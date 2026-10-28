"""AI reasoning layer. Knows nothing about HTTP or Tasks.

Flow per batch:
  1. canonical-content cache lookup (ids ignored)   -> no model call for known content
  2. ONE model call for all uncached packages        -> JSON decisions
  3. strict validation; evidence refs are taken from the paragraph the model
     points at (copied by code, never retyped by the model)
  4. rule-based fallback if the model is unavailable (never cached)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import httpx

log = logging.getLogger("decider")

ACTIONS = ("settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception")
PROMPT_VERSION = "2026-10-v1"  # bump to invalidate the decision cache
REF_RE = re.compile(r"\[([^\[\]\r\n]{1,120})\]")
EVIDENCE_BRACKETS = os.environ.get("EVIDENCE_BRACKETS", "1") != "0"
LLM_DEADLINE_S = float(os.environ.get("LLM_DEADLINE_S", "36"))
MAX_LLM_CALLS_PER_HOUR = int(os.environ.get("MAX_LLM_CALLS_PER_HOUR", "120"))


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Package reading
# --------------------------------------------------------------------------
_ID_KEYS = {"packageId", "package_id", "id", "batchId"}


def package_key(pkg) -> str:
    """Cache key = canonical content with the per-batch identifiers removed."""
    core = {k: v for k, v in pkg.items() if k not in _ID_KEYS} if isinstance(pkg, dict) else pkg
    return sha(PROMPT_VERSION + canon(core))


def package_id(pkg) -> str | None:
    if not isinstance(pkg, dict):
        return None
    for k in ("packageId", "package_id", "id"):
        v = pkg.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def flatten(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{path}[{i}]")
    elif obj is not None:
        yield path, obj


def unique_refs(text: str) -> list[str]:
    return list(dict.fromkeys(REF_RE.findall(text)))


@dataclass
class Para:
    idx: int | None  # numbered only when the paragraph carries bracketed references
    path: str
    text: str
    refs: list


def segment(pkg) -> list[Para]:
    """Schema-agnostic: every string leaf is split into paragraphs; paragraphs that
    contain bracketed references are numbered so the model can point at one."""
    paras: list[Para] = []
    n = 0
    for path, val in flatten(pkg):
        if not isinstance(val, str):
            paras.append(Para(None, path, json.dumps(val), []))
            continue
        for chunk in re.split(r"\n\s*\n", val):
            chunk = chunk.strip()
            if not chunk:
                continue
            pieces = [chunk]
            if len(unique_refs(chunk)) > 3 and "\n" in chunk:
                pieces = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
            for piece in pieces:
                refs = unique_refs(piece)
                if refs:
                    n += 1
                    paras.append(Para(n, path, piece, refs))
                else:
                    paras.append(Para(None, path, piece, []))
    return paras


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------
_ZERO_DEC = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "XAF", "XOF", "PYG", "RWF"}
_THREE_DEC = {"KWD", "BHD", "JOD", "OMR", "TND", "IQD", "LYD"}


def minor_exp(cur: str) -> int:
    return 0 if cur in _ZERO_DEC else 3 if cur in _THREE_DEC else 2


def parse_amount(s, cur="INR"):
    """'1,23,450.75' / '1.234,50' / '12345' -> integer minor units."""
    s = re.sub(r"[^\d.,]", "", str(s))
    if not re.search(r"\d", s):
        return None
    dec = None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
    elif "," in s:
        tail = s.rsplit(",", 1)[1]
        dec = "," if (s.count(",") == 1 and len(tail) in (1, 2)) else None
    elif "." in s:
        dec = "." if s.count(".") == 1 else None
    if dec:
        ip, fp = s.rsplit(dec, 1)
        ip = re.sub(r"[.,]", "", ip)
    else:
        ip, fp = re.sub(r"[.,]", "", s), ""
    try:
        d = Decimal(f"{ip or '0'}.{fp or '0'}")
    except InvalidOperation:
        return None
    return int((d * (10 ** minor_exp(cur))).to_integral_value())


def _coerce_int(v):
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
            return int(v)
    except (ValueError, OverflowError):
        pass
    return None


def _text_minors(paras, cur):
    out = set()
    for pa in paras:
        for tok in re.findall(r"\d[\d,.]*\d|\d", pa.text):
            m = parse_amount(tok, cur)
            if m is not None:
                out.add(m)
    return out


def resolve_amount(item, cur, paras):
    written = parse_amount(item.get("amountAsWritten"), cur) if item.get("amountAsWritten") not in (None, "") else None
    given = _coerce_int(item.get("amountMinor"))
    if written is not None and given is not None and written == given:
        return given
    cands = [c for c in (written, given) if c is not None]
    if not cands:
        return None
    seen = _text_minors(paras, cur)
    for c in cands:  # prefer the value that is actually printed in the package
        if c in seen:
            return c
    return cands[0]


# --------------------------------------------------------------------------
# Optional structured hints: a key that occurs exactly once in the whole package is trusted
# --------------------------------------------------------------------------
_HINT_KEYS = {
    "vendorName": {"vendorname", "vendor_name", "vendor", "suppliername"},
    "invoiceNumber": {"invoicenumber", "invoice_number", "invoiceno", "invoice_no", "invoiceid"},
    "amountMinor": {"amountminor", "amount_minor"},
    "currency": {"currency", "currencycode"},
}


def structured_hints(pkg) -> dict:
    found: dict[str, list] = {k: [] for k in _HINT_KEYS}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                nk = re.sub(r"[^a-z_]", "", str(k).lower())
                for field, names in _HINT_KEYS.items():
                    if nk in names and isinstance(v, (str, int)) and not isinstance(v, bool):
                        found[field].append(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(pkg)
    return {k: v[0] for k, v in found.items() if len(v) == 1}


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the control analyst for an accounts-payable team. Each invoice package mixes several documents: the operative finding for THIS invoice plus noise - cover sheets, archived or old examples, training material and decoys, hypothetical or negated statements, and unrelated words that merely look like actions.

For EACH package:
1. Find the ONE operative paragraph: the current, invoice-specific finding that states the real situation of this invoice and carries exactly three bracketed references. Cover sheets, archive/historical examples, training or sample material, templates and "for illustration" text are NOT operative, even when they contain action words and bracketed references. Negated statements ("no duplicate was found", "approval is not required", "hold released") describe what is NOT the case - do not act on the negated word.
2. Choose exactly one action, reading the operative paragraph together with any limits or policy it relies on elsewhere in the package:
- settle_invoice: invoice valid, records reconciled, and within autonomous payment authority.
- request_approval: commercially valid but above delegated authority; a higher approver must sign off.
- hold_invoice: payment pauses until a stated verification completes (for example a bank-detail callback, tax or compliance document, or goods-receipt confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number, same amount) was already paid. A payment to the same vendor for a DIFFERENT invoice number is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, receipt, bank details) and need an exception workflow instead of a simple pause.
3. Extract the facts of THIS invoice: vendorName and invoiceNumber exactly as written, the ISO currency code, and the invoice amount.

Output ONLY one JSON object, no prose, no markdown:
{"results":[{"n":<package number>,"p":<number of the operative paragraph, the P-number>,"reason":"<one or two sentences, max 220 chars, naming the decisive finding>","action":"<one of the five actions>","vendorName":"...","invoiceNumber":"...","currency":"INR","amountAsWritten":"<amount exactly as printed, digits and separators only>","amountMinor":<integer in minor units, e.g. 1,234.50 -> 123450>,"refs":["<the three bracketed references of paragraph p, copied verbatim without the brackets>"]}]}
Return one entry for every package, in order."""


def build_prompt(policy_revision, items):
    lines = [f"Policy revision: {policy_revision}", f"Packages: {len(items)}", ""]
    paras_per = []
    for n, (_, pkg) in enumerate(items, start=1):
        paras = segment(pkg)
        paras_per.append(paras)
        lines.append(f"##### PACKAGE {n} #####")
        for pa in paras:
            lines.append(f"P{pa.idx} ({pa.path}) {pa.text}" if pa.idx is not None else f"({pa.path}) {pa.text}")
        lines.append("")
    lines.append("Return the JSON now.")
    return "\n".join(lines), paras_per


def extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


# --------------------------------------------------------------------------
# Validation of one model item
# --------------------------------------------------------------------------
def normalize_item(item, paras, pkg):
    try:
        action = str(item.get("action", "")).strip()
        if action not in ACTIONS:
            return None
        p = item.get("p")
        if isinstance(p, str):
            m = re.search(r"\d+", p)
            p = int(m.group(0)) if m else None
        chosen = next((x for x in paras if x.idx is not None and x.idx == p), None)

        refs = None
        if chosen and len(chosen.refs) == 3:
            refs = list(chosen.refs)
        else:
            present = {r for pa in paras for r in pa.refs}
            said = [str(r).strip("[] ") for r in (item.get("refs") or []) if isinstance(r, (str, int))]
            said = [r for r in dict.fromkeys(said) if r in present]
            if said:
                refs = said[:3]
            elif chosen and chosen.refs:
                refs = list(chosen.refs[:3])
        if not refs:
            return None

        hints = structured_hints(pkg)
        vendor = hints.get("vendorName") or item.get("vendorName")
        invoice = hints.get("invoiceNumber") or item.get("invoiceNumber")
        cur = str(hints.get("currency") or item.get("currency") or "").strip().upper()
        if not (isinstance(vendor, str) and vendor.strip() and isinstance(invoice, str) and invoice.strip()):
            return None
        if not re.fullmatch(r"[A-Z]{3}", cur):
            return None
        hint_minor = _coerce_int(hints.get("amountMinor")) if "amountMinor" in hints else None
        amount = hint_minor if hint_minor is not None else resolve_amount(item, cur, paras)
        if amount is None:
            return None
        return {
            "action": action,
            "facts": {"vendorName": vendor.strip(), "invoiceNumber": invoice.strip(), "amountMinor": amount, "currency": cur},
            "refs": refs,
            "reason": str(item.get("reason") or "")[:400],
            "source": "llm",
        }
    except Exception:  # a malformed item must never break the batch
        log.exception("normalize_item failed")
        return None


# --------------------------------------------------------------------------
# Rule-based fallback (only used when the model is unavailable; never cached)
# --------------------------------------------------------------------------
_NOISE = re.compile(r"archiv|example|training|decoy|sample|cover|template|illustrat|historical|legacy|obsolete|superseded|hypothetic|reference only", re.I)
_NEG = re.compile(r"\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't|\bneither\b|\bnor\b|\blifted\b|\breleased\b|\bcleared\b", re.I)


def _has(text, pattern):
    for m in re.finditer(pattern, text, re.I):
        if not _NEG.search(text[max(0, m.start() - 40):m.start()]):
            return True
    return False


def heuristic_decision(pkg):
    paras = segment(pkg)
    cands = [p for p in paras if p.idx is not None and len(p.refs) == 3 and not _NOISE.search(p.text) and not _NOISE.search(p.path)]
    chosen = cands[0] if cands else next((p for p in paras if p.idx is not None), None)
    text = chosen.text if chosen else " ".join(p.text for p in paras)
    if _has(text, r"duplicate|already paid|previously paid"):
        action = "reject_duplicate"
    elif _has(text, r"conflict|mismatch|discrepan|do(?:es)? not match|exception"):
        action = "open_exception"
    elif _has(text, r"exceed|above (?:the )?(?:delegated|autonomous)|beyond (?:the )?(?:delegated|autonomous)|approval|delegated authority"):
        action = "request_approval"
    elif _has(text, r"\bhold\b|verification|verify|callback|pending|until"):
        action = "hold_invoice"
    else:
        action = "settle_invoice"

    blob = "\n".join(p.text for p in paras)
    hints = structured_hints(pkg)
    m = re.search(r"(?:vendor|supplier|payee)(?: name)?\s*[:=\-]\s*([^\n;,|]{2,80})", blob, re.I)
    vendor = hints.get("vendorName") or (m.group(1).strip() if m else "UNKNOWN")
    m = re.search(r"invoice\s*(?:no\.?|number|#|id)?\s*[:=#\-]?\s*([A-Z0-9][A-Z0-9\-/_.]{2,30})", blob, re.I)
    invoice = hints.get("invoiceNumber") or (m.group(1).strip(".") if m else "UNKNOWN")
    m = re.search(r"\b(INR|USD|EUR|GBP|AED|SGD|JPY)\b\s*([\d][\d,]*(?:\.\d+)?)", blob)
    cur = str(hints.get("currency") or (m.group(1) if m else "INR")).upper()
    amount = _coerce_int(hints.get("amountMinor")) if "amountMinor" in hints else None
    if amount is None:
        amount = parse_amount(m.group(2), cur) if m else 0
    reason = "Rule-based reading of the operative paragraph." if chosen else "No operative paragraph was found."
    return {
        "action": action,
        "facts": {"vendorName": vendor, "invoiceNumber": invoice, "amountMinor": amount or 0, "currency": cur},
        "refs": list(chosen.refs[:3]) if chosen else [],
        "reason": reason,
        "source": "heuristic",
    }


def display_refs(refs):
    return [f"[{r}]" if EVIDENCE_BRACKETS else r for r in refs]


def make_rationale(action, reason, refs_display):
    reason = re.sub(r"\s+", " ", reason or "").strip()[:900]
    if len(reason) < 20:
        reason = "The operative finding in this package supports this action."
    return f"Action {action}: {reason} Decisive evidence: {', '.join(refs_display)}."[:1500]


# --------------------------------------------------------------------------
# Decider
# --------------------------------------------------------------------------
class Decider:
    def __init__(self, store):
        self.store = store
        self.inflight: dict[str, asyncio.Future] = {}
        self.client = httpx.AsyncClient()
        self.calls = deque()  # timestamps, simple spend guard
        self.provider = os.environ.get("LLM_PROVIDER", "").lower()
        if not self.provider:
            self.provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai" if os.environ.get("OPENAI_API_KEY") else "none"
        log.info("LLM provider: %s", self.provider)

    # -- model transport ----------------------------------------------------
    async def _complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        if self.provider == "anthropic":
            r = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                json={
                    "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        if self.provider == "openai":  # any OpenAI-compatible endpoint (OpenAI, Groq, Gemini, OpenRouter, Ollama...)
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            body = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if os.environ.get("OPENAI_JSON_MODE", "0") == "1":
                body["response_format"] = {"type": "json_object"}
            r = await self.client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"},
                json=body,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("no LLM configured")

    def _budget_ok(self) -> bool:
        now = time.time()
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        if len(self.calls) >= MAX_LLM_CALLS_PER_HOUR:
            return False
        self.calls.append(now)
        return True

    async def _run_llm(self, policy_revision, items):
        """items: [(key, pkg)] -> {key: decision}. One call for the whole list."""
        out = {}
        if self.provider != "none" and self._budget_ok():
            prompt, paras_per = build_prompt(policy_revision, items)
            deadline = time.monotonic() + LLM_DEADLINE_S
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining < 6:
                    break
                try:
                    raw = await self._complete(SYSTEM_PROMPT, prompt, 400 * len(items) + 500, remaining - 1)
                    data = extract_json(raw)
                    results = data["results"] if isinstance(data, dict) else data
                    for it in results:
                        n = it.get("n") if isinstance(it, dict) else None
                        if isinstance(n, int) and 1 <= n <= len(items):
                            k, pkg = items[n - 1]
                            dec = normalize_item(it, paras_per[n - 1], pkg)
                            if dec and k not in out:
                                out[k] = dec
                    if len(out) == len(items):
                        break
                except httpx.HTTPStatusError as e:
                    log.warning("LLM HTTP %s", e.response.status_code)
                    if e.response.status_code in (429, 500, 502, 503, 529):
                        await asyncio.sleep(1.5)
                        continue
                    break
                except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as e:
                    log.warning("LLM call failed: %s", type(e).__name__)
                    if isinstance(e, httpx.TimeoutException):
                        break
        for k, pkg in items:
            if k not in out:
                log.warning("falling back to rules for one package")
                out[k] = heuristic_decision(pkg)
        return out

    # -- public ---------------------------------------------------------------
    async def decide(self, policy_revision, packages) -> list[dict]:
        loop = asyncio.get_running_loop()
        keys = [package_key(p) for p in packages]
        uniq: dict[str, object] = {}
        for k, p in zip(keys, packages):
            uniq.setdefault(k, p)

        decisions: dict[str, dict] = {}
        waits: dict[str, asyncio.Future] = {}
        mine: list[str] = []
        for k, p in uniq.items():
            cached = self.store.get_decision(k)
            if cached:
                decisions[k] = cached
            elif k in self.inflight:
                waits[k] = self.inflight[k]
            else:
                self.inflight[k] = loop.create_future()
                mine.append(k)
        try:
            if mine:
                out = await self._run_llm(policy_revision, [(k, uniq[k]) for k in mine])
                for k in mine:
                    decisions[k] = out[k]
                    if out[k]["source"] == "llm":
                        self.store.put_decision(k, out[k])
        finally:
            for k in mine:
                fut = self.inflight.pop(k, None)
                if fut and not fut.done():
                    fut.set_result(decisions.get(k))
        for k, fut in waits.items():
            decisions[k] = (await fut) or heuristic_decision(uniq[k])
        return [decisions[k] for k in keys]
