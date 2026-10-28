"""A2A 1.0 (HTTP+JSON) invoice-action agent - protocol layer.

Run with exactly ONE worker (state transitions rely on a single event loop + SQLite transactions):
    uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from decider import Decider, canon, display_refs, make_rationale, package_id
from store import COMPLETED, INPUT_REQUIRED, TERMINAL, WORKING, Store

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("a2a")

BASE_URL = os.environ.get("BASE_URL", "").strip()  # EXACTLY the URL submitted to the grader
BASE_PATH = urlparse(BASE_URL).path.rstrip("/") if BASE_URL else ""
STRICT_MEDIA_TYPE = os.environ.get("STRICT_MEDIA_TYPE", "0") == "1"
MAX_BODY = 8 * 1024 * 1024
MAX_RESPONSE = 500_000  # A2A responses must stay <= 512 KiB
REQUEST_BUDGET_S = 42.0  # per-request limit is 45 s

M_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
M_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
M_RESULTS = "application/vnd.ga5.invoice-action-results+json"
M_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

STATUS_NAMES = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
                405: "METHOD_NOT_ALLOWED", 409: "ABORTED", 413: "INVALID_ARGUMENT", 415: "INVALID_ARGUMENT"}

store = Store(os.environ.get("DB_PATH", "a2a_state.db"))
decider = Decider(store)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, reason, message, headers=None):
        self.status, self.reason, self.message, self.headers = status, reason, message, headers or {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def error(status, reason, message, headers=None) -> Response:
    body = {"error": {
        "code": status, "status": STATUS_NAMES.get(status, "ERROR"), "reason": reason, "message": message,
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason, "domain": "a2a-protocol.org"}],
    }}
    return Response(json.dumps(body), status_code=status, media_type="application/a2a+json",
                    headers={"A2A-Version": "1.0", "Cache-Control": "no-store", **(headers or {})})


def ok(obj, media="application/a2a+json") -> Response:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode()) > MAX_RESPONSE and isinstance(obj, dict):  # safety net, normally never reached
        task = obj.get("task", obj)
        for m in task.get("history", [])[1:]:
            m["parts"] = [{"mediaType": p.get("mediaType"), "data": {"truncated": True}} for p in m.get("parts", [])]
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return Response(text, media_type=media, headers={"A2A-Version": "1.0", "Cache-Control": "no-store"})


def authenticate(request: Request):
    h = request.headers.get("authorization", "").strip()
    m = re.fullmatch(r"(?i:bearer)\s+(\S{1,4096})", h)
    if not m:
        return None
    return hashlib.sha256(m.group(1).encode()).hexdigest()  # principal id; the raw token is never stored


def check_version(request: Request):
    v = request.headers.get("a2a-version")
    if v is None:
        v = request.query_params.get("A2A-Version", "")
    if v.strip() != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "Only A2A-Version 1.0 is supported.")


def check_content_type(request: Request):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    allowed = {"application/a2a+json"} | (set() if STRICT_MEDIA_TYPE else {"application/json"})
    if ct not in allowed:
        raise ApiError(415, "CONTENT_TYPE_NOT_SUPPORTED", "Content-Type must be application/a2a+json.")


def message_hash(msg) -> str:
    return hashlib.sha256(canon(msg).encode("utf-8")).hexdigest()  # `message` only; configuration is ignored


def base_url_for(request: Request) -> str:
    if BASE_URL:
        return BASE_URL
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    return f"{request.headers.get('x-forwarded-proto', 'https')}://{host}"


def agent_card(base: str) -> dict:
    modes_in = [M_BATCH, M_RESULTS]
    modes_out = [M_PROPOSALS, M_RECEIPTS]
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed business action per invoice, "
                       "and executes only the proposals the grader accepts. Task state is persisted so retries never repeat an action.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "provider": {"organization": "Invoice Action Agent", "url": base},
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearerAuth": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearerAuth": {"list": []}}}],
        "defaultInputModes": modes_in,
        "defaultOutputModes": modes_out,
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action decisions",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or open_exception "
                           "for each invoice package and cites the decisive evidence.",
            "tags": ["invoice", "accounts-payable", "approvals", "duplicate-detection"],
            "inputModes": modes_in,
            "outputModes": modes_out,
        }],
    }


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------
def validate_envelope(body) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        raise ApiError(400, "INVALID_ARGUMENT", "Request must contain a message object.")
    msg = body["message"]
    mid = msg.get("messageId")
    if not isinstance(mid, str) or not mid.strip() or len(mid) > 256:
        raise ApiError(400, "INVALID_ARGUMENT", "message.messageId is required.")
    if msg.get("role") != "ROLE_USER":
        raise ApiError(400, "INVALID_ARGUMENT", "message.role must be ROLE_USER.")
    parts = msg.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) for p in parts):
        raise ApiError(400, "INVALID_ARGUMENT", "message.parts must be a non-empty list.")
    for k in ("taskId", "contextId"):
        if k in msg and msg[k] is not None and not isinstance(msg[k], str):
            raise ApiError(400, "INVALID_ARGUMENT", f"message.{k} must be a string.")
    return msg


def extract_batch(msg) -> dict:
    parts = msg["parts"]
    if len(parts) != 1:
        raise ApiError(400, "INVALID_ARGUMENT", "Expected exactly one batch part.")
    if parts[0].get("mediaType") != M_BATCH:
        raise ApiError(400, "CONTENT_TYPE_NOT_SUPPORTED", "Unsupported part media type.")
    data = parts[0].get("data")
    if not isinstance(data, dict):
        raise ApiError(400, "INVALID_ARGUMENT", "Batch part needs a data object.")
    bid, rev, pk = data.get("batchId"), data.get("policyRevision"), data.get("packages")
    if not isinstance(bid, str) or not bid.strip():
        raise ApiError(400, "INVALID_ARGUMENT", "batchId is required.")
    if not isinstance(rev, (str, int)) or isinstance(rev, bool):
        raise ApiError(400, "INVALID_ARGUMENT", "policyRevision is required.")
    if not isinstance(pk, list) or not pk or len(pk) > 200:
        raise ApiError(400, "INVALID_ARGUMENT", "packages must be a non-empty list.")
    ids = [package_id(p) for p in pk]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ApiError(400, "INVALID_ARGUMENT", "Every package needs a unique packageId.")
    return data


def parse_results(task: dict, batch_id: str, data) -> list[dict]:
    """Validate a result continuation against the stored proposals. Returns accepted executions."""
    stored = task["artifacts"][0]["parts"][0]["data"]
    if not isinstance(data, dict) or data.get("batchId") != batch_id or data.get("batchId") != stored.get("batchId"):
        raise ApiError(400, "INVALID_ARGUMENT", "Batch does not match the task.")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ApiError(400, "INVALID_ARGUMENT", "results must be a non-empty list.")
    proposals = {p["packageId"]: p for p in stored["proposals"]}
    seen, accepted = set(), {}
    for r in results:
        if not isinstance(r, dict):
            raise ApiError(400, "INVALID_ARGUMENT", "Malformed result.")
        pid = r.get("packageId")
        p = proposals.get(pid) if isinstance(pid, str) else None
        if p is None:
            raise ApiError(400, "INVALID_ARGUMENT", "A result references an unknown package.")
        if pid in seen:
            raise ApiError(400, "INVALID_ARGUMENT", "A package appears twice in results.")
        seen.add(pid)
        if r.get("actionId") != p["actionId"]:
            raise ApiError(400, "INVALID_ARGUMENT", "A result's actionId does not match the proposal.")
        if r.get("action") != p["action"]:
            raise ApiError(400, "INVALID_ARGUMENT", "A result's action does not match the proposal.")
        outcome = r.get("outcome")
        if outcome not in ("ACCEPTED", "REJECTED"):
            raise ApiError(400, "INVALID_ARGUMENT", "A result has an invalid outcome.")
        if outcome == "ACCEPTED":
            nonce = r.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce.strip() or len(nonce) > 1024:
                raise ApiError(400, "INVALID_ARGUMENT", "An accepted result needs a receiptNonce.")
            accepted[pid] = nonce
    order = [p["packageId"] for p in stored["proposals"]]  # executions follow proposal order
    return [
        {"packageId": pid, "actionId": proposals[pid]["actionId"], "action": proposals[pid]["action"],
         "receiptNonce": accepted[pid], "facts": proposals[pid]["facts"], "evidenceRefs": proposals[pid]["evidenceRefs"]}
        for pid in order if pid in accepted
    ]


# --------------------------------------------------------------------------
# service logic
# --------------------------------------------------------------------------
class Service:
    def __init__(self):
        self.running: dict[tuple, asyncio.Task] = {}

    async def new_task(self, principal, msg, msg_hash) -> dict:
        batch = extract_batch(msg)  # validated before anything is stored
        task_id = str(uuid.uuid4())
        context_id = msg.get("contextId") or str(uuid.uuid4())
        stored_msg = {**msg, "taskId": task_id, "contextId": context_id}
        task = {"id": task_id, "contextId": context_id,
                "status": {"state": WORKING, "timestamp": now()}, "history": [stored_msg]}
        # message record + task are persisted BEFORE any model work (no await above this line)
        if not store.create_task(principal, msg["messageId"], msg_hash, batch["batchId"], task):
            row = store.get_message(principal, msg["messageId"])
            return await self.replay(principal, row["task_id"])
        return await self.run(principal, task_id)

    async def replay(self, principal, task_id) -> dict:
        task = store.get_task(principal, task_id)
        if task and task["status"]["state"] == WORKING:
            return await self.run(principal, task_id)  # join the running job, or resume after a crash
        return task

    async def run(self, principal, task_id) -> dict:
        key = (principal, task_id)
        job = self.running.get(key)
        if job is None:
            job = asyncio.create_task(self._process(principal, task_id))
            self.running[key] = job
            job.add_done_callback(lambda _j, k=key: self.running.pop(k, None))
        try:
            await asyncio.wait_for(asyncio.shield(job), timeout=REQUEST_BUDGET_S)
        except asyncio.TimeoutError:
            pass
        return store.get_task(principal, task_id)

    async def _process(self, principal, task_id):
        try:
            task = store.get_task(principal, task_id)
            if not task or task["status"]["state"] != WORKING:
                return
            data = task["history"][0]["parts"][0]["data"]
            packages = data["packages"]
            decisions = await decider.decide(data["policyRevision"], packages)
            proposals = []
            for pkg, d in zip(packages, decisions):
                pid = package_id(pkg)
                refs = display_refs(d["refs"])
                action_id = "act_" + hashlib.sha256(f"{principal}|{task_id}|{data['batchId']}|{pid}".encode()).hexdigest()[:24]
                proposals.append({
                    "packageId": pid, "actionId": action_id, "action": d["action"],
                    "facts": d["facts"], "evidenceRefs": refs,
                    "rationale": make_rationale(d["action"], d["reason"], refs),
                })
            artifact = {
                "artifactId": f"proposals-{task_id}", "name": "invoice-action-proposals",
                "parts": [{"mediaType": M_PROPOSALS, "data": {"batchId": data["batchId"], "proposals": proposals}}],
            }
            store.finish_proposals(principal, task_id, artifact, now())
        except Exception:
            log.exception("processing failed")
            store.fail_task(principal, task_id, now())

    def continue_task(self, principal, msg, msg_hash) -> dict:
        task_id = msg["taskId"]
        task = store.get_task(principal, task_id)
        if task is None:  # unknown id and someone else's id are indistinguishable
            raise ApiError(404, "TASK_NOT_FOUND", "Task not found.")
        state = task["status"]["state"]
        if state in TERMINAL:
            raise ApiError(409, "UNSUPPORTED_OPERATION", "The task has already finished and accepts no further messages.")
        if state != INPUT_REQUIRED:
            raise ApiError(409, "UNSUPPORTED_OPERATION", "The task is not waiting for results.")
        if msg.get("contextId") not in (None, task["contextId"]):
            raise ApiError(400, "INVALID_ARGUMENT", "Context does not match the task.")
        parts = msg["parts"]
        if len(parts) != 1 or parts[0].get("mediaType") != M_RESULTS:
            raise ApiError(400, "CONTENT_TYPE_NOT_SUPPORTED", "Expected one action-results part.")
        batch_id = store.get_batch_id(principal, task_id)
        executions = parse_results(task, batch_id, parts[0].get("data"))
        receipts = {
            "artifactId": f"receipts-{task_id}", "name": "invoice-action-receipts",
            "parts": [{"mediaType": M_RECEIPTS, "data": {"batchId": batch_id, "executions": executions}}],
        }
        stored_msg = {**msg, "taskId": task_id, "contextId": task["contextId"]}
        status, updated = store.complete(principal, task_id, stored_msg, msg["messageId"], msg_hash, receipts, executions, now())
        if status == "ok":
            return updated
        if status == "notfound":
            raise ApiError(404, "TASK_NOT_FOUND", "Task not found.")
        raise ApiError(409, "UNSUPPORTED_OPERATION", "The task has already finished.")  # lost a race (e.g. cancel)

    def cancel(self, principal, task_id) -> dict:
        status, task = store.cancel(principal, task_id, now())
        if status == "notfound":
            raise ApiError(404, "TASK_NOT_FOUND", "Task not found.")
        if status == "terminal":
            raise ApiError(409, "TASK_NOT_CANCELABLE", "The task has already finished.")
        return task


svc = Service()


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
async def handle_send(request: Request, principal: str) -> Response:
    check_content_type(request)
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body too large.")
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body too large.")
    try:
        body = json.loads(raw)
    except ValueError:
        raise ApiError(400, "INVALID_ARGUMENT", "Body is not valid JSON.")
    msg = validate_envelope(body)
    h = message_hash(msg)

    row = store.get_message(principal, msg["messageId"])  # (principal, messageId) idempotency
    if row:
        if row["hash"] != h:
            raise ApiError(409, "IDEMPOTENCY_CONFLICT", "This messageId was already used with different content.")
        return ok({"task": await svc.replay(principal, row["task_id"])})

    if msg.get("taskId"):
        return ok({"task": svc.continue_task(principal, msg, h)})
    return ok({"task": await svc.new_task(principal, msg, h)})


VALID_STATES = {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED", "TASK_STATE_FAILED",
                "TASK_STATE_CANCELED", "TASK_STATE_REJECTED", "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED"}


def _int_param(q, name, default=None, lo=0, hi=None):
    raw = q.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ApiError(400, "INVALID_ARGUMENT", f"{name} must be an integer.")
    if v < lo or (hi is not None and v > hi):
        raise ApiError(400, "INVALID_ARGUMENT", f"{name} is out of range.")
    return v


def shape_task(task: dict, history_length=None, include_artifacts=True) -> dict:
    """Spec 3.2.4: historyLength unset = full history, 0 = omit, N = last N messages."""
    out = dict(task)
    if history_length is not None:
        if history_length == 0:
            out.pop("history", None)
        elif "history" in out:
            out["history"] = out["history"][-history_length:]
    if not include_artifacts:
        out.pop("artifacts", None)
    return out


def _ts(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ApiError(400, "INVALID_ARGUMENT", "statusTimestampAfter must be an ISO 8601 timestamp.")


def handle_list(request: Request, principal: str) -> Response:
    """Spec 3.1.4: owner-only, sorted by status timestamp (newest first), cursor pagination,
    artifacts omitted unless includeArtifacts=true, history omitted unless historyLength given."""
    q = request.query_params
    page_size = _int_param(q, "pageSize", 50, 1, 100)
    history_length = _int_param(q, "historyLength", 0, 0)
    include_artifacts = q.get("includeArtifacts", "false").lower() == "true"
    status = q.get("status")
    if status and status not in VALID_STATES:
        raise ApiError(400, "INVALID_ARGUMENT", "status is not a valid task state.")
    after = _ts(q["statusTimestampAfter"]) if q.get("statusTimestampAfter") else None
    offset = 0
    if q.get("pageToken"):
        try:
            offset = int(base64.urlsafe_b64decode(q["pageToken"].encode()).decode().split(":")[1])
            assert offset >= 0
        except Exception:
            raise ApiError(400, "INVALID_ARGUMENT", "pageToken is invalid.")

    tasks = store.list_tasks(principal)  # only this principal's tasks, by construction
    tasks.sort(key=lambda t: t["status"].get("timestamp", ""), reverse=True)  # stable
    if q.get("contextId"):
        tasks = [t for t in tasks if t.get("contextId") == q["contextId"]]
    if status:
        tasks = [t for t in tasks if t["status"]["state"] == status]
    if after:
        tasks = [t for t in tasks if _ts(t["status"].get("timestamp", "1970-01-01T00:00:00Z")) >= after]
    total = len(tasks)

    page, size = [], 0
    for t in tasks[offset:offset + page_size]:
        shaped = shape_task(t, history_length, include_artifacts)
        size += len(json.dumps(shaped))
        if size > 400_000 and page:  # keep the response well under 512 KiB
            break
        page.append(shaped)
    end = offset + len(page)
    token = base64.urlsafe_b64encode(f"o:{end}".encode()).decode() if end < total else ""
    return ok({"tasks": page, "nextPageToken": token, "pageSize": page_size, "totalSize": total})


async def route(request: Request) -> Response:
    path, method = request.url.path, request.method

    if path == "/.well-known/agent-card.json":
        if method not in ("GET", "HEAD"):
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
        body = json.dumps(agent_card(base_url_for(request)))
        return Response(body, media_type="application/json",
                        headers={"Cache-Control": "public, max-age=30",
                                 "ETag": '"' + hashlib.sha256(body.encode()).hexdigest()[:16] + '"'})
    if path == "/healthz":
        return Response('{"ok":true}', media_type="application/json")

    if BASE_PATH and not (path == BASE_PATH or path.startswith(BASE_PATH + "/")):
        raise ApiError(404, "NOT_FOUND", "Not found.")
    rel = path[len(BASE_PATH):].rstrip("/") or "/"

    principal = authenticate(request)
    if principal is None:
        raise ApiError(401, "UNAUTHENTICATED", "Bearer token required.", {"WWW-Authenticate": "Bearer"})
    check_version(request)

    if rel == "/message:send":
        if method != "POST":
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed.", {"Allow": "POST"})
        return await handle_send(request, principal)
    if rel == "/tasks":
        if method != "GET":
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed.", {"Allow": "GET"})
        return handle_list(request, principal)
    m = re.fullmatch(r"/tasks/([^/:]+):cancel", rel)
    if m:
        if method != "POST":
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed.", {"Allow": "POST"})
        return ok(svc.cancel(principal, m.group(1)))
    m = re.fullmatch(r"/tasks/([^/:]+)", rel)
    if m:
        if method != "GET":
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed.", {"Allow": "GET"})
        task = store.get_task(principal, m.group(1))
        if task is None:
            raise ApiError(404, "TASK_NOT_FOUND", "Task not found.")
        return ok(shape_task(task, _int_param(request.query_params, "historyLength", None, 0)))
    raise ApiError(404, "NOT_FOUND", "Not found.")


async def endpoint(request: Request) -> Response:
    try:
        return await route(request)
    except ApiError as e:
        return error(e.status, e.reason, e.message, e.headers)
    except Exception:
        log.exception("unhandled error")
        return error(500, "INTERNAL", "Internal error.")


app = Starlette(routes=[Route("/{path:path}", endpoint, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])])
