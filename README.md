# A2A Invoice Agent

Layers: `app.py` (A2A protocol) -> `store.py` (SQLite, atomic transitions) -> `decider.py` (AI + cache + fallback).

## Run
    pip install -r requirements.txt
    export BASE_URL="https://YOUR-HOST/a2a/"     # EXACTLY what you submit to the grader (routes mount under its path)
    export ANTHROPIC_API_KEY=...                  # or OPENAI_API_KEY (+ OPENAI_BASE_URL, OPENAI_MODEL) for any OpenAI-compatible API
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

If BASE_URL is unset it is derived from the Host header (base path "/", no trailing slash) - submit that exact form.
Quick public HTTPS for testing: `cloudflared tunnel --url http://localhost:8000`.

## Settings
| env | default | meaning |
|---|---|---|
| ANTHROPIC_MODEL | claude-haiku-5-5 | model for the Anthropic provider |
| OPENAI_BASE_URL / OPENAI_MODEL | api.openai.com/v1 / gpt-4.1-mini | e.g. Groq, Gemini (OpenAI-compat), OpenRouter, Ollama |
| EVIDENCE_BRACKETS | 1 | 1: evidenceRefs like "[AB-12]" (as printed); 0: "AB-12" |
| STRICT_MEDIA_TYPE | 0 | 1: reject Content-Type application/json (only application/a2a+json) |
| DB_PATH | a2a_state.db | keep on a persistent disk |
| MAX_LLM_CALLS_PER_HOUR | 120 | spend guard |

## Test locally
    python test_local.py
