"""LLM provider abstraction. Keys are read from environment variables and never leave the backend."""
import json
import urllib.error
import urllib.request

from ..core.config import env, settings
from ..core.errors import AppError

# OpenAI-style providers: id -> (label, key env, base-url env, default base, model env, default model, hint)
_OPENAI_STYLE = {
    "openai_compatible": ("OpenAI-compatible API", "OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1",
                          "OPENAI_MODEL", "gpt-4o-mini",
                          "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) in .env"),
    "gemini": ("Google Gemini", "GEMINI_API_KEY", "GEMINI_BASE_URL",
               "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_MODEL", "gemini-2.5-flash",
               "Set GEMINI_API_KEY (free key from aistudio.google.com) in .env"),
    "groq": ("Groq", "GROQ_API_KEY", "GROQ_BASE_URL", "https://api.groq.com/openai/v1", "GROQ_MODEL",
             "llama-3.3-70b-versatile", "Set GROQ_API_KEY (free key from console.groq.com) in .env"),
}


def _cfg(pid: str) -> dict:
    label, key_env, base_env, base_def, model_env, model_def, hint = _OPENAI_STYLE[pid]
    return {"label": label, "key": env(key_env), "base": env(base_env, base_def).rstrip("/"),
            "model": env(model_env, model_def), "hint": hint, "default_base": base_def}


def providers() -> list[dict]:
    out = []
    for pid in _OPENAI_STYLE:
        c = _cfg(pid)
        # OpenAI-compatible also counts as ready with no key when pointed at a local server (Ollama, LM Studio)
        ready = bool(c["key"]) or (pid == "openai_compatible" and c["base"] != c["default_base"])
        out.append({"id": pid, "label": c["label"], "kind": "external", "model": c["model"], "ready": ready,
                    "hint": c["hint"]})
    out.append({"id": "anthropic", "label": "Anthropic Claude", "kind": "external",
                "model": env("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "ready": bool(env("ANTHROPIC_API_KEY")),
                "hint": "Set ANTHROPIC_API_KEY (and optionally ANTHROPIC_MODEL) in .env"})
    out.append({"id": "demo", "label": "Local demo (rule-based, not an LLM)", "kind": "demo", "model": "none",
                "ready": True, "hint": "Always available. Answers a few built-in questions only."})
    return out


def default_provider() -> str:
    wanted = env("LLM_PROVIDER")
    ps = providers()
    if wanted and any(p["id"] == wanted and p["ready"] for p in ps):
        return wanted
    return next((p["id"] for p in ps if p["kind"] == "external" and p["ready"]), "demo")


def _provider_detail(body: bytes) -> str:
    """Pull the human-readable message out of a provider's JSON error body."""
    try:
        j = json.loads(body.decode("utf-8", "ignore"))
        j = j[0] if isinstance(j, list) and j else j
        e = j.get("error", j) if isinstance(j, dict) else {}
        msg = e.get("message") if isinstance(e, dict) else str(e)
        return (msg or "").strip().replace("\n", " ")[:300]
    except Exception:
        return ""


def _post(url: str, headers: dict, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=settings.request_timeout)
    except urllib.error.HTTPError as e:
        base = {400: "The provider rejected the request (check the model name).",
                401: "The provider rejected the API key.", 403: "The provider denied access to this model or key.",
                404: "Model or endpoint not found. Check the model name and base URL.",
                429: "The provider's rate limit or quota was reached. Try again later."}.get(
            e.code, f"The provider returned HTTP {e.code}.")
        detail = _provider_detail(e.read())
        raise AppError(502, "provider_error", f"{base} {('Details: ' + detail) if detail else ''}".strip())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AppError(504, "provider_unreachable",
                       f"Could not reach the provider ({getattr(e, 'reason', e)}). Check your internet connection, "
                       "base URL, or proxy/firewall.")


def _sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _openai(pid: str, messages: list[dict], max_tokens: int | None = None):
    c = _cfg(pid)
    headers = {"Authorization": f"Bearer {c['key']}"} if c["key"] else {}
    payload = {"model": c["model"], "stream": True,
               "messages": [{"role": "system", "content": settings.system_prompt}] + messages}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    resp = _post(f"{c['base']}/chat/completions", headers, payload)

    def gen():
        with resp:
            for ev in _sse(resp):
                if isinstance(ev, list):  # Gemini sometimes wraps events in a list
                    ev = ev[0] if ev else {}
                if isinstance(ev, dict) and ev.get("error"):
                    raise AppError(502, "provider_error", f"The provider reported an error: {ev['error']}"[:300])
                try:
                    t = ev["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, TypeError, AttributeError):
                    t = None
                if t:
                    yield t
    return gen()


def _anthropic(messages: list[dict], max_tokens: int = 2048):
    resp = _post(env("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages",
                 {"x-api-key": env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
                 {"model": env("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": max_tokens, "stream": True,
                  "system": settings.system_prompt, "messages": messages})

    def gen():
        with resp:
            for ev in _sse(resp):
                typ = ev.get("type")
                if typ == "content_block_delta":
                    t = ev.get("delta", {}).get("text")
                    if t:
                        yield t
                elif typ == "error":
                    raise AppError(502, "provider_error", "The provider reported an error: "
                                   f"{ev.get('error', {}).get('message', 'unknown')}"[:300])
    return gen()


def _demo(messages: list[dict]):
    q = messages[-1]["content"].lower() if messages else ""
    head = "**Local demo mode.** This reply comes from a tiny rule-based helper, *not* a language model.\n\n"
    if q.strip() in ("hi", "hello", "hey") or q.startswith(("hello", "hi ", "hey ")):
        body = "Hello! I can point you to the right Nexora module."
    elif "csv" in q or "data" in q or "analy" in q:
        body = "Open **Analytics Studio** to upload a CSV/XLSX, see statistics, build charts and clean data."
    elif any(w in q for w in ("sentiment", "summar", "nlp", "keyword")):
        body = "The **NLP Lab** has preprocessing, sentiment, classification, summarization and keyword tools."
    elif any(w in q for w in ("neural", "deep", "train")):
        body = "The **Deep Learning Lab** trains a small neural network on the digits dataset and shows real curves."
    else:
        body = ("I can't answer open-ended questions without a real model. Add an API key in `.env` "
                "(see README) and choose that provider, or use the other modules.")
    return iter([w + " " for w in (head + body).split(" ")])


def _clean_history(messages: list[dict]) -> list[dict]:
    """Providers (esp. Anthropic) need the conversation to start with a user turn; also drop empty turns."""
    msgs = [m for m in messages if (m.get("content") or "").strip()]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def open_stream(provider_id: str, messages: list[dict]):
    """Opens the provider connection eagerly so errors surface before streaming starts."""
    info = next((p for p in providers() if p["id"] == provider_id), None)
    if not info:
        raise AppError(400, "unknown_provider", f"Unknown provider '{provider_id}'.")
    if not info["ready"]:
        raise AppError(409, "provider_not_ready", f"{info['label']} is not configured. {info['hint']}")
    messages = _clean_history(messages)
    if provider_id == "demo":
        return _demo(messages)
    if provider_id == "anthropic":
        return _anthropic(messages)
    return _openai(provider_id, messages)


def check_provider(provider_id: str) -> str:
    """Sends a tiny real request and returns the reply text; raises AppError with a readable reason on failure."""
    info = next((p for p in providers() if p["id"] == provider_id), None)
    if not info:
        raise AppError(400, "unknown_provider", f"Unknown provider '{provider_id}'.")
    if not info["ready"]:
        raise AppError(409, "provider_not_ready", f"{info['label']} is not configured. {info['hint']}")
    if provider_id == "demo":
        return "Demo mode needs no connection."
    msgs = [{"role": "user", "content": "Reply with the single word: OK"}]
    stream = _anthropic(msgs, 20) if provider_id == "anthropic" else _openai(provider_id, msgs, 20)
    text = "".join(stream).strip()
    if not text:
        raise AppError(502, "provider_error", "The provider answered but returned no text. Check the model name.")
    return text[:80]
