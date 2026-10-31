from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core import db
from ..core.config import settings
from ..core.errors import AppError
from ..schemas import ChatRequest, RenameRequest, TestIn
from ..services import llm

router = APIRouter(prefix="/api")


@router.post("/chat")
def chat(req: ChatRequest):
    provider = req.provider or llm.default_provider()
    text = (req.message or "").strip()
    if not req.regenerate:
        if not text:
            raise AppError(422, "empty_message", "Message cannot be empty.")
        if len(text) > settings.max_message_chars:
            raise AppError(413, "message_too_long", f"Messages are limited to {settings.max_message_chars} characters.")
    elif not req.conversation_id:
        raise AppError(422, "nothing_to_regenerate", "Nothing to regenerate.")
    if req.conversation_id and not db.get_conversation(req.conversation_id):
        raise AppError(404, "conversation_not_found", "Conversation not found.")

    # Build the prompt in memory and open the provider connection BEFORE writing anything to the database,
    # so a failed request (bad key, network error...) leaves no orphan conversation or dangling message.
    past = [{"role": m["role"], "content": m["content"]} for m in db.get_messages(req.conversation_id)] \
        if req.conversation_id else []
    if req.regenerate:
        if past and past[-1]["role"] == "assistant":
            past.pop()
        if not past:
            raise AppError(422, "nothing_to_regenerate", "Nothing to regenerate.")
    else:
        past.append({"role": "user", "content": text})
    stream = llm.open_stream(provider, past[-settings.context_messages:])  # raises a readable error first

    cid = req.conversation_id
    if not cid:
        cid = db.create_conversation(text[:50])
    if req.regenerate:
        db.delete_last_assistant(cid)
    else:
        db.add_message(cid, "user", text)

    def gen():
        parts: list[str] = []
        try:
            for chunk in stream:
                parts.append(chunk)
                yield chunk
        except AppError as e:  # provider failed mid-stream: show why instead of silently cutting off
            note = f"\n\n⚠️ {e.message}"
            parts.append(note)
            yield note
        except Exception:
            note = "\n\n⚠️ The connection to the provider was interrupted."
            parts.append(note)
            yield note
        finally:  # also runs when the user presses Stop, so partial answers are kept
            out = "".join(parts)
            if out:
                db.add_message(cid, "assistant", out, provider)

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"X-Conversation-Id": cid, "X-Provider": provider,
                                      "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/chat/test")
def chat_test(body: TestIn):
    """Makes one tiny real request so the Settings page can show whether a provider actually works."""
    reply = llm.check_provider(body.provider)
    return {"ok": True, "reply": reply}


@router.get("/conversations")
def list_conversations():
    return db.list_conversations()


@router.get("/conversations/{cid}")
def get_conversation(cid: str):
    conv = db.get_conversation(cid)
    if not conv:
        raise AppError(404, "conversation_not_found", "Conversation not found.")
    return {**conv, "messages": db.get_messages(cid)}


@router.patch("/conversations/{cid}")
def rename(cid: str, body: RenameRequest):
    if not db.rename_conversation(cid, body.title.strip()):
        raise AppError(404, "conversation_not_found", "Conversation not found.")
    return {"ok": True}


@router.delete("/conversations/{cid}")
def delete(cid: str):
    if not db.delete_conversation(cid):
        raise AppError(404, "conversation_not_found", "Conversation not found.")
    return {"ok": True}
