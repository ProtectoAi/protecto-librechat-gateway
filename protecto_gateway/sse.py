"""SSE formatting helpers for the OpenAI-compatible public API."""
import json
from typing import Any, Dict
from .config import LOG_SSE, logger


def chat_chunk(delta: Dict[str, Any], finish_reason: str | None,
               response_id: str, model: str, created: int) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    if LOG_SSE:
        logger.info("[SSE OUT] %s", encoded)
    return f"data: {encoded}\n\n"


def completed_chat_text_response(
    text: str,
    response_id: str,
    model: str,
    created: int,
) -> dict[str, Any]:
    """Return a short message as a normal Chat Completions response."""
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def masking_progress_chunk(
    message: str,
    tools_enabled: bool,
    response_id: str,
    model: str,
    created: int,
) -> str:
    """Keep tool turns content-free while still keeping their SSE stream alive."""
    if tools_enabled:
        return f": protecto-progress {message}\n\n"
    return chat_chunk(
        {"content": f"\n\n_⏳ {message}_\n\n"},
        None,
        response_id,
        model,
        created,
    )


async def completed_chat_text_stream(
    text: str,
    response_id: str,
    model: str,
    created: int,
):
    """Return a short message as a normal completed assistant stream."""
    yield chat_chunk(
        {"role": "assistant", "content": text},
        None,
        response_id,
        model,
        created,
    )
    yield chat_chunk({}, "stop", response_id, model, created)
    yield "data: [DONE]\n\n"


def responses_event(event_type: str, payload: dict, sequence: int) -> str:
    body = {"type": event_type, "sequence_number": sequence, **payload}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
