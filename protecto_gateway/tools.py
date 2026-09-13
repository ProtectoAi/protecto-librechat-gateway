import json
import re
import uuid
import httpx
from typing import List,Dict,Any
from .config import NORMALIZE_TOOL_CALL_IDS,logger
from .protecto import _unmask_call_async

def normalize_tool_call_ids(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return tool calls with unique OpenAI-compatible ids and contiguous indexes."""
    seen_ids = set()
    normalized = []
    for call in tool_calls:
        call = dict(call)
        original = call.get("id")
        if NORMALIZE_TOOL_CALL_IDS:
            if (
                not re.fullmatch(r"call_[A-Za-z0-9]{24}", original or "")
                or original in seen_ids
            ):
                call["id"] = f"call_{uuid.uuid4().hex[:24]}"
                logger.info(
                    "Normalised tool call id %s -> %s (duplicate_or_invalid=%s)",
                    original, call["id"], original in seen_ids,
                )
        if not call.get("id"):
            call["id"] = f"call_{uuid.uuid4().hex[:24]}"
        seen_ids.add(call["id"])
        normalized.append(call)

    for index, call in enumerate(normalized):
        call["index"] = index
    return normalized


async def unmask_tool_calls_async(
        client: httpx.AsyncClient,
        tool_calls: List[Dict[str, Any]],
        protecto_unmask_url: str,
        headers: Dict,
) -> List[Dict[str, Any]]:
    """
    The model only ever sees masked text, so the tool_calls it produces carry
    masked argument values. LibreChat/MCP need REAL values to actually run the
    tool, so unmask arguments on the way out.

    The masked form is not kept anywhere: on the next request LibreChat
    replays these (unmasked) tool_calls back, and they are re-masked live in
    build_masked_history()'s pending region before going upstream.
    """
    unmasked = []
    for order, tc in enumerate(tool_calls):
        fn = dict(tc.get("function") or {})
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, str) and "<" in arguments:
            arguments = await _unmask_call_async(
                client, arguments, order, protecto_unmask_url, headers,
            )
        fn["arguments"] = arguments
        unmasked.append({**tc, "function": fn})
    return unmasked
