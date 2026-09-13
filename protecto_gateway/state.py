"""Short-lived Gemini interaction state for stateful tool continuations."""
from typing import Any
from .config import logger

_STATE: dict[str, dict[str, Any]] = {}
_MAX_STATE = 4000

def remember_gemini_turn(client_calls: list[dict], interaction_id: str | None) -> None:
    if not client_calls or not interaction_id:
        logger.warning("Gemini tool turn missing call ids or interaction id")
        return
    call_ids = {c["id"] for c in client_calls if c.get("id")}
    provider_ids = {c["id"]: c.get("provider_id", c["id"]) for c in client_calls if c.get("id")}
    entry = {"interaction_id": interaction_id, "call_ids": call_ids, "provider_ids": provider_ids}
    for call_id in call_ids:
        _STATE[call_id] = entry
    while len(_STATE) > _MAX_STATE:
        _STATE.pop(next(iter(_STATE)))
    logger.info(
        "[GEMINI STATE STORE] interaction=%s call_ids=%s provider_ids=%s state_entries=%d",
        interaction_id, sorted(call_ids), provider_ids, len(_STATE),
    )

def lookup_gemini_state(messages: list[dict]) -> dict[str, Any] | None:
    logger.info("[GEMINI STATE LOOKUP] messages=%d", len(messages))
    for mi, msg in enumerate(reversed(messages)):
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or ""
            state = _STATE.get(call_id)
            logger.info(
                "[GEMINI STATE LOOKUP] message=%d call_id=%s found=%s",
                mi, call_id, bool(state),
            )
            if state:
                logger.info(
                    "[GEMINI STATE LOOKUP HIT] interaction=%s call_ids=%s provider_ids=%s",
                    state.get("interaction_id"), sorted(state.get("call_ids", set())), state.get("provider_ids", {}),
                )
                return state
    logger.info("[GEMINI STATE LOOKUP] MISS")
    return None

def forget_gemini_turn(messages: list[dict]) -> None:
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            _STATE.pop(call.get("id") or "", None)
