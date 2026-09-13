from typing import *
import json
import re
import traceback
import time
from typing import List,Dict,Any
import httpx
from .config import *
from .state import remember_gemini_turn,lookup_gemini_state,forget_gemini_turn
from .tools import normalize_tool_call_ids,unmask_tool_calls_async
from .protecto import _unmask_call_async, decode_json_unicode_escapes
from .artifacts import build_behind_scenes_artifact
from .providers import create_provider_stream
from .sse import chat_chunk as sse_chunk


_MASKED_ENTITY_RE = re.compile(r"<[^>]+>.*?</[^>]+>", re.DOTALL)
_MASKED_ENTITY_VALUE_RE = re.compile(
    r"<[^>/]+>(.*?)</[^>]+>",
    re.DOTALL,
)
_CLOSING_TAG_RE = re.compile(r"</[^>]+>")


def _replace_first_masked_entity(text: str, original_value: str) -> str:
    """Insert an original value literally, without parsing backslash escapes."""
    decoded_value = decode_json_unicode_escapes(original_value)
    return _MASKED_ENTITY_RE.sub(lambda _match: decoded_value, text, count=1)


def _masked_entity_value(text: str) -> str:
    """Extract a token even when the LLM changed the closing tag name."""
    match = _MASKED_ENTITY_VALUE_RE.search(text)
    return match.group(1) if match else ""


async def _entity_safe_split_async(async_gen):
    buffer = ""
    async for chunk in async_gen:
        buffer += chunk
        while True:
            if "<" not in buffer:
                yield buffer, False
                buffer = ""
                break
            if len(buffer) >= (CHUNK_SIZE + OVERLAP) or buffer.endswith(END_TAG):
                window = buffer[: CHUNK_SIZE + OVERLAP]
                found = None
                idx_start = len(buffer)
                for start, end in START_TAGS.items():
                    idx = window.find(start)
                    if idx != -1 and idx < idx_start:
                        found = (start, end, idx)
                        idx_start = idx
                if not found:
                    piece = buffer[:CHUNK_SIZE]
                    buffer = buffer[CHUNK_SIZE:]
                    yield piece, False
                else:
                    start, end, idx = found
                    closing_match = _CLOSING_TAG_RE.search(
                        buffer,
                        idx + len(start),
                    )
                    if closing_match is None:
                        break
                    actual_end = closing_match.group(0)
                    if actual_end != end:
                        logger.warning(
                            "[MASKED TAG MISMATCH] expected=%s received=%s",
                            end,
                            actual_end,
                        )
                    piece = buffer[:closing_match.end()]
                    buffer = buffer[closing_match.end():]
                    yield piece, True
            else:
                break
    if buffer:
        yield buffer, False


async def core_unmasked_stream(
        provider: str,
        model: str,
        masked_messages: List[Dict[str, Any]],
        protecto_unmask_url: str,
        headers: Dict,
        provider_api_key: str,
    token_map: Dict[str, str],
    tools=None,
    tool_choice=None,
):
    logger.info("Entered core_unmasked_stream")
    raw_output = ""
    order_counter = 0
    tool_calls: List[Dict[str, Any]] = []
    gemini_tool_event_count = [0]
    provider_completed = False
    tool_turn = False
    hold_text = bool(tools) and BUFFER_TEXT_WHEN_TOOLS
    held_chunks: List[str] = []
    nonlocal_state: Dict[str, Any] = {}

    try:
        raw_gen = create_provider_stream(
            provider=provider,
            model=model,
            masked_messages=masked_messages,
            provider_api_key=provider_api_key,
            tools=tools,
            tool_choice=tool_choice,
            gemini_state=lookup_gemini_state(masked_messages),
        )

        async def text_gen():
            nonlocal provider_completed, tool_turn
            async for event in raw_gen:
                event_type = event.get("type")
                if event_type == "text":
                    if not tool_turn:
                        yield event.get("text", "")
                elif event_type == "interaction_id":
                    nonlocal_state["interaction_id"] = event.get("id")
                elif event_type == "tool_pending":
                    # A tool call has STARTED. Stop emitting text right here,
                    # before its arguments have even finished streaming.
                    tool_turn = True
                elif event_type == "tool_call":
                    gemini_tool_event_count[0] += 1
                    tool_turn = True
                    logger.info(
                        "[TOOL BUFFER] received provider tool_call #%d id=%s name=%s args=%s",
                        gemini_tool_event_count[0], event.get("id"), event.get("name"), event.get("arguments"),
                    )
                    tool_calls.append(event)
                    logger.info(
                        "[TOOL BUFFER] buffered tool_calls=%d ids=%s",
                        len(tool_calls), [c.get("id") for c in tool_calls],
                    )
                elif event_type == "completed":
                    provider_completed = True
                    logger.info(
                        "[PROVIDER COMPLETED] tool_turn=%s buffered_tool_calls=%d ids=%s",
                        tool_turn, len(tool_calls), [c.get("id") for c in tool_calls],
                    )
                    # END_TAG belongs only to an ordinary text completion.
                    if not tool_turn:
                        yield END_TAG
                    return

        entity_stream = _entity_safe_split_async(text_gen())

        async with httpx.AsyncClient() as client:
            async for piece, has_entity in entity_stream:
                if tool_turn:
                    held_chunks.clear()
                    continue

                fixed_piece = piece
                piece_value = ""
                if has_entity or piece.endswith(END_TAG):
                    piece_value = _masked_entity_value(piece)
                    if piece_value and piece_value in token_map:
                        fixed_piece = _replace_first_masked_entity(
                            piece,
                            token_map[piece_value],
                        )
                    else:
                        fixed_piece = await _unmask_call_async(
                            client, piece, order_counter, protecto_unmask_url, headers,
                        )

                clean_output = (
                    fixed_piece.rsplit(END_TAG, 1)[0]
                    if END_TAG in fixed_piece
                    else fixed_piece
                )
                raw_piece = (
                    piece.rsplit(END_TAG, 1)[0] if END_TAG in piece else piece
                )
                raw_output += raw_piece

                if clean_output:
                    if hold_text:
                        held_chunks.append(clean_output)
                    else:
                        yield {"kind": "text", "text": clean_output}
                order_counter += 1

            # ------------------ TOOL CALL TERMINAL ------------------
            logger.info(
                "[TOOL BUFFER FINAL] provider=%s buffered=%d ids=%s provider_completed=%s",
                provider, len(tool_calls), [c.get("id") for c in tool_calls], provider_completed,
            )
            if tool_calls:
                delta_calls = []
                for index, call in enumerate(tool_calls):
                    call_id = call.get("id")
                    name = call.get("name")
                    if not call_id or not name:
                        logger.warning("Ignoring incomplete streamed tool call: %s", call)
                        continue
                    arguments = call.get("arguments") or "{}"
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    delta_calls.append({
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    })

                if not delta_calls:
                    logger.error(
                        "Provider signalled a tool-call turn but no complete "
                        "tool call could be constructed."
                    )
                    yield {
                        "kind": "error",
                        "message": (
                            "The model's tool call could not be parsed. "
                            "Please try again."
                        ),
                    }
                    return

                logger.info(
                    "[TOOL NORMALIZE INPUT] delta_calls=%s",
                    json.dumps(delta_calls, ensure_ascii=False),
                )
                if provider.lower() == "gemini":
                    for call in delta_calls:
                        call["provider_id"] = call.get("id")
                delta_calls = normalize_tool_call_ids(delta_calls)
                logger.info(
                    "[TOOL NORMALIZE OUTPUT] delta_calls=%s",
                    json.dumps(delta_calls, ensure_ascii=False),
                )
                if provider.lower() == "gemini":
                    logger.info(
                        "[GEMINI STATE BEFORE STORE] interaction_id=%s call_ids=%s",
                        nonlocal_state.get("interaction_id"), [c.get("id") for c in delta_calls],
                    )
                    remember_gemini_turn(delta_calls, nonlocal_state.get("interaction_id"))
                    for call in delta_calls:
                        call.pop("provider_id", None)

                # LibreChat/MCP need real values to actually execute the tool.
                delta_calls = await unmask_tool_calls_async(
                    client, delta_calls, protecto_unmask_url, headers,
                )
                logger.info(
                    "[TOOL UNMASK OUTPUT] count=%d calls=%s",
                    len(delta_calls), json.dumps(delta_calls, ensure_ascii=False),
                )
                logger.info("Emitting %d tool call(s)", len(delta_calls))
                yield {"kind": "tool_calls", "calls": delta_calls}
                return

        # ------------------ TEXT TERMINAL ------------------
        if not provider_completed:
            logger.warning("Provider stream ended without explicit completed event.")

        forget_gemini_turn(masked_messages)

        if hold_text:
            for held_chunk in held_chunks:
                yield {"kind": "text", "text": held_chunk}

        yield {
            "kind": "end",
            "raw_output": raw_output,
            "artifact": build_behind_scenes_artifact(
                masked_messages=masked_messages,
                token_map=token_map,
                assistant_raw=raw_output,
            ),
        }

    except Exception as e:
        logger.error("[STREAM EXCEPTION] %s: %s", type(e).__name__, e)
        logger.error(traceback.format_exc())
        yield {"kind": "error", "message": f"{type(e).__name__}: {e}"}


async def stream_and_unmask_generator(
        provider: str,
        model: str,
        model_label: str,
        masked_messages: List[Dict[str, Any]],
        protecto_unmask_url: str,
        headers: Dict,
        provider_api_key: str,
        token_map: Dict[str, str],
        response_id: str,
        conversation_id: str | None = None,
        tools=None,
        tool_choice=None,
):
    created = int(time.time())
    async for ev in core_unmasked_stream(
        provider=provider,
        model=model,
        masked_messages=masked_messages,
        protecto_unmask_url=protecto_unmask_url,
        headers=headers,
        provider_api_key=provider_api_key,
        token_map=token_map,
        tools=tools,
        tool_choice=tool_choice,
    ):
        kind = ev["kind"]
        if kind == "text":
            yield sse_chunk(
                {"content": ev["text"]}, None, response_id, model_label, created,
            )
        elif kind == "tool_calls":
            calls = ev["calls"]
            logger.info("Emitting %d tool call(s)", len(calls))

            # Emit one OpenAI Chat Completions SSE delta per tool call.
            # LibreChat accumulates calls by index; this matches the stream
            # shape used by the working OpenAI multi-tool path.
            for position, call in enumerate(calls):
                tool_delta = {
                    "tool_calls": [{
                        "index": position,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"].get("arguments", "{}"),
                        },
                    }],
                }
                if position == 0:
                    tool_delta["role"] = "assistant"
                    tool_delta["content"] = None

                logger.info(
                    "[TOOL CALL CHUNK] %s",
                    json.dumps(tool_delta["tool_calls"], ensure_ascii=False),
                )
                logger.info(
                    "[SSE OUT TOOL] position=%d role=%s id=%s name=%s args=%s",
                    position, tool_delta.get("role"), call.get("id"),
                    call.get("function", {}).get("name"), call.get("function", {}).get("arguments"),
                )

                chunk_delta = sse_chunk(
                    tool_delta, None, response_id, model_label, created,
                )

                logger.info("SSE CHUNK DELTA: %s", chunk_delta.rstrip())
                yield chunk_delta
                # yield sse_chunk(
                #     tool_delta, None, response_id, model_label, created,
                # )

            logger.info("[SSE OUT TERMINAL] finish_reason=tool_calls count=%d", len(calls))
            chunk = sse_chunk({}, "tool_calls", response_id, model_label, created)
            logger.info("SSE TERMINAL CHUNK: %s", chunk.rstrip())
            yield chunk
            # yield sse_chunk({}, "tool_calls", response_id, model_label, created)
            logger.info(
                "[SSE END] chat/completions tool_calls terminal sent (%d call(s), ids=%s)",
                len(calls), [c.get("id") for c in calls],
            )
            yield "data: [DONE]\n\n"
            return
        elif kind == "end":
            yield sse_chunk(
                {"content": ev["artifact"]},
                "stop",
                response_id,
                model_label,
                created,
            )
            yield "data: [DONE]\n\n"
            return
        elif kind == "error":
            yield sse_chunk({"content": f"\n\n_[Protecto gateway error: {ev['message']}]_"}, None, response_id, model_label, created)
            yield sse_chunk({}, "stop", response_id, model_label, created)
            yield "data: [DONE]\n\n"
            return
