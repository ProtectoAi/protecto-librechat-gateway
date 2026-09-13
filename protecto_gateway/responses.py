from typing import *
import json
import time
import uuid
from typing import List,Dict,Any,Tuple
from .config import logger
from .streaming import core_unmasked_stream
from uuid import uuid4

def _responses_event(event_type: str, payload: dict, seq: int) -> str:
    body = {"type": event_type, "sequence_number": seq, **payload}
    return (
        f"event: {event_type}\n"
        "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
    )


def _responses_object(
        response_id: str,
        model_label: str,
        created: int,
        output: list,
        status: str = "completed",
        usage: dict | None = None,
) -> dict:
    """
    A spec-shaped Response object. The optional fields matter: clients (and
    LangChain, which LibreChat's agent layer wraps) read `usage`, `error` and
    `incomplete_details` off this object, and an unexpected `undefined` there
    can abort the run - which the UI reports as a cancelled tool call rather
    than as a parse error.
    """
    text_out = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model_label,
        "output": output,
        "output_text": text_out,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "top_p": None,
        "truncation": "disabled",
        "metadata": {},
        "text": {"format": {"type": "text"}},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage or {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def response_text_output(response_id: str, text: str) -> list[dict]:
    """Build a completed Responses API assistant text output item."""
    return [{
        "id": f"msg_{response_id}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
        }],
    }]


async def completed_response_text_stream(
    text: str,
    response_id: str,
    model_label: str,
    created: int,
):
    """Return a short message as a normal completed Responses API stream."""
    item = response_text_output(response_id, text)[0]
    in_progress_item = {**item, "status": "in_progress", "content": []}
    yield _responses_event(
        "response.created",
        {"response": _responses_object(
            response_id, model_label, created, [], status="in_progress",
        )},
        1,
    )
    yield _responses_event(
        "response.output_item.added",
        {"output_index": 0, "item": in_progress_item},
        2,
    )
    yield _responses_event(
        "response.output_text.delta",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        3,
    )
    yield _responses_event(
        "response.output_text.done",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
        4,
    )
    yield _responses_event(
        "response.output_item.done",
        {"output_index": 0, "item": item},
        5,
    )
    yield _responses_event(
        "response.completed",
        {"response": _responses_object(
            response_id, model_label, created, [item],
        )},
        6,
    )


async def stream_responses_generator(
        provider: str,
        model: str,
        model_label: str,
        masked_messages: List[Dict[str, Any]],
        protecto_unmask_url: str,
        headers: Dict,
        provider_api_key: str,
        token_map: Dict[str, str],
        response_id: str,
        tools=None,
        tool_choice=None,
):
    created = int(time.time())
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    msg_item_id = f"msg_{response_id}"
    text_parts: List[str] = []
    message_open = False
    output_index = 0

    yield _responses_event(
        "response.created",
        {"response": _responses_object(
            response_id, model_label, created, [], status="in_progress",
        )},
        nxt(),
    )

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
            if not message_open:
                message_open = True
                yield _responses_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {
                            "id": msg_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                    nxt(),
                )
            text_parts.append(ev["text"])
            yield _responses_event(
                "response.output_text.delta",
                {
                    "item_id": msg_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": ev["text"],
                },
                nxt(),
            )

        elif kind == "tool_calls":
            # A tool turn never carries assistant text, so no message item is
            # opened; emit one function_call item per call.
            output = []
            for call in ev["calls"]:
                fn = call.get("function") or {}
                item_id = f"fc_{call['id']}"
                item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call["id"],
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", "{}"),
                }
                yield _responses_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {**item, "status": "in_progress", "arguments": ""},
                    },
                    nxt(),
                )
                yield _responses_event(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": item["arguments"],
                    },
                    nxt(),
                )
                yield _responses_event(
                    "response.function_call_arguments.done",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": item["arguments"],
                    },
                    nxt(),
                )
                yield _responses_event(
                    "response.output_item.done",
                    {"output_index": output_index, "item": item},
                    nxt(),
                )
                output.append(item)
                output_index += 1
            yield _responses_event(
                "response.completed",
                {"response": _responses_object(
                    response_id, model_label, created, output,
                )},
                nxt(),
            )
            logger.info(
                "[SSE END] /v1/responses tool_calls terminal sent "
                "(%d call(s), call_ids=%s). Awaiting LibreChat's tool results.",
                len(output),
                [i.get("call_id") for i in output],
            )
            return

        elif kind in {"end", "error"}:
            if kind == "error":
                err = f"\n\n_[Protecto gateway error: {ev['message']}]_"
                if not message_open:
                    message_open = True
                    yield _responses_event(
                        "response.output_item.added",
                        {
                            "output_index": output_index,
                            "item": {
                                "id": msg_item_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                        nxt(),
                    )
                text_parts.append(err)
                yield _responses_event(
                    "response.output_text.delta",
                    {
                        "item_id": msg_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": err,
                    },
                    nxt(),
                )
            else:
                # Carry the masked-history artifact in the assistant text, the
                # same way the Chat Completions path does.
                artifact = ev.get("artifact") or ""
                if artifact:
                    if not message_open:
                        message_open = True
                        yield _responses_event(
                            "response.output_item.added",
                            {
                                "output_index": output_index,
                                "item": {
                                    "id": msg_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                            nxt(),
                        )
                    text_parts.append(artifact)
                    yield _responses_event(
                        "response.output_text.delta",
                        {
                            "item_id": msg_item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": artifact,
                        },
                        nxt(),
                    )

            full_text = "".join(text_parts)
            output = []
            if message_open:
                yield _responses_event(
                    "response.output_text.done",
                    {
                        "item_id": msg_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": full_text,
                    },
                    nxt(),
                )
                item = {
                    "id": msg_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }],
                }
                yield _responses_event(
                    "response.output_item.done",
                    {"output_index": output_index, "item": item},
                    nxt(),
                )
                output.append(item)
            yield _responses_event(
                "response.completed",
                {"response": _responses_object(
                    response_id, model_label, created, output,
                )},
                nxt(),
            )
            return


def responses_request_to_messages(body: dict) -> Tuple[List[Dict[str, Any]], list]:
    """
    Convert an incoming /v1/responses request into the Chat-Completions-shaped
    messages + tools this gateway works with internally, so both endpoints
    share one pipeline.

      instructions          -> system message
      input (str)           -> single user message
      input[] role items    -> user/assistant messages
      input[] function_call -> assistant.tool_calls
      input[] function_call_output -> tool message
      tools[] (flat)        -> nested {"type":"function","function":{...}}
    """
    messages: List[Dict[str, Any]] = []

    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        raw_input = []
    elif raw_input is None:
        raw_input = []

    pending_tool_calls: List[Dict[str, Any]] = []

    def flush_tool_calls():
        if pending_tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": list(pending_tool_calls),
            })
            pending_tool_calls.clear()

    for item in raw_input:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                },
            })
            continue

        flush_tool_calls()

        if itype == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id"),
                "content": output,
            })
            continue

        role = item.get("role")
        if role not in {"user", "assistant", "system", "developer"}:
            logger.warning("Ignoring unsupported Responses input item: %s", itype)
            continue

        content = item.get("content")
        if isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {
                    "input_text", "output_text", "text", "summary_text",
                }:
                    texts.append(block.get("text", ""))
            content = "".join(texts)
        if content:
            messages.append({"role": role, "content": content})

    flush_tool_calls()

    converted_tools = []
    for tool in body.get("tools") or []:
        if tool.get("type") != "function":
            logger.warning("Ignoring non-function Responses tool: %s", tool.get("type"))
            continue
        # Responses tools are flat; the rest of the gateway expects nested.
        if "function" in tool:
            converted_tools.append(tool)
            continue
        converted_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                **({"strict": tool["strict"]} if "strict" in tool else {}),
            },
        })

    return messages, converted_tools
