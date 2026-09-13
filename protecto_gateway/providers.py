import json
import httpx
from typing import Any,Dict,List
from .config import OPENAI_API_URL,GEMINI_INTERACTIONS_URL,logger
from .openai_provider import build_openai_payload,extract_openai_response_text,extract_openai_tool_calls
from .gemini import build_gemini_payload,extract_gemini_interaction_text,extract_gemini_tool_calls


async def _openai_stream_async(openai_payload: dict, openai_headers: dict):
    """True Responses API streaming for text and function calls."""
    logger.info("Entered _openai_stream_async (text + tools)")
    calls = {}
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", OPENAI_API_URL, json=openai_payload, headers=openai_headers,
        ) as res:
            if res.status_code != 200:
                body = (await res.aread()).decode(errors="replace")
                logger.error("[OPENAI RESPONSES ERROR] %s %s", res.status_code, body)
                raise RuntimeError(f"OpenAI HTTP {res.status_code}: {body}")
            async for line in res.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                event = json.loads(raw)
                typ = event.get("type")
                if typ == "response.output_text.delta":
                    if event.get("delta"):
                        yield {"type": "text", "text": event["delta"]}
                elif typ == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        idx = event.get("output_index", 0)
                        calls[idx] = {
                            "id": item.get("call_id") or item.get("id"),
                            "name": item.get("name"),
                            "arguments": item.get("arguments", ""),
                        }
                        # Signal the tool turn as soon as the call STARTS, so
                        # downstream can stop emitting text immediately rather
                        # than waiting for arguments to finish streaming.
                        yield {"type": "tool_pending"}
                elif typ == "response.function_call_arguments.delta":
                    idx = event.get("output_index", 0)
                    c = calls.setdefault(
                        idx,
                        {
                            "id": event.get("call_id"),
                            "name": event.get("name"),
                            "arguments": "",
                        },
                    )
                    c["arguments"] += event.get("delta", "")
                elif typ == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        idx = event.get("output_index", 0)
                        c = calls.pop(idx, {})
                        yield {
                            "type": "tool_call",
                            "id": item.get("call_id") or c.get("id") or item.get("id"),
                            "name": item.get("name") or c.get("name"),
                            "arguments": (
                                item.get("arguments") or c.get("arguments") or "{}"
                            ),
                        }
                elif typ == "response.completed":
                    yield {"type": "completed"}
                    return
                elif typ in {"response.failed", "error"}:
                    raise RuntimeError(f"OpenAI stream error: {event}")
    yield {"type": "completed"}



async def _gemini_stream_async(gemini_payload:dict,gemini_api_key:str):
    """Stream Gemini Interactions events with detailed diagnostic logging."""
    headers={"x-goog-api-key":gemini_api_key,"Content-Type":"application/json","Accept":"text/event-stream"}

    logger.info("[GEMINI DEBUG] ===== STREAM START =====")
    logger.info("[GEMINI DEBUG] model=%s stream=%s", gemini_payload.get("model"), gemini_payload.get("stream"))
    logger.info("[GEMINI DEBUG] previous_interaction_id=%s", gemini_payload.get("previous_interaction_id"))
    logger.info("[GEMINI DEBUG] tools=%d", len(gemini_payload.get("tools") or []))
    logger.info("[GEMINI DEBUG] input_steps=%d", len(gemini_payload.get("input") or []))
    for i, step in enumerate(gemini_payload.get("input") or []):
        logger.info(
            "[GEMINI DEBUG] INPUT[%d] type=%s keys=%s call_id=%s name=%s",
            i, step.get("type"), sorted(step.keys()), step.get("call_id"), step.get("name"),
        )
    if gemini_payload.get("system_instruction"):
        logger.info("[GEMINI DEBUG] system_instruction present len=%d", len(gemini_payload["system_instruction"]))
    else:
        logger.info("[GEMINI DEBUG] system_instruction NOT present in payload")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0,connect=10.0)) as client:
        async with client.stream("POST",GEMINI_INTERACTIONS_URL,json=gemini_payload,headers=headers) as res:
            logger.info("[GEMINI DEBUG] HTTP status=%s", res.status_code)
            if res.status_code!=200:
                body=(await res.aread()).decode(errors="replace")
                logger.error("[GEMINI STREAM ERROR] %s %s",res.status_code,body)
                logger.error("[GEMINI PAYLOAD] %s",json.dumps(gemini_payload,ensure_ascii=False)[:8000])
                raise RuntimeError(f"Gemini HTTP {res.status_code}: {body}")

            current_step=None
            event_count=0
            function_call_count=0
            interaction_id_seen=None

            async for line in res.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw=line[5:].strip()
                if not raw or raw=="[DONE]":
                    if raw=="[DONE]":
                        logger.info("[GEMINI DEBUG] RECEIVED [DONE]")
                    continue

                event_count += 1
                try:
                    event=json.loads(raw)
                except Exception:
                    logger.exception("[GEMINI DEBUG] Failed to parse SSE event #%d raw=%r", event_count, raw[:4000])
                    continue

                et=event.get("event_type")
                interaction_id=(event.get("interaction") or {}).get("id") or event.get("interaction_id")
                logger.info(
                    "[GEMINI DEBUG] EVENT #%d type=%s keys=%s interaction_id=%s",
                    event_count, et, sorted(event.keys()), interaction_id,
                )

                # Full event is intentionally logged at debug level so the exact
                # difference between single- and multi-call streams is visible.
                logger.debug("[GEMINI DEBUG] EVENT #%d BODY=%s", event_count, json.dumps(event, ensure_ascii=False))

                if interaction_id:
                    interaction_id_seen=interaction_id
                    logger.info("[GEMINI DEBUG] interaction_id=%s", interaction_id)
                    yield {"type":"interaction_id","id":interaction_id}

                if et=="step.start":
                    current_step=dict(event.get("step") or {})
                    logger.info(
                        "[GEMINI DEBUG] STEP.START type=%s id=%s name=%s keys=%s",
                        current_step.get("type"), current_step.get("id") or current_step.get("call_id"),
                        current_step.get("name"), sorted(current_step.keys()),
                    )
                    logger.debug("[GEMINI DEBUG] STEP.START BODY=%s", json.dumps(current_step, ensure_ascii=False))
                    if current_step.get("type")=="function_call":
                        logger.info("[GEMINI DEBUG] TOOL_PENDING id=%s name=%s", current_step.get("id") or current_step.get("call_id"), current_step.get("name"))
                        yield {"type":"tool_pending"}

                elif et=="step.delta":
                    delta=event.get("delta") or {}
                    logger.info(
                        "[GEMINI DEBUG] STEP.DELTA current_type=%s delta_type=%s keys=%s",
                        (current_step or {}).get("type"), delta.get("type"), sorted(delta.keys()),
                    )
                    logger.debug("[GEMINI DEBUG] STEP.DELTA BODY=%s", json.dumps(delta, ensure_ascii=False))
                    if delta.get("type")=="text" and delta.get("text"):
                        yield {"type":"text","text":delta["text"]}
                    if current_step is not None:
                        if delta.get("arguments") is not None:
                            previous=current_step.get("arguments")
                            current_step["arguments"]=delta["arguments"]
                            logger.info(
                                "[GEMINI DEBUG] ARGUMENTS UPDATE old_len=%s new_len=%s value=%r",
                                len(previous) if isinstance(previous,str) else None,
                                len(current_step["arguments"]) if isinstance(current_step["arguments"],str) else None,
                                current_step["arguments"] if len(str(current_step["arguments"])) < 1000 else str(current_step["arguments"])[:1000] + "...",
                            )

                elif et=="step.stop":
                    step={**(current_step or {}),**(event.get("step") or {})}
                    logger.info(
                        "[GEMINI DEBUG] STEP.STOP type=%s id=%s name=%s keys=%s",
                        step.get("type"), step.get("id") or step.get("call_id"), step.get("name"), sorted(step.keys()),
                    )
                    logger.debug("[GEMINI DEBUG] STEP.STOP BODY=%s", json.dumps(step, ensure_ascii=False))
                    if step.get("type")=="function_call":
                        function_call_count += 1
                        args=step.get("arguments",{})
                        if not isinstance(args,str):
                            args=json.dumps(args,ensure_ascii=False)
                        call={"type":"tool_call","id":step.get("id") or step.get("call_id"),"name":step.get("name"),"arguments":args}
                        logger.info(
                            "[GEMINI DEBUG] FUNCTION_CALL #%d id=%s name=%s args=%s",
                            function_call_count, call.get("id"), call.get("name"), call.get("arguments"),
                        )
                        logger.info("[GEMINI DEBUG] ABOUT TO YIELD tool_call #%d", function_call_count)
                        yield call
                        logger.info("[GEMINI DEBUG] RETURNED FROM YIELD tool_call #%d", function_call_count)
                    else:
                        logger.info("[GEMINI DEBUG] STEP.STOP non-tool type=%s", step.get("type"))
                    current_step=None

                elif et=="interaction.completed":
                    logger.info(
                        "[GEMINI DEBUG] INTERACTION COMPLETED events=%d function_calls=%d interaction_id=%s",
                        event_count, function_call_count, interaction_id_seen,
                    )
                    yield {"type":"completed"}
                    logger.info("[GEMINI DEBUG] ===== STREAM END (interaction.completed) =====")
                    return

                elif et=="error":
                    logger.error("[GEMINI DEBUG] GEMINI ERROR EVENT=%s", json.dumps(event, ensure_ascii=False))
                    raise RuntimeError(f"Gemini stream error: {event}")
                else:
                    logger.info("[GEMINI DEBUG] UNHANDLED EVENT TYPE=%s", et)

    logger.info(
        "[GEMINI DEBUG] HTTP STREAM CLOSED events=%d function_calls=%d interaction_id=%s",
        event_count, function_call_count, interaction_id_seen,
    )
    logger.info("[GEMINI DEBUG] ===== STREAM END (connection close) =====")
    yield {"type":"completed"}

def create_provider_stream(provider,model,masked_messages,provider_api_key,tools=None,tool_choice=None,gemini_state=None):
    provider=provider.lower()
    if provider=="openai":
        return _openai_stream_async(build_openai_payload(model,masked_messages,tools,tool_choice,True),{"Authorization":f"Bearer {provider_api_key}","Content-Type":"application/json"})
    if provider=="gemini":
        return _gemini_stream_async(build_gemini_payload(model,masked_messages,tools,True,gemini_state),provider_api_key)
    raise ValueError(f"Unsupported provider: {provider}")

async def call_provider_non_streaming(provider,model,masked_messages,provider_api_key,tools=None,tool_choice=None,gemini_state=None):
    provider=provider.lower()
    async with httpx.AsyncClient(timeout=60.0) as client:
        if provider=="openai":
            payload=build_openai_payload(model,masked_messages,tools,tool_choice,False)
            res=await client.post(OPENAI_API_URL,json=payload,headers={"Authorization":f"Bearer {provider_api_key}","Content-Type":"application/json"})
        elif provider=="gemini":
            payload=build_gemini_payload(model,masked_messages,tools,False,gemini_state)
            res=await client.post(GEMINI_INTERACTIONS_URL,json=payload,headers={"x-goog-api-key":provider_api_key,"Content-Type":"application/json"})
        else: raise ValueError(f"Unsupported provider: {provider}")
        if res.status_code!=200: logger.error("[%s ERROR] %s: %s",provider.upper(),res.status_code,res.text)
        res.raise_for_status(); return res.json()

def extract_provider_tool_calls(provider,response_json):
    return extract_openai_tool_calls(response_json) if provider=="openai" else extract_gemini_tool_calls(response_json)

def extract_provider_text(provider,response_json):
    return extract_openai_response_text(response_json) if provider=="openai" else extract_gemini_interaction_text(response_json)
