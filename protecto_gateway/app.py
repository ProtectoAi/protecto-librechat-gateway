import asyncio
import json
import time
import traceback
from contextlib import suppress
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .artifacts import (
    build_behind_scenes_artifact,
)
from .config import (
    ASYNC_MASK_MAX_RETRIES,
    ASYNC_MASK_POLL_SECONDS,
    ASYNC_MASK_SCALE_POLL_SECONDS,
    ASYNC_MASK_SCALE_RETRIES,
    ASYNC_MASK_THRESHOLD_KB,
    BUFFER_TEXT_WHEN_TOOLS,
    CHAT_NAME_HEADER,
    GATEWAY_MODELS,
    MASK_TOOL_RESULTS,
    logger,
)
from .history import (
    PROVIDER_FILE_UNSUPPORTED_MESSAGE,
    add_chat_name_prefix,
    build_model_catalog,
    is_title_generation_request,
    latest_user_message_has_provider_file,
    log_librechat_replay,
    log_masked_replay_payload,
    prepare_request,
)
from .embeddings import create_embeddings, validate_embeddings_request
from .ocr_api import router as ocr_router
from .protecto import _unmask_call_async
from .providers import (
    call_provider_non_streaming,
    extract_provider_text,
    extract_provider_tool_calls,
)
from .responses import (
    _responses_object,
    completed_response_text_stream,
    response_text_output,
    responses_request_to_messages,
    stream_responses_generator,
)
from .sse import (
    chat_chunk as sse_chunk,
    completed_chat_text_stream,
    completed_chat_text_response,
    masking_progress_chunk,
)
from .state import forget_gemini_turn, lookup_gemini_state, remember_gemini_turn
from .streaming import stream_and_unmask_generator
from .tools import normalize_tool_call_ids, unmask_tool_calls_async


app = FastAPI(title="Protecto Gateway for LibreChat")
app.include_router(ocr_router)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    payload, provider_api_key = validate_embeddings_request(request, body)
    return await create_embeddings(payload, provider_api_key)


async def _stream_chat_after_preparation(
    request: Request,
    raw_messages: list[dict[str, Any]],
    request_tools: list,
    request_tool_choice: Any,
    model_string: str,
    response_id: str,
):
    """Start SSE immediately so large background masking can report progress."""
    progress_queue: asyncio.Queue[str] = asyncio.Queue()
    preparation_task = asyncio.create_task(
        prepare_request(
            request,
            raw_messages,
            request_tools,
            model_string,
            progress_callback=progress_queue.put,
        ),
    )
    created = int(time.time())
    tools_enabled = bool(request_tools)
    if tools_enabled:
        logger.info(
            "[ASYNC MASK PROGRESS] mode=sse-heartbeat reason=tools-enabled",
        )

    try:
        while not preparation_task.done():
            while not progress_queue.empty():
                message = progress_queue.get_nowait()
                yield masking_progress_chunk(
                    message=message,
                    tools_enabled=tools_enabled,
                    response_id=response_id,
                    model=model_string,
                    created=created,
                )
            try:
                await asyncio.wait_for(
                    asyncio.shield(preparation_task),
                    timeout=0.25,
                )
            except TimeoutError:
                pass

        while not progress_queue.empty():
            message = progress_queue.get_nowait()
            yield masking_progress_chunk(
                message=message,
                tools_enabled=tools_enabled,
                response_id=response_id,
                model=model_string,
                created=created,
            )

        ctx = await preparation_task
        masked_messages = ctx["masked_messages"]
        log_masked_replay_payload(masked_messages)
        async for chunk in stream_and_unmask_generator(
            provider=ctx["provider"],
            model=ctx["selected_model"],
            model_label=ctx["provider_selected_model"],
            masked_messages=masked_messages,
            protecto_unmask_url=ctx["protect_url"] + "/unmask",
            headers=ctx["protect_headers"],
            provider_api_key=ctx["provider_api_key"],
            token_map=ctx["token_map"],
            response_id=response_id,
            conversation_id=ctx["conversation_id"],
            tools=request_tools,
            tool_choice=request_tool_choice,
        ):
            yield chunk
    except Exception as exc:
        logger.error(
            "[STREAM PREPARATION EXCEPTION] %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.error(traceback.format_exc())
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        yield sse_chunk(
            {"content": f"\n\n_[Protecto gateway error: {detail}]_"},
            None,
            response_id,
            model_string,
            created,
        )
        yield sse_chunk({}, "stop", response_id, model_string, created)
        yield "data: [DONE]\n\n"
    finally:
        if not preparation_task.done():
            preparation_task.cancel()
            with suppress(asyncio.CancelledError):
                await preparation_task

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        logger.info("Entered chat_completions")
        body = await request.json()
        raw_messages = body.get("messages", [])
        is_stream = body.get("stream", True)
        is_title_request = is_title_generation_request(body)
        request_tools = body.get("tools") or []
        request_tool_choice = body.get("tool_choice")

        log_librechat_replay(raw_messages)

        # What LibreChat actually sent. This is the fastest way to tell a
        # genuine tool continuation (contains role="tool") apart from a
        # title-generation call (no tools, short history) when a run stalls.
        roles = [m.get("role") for m in raw_messages]
        n_tool_results = sum(1 for m in raw_messages if m.get("role") == "tool")
        # How many tool round-trips LibreChat has already completed in this
        # conversation. This is the number that matters when a multi-round run
        # dies: it says exactly which round failed to come back.
        tool_rounds = sum(
            1 for m in raw_messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        logger.info(
            "[REQUEST IN] >>> TOOL ROUND %d <<< model=%s stream=%s tools=%d "
            "tool_choice=%s messages=%d roles=%s tool_results=%d%s",
            tool_rounds + 1,
            body.get("model"),
            is_stream,
            len(request_tools),
            request_tool_choice,
            len(raw_messages),
            roles,
            n_tool_results,
            "  <<< TOOL CONTINUATION" if n_tool_results else "",
        )
        if is_title_request:
            logger.info(
                "[REQUEST CLASSIFICATION] title_generation=true reason=stream_false",
            )
        if request_tools:
            logger.info(
                "[REQUEST TOOLS] %s",
                [(t.get("function") or {}).get("name") for t in request_tools],
            )

        model_string = body.get("model", "")
        response_id = f"chatcmpl-{uuid4().hex}"
        if (
            not is_title_request
            and latest_user_message_has_provider_file(raw_messages)
        ):
            created = int(time.time())
            logger.info(
                "[PROVIDER FILE REJECTED] endpoint=chat/completions model=%s",
                model_string,
            )
            if is_stream:
                return StreamingResponse(
                    completed_chat_text_stream(
                        PROVIDER_FILE_UNSUPPORTED_MESSAGE,
                        response_id,
                        model_string,
                        created,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return completed_chat_text_response(
                PROVIDER_FILE_UNSUPPORTED_MESSAGE,
                response_id,
                model_string,
                created,
            )

        # For streaming calls, defer request preparation until after the SSE
        # connection opens. Large Protecto jobs can then report each polling
        # attempt instead of leaving LibreChat with a silent connection.
        if is_stream:
            return StreamingResponse(
                _stream_chat_after_preparation(
                    request=request,
                    raw_messages=raw_messages,
                    request_tools=request_tools,
                    request_tool_choice=request_tool_choice,
                    model_string=model_string,
                    response_id=response_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Content-Type": "text/event-stream",
                },
            )

        ctx = await prepare_request(
            request, raw_messages, request_tools, model_string,
        )
        provider = ctx["provider"]
        selected_model = ctx["selected_model"]
        provider_selected_model = ctx["provider_selected_model"]
        provider_api_key = ctx["provider_api_key"]
        protect_url = ctx["protect_url"]
        protect_headers = ctx["protect_headers"]
        masked_messages = ctx["masked_messages"]
        token_map = ctx["token_map"]
        conversation_id = ctx["conversation_id"]
        message_id = ctx["message_id"]
        log_masked_replay_payload(masked_messages)
        # response_id = f"chatcmpl-{message_id}"

        # ========================================================
        # Non-streaming
        # ========================================================
        if not is_stream:
            response_json = await call_provider_non_streaming(
                provider=provider,
                model=selected_model,
                masked_messages=masked_messages,
                provider_api_key=provider_api_key,
                tools=request_tools,
                tool_choice=request_tool_choice,
                gemini_state=lookup_gemini_state(masked_messages),
            )
            masked_tool_calls = extract_provider_tool_calls(provider, response_json)
            assistant_text = extract_provider_text(provider, response_json)

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Tool-call turn: tool_calls only, no content.
                if masked_tool_calls:
                    if provider == "gemini":
                        for call in masked_tool_calls:
                            call["provider_id"] = call.get("id")
                    masked_tool_calls = normalize_tool_call_ids(masked_tool_calls)
                    if provider == "gemini":
                        remember_gemini_turn(masked_tool_calls, response_json.get("id"))
                        for call in masked_tool_calls:
                            call.pop("provider_id", None)
                    unmasked_tool_calls = await unmask_tool_calls_async(
                        client,
                        masked_tool_calls,
                        protect_url + "/unmask",
                        protect_headers,
                    )
                    return {
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": provider_selected_model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": unmasked_tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }],
                    }

                # Turn resolved as text; drop any held reasoning state.
                forget_gemini_turn(masked_messages)

                if not assistant_text:
                    raise RuntimeError(
                        f"{provider} returned neither text nor tool calls. "
                        f"Response: {json.dumps(response_json, ensure_ascii=False)}"
                    )

                unmasked_text = await _unmask_call_async(
                    client, assistant_text, 0, protect_url + "/unmask", protect_headers,
                )

            artifact_payload = ""
            if not is_title_request:
                artifact_payload = build_behind_scenes_artifact(
                    masked_messages=masked_messages,
                    token_map=token_map,
                    assistant_raw=assistant_text,
                )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": provider_selected_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": unmasked_text + artifact_payload,
                    },
                    "finish_reason": "stop",
                }],
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[UNHANDLED EXCEPTION] %s: %s", type(e).__name__, e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": type(e).__name__, "message": str(e)},
        )


@app.post("/v1/responses")
async def responses(request: Request):
    try:
        logger.info("Entered responses (/v1/responses)")
        body = await request.json()
        is_stream = body.get("stream", False)

        if latest_user_message_has_provider_file(body.get("input")):
            model_label = body.get("model", "")
            response_id = f"resp_{uuid4().hex}"
            created = int(time.time())
            logger.info(
                "[PROVIDER FILE REJECTED] endpoint=responses model=%s",
                model_label,
            )
            output = response_text_output(
                response_id,
                PROVIDER_FILE_UNSUPPORTED_MESSAGE,
            )
            if is_stream:
                return StreamingResponse(
                    completed_response_text_stream(
                        PROVIDER_FILE_UNSUPPORTED_MESSAGE,
                        response_id,
                        model_label,
                        created,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return _responses_object(
                response_id,
                model_label,
                created,
                output,
            )

        raw_messages, request_tools = responses_request_to_messages(body)
        request_tool_choice = body.get("tool_choice")

        log_librechat_replay(raw_messages)

        n_tool_results = sum(1 for m in raw_messages if m.get("role") == "tool")
        logger.info(
            "responses: stream=%s tools=%d messages=%d tool_results=%d%s",
            is_stream,
            len(request_tools),
            len(raw_messages),
            n_tool_results,
            " (TOOL CONTINUATION)" if n_tool_results else "",
        )

        # This gateway is stateless: it has no stored Response to continue
        # from, so if a client relies on previous_response_id instead of
        # replaying history, the earlier turns are simply absent.
        if body.get("previous_response_id"):
            logger.warning(
                "Client sent previous_response_id=%s but this gateway does not "
                "store responses; only the items in this request are visible.",
                body["previous_response_id"],
            )

        ctx = await prepare_request(
            request, raw_messages, request_tools, body.get("model", ""),
        )
        provider = ctx["provider"]
        selected_model = ctx["selected_model"]
        provider_selected_model = ctx["provider_selected_model"]
        provider_api_key = ctx["provider_api_key"]
        protect_url = ctx["protect_url"]
        protect_headers = ctx["protect_headers"]
        masked_messages = ctx["masked_messages"]
        token_map = ctx["token_map"]
        conversation_id = ctx["conversation_id"]
        log_masked_replay_payload(masked_messages)
        # response_id = f"resp_{ctx['message_id']}"
        response_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        # ---------------- Streaming ----------------
        if is_stream:
            return StreamingResponse(
                stream_responses_generator(
                    provider=provider,
                    model=selected_model,
                    model_label=provider_selected_model,
                    masked_messages=masked_messages,
                    protecto_unmask_url=protect_url + "/unmask",
                    headers=protect_headers,
                    provider_api_key=provider_api_key,
                    token_map=token_map,
                    response_id=response_id,
                    tools=request_tools,
                    tool_choice=request_tool_choice,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ---------------- Non-streaming ----------------
        response_json = await call_provider_non_streaming(
            provider=provider,
            model=selected_model,
            masked_messages=masked_messages,
            provider_api_key=provider_api_key,
            tools=request_tools,
            tool_choice=request_tool_choice,
            gemini_state=lookup_gemini_state(masked_messages),
        )
        masked_tool_calls = extract_provider_tool_calls(provider, response_json)
        assistant_text = extract_provider_text(provider, response_json)

        output = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            if masked_tool_calls:
                if provider == "gemini":
                    for call in masked_tool_calls:
                        call["provider_id"] = call.get("id")
                masked_tool_calls = normalize_tool_call_ids(masked_tool_calls)
                if provider == "gemini":
                    # Preserve the provider IDs before removing this internal
                    # field from the OpenAI-compatible response below.
                    remember_gemini_turn(
                        masked_tool_calls,
                        response_json.get("id"),
                    )
                    for call in masked_tool_calls:
                        call.pop("provider_id", None)
                unmasked_tool_calls = await unmask_tool_calls_async(
                    client, masked_tool_calls, protect_url + "/unmask", protect_headers,
                )
                for call in unmasked_tool_calls:
                    fn = call.get("function") or {}
                    output.append({
                        "id": f"fc_{call['id']}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call["id"],
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", "{}"),
                    })
                return _responses_object(
                    response_id, provider_selected_model, created, output,
                )

            forget_gemini_turn(masked_messages)
            unmasked_text = ""
            if assistant_text:
                unmasked_text = await _unmask_call_async(
                    client, assistant_text, 0, protect_url + "/unmask", protect_headers,
                )

        artifact_payload = build_behind_scenes_artifact(
            masked_messages=masked_messages,
            token_map=token_map,
            assistant_raw=assistant_text,
        )
        output.append({
            "id": f"msg_{response_id}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": unmasked_text + artifact_payload,
                "annotations": [],
            }],
        })
        return _responses_object(
            response_id, provider_selected_model, created, output,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[RESPONSES EXCEPTION] %s: %s", type(e).__name__, e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": type(e).__name__, "message": str(e)},
        )


@app.get("/v1/models")
async def list_models(request: Request):
    created = int(time.time())
    chat_name = request.headers.get(CHAT_NAME_HEADER)
    return build_model_catalog(GATEWAY_MODELS, chat_name, created)


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request):
    return {
        "id": add_chat_name_prefix(
            model_id,
            request.headers.get(CHAT_NAME_HEADER),
        ),
        "object": "model",
        "created": int(time.time()),
        "owned_by": "protecto",
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        logger.error(
            "[UNHANDLED ROUTE] %s %s -> 404. The client expected an endpoint "
            "this gateway does not implement.",
            request.method,
            request.url.path,
        )
    else:
        logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "buffer_text_when_tools": BUFFER_TEXT_WHEN_TOOLS,
        "mask_tool_results": MASK_TOOL_RESULTS,
        "async_mask": {
            "threshold_kb": ASYNC_MASK_THRESHOLD_KB,
            "base_poll_seconds": ASYNC_MASK_POLL_SECONDS,
            "base_retries": ASYNC_MASK_MAX_RETRIES,
            "scale_poll_seconds": ASYNC_MASK_SCALE_POLL_SECONDS,
            "scale_retries": ASYNC_MASK_SCALE_RETRIES,
        },
        "endpoints": [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/embeddings",
            "/v1/models",
            "/v1/ocr",
        ],
    }
