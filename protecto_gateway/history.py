import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import HTTPException, Request
from .artifacts import extract_behind_scenes_data
from .config import (
    CHAT_NAME_HEADER,
    ARTIFACT_RE,
    MASK_TOOL_RESULTS,
    PROTECTO_MASTER_TOKEN,
    PROTECTO_NAMESPACE,
    PROTECTO_URL,
    build_system_prompt,
    logger,
    validate_protecto_settings,
)
from .protecto import get_or_create_auth_token, mask_values_async


_REPLAY_ROLES = {"system", "developer", "user", "assistant", "tool"}
_TEXT_CONTENT_TYPES = {"text", "input_text", "output_text", "summary_text"}
_ARTIFACT_RESOLVED_KEY = "_protecto_artifact_resolved"
_GATEWAY_OWNED_KEY = "_protecto_gateway_owned"
_MASKED_VALUE_RE = re.compile(r"<([A-Z][A-Z0-9_]*)>(.*?)</\1>", re.DOTALL)
PROVIDER_FILE_UNSUPPORTED_MESSAGE = (
    "The 'Upload to Provider' option sends the entire file as raw bytes to the LLM provider. "
    "This can expose sensitive information without the gateway's text-masking protection, "
    "so it is not allowed in Secured-Chat. "
    "Please use 'Upload as Text' instead so the file's text can be extracted "
    "and masked before it is sent to the LLM."
)


def contains_provider_file_upload(value: Any) -> bool:
    """Return whether a request value contains provider-uploaded file bytes."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, list):
        return any(contains_provider_file_upload(item) for item in value)
    if not isinstance(value, dict):
        return False

    value_type = str(value.get("type", "")).lower()
    if value_type == "input_file" or "file_data" in value:
        return True
    if value_type == "buffer" and isinstance(value.get("data"), list):
        return True
    if value.get("documents"):
        return True
    return any(contains_provider_file_upload(item) for item in value.values())


def latest_user_message_has_provider_file(value: Any) -> bool:
    """Check only the most recent user turn for a provider-uploaded file."""
    if not isinstance(value, list):
        return contains_provider_file_upload(value)
    for item in reversed(value):
        if isinstance(item, dict) and item.get("role") == "user":
            return contains_provider_file_upload(item)
    return False


def reject_provider_file_upload(value: Any) -> None:
    """Reject file bytes that cannot safely pass through text masking."""
    if latest_user_message_has_provider_file(value):
        raise HTTPException(
            status_code=400,
            detail=PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        )


def normalize_chat_name(value: str | None) -> str | None:
    """Normalize the optional LibreChat display name carried in a header."""
    if value is None:
        return None

    chat_name = value.strip().rstrip("-")
    if not chat_name:
        return None
    if len(chat_name) > 128 or any(ord(char) < 32 for char in chat_name):
        raise HTTPException(status_code=400, detail="Invalid x-chat-name header")
    return chat_name


def add_chat_name_prefix(model_id: str, chat_name: str | None) -> str:
    """Return the UI model ID, adding the configured chat name once."""
    normalized_name = normalize_chat_name(chat_name)
    if normalized_name is None:
        return model_id

    prefix = f"{normalized_name}-"
    if model_id.lower().startswith(prefix.lower()):
        return model_id
    return f"{prefix}{model_id}"


def remove_chat_name_prefix(model_id: str, chat_name: str | None) -> str:
    """Remove only the display prefix supplied by LibreChat for this request."""
    normalized_name = normalize_chat_name(chat_name)
    if normalized_name is None:
        return model_id

    prefix = f"{normalized_name}-"
    if model_id.lower().startswith(prefix.lower()):
        return model_id[len(prefix):]
    return model_id


def build_model_catalog(
        model_ids: list[str],
        chat_name: str | None,
        created: int,
) -> dict[str, Any]:
    """Build an OpenAI-compatible model list with UI display prefixes."""
    return {
        "object": "list",
        "data": [
            {
                "id": add_chat_name_prefix(model_id, chat_name),
                "object": "model",
                "created": created,
                "owned_by": "protecto",
            }
            for model_id in model_ids
        ],
    }


def log_librechat_replay(raw_messages: list[dict[str, Any]]) -> None:
    """Log replay shape without logging any replayed values."""
    summary: list[dict[str, Any]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            summary.append({"index": index, "role": "unknown", "invalid": True})
            continue

        raw_role = message.get("role")
        role = raw_role if raw_role in _REPLAY_ROLES else "unknown"
        content = message.get("content")
        entry: dict[str, Any] = {
            "index": index,
            "role": role,
            "content_type": type(content).__name__,
            "content_chars": len(content) if isinstance(content, str) else 0,
        }

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                tool_calls = []
            entry["has_artifact"] = bool(
                isinstance(content, str) and ARTIFACT_RE.search(content)
            )
            entry["tool_calls"] = len(tool_calls)

            argument_chars = []
            for call in tool_calls:
                function = call.get("function") if isinstance(call, dict) else None
                arguments = (
                    function.get("arguments")
                    if isinstance(function, dict)
                    else None
                )
                argument_chars.append(
                    len(arguments) if isinstance(arguments, str) else 0
                )
            entry["argument_chars"] = argument_chars

        summary.append(entry)

    logger.info(
        "[LIBRECHAT REPLAY] messages=%d shape=%s",
        len(raw_messages),
        json.dumps(summary, separators=(",", ":"), ensure_ascii=True),
    )


def log_masked_replay_payload(masked_messages: list[dict[str, Any]]) -> None:
    """Log the complete replay after the masking pipeline has finished."""
    logger.info(
        "[LIBRECHAT MASKED REPLAY PAYLOAD] %s",
        json.dumps(masked_messages, ensure_ascii=False, separators=(",", ":")),
    )


def _assistant_has_tool_calls(message: dict[str, Any]) -> bool:
    """A tool-call turn is never a turn boundary: it carries no artifact."""
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _has_visible_text(content: Any) -> bool:
    """
    Tolerant text check used only to pick a replay anchor.

    Deliberately never raises: _message_text_content rejects unknown content
    with a 400, and failing a whole request while merely choosing where to
    start the replay would be worse than replaying a little too much.
    """
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            (isinstance(part, str) and part.strip())
            or (
                isinstance(part, dict)
                and part.get("type") in _TEXT_CONTENT_TYPES
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            )
            for part in content
        )
    return False


def _artifactless_pending_start(raw_messages: list[dict[str, Any]]) -> int:
    """
    Choose where replay starts when no behind-the-scenes artifact exists.

    Replay begins at the start of the turn that is still open. A turn is closed
    only by an assistant TEXT message - no tool_calls, non-empty content - that
    is followed by a later user message. A tool-call turn deliberately carries
    neither content nor an artifact, so it must never end the replay window:
    anchoring on it drops the assistant tool_calls and tool results the model
    needs to see that it has already called the tool, and the model then repeats
    the same call forever.

    Requiring a later user message keeps the original intent of this fallback -
    do not replay the whole raw history after a turn that ended without an
    artifact (e.g. a rejected provider upload) - while guaranteeing the pending
    region always still contains a real prompt.
    """
    last_user_idx = next(
        (
            i
            for i in range(len(raw_messages) - 1, -1, -1)
            if raw_messages[i].get("role") == "user"
        ),
        -1,
    )
    if last_user_idx < 0:
        return 0

    for i in range(last_user_idx - 1, -1, -1):
        message = raw_messages[i]
        if message.get("role") != "assistant":
            continue
        if _assistant_has_tool_calls(message):
            continue
        if _has_visible_text(message.get("content")):
            return i + 1
    return 0


def build_masked_history(
    raw_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """
    Split the replayed conversation into a RESOLVED region (masked text read
    from our artifacts) and a PENDING region (arrives unmasked from LibreChat,
    must be masked live by the caller).

    Returns (masked_messages, token_map, pending_messages).

    The boundary is the LAST assistant message carrying a behind-the-scenes
    artifact. Everything at or before it has already been through Protecto and
    its masked form is recorded. Everything after it - the current user prompt
    and, on a tool continuation, the assistant tool_calls plus tool results -
    is still in the clear and is returned as `pending`.
    """
    logger.info("Entered build_masked_history")

    last_artifact_idx = -1
    for i, msg in enumerate(raw_messages):
        if msg.get("role") == "assistant" and ARTIFACT_RE.search(
            msg.get("content") or ""
        ):
            last_artifact_idx = i

    pending_start_idx = last_artifact_idx + 1
    if last_artifact_idx < 0:
        # If an earlier turn ended before an artifact could be created (for
        # example, an unsupported provider upload), do not replay the whole
        # raw LibreChat history. Start at the still-open turn, which keeps the
        # current prompt together with the assistant tool calls and tool
        # results that belong to it.
        pending_start_idx = _artifactless_pending_start(raw_messages)

    masked_messages: list[dict[str, Any]] = []
    token_map: dict[str, str] = {}
    pending: list[dict[str, Any]] = []
    pending_tool_call_ids: set[str] = set()
    orphaned_tool_results = 0
    artifact_messages: list[dict[str, Any]] = []
    artifact_instructions: list[dict[str, str]] = []
    live_instructions: list[dict[str, Any]] = []
    artifact_count = 0

    for i, msg in enumerate(raw_messages):
        role = msg.get("role")

        if role in {"system", "developer"}:
            live_instructions.append({
                "role": role,
                "content": msg.get("content", ""),
            })
            continue

        if last_artifact_idx < 0 and i < pending_start_idx:
            continue

        # ---- RESOLVED region: rebuild from artifacts only ----
        if i <= last_artifact_idx:
            if role != "assistant":
                # Raw user text here is the unmasked original; its masked twin
                # comes from the artifact on the assistant reply that follows.
                continue
            content = msg.get("content") or ""
            if not ARTIFACT_RE.search(content):
                # Assistant text with no artifact is not replayed: it may be
                # unmasked UI text.
                continue
            data = extract_behind_scenes_data(content)
            if data is None:
                logger.warning("[ARTIFACT DATA ERROR] Could not decode replay data")
                continue
            if data.get("user"):
                rebuilt_user = {"role": "user", "content": data["user"]}
                masked_messages.append(rebuilt_user)
                artifact_messages.append(rebuilt_user)
            if data.get("assistant"):
                rebuilt_assistant = {
                    "role": "assistant",
                    "content": data["assistant"],
                }
                masked_messages.append(rebuilt_assistant)
                artifact_messages.append(rebuilt_assistant)
            token_map.update(data.get("token_map") or {})
            stored_instructions = data.get("instructions")
            artifact_instructions = []
            if isinstance(stored_instructions, list):
                artifact_instructions = [
                    {"role": item["role"], "content": item["content"]}
                    for item in stored_instructions
                    if isinstance(item, dict)
                    and item.get("role") in {"system", "developer"}
                    and isinstance(item.get("content"), str)
                ]
            artifact_count += 1
            continue

        # ---- PENDING region: needs live masking ----
        if role == "user":
            pending.append(msg)
        elif role == "assistant":
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    pending_tool_call_ids.add(call["id"])
            pending.append(msg)
        elif role == "tool":
            # A tool result whose assistant tool_call is not in the pending
            # region cannot be replayed: artifacts never store tool_calls, so
            # the pairing is unrecoverable. Providers reject an unpaired tool
            # result outright, so drop it rather than send a 400.
            call_id = msg.get("tool_call_id")
            if call_id not in pending_tool_call_ids:
                orphaned_tool_results += 1
                logger.warning(
                    "[REPLAY ORPHAN TOOL DROPPED] index=%d has_tool_call_id=%s",
                    i,
                    bool(call_id),
                )
                continue
            pending.append(msg)

    # Reuse artifact instructions only when the live copies are absent or are
    # exactly the same after local token-map reconstruction. A changed live
    # instruction may contain newly retrieved file context and must be masked.
    live_instructions_match = bool(live_instructions) and bool(
        artifact_instructions
    ) and _instructions_match_artifact(
            live_instructions,
            artifact_instructions,
            token_map,
    )
    reuse_artifact_instructions = bool(artifact_instructions) and (
        not live_instructions or live_instructions_match
    )
    if reuse_artifact_instructions:
        resolved_instructions = [
            {**message, _ARTIFACT_RESOLVED_KEY: True}
            for message in artifact_instructions
        ]
        masked_messages[0:0] = resolved_instructions
        artifact_messages[0:0] = artifact_instructions
        logger.info(
            "[INSTRUCTION REPLAY] source=artifact live_present=%s exact_match=%s "
            "instructions=%d",
            bool(live_instructions),
            live_instructions_match,
            len(artifact_instructions),
        )
    elif live_instructions:
        masked_messages[0:0] = live_instructions
        logger.info(
            "[INSTRUCTION REPLAY] source=live exact_match=false instructions=%d",
            len(live_instructions),
        )

    logger.info(
        "[ARTIFACT HISTORY REBUILT] artifacts=%d messages=%d roles=%s "
        "masked_tokens=%d pending=%d pending_start=%d",
        artifact_count,
        len(artifact_messages),
        [message["role"] for message in artifact_messages],
        len(token_map),
        len(pending),
        pending_start_idx,
    )

    # Counts and indices only - never message content.
    raw_tool_results = sum(
        1 for message in raw_messages if message.get("role") == "tool"
    )
    pending_tool_results = sum(
        1 for message in pending if message.get("role") == "tool"
    )
    logger.info(
        "[REPLAY ANCHOR] total=%d last_artifact_idx=%d pending_start_idx=%d "
        "source=%s raw_tool_results=%d pending_tool_results=%d "
        "orphaned_tool_results=%d",
        len(raw_messages),
        last_artifact_idx,
        pending_start_idx,
        "artifact" if last_artifact_idx >= 0 else "artifactless-scan",
        raw_tool_results,
        pending_tool_results,
        orphaned_tool_results,
    )
    if raw_tool_results and not pending_tool_results:
        # The signature of the skill-loop bug: tool results came in, none went
        # out, so the model cannot tell it has already called the tool.
        logger.warning(
            "[REPLAY ANCHOR DROPPED TOOL CHAIN] raw_tool_results=%d "
            "pending_start_idx=%d - provider will not see any tool result",
            raw_tool_results,
            pending_start_idx,
        )
    logger.info(
        "[ARTIFACT HISTORY REBUILT PAYLOAD] %s",
        json.dumps(artifact_messages, ensure_ascii=False, separators=(",", ":")),
    )

    return masked_messages, token_map, pending


def _resolve_artifact_tokens(text: str, token_map: dict[str, str]) -> str:
    """Locally reconstruct artifact text only for an exact replay comparison."""
    def replace(match: re.Match[str]) -> str:
        full_token = match.group(0)
        token_value = match.group(2)
        return token_map.get(token_value, token_map.get(full_token, full_token))

    return _MASKED_VALUE_RE.sub(replace, text)


def _instructions_match_artifact(
    live_instructions: list[dict[str, Any]],
    artifact_instructions: list[dict[str, str]],
    token_map: dict[str, str],
) -> bool:
    """Return whether live instructions exactly match stored masked copies."""
    if len(live_instructions) != len(artifact_instructions):
        return False

    for live, stored in zip(live_instructions, artifact_instructions):
        if live.get("role") != stored["role"]:
            return False
        live_content = _message_text_content(live.get("content"), stored["role"])
        if live_content != _resolve_artifact_tokens(stored["content"], token_map):
            return False
    return True


def _message_text_content(content: Any, role: str) -> str:
    """Normalize OpenAI-compatible text content without forwarding unknown parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {role} message content structure",
        )

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict) or part.get("type") not in _TEXT_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported non-text content in {role} message",
            )
        text = part.get("text", "")
        if not isinstance(text, str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid text content in {role} message",
            )
        text_parts.append(text)
    return "".join(text_parts)


def _tool_result_text(content: Any) -> str:
    """Serialize structured tool output before masking it as one text value."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Tool result content must be JSON serializable",
        ) from exc


def add_trusted_system_prompt(
    masked_messages: list[dict[str, Any]],
    trusted_prompt: str,
) -> None:
    """Add gateway-owned instructions after all client-owned text is masked."""
    masked_messages.insert(0, {
        "role": "system",
        "content": trusted_prompt,
        _GATEWAY_OWNED_KEY: True,
    })


async def mask_pending_messages(
    client: httpx.AsyncClient,
    pending: list[dict[str, Any]],
    masked_messages: list[dict[str, Any]],
    token_map: dict[str, str],
    protecto_mask_url: str,
    protecto_headers: dict[str, str],
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """
    Mask every untrusted clear-text value in ONE Protecto call and append the
    pending messages to masked_messages, preserving order.

    Masked here:
      * LibreChat system/developer instructions, including injected RAG text
      * user prompts (including the ORIGINAL prompt replayed on a tool
        continuation - this is why no cross-request cache is needed)
      * structured standard text content, normalized to a string
      * assistant tool_call arguments (LibreChat replays these with the REAL
        values we handed it for execution, so they must be re-masked before
        going back upstream)
      * text or structured JSON tool results, when MASK_TOOL_RESULTS is enabled
    """
    to_mask: list[str] = []
    instruction_messages: list[dict[str, Any]] = []
    for message in masked_messages:
        role = message.get("role")
        if role not in {"system", "developer"}:
            continue
        if message.get(_ARTIFACT_RESOLVED_KEY):
            continue
        content = _message_text_content(message.get("content"), role)
        message["content"] = content
        if content:
            instruction_messages.append(message)
            to_mask.append(content)

    normalized_pending_content: dict[int, str] = {}
    for msg in pending:
        role = msg.get("role")
        if role == "user":
            content = _message_text_content(msg.get("content"), role)
            normalized_pending_content[id(msg)] = content
            if content:
                to_mask.append(content)
        elif role == "assistant":
            for tc in msg.get("tool_calls") or []:
                args = (tc.get("function") or {}).get("arguments")
                if args:
                    to_mask.append(
                        args if isinstance(args, str)
                        else json.dumps(args, ensure_ascii=False)
                    )
        elif role == "tool":
            content = _tool_result_text(msg.get("content"))
            normalized_pending_content[id(msg)] = content
            if MASK_TOOL_RESULTS and content:
                to_mask.append(content)

    masked_values = await mask_values_async(
        client,
        to_mask,
        protecto_mask_url,
        protecto_headers,
        token_map,
        progress_callback,
    )

    idx = 0
    for message in instruction_messages:
        message["content"] = masked_values[idx]
        idx += 1

    for msg in pending:
        role = msg.get("role")

        if role == "user":
            content = normalized_pending_content.get(id(msg), "")
            if content:
                masked_messages.append({
                    "role": "user",
                    "content": masked_values[idx],
                })
                idx += 1
            continue

        if role == "assistant":
            new_tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = dict(tc.get("function") or {})
                if fn.get("arguments"):
                    fn["arguments"] = masked_values[idx]
                    idx += 1
                new_tool_calls.append({**tc, "function": fn})
            if new_tool_calls:
                masked_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": new_tool_calls,
                })
            # Assistant text in the pending region carries no artifact, so it
            # is unmasked UI text and is intentionally not replayed.
            continue

        if role == "tool":
            content = normalized_pending_content.get(id(msg), "")
            if MASK_TOOL_RESULTS and content:
                content = masked_values[idx]
                idx += 1
            masked_messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "name": msg.get("name"),
                "content": content,
            })
            continue


async def prepare_request(
    request: Request,
    raw_messages: list[dict[str, Any]],
    request_tools: list,
    model_string: str,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """
    Shared pipeline for BOTH public endpoints: validate, authenticate with
    Protecto, and build the masked history. Returns everything the provider
    call needs.
    """
    request_headers = dict(request.headers)
    conversation_id = request.headers.get("x-conversation-id", "unknown")
    message_id = request.headers.get("x-message-id", "unknown")

    # ========================================================
    # Parse provider:model
    # ========================================================
    provider_selected_model = model_string
    chat_name = request.headers.get(CHAT_NAME_HEADER)
    routed_model = remove_chat_name_prefix(model_string, chat_name)
    provider, separator, selected_model = routed_model.partition(":")
    if not separator:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model must include provider prefix. Examples: "
                "openai:gpt-5.6 or gemini:gemini-3.6-flash"
            ),
        )
    # Keep compatibility with model IDs produced before X-Chat-Name became
    # available. New deployments should use the request header instead.
    provider = provider.lower().strip().removeprefix("protecto-")
    selected_model = selected_model.strip()
    if provider not in {"openai", "gemini"}:
        raise HTTPException(
            status_code=400, detail=f"Unsupported provider: {provider}",
        )

    # ========================================================
    # Private Protecto service configuration
    # ========================================================
    # Logged-in user email is used as the Protecto user id.
    logged_in_user = request_headers.get("x-user-email", "").strip().lower()

    try:
        (
            protect_url,
            protect_master_token,
            protect_namespace,
        ) = validate_protecto_settings(
            PROTECTO_URL,
            PROTECTO_MASTER_TOKEN,
            PROTECTO_NAMESPACE,
        )
    except ValueError as exc:
        logger.error("Protecto gateway service configuration is invalid")
        raise HTTPException(
            status_code=503,
            detail="Protecto service is not configured in the gateway",
        ) from exc
    if not logged_in_user:
        raise HTTPException(status_code=400, detail="Missing x-user-email")

    # ========================================================
    # Provider API key
    # ========================================================
    if provider == "openai":
        provider_api_key = request_headers.get("x-openai-api-key")
        if not provider_api_key:
            raise HTTPException(status_code=400, detail="Missing x-openai-api-key")
    else:
        provider_api_key = request_headers.get("x-gemini-api-key")
        if not provider_api_key:
            raise HTTPException(status_code=400, detail="Missing x-gemini-api-key")

    # ========================================================
    # Validate messages
    # ========================================================
    if not raw_messages:
        raise HTTPException(status_code=400, detail="No message history provided")

    reject_provider_file_upload(raw_messages)

    last_role = raw_messages[-1].get("role")
    if last_role not in {"user", "tool"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Last conversation message must be a user message "
                "or a tool result."
            ),
        )

    # ========================================================
    # Protecto master auth
    # ========================================================
    master_headers = {
        "Authorization": f"Bearer {protect_master_token}",
        "Content-Type": "application/json",
    }
    protect_auth_token = await get_or_create_auth_token(
        namespace_name=protect_namespace,
        user_id=logged_in_user,
        headers=master_headers,
        protecto_url=protect_url,
    )
    protect_headers = {
        "Authorization": f"Bearer {protect_auth_token}",
        "Content-Type": "application/json",
    }

    # ========================================================
    # Build masked history
    #
    # Resolved region comes from artifacts; the pending region (current
    # prompt + any replayed tool_calls/tool results) is masked live. On a
    # tool continuation the ORIGINAL prompt is simply re-masked from what
    # LibreChat replayed - no cross-request state is kept anywhere.
    # ========================================================
    masked_messages, token_map, pending = build_masked_history(raw_messages)
    async with httpx.AsyncClient(timeout=30.0) as client:
        await mask_pending_messages(
            client=client,
            pending=pending,
            masked_messages=masked_messages,
            token_map=token_map,
            protecto_mask_url=protect_url + "/mask",
            protecto_headers=protect_headers,
            progress_callback=progress_callback,
        )

    # Add gateway-owned instructions only after every client-owned text field
    # has passed through Protecto. This prevents LibreChat RAG/file context in
    # a system message from bypassing masking while keeping our fixed policy
    # prompt outside the untrusted masking batch.
    add_trusted_system_prompt(
        masked_messages,
        build_system_prompt(bool(request_tools)),
    )

    return {
        "provider": provider,
        "selected_model": selected_model,
        "provider_selected_model": provider_selected_model,
        "provider_api_key": provider_api_key,
        "protect_url": protect_url,
        "protect_headers": protect_headers,
        "masked_messages": masked_messages,
        "token_map": token_map,
        "conversation_id": conversation_id,
        "message_id": message_id,
    }

def is_title_generation_request(body: dict[str, Any]) -> bool:
    """Treat an explicit non-streaming Chat Completions call as a title call."""
    return body.get("stream") is False
