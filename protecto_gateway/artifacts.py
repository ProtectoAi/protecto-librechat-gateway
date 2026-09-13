import base64
import html
import json
import re
from typing import Any

from .config import ARTIFACT_RE, logger
from .protecto import decode_json_unicode_escapes


_REPLAY_DATA_RE = re.compile(
    r"<!--protecto-replay-data:([A-Za-z0-9_=-]+)-->",
)
_REPLAY_DATA_ATTRIBUTE_RE = re.compile(
    r':::artifact\{[^}]*\breplay-data="([A-Za-z0-9_=-]+)"',
)
_MASKED_TAG_RE = re.compile(
    r"<([A-Z][A-Z0-9_]*)>(.*?)</\1>",
    re.DOTALL,
)
def _encode_replay_data(data: dict[str, Any]) -> str:
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(serialized.encode("utf-8")).decode("ascii")


def _normalize_text(value: str | None) -> str:
    if not value:
        return "None"
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _masked_tokens(*values: str | None) -> list[tuple[str, str]]:
    tokens: dict[str, str] = {}
    for value in values:
        if not value:
            continue
        for match in _MASKED_TAG_RE.finditer(value):
            tokens[match.group(0)] = match.group(2)
    return sorted(tokens.items())


def _original_values(
    user_prompt: str,
    assistant_response: str | None,
    token_map: dict[str, str],
    tool_results: list[str] | None = None,
    instructions: list[str] | None = None,
) -> list[str]:
    values: dict[str, None] = {}
    for full_token, inner_value in _masked_tokens(
            user_prompt,
            assistant_response,
            *(tool_results or []),
            *(instructions or []),
    ):
        original_value = token_map.get(inner_value)
        if original_value is None:
            original_value = token_map.get(full_token)
        if original_value is not None:
            values[decode_json_unicode_escapes(original_value)] = None
    return list(values)


def _render_html_artifact(
    user_prompt: str,
    assistant_response: str | None,
    token_map: dict[str, str],
    tool_results: list[tuple[str, str]] | None = None,
    instructions: list[str] | None = None,
) -> str:
    tool_results = tool_results or []
    original_values = _original_values(
        user_prompt,
        assistant_response,
        token_map,
        [content for _, content in tool_results],
        instructions,
    )
    original_value_items = "".join(
        f"<li><code>{html.escape(value)}</code></li>"
        for value in original_values
    ) or "<li>None</li>"
    retrieved_context = ""
    if tool_results:
        retrieved_items = "".join(
            "    <h3>" + html.escape(name) + "</h3>\n"
            "    <pre>" + html.escape(_normalize_text(content)) + "</pre>\n"
            for name, content in tool_results
        )
        retrieved_context = (
            "  <section>\n"
            "    <h2><strong>📄 RETRIEVED FILE CONTEXT SENT TO LLM</strong></h2>\n"
            f"{retrieved_items}"
            "  </section>\n"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "  <title>Behind the Scene</title>\n"
        "  <style>\n"
        "    body { font-family: system-ui, sans-serif; line-height: 1.5; "
        "font-size: 12px; margin: 0; padding: 24px; color: #1f2937; "
        "background: #ffffff; }\n"
        "    section { margin: 0 0 28px; }\n"
        "    h2 { font-size: 16px; font-weight: 800; margin: 0 0 12px; }\n"
        "    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; "
        "padding: 16px; border: 1px solid #d1d5db; border-radius: 8px; "
        "background: #f9fafb; font-family: system-ui, sans-serif; }\n"
        "    code { font-family: system-ui, sans-serif; }\n"
        "    ul { margin: 0; padding-left: 24px; }\n"
        "    li { margin: 6px 0; overflow-wrap: anywhere; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <section>\n"
        "    <h2><strong>👤 USER PROMPT</strong></h2>\n"
        f"    <pre>{html.escape(_normalize_text(user_prompt))}</pre>\n"
        "  </section>\n"
        "  <section>\n"
        "    <h2><strong>🤖 AI RESPONSE</strong></h2>\n"
        f"    <pre>{html.escape(_normalize_text(assistant_response))}</pre>\n"
        "  </section>\n"
        f"{retrieved_context}"
        "  <section>\n"
        "    <h2>🔐 SENSITIVE INFORMATION IDENTIFIED</h2>\n"
        f"    <ul>{original_value_items}</ul>\n"
        "  </section>\n"
        "</body>\n"
        "</html>"
    )


def _artifact_fence(content: str) -> str:
    longest_backtick_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", content)),
        default=0,
    )
    return "`" * max(4, longest_backtick_run + 1)


def extract_behind_scenes_data(content: str) -> dict[str, Any] | None:
    """Read current encoded artifacts and legacy JSON artifacts."""
    artifact_match = ARTIFACT_RE.search(content)
    if not artifact_match:
        return None

    artifact_body = artifact_match.group(1).strip()
    attribute_match = _REPLAY_DATA_ATTRIBUTE_RE.search(content)
    comment_match = _REPLAY_DATA_RE.search(artifact_body)
    encoded_match = attribute_match or comment_match
    source = (
        "metadata_attribute"
        if attribute_match
        else "metadata_comment"
        if comment_match
        else "legacy_json"
    )
    try:
        if encoded_match:
            decoded = base64.urlsafe_b64decode(encoded_match.group(1)).decode("utf-8")
            data = json.loads(decoded)
        else:
            data = json.loads(artifact_body)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("[ARTIFACT DATA DECODE] source=%s success=false", source)
        return None

    if not isinstance(data, dict):
        logger.warning("[ARTIFACT DATA DECODE] source=%s success=false", source)
        return None

    token_map = data.get("token_map")
    logger.info(
        "[ARTIFACT DATA DECODE] source=%s success=true encoded_chars=%d "
        "user_chars=%d assistant_chars=%d masked_tokens=%d",
        source,
        len(encoded_match.group(1)) if encoded_match else 0,
        len(data.get("user") or ""),
        len(data.get("assistant") or ""),
        len(token_map) if isinstance(token_map, dict) else 0,
    )
    return data


def build_behind_scenes_artifact(
    masked_messages: list[dict[str, Any]],
    token_map: dict[str, str],
    assistant_raw: str | None,
) -> str:
    """
    Persist the masked user turn + masked assistant reply so later requests can
    reconstruct history without re-masking everything.

    Only ever attached to a NORMAL text turn. A tool-call turn deliberately
    carries no content at all (see stream_and_unmask_generator).
    """
    last_user = ""
    for msg in reversed(masked_messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break
    tool_results = []
    for msg in masked_messages:
        if msg.get("role") != "tool" or not msg.get("content"):
            continue
        content = msg["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        tool_results.append((msg.get("name") or "File search result", content))
    instructions = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in masked_messages
        if msg.get("role") in {"system", "developer"}
        and not msg.get("_protecto_gateway_owned")
        and isinstance(msg.get("content"), str)
    ]
    replay_data = {
        "user": last_user,
        "assistant": assistant_raw,
        "token_map": token_map,
    }
    if instructions:
        replay_data["instructions"] = instructions
    encoded_replay_data = _encode_replay_data(replay_data)
    readable_content = _render_html_artifact(
        last_user,
        assistant_raw,
        token_map,
        tool_results,
        [item["content"] for item in instructions],
    )
    fence = _artifact_fence(readable_content)
    return (
        "\n\n"
        ':::artifact{identifier="behind-the-scenes" '
        'type="text/html" title="Behind the Scene" '
        f'replay-data="{encoded_replay_data}"}}\n'
        f"{fence}\n{readable_content}\n{fence}\n"
        ":::\n\n"
    )
