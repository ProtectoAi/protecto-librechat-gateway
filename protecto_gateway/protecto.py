import asyncio
from typing import *
from weakref import WeakValueDictionary
import json
import httpx
import re
from datetime import date,timedelta,datetime
from fastapi import HTTPException
from .config import (
    ASYNC_MASK_MAX_RETRIES,
    ASYNC_MASK_POLL_SECONDS,
    ASYNC_MASK_SCALE_POLL_SECONDS,
    ASYNC_MASK_SCALE_RETRIES,
    ASYNC_MASK_THRESHOLD_BYTES,
    logger,
)
from .batch_mask import (
    MaskProgressCallback,
    mask_payload_in_background,
    scaled_polling_settings,
)


_JSON_UNICODE_ESCAPE_RE = re.compile(
    r"\\u([0-9a-fA-F]{4})(?:\\u([0-9a-fA-F]{4}))?",
)

# LibreChat starts title generation and the main response concurrently. Protecto
# may invalidate an earlier token when another token is generated for the same
# namespace/user, so serialize that user's fetch-or-generate transaction.
_AUTH_TOKEN_LOCKS: WeakValueDictionary[
    tuple[str, str, str], asyncio.Lock
] = WeakValueDictionary()


def decode_json_unicode_escapes(value: str) -> str:
    """Decode only valid JSON ``\\uXXXX`` escapes, including surrogate pairs."""
    def replace_escape(match: re.Match) -> str:
        first = int(match.group(1), 16)
        second_group = match.group(2)
        if second_group is None:
            if 0xD800 <= first <= 0xDFFF:
                return match.group(0)
            return chr(first)

        second = int(second_group, 16)
        if 0xD800 <= first <= 0xDBFF and 0xDC00 <= second <= 0xDFFF:
            codepoint = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
            return chr(codepoint)
        if 0xD800 <= first <= 0xDFFF or 0xD800 <= second <= 0xDFFF:
            return match.group(0)
        return chr(first) + chr(second)

    return _JSON_UNICODE_ESCAPE_RE.sub(replace_escape, value)

async def get_or_create_auth_token(
        namespace_name: str,
        user_id: str,
        headers: dict,
        protecto_url: str,
) -> str:
    logger.info("Entered get_or_create_auth_token")
    lock_key = (protecto_url, namespace_name, user_id)
    auth_lock = _AUTH_TOKEN_LOCKS.setdefault(lock_key, asyncio.Lock())
    async with auth_lock:
        return await _fetch_or_generate_auth_token(
            namespace_name=namespace_name,
            user_id=user_id,
            headers=headers,
            protecto_url=protecto_url,
        )


async def _fetch_or_generate_auth_token(
        namespace_name: str,
        user_id: str,
        headers: dict,
        protecto_url: str,
) -> str:
    fetch_url = f"{protecto_url}/super-admin/namespace/auth-token/fetch"
    generate_url = f"{protecto_url}/super-admin/namespace/auth-token/generate"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ----------------------------------------------------
        # Fetch existing token
        # ----------------------------------------------------
        fetch_payload = {
            "data": {
                "_id": user_id,
                "namespace_name": namespace_name,
            }
        }
        response = await client.put(fetch_url, json=fetch_payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", {})

        create_token = False
        if not data:
            create_token = True
        else:
            end_date = data.get("end_date")
            if not end_date:
                create_token = True
            else:
                expiry = datetime.strptime(end_date.split()[0], "%Y-%m-%d").date()
                if expiry <= date.today():
                    create_token = True

        # ----------------------------------------------------
        # Existing token
        # ----------------------------------------------------
        if not create_token:
            logger.info("Fetched Protecto auth token for %s", user_id)
            return data["auth_token"]

        # ----------------------------------------------------
        # Generate new token
        # ----------------------------------------------------
        today = date.today()
        generate_payload = {
            "data": {
                "namespace_name": namespace_name,
                "_id": user_id,
                "permissions": ["mask", "unmask"],
                "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
            }
        }
        response = await client.put(generate_url, json=generate_payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", {})
        if "auth_key" not in data:
            raise RuntimeError("Protecto auth token generation failed.")
        logger.info("Created Protecto auth token for %s", user_id)
        return data["auth_key"]


async def mask_values_async(
        client: httpx.AsyncClient,
        values: List[str],
        protecto_mask_url: str,
        headers: Dict,
        token_map: Dict[str, str],
        progress_callback: MaskProgressCallback | None = None,
) -> List[str]:
    """
    Mask a batch of strings in ONE Protecto call and fold the returned
    individual tokens into token_map. Returns masked strings in input order.
    """
    if not values:
        return []
    payload = {"mask": [{"value": v} for v in values]}
    payload_bytes = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    use_background_mask = payload_bytes > ASYNC_MASK_THRESHOLD_BYTES
    logger.info(
        "[PROTECTO MASK ROUTE] values=%d payload_bytes=%d payload_kib=%.2f "
        "threshold_bytes=%d endpoint=%s",
        len(values),
        payload_bytes,
        payload_bytes / 1024,
        ASYNC_MASK_THRESHOLD_BYTES,
        "/mask/async" if use_background_mask else "/mask",
    )

    if use_background_mask:
        poll_seconds, max_polls = scaled_polling_settings(
            payload_bytes=payload_bytes,
            threshold_bytes=ASYNC_MASK_THRESHOLD_BYTES,
            base_poll_seconds=ASYNC_MASK_POLL_SECONDS,
            base_retries=ASYNC_MASK_MAX_RETRIES,
            scale_poll_seconds=ASYNC_MASK_SCALE_POLL_SECONDS,
            scale_retries=ASYNC_MASK_SCALE_RETRIES,
        )
        logger.info(
            "[PROTECTO ASYNC MASK POLICY] poll_seconds=%.2f retries=%d "
            "scale_poll_seconds=%s scale_retries=%s",
            poll_seconds,
            max_polls,
            ASYNC_MASK_SCALE_POLL_SECONDS,
            ASYNC_MASK_SCALE_RETRIES,
        )
        data = await mask_payload_in_background(
            client=client,
            payload=payload,
            protecto_mask_url=protecto_mask_url,
            headers=headers,
            progress_callback=progress_callback,
            poll_seconds=poll_seconds,
            max_polls=max_polls,
        )
    else:
        res = await client.put(protecto_mask_url, json=payload, headers=headers)
        if res.status_code != 200:
            logger.error("[PROTECTO MASK ERROR] status=%s", res.status_code)
        res.raise_for_status()
        data = res.json().get("data", [])
    if len(data) != len(values):
        raise HTTPException(
            status_code=502,
            detail=(
                "Protecto mask API returned "
                f"{len(data)} result(s) for {len(values)} input(s)."
            ),
        )
    masked = []
    for item in data:
        masked.append(item["token_value"])
        for token in item.get("individual_tokens", []) or []:
            token_map[token["token"]] = decode_json_unicode_escapes(
                token["value"],
            )
    return masked


async def _unmask_call_async(
        client: httpx.AsyncClient,
        text_piece: str,
        order: int,
        protect_unmask_url: str,
        headers: Dict,
) -> str:
    try:
        payload = {
            "unmask": [{
                "token_value": text_piece,
                "attributes": {"order": order},
            }]
        }
        res = await client.put(
            protect_unmask_url,
            json=payload,
            headers=headers,
            timeout=10.0,
        )
        res.raise_for_status()
        return decode_json_unicode_escapes(res.json()["data"][0]["value"])
    except Exception as e:
        logger.error("[UNMASK EXCEPTION] %s", e)
        return text_piece
