"""Protecto asynchronous masking submission and status polling."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import HTTPException

from .config import (
    ASYNC_MASK_MAX_RETRIES,
    ASYNC_MASK_POLL_SECONDS,
    logger,
)


MaskProgressCallback = Callable[[str], Awaitable[None]]


def _polling_progress_percentage(completed_polls: int, max_polls: int) -> int:
    """Estimate progress without claiming completion before API success."""
    if max_polls <= 0:
        return 0
    return min(99, math.floor(completed_polls * 100 / max_polls))


def scaled_polling_settings(
    payload_bytes: int,
    threshold_bytes: int,
    base_poll_seconds: float,
    base_retries: int,
    scale_poll_seconds: bool,
    scale_retries: bool,
) -> tuple[float, int]:
    """Scale polling settings relative to the configured threshold size."""
    scale_factor = max(1.0, payload_bytes / threshold_bytes)
    poll_seconds = (
        base_poll_seconds * scale_factor
        if scale_poll_seconds
        else base_poll_seconds
    )
    retries = (
        math.ceil(base_retries * scale_factor)
        if scale_retries
        else base_retries
    )
    return poll_seconds, retries


def _protecto_batch_urls(mask_url: str) -> tuple[str, str]:
    """Build the async submit and status URLs from the existing mask URL."""
    base_url = mask_url.rstrip("/")
    if base_url.endswith("/mask"):
        base_url = base_url[:-len("/mask")]
    return f"{base_url}/mask/async", f"{base_url}/async-status"


def _tracking_ids(response_body: dict[str, Any]) -> list[str]:
    items = response_body.get("data")
    if not isinstance(items, list):
        items = []
    tracking_ids = [
        item.get("tracking_id")
        for item in items
        if isinstance(item, dict) and item.get("tracking_id")
    ]
    if not response_body.get("success") or not tracking_ids:
        message = (response_body.get("error") or {}).get("message")
        raise HTTPException(
            status_code=502,
            detail=message or "Protecto async mask submission returned no tracking ID.",
        )
    return tracking_ids


def _completed_results(
    response_body: dict[str, Any],
    tracking_ids: list[str],
) -> list[dict[str, Any]] | None:
    """Return completed mask results, or None while any job is still active."""
    if not response_body.get("success"):
        message = (response_body.get("error") or {}).get("message")
        raise HTTPException(
            status_code=502,
            detail=message or "Protecto async status request failed.",
        )

    items = response_body.get("data")
    if not isinstance(items, list) or not items:
        raise HTTPException(
            status_code=502,
            detail="Protecto async status returned no job data.",
        )

    by_tracking_id = {
        item.get("tracking_id"): item
        for item in items
        if isinstance(item, dict) and item.get("tracking_id")
    }
    ordered_items = [by_tracking_id.get(tracking_id) for tracking_id in tracking_ids]
    if any(item is None for item in ordered_items):
        raise HTTPException(
            status_code=502,
            detail="Protecto async status omitted a submitted tracking ID.",
        )

    statuses = [str(item.get("status", "")).upper() for item in ordered_items]
    if any(status == "FAILED" for status in statuses):
        failed_item = ordered_items[statuses.index("FAILED")]
        raise HTTPException(
            status_code=502,
            detail=(
                failed_item.get("error_msg")
                or "Protecto asynchronous masking failed."
            ),
        )

    allowed_statuses = {"PENDING", "IN-PROGRESS", "SUCCESS"}
    unexpected = [status for status in statuses if status not in allowed_statuses]
    if unexpected:
        raise HTTPException(
            status_code=502,
            detail=f"Protecto returned unexpected async status: {unexpected[0] or 'empty'}.",
        )

    if not all(status == "SUCCESS" for status in statuses):
        return None

    results: list[dict[str, Any]] = []
    for item in ordered_items:
        item_results = item.get("result")
        if not isinstance(item_results, list):
            raise HTTPException(
                status_code=502,
                detail="Protecto async mask success response has no result list.",
            )
        results.extend(item_results)
    return results


async def mask_payload_in_background(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    protecto_mask_url: str,
    headers: dict[str, str],
    progress_callback: MaskProgressCallback | None = None,
    poll_seconds: float = ASYNC_MASK_POLL_SECONDS,
    max_polls: int = ASYNC_MASK_MAX_RETRIES,
) -> list[dict[str, Any]]:
    """Submit a large mask payload and poll until it succeeds or times out."""
    async_url, status_url = _protecto_batch_urls(protecto_mask_url)
    submit_response = await client.put(async_url, json=payload, headers=headers)
    if not 200 <= submit_response.status_code < 300:
        logger.error(
            "[PROTECTO ASYNC MASK ERROR] status=%s",
            submit_response.status_code,
        )
    submit_response.raise_for_status()
    tracking_ids = _tracking_ids(submit_response.json())
    logger.info(
        "[PROTECTO ASYNC MASK SUBMITTED] jobs=%d max_polls=%d interval_seconds=%s",
        len(tracking_ids),
        max_polls,
        poll_seconds,
    )

    status_payload = {
        "status": [{"tracking_id": tracking_id} for tracking_id in tracking_ids],
    }
    if progress_callback is not None:
        await progress_callback("Context masking is 0% complete...")

    for poll_number in range(1, max_polls + 1):
        await asyncio.sleep(poll_seconds)

        status_response = await client.put(
            status_url,
            json=status_payload,
            headers=headers,
        )
        if not 200 <= status_response.status_code < 300:
            logger.error(
                "[PROTECTO ASYNC STATUS ERROR] poll=%d status=%s",
                poll_number,
                status_response.status_code,
            )
        status_response.raise_for_status()
        status_body = status_response.json()
        status_items = status_body.get("data")
        if not isinstance(status_items, list):
            status_items = []
        statuses = [
            str(item.get("status", "")).upper()
            for item in status_items
            if isinstance(item, dict)
        ]
        logger.info(
            "[PROTECTO ASYNC MASK STATUS] poll=%d statuses=%s",
            poll_number,
            statuses,
        )
        results = _completed_results(status_body, tracking_ids)
        if results is not None:
            if progress_callback is not None:
                await progress_callback("Context masking is 100% complete.")
            logger.info(
                "[PROTECTO ASYNC MASK COMPLETED] poll=%d results=%d",
                poll_number,
                len(results),
            )
            return results

        if progress_callback is not None:
            percentage = _polling_progress_percentage(poll_number, max_polls)
            await progress_callback(
                f"Context masking is {percentage}% complete...",
            )

    raise HTTPException(
        status_code=504,
        detail=(
            "Protecto asynchronous masking did not complete after "
            f"{max_polls} status checks."
        ),
    )
