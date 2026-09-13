"""Protecto-masked, OpenAI-compatible embeddings support."""

from typing import Any

import httpx
from fastapi import HTTPException, Request

from .config import (
    OPENAI_EMBEDDINGS_URL,
    PROTECTO_MASTER_TOKEN,
    PROTECTO_NAMESPACE,
    PROTECTO_URL,
    RAG_PROTECTO_USER_ID,
    logger,
    validate_protecto_settings,
)
from .protecto import get_or_create_auth_token, mask_values_async


SUPPORTED_EMBEDDING_MODEL = "text-embedding-3-small"
ALLOWED_REQUEST_FIELDS = {"model", "input", "encoding_format", "dimensions", "user"}


def _bearer_token(request: Request) -> str:
    scheme, separator, token = request.headers.get("authorization", "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Missing bearer API key")
    return token.strip()


def validate_embeddings_request(
        request: Request,
        body: Any,
) -> tuple[dict[str, Any], str]:
    """Validate RAG input and capture its forwarded OpenAI Bearer key."""
    if not RAG_PROTECTO_USER_ID:
        raise HTTPException(
            status_code=503,
            detail="Protecto RAG identity is not configured in the gateway",
        )
    provider_api_key = _bearer_token(request)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    if body.get("model") != SUPPORTED_EMBEDDING_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported embedding model: {body.get('model')!r}",
        )

    embedding_input = body.get("input")
    valid_input = isinstance(embedding_input, str) and bool(embedding_input)
    if isinstance(embedding_input, list):
        valid_input = bool(embedding_input) and all(
            isinstance(item, str) and bool(item) for item in embedding_input
        )
    if not valid_input:
        raise HTTPException(
            status_code=400,
            detail=(
                "input must be a non-empty string or list of non-empty strings; "
                "set RAG_CHECK_EMBEDDING_CTX_LENGTH=false in LibreChat"
            ),
        )

    unknown_fields = set(body) - ALLOWED_REQUEST_FIELDS
    if unknown_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported request fields: {', '.join(sorted(unknown_fields))}",
        )
    return body, provider_api_key


async def _mask_inputs(values: list[str]) -> list[str]:
    try:
        protecto_url, master_token, namespace = validate_protecto_settings(
            PROTECTO_URL,
            PROTECTO_MASTER_TOKEN,
            PROTECTO_NAMESPACE,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Protecto service is not configured in the gateway",
        ) from exc

    master_headers = {
        "Authorization": f"Bearer {master_token}",
        "Content-Type": "application/json",
    }
    try:
        auth_token = await get_or_create_auth_token(
            namespace_name=namespace,
            user_id=RAG_PROTECTO_USER_ID,
            headers=master_headers,
            protecto_url=protecto_url,
        )
        protecto_headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await mask_values_async(
                client=client,
                values=values,
                protecto_mask_url=f"{protecto_url}/mask",
                headers=protecto_headers,
                token_map={},
            )
    except HTTPException:
        raise
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        logger.error("Protecto embedding-input masking failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Protecto masking failed") from exc


async def _request_openai(
        payload: dict[str, Any],
        provider_api_key: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {provider_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            response = await client.post(
                OPENAI_EMBEDDINGS_URL,
                json=payload,
                headers=headers,
            )
    except httpx.RequestError as exc:
        logger.error("OpenAI embeddings request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Embedding provider unavailable") from exc

    if response.status_code != 200:
        logger.error("OpenAI embeddings returned HTTP %s", response.status_code)
        if response.status_code in {400, 422}:
            raise HTTPException(status_code=400, detail="Embedding provider rejected request")
        if response.status_code in {401, 403}:
            raise HTTPException(status_code=502, detail="Embedding provider authentication failed")
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Embedding provider rate limit exceeded")
        raise HTTPException(status_code=502, detail="Embedding provider failed")

    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Invalid embedding provider response") from exc
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise HTTPException(status_code=502, detail="Invalid embedding provider response")
    return result


async def create_embeddings(
        body: dict[str, Any],
        provider_api_key: str,
) -> dict[str, Any]:
    """Mask input transiently, embed it, and return vectors without text."""
    original_input = body["input"]
    values = [original_input] if isinstance(original_input, str) else original_input
    masked_values = await _mask_inputs(values)

    upstream_payload = dict(body)
    upstream_payload["input"] = (
        masked_values[0] if isinstance(original_input, str) else masked_values
    )
    logger.info(
        "Creating masked embeddings model=%s inputs=%d",
        body["model"],
        len(values),
    )
    return await _request_openai(upstream_payload, provider_api_key)
