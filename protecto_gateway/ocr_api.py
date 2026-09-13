"""Bounded, authenticated Mistral-compatible endpoint for LibreChat local OCR."""

import asyncio
import base64
import binascii
import hmac
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import ClientDisconnect

from .config import OCR_SETTINGS, OCRSettings, logger
from .ocr import IMAGE_FORMATS, OCRError


class OCRDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    type: Literal["document_url", "image_url"]
    document_url: str | None = None
    image_url: str | None = None


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    model: str = Field(default="local-tesseract", max_length=128)
    document: OCRDocument
    image_limit: int = Field(default=0, ge=0)
    include_image_base64: bool = False


def authenticate_ocr(authorization: str, settings: OCRSettings) -> None:
    if not settings.enabled:
        raise HTTPException(503, "Local OCR is disabled. Configure OCR_ENABLED and OCR_API_KEY.")
    scheme, _, key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        key.encode("utf-8"), settings.api_key.encode("utf-8"),
    ):
        raise HTTPException(401, "Invalid OCR credentials", headers={"WWW-Authenticate": "Bearer"})


def decode_ocr_request(raw: bytes | bytearray, settings: OCRSettings) -> tuple[bytes, str]:
    """Accept inline files only; never fetch URLs or read client-supplied paths."""
    if len(raw) > settings.max_body_bytes:
        raise OCRError("OCR request exceeds the upload limit.", 413)
    try:
        payload = OCRRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise OCRError("Invalid OCR request format.", 400) from exc
    document = payload.document
    value = document.document_url if document.type == "document_url" else document.image_url
    other = document.image_url if document.type == "document_url" else document.document_url
    if not value or other is not None:
        raise OCRError("Provide exactly one inline document matching its type.", 400)
    header, separator, encoded = value.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise OCRError("Only base64 inline files are supported. Use LibreChat azure_mistral_ocr strategy.", 400)
    mime_type = header[5:-7].lower()
    if mime_type not in {"application/pdf", *IMAGE_FORMATS}:
        raise OCRError("OCR supports PDF, PNG, JPEG, WebP, GIF, TIFF and BMP only.", 415)
    if (document.type == "image_url") != mime_type.startswith("image/"):
        raise OCRError("Document type does not match its MIME type.", 400)
    if len(encoded) > 4 * ((settings.max_file_bytes + 2) // 3):
        raise OCRError("File exceeds the OCR upload limit.", 413)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OCRError("Invalid base64 file content.", 400) from exc
    if not data:
        raise OCRError("The uploaded file is empty.", 400)
    if len(data) > settings.max_file_bytes:
        raise OCRError("File exceeds the OCR upload limit.", 413)
    return data, mime_type


async def read_ocr_body(request: Request, settings: OCRSettings) -> bytearray:
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise OCRError("OCR requires an application/json request.", 415)
    body = bytearray()
    async with asyncio.timeout(30):
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_body_bytes:
                raise OCRError("OCR request exceeds the upload limit.", 413)
            body.extend(chunk)
    return body


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    # Kill the group, including a Poppler/Tesseract child, before deleting files.
    if process.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def run_ocr_worker(data: bytes, mime_type: str, request: Request) -> dict:
    with TemporaryDirectory(prefix="protecto-ocr-") as directory:
        source = Path(directory) / "document"
        output = Path(directory) / "result.json"
        await asyncio.to_thread(source.write_bytes, data)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "protecto_gateway.ocr", str(source), mime_type, str(output),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        try:
            async with asyncio.timeout(OCR_SETTINGS.timeout_seconds):
                while process.returncode is None:
                    if await request.is_disconnected():
                        raise OCRError("OCR request was disconnected.", 499)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=0.25)
                    except TimeoutError:
                        continue
                if process.returncode != 0 or not output.exists():
                    raise OCRError("OCR worker stopped before completion. Try a smaller file.")
                if output.stat().st_size > OCR_SETTINGS.max_text_chars * 12 + 65536:
                    raise OCRError("Extracted text exceeds the OCR output limit.", 413)
                result = json.loads(await asyncio.to_thread(output.read_text, encoding="utf-8"))
                if "detail" in result:
                    raise OCRError(result["detail"], result["status_code"])
                return result
        finally:
            await _stop_worker(process)


router = APIRouter()
_slots = asyncio.Semaphore(OCR_SETTINGS.max_concurrent)


@router.post("/v1/ocr")
async def local_ocr(request: Request) -> JSONResponse:
    authenticate_ocr(request.headers.get("authorization", ""), OCR_SETTINGS)
    if _slots.locked():
        raise HTTPException(429, "Local OCR is busy. Please retry shortly.", headers={"Retry-After": "5"})
    request_id = uuid4().hex
    started = time.monotonic()
    async with _slots:
        try:
            raw = await read_ocr_body(request, OCR_SETTINGS)
            data, mime_type = await asyncio.to_thread(decode_ocr_request, raw, OCR_SETTINGS)
            del raw
            logger.info("[OCR START] request_id=%s type=%s bytes=%d", request_id, mime_type, len(data))
            result = await run_ocr_worker(data, mime_type, request)
        except ClientDisconnect as exc:
            logger.info("[OCR DISCONNECTED] request_id=%s", request_id)
            raise HTTPException(499, "OCR upload was disconnected.") from exc
        except TimeoutError as exc:
            logger.warning("[OCR TIMEOUT] request_id=%s", request_id)
            raise HTTPException(504, "OCR timed out. Please split the document into smaller files.") from exc
        except OCRError as exc:
            logger.warning("[OCR REJECTED] request_id=%s status=%d", request_id, exc.status_code)
            raise HTTPException(exc.status_code, exc.detail) from exc
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[OCR FAILED] request_id=%s error_type=%s", request_id, type(exc).__name__)
            raise HTTPException(503, "Local OCR is unavailable. Please contact the administrator.") from exc
    logger.info(
        "[OCR COMPLETE] request_id=%s pages=%d elapsed_seconds=%.2f",
        request_id, result["usage_info"]["pages_processed"], time.monotonic() - started,
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store", "X-Request-ID": request_id})
