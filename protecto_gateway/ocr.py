"""Local PDF/image extraction in a disposable, resource-limited worker process.

No network requests, persistent document store, or document-content logging.
The API owns the temporary directory and kills the process group on timeout.
"""

import json
import os
import re
import resource
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import OCR_SETTINGS, OCRSettings, logger


IMAGE_FORMATS = {
    "image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP",
    "image/gif": "GIF", "image/tiff": "TIFF", "image/bmp": "BMP",
}


class OCRError(Exception):
    """Safe, user-facing extraction failure (never include document content)."""

    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class OCRPage(TypedDict):
    index: int
    markdown: str
    images: list


def _command(args: list[str], output: Path, settings: OCRSettings) -> None:
    """Discard parser diagnostics: they may contain document data."""
    try:
        with output.open("wb") as stream:
            result = subprocess.run(
                args, stdout=stream, stderr=subprocess.DEVNULL,
                timeout=settings.timeout_seconds, check=False,
                env={**os.environ, "LC_ALL": "C", "OMP_THREAD_LIMIT": "1"},
            )
    except FileNotFoundError as exc:
        raise OCRError("Local OCR dependencies are unavailable. Rebuild the gateway image.", 503) from exc
    except subprocess.TimeoutExpired as exc:
        raise OCRError("OCR timed out. Please split the document into smaller files.", 504) from exc
    if result.returncode:
        raise OCRError("Unable to read this file. Check that it is valid and not password-protected.")


def _read_text(path: Path, settings: OCRSettings) -> str:
    if path.stat().st_size > settings.max_text_chars * 4:
        raise OCRError("Extracted text exceeds the OCR output limit.", 413)
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x0c", "").strip()
    if len(text) > settings.max_text_chars:
        raise OCRError("Extracted text exceeds the OCR output limit.", 413)
    return text


def _image_text(image_path: Path, workdir: Path, settings: OCRSettings) -> str:
    text_path = workdir / "text.txt"
    _command(
        ["tesseract", str(image_path), "stdout", "-l", settings.languages],
        text_path, settings,
    )
    return _read_text(text_path, settings)


def _pdf_pages(path: Path, workdir: Path, settings: OCRSettings) -> list[OCRPage]:
    info_path = workdir / "info.txt"
    _command(["pdfinfo", str(path)], info_path, settings)
    info = _read_text(info_path, settings)
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match or re.search(r"^Encrypted:\s+yes", info, re.MULTILINE):
        raise OCRError("PDF is invalid or password-protected. Upload an unlocked PDF.")
    count = int(match.group(1))
    if not 1 <= count <= settings.max_pages:
        raise OCRError("PDF exceeds the OCR page limit. Please split the document.", 413)
    pages: list[OCRPage] = []
    total_chars = 0
    for page_number in range(1, count + 1):
        text = ""
        if settings.pdf_mode == "auto":
            text_path = workdir / "text.txt"
            _command(
                ["pdftotext", "-f", str(page_number), "-l", str(page_number),
                 "-layout", "-enc", "UTF-8", str(path), "-"],
                text_path, settings,
            )
            text = _read_text(text_path, settings)
        if not text:
            image_prefix = workdir / "page"
            _command(
                ["pdftoppm", "-f", str(page_number), "-l", str(page_number),
                 "-singlefile", "-scale-to", str(settings.render_max_side),
                 "-gray", "-png", str(path), str(image_prefix)],
                workdir / "render.txt", settings,
            )
            text = _image_text(workdir / "page.png", workdir, settings)
            (workdir / "page.png").unlink()
        total_chars += len(text)
        if total_chars > settings.max_text_chars:
            raise OCRError("Extracted text exceeds the OCR output limit.", 413)
        pages.append({"index": page_number - 1, "markdown": text, "images": []})
    return pages


def _image_pages(
    path: Path, mime_type: str, workdir: Path, settings: OCRSettings,
) -> list[OCRPage]:
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    pages: list[OCRPage] = []
    total_chars = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != IMAGE_FORMATS[mime_type]:
                    raise OCRError("Image content does not match its declared file type.")
                count = getattr(image, "n_frames", 1)
                if count > settings.max_pages:
                    raise OCRError("Image exceeds the OCR frame/page limit.", 413)
                for index in range(count):
                    image.seek(index)
                    if image.width * image.height > settings.max_image_pixels:
                        raise OCRError("Image exceeds the OCR pixel limit.", 413)
                    with ImageOps.exif_transpose(image) as oriented:
                        with oriented.convert("RGBA") as rgba:
                            with Image.new("RGBA", rgba.size, "white") as background:
                                background.alpha_composite(rgba)
                                with background.convert("RGB") as normalized:
                                    normalized.thumbnail((settings.render_max_side,) * 2)
                                    normalized.save(workdir / "page.png")
                    text = _image_text(workdir / "page.png", workdir, settings)
                    (workdir / "page.png").unlink()
                    total_chars += len(text)
                    if total_chars > settings.max_text_chars:
                        raise OCRError("Extracted text exceeds the OCR output limit.", 413)
                    pages.append({"index": index, "markdown": text, "images": []})
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise OCRError("Image exceeds the OCR pixel limit.", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError, EOFError) as exc:
        raise OCRError("Unable to read this image. Upload a valid supported image.") from exc
    return pages


def extract_document(path: Path, mime_type: str, settings: OCRSettings) -> dict:
    """Extract ordered pages using text-first PDF parsing or local Tesseract."""
    if path.stat().st_size > settings.max_file_bytes:
        raise OCRError("File exceeds the OCR upload limit.", 413)
    if mime_type == "application/pdf":
        with path.open("rb") as stream:
            if not stream.read(1024).lstrip().startswith(b"%PDF-"):
                raise OCRError("File content is not a PDF.")
        pages = _pdf_pages(path, path.parent, settings)
    elif mime_type in IMAGE_FORMATS:
        pages = _image_pages(path, mime_type, path.parent, settings)
    else:
        raise OCRError("OCR supports PDF, PNG, JPEG, WebP, GIF, TIFF and BMP only.", 415)
    if not any(page["markdown"].strip() for page in pages):
        raise OCRError("No readable text was found. Try a clearer scan or another OCR language.")
    return {
        "pages": pages,
        "model": "local-tesseract",
        "usage_info": {"pages_processed": len(pages), "doc_size_bytes": path.stat().st_size},
    }


def _worker_main() -> None:
    # Limits also apply to Poppler/Tesseract children. Parent enforces wall time.
    if sys.platform.startswith("linux"):
        memory = OCR_SETTINGS.worker_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    file_limit = max(64 * 1024 * 1024, OCR_SETTINGS.max_text_chars * 12)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    try:
        result = extract_document(Path(sys.argv[1]), sys.argv[2], OCR_SETTINGS)
    except OCRError as exc:
        result = {"detail": exc.detail, "status_code": exc.status_code}
    except Exception as exc:
        # The worker boundary must never serialize parser messages or a traceback.
        logger.error("[OCR WORKER FAILED] error_type=%s", type(exc).__name__)
        result = {"detail": "Local OCR could not process this document.", "status_code": 422}
    Path(sys.argv[3]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _worker_main()
