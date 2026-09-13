import asyncio
import base64
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request

from protecto_gateway.config import OCRSettings
from protecto_gateway.ocr import OCRError, _command, _image_pages, _pdf_pages, extract_document
from protecto_gateway.ocr_api import (
    authenticate_ocr, decode_ocr_request, local_ocr, read_ocr_body, run_ocr_worker,
)


def inline_request(data=b"%PDF-1.4\n", mime="application/pdf"):
    kind = "image_url" if mime.startswith("image/") else "document_url"
    return json.dumps({
        "model": "local-tesseract", "image_limit": 0, "include_image_base64": False,
        "document": {"type": kind, kind: f"data:{mime};base64,{base64.b64encode(data).decode()}"},
    }).encode()


def http_request(body, headers=()):
    receive = AsyncMock(return_value={"type": "http.request", "body": body, "more_body": False})
    return Request({
        "type": "http", "method": "POST", "path": "/v1/ocr",
        "headers": [(b"content-type", b"application/json"), *headers],
    }, receive)


class OCRValidationTests(unittest.TestCase):
    def setUp(self):
        self.settings = OCRSettings()

    def test_config_rejects_invalid_values(self):
        for kwargs in ({"enabled": True}, {"languages": "../../eng"},
                       {"max_pages": 0}, {"pdf_mode": "bad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OCRSettings(**kwargs)

    def test_config_never_displays_key(self):
        self.assertNotIn("private-secret", repr(OCRSettings(api_key="private-secret")))

    def test_auth_disabled_and_invalid(self):
        with self.assertRaises(HTTPException) as error:
            authenticate_ocr("Bearer test", self.settings)
        self.assertEqual(error.exception.status_code, 503)
        settings = replace(self.settings, enabled=True, api_key="x" * 32)
        for value in ("", "Basic " + "x" * 32, "Bearer invalid", "Bearer \u2603"):
            with self.assertRaises(HTTPException) as error:
                authenticate_ocr(value, settings)
            self.assertEqual(error.exception.status_code, 401)
        authenticate_ocr("Bearer " + "x" * 32, settings)

    def test_inline_pdf_and_image(self):
        for mime, data in (("application/pdf", b"%PDF-test"), ("image/png", b"image")):
            self.assertEqual(decode_ocr_request(inline_request(data, mime), self.settings), (data, mime))

    def test_invalid_requests_are_safe(self):
        requests = [b"not json", b"[]", b"{}", inline_request(b"", "image/png")]
        for url in ("http://169.254.169.254/secret", "file:///etc/passwd", "data:image/png;base64,???"):
            requests.append(json.dumps({"document": {"type": "image_url", "image_url": url}}).encode())
        requests.append(inline_request(b"file", "image/svg+xml"))
        requests.append(json.dumps({"document": {"type": "image_url", "document_url": "secret"}}).encode())
        requests.append(inline_request().replace(b'"document_url",', b'"image_url",'))
        for body in requests:
            with self.subTest(body=body), self.assertRaises(OCRError) as error:
                decode_ocr_request(body, self.settings)
            self.assertNotIn("secret", str(error.exception))

    def test_decoded_file_size_boundary(self):
        settings = replace(self.settings, max_file_mb=1)
        data = b"a" * settings.max_file_bytes
        self.assertEqual(len(decode_ocr_request(inline_request(data), settings)[0]), len(data))
        with self.assertRaises(OCRError) as error:
            decode_ocr_request(inline_request(data + b"a"), settings)
        self.assertEqual(error.exception.status_code, 413)


class OCREngineTests(unittest.TestCase):
    def test_pdf_auto_preserves_text_and_ocrs_empty_page(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            def fake_command(args, output, settings):
                if args[0] == "pdfinfo":
                    output.write_text("Pages: 2\nEncrypted: no\n")
                elif args[0] == "pdftotext":
                    output.write_text("Searchable text" if args[2] == "1" else "\x0c")
                else:
                    (workdir / "page.png").touch()
            with patch("protecto_gateway.ocr._command", side_effect=fake_command), \
                    patch("protecto_gateway.ocr._image_text", return_value="Scanned text") as ocr:
                pages = _pdf_pages(workdir / "document", workdir, OCRSettings())
            self.assertEqual([page["markdown"] for page in pages], ["Searchable text", "Scanned text"])
            self.assertEqual([page["index"] for page in pages], [0, 1])
            ocr.assert_called_once()

    def test_pdf_always_uses_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            def fake_command(args, output, settings):
                if args[0] == "pdfinfo":
                    output.write_text("Pages: 1\nEncrypted: no\n")
                else:
                    self.assertEqual(args[0], "pdftoppm")
                    (workdir / "page.png").touch()
            with patch("protecto_gateway.ocr._command", side_effect=fake_command), \
                    patch("protecto_gateway.ocr._image_text", return_value="OCR text"):
                pages = _pdf_pages(workdir / "doc", workdir, OCRSettings(pdf_mode="always"))
            self.assertEqual(pages[0]["markdown"], "OCR text")

    def test_encrypted_or_too_many_pdf_pages_rejected_before_rendering(self):
        for info in ("Pages: 51\nEncrypted: no", "Pages: 1\nEncrypted: yes"):
            with tempfile.TemporaryDirectory() as directory:
                workdir = Path(directory)
                with patch("protecto_gateway.ocr._command", side_effect=lambda a, o, s: o.write_text(info)) as command:
                    with self.assertRaises(OCRError):
                        _pdf_pages(workdir / "doc", workdir, OCRSettings())
                command.assert_called_once()

    def test_invalid_pdf_signature_and_unsupported_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document"
            path.write_bytes(b"not a pdf")
            for mime in ("application/pdf", "application/msword"):
                with self.assertRaises(OCRError):
                    extract_document(path, mime, OCRSettings())

    def test_image_validation_frames_and_blank_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image"
            with Image.new("RGB", (20, 20), "white") as image:
                image.save(path, format="PNG")
            with self.assertRaises(OCRError):
                _image_pages(path, "image/jpeg", path.parent, OCRSettings())
            with self.assertRaises(OCRError) as error:
                _image_pages(path, "image/png", path.parent, OCRSettings(max_image_pixels=100))
            self.assertEqual(error.exception.status_code, 413)
            with patch("protecto_gateway.ocr._image_text", return_value=""):
                with self.assertRaises(OCRError):
                    extract_document(path, "image/png", OCRSettings())
            with Image.new("RGB", (20, 20), "white") as first, Image.new("RGB", (20, 20), "black") as second:
                first.save(path, format="TIFF", save_all=True, append_images=[second])
            with self.assertRaises(OCRError):
                _image_pages(path, "image/tiff", path.parent, OCRSettings(max_pages=1))
            with patch("protecto_gateway.ocr._image_text", side_effect=["First", "Second"]):
                self.assertEqual(len(_image_pages(path, "image/tiff", path.parent, OCRSettings())), 2)

    def test_subprocess_errors_do_not_expose_diagnostics(self):
        errors = (FileNotFoundError("secret"), subprocess.TimeoutExpired("secret", 1))
        with tempfile.TemporaryDirectory() as directory:
            for error in errors:
                with patch("subprocess.run", side_effect=error), self.assertRaises(OCRError) as caught:
                    _command(["test"], Path(directory) / "out", OCRSettings())
                self.assertNotIn("secret", str(caught.exception))


class OCRAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_timeout_and_disconnect_cleanup(self):
        for disconnected in (False, True):
            with self.subTest(disconnected=disconnected):
                process = Mock(pid=12345, returncode=None)
                async def waiting():
                    if process.returncode is None:
                        await asyncio.sleep(30)
                    return -9
                process.wait = waiting
                request = Mock(is_disconnected=AsyncMock(return_value=disconnected))
                def killed(*args):
                    process.returncode = -9
                with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn, \
                        patch("os.killpg", side_effect=killed) as kill, \
                        patch("protecto_gateway.ocr_api.OCR_SETTINGS", OCRSettings(timeout_seconds=1)):
                    expected = OCRError if disconnected else TimeoutError
                    with self.assertRaises(expected):
                        await run_ocr_worker(b"document", "application/pdf", request)
                    kill.assert_called_once()
                    self.assertFalse(Path(spawn.call_args.args[3]).parent.exists())

    async def test_failed_route_releases_capacity_without_logging_content(self):
        settings = OCRSettings(enabled=True, api_key="x" * 32)
        request = http_request(inline_request(b"private-sample"), [(b"authorization", b"Bearer " + b"x" * 32)])
        slots = asyncio.Semaphore(1)
        with patch("protecto_gateway.ocr_api.OCR_SETTINGS", settings), \
                patch("protecto_gateway.ocr_api._slots", slots), \
                patch("protecto_gateway.ocr_api.run_ocr_worker", AsyncMock(side_effect=TimeoutError())), \
                self.assertLogs("protecto_gateway", level="INFO") as logs:
            with self.assertRaises(HTTPException) as error:
                await local_ocr(request)
        self.assertEqual(error.exception.status_code, 504)
        self.assertFalse(slots.locked())
        self.assertNotIn("private-sample", " ".join(logs.output))
        self.assertNotIn("x" * 32, " ".join(logs.output))

    async def test_streamed_body_limit_does_not_trust_content_length(self):
        settings = OCRSettings(max_file_mb=1)
        request = http_request(b"x" * (settings.max_body_bytes + 1))
        with self.assertRaises(OCRError) as error:
            await read_ocr_body(request, settings)
        self.assertEqual(error.exception.status_code, 413)

    async def test_auth_happens_before_body_read(self):
        request = http_request(b"secret")
        with patch("protecto_gateway.ocr_api.OCR_SETTINGS", OCRSettings()), \
                patch("protecto_gateway.ocr_api.read_ocr_body", new_callable=AsyncMock) as read:
            with self.assertRaises(HTTPException):
                await local_ocr(request)
        read.assert_not_awaited()

    async def test_route_success_and_busy(self):
        settings = OCRSettings(enabled=True, api_key="x" * 32)
        request = http_request(inline_request(), [(b"authorization", b"Bearer " + b"x" * 32)])
        result = {"pages": [{"index": 0, "markdown": "Sample", "images": []}], "usage_info": {"pages_processed": 1}}
        slots = asyncio.Semaphore(1)
        with patch("protecto_gateway.ocr_api.OCR_SETTINGS", settings), \
                patch("protecto_gateway.ocr_api._slots", slots), \
                patch("protecto_gateway.ocr_api.run_ocr_worker", AsyncMock(return_value=result)):
            response = await local_ocr(request)
            self.assertEqual(json.loads(response.body), result)
            self.assertEqual(response.headers["cache-control"], "no-store")
            await slots.acquire()
            try:
                with self.assertRaises(HTTPException) as error:
                    await local_ocr(request)
                self.assertEqual(error.exception.status_code, 429)
            finally:
                slots.release()

    async def test_worker_cancellation_kills_children_and_cleans_files(self):
        process = Mock(pid=12345, returncode=None, wait=AsyncMock())
        started = asyncio.Event()
        request = Mock()
        async def connected():
            started.set()
            await asyncio.sleep(30)
            return False
        request.is_disconnected = connected
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn, \
                patch("os.killpg") as kill:
            task = asyncio.create_task(run_ocr_worker(b"document", "application/pdf", request))
            await started.wait()
            path = Path(spawn.call_args.args[3])
            self.assertTrue(path.exists())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            kill.assert_called_once()
            self.assertFalse(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
