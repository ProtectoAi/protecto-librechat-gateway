"""Opt-in real engine/API tests using synthetic, non-sensitive fixtures."""

import io
import os
import unittest
from unittest.mock import patch

import httpx
from PIL import Image, ImageDraw, ImageFont

from protecto_gateway.config import OCRSettings


def searchable_pdf() -> bytes:
    # Tiny deterministic fixture with a real PDF text layer (no extra PDF dependency).
    stream = b"BT /F1 24 Tf 50 740 Td (LOCAL OCR TEXT SAMPLE) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(result)


def image_fixture(file_format: str) -> bytes:
    with Image.new("RGB", (1400, 300), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.text((60, 100), "LOCAL OCR IMAGE SAMPLE", fill="black", font=ImageFont.load_default(size=60))
        stream = io.BytesIO()
        image.save(stream, format=file_format)
        return stream.getvalue()


@unittest.skipUnless(os.getenv("OCR_RUN_INTEGRATION") == "true", "Opt-in real OCR integration tests")
class OCRIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from protecto_gateway.app import app
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.settings = patch("protecto_gateway.ocr_api.OCR_SETTINGS", OCRSettings(enabled=True, api_key="x" * 32))
        self.settings.start()

    async def asyncTearDown(self):
        self.settings.stop()
        await self.client.aclose()

    async def test_real_pdf_image_and_scanned_pdf(self):
        from test_ocr import inline_request
        for mime, data, expected in (
            ("application/pdf", searchable_pdf(), "LOCAL OCR TEXT SAMPLE"),
            ("image/png", image_fixture("PNG"), "LOCAL OCR IMAGE SAMPLE"),
            ("image/jpeg", image_fixture("JPEG"), "LOCAL OCR IMAGE SAMPLE"),
            ("image/webp", image_fixture("WEBP"), "LOCAL OCR IMAGE SAMPLE"),
            ("image/gif", image_fixture("GIF"), "LOCAL OCR IMAGE SAMPLE"),
            ("image/tiff", image_fixture("TIFF"), "LOCAL OCR IMAGE SAMPLE"),
            ("image/bmp", image_fixture("BMP"), "LOCAL OCR IMAGE SAMPLE"),
            ("application/pdf", image_fixture("PDF"), "LOCAL OCR IMAGE SAMPLE"),
        ):
            with self.subTest(mime=mime, expected=expected):
                response = await self.client.post(
                    "/v1/ocr", content=inline_request(data, mime),
                    headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertIn(expected, result["pages"][0]["markdown"])
                self.assertEqual(result["usage_info"]["pages_processed"], 1)
                self.assertEqual(result["usage_info"]["doc_size_bytes"], len(data))
                self.assertEqual(result["model"], "local-tesseract")

    async def test_real_http_auth_invalid_file_and_no_echo(self):
        from test_ocr import inline_request
        response = await self.client.post("/v1/ocr", json={})
        self.assertEqual(response.status_code, 401)
        response = await self.client.post(
            "/v1/ocr", content=inline_request(b"confidential-invalid-file"),
            headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("confidential-invalid-file", response.text)
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/ocr", response.json()["endpoints"])
