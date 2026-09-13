import asyncio
import json
import unittest

from fastapi import HTTPException

from protecto_gateway.history import (
    PROVIDER_FILE_UNSUPPORTED_MESSAGE,
    contains_provider_file_upload,
    latest_user_message_has_provider_file,
    reject_provider_file_upload,
)
from protecto_gateway.responses import (
    _responses_object,
    completed_response_text_stream,
    response_text_output,
)
from protecto_gateway.sse import (
    completed_chat_text_response,
    completed_chat_text_stream,
)


class ProviderFileUploadTests(unittest.TestCase):
    def test_upload_guidance_explains_security_and_text_alternative(self):
        self.assertIn("entire file as raw bytes to the LLM provider", PROVIDER_FILE_UNSUPPORTED_MESSAGE)
        self.assertIn("expose sensitive information", PROVIDER_FILE_UNSUPPORTED_MESSAGE)
        self.assertIn("not allowed in Secured-Chat", PROVIDER_FILE_UNSUPPORTED_MESSAGE)
        self.assertIn("Please use 'Upload as Text'", PROVIDER_FILE_UNSUPPORTED_MESSAGE)
        self.assertIn("masked before it is sent to the LLM", PROVIDER_FILE_UNSUPPORTED_MESSAGE)
        self.assertNotIn("[Protecto gateway error:", PROVIDER_FILE_UNSUPPORTED_MESSAGE)

    @staticmethod
    def _provider_file_content():
        return [{
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
        }]

    def test_detects_responses_input_file_data(self):
        content = [{
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": "report.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
            }],
        }]

        self.assertTrue(contains_provider_file_upload(content))

    def test_detects_chat_message_documents(self):
        messages = [{
            "role": "user",
            "content": "Review this file",
            "documents": [{"filename": "report.pdf"}],
        }]

        self.assertTrue(contains_provider_file_upload(messages))

    def test_detects_node_buffer_and_python_bytes(self):
        self.assertTrue(contains_provider_file_upload(b"binary"))
        self.assertTrue(contains_provider_file_upload({
            "type": "Buffer",
            "data": [37, 80, 68, 70],
        }))

    def test_allows_plain_text_and_image_urls(self):
        content = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}, {
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png"},
            }],
        }]

        self.assertFalse(contains_provider_file_upload(content))

    def test_rejection_has_user_facing_message(self):
        with self.assertRaises(HTTPException) as raised:
            reject_provider_file_upload({
                "type": "input_file",
                "file_data": "data:text/plain;base64,SGVsbG8=",
            })

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            raised.exception.detail,
            PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        )

    def test_only_current_user_upload_is_rejected(self):
        history = [{
            "role": "user",
            "content": self._provider_file_content(),
        }, {
            "role": "assistant",
            "content": PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        }, {
            "role": "user",
            "content": "Continue without the file",
        }]

        self.assertTrue(contains_provider_file_upload(history))
        self.assertFalse(latest_user_message_has_provider_file(history))

    def test_chat_completions_returns_normal_assistant_response(self):
        response = completed_chat_text_response(
            PROVIDER_FILE_UNSUPPORTED_MESSAGE,
            "chatcmpl-test",
            "OpenAI:test",
            123,
        )

        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(
            choice["message"]["content"],
            PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        )

    def test_streaming_chat_returns_content_without_error_wrapper(self):
        async def collect():
            return "".join([
                chunk async for chunk in completed_chat_text_stream(
                    PROVIDER_FILE_UNSUPPORTED_MESSAGE,
                    "chatcmpl-test",
                    "OpenAI:test",
                    123,
                )
            ])

        response_text = asyncio.run(collect())
        self.assertIn(PROVIDER_FILE_UNSUPPORTED_MESSAGE, response_text)
        self.assertNotIn("Protecto gateway error", response_text)
        self.assertIn("data: [DONE]", response_text)

    def test_responses_api_returns_normal_assistant_response(self):
        response_id = "resp_test"
        output = response_text_output(
            response_id,
            PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        )
        payload = _responses_object(
            response_id,
            "OpenAI:test",
            123,
            output,
        )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(
            payload["output_text"],
            PROVIDER_FILE_UNSUPPORTED_MESSAGE,
        )

    def test_streaming_responses_api_completes_normally(self):
        async def collect():
            return "".join([
                chunk async for chunk in completed_response_text_stream(
                    PROVIDER_FILE_UNSUPPORTED_MESSAGE,
                    "resp_test",
                    "OpenAI:test",
                    123,
                )
            ])

        events = asyncio.run(collect())
        self.assertIn("event: response.completed", events)
        self.assertIn(PROVIDER_FILE_UNSUPPORTED_MESSAGE, events)
        self.assertNotIn("Protecto gateway error", events)
        completed_data = [
            json.loads(line.removeprefix("data: "))
            for line in events.splitlines()
            if line.startswith("data: ")
        ][-1]
        self.assertEqual(completed_data["response"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
