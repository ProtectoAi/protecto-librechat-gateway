import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.requests import Request

from protecto_gateway.embeddings import (
    create_embeddings,
    validate_embeddings_request,
)


def request_with_key(key: str) -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/v1/embeddings",
        "headers": [(b"authorization", f"Bearer {key}".encode())],
    })


class EmbeddingsValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.multiple(
            "protecto_gateway.embeddings",
            RAG_PROTECTO_USER_ID="rag-service",
        )
        self.config.start()

    def tearDown(self):
        self.config.stop()

    def test_accepts_text_inputs(self):
        body = {
            "model": "text-embedding-3-small",
            "input": ["document chunk", "retrieval query"],
        }
        result, provider_api_key = validate_embeddings_request(
            request_with_key("user-openai-key"), body,
        )
        self.assertIs(result, body)
        self.assertEqual(provider_api_key, "user-openai-key")

    def test_rejects_token_arrays_that_cannot_be_masked(self):
        with self.assertRaises(HTTPException) as raised:
            validate_embeddings_request(
                request_with_key("user-openai-key"),
                {"model": "text-embedding-3-small", "input": [[1, 2, 3]]},
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("RAG_CHECK_EMBEDDING_CTX_LENGTH=false", raised.exception.detail)

    def test_rejects_missing_forwarded_provider_key(self):
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/embeddings",
            "headers": [],
        })
        with self.assertRaises(HTTPException) as raised:
            validate_embeddings_request(
                request,
                {"model": "text-embedding-3-small", "input": "query"},
            )
        self.assertEqual(raised.exception.status_code, 401)


class EmbeddingsMaskingTests(unittest.IsolatedAsyncioTestCase):
    @patch("protecto_gateway.embeddings._request_openai", new_callable=AsyncMock)
    @patch("protecto_gateway.embeddings._mask_inputs", new_callable=AsyncMock)
    async def test_only_masked_text_is_sent_upstream(self, mask_inputs, request_openai):
        mask_inputs.return_value = ["Contact <PER>person-1</PER>"]
        expected_response = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        request_openai.return_value = expected_response
        original_body = {
            "model": "text-embedding-3-small",
            "input": ["Contact Alice"],
        }

        result = await create_embeddings(original_body, "user-openai-key")

        self.assertEqual(result, expected_response)
        self.assertEqual(original_body["input"], ["Contact Alice"])
        mask_inputs.assert_awaited_once_with(["Contact Alice"])
        upstream_payload = request_openai.await_args.args[0]
        upstream_api_key = request_openai.await_args.args[1]
        self.assertEqual(upstream_payload["input"], ["Contact <PER>person-1</PER>"])
        self.assertNotIn("Contact Alice", str(upstream_payload))
        self.assertEqual(upstream_api_key, "user-openai-key")
        self.assertNotIn("user-openai-key", str(upstream_payload))


if __name__ == "__main__":
    unittest.main()
