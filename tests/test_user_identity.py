import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import Headers

from protecto_gateway import history


class _StopAfterAuth(Exception):
    """Raised by the patched auth call so the test stops at the identity step."""


class _FakeRequest:
    def __init__(self, headers):
        self.headers = Headers(headers)


def _headers(**extra):
    base = {
        "x-openai-api-key": "provider-key",
    }
    base.update(extra)
    return base


class UserIdentityHeaderTests(unittest.IsolatedAsyncioTestCase):
    """The Protecto user id comes from X-User-Email, normalized."""

    async def _prepare(self, headers):
        """Run prepare_request far enough to capture the Protecto user id."""
        captured = {}

        async def fake_auth(*, user_id, **_kwargs):
            captured["user_id"] = user_id
            raise _StopAfterAuth

        with patch.multiple(
            history,
            PROTECTO_URL="https://protecto.example.com/vault",
            PROTECTO_MASTER_TOKEN="master-token",
            PROTECTO_NAMESPACE="namespace",
            get_or_create_auth_token=fake_auth,
        ):
            try:
                await history.prepare_request(
                    request=_FakeRequest(headers),
                    raw_messages=[{"role": "user", "content": "hello"}],
                    request_tools=[],
                    model_string="openai:gpt-5.6-terra",
                )
            except _StopAfterAuth:
                pass
        return captured

    async def test_email_header_becomes_the_protecto_user_id(self):
        captured = await self._prepare(
            _headers(**{"x-user-email": "abhishek@protecto.ai"}),
        )
        self.assertEqual(captured["user_id"], "abhishek@protecto.ai")

    async def test_email_is_trimmed_and_lowercased(self):
        captured = await self._prepare(
            _headers(**{"x-user-email": "  Abhishek@Protecto.AI  "}),
        )
        self.assertEqual(captured["user_id"], "abhishek@protecto.ai")

    async def test_missing_email_header_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._prepare(_headers())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Missing x-user-email")

    async def test_blank_email_header_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._prepare(_headers(**{"x-user-email": "   "}))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Missing x-user-email")

    async def test_username_header_is_not_a_fallback(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._prepare(_headers(**{"x-user-username": "abhishek"}))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Missing x-user-email")


if __name__ == "__main__":
    unittest.main()
