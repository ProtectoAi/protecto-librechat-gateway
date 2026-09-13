import asyncio
import unittest
from unittest.mock import patch

from protecto_gateway.protecto import (
    _AUTH_TOKEN_LOCKS,
    get_or_create_auth_token,
)


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self._data}


class _ProtectoClient:
    def __init__(self):
        self.token = None
        self.generate_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def put(self, url, **_kwargs):
        if url.endswith("/fetch"):
            await asyncio.sleep(0)
            if self.token is None:
                return _Response({})
            return _Response({
                "auth_token": self.token,
                "end_date": "2099-01-01 00:00:00",
            })

        self.generate_count += 1
        await asyncio.sleep(0.01)
        self.token = "shared-auth-token"
        return _Response({"auth_key": self.token})


class ProtectoAuthConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _AUTH_TOKEN_LOCKS.clear()

    async def test_concurrent_requests_generate_only_one_user_token(self):
        client = _ProtectoClient()
        kwargs = {
            "namespace_name": "namespace",
            "user_id": "user",
            "headers": {"Authorization": "Bearer master"},
            "protecto_url": "https://protecto.example.com/vault",
        }

        with patch(
            "protecto_gateway.protecto.httpx.AsyncClient",
            return_value=client,
        ):
            tokens = await asyncio.gather(
                get_or_create_auth_token(**kwargs),
                get_or_create_auth_token(**kwargs),
            )

        self.assertEqual(tokens, ["shared-auth-token", "shared-auth-token"])
        self.assertEqual(client.generate_count, 1)


if __name__ == "__main__":
    unittest.main()
