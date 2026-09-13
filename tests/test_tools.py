import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from protecto_gateway.tools import unmask_tool_calls_async


class ToolUnmaskingTests(unittest.TestCase):
    def test_unmask_tool_calls_uses_protecto_unmask(self):
        unmask = AsyncMock(return_value='{"query":"Alice"}')
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "file_search",
                    "arguments": '{"query":"<PERSON_1>"}',
                },
            }
        ]

        with patch("protecto_gateway.tools._unmask_call_async", unmask):
            result = asyncio.run(unmask_tool_calls_async(
                AsyncMock(),
                tool_calls,
                "https://protecto.example.com/unmask",
                {"Authorization": "Bearer test"},
            ))

        self.assertEqual(
            result[0]["function"]["arguments"],
            '{"query":"Alice"}',
        )
        unmask.assert_awaited_once()
