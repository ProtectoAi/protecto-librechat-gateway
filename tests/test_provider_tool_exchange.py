"""
Regression tests for the tool exchange that each provider actually receives.

The masking tests assert on intermediate state (what is submitted to Protecto).
What actually broke in the skill-loop bug is whether the PROVIDER ever sees the
prior assistant tool_calls and tool results: when history was truncated to the
trailing user message, every round sent a byte-identical payload and the model
re-emitted the same tool call forever.

These tests drive the real path - build_masked_history -> mask_pending_messages
-> build_<provider>_payload - and assert on the provider payload itself.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from protecto_gateway.gemini import build_gemini_payload
from protecto_gateway.history import build_masked_history, mask_pending_messages
from protecto_gateway.openai_provider import build_openai_payload


SKILL_INSTRUCTIONS = (
    "Review the user's text for personally identifiable information (PII)."
)


def looping_skill_conversation() -> list[dict]:
    """
    The shape LibreChat replays during a Skill loop.

    No assistant message carries a behind-the-scenes artifact: a tool-call turn
    is content-free by the BUFFER_TEXT_WHEN_TOOLS invariant. LibreChat injects
    the skill instructions as a NEW user message after each tool result, so the
    last message in the list is a user message rather than the tool result.
    """
    return [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Check this text for PII please"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "skill",
                "arguments": '{"skillName":"pii-reviewer"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "call-1", "name": "skill",
         "content": "skill loaded: pii-reviewer"},
        {"role": "user", "content": SKILL_INSTRUCTIONS},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-2",
            "type": "function",
            "function": {
                "name": "skill",
                "arguments": '{"skillName":"pii-reviewer"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "call-2", "name": "skill",
         "content": "skill loaded: pii-reviewer"},
        {"role": "user", "content": SKILL_INSTRUCTIONS},
    ]


class ProviderToolExchangeTests(unittest.TestCase):
    @staticmethod
    def _mask_values(values):
        return [f"<MASKED>{index}</MASKED>" for index, _ in enumerate(values)]

    def _masked_messages(self, raw_messages, mask_tool_results=True):
        masked_messages, token_map, pending = build_masked_history(raw_messages)
        fake_mask = AsyncMock(
            side_effect=lambda _client, values, *_args: self._mask_values(values)
        )
        with (
            patch("protecto_gateway.history.mask_values_async", fake_mask),
            patch("protecto_gateway.history.MASK_TOOL_RESULTS", mask_tool_results),
        ):
            asyncio.run(mask_pending_messages(
                client=AsyncMock(),
                pending=pending,
                masked_messages=masked_messages,
                token_map=token_map,
                protecto_mask_url="https://protecto.example/mask",
                protecto_headers={"Authorization": "Bearer test"},
            ))
        return masked_messages

    @staticmethod
    def _steps_of_type(payload, step_type):
        return [
            item for item in payload["input"]
            if isinstance(item, dict) and item.get("type") == step_type
        ]

    # ---- OpenAI (Responses API) ----

    def test_openai_payload_carries_full_tool_exchange(self):
        masked = self._masked_messages(looping_skill_conversation())
        payload = build_openai_payload("openai-test", masked, None, None, False)

        calls = self._steps_of_type(payload, "function_call")
        outputs = self._steps_of_type(payload, "function_call_output")

        self.assertEqual(
            ["call-1", "call-2"], [item["call_id"] for item in calls],
            "both prior assistant tool calls must reach the provider",
        )
        self.assertEqual(
            ["call-1", "call-2"], [item["call_id"] for item in outputs],
            "both prior tool results must reach the provider",
        )
        for item in calls:
            self.assertEqual("skill", item["name"])
            # OpenAI receives arguments as a JSON STRING.
            self.assertIsInstance(item["arguments"], str)

    def test_openai_payload_has_no_orphaned_function_call_output(self):
        masked = self._masked_messages(looping_skill_conversation())
        payload = build_openai_payload("openai-test", masked, None, None, False)

        seen_call_ids = set()
        for item in payload["input"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                seen_call_ids.add(item["call_id"])
            elif item.get("type") == "function_call_output":
                self.assertIn(
                    item["call_id"], seen_call_ids,
                    "a function_call_output without a preceding function_call "
                    "of the same call_id is rejected by the provider",
                )

    # ---- Gemini (Interactions API) ----

    def test_gemini_payload_carries_full_tool_exchange(self):
        masked = self._masked_messages(looping_skill_conversation())
        # Force the stateless full-history branch: lookup_gemini_state reads a
        # process-global that other tests may have populated.
        payload = build_gemini_payload("gemini-test", masked, None, False, {})

        calls = self._steps_of_type(payload, "function_call")
        results = self._steps_of_type(payload, "function_result")

        self.assertEqual(
            ["call-1", "call-2"], [item["id"] for item in calls],
            "both prior assistant tool calls must reach the provider",
        )
        self.assertEqual(
            ["call-1", "call-2"], [item["call_id"] for item in results],
            "both prior tool results must reach the provider",
        )
        for item in calls:
            self.assertEqual("skill", item["name"])
            # Gemini receives arguments as a DICT, unlike OpenAI.
            self.assertIsInstance(item["arguments"], dict)

    def test_gemini_payload_has_no_orphaned_function_result(self):
        masked = self._masked_messages(looping_skill_conversation())
        payload = build_gemini_payload("gemini-test", masked, None, False, {})

        seen_call_ids = set()
        for item in payload["input"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                seen_call_ids.add(item["id"])
            elif item.get("type") == "function_result":
                self.assertIn(item["call_id"], seen_call_ids)


if __name__ == "__main__":
    unittest.main()
