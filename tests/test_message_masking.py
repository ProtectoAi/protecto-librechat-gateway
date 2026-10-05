import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from protecto_gateway.artifacts import build_behind_scenes_artifact
from protecto_gateway.gemini import build_gemini_payload
from protecto_gateway.history import (
    add_trusted_system_prompt,
    build_masked_history,
    mask_pending_messages,
)
from protecto_gateway.openai_provider import build_openai_payload
from protecto_gateway.responses import responses_request_to_messages


class MessageMaskingTests(unittest.TestCase):
    @staticmethod
    def _mask_values(values):
        return [f"<MASKED>{index}</MASKED>" for index, _ in enumerate(values)]

    def _run_masking(self, raw_messages, mask_tool_results=True):
        masked_messages, token_map, pending = build_masked_history(raw_messages)
        fake_mask = AsyncMock(side_effect=lambda _client, values, *_args: self._mask_values(values))
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
        return masked_messages, fake_mask.await_args.args[1]

    def test_masks_librechat_system_context_before_both_provider_conversions(self):
        original_context = "Retrieved bank statement belongs to Private Person"
        raw_messages = [
            {"role": "system", "content": original_context},
            {"role": "developer", "content": "Private developer context"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Summarize Private Person"}],
            },
        ]

        masked_messages, submitted_values = self._run_masking(raw_messages)
        add_trusted_system_prompt(masked_messages, "Trusted gateway policy")

        self.assertEqual(submitted_values, [
            original_context,
            "Private developer context",
            "Summarize Private Person",
        ])
        gemini_payload = build_gemini_payload(
            "gemini-test", masked_messages, None, False,
        )
        openai_payload = build_openai_payload(
            "openai-test", masked_messages, None, None, False,
        )
        serialized = json.dumps(
            {"gemini": gemini_payload, "openai": openai_payload},
            ensure_ascii=False,
        )
        self.assertNotIn("Private Person", serialized)
        self.assertNotIn("Private developer context", serialized)
        self.assertIn("Trusted gateway policy", gemini_payload["system_instruction"])
        self.assertIn("<MASKED>0</MASKED>", gemini_payload["system_instruction"])
        self.assertIn("<MASKED>0</MASKED>", openai_payload["instructions"])

    def test_masks_responses_api_top_level_instructions(self):
        original_instructions = "Retrieved file context for Private Person"
        raw_messages, _ = responses_request_to_messages({
            "instructions": original_instructions,
            "input": "Summarize this file",
        })

        masked_messages, submitted_values = self._run_masking(raw_messages)

        self.assertEqual(submitted_values, [
            original_instructions,
            "Summarize this file",
        ])
        openai_payload = build_openai_payload(
            "openai-test", masked_messages, None, None, False,
        )
        self.assertEqual(openai_payload["instructions"], "<MASKED>0</MASKED>")
        self.assertNotIn(original_instructions, json.dumps(openai_payload))

    def test_tool_metadata_is_not_masked_or_modified(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Look up a customer account",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                },
            },
        }]
        masked_messages, submitted_values = self._run_masking([
            {"role": "user", "content": "Find Private Person"},
        ])

        self.assertEqual(submitted_values, ["Find Private Person"])
        openai_payload = build_openai_payload(
            "openai-test", masked_messages, tools, None, False,
        )
        provider_tool = openai_payload["tools"][0]
        self.assertEqual(provider_tool["name"], "lookup_customer")
        self.assertEqual(provider_tool["description"], "Look up a customer account")
        self.assertEqual(
            provider_tool["parameters"], tools[0]["function"]["parameters"],
        )

    def test_reuses_identical_live_instruction_without_masking_it_again(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {"role": "system", "content": "File for <PER>person-1</PER>"},
                {"role": "user", "content": "Earlier prompt"},
            ],
            token_map={"person-1": "Private Person"},
            assistant_raw="Earlier response",
        )
        raw_messages = [
            {"role": "system", "content": "File for Private Person"},
            {"role": "assistant", "content": artifact},
            {"role": "user", "content": "Continue"},
        ]

        masked_messages, submitted_values = self._run_masking(raw_messages)

        self.assertEqual(submitted_values, ["Continue"])
        self.assertEqual(masked_messages[0]["content"], "File for <PER>person-1</PER>")
        self.assertTrue(masked_messages[0]["_protecto_artifact_resolved"])

    def test_masks_live_instruction_again_when_document_context_changes(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {"role": "system", "content": "File for <PER>person-1</PER>"},
                {"role": "user", "content": "Earlier prompt"},
            ],
            token_map={"person-1": "Private Person"},
            assistant_raw="Earlier response",
        )
        raw_messages = [
            {"role": "system", "content": "Different uploaded document"},
            {"role": "assistant", "content": artifact},
            {"role": "user", "content": "Continue"},
        ]

        _masked_messages, submitted_values = self._run_masking(raw_messages)

        self.assertEqual(submitted_values, [
            "Different uploaded document",
            "Continue",
        ])

    def test_masks_structured_tool_result_as_json(self):
        raw_messages = [{
            "role": "user",
            "content": "Lookup account",
        }, {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"owner": "Private Person"},
                },
            }],
        }, {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "lookup",
            "content": {"account": "Private Account"},
        }]

        masked_messages, submitted_values = self._run_masking(raw_messages)

        self.assertEqual(submitted_values, [
            "Lookup account",
            '{"owner": "Private Person"}',
            '{"account":"Private Account"}',
        ])
        tool_message = masked_messages[-1]
        self.assertEqual(tool_message["content"], "<MASKED>2</MASKED>")

    def test_rejects_unknown_non_text_user_content(self):
        raw_messages = [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.test/private.png"},
            }],
        }]

        with self.assertRaises(HTTPException) as raised:
            self._run_masking(raw_messages)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("Unsupported non-text content", raised.exception.detail)

    # ---- replay anchor when no artifact exists (skill/tool loop) ----

    @staticmethod
    def _tool_call(call_id, arguments='{"skillName":"pii-reviewer"}'):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "skill", "arguments": arguments},
        }

    def test_replays_full_tool_loop_when_no_artifact_exists(self):
        """
        A tool-call turn carries no artifact, so the artifact-less fallback
        decides the replay window. It must keep the whole open turn: anchoring
        on the trailing injected user message drops every tool call and result,
        and the model then repeats the same call forever.
        """
        raw_messages = [
            {"role": "user", "content": "Check this text for PII"},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tool_call("call-1")]},
            {"role": "tool", "tool_call_id": "call-1", "name": "skill",
             "content": "skill loaded"},
            {"role": "user", "content": "Skill instructions"},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tool_call("call-2")]},
            {"role": "tool", "tool_call_id": "call-2", "name": "skill",
             "content": "skill loaded"},
            {"role": "user", "content": "Skill instructions"},
        ]

        masked_messages, submitted = self._run_masking(raw_messages)

        self.assertEqual(submitted, [
            "Check this text for PII",
            '{"skillName":"pii-reviewer"}',
            "skill loaded",
            "Skill instructions",
            '{"skillName":"pii-reviewer"}',
            "skill loaded",
            "Skill instructions",
        ])
        self.assertEqual(
            [message["role"] for message in masked_messages],
            ["user", "assistant", "tool", "user", "assistant", "tool", "user"],
        )

    def test_artifactless_tool_loop_ignores_assistant_with_content_and_tool_calls(self):
        """An assistant that both narrates and calls a tool is mid-turn."""
        raw_messages = [
            {"role": "user", "content": "Check this text for PII"},
            {"role": "assistant", "content": "Let me load the skill.",
             "tool_calls": [self._tool_call("call-1")]},
            {"role": "tool", "tool_call_id": "call-1", "content": "skill loaded"},
            {"role": "user", "content": "Skill instructions"},
        ]

        _masked_messages, submitted = self._run_masking(raw_messages)

        self.assertIn("Check this text for PII", submitted)
        self.assertIn("skill loaded", submitted)

    def test_tool_loop_after_artifactless_text_turn_drops_the_earlier_turn(self):
        """An assistant TEXT message closes the previous turn."""
        raw_messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Plain reply without artifact"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tool_call("call-1")]},
            {"role": "tool", "tool_call_id": "call-1", "content": "skill loaded"},
        ]

        _masked_messages, submitted = self._run_masking(raw_messages)

        self.assertNotIn("First question", submitted)
        self.assertNotIn("Plain reply without artifact", submitted)
        self.assertEqual(submitted[0], "Second question")

    def test_assistant_without_content_or_tool_calls_is_not_a_turn_boundary(self):
        """An empty assistant turn is not evidence that a turn completed."""
        raw_messages = [
            {"role": "user", "content": "Check this text for PII"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tool_call("call-1")]},
            {"role": "tool", "tool_call_id": "call-1", "content": "skill loaded"},
            {"role": "user", "content": "Skill instructions"},
        ]

        _masked_messages, submitted = self._run_masking(raw_messages)

        self.assertEqual(submitted[0], "Check this text for PII")

    def test_drops_tool_result_without_matching_assistant_tool_call(self):
        """
        An unpaired tool result is rejected by providers. Dropping it must
        happen before mask_pending_messages, whose collect and rebuild passes
        share one cursor into the masked values.
        """
        raw_messages = [
            {"role": "user", "content": "Check this text for PII"},
            {"role": "tool", "tool_call_id": "call-unknown",
             "content": "orphaned result"},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tool_call("call-1")]},
            {"role": "tool", "tool_call_id": "call-1", "content": "skill loaded"},
            {"role": "user", "content": "Skill instructions"},
        ]

        masked_messages, submitted = self._run_masking(raw_messages)

        self.assertNotIn("orphaned result", submitted)
        self.assertEqual(
            ["call-1"],
            [m["tool_call_id"] for m in masked_messages if m["role"] == "tool"],
        )
        # The shared cursor stayed in lockstep: each value landed on its message.
        self.assertEqual(
            submitted.index("skill loaded"),
            int(
                next(
                    m["content"] for m in masked_messages
                    if m["role"] == "tool"
                ).removeprefix("<MASKED>").removesuffix("</MASKED>")
            ),
        )

    def test_conversation_starting_with_tool_message_is_not_replayed_as_orphan(self):
        raw_messages = [
            {"role": "tool", "tool_call_id": "call-stale", "content": "stale"},
            {"role": "user", "content": "Check this text for PII"},
        ]

        masked_messages, submitted = self._run_masking(raw_messages)

        self.assertEqual(submitted, ["Check this text for PII"])
        self.assertEqual([m["role"] for m in masked_messages], ["user"])


if __name__ == "__main__":
    unittest.main()
