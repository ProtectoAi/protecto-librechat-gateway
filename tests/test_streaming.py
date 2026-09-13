import asyncio
import json
import unittest
from unittest.mock import patch

from protecto_gateway.protecto import decode_json_unicode_escapes
from protecto_gateway.sse import masking_progress_chunk
from protecto_gateway.streaming import (
    _entity_safe_split_async,
    _masked_entity_value,
    _replace_first_masked_entity,
    stream_and_unmask_generator,
)


class StreamingTests(unittest.TestCase):
    def test_malformed_closing_tag_does_not_block_following_entities(self):
        async def source():
            yield (
                "CVV <CVV>209</CV>. Passport "
                "<PASSPORT_NO>WngTo</PASSPORT_NO>."
            )

        async def collect():
            return [item async for item in _entity_safe_split_async(source())]

        pieces = asyncio.run(collect())
        entity_pieces = [piece for piece, has_entity in pieces if has_entity]

        self.assertEqual(len(entity_pieces), 2)
        self.assertEqual(_masked_entity_value(entity_pieces[0]), "209")
        self.assertEqual(_masked_entity_value(entity_pieces[1]), "WngTo")

    def test_completed_text_turn_streams_answer_and_artifact(self):
        async def fake_core_stream(**_kwargs):
            yield {"kind": "text", "text": "Visible response"}
            yield {
                "kind": "end",
                "artifact": "\n:::artifact{identifier=\"behind-the-scenes\"}\n:::",
            }

        async def collect_chunks():
            with patch(
                "protecto_gateway.streaming.core_unmasked_stream",
                fake_core_stream,
            ):
                return [chunk async for chunk in stream_and_unmask_generator(
                    provider="openai",
                    model="test",
                    model_label="Secured-Chat-OpenAI:test",
                    masked_messages=[],
                    protecto_unmask_url="https://protecto.example.com/unmask",
                    headers={},
                    provider_api_key="test-key",
                    token_map={},
                    response_id="chatcmpl-test",
                )]

        chunks = asyncio.run(collect_chunks())
        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk.startswith("data: {")
        ]

        self.assertEqual(
            payloads[0]["choices"][0]["delta"],
            {"content": "Visible response"},
        )
        self.assertIsNone(payloads[0]["choices"][0]["finish_reason"])
        self.assertIn(
            ":::artifact",
            payloads[-1]["choices"][0]["delta"]["content"],
        )
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "stop")

    def test_async_mask_progress_is_sse_comment_when_tools_are_enabled(self):
        chunk = masking_progress_chunk(
            message="Context masking is 16% complete...",
            tools_enabled=True,
            response_id="chatcmpl-test",
            model="openai:test",
            created=123,
        )

        self.assertTrue(chunk.startswith(": protecto-progress "))
        self.assertNotIn("data:", chunk)
        self.assertNotIn('"content"', chunk)

    def test_async_mask_progress_remains_visible_without_tools(self):
        chunk = masking_progress_chunk(
            message="Context masking is 16% complete...",
            tools_enabled=False,
            response_id="chatcmpl-test",
            model="openai:test",
            created=123,
        )

        self.assertTrue(chunk.startswith("data: "))
        self.assertIn('"content"', chunk)
        self.assertIn("Context masking is 16% complete", chunk)

    def test_original_value_backslashes_are_inserted_literally(self):
        original_value = r'{"path":"C:\users","backref":"\1"}'

        result = _replace_first_masked_entity(
            "before <PER>masked-value</PER> after",
            original_value,
        )

        self.assertEqual(
            result,
            f"before {original_value} after",
        )

    def test_json_unicode_escape_is_decoded_for_display(self):
        result = _replace_first_masked_entity(
            "<PER>masked-value</PER>",
            r"Baskaran\u0027s workspace",
        )

        self.assertEqual(result, "Baskaran's workspace")

    def test_json_surrogate_pair_is_decoded(self):
        self.assertEqual(
            decode_json_unicode_escapes(r"Meeting \ud83d\udcdd"),
            "Meeting 📝",
        )


if __name__ == "__main__":
    unittest.main()
