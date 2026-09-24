import asyncio
import json
import unittest
from unittest.mock import patch

from protecto_gateway.config import ENTITIES, END_TAG
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

    def test_entity_not_in_known_list_does_not_swallow_the_next_tag(self):
        # Regression test: if an entity tag isn't in config.ENTITIES (and so
        # isn't in START_TAGS), the splitter doesn't recognize it as its own
        # boundary and bundles it together with the next *known* tag into a
        # single "has_entity" piece. core_unmasked_stream's fast path then
        # extracts only the FIRST tag/value pair in that piece and replaces
        # only it (_replace_first_masked_entity uses count=1), leaving any
        # later tag in the same piece as raw, still-masked text delivered to
        # the user untouched. This previously happened for real with STATE,
        # which was missing from ENTITIES: "<STATE>qmD</STATE>
        # <PINCODE>66781</PINCODE>" rendered to the user as
        # "NF <PINCODE>66781</PINCODE>" instead of "NF 30021".
        async def source():
            yield "<STATE>qmD</STATE> <PINCODE>66781</PINCODE>"
            yield END_TAG

        async def collect():
            return [item async for item in _entity_safe_split_async(source())]

        pieces = asyncio.run(collect())
        entity_pieces = [piece for piece, has_entity in pieces if has_entity]

        # STATE and PINCODE must be split into separate pieces so each one
        # is routed through unmasking on its own.
        self.assertEqual(len(entity_pieces), 2)
        self.assertEqual(_masked_entity_value(entity_pieces[0]), "qmD")
        self.assertEqual(_masked_entity_value(entity_pieces[1]), "66781")

    def test_entities_list_covers_all_tags_seen_in_gateway_logs(self):
        # Guards against ENTITIES silently drifting out of sync with what
        # Protecto actually returns for this namespace -- any tag it emits
        # that isn't in ENTITIES risks the swallowing bug above.
        observed_tags = {
            "AADHAAR", "ACC_NO", "ADDRESS", "ADMIT_DATE", "BLOOD_TYPE",
            "CANADIAN_SIN", "CITY", "COUNTRY", "CRD", "CRD_EXP_DATE",
            "CRD_PIN", "CVV", "DEVICE_ID", "DISCHARGE_DATE", "DL_NO", "DOB",
            "EMAIL", "FAX_NO", "GSTIN", "HEALTH_BENEFICIARY_NO",
            "INSURANCE_NO", "IPA", "MEDICAL_CODE", "MEDICAL_CONDITION",
            "MEDICAL_PROCEDURE", "MEDICATION", "MRN", "NATIONALITY",
            "NATIONAL_ID", "ORG", "PAN", "PASSPORT_NO", "PASSWORD", "PER",
            "PHN", "PH_TIN", "PINCODE", "POLICY_NO", "ROUTING_NO", "SSN",
            "STATE", "SWIFT", "TAN", "UK_NIN", "UK_UTR", "URL", "US_ITIN",
            "VEHICLE_REG_NO", "VIN",
        }
        self.assertEqual(observed_tags - set(ENTITIES), set())

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
