import base64
import json
import logging
import unittest

from protecto_gateway.artifacts import (
    build_behind_scenes_artifact,
    extract_behind_scenes_data,
)
from protecto_gateway.history import build_masked_history


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class ArtifactTests(unittest.TestCase):
    def test_no_artifact_replay_keeps_only_latest_user_turn(self):
        old_file_content = [{
            "type": "input_file",
            "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
        }]
        raw_messages = [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": old_file_content},
            {"role": "assistant", "content": "Upload is not supported"},
            {"role": "user", "content": "Continue without the file"},
        ]

        masked_messages, token_map, pending = build_masked_history(raw_messages)

        self.assertEqual(masked_messages, [{
            "role": "system",
            "content": "System instructions",
        }])
        self.assertEqual(token_map, {})
        self.assertEqual(pending, [{
            "role": "user",
            "content": "Continue without the file",
        }])

    def test_builds_artifact_with_complete_replay_payload(self):
        masked_messages = [{
            "role": "user",
            "content": "Email <EMAIL>person-1</EMAIL>",
        }]
        token_map = {"person-1": "private@example.com"}
        assistant_raw = "Hello <EMAIL>person-1</EMAIL>"

        artifact = build_behind_scenes_artifact(
            masked_messages=masked_messages,
            token_map=token_map,
            assistant_raw=assistant_raw,
        )
        self.assertIn('identifier="behind-the-scenes"', artifact)
        self.assertIsNotNone(extract_behind_scenes_data(artifact))

    def test_persists_masked_instructions_and_rebuilds_them_as_fallback(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {
                    "role": "system",
                    "content": "Gateway policy",
                    "_protecto_gateway_owned": True,
                },
                {"role": "system", "content": "File for <PER>person-1</PER>"},
                {"role": "developer", "content": "Protect <PER>person-1</PER>"},
                {"role": "user", "content": "Summarize it"},
            ],
            token_map={"person-1": "Private Person"},
            assistant_raw="Summary",
        )

        data = extract_behind_scenes_data(artifact)
        self.assertEqual(data["instructions"], [
            {"role": "system", "content": "File for <PER>person-1</PER>"},
            {"role": "developer", "content": "Protect <PER>person-1</PER>"},
        ])
        self.assertNotIn("Gateway policy", json.dumps(data))
        self.assertIn("<li><code>Private Person</code></li>", artifact)
        self.assertNotIn("File for &lt;PER&gt;person-1", artifact)

        masked_messages, token_map, pending = build_masked_history([
            {"role": "assistant", "content": artifact},
            {"role": "user", "content": "Continue"},
        ])
        self.assertEqual(
            [message["content"] for message in masked_messages[:2]],
            ["File for <PER>person-1</PER>", "Protect <PER>person-1</PER>"],
        )
        self.assertTrue(masked_messages[0]["_protecto_artifact_resolved"])
        self.assertEqual(token_map, {"person-1": "Private Person"})
        self.assertEqual(pending, [{"role": "user", "content": "Continue"}])

    def test_logs_decoding_and_exact_masked_rebuilt_history(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {"role": "user", "content": "Hello <PER>person-1</PER>"},
            ],
            token_map={"<PER>person-1</PER>": "Private Person"},
            assistant_raw="Welcome <PER>person-1</PER>",
        )
        capture = _Capture()
        logger = logging.getLogger("protecto_gateway")
        logger.addHandler(capture)
        try:
            build_masked_history([{"role": "assistant", "content": artifact}])
        finally:
            logger.removeHandler(capture)

        output = "\n".join(capture.records)
        self.assertIn(
            "[ARTIFACT DATA DECODE] source=metadata_attribute success=true",
            output,
        )
        self.assertIn("[ARTIFACT HISTORY REBUILT] artifacts=1 messages=2", output)
        self.assertIn(
            '[ARTIFACT HISTORY REBUILT PAYLOAD] '
            '[{"role":"user","content":"Hello <PER>person-1</PER>"},'
            '{"role":"assistant","content":"Welcome <PER>person-1</PER>"}]',
            output,
        )
        self.assertNotIn("Private Person", output)

    def test_renders_readable_sections_with_only_original_values(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {"role": "user", "content": "Email <EMAIL>person-1</EMAIL>"},
            ],
            token_map={"person-1": "private@example.com"},
            assistant_raw="Hello <EMAIL>person-1</EMAIL>",
        )

        self.assertIn("type=\"text/html\"", artifact)
        self.assertIn("}\n````\n<!DOCTYPE html>", artifact)
        self.assertIn("<h2><strong>👤 USER PROMPT</strong></h2>", artifact)
        self.assertIn("<h2><strong>🤖 AI RESPONSE</strong></h2>", artifact)
        self.assertIn(
            "<h2>🔐 SENSITIVE INFORMATION IDENTIFIED</h2>",
            artifact,
        )
        self.assertNotIn("ORIGINAL VALUES", artifact)
        self.assertNotIn("MASKED TOKENS", artifact)
        self.assertIn(
            "background: #f9fafb; font-family: system-ui, sans-serif;",
            artifact,
        )
        self.assertIn("font-size: 12px; margin: 0; padding: 24px;", artifact)
        self.assertIn("code { font-family: system-ui, sans-serif; }", artifact)
        self.assertNotIn("ui-monospace", artifact)
        self.assertIn("\n````\n:::", artifact)
        self.assertNotIn("\u2028", artifact)
        self.assertNotIn("\u2029", artifact)
        self.assertIn("Email &lt;EMAIL&gt;person-1&lt;/EMAIL&gt;", artifact)
        self.assertIn("Hello &lt;EMAIL&gt;person-1&lt;/EMAIL&gt;", artifact)
        self.assertIn("<pre>", artifact)
        self.assertNotIn("<br>", artifact)
        self.assertIn(
            "<li><code>private@example.com</code></li>",
            artifact,
        )

    def test_renders_masked_retrieved_file_context(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[
                {"role": "user", "content": "Who owns the account?"},
                {
                    "role": "tool",
                    "name": "file_search",
                    "content": (
                        "The owner is <PERSON>person-1</PERSON> with account "
                        "<ACCOUNT>account-1</ACCOUNT>."
                    ),
                },
            ],
            token_map={
                "person-1": "Alice Smith",
                "account-1": "12345678",
            },
            assistant_raw="The owner is <PERSON>person-1</PERSON>.",
        )

        self.assertIn(
            "📄 RETRIEVED FILE CONTEXT SENT TO LLM",
            artifact,
        )
        self.assertIn("<h3>file_search</h3>", artifact)
        self.assertIn(
            "The owner is &lt;PERSON&gt;person-1&lt;/PERSON&gt; with account "
            "&lt;ACCOUNT&gt;account-1&lt;/ACCOUNT&gt;.",
            artifact,
        )
        self.assertNotIn("The owner is Alice Smith", artifact)
        self.assertIn("<li><code>Alice Smith</code></li>", artifact)
        self.assertIn("<li><code>12345678</code></li>", artifact)

    def test_omits_retrieved_context_section_without_tool_results(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{"role": "user", "content": "Hello"}],
            token_map={},
            assistant_raw="Hi",
        )

        self.assertNotIn("RETRIEVED FILE CONTEXT SENT TO LLM", artifact)

    def test_escapes_original_values_and_omits_values_not_in_current_turn(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{
                "role": "user",
                "content": "Email <EMAIL>person-1</EMAIL>",
            }],
            token_map={
                "person-1": "<private@example.com>",
                "old-person": "Previous Person",
            },
            assistant_raw="Done",
        )

        self.assertIn(
            "<li><code>&lt;private@example.com&gt;</code></li>",
            artifact,
        )
        self.assertNotIn("<li><code>Previous Person</code></li>", artifact)

    def test_supports_legacy_token_map_with_complete_tag_keys(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{
                "role": "user",
                "content": "Hello <PER>person-1</PER>",
            }],
            token_map={"<PER>person-1</PER>": "Private Person"},
            assistant_raw="Hello",
        )

        self.assertIn(
            "<li><code>Private Person</code></li>",
            artifact,
        )

    def test_preserves_ai_response_paragraphs_and_blank_lines(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{"role": "user", "content": "make it sarcastic"}],
            token_map={},
            assistant_raw="Subject: Delay\n\nDear <PER>person-1</PER>,\n\nSincerely,\nSupport",
        )

        self.assertIn(
            "Subject: Delay\n\nDear &lt;PER&gt;person-1&lt;/PER&gt;,\n\n"
            "Sincerely,\nSupport</pre>",
            artifact,
        )

    def test_preserves_backticks_and_neutralizes_non_masking_angle_brackets(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{
                "role": "user",
                "content": "Use ``` and <unknown> in the reply",
            }],
            token_map={},
            assistant_raw="Done",
        )

        self.assertIn("}\n````\n<!DOCTYPE html>", artifact)
        self.assertIn("Use ``` and &lt;unknown&gt; in the reply", artifact)

    def test_uses_fence_longer_than_four_backticks_in_content(self):
        artifact = build_behind_scenes_artifact(
            masked_messages=[{"role": "user", "content": "Use ```` exactly"}],
            token_map={},
            assistant_raw="Done",
        )

        self.assertIn("}\n`````\n<!DOCTYPE html>", artifact)
        self.assertIn("Use ```` exactly", artifact)
        self.assertIn("\n`````\n:::", artifact)

    def test_extracts_complete_data_from_readable_artifact(self):
        token_map = {"<PER>person-1</PER>": "Private Person"}
        artifact = build_behind_scenes_artifact(
            masked_messages=[{"role": "user", "content": "Hi <PER>person-1</PER>"}],
            token_map=token_map,
            assistant_raw="Hello <PER>person-1</PER>",
        )

        self.assertEqual(
            extract_behind_scenes_data(artifact),
            {
                "user": "Hi <PER>person-1</PER>",
                "assistant": "Hello <PER>person-1</PER>",
                "token_map": token_map,
            },
        )

    def test_extracts_legacy_json_artifact(self):
        data = {
            "user": "legacy user",
            "assistant": "legacy assistant",
            "token_map": {"token": "value"},
        }
        artifact = (
            ':::artifact{identifier="behind-the-scenes" '
            'type="text/markdown" title="Behind the Scene"}\n'
            f"{json.dumps(data)}\n"
            ":::"
        )

        self.assertEqual(extract_behind_scenes_data(artifact), data)

    def test_extracts_previous_markdown_artifact_metadata(self):
        data = {
            "user": "previous user",
            "assistant": "previous assistant",
            "token_map": {"<PER>person-1</PER>": "Private Person"},
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(data, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        artifact = (
            ':::artifact{identifier="behind-the-scenes" '
            'type="text/markdown" title="Behind the Scene"}\n'
            "Previous readable content\n\n"
            f"<!--protecto-replay-data:{encoded}-->\n"
            ":::"
        )

        self.assertEqual(extract_behind_scenes_data(artifact), data)


if __name__ == "__main__":
    unittest.main()
