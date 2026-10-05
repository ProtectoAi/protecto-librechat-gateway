import json
import logging
import unittest

from protecto_gateway.history import (
    build_masked_history,
    log_librechat_replay,
    log_masked_replay_payload,
)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class ReplayLoggingTests(unittest.TestCase):
    def test_logs_complete_masked_replay_payload(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Contact <EMAIL>masked@example.com</EMAIL>"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_customer",
                        "arguments": '{"email":"<EMAIL>masked@example.com</EMAIL>"}',
                    },
                }],
            },
        ]

        capture = _Capture()
        logger = logging.getLogger("protecto_gateway")
        logger.addHandler(capture)
        try:
            log_masked_replay_payload(messages)
        finally:
            logger.removeHandler(capture)

        expected = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(
            capture.records[-1],
            f"[LIBRECHAT MASKED REPLAY PAYLOAD] {expected}",
        )

    def test_logs_shape_without_replayed_values(self):
        secrets = {
            "prompt": "customer secret prompt",
            "arguments": json.dumps({"email": "private@example.com"}),
            "tool_result": "private tool result",
            "call_id": "secret-call-id",
        }
        messages = [
            {"role": "user", "content": secrets["prompt"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": secrets["call_id"],
                    "function": {
                        "name": "lookup_customer",
                        "arguments": secrets["arguments"],
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": secrets["call_id"],
                "content": secrets["tool_result"],
            },
        ]

        capture = _Capture()
        logger = logging.getLogger("protecto_gateway")
        logger.addHandler(capture)
        try:
            log_librechat_replay(messages)
        finally:
            logger.removeHandler(capture)

        line = capture.records[-1]
        for secret in secrets.values():
            self.assertNotIn(secret, line)
        self.assertIn('"role":"user"', line)
        self.assertIn('"tool_calls":1', line)
        self.assertIn(f'"argument_chars":[{len(secrets["arguments"])}]', line)

    def test_does_not_log_untrusted_role_or_non_string_content(self):
        unsafe_role = "user\nFORGED LOG LINE"
        capture = _Capture()
        logger = logging.getLogger("protecto_gateway")
        logger.addHandler(capture)
        try:
            log_librechat_replay([{"role": unsafe_role, "content": {"secret": 1}}])
        finally:
            logger.removeHandler(capture)

        line = capture.records[-1]
        self.assertNotIn(unsafe_role, line)
        self.assertIn('"role":"unknown"', line)
        self.assertIn('"content_type":"dict"', line)
        self.assertIn('"content_chars":0', line)

    def _capture_build(self, raw_messages):
        from protecto_gateway.config import logger

        capture = _Capture()
        logger.addHandler(capture)
        try:
            build_masked_history(raw_messages)
        finally:
            logger.removeHandler(capture)
        return capture.records

    def test_logs_replay_anchor_decision(self):
        records = self._capture_build([
            {"role": "user", "content": "Check this text"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "skill", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "loaded"},
            {"role": "user", "content": "Injected instructions"},
        ])

        line = next(r for r in records if r.startswith("[REPLAY ANCHOR]"))
        self.assertIn("source=artifactless-scan", line)
        self.assertIn("raw_tool_results=1", line)
        self.assertIn("pending_tool_results=1", line)
        self.assertFalse(
            [r for r in records if "[REPLAY ANCHOR DROPPED TOOL CHAIN]" in r]
        )

    def test_warns_when_anchor_drops_every_tool_result(self):
        """
        A completed assistant text turn closes the loop, so the earlier tool
        results fall outside the replay window. That is correct here, but it is
        the exact shape that broke the skill loop, so it must be visible.
        """
        records = self._capture_build([
            {"role": "user", "content": "Check this text"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "skill", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "loaded"},
            {"role": "assistant", "content": "Here is the answer"},
            {"role": "user", "content": "Next question"},
        ])

        line = next(
            r for r in records if r.startswith("[REPLAY ANCHOR DROPPED TOOL CHAIN]")
        )
        self.assertIn("raw_tool_results=1", line)


if __name__ == "__main__":
    unittest.main()
