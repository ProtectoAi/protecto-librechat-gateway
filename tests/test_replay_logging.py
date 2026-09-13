import json
import logging
import unittest

from protecto_gateway.history import (
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


if __name__ == "__main__":
    unittest.main()
