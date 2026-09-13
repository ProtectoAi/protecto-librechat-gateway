import unittest
from protecto_gateway.history import (
    add_chat_name_prefix,
    build_model_catalog,
    remove_chat_name_prefix,
)


class ModelIdTests(unittest.TestCase):
    def test_adds_chat_name_to_model_discovery_id(self):
        self.assertEqual(
            add_chat_name_prefix("OpenAI:gpt-5.6", "Protecto-Chats"),
            "Protecto-Chats-OpenAI:gpt-5.6",
        )

    def test_removes_matching_chat_name_case_insensitively(self):
        self.assertEqual(
            remove_chat_name_prefix(
                "protecto-chats-OpenAI:gpt-5.6",
                "Protecto-Chats",
            ),
            "OpenAI:gpt-5.6",
        )

    def test_does_not_remove_a_different_prefix(self):
        self.assertEqual(
            remove_chat_name_prefix(
                "Other-OpenAI:gpt-5.6",
                "Protecto-Chats",
            ),
            "Other-OpenAI:gpt-5.6",
        )

    def test_models_endpoint_prefixes_ids_from_header(self):
        result = build_model_catalog(
            ["OpenAI:gpt-5.6", "Gemini:gemini-3.6-flash"],
            "Protecto-Chats",
            123,
        )

        self.assertEqual(
            [model["id"] for model in result["data"]],
            [
                "Protecto-Chats-OpenAI:gpt-5.6",
                "Protecto-Chats-Gemini:gemini-3.6-flash",
            ],
        )

    def test_models_endpoint_returns_clean_ids_without_header(self):
        result = build_model_catalog(["OpenAI:gpt-5.6"], None, 123)

        self.assertEqual(result["data"][0]["id"], "OpenAI:gpt-5.6")


if __name__ == "__main__":
    unittest.main()
