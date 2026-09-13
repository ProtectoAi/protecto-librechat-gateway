import unittest

from protecto_gateway.config import parse_gateway_models


class ModelConfigTests(unittest.TestCase):
    def test_builds_provider_model_ids_from_two_level_environment(self):
        models = parse_gateway_models(
            '{"OpenAI":["gpt-5.6","gpt-5.5"],'
            '"Gemini":["gemini-3.6-flash"]}',
        )

        self.assertEqual(
            models,
            [
                "OpenAI:gpt-5.6",
                "OpenAI:gpt-5.5",
                "Gemini:gemini-3.6-flash",
            ],
        )

    def test_requires_models_for_each_provider(self):
        with self.assertRaisesRegex(ValueError, "non-empty JSON array"):
            parse_gateway_models('{"OpenAI":[]}')

    def test_removes_duplicate_provider_model_ids(self):
        models = parse_gateway_models(
            '{"OpenAI":["gpt-5.6","gpt-5.6"]}',
        )

        self.assertEqual(models, ["OpenAI:gpt-5.6"])

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "must be valid JSON"):
            parse_gateway_models('{"OpenAI":["gpt-5.6"]')


if __name__ == "__main__":
    unittest.main()
