import unittest

from protecto_gateway.config import validate_protecto_settings


class ProtectoConfigTests(unittest.TestCase):
    def test_normalizes_valid_gateway_settings(self):
        settings = validate_protecto_settings(
            " https://protecto.example.com/ ",
            " master-token ",
            " librechat_namespace_vault ",
        )

        self.assertEqual(
            settings,
            (
                "https://protecto.example.com",
                "master-token",
                "librechat_namespace_vault",
            ),
        )

    def test_rejects_missing_gateway_settings(self):
        with self.assertRaisesRegex(ValueError, "configuration is incomplete"):
            validate_protecto_settings("", "", "")

    def test_rejects_non_http_service_url(self):
        with self.assertRaisesRegex(ValueError, r"HTTP\(S\) service URL"):
            validate_protecto_settings(
                "file:///tmp/protecto",
                "master-token",
                "librechat_namespace_vault",
            )

    def test_rejects_url_query_or_fragment(self):
        for url in (
            "https://protecto.example.com?token=unsafe",
            "https://protecto.example.com#fragment",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, r"HTTP\(S\) service URL"):
                    validate_protecto_settings(
                        url,
                        "master-token",
                        "librechat_namespace_vault",
                    )

    def test_rejects_invalid_namespace(self):
        with self.assertRaisesRegex(ValueError, "PROTECTO_NAMESPACE is invalid"):
            validate_protecto_settings(
                "https://protecto.example.com",
                "master-token",
                "invalid\nnamespace",
            )


if __name__ == "__main__":
    unittest.main()
