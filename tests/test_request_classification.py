import unittest

from protecto_gateway.history import is_title_generation_request


class RequestClassificationTests(unittest.TestCase):
    def test_explicit_stream_false_is_title_generation(self):
        self.assertTrue(is_title_generation_request({"stream": False}))

    def test_streaming_request_is_not_title_generation(self):
        self.assertFalse(is_title_generation_request({"stream": True}))

    def test_missing_stream_is_not_title_generation(self):
        self.assertFalse(is_title_generation_request({}))

    def test_non_boolean_false_values_are_not_title_generation(self):
        self.assertFalse(is_title_generation_request({"stream": 0}))
        self.assertFalse(is_title_generation_request({"stream": None}))


if __name__ == "__main__":
    unittest.main()
