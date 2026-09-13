import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from protecto_gateway.batch_mask import (
    mask_payload_in_background,
    scaled_polling_settings,
)
from protecto_gateway.config import ASYNC_MASK_THRESHOLD_BYTES
from protecto_gateway.protecto import mask_values_async


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def put(self, url, json, headers, **kwargs):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def _mask_result(token_value="<PER>masked</PER>"):
    return {
        "token_value": token_value,
        "individual_tokens": [{
            "token": "masked",
            "value": "Original Person",
        }],
    }


class BatchMaskTests(unittest.IsolatedAsyncioTestCase):
    def test_poll_seconds_and_retries_can_scale_independently(self):
        twenty_kib = 20 * 1024
        ten_kib = 10 * 1024

        self.assertEqual(
            scaled_polling_settings(
                twenty_kib, ten_kib, 10.0, 6, True, True,
            ),
            (20.0, 12),
        )
        self.assertEqual(
            scaled_polling_settings(
                twenty_kib, ten_kib, 10.0, 6, False, True,
            ),
            (10.0, 12),
        )
        self.assertEqual(
            scaled_polling_settings(
                twenty_kib, ten_kib, 10.0, 6, True, False,
            ),
            (20.0, 6),
        )

    async def test_small_payload_uses_realtime_mask_endpoint(self):
        client = _Client([_Response({"data": [_mask_result()]})])
        token_map = {}

        masked = await mask_values_async(
            client,
            ["Original Person"],
            "https://protecto.example/mask",
            {"Authorization": "Bearer test"},
            token_map,
        )

        self.assertEqual(masked, ["<PER>masked</PER>"])
        self.assertEqual(client.calls[0]["url"], "https://protecto.example/mask")
        self.assertEqual(token_map, {"masked": "Original Person"})

    @patch("protecto_gateway.batch_mask.asyncio.sleep", new_callable=AsyncMock)
    async def test_payload_over_10_kib_uses_async_api_and_extracts_result(
        self,
        sleep_mock,
    ):
        tracking_id = "tracking-1"
        client = _Client([
            _Response({
                "data": [{"tracking_id": tracking_id, "status": "PENDING"}],
                "success": True,
                "error": {"message": ""},
            }),
            _Response({
                "data": [{
                    "tracking_id": tracking_id,
                    "status": "IN-PROGRESS",
                }],
                "success": True,
                "error": {"message": ""},
            }),
            _Response({
                "data": [{
                    "tracking_id": tracking_id,
                    "status": "SUCCESS",
                    "result": [_mask_result()],
                }],
                "success": True,
                "error": {"message": ""},
            }),
        ])
        progress = AsyncMock()
        token_map = {}

        masked = await mask_values_async(
            client,
            ["x" * ASYNC_MASK_THRESHOLD_BYTES],
            "https://protecto.example/mask",
            {},
            token_map,
            progress,
        )

        self.assertEqual(masked, ["<PER>masked</PER>"])
        self.assertEqual(
            [call["url"] for call in client.calls],
            [
                "https://protecto.example/mask/async",
                "https://protecto.example/async-status",
                "https://protecto.example/async-status",
            ],
        )
        self.assertEqual(client.calls[1]["json"], {
            "status": [{"tracking_id": tracking_id}],
        })
        self.assertEqual(progress.await_count, 3)
        progress_messages = [
            call.args[0] for call in progress.await_args_list
        ]
        self.assertEqual(
            progress_messages[0],
            "Context masking is 0% complete...",
        )
        self.assertRegex(
            progress_messages[1],
            r"^Context masking is \d+% complete\.\.\.$",
        )
        self.assertEqual(
            progress_messages[-1],
            "Context masking is 100% complete.",
        )
        self.assertEqual(sleep_mock.await_count, 2)
        self.assertEqual(token_map, {"masked": "Original Person"})

    @patch("protecto_gateway.batch_mask.asyncio.sleep", new_callable=AsyncMock)
    async def test_pending_status_reports_percentage_from_poll_budget(
        self,
        sleep_mock,
    ):
        pending = {
            "data": [{"tracking_id": "tracking-1", "status": "PENDING"}],
            "success": True,
            "error": {"message": ""},
        }
        success = {
            "data": [{
                "tracking_id": "tracking-1",
                "status": "SUCCESS",
                "result": [_mask_result()],
            }],
            "success": True,
            "error": {"message": ""},
        }
        client = _Client([
            _Response(pending),
            _Response(pending),
            _Response(success),
        ])
        progress = AsyncMock()

        await mask_payload_in_background(
            client,
            {"mask": [{"value": "large content"}]},
            "https://protecto.example/mask",
            {},
            progress_callback=progress,
            poll_seconds=10,
            max_polls=6,
        )

        self.assertEqual(
            [call.args[0] for call in progress.await_args_list],
            [
                "Context masking is 0% complete...",
                "Context masking is 16% complete...",
                "Context masking is 100% complete.",
            ],
        )
        self.assertEqual(sleep_mock.await_count, 2)

    @patch("protecto_gateway.batch_mask.asyncio.sleep", new_callable=AsyncMock)
    async def test_failed_job_stops_polling(self, sleep_mock):
        client = _Client([
            _Response({
                "data": [{"tracking_id": "tracking-1", "status": "PENDING"}],
                "success": True,
                "error": {"message": ""},
            }),
            _Response({
                "data": [{
                    "tracking_id": "tracking-1",
                    "status": "FAILED",
                    "error_msg": "Masking rejected",
                }],
                "success": True,
                "error": {"message": ""},
            }),
        ])

        with self.assertRaises(HTTPException) as raised:
            await mask_payload_in_background(
                client,
                {"mask": [{"value": "large content"}]},
                "https://protecto.example/mask",
                {},
                poll_seconds=10,
                max_polls=6,
            )

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail, "Masking rejected")
        self.assertEqual(sleep_mock.await_count, 1)

    @patch("protecto_gateway.batch_mask.asyncio.sleep", new_callable=AsyncMock)
    async def test_active_job_fails_after_six_polls(self, sleep_mock):
        pending = {
            "data": [{"tracking_id": "tracking-1", "status": "PENDING"}],
            "success": True,
            "error": {"message": ""},
        }
        client = _Client([_Response(pending)] + [_Response(pending) for _ in range(6)])

        with self.assertRaises(HTTPException) as raised:
            await mask_payload_in_background(
                client,
                {"mask": [{"value": "large content"}]},
                "https://protecto.example/mask",
                {},
                poll_seconds=10,
                max_polls=6,
            )

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(sleep_mock.await_count, 6)
        self.assertEqual(len(client.calls), 7)


if __name__ == "__main__":
    unittest.main()
