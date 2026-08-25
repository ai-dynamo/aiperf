# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.ui_dispatch.

Focuses on:
- print_progress_message dispatches correctly for each message type
- print_realtime_metrics filters and formats key metrics
- stream_progress wires up WebSocket streaming with correct parameters
- WS_MESSAGE_TYPES and constants are correct
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import MessageType
from aiperf.kubernetes.ui_dispatch import (
    API_WS_PATH,
    WS_MAX_RETRIES,
    WS_MESSAGE_TYPES,
    print_progress_message,
    print_realtime_metrics,
    stream_progress,
)

# Lazy imports in ui_dispatch pull from these modules, so we patch at the source.
_CONSOLE_LOGGER = "aiperf.kubernetes.console.logger"
_CONSOLE_PRINT_STEP = "aiperf.kubernetes.console.print_step"
_STREAM_FROM_API = "aiperf.kubernetes.port_forward.stream_progress_from_api"


# ============================================================
# Constants
# ============================================================


class TestConstants:
    """Verify module-level constants."""

    def test_ws_message_types_contains_expected_types(self) -> None:
        expected = {
            MessageType.CREDIT_PHASE_START,
            MessageType.CREDIT_PHASE_PROGRESS,
            MessageType.CREDIT_PHASE_COMPLETE,
            MessageType.REALTIME_METRICS,
            MessageType.WORKER_STATUS_SUMMARY,
            MessageType.ALL_RECORDS_RECEIVED,
        }
        assert set(WS_MESSAGE_TYPES) == expected

    def test_ws_message_types_length(self) -> None:
        assert len(WS_MESSAGE_TYPES) == 6

    def test_ws_max_retries_value(self) -> None:
        assert WS_MAX_RETRIES == 10

    def test_api_ws_path_value(self) -> None:
        assert API_WS_PATH == "/ws"


# ============================================================
# print_progress_message
# ============================================================


class TestPrintProgressMessage:
    """Verify print_progress_message dispatches to logger for each message type."""

    @patch(_CONSOLE_LOGGER)
    def test_subscribed_message_does_not_log(self, mock_logger: MagicMock) -> None:
        print_progress_message({"message_type": "subscribed"})
        mock_logger.info.assert_not_called()

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_start_logs_phase_name(self, mock_logger: MagicMock) -> None:
        print_progress_message(
            {
                "message_type": MessageType.CREDIT_PHASE_START,
                "stats": {"phase": "warmup"},
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "Starting warmup phase" in logged

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_start_missing_phase_defaults_to_unknown(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message({"message_type": MessageType.CREDIT_PHASE_START})
        logged = mock_logger.info.call_args[0][0]
        assert "unknown" in logged

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_progress_logs_counts_and_percent(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message(
            {
                "message_type": MessageType.CREDIT_PHASE_PROGRESS,
                "stats": {
                    "phase": "benchmark",
                    "requests_completed": 50,
                    "total_expected_requests": 200,
                },
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "benchmark" in logged
        assert "50/200" in logged
        assert "25.0%" in logged

    @pytest.mark.parametrize(
        "stats_data,expected_fraction,expected_percent",
        [
            ({"requests_completed": 0, "total_expected_requests": 0}, "0/0", "0.0%"),
            ({"requests_completed": 100, "total_expected_requests": 100}, "100/100", "100.0%"),
            ({}, "0/0", "0.0%"),
        ],
    )  # fmt: skip
    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_progress_edge_cases(
        self,
        mock_logger: MagicMock,
        stats_data: dict,
        expected_fraction: str,
        expected_percent: str,
    ) -> None:
        print_progress_message(
            {
                "message_type": MessageType.CREDIT_PHASE_PROGRESS,
                "stats": {"phase": "test", **stats_data},
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert expected_fraction in logged
        assert expected_percent in logged

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_progress_missing_stats_key(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message(
            {
                "message_type": MessageType.CREDIT_PHASE_PROGRESS,
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "0/0" in logged
        assert "0.0%" in logged

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_complete_logs_phase_name(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message(
            {
                "message_type": MessageType.CREDIT_PHASE_COMPLETE,
                "stats": {"phase": "cooldown"},
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "Completed cooldown phase" in logged

    @patch(_CONSOLE_LOGGER)
    def test_credit_phase_complete_missing_phase_defaults_to_unknown(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message({"message_type": MessageType.CREDIT_PHASE_COMPLETE})
        logged = mock_logger.info.call_args[0][0]
        assert "unknown" in logged

    @patch("aiperf.kubernetes.ui_dispatch.print_realtime_metrics")
    def test_realtime_metrics_delegates_to_helper(self, mock_fn: MagicMock) -> None:
        data = {"message_type": MessageType.REALTIME_METRICS, "metrics": []}
        print_progress_message(data)
        mock_fn.assert_called_once_with(data)

    @patch(_CONSOLE_LOGGER)
    def test_worker_status_summary_logs_healthy_count(
        self, mock_logger: MagicMock
    ) -> None:
        print_progress_message(
            {
                "message_type": MessageType.WORKER_STATUS_SUMMARY,
                "workers": {
                    "w1": {"status": "HEALTHY"},
                    "w2": {"status": "HEALTHY"},
                    "w3": {"status": "ERROR"},
                },
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "2/3 healthy" in logged

    @patch(_CONSOLE_LOGGER)
    def test_worker_status_summary_empty_workers(self, mock_logger: MagicMock) -> None:
        print_progress_message(
            {
                "message_type": MessageType.WORKER_STATUS_SUMMARY,
                "workers": {},
            }
        )
        logged = mock_logger.info.call_args[0][0]
        assert "0/0 healthy" in logged

    @patch(_CONSOLE_LOGGER)
    def test_all_records_received_logs_complete(self, mock_logger: MagicMock) -> None:
        print_progress_message({"message_type": MessageType.ALL_RECORDS_RECEIVED})
        logged = mock_logger.info.call_args[0][0]
        assert "COMPLETE" in logged
        assert "All records received" in logged

    @patch(_CONSOLE_LOGGER)
    def test_unknown_message_type_does_not_log(self, mock_logger: MagicMock) -> None:
        print_progress_message({"message_type": "some_unknown_type"})
        mock_logger.info.assert_not_called()

    @patch(_CONSOLE_LOGGER)
    def test_empty_data_does_not_log(self, mock_logger: MagicMock) -> None:
        print_progress_message({})
        mock_logger.info.assert_not_called()


# ============================================================
# print_realtime_metrics
# ============================================================


class TestPrintRealtimeMetrics:
    """Verify print_realtime_metrics filters and formats metrics."""

    @patch(_CONSOLE_LOGGER)
    def test_filters_key_metrics_from_list(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {
                    "tag": "request_throughput",
                    "header": "Request Throughput",
                    "avg": 42.5,
                    "unit": " req/s",
                },
                {
                    "tag": "request_latency",
                    "header": "Request Latency",
                    "avg": 123.0,
                    "unit": "ms",
                },
                {
                    "tag": "irrelevant_metric",
                    "header": "Irrelevant",
                    "avg": 999.0,
                    "unit": "x",
                },
            ]
        }
        print_realtime_metrics(data)
        assert mock_logger.info.call_count == 2
        calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("request_throughput" in c and "42.50" in c for c in calls)
        assert any("request_latency" in c and "123.00" in c for c in calls)

    @patch(_CONSOLE_LOGGER)
    def test_shows_all_four_key_metrics(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {
                    "tag": "request_throughput",
                    "header": "Throughput",
                    "avg": 1.0,
                    "unit": "",
                },
                {"tag": "request_latency", "header": "Latency", "avg": 2.0, "unit": ""},
                {
                    "tag": "time_to_first_token",
                    "header": "TTFT",
                    "avg": 3.0,
                    "unit": "",
                },
                {
                    "tag": "output_token_throughput",
                    "header": "Token TPS",
                    "avg": 4.0,
                    "unit": "",
                },
                {"tag": "some_other_metric", "header": "Other", "avg": 5.0, "unit": ""},
            ]
        }
        print_realtime_metrics(data)
        assert mock_logger.info.call_count == 4

    @patch(_CONSOLE_LOGGER)
    def test_empty_metrics_list_logs_nothing(self, mock_logger: MagicMock) -> None:
        print_realtime_metrics({"metrics": []})
        mock_logger.info.assert_not_called()

    @patch(_CONSOLE_LOGGER)
    def test_missing_metrics_key_logs_nothing(self, mock_logger: MagicMock) -> None:
        print_realtime_metrics({})
        mock_logger.info.assert_not_called()

    @patch(_CONSOLE_LOGGER)
    def test_uses_unit_field(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {
                    "tag": "request_throughput",
                    "header": "Throughput",
                    "avg": 10.0,
                    "unit": " req/s",
                },
            ]
        }
        print_realtime_metrics(data)
        logged = mock_logger.info.call_args[0][0]
        assert " req/s" in logged

    @patch(_CONSOLE_LOGGER)
    def test_uses_avg_for_value(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {
                    "tag": "request_latency",
                    "header": "Latency",
                    "avg": 0.5,
                    "unit": "s",
                },
            ]
        }
        print_realtime_metrics(data)
        logged = mock_logger.info.call_args[0][0]
        assert "0.50 s" in logged

    @patch(_CONSOLE_LOGGER)
    def test_falls_back_to_current_when_no_avg(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {
                    "tag": "request_throughput",
                    "header": "Throughput",
                    "current": 7.0,
                    "unit": "",
                },
            ]
        }
        print_realtime_metrics(data)
        assert mock_logger.info.call_count == 1
        logged = mock_logger.info.call_args[0][0]
        assert "7.00" in logged

    @patch(_CONSOLE_LOGGER)
    def test_metric_with_missing_tag_is_skipped(self, mock_logger: MagicMock) -> None:
        data = {
            "metrics": [
                {"avg": 1.0, "unit": ""},
            ]
        }
        print_realtime_metrics(data)
        mock_logger.info.assert_not_called()


# ============================================================
# stream_progress
# ============================================================


class TestStreamProgress:
    """Verify stream_progress wires up WebSocket streaming correctly."""

    @pytest.mark.asyncio
    @patch(_STREAM_FROM_API, new_callable=AsyncMock)
    @patch(_CONSOLE_LOGGER)
    @patch(_CONSOLE_PRINT_STEP)
    async def test_calls_stream_api_with_correct_params(
        self,
        mock_print_step: MagicMock,
        mock_logger: MagicMock,
        mock_stream_api: AsyncMock,
    ) -> None:
        await stream_progress("ws://localhost:9090/ws")

        mock_print_step.assert_called_once_with("Streaming progress...")
        mock_stream_api.assert_awaited_once()
        _, kwargs = mock_stream_api.call_args
        assert mock_stream_api.call_args[0][0] == "ws://localhost:9090/ws"
        assert kwargs["message_types"] == WS_MESSAGE_TYPES
        assert kwargs["max_retries"] == WS_MAX_RETRIES

    @pytest.mark.asyncio
    @patch(_STREAM_FROM_API, new_callable=AsyncMock)
    @patch(_CONSOLE_LOGGER)
    @patch(_CONSOLE_PRINT_STEP)
    async def test_handle_message_returns_true_on_all_records_received(
        self,
        mock_print_step: MagicMock,
        mock_logger: MagicMock,
        mock_stream_api: AsyncMock,
    ) -> None:
        await stream_progress("ws://localhost:9090/ws")

        on_message = mock_stream_api.call_args[1]["on_message"]
        result = await on_message({"message_type": MessageType.ALL_RECORDS_RECEIVED})
        assert result is True

    @pytest.mark.asyncio
    @patch(_STREAM_FROM_API, new_callable=AsyncMock)
    @patch(_CONSOLE_LOGGER)
    @patch(_CONSOLE_PRINT_STEP)
    async def test_handle_message_returns_false_for_other_types(
        self,
        mock_print_step: MagicMock,
        mock_logger: MagicMock,
        mock_stream_api: AsyncMock,
    ) -> None:
        await stream_progress("ws://localhost:9090/ws")

        on_message = mock_stream_api.call_args[1]["on_message"]
        result = await on_message(
            {
                "message_type": MessageType.CREDIT_PHASE_START,
                "stats": {"phase": "warmup"},
            }
        )
        assert result is False

    @pytest.mark.asyncio
    @patch(_STREAM_FROM_API, new_callable=AsyncMock)
    @patch(_CONSOLE_LOGGER)
    @patch(_CONSOLE_PRINT_STEP)
    async def test_handle_message_returns_false_when_no_message_type(
        self,
        mock_print_step: MagicMock,
        mock_logger: MagicMock,
        mock_stream_api: AsyncMock,
    ) -> None:
        await stream_progress("ws://localhost:9090/ws")

        on_message = mock_stream_api.call_args[1]["on_message"]
        result = await on_message({})
        assert result is False
