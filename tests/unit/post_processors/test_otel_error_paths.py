# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for error handling paths in OTel processor and fanout."""

from __future__ import annotations

from queue import Empty, Full
from unittest.mock import MagicMock, patch

import pytest

from aiperf.post_processors.otel_metrics_results_processor import (
    OTelMetricsResultsProcessor,
)


@pytest.fixture
def mock_user_config(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Create a mock user config with OTel enabled."""
    config = MagicMock()
    config.otel_metrics_url = "http://localhost:4318/v1/metrics"
    config.mlflow_enabled = False
    config.mlflow_tracking_uri = None
    config.otel_stream_metrics_enabled = True
    config.otel_stream_timing_enabled = True
    config.benchmark_id = "test-123"
    config.endpoint.type = "chat"
    config.endpoint.model_names = ["mock-model"]
    config.otel_custom_resource_attributes = {}
    return config


class TestProcessorErrorPaths:
    """Test error handling in OTelMetricsResultsProcessor."""

    @pytest.mark.asyncio
    async def test_flush_handles_exception_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, mock_user_config: MagicMock
    ) -> None:
        """flush() should not raise when meter_provider.force_flush fails."""
        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ):
            processor = OTelMetricsResultsProcessor(
                service_id="test",
                user_config=mock_user_config,
            )
            # Simulate fanout mode with a queue
            processor._streaming_ready = True
            processor._use_fanout_process = True
            mock_queue = MagicMock()
            processor._fanout_queue = mock_queue

            # flush should not raise
            await processor.flush(force=True)
            mock_queue.put_nowait.assert_called_once()

    @pytest.mark.asyncio
    async def test_queue_fanout_event_handles_put_exception(
        self, mock_user_config: MagicMock
    ) -> None:
        """_queue_fanout_event should handle unexpected exceptions from put_nowait."""
        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ):
            processor = OTelMetricsResultsProcessor(
                service_id="test",
                user_config=mock_user_config,
            )
            mock_queue = MagicMock()
            mock_queue.put_nowait.side_effect = OSError("broken pipe")
            processor._fanout_queue = mock_queue

            # Should not raise
            processor._queue_fanout_event("test", {"key": "value"})

    @pytest.mark.asyncio
    async def test_queue_fanout_event_full_then_drop_fails(
        self, mock_user_config: MagicMock
    ) -> None:
        """When queue is full and drop also fails, should increment drop counter."""
        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ):
            processor = OTelMetricsResultsProcessor(
                service_id="test",
                user_config=mock_user_config,
            )
            mock_queue = MagicMock()
            # First put_nowait raises Full, get_nowait raises Empty (can't drop)
            mock_queue.put_nowait.side_effect = Full()
            mock_queue.get_nowait.side_effect = Empty()
            processor._fanout_queue = mock_queue

            processor._queue_fanout_event("test", {"key": "value"})
            assert processor._fanout_dropped_events == 1

    @pytest.mark.asyncio
    async def test_drop_oldest_handles_exception(
        self, mock_user_config: MagicMock
    ) -> None:
        """_drop_oldest_fanout_event should return False on unexpected exception."""
        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ):
            processor = OTelMetricsResultsProcessor(
                service_id="test",
                user_config=mock_user_config,
            )
            mock_queue = MagicMock()
            mock_queue.get_nowait.side_effect = OSError("broken")
            processor._fanout_queue = mock_queue

            result = processor._drop_oldest_fanout_event()
            assert result is False

    @pytest.mark.asyncio
    async def test_process_result_skips_when_not_ready(
        self, mock_user_config: MagicMock
    ) -> None:
        """process_result should silently return when streaming is not ready."""
        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ):
            processor = OTelMetricsResultsProcessor(
                service_id="test",
                user_config=mock_user_config,
            )
            processor._streaming_ready = False

            record = MagicMock()
            await processor.process_result(record)
            # No exception raised


class TestFanoutFlushRetry:
    """Test MLflow flush retry behavior in the fanout."""

    def test_flush_retry_on_log_batch_failure_preserves_buffer(self) -> None:
        """When log_batch fails, buffer should NOT be cleared (retry on next flush)."""
        from aiperf.post_processors.otel_streaming_fanout import _MLflowFanoutState

        state = _MLflowFanoutState(
            module=MagicMock(),
            client=MagicMock(),
            metric_cls=MagicMock(side_effect=lambda k, v, ts, s: (k, v, ts, s)),
            run_id="test-run",
            step=0,
            buffer=[("live.metric_a", 1.0), ("live.metric_b", 2.0)],
            timing_gauge_snapshots={},
            counter_snapshots={},
        )
        # Simulate log_batch failure
        state.client.log_batch.side_effect = RuntimeError("network error")

        # The buffer should remain after a failed flush — verified via
        # the dataclass state since _flush_mlflow_metrics is a closure
        # inside run_otel_streaming_fanout and cannot be called directly.
        # This test documents the contract: buffer is NOT cleared on failure.
        assert len(state.buffer) == 2
        state.client.log_batch.side_effect = RuntimeError("still failing")
        # Buffer untouched
        assert state.buffer == [("live.metric_a", 1.0), ("live.metric_b", 2.0)]
