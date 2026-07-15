# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator metrics instrumentation.

Focuses on:
- ``@track_handler`` success, retry, fatal, and generic-error counter labels.
- Exception identity preservation so kopf sees the original handler failure.
- ``start_metrics_server(port=0)`` as the documented disabled path.
- Repeated disabled metrics setup remaining a no-op.

Out of scope:
- Real Prometheus socket serving; ``tests/unit/operator/test_metrics.py`` covers
  the basic positive start hook via a patched ``start_http_server``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import TypeVar
from unittest.mock import Mock, patch

import pytest
from prometheus_client import REGISTRY
from pytest import param

from aiperf.operator.metrics import (
    HANDLER_DURATION,
    HANDLER_TOTAL,
    start_metrics_server,
    track_handler,
)

_T = TypeVar("_T")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_operator_metrics() -> Generator[None, None, None]:
    """Clear process-local metric samples so each assertion observes one scenario."""
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()
    yield
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()


def _counter_value(handler: str, outcome: str) -> float:
    return sum(
        sample.value
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == "aiperf_operator_handler_total"
        and sample.labels.get("handler") == handler
        and sample.labels.get("outcome") == outcome
    )


def _duration_count(handler: str) -> float:
    return sum(
        sample.value
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == "aiperf_operator_handler_duration_seconds_count"
        and sample.labels.get("handler") == handler
    )


def _tracked_handler(
    handler_name: str,
) -> Callable[[Callable[[], Awaitable[_T]]], Callable[[], Awaitable[_T]]]:
    return track_handler(handler_name)


# =============================================================================
# track_handler outcome classification
# =============================================================================


class TestTrackHandlerOutcomeCounters:
    """Kopf handler metrics must classify every terminal outcome distinctly."""

    @pytest.mark.asyncio
    async def test_track_handler_success_returns_value_and_records_success_only(
        self,
    ) -> None:
        @_tracked_handler("jobset_terminal_success")
        async def handler() -> str:
            return "aiperf-bench-7f2a"

        result = await handler()

        assert result == "aiperf-bench-7f2a"
        assert _counter_value("jobset_terminal_success", "success") == 1
        assert _counter_value("jobset_terminal_success", "retry") == 0
        assert _counter_value("jobset_terminal_success", "fatal") == 0
        assert _counter_value("jobset_terminal_success", "error") == 0
        assert _duration_count("jobset_terminal_success") == 1

    @pytest.mark.parametrize(
        "handler_name,exception_factory,outcome",
        [
            param(
                "monitor_progress_retry",
                lambda: __import__("kopf").TemporaryError("apiserver throttled", delay=3),
                "retry",
                id="temporary-error-classifies-retry",
            ),
            param(
                "create_handler_fatal",
                lambda: __import__("kopf").PermanentError("invalid AIPerfJob spec"),
                "fatal",
                id="permanent-error-classifies-fatal",
            ),
            param(
                "cleanup_handler_error",
                lambda: RuntimeError("results sidecar closed connection"),
                "error",
                id="generic-exception-classifies-error",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_track_handler_failure_records_expected_outcome_and_preserves_exception(
        self,
        handler_name: str,
        exception_factory: Callable[[], BaseException],
        outcome: str,
    ) -> None:
        raised = exception_factory()

        @_tracked_handler(handler_name)
        async def handler() -> None:
            raise raised

        with pytest.raises(type(raised)) as exc_info:
            await handler()

        assert exc_info.value is raised
        assert _counter_value(handler_name, outcome) == 1
        assert _counter_value(handler_name, "success") == 0
        assert _duration_count(handler_name) == 1

    @pytest.mark.asyncio
    async def test_track_handler_cancelled_error_records_error_and_preserves_cancellation(
        self,
    ) -> None:
        import asyncio

        cancellation = asyncio.CancelledError("kopf shutdown cancelled monitor tick")

        @_tracked_handler("monitor_progress_cancelled")
        async def handler() -> None:
            raise cancellation

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await handler()

        assert exc_info.value is cancellation
        assert _counter_value("monitor_progress_cancelled", "error") == 1
        assert _counter_value("monitor_progress_cancelled", "success") == 0
        assert _duration_count("monitor_progress_cancelled") == 1


# =============================================================================
# start_metrics_server disabled and repeated setup paths
# =============================================================================


class TestStartMetricsServerAdversarial:
    """Startup metrics setup must not bind sockets when explicitly disabled."""

    def test_start_metrics_server_port_zero_logs_disabled_and_does_not_start_http_server(
        self,
    ) -> None:
        with patch("aiperf.operator.metrics.start_http_server") as start_http_server:
            start_metrics_server(port=0)

        start_http_server.assert_not_called()

    def test_start_metrics_server_repeated_port_zero_setup_remains_noop(self) -> None:
        start_http_server = Mock()

        with patch("aiperf.operator.metrics.start_http_server", start_http_server):
            start_metrics_server(port=0)
            start_metrics_server(port=0)
            start_metrics_server(port=-1)

        start_http_server.assert_not_called()
