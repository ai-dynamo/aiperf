# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.operator.metrics."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from aiperf.operator.metrics import (
    HANDLER_DURATION,
    HANDLER_TOTAL,
    track_handler,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    """Reset registry-wide samples between tests (not strictly required but isolates assertions)."""
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()
    yield


@pytest.mark.asyncio
async def test_track_handler_increments_success_counter() -> None:
    @track_handler("test_handler")
    async def fake_handler() -> None:
        return None

    await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_total"
        and s.labels.get("handler") == "test_handler"
        and s.labels.get("outcome") == "success"
    ]
    assert any(s.value >= 1 for s in samples)


@pytest.mark.asyncio
async def test_track_handler_increments_error_counter_on_exception() -> None:
    @track_handler("flaky_handler")
    async def fake_handler() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_total"
        and s.labels.get("handler") == "flaky_handler"
        and s.labels.get("outcome") == "error"
    ]
    assert any(s.value >= 1 for s in samples)


@pytest.mark.asyncio
async def test_track_handler_records_duration_histogram() -> None:
    @track_handler("timed_handler")
    async def fake_handler() -> None:
        return None

    await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_duration_seconds_count"
        and s.labels.get("handler") == "timed_handler"
    ]
    assert any(s.value >= 1 for s in samples)


def _outcome_counter(handler: str, outcome: str) -> float:
    """Sum the handler_total counter across whatever samples are present."""
    total = 0.0
    for m in REGISTRY.collect():
        for s in m.samples:
            if (
                s.name == "aiperf_operator_handler_total"
                and s.labels.get("handler") == handler
                and s.labels.get("outcome") == outcome
            ):
                total += s.value
    return total


class TestTrackHandlerAdversarial:
    """Adversarial tests for ``track_handler`` covering exception classes
    that aren't subclasses of ``Exception``, sync-misuse, concurrent
    accounting, and stable-name collisions.

    Why this matters: the kopf operator runs handlers under an asyncio
    event loop where ``asyncio.CancelledError`` is a ``BaseException`` (not
    ``Exception``) — a naive ``except Exception:`` would silently classify
    a cancelled handler as ``outcome=success``, hiding shutdown / hang
    pathologies. Operators rely on this counter to alert on stuck pods.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_is_classified_as_error(self) -> None:
        """asyncio.CancelledError must increment outcome=error AND propagate.

        CancelledError is a BaseException in Python 3.8+, so ``except
        Exception`` would miss it — that classifies a handler that timed
        out and got cancelled by kopf as success, which is wrong: it hides
        the very stuckness operators alert on.
        """
        import asyncio

        @track_handler("cancellable_handler")
        async def fake_handler() -> None:
            raise asyncio.CancelledError("kopf timed out")

        with pytest.raises(asyncio.CancelledError):
            await fake_handler()

        assert _outcome_counter("cancellable_handler", "error") >= 1
        assert _outcome_counter("cancellable_handler", "success") == 0

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_is_classified_as_error(self) -> None:
        """SIGINT during a handler invocation must record error before propagating."""

        @track_handler("kbi_handler")
        async def fake_handler() -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            await fake_handler()

        assert _outcome_counter("kbi_handler", "error") >= 1
        assert _outcome_counter("kbi_handler", "success") == 0

    @pytest.mark.asyncio
    async def test_system_exit_is_classified_as_error(self) -> None:
        """SystemExit (BaseException in 3.10+) must record error before propagating."""

        @track_handler("sysexit_handler")
        async def fake_handler() -> None:
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            await fake_handler()

        assert _outcome_counter("sysexit_handler", "error") >= 1
        assert _outcome_counter("sysexit_handler", "success") == 0

    @pytest.mark.asyncio
    async def test_concurrent_invocations_each_increment(self) -> None:
        """Counter accounting under asyncio.gather: N invocations → N increments.

        prometheus_client.Counter is process-safe under the GIL but pin it
        explicitly to catch a regression to a non-atomic increment scheme.
        """
        import asyncio

        @track_handler("gathered_handler")
        async def fake_handler() -> None:
            await asyncio.sleep(0)
            return None

        await asyncio.gather(*(fake_handler() for _ in range(20)))
        assert _outcome_counter("gathered_handler", "success") >= 20

    @pytest.mark.asyncio
    async def test_two_handlers_same_name_share_counter(self) -> None:
        """Two `@track_handler("X")` definitions both increment ONE shared
        Counter labelset. That's intentional (label cardinality bound) —
        pin it so future "improvements" don't silently change semantics.
        """

        @track_handler("shared_name")
        async def first() -> None:
            return None

        @track_handler("shared_name")
        async def second() -> None:
            return None

        await first()
        await second()
        assert _outcome_counter("shared_name", "success") >= 2

    @pytest.mark.asyncio
    async def test_handler_running_past_60s_falls_in_inf_bucket(self) -> None:
        """Default histogram tops at 60s; longer handlers must still be
        observed in the +Inf bucket (i.e. duration_seconds_count increments
        even when the value exceeds the largest finite bucket).

        Uses a perf_counter monkeypatch so the test stays fast.
        """
        import time as time_module

        from aiperf.operator import metrics as metrics_mod

        # Simulate a 65-second handler by patching perf_counter to jump.
        ticks = iter([1000.0, 1065.0])

        def fake_perf_counter() -> float:
            return next(ticks)

        @track_handler("slow_handler")
        async def fake_handler() -> None:
            return None

        original = time_module.perf_counter
        metrics_mod.time.perf_counter = fake_perf_counter
        try:
            await fake_handler()
        finally:
            metrics_mod.time.perf_counter = original

        # The +Inf bucket count must include this run.
        inf_bucket_samples = [
            s
            for m in REGISTRY.collect()
            for s in m.samples
            if s.name == "aiperf_operator_handler_duration_seconds_bucket"
            and s.labels.get("handler") == "slow_handler"
            and s.labels.get("le") == "+Inf"
        ]
        assert any(s.value >= 1 for s in inf_bucket_samples), (
            "+Inf bucket must collect handlers running longer than the largest finite bucket"
        )

    def test_track_handler_on_sync_function_yields_coroutine_at_call_time(self) -> None:
        """Decorating a sync function with @track_handler doesn't fail at
        decoration time — wrapper is unconditionally async. Calling the
        wrapped sync fn returns a coroutine that, when awaited, will await
        the (non-awaitable) sync return and raise TypeError.

        This is intentional: track_handler is for kopf async handlers only.
        Pin the failure mode so a misuse surfaces visibly rather than
        silently double-running the function or yielding wrong metrics.
        """
        import asyncio
        import inspect

        @track_handler("sync_misuse")
        def sync_fn() -> int:  # type: ignore[misc]
            return 42

        # Decoration succeeds.
        result = sync_fn()
        # The wrapper is async, so calling returns a coroutine.
        assert inspect.iscoroutine(result)

        # Awaiting the coroutine raises TypeError because `await 42` is invalid.
        with pytest.raises(TypeError):
            asyncio.run(result)


class TestStartMetricsServerAdversarial:
    """Adversarial tests for ``start_metrics_server`` — the single entry
    point in the kopf startup hook. A misbehaving server fn here can
    crash the operator before any reconcile has run."""

    def test_port_zero_is_disabled(self) -> None:
        """port=0 → no http server. Documented "disabled" sentinel."""
        from unittest.mock import patch

        from aiperf.operator.metrics import start_metrics_server

        with patch("aiperf.operator.metrics.start_http_server") as mock_start:
            start_metrics_server(port=0)
        mock_start.assert_not_called()

    def test_negative_port_is_disabled(self) -> None:
        """port=-1 must be treated identically to 0 — no crash, no listener.

        While Pydantic guards the OperatorEnvironment field at config-load
        time, programmatic / test callers can bypass that. Defensive guard
        in start_metrics_server is the second line of defense.
        """
        from unittest.mock import patch

        from aiperf.operator.metrics import start_metrics_server

        with patch("aiperf.operator.metrics.start_http_server") as mock_start:
            start_metrics_server(port=-1)
        mock_start.assert_not_called()

    def test_positive_port_starts_server(self) -> None:
        """A real positive port must actually call start_http_server."""
        from unittest.mock import patch

        from aiperf.operator.metrics import start_metrics_server

        with patch("aiperf.operator.metrics.start_http_server") as mock_start:
            start_metrics_server(port=9090)
        mock_start.assert_called_once_with(9090)

    def test_double_start_propagates_oserror(self) -> None:
        """Double-start (port already in use) currently raises OSError out
        of start_http_server. Pin the behavior so callers know they own
        retry/swallow semantics — kopf's startup hook will surface this
        as an operator startup failure, which is the correct loud
        behavior for a port-conflict misconfiguration.
        """
        from unittest.mock import patch

        from aiperf.operator.metrics import start_metrics_server

        with (
            patch(
                "aiperf.operator.metrics.start_http_server",
                side_effect=OSError("Address already in use"),
            ),
            pytest.raises(OSError, match="Address already in use"),
        ):
            start_metrics_server(port=9090)
