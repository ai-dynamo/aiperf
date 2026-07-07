# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseOrchestrator initialization, cancellation, and phase configuration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from tests.unit.timing.conftest import make_phase_config, make_timing_config


def make_dataset(num_convs: int = 3, turns: int = 1) -> DatasetMetadata:
    convs = [
        ConversationMetadata(
            conversation_id=f"conv-{i}", turns=[TurnMetadata() for _ in range(turns)]
        )
        for i in range(num_convs)
    ]
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def make_router() -> MagicMock:
    r = MagicMock()
    r.send_credit = AsyncMock()
    r.cancel_all_credits = AsyncMock()
    r.mark_credits_complete = MagicMock()
    r.set_return_callback = MagicMock()
    r.set_first_token_callback = MagicMock()
    r.reset = MagicMock()
    return r


def make_publisher() -> MagicMock:
    p = MagicMock()
    p.publish_phase_start = AsyncMock()
    p.publish_phase_complete = AsyncMock()
    p.publish_phase_sending_complete = AsyncMock()
    p.publish_progress = AsyncMock()
    p.publish_credits_complete = AsyncMock()
    return p


@pytest.mark.asyncio
class TestOrchestratorInit:
    """Tests for PhaseOrchestrator initialization behavior."""

    async def test_registers_callbacks_with_router(self) -> None:
        """Orchestrator registers credit return and first token callbacks during init."""
        router = make_router()
        orch = PhaseOrchestrator(
            config=make_timing_config(
                TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
            ),
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        router.set_return_callback.assert_called_once_with(
            orch._callback_handler.on_credit_return
        )
        router.set_first_token_callback.assert_called_once_with(
            orch._callback_handler.on_first_token
        )

    async def test_conversation_source_samples_from_dataset(self) -> None:
        """Conversation source is initialized and can sample conversations."""
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        sampled = orch.conversation_source.next()
        assert sampled is not None
        assert sampled.conversation_id.startswith("conv-")


@pytest.mark.asyncio
class TestCancellation:
    """Tests for PhaseOrchestrator cancellation behavior."""

    async def test_cancel_cancels_router_credits(self) -> None:
        """Calling cancel() triggers cancellation of all in-flight credits."""
        router = make_router()
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        await orch.cancel()
        router.cancel_all_credits.assert_called_once()

    async def test_cancel_can_be_called_multiple_times(self) -> None:
        """Calling cancel() multiple times does not raise errors."""
        router = make_router()
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        await orch.cancel()
        await orch.cancel()
        assert router.cancel_all_credits.call_count == 2


class _CancelDuringRunFakeRunner:
    """Fake PhaseRunner reproducing the Ctrl-C race in ``_execute_phases``.

    Its ``run()`` invokes the orchestrator's ``cancel()`` (which runs
    ``_cancel_active_runners`` -> ``_active_runners.clear()``) and THEN
    returns normally, forcing the exact interleaving of a SIGINT that lands
    while ``runner.run()`` is suspended in the returns/sending-completion
    window: the active-runners list is emptied *before* control returns to
    the unconditional ``remove(runner)`` line. Deterministic — no sleeps.
    """

    def __init__(self, orch: PhaseOrchestrator, *, phase: str = "profiling") -> None:
        self._orch = orch
        self.phase = phase
        self.cancel_calls = 0
        self.run_calls = 0

    async def run(self, *, is_final_phase: bool) -> None:
        self.run_calls += 1
        # SIGINT arrives here (cooperative-cancellation window): cancel()
        # clears _active_runners while run() is "in flight", then run()
        # returns normally.
        await self._orch.cancel()

    def cancel(self) -> None:
        self.cancel_calls += 1


@pytest.mark.asyncio
class TestCancelDuringRunRace:
    """Regression tests for the cancel()-vs-_execute_phases removal race.

    Before the fix, ``_cancel_active_runners`` calling ``list.clear()``
    concurrently with the unconditional ``_active_runners.remove(runner)``
    in ``_execute_phases`` raised ``ValueError: list.remove(x): x not in
    list``, crashing phase execution and losing the aggregated exports.
    """

    def _make_orch(self) -> tuple[PhaseOrchestrator, MagicMock, MagicMock]:
        router = make_router()
        publisher = make_publisher()
        orch = PhaseOrchestrator(
            config=make_timing_config(
                TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
            ),
            phase_publisher=publisher,
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        return orch, router, publisher

    async def test_cancel_mid_run_does_not_raise_and_removes_once(
        self, monkeypatch
    ) -> None:
        """cancel() clearing the list mid-run must not crash the remove line."""
        orch, router, _ = self._make_orch()
        await orch.initialize()

        fake = _CancelDuringRunFakeRunner(orch)
        monkeypatch.setattr(
            "aiperf.timing.phase_orchestrator.PhaseRunner",
            lambda **kwargs: fake,
        )

        # Must NOT raise ValueError: list.remove(x): x not in list.
        await orch._execute_phases()

        assert fake.run_calls == 1
        # Removed exactly once (by cancel()'s clear); the guarded remove skips.
        assert fake.cancel_calls == 1
        assert orch._active_runners == []
        router.cancel_all_credits.assert_called_once()

    async def test_cancel_mid_run_still_runs_export_cleanup(self, monkeypatch) -> None:
        """The start path's finally (exports/cleanup) still runs after the race."""
        orch, router, publisher = self._make_orch()
        await orch.initialize()

        fake = _CancelDuringRunFakeRunner(orch)
        monkeypatch.setattr(
            "aiperf.timing.phase_orchestrator.PhaseRunner",
            lambda **kwargs: fake,
        )

        # Drives _execute_phases inside the try/finally; must complete cleanly
        # and reach the credits-complete cleanup (analogue of "exports run").
        await orch._start_orchestrator()

        router.mark_credits_complete.assert_called_once()
        publisher.publish_credits_complete.assert_awaited_once()
        assert orch._active_runners == []


@pytest.mark.asyncio
class TestOrchestratorStop:
    """Tests for PhaseOrchestrator @on_stop cleanup.

    Without the @on_stop hook, runners tracked in ``_active_runners`` are
    leaked on the normal (non-cancellation) shutdown path.
    """

    async def test_stop_cancels_and_clears_active_runners(self) -> None:
        """@on_stop cancels every tracked runner and empties the active list."""
        router = make_router()
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()

        runner_a = MagicMock()
        runner_a.phase = "warmup"
        runner_b = MagicMock()
        runner_b.phase = "profiling"
        orch._active_runners.extend([runner_a, runner_b])

        await orch.stop()

        runner_a.cancel.assert_called_once()
        runner_b.cancel.assert_called_once()
        assert orch._active_runners == []


@pytest.mark.asyncio
class TestPhaseConfig:
    """Tests for phase configuration handling."""

    async def test_warmup_and_profiling_phases_in_order(self) -> None:
        """When both warmup and profiling are configured, they execute in order."""
        warmup = make_phase_config(
            "warmup",
            TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
        )
        profiling = make_phase_config(
            "profiling",
            TimingMode.REQUEST_RATE,
            request_count=10,
            request_rate=10.0,
        )
        cfg = TimingConfig(phase_configs=[warmup, profiling])
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert phases == ["warmup", "profiling"]

    async def test_profiling_only_excludes_warmup(self) -> None:
        """When only profiling is configured, warmup phase is not present."""
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert "profiling" in phases
        assert "warmup" not in phases
