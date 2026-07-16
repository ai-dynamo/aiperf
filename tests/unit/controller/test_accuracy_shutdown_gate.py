# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SystemController accuracy shutdown-gate.

The controller sets ``_should_wait_for_accuracy`` from config at construction and
clears it only when a ``ProcessAccuracyResultMessage`` arrives. While the flag is
True and no accuracy summary has been received, ``_check_and_trigger_shutdown``
must NOT trigger shutdown; once the message handler runs (with a real summary OR a
terminal ``results=None``), the flag clears and the accuracy term stops blocking.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.accuracy.models import AccuracySummary, ProcessAccuracyResult
from aiperf.common.messages import ProcessAccuracyResultMessage
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import AccuracyBenchmarkType


def _build_controller(benchmark_run, mock_service_manager, *, accuracy: bool):
    """Construct a SystemController mirroring the shared fixture, with accuracy
    optionally enabled at construction time so ``_should_wait_for_accuracy`` is
    computed by the real ``__init__`` logic rather than being set after the fact.
    """
    if accuracy:
        from aiperf.config.accuracy import AccuracyConfig

        benchmark_run.cfg.accuracy = AccuracyConfig(
            benchmark=AccuracyBenchmarkType.MMLU
        )

    mock_ui = AsyncMock()
    mock_comm = AsyncMock()

    def mock_get_class(protocol, name):
        if protocol == "service_manager":
            return lambda **kwargs: mock_service_manager
        if protocol == "ui":
            return lambda **kwargs: mock_ui
        if protocol == "communication":
            return lambda **kwargs: mock_comm
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.controller.system_controller.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            side_effect=mock_get_class,
        ),
    ):  # fmt: skip
        mock_proxy.return_value = AsyncMock()
        controller = SystemController(run=benchmark_run, service_id="test_controller")

    controller.stop = AsyncMock()
    return controller


def _summary() -> AccuracySummary:
    return AccuracySummary(
        total_evaluated=4,
        total_passed=3,
        accuracy_rate=0.75,
        overall_unparsed=0,
        grader_name="multiple_choice",
    )


class TestAccuracyShutdownGateEnabled:
    """Accuracy ENABLED: the flag blocks shutdown until the message clears it."""

    @pytest.mark.asyncio
    async def test_startup_sets_wait_flag_true(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        assert controller._should_wait_for_accuracy is True

    @pytest.mark.asyncio
    async def test_gate_blocks_shutdown_while_waiting(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        # Profile results present so only the accuracy term can gate.
        controller._profile_results_received = True
        controller._accuracy_results = None

        await controller._check_and_trigger_shutdown()

        assert controller._shutdown_triggered is False
        controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_message_clears_flag_and_unblocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        controller._profile_results_received = True

        summary = _summary()
        await controller._on_process_accuracy_result_message(
            ProcessAccuracyResultMessage(
                service_id="rm",
                accuracy_result=ProcessAccuracyResult(results=summary),
            )
        )

        assert controller._should_wait_for_accuracy is False
        assert controller._accuracy_results == summary
        assert controller._shutdown_triggered is True
        controller.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminal_none_message_clears_flag_and_unblocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        """A ``results=None`` terminal message must still clear the wait flag so a
        summary that could not be computed does not hang shutdown forever."""
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        controller._profile_results_received = True

        await controller._on_process_accuracy_result_message(
            ProcessAccuracyResultMessage(
                service_id="rm",
                accuracy_result=ProcessAccuracyResult(results=None),
            )
        )

        assert controller._should_wait_for_accuracy is False
        assert controller._accuracy_results is None
        assert controller._shutdown_triggered is True
        controller.stop.assert_awaited_once()


class TestAccuracyResultsInjection:
    """The dedicated-channel summary is materialized into the profile records
    exactly once at export time so legacy exporters read ``accuracy.*``."""

    def _controller_with_records(self, benchmark_run, mock_service_manager):
        from aiperf.common.models.record_models import (
            ProcessRecordsResult,
            ProfileResults,
        )

        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        controller._profile_results = ProcessRecordsResult(
            results=ProfileResults(records=[], completed=0, start_ns=0, end_ns=1),
        )
        controller._accuracy_results = _summary()
        return controller

    def test_injects_accuracy_metric_results_once(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = self._controller_with_records(benchmark_run, mock_service_manager)

        controller._inject_accuracy_results_into_records()

        records = controller._profile_results.results.records
        tags = [r.tag for r in records]
        assert tags == ["accuracy.overall", "accuracy.unparsed"]
        assert controller._accuracy_results_injected is True

        # Re-export must not double-append.
        controller._inject_accuracy_results_into_records()
        assert [r.tag for r in controller._profile_results.results.records] == tags

    def test_no_injection_when_no_summary(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = self._controller_with_records(benchmark_run, mock_service_manager)
        controller._accuracy_results = None

        controller._inject_accuracy_results_into_records()

        assert controller._profile_results.results.records == []
        assert controller._accuracy_results_injected is False


class TestAccuracyShutdownGateDisabled:
    """Accuracy DISABLED: the accuracy term never blocks shutdown."""

    @pytest.mark.asyncio
    async def test_startup_wait_flag_false(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=False
        )
        assert controller._should_wait_for_accuracy is False

    @pytest.mark.asyncio
    async def test_accuracy_term_never_blocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=False
        )
        controller._profile_results_received = True
        controller._accuracy_results = None

        await controller._check_and_trigger_shutdown()

        assert controller._shutdown_triggered is True
        controller.stop.assert_awaited_once()
