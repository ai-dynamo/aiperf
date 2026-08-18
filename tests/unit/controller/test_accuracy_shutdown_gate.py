# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SystemController accuracy shutdown-gate.

Accuracy is a domain in the ``ResultJoinCoordinator`` shutdown barrier: RecordsManager
advertises a ``result_producer:accuracy`` capability (iff accuracy is enabled), which
registers the domain. While that domain is pending, ``_check_and_trigger_shutdown``
must NOT trigger shutdown; once the ``ProcessAccuracyResultMessage`` handler runs
(with a real summary OR a terminal ``results=None``), the domain completes and the
barrier stops blocking.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.accuracy.models import AccuracySummary, ProcessAccuracyResult
from aiperf.common.enums import SystemState
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
    # Not started, so pub_client is unset; _set_system_state publishes.
    controller.publish = AsyncMock()
    controller._system_state = SystemState.PROFILING
    return controller


def _summary() -> AccuracySummary:
    return AccuracySummary(
        total_evaluated=4,
        total_passed=3,
        accuracy_rate=0.75,
        overall_unparsed=0,
        grader_name="multiple_choice",
    )


def _register_records_manager(controller, *, accuracy: bool) -> None:
    """Simulate RecordsManager registering its result-producer domains."""
    controller._result_join_coordinator.register("profile", "rm")
    if accuracy:
        controller._result_join_coordinator.register("accuracy", "rm")


class TestAccuracyShutdownGateEnabled:
    """Accuracy ENABLED: the accuracy domain blocks shutdown until its message
    completes it."""

    @pytest.mark.asyncio
    async def test_gate_blocks_shutdown_while_waiting(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        _register_records_manager(controller, accuracy=True)
        # Profile complete so only the accuracy domain can gate.
        controller._result_join_coordinator.complete_domain("profile")

        await controller._check_and_trigger_shutdown()

        assert "accuracy" in controller._result_join_coordinator.pending_domains
        assert controller._shutdown_triggered is False
        controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_message_completes_domain_and_unblocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        _register_records_manager(controller, accuracy=True)
        controller._result_join_coordinator.complete_domain("profile")

        summary = _summary()
        await controller._on_process_accuracy_result_message(
            ProcessAccuracyResultMessage(
                service_id="rm",
                accuracy_result=ProcessAccuracyResult(results=summary),
            )
        )

        assert controller._result_join_coordinator.ready is True
        assert controller._accuracy_results == summary
        assert controller._shutdown_triggered is True
        controller.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminal_none_message_completes_domain_and_unblocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        """A ``results=None`` terminal message must still complete the accuracy
        domain so a summary that could not be computed does not hang shutdown."""
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=True
        )
        _register_records_manager(controller, accuracy=True)
        controller._result_join_coordinator.complete_domain("profile")

        await controller._on_process_accuracy_result_message(
            ProcessAccuracyResultMessage(
                service_id="rm",
                accuracy_result=ProcessAccuracyResult(results=None),
            )
        )

        assert controller._result_join_coordinator.ready is True
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
    """Accuracy DISABLED: RecordsManager advertises no accuracy domain, so it
    never blocks shutdown."""

    @pytest.mark.asyncio
    async def test_accuracy_domain_never_registered(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=False
        )
        _register_records_manager(controller, accuracy=False)

        assert "accuracy" not in controller._result_join_coordinator.pending_domains

    @pytest.mark.asyncio
    async def test_completing_profile_alone_unblocks(
        self, benchmark_run, mock_service_manager
    ) -> None:
        controller = _build_controller(
            benchmark_run, mock_service_manager, accuracy=False
        )
        _register_records_manager(controller, accuracy=False)
        controller._result_join_coordinator.complete_domain("profile")

        await controller._check_and_trigger_shutdown()

        assert controller._shutdown_triggered is True
        controller.stop.assert_awaited_once()
