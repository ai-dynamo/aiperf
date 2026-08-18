# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A degraded run must still export, and must never report success.

These pin the boundary between three distinct outcomes that the controller
previously collapsed into two:

* complete -- results exported, exit 0
* degraded -- results exported, exit non-zero, degradation named
* no results -- nothing to export, exit non-zero

The regressions guarded here all pushed a *degraded* run into the wrong bucket:
an aggregation diagnostic or a reaped producer either discarded the whole
export (silent total loss) or released the shutdown barrier and exited 0
(silent success).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CommandType, ExportLevel, SystemState
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    FinalizeArtifactsCommand,
    ProcessRecordsResultMessage,
)
from aiperf.common.models import ErrorDetails, ProcessRecordsResult
from aiperf.common.models.record_models import MetricResult, ProfileResults
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _records_result(*errors: ErrorDetails) -> ProcessRecordsResult:
    """A complete, exportable record set that also carries diagnostics."""
    return ProcessRecordsResult(
        results=ProfileResults(
            records=[
                MetricResult(tag="request_latency", header="Request Latency", unit="ms")
            ],
            completed=100,
            start_ns=0,
            end_ns=1_000_000,
        ),
        errors=list(errors),
    )


async def _run_stop_hook(controller: SystemController) -> int:
    """Drive ``_stop_system_controller`` and return the process exit code."""
    controller.ui = AsyncMock()
    controller.publish = AsyncMock()
    controller.comms = AsyncMock()
    controller.proxy_manager = AsyncMock()
    controller._announce_benchmark_complete = AsyncMock()
    controller._print_post_benchmark_info_and_metrics = AsyncMock()
    controller._print_exit_errors_and_log_file = MagicMock()

    exit_codes: list[int] = []
    with (
        patch(
            "aiperf.controller.system_controller.os._exit",
            side_effect=exit_codes.append,
        ),
        patch(
            "aiperf.controller.system_controller.cleanup_global_log_queue",
            AsyncMock(),
        ),
    ):  # fmt: skip
        await controller._stop_system_controller()

    assert exit_codes, "the stop hook must always terminate the process"
    return exit_codes[0]


class TestAggregationErrorsDoNotDiscardTheExport:
    """C3 / H6: ``results.errors`` is a diagnostic list, not a fatal verdict."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stage",
        [
            param("gpu_telemetry_drain", id="h6_gpu_telemetry_drain"),
            param("stream_export_finalize", id="c3_stream_export_finalize"),
        ],
    )  # fmt: skip
    async def test_aggregation_error_does_not_set_export_failed(
        self, system_controller: SystemController, stage: str
    ) -> None:
        """``_export_failed`` withholds the k8s ready marker; aggregation must not.

        Before the fix this set ``_export_failed = True``, so an OOMKilled
        GPU-telemetry container blocked publication of a fully valid inference
        result set.
        """
        system_controller._check_and_trigger_shutdown = AsyncMock()
        error = ErrorDetails(message="drain timed out", details={"stage": stage})

        await system_controller._on_process_records_result_message(
            ProcessRecordsResultMessage(
                service_id="records_manager",
                results=_records_result(error),
            )
        )

        assert system_controller._export_failed is False
        # Still reported, and still forces a non-zero exit.
        assert [e.operation for e in system_controller._exit_errors] == [
            "process_records"
        ]

    @pytest.mark.asyncio
    async def test_a_run_with_aggregation_errors_still_writes_its_export(
        self, system_controller: SystemController
    ) -> None:
        """The C3 headline: one diagnostic must not delete the whole export.

        Before the fix ``_stop_system_controller`` gated the export on
        ``not self._exit_errors``, so this produced no profile_export.csv/.json,
        no console summary, no auto-plot and no ready marker.
        """
        system_controller._check_and_trigger_shutdown = AsyncMock()
        await system_controller._on_process_records_result_message(
            ProcessRecordsResultMessage(
                service_id="records_manager",
                results=_records_result(ErrorDetails(message="one bad record")),
            )
        )

        exit_code = await _run_stop_hook(system_controller)

        system_controller._print_post_benchmark_info_and_metrics.assert_awaited_once()
        system_controller._print_exit_errors_and_log_file.assert_called_once()
        assert exit_code == 1, "a degraded run must not report success"

    @pytest.mark.asyncio
    async def test_a_run_with_no_results_still_fails_without_exporting(
        self, system_controller: SystemController
    ) -> None:
        """Fail-closed half of the same gate: nothing to export, exit non-zero."""
        system_controller._profile_results = None
        system_controller._exit_errors.append(
            MagicMock(
                operation="service_runtime", error_details=ErrorDetails(message="x")
            )
        )

        exit_code = await _run_stop_hook(system_controller)

        system_controller._print_post_benchmark_info_and_metrics.assert_not_awaited()
        system_controller._print_exit_errors_and_log_file.assert_called_once()
        assert exit_code == 1


class TestEvictedProducersReachTheOutcome:
    """C4: barrier eviction released the run, which then exited 0."""

    @pytest.mark.asyncio
    async def test_a_reaped_producer_makes_the_run_exit_non_zero(
        self, system_controller: SystemController
    ) -> None:
        """8 worker pods, one OOMKilled: the table is built from 7 of them.

        Before the fix ``_on_service_reaped`` only logged a warning, so a sweep
        or CI job reading the exit code recorded a successful benchmark with
        wrong throughput.
        """
        system_controller._system_state = SystemState.PROFILING
        system_controller._check_and_trigger_shutdown = AsyncMock()
        system_controller._result_join_coordinator.register("profile", "worker_group_7")
        system_controller._profile_results = _records_result()

        await system_controller._on_service_reaped("worker_group_7", "pod OOMKilled")

        assert system_controller._result_join_coordinator.evicted == {
            "worker_group_7": "pod OOMKilled"
        }
        assert [e.operation for e in system_controller._exit_errors] == [
            "result_producer_reaped"
        ]

        exit_code = await _run_stop_hook(system_controller)

        # The surviving producers' results are still exported...
        system_controller._print_post_benchmark_info_and_metrics.assert_awaited_once()
        # ...but the run is not reported as a success.
        assert exit_code == 1

    def test_evicted_producers_are_named_in_the_console_output(
        self, system_controller: SystemController
    ) -> None:
        """``ResultJoinCoordinator.evicted`` needs a production reader."""
        system_controller._result_join_coordinator.register("profile", "worker_group_7")
        system_controller._result_join_coordinator.evict_service(
            "worker_group_7", "pod OOMKilled"
        )
        console = MagicMock()

        system_controller._print_degraded_producers(console)

        printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
        assert "DEGRADED" in printed
        assert "worker_group_7" in printed
        assert "OOMKilled" in printed

    def test_a_complete_run_prints_no_degradation_banner(
        self, system_controller: SystemController
    ) -> None:
        console = MagicMock()
        system_controller._print_degraded_producers(console)
        console.print.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_reaped_service_is_dropped_from_the_command_target_maps(
        self, system_controller: SystemController
    ) -> None:
        """H9 half: finalize must not target a peer known to be dead."""
        system_controller._system_state = SystemState.PROFILING
        system_controller._check_and_trigger_shutdown = AsyncMock()
        info = MagicMock(
            service_id="record_processor_2",
            service_type=ServiceType.RECORD_PROCESSOR,
        )
        system_controller.service_manager.service_id_map = {
            "record_processor_2": info,
        }
        system_controller.service_manager.service_map = {
            ServiceType.RECORD_PROCESSOR: [info]
        }
        system_controller._result_join_coordinator.register(
            "profile", "record_processor_2"
        )

        await system_controller._on_service_reaped(
            "record_processor_2", "missed heartbeats"
        )

        assert (
            "record_processor_2" not in system_controller.service_manager.service_id_map
        )
        assert (
            system_controller.service_manager.service_map[ServiceType.RECORD_PROCESSOR]
            == []
        )
        assert "record_processor_2" in system_controller._reaped_service_ids


class TestFinalizeArtifactsIsReportable:
    """H9: a finalize failure must be a reported failure, never a hang."""

    @pytest.mark.asyncio
    async def test_finalize_skips_reaped_record_processors(
        self, system_controller: SystemController
    ) -> None:
        """A dead peer would otherwise burn a full command timeout, then raise."""
        system_controller._is_kubernetes = MagicMock(return_value=False)
        alive = MagicMock(
            service_id="record_processor_0", service_type=ServiceType.RECORD_PROCESSOR
        )
        dead = MagicMock(
            service_id="record_processor_1", service_type=ServiceType.RECORD_PROCESSOR
        )
        system_controller.service_manager.service_id_map = {
            "record_processor_0": alive,
            "record_processor_1": dead,
        }
        system_controller._reaped_service_ids.add("record_processor_1")

        targeted: list[list[str]] = []

        async def respond(command, service_ids, timeout):  # noqa: ANN001, ARG001
            targeted.append(list(service_ids))
            return [
                CommandAcknowledgedResponse.from_command_message(command, service_id)
                for service_id in service_ids
            ]

        system_controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )

        await system_controller._handle_finalize_artifacts_command(
            FinalizeArtifactsCommand(service_id="records_manager")
        )

        assert targeted == [["record_processor_0"]]
        assert system_controller._exit_errors == []

    @pytest.mark.asyncio
    async def test_finalize_failure_is_recorded_before_it_is_raised(
        self, system_controller: SystemController
    ) -> None:
        """The caller may drop the response; the exit code must not depend on it."""
        system_controller._is_kubernetes = MagicMock(return_value=False)
        system_controller.service_manager.service_id_map = {}

        with pytest.raises(RuntimeError):
            await system_controller._handle_finalize_artifacts_command(
                FinalizeArtifactsCommand(service_id="records_manager")
            )

        assert [e.operation for e in system_controller._exit_errors] == [
            "finalize_artifacts"
        ]


class TestRawFinalizeHonoursThePodLossTolerance:
    """H10: the finalize barrier and the abort threshold must agree."""

    @pytest.fixture
    def k8s_controller(self) -> SystemController:
        # Built field-by-field rather than from the shared fixture: the real
        # BenchmarkRun's artifacts config rejects assignment, and this barrier
        # only reads export_level and the pod topology.
        ctrl = SystemController.__new__(SystemController)
        ctrl.run = MagicMock()
        ctrl.run.cfg.artifacts.export_level = ExportLevel.RAW
        ctrl._is_kubernetes = MagicMock(return_value=True)
        ctrl._k8s_topology = MagicMock(num_worker_pods=4)
        ctrl.service_id = "system_controller"
        ctrl._exit_errors = []
        ctrl._export_failed = False
        ctrl._reaped_service_ids = set()
        ctrl.info = MagicMock()
        ctrl.warning = MagicMock()
        ctrl.error = MagicMock()
        return ctrl

    @pytest.mark.asyncio
    async def test_a_tolerated_pod_loss_still_finalizes_the_survivors(
        self, k8s_controller: SystemController
    ) -> None:
        """4 pods, 1 lost = 25% < the 50% abort threshold, so the run continues.

        Before the fix the exact-equality check failed the barrier without ever
        contacting the three healthy pods.
        """
        registered = [
            MagicMock(service_id=f"worker_group_manager_{i}") for i in range(3)
        ]
        targeted: list[list[str]] = []

        async def respond(command, service_ids, timeout):  # noqa: ANN001, ARG001
            targeted.append(list(service_ids))
            return [
                CommandAcknowledgedResponse.from_command_message(command, service_id)
                for service_id in service_ids
            ]

        k8s_controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=registered,
        ):
            await k8s_controller._finalize_kubernetes_raw_artifacts()

        assert targeted == [
            [
                "worker_group_manager_0",
                "worker_group_manager_1",
                "worker_group_manager_2",
            ]
        ]
        # Tolerated, but not silent: readiness is still published while the run
        # is marked degraded.
        assert k8s_controller._export_failed is False
        assert [e.operation for e in k8s_controller._exit_errors] == [
            "finalize_raw_artifacts_degraded"
        ]

    @pytest.mark.asyncio
    async def test_a_loss_beyond_the_threshold_still_fails_closed(
        self, k8s_controller: SystemController
    ) -> None:
        """4 pods, 3 lost = 75% >= threshold: withhold readiness, contact nobody."""
        k8s_controller.send_command_and_wait_for_all_responses = AsyncMock()
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=[MagicMock(service_id="worker_group_manager_0")],
        ):
            await k8s_controller._finalize_kubernetes_raw_artifacts()

        k8s_controller.send_command_and_wait_for_all_responses.assert_not_awaited()
        assert k8s_controller._export_failed is True
        assert [e.operation for e in k8s_controller._exit_errors] == [
            "finalize_raw_artifacts"
        ]

    @pytest.mark.asyncio
    async def test_losing_every_worker_group_fails_closed(
        self, k8s_controller: SystemController
    ) -> None:
        """A barrier with no members left proves nothing, at any threshold."""
        k8s_controller.send_command_and_wait_for_all_responses = AsyncMock()
        with (
            patch(
                "aiperf.controller.system_controller.ServiceRegistry.get_services",
                return_value=[],
            ),
            patch(
                "aiperf.common.environment.Environment.POD.FAILURE_ABORT_THRESHOLD_PERCENT",
                0.0,
            ),
        ):  # fmt: skip
            await k8s_controller._finalize_kubernetes_raw_artifacts()

        k8s_controller.send_command_and_wait_for_all_responses.assert_not_awaited()
        assert k8s_controller._export_failed is True


def _finalize_command() -> FinalizeArtifactsCommand:
    return FinalizeArtifactsCommand(
        service_id="records_manager",
        target_service_type=ServiceType.RECORD_PROCESSOR,
    )


def test_finalize_command_type_is_stable() -> None:
    """Guards the ACK matching in ``_raw_artifact_finalize_response_error``."""
    assert _finalize_command().command == CommandType.FINALIZE_ARTIFACTS
