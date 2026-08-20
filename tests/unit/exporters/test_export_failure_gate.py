# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed export must not be advertised as a complete result set.

The results-ready marker is what tells the sidecar it may serve top-level
artifacts and the operator that the run is harvestable. Writing it after an
exporter failed publishes a truncated result set as authoritative -- an ENOSPC
or a partial write becomes a job marked Completed with artifacts missing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.models import ErrorDetails, ProcessRecordsResult, ProfileResults
from aiperf.controller.system_controller import SystemController
from aiperf.exporters.exporter_manager import ExporterFailure, ExporterManager


def _manager() -> ExporterManager:
    mgr = ExporterManager.__new__(ExporterManager)
    mgr._tasks = set()
    mgr._exporter_config = MagicMock()
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    return mgr


class TestExportFailureIsReported:
    @pytest.mark.asyncio
    async def test_local_failure_is_structured_and_blocks_readiness(self) -> None:
        mgr = _manager()

        class FailingExporter:
            async def export(self) -> None:
                raise OSError("No space left on device")

        failures = await mgr._run_data_exporters(
            [FailingExporter()],
            is_deferred=False,
        )

        assert len(failures) == 1
        assert failures[0].exporter == "FailingExporter"
        assert isinstance(failures[0].error, OSError)
        assert failures[0].is_deferred is False

    @pytest.mark.asyncio
    async def test_deferred_failure_retains_remote_upload_classification(self) -> None:
        mgr = _manager()

        class FailingUploader:
            async def export(self) -> None:
                raise RuntimeError("upload failed")

        failures = await mgr._run_data_exporters(
            [FailingUploader()],
            is_deferred=True,
        )

        assert len(failures) == 1
        assert failures[0].is_deferred is True

    @pytest.mark.asyncio
    async def test_phase_artifact_failure_is_marker_blocking(self) -> None:
        mgr = _manager()
        mgr._export_phase_metric_artifacts = AsyncMock(
            side_effect=OSError("phase export disk full")
        )

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[],
        ):
            failures = await mgr.export_data()

        assert len(failures) == 1
        assert failures[0].exporter == "PhaseMetricArtifacts"
        assert failures[0].is_deferred is False

    @pytest.mark.asyncio
    async def test_local_constructor_failure_is_marker_blocking(self) -> None:
        mgr = _manager()
        mgr._export_phase_metric_artifacts = AsyncMock()
        entry = MagicMock(name="broken-local-entry")
        entry.name = "broken_local"

        class BrokenLocalExporter:
            def __init__(self, **_: object) -> None:
                raise OSError("constructor could not open output")

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[(entry, BrokenLocalExporter)],
        ):
            failures = await mgr.export_data()

        assert len(failures) == 1
        assert failures[0].exporter == "broken_local"
        assert isinstance(failures[0].error, OSError)
        assert failures[0].is_deferred is False


class TestExportFailureSurface:
    """Only local export failures block readiness and force non-zero exit."""

    @staticmethod
    def _controller():
        from aiperf.controller.system_controller import SystemController

        ctrl = SystemController.__new__(SystemController)
        ctrl.service_id = "controller"
        ctrl._exit_errors = []
        ctrl.warning = MagicMock()
        return ctrl

    def test_local_failure_blocks_marker_and_records_exit_error(self) -> None:
        ctrl = self._controller()
        failure = ExporterFailure(
            exporter="MetricsJsonExporter",
            error=OSError("No space left on device"),
            is_deferred=False,
        )

        assert ctrl._surface_export_failures([failure]) is True
        assert len(ctrl._exit_errors) == 1
        assert ctrl._exit_errors[0].operation == "export:MetricsJsonExporter"

    def test_deferred_failure_does_not_block_local_results(self) -> None:
        ctrl = self._controller()
        failure = ExporterFailure(
            exporter="WandbDataExporter",
            error=RuntimeError("upload failed"),
            is_deferred=True,
        )

        assert ctrl._surface_export_failures([failure]) is False
        assert ctrl._exit_errors == []
        ctrl.warning.assert_called_once()


class TestProcessResultFailureSurface:
    @pytest.mark.asyncio
    async def test_processing_error_is_reported_without_withholding_results(
        self,
    ) -> None:
        """Aggregation diagnostics are reported, but never gate publication.

        ``results.errors`` is an aggregation-side diagnostic list, not a verdict
        on the export. Setting ``_export_failed`` from it meant a GPU-telemetry
        drain timeout or one malformed record withheld the results-ready marker
        and ResultsExportedMessage for a fully valid inference result set.
        """
        ctrl = SystemController.__new__(SystemController)
        ctrl.trace_or_debug = MagicMock()
        ctrl.error = MagicMock()
        ctrl.debug = MagicMock()
        ctrl._exit_errors = []
        ctrl._export_failed = False
        ctrl._profile_results = None
        ctrl._server_metrics_results = None
        ctrl._result_join_coordinator = MagicMock()
        ctrl._check_and_trigger_shutdown = AsyncMock()
        error = ErrorDetails(
            type="OSError",
            message="stream flush disk full",
            details={"stage": "stream_export_finalize"},
        )

        await ctrl._on_process_records_result_message(
            ProcessRecordsResultMessage(
                service_id="records-manager",
                results=ProcessRecordsResult(
                    results=ProfileResults(
                        records=[],
                        completed=0,
                        start_ns=1,
                        end_ns=2,
                    ),
                    errors=[error],
                ),
            )
        )

        assert ctrl._export_failed is False
        assert len(ctrl._exit_errors) == 1
        assert ctrl._exit_errors[0].operation == "process_records"
        assert ctrl._exit_errors[0].service_id == "records-manager"
        assert ctrl._exit_errors[0].error_details == error
        ctrl._check_and_trigger_shutdown.assert_awaited_once()

        ctrl.service_id = "system_controller"
        ctrl._is_kubernetes = MagicMock(return_value=True)
        ctrl._was_cancelled = False
        ctrl.run = MagicMock()
        ctrl.warning = MagicMock()
        with (
            patch(
                "aiperf.controller.system_controller.write_ready_marker"
            ) as write_ready_marker,
            patch(
                "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
                AsyncMock(),
            ),
        ):  # fmt: skip
            ctrl.publish = AsyncMock()
            await ctrl._announce_results_exported()

        write_ready_marker.assert_called_once()
        ctrl.publish.assert_awaited_once()
