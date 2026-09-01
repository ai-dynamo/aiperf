# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed export must not be advertised as a complete result set.

The results-ready marker is what tells the sidecar it may serve top-level
artifacts and the operator that the run is harvestable. Writing it after an
exporter failed publishes a truncated result set as authoritative -- an ENOSPC
or a partial write becomes a job marked Completed with artifacts missing.
"""

import asyncio
import gc
import warnings
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
        ctrl._failed_exporters = []
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

    def test_deferred_failure_records_an_exit_error(self) -> None:
        """Remote-upload CI must fail even when local artifacts are intact."""
        ctrl = self._controller()
        failure = ExporterFailure(
            exporter="WandbDataExporter",
            error=RuntimeError("upload failed"),
            is_deferred=True,
        )

        assert ctrl._surface_export_failures([failure]) is True
        assert len(ctrl._exit_errors) == 1
        assert ctrl._exit_errors[0].operation == "export:WandbDataExporter"

    def test_phase_artifact_failure_does_not_record_an_exit_error(self) -> None:
        """A supplemental phase artifact cannot invalidate root exports."""
        ctrl = self._controller()
        failure = ExporterFailure(
            exporter="PhaseMetricArtifacts:warmup:metrics_csv",
            error=OSError("phase disk full"),
            is_deferred=False,
            is_exit_failure=False,
        )

        assert ctrl._surface_export_failures([failure]) is False
        assert ctrl._exit_errors == []
        ctrl.warning.assert_called_once()


class TestPartialLocalFailureStillPublishesReadyMarker:
    """A failed exporter must not black-hole a sibling exporter's artifact.

    Before this fix, any non-deferred exporter failure set ``_export_failed``,
    which made ``_announce_results_exported`` return before ever calling
    ``write_ready_marker`` -- withholding readiness for the ENTIRE run even
    when another exporter (e.g. JSON) already wrote a valid artifact to disk.
    """

    @staticmethod
    def _controller() -> SystemController:
        ctrl = SystemController.__new__(SystemController)
        ctrl.service_id = "system_controller"
        ctrl._exit_errors = []
        ctrl._export_failed = False
        ctrl._failed_exporters = []
        ctrl.warning = MagicMock()
        ctrl.error = MagicMock()
        ctrl._was_cancelled = False
        ctrl.run = MagicMock()
        ctrl.publish = AsyncMock()
        return ctrl

    @pytest.mark.asyncio
    async def test_partial_local_failure_still_writes_ready_marker_with_failed_exporters(
        self,
    ) -> None:
        ctrl = self._controller()
        failure = ExporterFailure(
            exporter="MetricsCsvExporter",
            error=OSError("No space left on device"),
            is_deferred=False,
        )
        # Only the CSV exporter failed; the JSON exporter (not represented in
        # `failures`, since export_data() only returns failures) succeeded.
        marker_blocking = ctrl._surface_export_failures([failure])
        assert marker_blocking is True

        with (
            patch(
                "aiperf.controller.system_controller.write_ready_marker"
            ) as write_ready_marker,
            patch(
                "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
                AsyncMock(),
            ),
        ):  # fmt: skip
            await ctrl._announce_results_exported()

        write_ready_marker.assert_called_once()
        _, kwargs = write_ready_marker.call_args
        assert kwargs.get("partial") is True
        assert kwargs.get("failed_exporters") == ["MetricsCsvExporter"]


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
        # The results-ready marker asserted below is a Kubernetes artifact, so
        # this exercises the operator path, where aggregation diagnostics do
        # reach _exit_errors. Locally they stay log-only; see
        # tests/unit/controller/test_advisory_record_diagnostics.py.
        ctrl._is_kubernetes = MagicMock(return_value=True)
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
        ctrl._was_cancelled = False
        ctrl._failed_exporters = []
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


class TestPhaseArtifactCoroutinesAreNotAbandoned:
    """A BaseException escaping _export_one_phase must not orphan its writers."""

    @pytest.mark.asyncio
    async def test_cancellation_leaves_no_unawaited_artifact_coroutines(self) -> None:
        """The per-artifact handler catches Exception, so CancelledError unwinds.

        Building all four writer coroutines up front meant the ones the loop
        had not reached yet were discarded without ever being awaited: silently
        abandoned work, plus a "coroutine was never awaited" RuntimeWarning
        pointing at the exporter rather than at the cancellation. Constructing
        each coroutine only as it is about to be awaited makes that impossible.
        """
        mgr = _manager()
        mgr._run = MagicMock()
        mgr._results = MagicMock(
            start_ns=0, end_ns=1, is_complete=True, incomplete_reason=None
        )
        started: list[str] = []

        async def _record_and_maybe_cancel(**kwargs) -> None:
            key = kwargs["manifest_key"]
            started.append(key)
            if len(started) == 1:
                raise asyncio.CancelledError

        async def _observability(**kwargs) -> None:
            started.append(kwargs["manifest_key"])

        mgr._write_phase_export = _record_and_maybe_cancel
        mgr._write_phase_observability_export = _observability

        phase_result = MagicMock(
            phase_name="warmup",
            records=[],
            start_ns=0,
            end_ns=1,
            was_cancelled=False,
            successful_request_count=0,
            error_request_count=0,
            error_summary=[],
            branch_stats=None,
        )

        with (
            warnings.catch_warnings(record=True) as caught,
            patch("asyncio.to_thread", new=AsyncMock()),
            pytest.raises(asyncio.CancelledError),
        ):
            warnings.simplefilter("always")
            await mgr._export_one_phase(
                phase_result=phase_result,
                manifest_entry={"total_request_count": 0},
            )
            gc.collect()

        gc.collect()
        unawaited = [w for w in caught if "never awaited" in str(w.message)]
        assert not unawaited, (
            f"orphaned coroutines: {[str(w.message) for w in unawaited]}"
        )
        assert started == ["metrics_json"], (
            "writers after the cancellation must never be constructed at all"
        )
