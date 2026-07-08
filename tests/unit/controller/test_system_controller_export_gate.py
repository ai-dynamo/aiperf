# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-integrity gate: export failures must not become a lying ready marker.

Regression: when the disk fills mid-export the JSON exporter leaves a
truncated ``profile_export_aiperf.json`` and raises OSError(ENOSPC). The old
``_export_results_data`` gathered exporter tasks with ``return_exceptions=True``
(swallowing the error) and then UNCONDITIONALLY wrote the K8s results-ready
marker ``{ready: true}``. The results-sidecar then served the truncated file
and the operator marked the run Complete over corrupt artifacts.

The fix surfaces per-exporter failures from ``ExporterManager.export_data()``
and gates ``write_ready_marker`` on export success.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.exporters.exporter_manager import ExporterFailure
from aiperf.plugin.enums import ServiceRunType


def _local_failure() -> ExporterFailure:
    return ExporterFailure(
        exporter="MetricsJsonExporter",
        error=OSError(28, "No space left on device"),
        is_deferred=False,
    )


def _deferred_failure() -> ExporterFailure:
    return ExporterFailure(
        exporter="WandbDataExporter",
        error=RuntimeError("wandb upload failed"),
        is_deferred=True,
    )


class TestSurfaceExportFailures:
    """``_surface_export_failures`` classifies which failures block the marker."""

    def test_local_failure_blocks_marker_and_records_exit_error(
        self, system_controller
    ) -> None:
        blocking = system_controller._surface_export_failures([_local_failure()])

        assert blocking is True
        assert len(system_controller._exit_errors) == 1
        exit_error = system_controller._exit_errors[0]
        assert exit_error.operation == "export:MetricsJsonExporter"

    def test_deferred_failure_does_not_block_marker(self, system_controller) -> None:
        blocking = system_controller._surface_export_failures([_deferred_failure()])

        assert blocking is False
        assert system_controller._exit_errors == []

    def test_clean_export_does_not_block_marker(self, system_controller) -> None:
        assert system_controller._surface_export_failures([]) is False
        assert system_controller._exit_errors == []


class TestExportResultsMarkerGate:
    """``_export_results_data`` must withhold the ready marker on failure."""

    def _prime_controller(self, controller, tmp_path) -> None:
        controller.run.cfg.runtime.service_run_type = ServiceRunType.KUBERNETES
        controller.run.cfg.artifacts.dir = tmp_path
        controller.service_manager = AsyncMock()
        controller._compute_cross_input_analyzers = MagicMock()
        controller._profile_results = MagicMock()

    @pytest.mark.asyncio
    async def test_marker_withheld_when_local_exporter_fails(
        self, system_controller, tmp_path
    ) -> None:
        self._prime_controller(system_controller, tmp_path)
        fake_manager = MagicMock()
        fake_manager.export_data = AsyncMock(return_value=[_local_failure()])

        with (
            patch(
                "aiperf.controller.system_controller.ExporterManager",
                return_value=fake_manager,
            ),
            patch(
                "aiperf.kubernetes.results_sidecar.write_ready_marker"
            ) as mock_write_marker,
        ):
            await system_controller._export_results_data()

        mock_write_marker.assert_not_called()
        assert system_controller._exit_errors, "export failure must surface"
        assert system_controller._results_exported is True

    @pytest.mark.asyncio
    async def test_marker_written_when_export_succeeds(
        self, system_controller, tmp_path
    ) -> None:
        self._prime_controller(system_controller, tmp_path)
        fake_manager = MagicMock()
        fake_manager.export_data = AsyncMock(return_value=[])

        with (
            patch(
                "aiperf.controller.system_controller.ExporterManager",
                return_value=fake_manager,
            ),
            patch(
                "aiperf.kubernetes.results_sidecar.write_ready_marker"
            ) as mock_write_marker,
        ):
            await system_controller._export_results_data()

        mock_write_marker.assert_called_once()
        assert system_controller._exit_errors == []

    @pytest.mark.asyncio
    async def test_marker_written_when_only_deferred_upload_fails(
        self, system_controller, tmp_path
    ) -> None:
        """A remote-upload (wandb/mlflow) outage leaves local artifacts intact,
        so the local results-ready marker is still written."""
        self._prime_controller(system_controller, tmp_path)
        fake_manager = MagicMock()
        fake_manager.export_data = AsyncMock(return_value=[_deferred_failure()])

        with (
            patch(
                "aiperf.controller.system_controller.ExporterManager",
                return_value=fake_manager,
            ),
            patch(
                "aiperf.kubernetes.results_sidecar.write_ready_marker"
            ) as mock_write_marker,
        ):
            await system_controller._export_results_data()

        mock_write_marker.assert_called_once()
        assert system_controller._exit_errors == []
