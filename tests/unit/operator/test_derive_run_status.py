# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for derive_run_status — the per-epoch status normalizer."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.operator.routers.jobs import derive_run_status
from aiperf.operator.runs_index_models import RunIndexRow


def _row(
    *,
    epoch: str = "1714069400",
    phase: str = "Succeeded",
    error: str | None = None,
    is_latest: bool = True,
) -> RunIndexRow:
    return RunIndexRow(
        namespace="bench",
        job_id="j1",
        epoch=epoch,
        phase=phase,
        is_latest=is_latest,
        start_time=None,
        end_time=None,
        created_unix=0,
        mtime_epoch=0,
        error=error,
        model=None,
        endpoint=None,
        gpu_count=0,
        gpu_name=None,
        file_count=0,
        total_size_bytes=0,
        sweep_namespace=None,
        sweep_name=None,
        sweep_epoch=None,
        sweep_variation_idx=None,
    )


@pytest.mark.parametrize(
    "row, live_running_epoch, expected",
    [
        param(_row(epoch="100", phase="Running"), "100", "running",
              id="live-running-wins-over-phase"),
        param(_row(epoch="100", phase="Succeeded"), "100", "running",
              id="live-running-wins-even-when-index-stale"),
        param(_row(epoch="100", phase="Succeeded"), None, "succeeded",
              id="phase-succeeded-no-live"),
        param(_row(epoch="100", phase="Succeeded"), "999", "succeeded",
              id="phase-succeeded-different-live-epoch"),
        param(_row(epoch="100", phase="Failed"), None, "failed",
              id="phase-failed"),
        param(_row(epoch="100", phase="Cancelled"), None, "cancelled",
              id="phase-cancelled"),
        param(_row(epoch="100", phase="Succeeded", error="boom"), None, "failed",
              id="error-overrides-succeeded"),
        param(_row(epoch="100", phase="Pending"), None, "unknown",
              id="phase-pending-falls-to-unknown"),
        param(_row(epoch="100", phase=""), None, "unknown",
              id="empty-phase-falls-to-unknown"),
        param(_row(epoch="100", phase="SUCCEEDED"), None, "succeeded",
              id="phase-case-insensitive-uppercase"),
        param(_row(epoch="100", phase="failed"), None, "failed",
              id="phase-case-insensitive-lowercase"),
    ],
)  # fmt: skip
def test_derive_run_status(
    row: RunIndexRow, live_running_epoch: str | None, expected: str
) -> None:
    assert derive_run_status(row, live_running_epoch=live_running_epoch) == expected
