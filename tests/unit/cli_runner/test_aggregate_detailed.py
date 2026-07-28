# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for detailed aggregation wiring in ``cli_runner``."""

from __future__ import annotations

from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.cli_runner._aggregate import _maybe_compute_detailed
from aiperf.orchestrator.models import RunResult


def _write_profile_record(run_dir, metric_value: float) -> None:
    run_dir.mkdir()
    record = {
        "metadata": {"benchmark_phase": "profiling"},
        "metrics": {"request_latency": {"value": metric_value, "unit": "ms"}},
        "error": None,
    }
    (run_dir / "profile_export.jsonl").write_bytes(orjson.dumps(record) + b"\n")


def test_maybe_compute_detailed_uses_default_jsonl_filename(tmp_path) -> None:
    """Regression: ``export_jsonl_file=None`` must not pass an empty filename.

    ``Path(run_dir) / ""`` resolves to ``run_dir`` itself, causing the JSONL
    loader to open the run directory as a file and skip collated metrics.
    """
    run_dir = tmp_path / "run_0001"
    _write_profile_record(run_dir, 12.5)

    plan = MagicMock()
    plan.use_adaptive = True
    plan.export_jsonl_file = None
    plan.cooldown_seconds = 0.0

    result = _maybe_compute_detailed(
        plan,
        [RunResult(label="run_0001", success=True, artifacts_path=run_dir)],
    )

    assert result is not None
    metric = result.metrics["request_latency"]
    assert metric["combined"]["count"] == 1
    assert metric["combined"]["mean"] == pytest.approx(12.5)
