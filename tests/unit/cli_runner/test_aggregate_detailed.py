# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for detailed aggregation wiring in ``cli_runner``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.cli_runner._aggregate import _maybe_compute_detailed, aggregate_and_export
from aiperf.orchestrator.models import RunResult


def _write_profile_record(run_dir: Path, metric_value: float) -> None:
    run_dir.mkdir()
    record = {
        "metadata": {"benchmark_phase": "profiling"},
        "metrics": {"request_latency": {"value": metric_value, "unit": "ms"}},
        "error": None,
    }
    (run_dir / "profile_export.jsonl").write_bytes(orjson.dumps(record) + b"\n")


@pytest.mark.asyncio
async def test_maybe_compute_detailed_uses_default_jsonl_filename(
    tmp_path: Path,
) -> None:
    """Regression: ``export_jsonl_file=None`` must not pass an empty filename.

    ``Path(run_dir) / ""`` resolves to ``run_dir`` itself, causing the JSONL
    loader to open the run directory as a file and skip collated metrics.
    """
    run_dir = tmp_path / "run_0001"
    await asyncio.to_thread(_write_profile_record, run_dir, 12.5)

    plan = MagicMock()
    plan.use_adaptive = True
    plan.export_jsonl_file = None
    plan.cooldown_seconds = 0.0
    plan.confidence_level = 0.95
    plan.is_sweep = False
    base_config = MagicMock()
    base_config.scenario = None
    plan.configs = [base_config]

    results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]

    result = await asyncio.to_thread(_maybe_compute_detailed, plan, results)

    assert result is not None
    metric = result.metrics["request_latency"]
    assert metric["combined"]["count"] == 1
    assert metric["combined"]["mean"] == pytest.approx(12.5)

    strategy = MagicMock()
    strategy.get_aggregate_path.return_value = tmp_path / "aggregate"

    await aggregate_and_export(
        results,
        plan,
        strategy=strategy,
        base_dir=tmp_path,
        logger=MagicMock(),
    )

    collated_path = tmp_path / "aggregate" / "profile_export_aiperf_collated.json"
    assert collated_path.exists()
    collated = json.loads(await asyncio.to_thread(collated_path.read_text))
    assert collated["metrics"]["request_latency"]["combined"]["count"] == 1
