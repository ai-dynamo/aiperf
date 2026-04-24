# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for aggregation tests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import orjson
import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult


@pytest.fixture
def make_results_with_jsonl(
    tmp_path: Path,
) -> Callable[..., list[RunResult]]:
    """Write JSONL to tmp_path, return list[RunResult] with artifacts_path pointing to real files.

    Args:
        run_latencies: List of numpy arrays, one per run. Each array contains
            per-request metric values for that run.
        metric: Metric name to write into JSONL records.

    Returns:
        Factory function producing list[RunResult] backed by real JSONL files.
    """

    def _factory(
        run_latencies: list[np.ndarray],
        metric: str = "time_to_first_token",
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for i, latencies in enumerate(run_latencies):
            run_dir = tmp_path / f"run_{i + 1:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = run_dir / "profile_export.jsonl"

            with open(jsonl_path, "wb") as f:
                for j, val in enumerate(latencies):
                    record = {
                        "metadata": {
                            "session_num": j,
                            "benchmark_phase": "profiling",
                            "request_start_ns": 0,
                            "request_end_ns": 1,
                            "worker_id": "worker_0",
                            "record_processor_id": "rp_0",
                        },
                        "metrics": {
                            metric: {"value": float(val), "unit": "ms"},
                        },
                        "error": None,
                    }
                    f.write(orjson.dumps(record))
                    f.write(b"\n")

            avg_val = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
            results.append(
                RunResult(
                    label=f"run_{i + 1:04d}",
                    success=True,
                    summary_metrics={
                        metric: JsonMetricResult(unit="ms", avg=avg_val),
                    },
                    artifacts_path=run_dir,
                )
            )
        return results

    return _factory
