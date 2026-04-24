# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DistributionConvergence criterion.

Feature: adaptive-sweep-and-detailed-aggregation
Property 5: DistributionConvergence same-distribution convergence
"""

import numpy as np
import orjson

from aiperf.orchestrator.convergence.distribution import DistributionConvergence
from aiperf.orchestrator.models import RunResult


class TestDistributionConvergence:
    """Tests for DistributionConvergence.is_converged."""

    def test_same_distribution_converged(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is True

    def test_different_distributions_not_converged(
        self, make_results_with_jsonl
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [
            rng.normal(100, 5, size=200),
            rng.normal(100, 5, size=200),
            rng.normal(200, 5, size=200),
        ]
        results = make_results_with_jsonl(runs)

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_fewer_than_min_runs_returns_false(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(2)]
        results = make_results_with_jsonl(runs)

        criterion = DistributionConvergence(
            metric="time_to_first_token", min_runs=3, p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_missing_jsonl_for_one_run_skips(
        self, make_results_with_jsonl, tmp_path
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        # Remove JSONL from the last run (distribution B becomes empty)
        jsonl_path = results[-1].artifacts_path / "profile_export.jsonl"
        jsonl_path.unlink()

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_missing_jsonl_for_middle_run_still_works(
        self, make_results_with_jsonl
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(4)]
        results = make_results_with_jsonl(runs)

        # Remove JSONL from a middle run; dist A still has runs 0 and 2, dist B has run 3
        jsonl_path = results[1].artifacts_path / "profile_export.jsonl"
        jsonl_path.unlink()

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is True

    def test_metric_not_in_jsonl_returns_false(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs, metric="time_to_first_token")

        criterion = DistributionConvergence(
            metric="nonexistent_metric", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_distribution_a_fewer_than_2_points(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        # First two runs have 1 point each (dist A = 2 points total), last run has 200
        runs = [
            rng.normal(100, 5, size=1),
            rng.normal(100, 5, size=1),
            rng.normal(100, 5, size=200),
        ]
        results = make_results_with_jsonl(runs)

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        # dist A has exactly 2 points, dist B has 200 -> should work
        assert criterion.is_converged(results) is True

    def test_distribution_b_fewer_than_2_points(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [
            rng.normal(100, 5, size=200),
            rng.normal(100, 5, size=200),
            rng.normal(100, 5, size=1),
        ]
        results = make_results_with_jsonl(runs)

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_distribution_a_empty_returns_false(
        self, make_results_with_jsonl, tmp_path
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        # Remove JSONL from all runs except the last
        for r in results[:-1]:
            (r.artifacts_path / "profile_export.jsonl").unlink()

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_all_failed_runs_returns_false(self) -> None:
        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        results = [RunResult(label=f"run_{i:04d}", success=False) for i in range(5)]
        assert criterion.is_converged(results) is False

    def test_successful_runs_without_artifacts_path_excluded(
        self, make_results_with_jsonl
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        # Set artifacts_path to None on all runs
        for r in results:
            r.artifacts_path = None

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_exactly_at_p_value_threshold_not_converged(
        self, make_results_with_jsonl
    ) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        # Use p_value_threshold=1.0 so any p_value > 1.0 is impossible -> not converged
        # Actually p_value is always <= 1.0, so threshold=1.0 means p > 1.0 never true
        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=1.0
        )
        assert criterion.is_converged(results) is False

    def test_empty_jsonl_files_returns_false(self, make_results_with_jsonl) -> None:
        rng = np.random.default_rng(seed=42)
        runs = [rng.normal(100, 5, size=200) for _ in range(3)]
        results = make_results_with_jsonl(runs)

        # Truncate all JSONL files to empty
        for r in results:
            (r.artifacts_path / "profile_export.jsonl").write_bytes(b"")

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        assert criterion.is_converged(results) is False

    def test_warmup_records_excluded(self, tmp_path) -> None:
        """Verify that warmup-phase records are filtered out by _load_request_metrics."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        records = []
        rng = np.random.default_rng(seed=42)
        # 5 warmup records
        for val in rng.normal(100, 5, size=5):
            records.append(
                {
                    "metadata": {"benchmark_phase": "warmup"},
                    "metrics": {
                        "time_to_first_token": {"value": float(val), "unit": "ms"}
                    },
                    "error": None,
                }
            )
        # 200 profiling records
        for val in rng.normal(100, 5, size=200):
            records.append(
                {
                    "metadata": {"benchmark_phase": "profiling"},
                    "metrics": {
                        "time_to_first_token": {"value": float(val), "unit": "ms"}
                    },
                    "error": None,
                }
            )

        with open(jsonl_path, "wb") as f:
            for rec in records:
                f.write(orjson.dumps(rec))
                f.write(b"\n")

        criterion = DistributionConvergence(
            metric="time_to_first_token", p_value_threshold=0.05
        )
        values = criterion._load_request_metrics(run_dir, "time_to_first_token")
        assert len(values) == 200
