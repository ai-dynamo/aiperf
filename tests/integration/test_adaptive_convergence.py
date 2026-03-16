# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for adaptive convergence and detailed aggregation."""

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestAdaptiveConvergence:
    """Integration tests for adaptive convergence with early stopping."""

    async def test_adaptive_ci_width_stops_early(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test adaptive strategy stops before max_runs when mock server returns stable metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 5 \
                --convergence-metric time_to_first_token \
                --convergence-mode ci_width \
                --convergence-threshold 0.20 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui none
            """,
            timeout=300.0,
        )

        assert result.exit_code == 0

        # Verify per-run artifacts exist
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory should exist"

        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) >= 2, (
            "Should have at least 2 run directories (min_runs floor)"
        )
        assert len(run_dirs) <= 5, (
            "Should have at most 5 run directories (max_runs cap)"
        )

        # Verify each run has artifacts
        for run_dir in run_dirs:
            json_file = run_dir / "profile_export_aiperf.json"
            assert json_file.exists(), f"{run_dir.name} should have JSON artifact"

        # Verify aggregate directory exists
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory should exist"

        # Verify confidence aggregate JSON
        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
        assert agg_json.exists(), "Confidence aggregate JSON should exist"

        with open(agg_json, "rb") as f:
            agg_data = orjson.loads(f.read())
            assert agg_data["metadata"]["aggregation_type"] == "confidence"
            assert agg_data["metadata"]["num_successful_runs"] >= 2
            assert "metrics" in agg_data
            assert len(agg_data["metrics"]) > 0

        # Verify detailed aggregate JSON
        detailed_json = aggregate_dir / "profile_export_aiperf_collated.json"
        assert detailed_json.exists(), "Collated aggregate JSON should exist"

        with open(detailed_json, "rb") as f:
            detailed_data = orjson.loads(f.read())
            assert detailed_data["metadata"]["aggregation_type"] == "detailed"
            assert detailed_data["metadata"]["num_successful_runs"] >= 2
            assert "metrics" in detailed_data

            # Verify detailed metrics schema
            metrics = detailed_data["metrics"]
            if len(metrics) > 0:
                sample_metric = next(iter(metrics.values()))
                assert "combined" in sample_metric
                combined = sample_metric["combined"]
                for field in ["mean", "std", "p50", "p90", "p95", "p99", "count"]:
                    assert field in combined, f"Combined stats should have {field}"
                assert "per_run" in sample_metric
                assert len(sample_metric["per_run"]) >= 2

    async def test_backward_compat_no_convergence_flags(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that multi-run without convergence flags uses FixedTrialsStrategy."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 3 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui none
            """,
            timeout=300.0,
        )

        assert result.exit_code == 0

        # Verify exactly 3 runs (fixed, no early stopping)
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists()
        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) == 3, "FixedTrialsStrategy should run exactly 3 times"

        # Verify only confidence aggregate exists (no detailed)
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists()

        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
        assert agg_json.exists(), "Confidence aggregate should exist"

        detailed_json = aggregate_dir / "profile_export_aiperf_collated.json"
        assert not detailed_json.exists(), (
            "Collated aggregate should NOT exist without convergence flags"
        )

    async def test_adaptive_cv_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test adaptive convergence with CV mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 5 \
                --convergence-metric time_to_first_token \
                --convergence-mode cv \
                --convergence-threshold 0.20 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui none
            """,
            timeout=300.0,
        )

        assert result.exit_code == 0

        profile_runs_dir = temp_output_dir / "profile_runs"
        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) >= 2
        assert len(run_dirs) <= 5

        # Verify both aggregation outputs
        aggregate_dir = temp_output_dir / "aggregate"
        assert (aggregate_dir / "profile_export_aiperf_aggregate.json").exists()
        assert (aggregate_dir / "profile_export_aiperf_collated.json").exists()

    async def test_adaptive_request_rate_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test adaptive convergence works with request-rate benchmarking mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 5 \
                --convergence-metric time_to_first_token \
                --convergence-mode ci_width \
                --convergence-threshold 0.20 \
                --request-rate 5.0 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui none
            """,
            timeout=300.0,
        )

        assert result.exit_code == 0

        profile_runs_dir = temp_output_dir / "profile_runs"
        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) >= 2
        assert len(run_dirs) <= 5

    async def test_convergence_metric_without_multi_run_fails(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that --convergence-metric without --num-profile-runs > 1 raises error."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --convergence-metric time_to_first_token \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui none
            """,
            timeout=30.0,
            assert_success=False,
        )

        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "convergence" in output.lower() or "num-profile-runs" in output.lower()
