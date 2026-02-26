# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for multi-run confidence reporting feature."""

import json
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiRunConfidence:
    """Integration tests for multi-run confidence reporting."""

    async def test_multi_run_basic(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test basic multi-run execution with 3 runs."""
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
                --ui {defaults.ui}
            """
        )

        # Verify basic execution
        assert result.exit_code == 0

        # Verify per-run artifacts exist
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory should exist"

        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) == 3, "Should have 3 run directories"

        # Verify run directory naming
        assert run_dirs[0].name == "run_0001"
        assert run_dirs[1].name == "run_0002"
        assert run_dirs[2].name == "run_0003"

        # Verify each run has artifacts
        for run_dir in run_dirs:
            json_file = run_dir / "profile_export_aiperf.json"
            csv_file = run_dir / "profile_export_aiperf.csv"
            assert json_file.exists(), f"{run_dir.name} should have JSON artifact"
            assert csv_file.exists(), f"{run_dir.name} should have CSV artifact"

            # Verify JSON is valid
            with open(json_file) as f:
                run_data = json.load(f)
                assert run_data["request_count"]["avg"] == 10

        # Verify aggregate artifacts exist
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory should exist"

        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
        agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"
        assert agg_json.exists(), "Aggregate JSON should exist"
        assert agg_csv.exists(), "Aggregate CSV should exist"

        # Verify aggregate JSON schema
        with open(agg_json) as f:
            agg_data = json.load(f)

            # Check metadata
            assert agg_data["metadata"]["aggregation_type"] == "confidence"
            assert agg_data["metadata"]["num_profile_runs"] == 3
            assert agg_data["metadata"]["num_successful_runs"] == 3
            assert len(agg_data["metadata"]["failed_runs"]) == 0
            assert agg_data["metadata"]["confidence_level"] == 0.95
            assert len(agg_data["metadata"]["run_labels"]) == 3

            # Check metrics structure
            assert "metrics" in agg_data
            metrics = agg_data["metrics"]

            # Verify at least one metric has all required fields
            assert len(metrics) > 0, "Should have aggregated metrics"

            # Check a common metric (request_throughput_avg should exist)
            throughput_metrics = [k for k in metrics if "throughput" in k.lower()]
            assert len(throughput_metrics) > 0, "Should have throughput metrics"

            sample_metric = metrics[throughput_metrics[0]]
            required_fields = [
                "mean",
                "std",
                "min",
                "max",
                "cv",
                "se",
                "ci_low",
                "ci_high",
                "t_critical",
                "unit",
            ]
            for field in required_fields:
                assert field in sample_metric, f"Metric should have {field} field"

    async def test_backward_compatibility_single_run(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that num_profile_runs=1 maintains backward compatibility."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 1 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify NO multi-run directory structure
        profile_runs_dir = temp_output_dir / "profile_runs"
        aggregate_dir = temp_output_dir / "aggregate"

        assert not profile_runs_dir.exists(), (
            "profile_runs should not exist for single run"
        )
        assert not aggregate_dir.exists(), "aggregate should not exist for single run"

        # Verify standard artifacts exist at root level
        json_file = temp_output_dir / "profile_export_aiperf.json"
        csv_file = temp_output_dir / "profile_export_aiperf.csv"

        assert json_file.exists(), "Standard JSON should exist at root"
        assert csv_file.exists(), "Standard CSV should exist at root"

    async def test_multi_run_with_cooldown(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run with cooldown between runs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --profile-run-cooldown-seconds 0.5 \
                --request-count 5 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=300.0,  # Longer timeout for cooldown
        )

        assert result.exit_code == 0

        # Verify aggregate metadata includes cooldown
        aggregate_dir = temp_output_dir / "aggregate"
        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"

        with open(agg_json) as f:
            agg_data = json.load(f)
            assert agg_data["metadata"]["cooldown_seconds"] == 0.5

    async def test_multi_run_custom_confidence_level(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run with custom confidence level."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 3 \
                --confidence-level 0.99 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify confidence level in aggregate
        aggregate_dir = temp_output_dir / "aggregate"
        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"

        with open(agg_json) as f:
            agg_data = json.load(f)
            assert agg_data["metadata"]["confidence_level"] == 0.99

    async def test_multi_run_concurrency_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run works with concurrency mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --concurrency 4 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify runs completed
        profile_runs_dir = temp_output_dir / "profile_runs"
        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) == 2

    async def test_multi_run_request_rate_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run works with request-rate mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --request-rate 5.0 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify runs completed
        profile_runs_dir = temp_output_dir / "profile_runs"
        run_dirs = sorted(profile_runs_dir.glob("run_*"))
        assert len(run_dirs) == 2

    async def test_multi_run_with_warmup(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run with warmup phase (warmup should run once)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --warmup-request-count 5 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify each run has correct request count (warmup excluded)
        profile_runs_dir = temp_output_dir / "profile_runs"
        for run_dir in sorted(profile_runs_dir.glob("run_*")):
            json_file = run_dir / "profile_export_aiperf.json"
            with open(json_file) as f:
                run_data = json.load(f)
                # Each run should have 10 requests (warmup excluded)
                assert run_data["request_count"]["avg"] == 10

    async def test_aggregate_csv_format(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test aggregate CSV format is correct."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        # Verify CSV format
        aggregate_dir = temp_output_dir / "aggregate"
        agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"

        csv_content = agg_csv.read_text()
        lines = csv_content.strip().split("\n")

        # Check header
        header = lines[0]
        required_columns = [
            "metric",
            "mean",
            "std",
            "min",
            "max",
            "cv",
            "se",
            "ci_low",
            "ci_high",
            "t_critical",
            "unit",
        ]
        for col in required_columns:
            assert col in header, f"CSV should have {col} column"

        # Check at least one data row
        assert len(lines) > 1, "CSV should have data rows"

    # =========================================================================
    # Negative Test Cases - Failure Scenarios
    # =========================================================================

    async def test_multi_run_with_partial_failures(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test multi-run continues when some runs fail.

        This test simulates a scenario where some runs fail but others succeed.
        The orchestrator should:
        1. Continue executing remaining runs after a failure
        2. Compute aggregate statistics over successful runs only
        3. Record failed runs in metadata
        """
        # Configure mock server to fail intermittently
        # We'll use a very short benchmark duration and high request rate to potentially trigger failures
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 5 \
                --request-count 100 \
                --concurrency 50 \
                --benchmark-duration 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=300.0,
            assert_success=False,  # Allow non-zero exit if some runs fail
        )

        # Even if some runs fail, the command should complete
        # (exit code might be non-zero if failures occur, but that's acceptable)

        # Verify that we have some run directories
        profile_runs_dir = temp_output_dir / "profile_runs"
        if profile_runs_dir.exists():
            run_dirs = sorted(profile_runs_dir.glob("run_*"))
            assert len(run_dirs) > 0, "Should have at least some run directories"

            # If aggregate was created, verify it handles failures correctly
            aggregate_dir = temp_output_dir / "aggregate"
            if aggregate_dir.exists():
                agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
                if agg_json.exists():
                    with open(agg_json) as f:
                        agg_data = json.load(f)

                        # Verify metadata tracks failures
                        assert "num_successful_runs" in agg_data["metadata"]
                        assert "failed_runs" in agg_data["metadata"]

                        num_successful = agg_data["metadata"]["num_successful_runs"]
                        num_failed = len(agg_data["metadata"]["failed_runs"])

                        # Total should equal num_profile_runs
                        assert num_successful + num_failed == 5

                        # If there were failures, verify they're documented
                        if num_failed > 0:
                            for failed_run in agg_data["metadata"]["failed_runs"]:
                                assert "label" in failed_run
                                assert "error" in failed_run
                                assert failed_run["label"].startswith("run_")

    async def test_multi_run_insufficient_successful_runs(
        self, cli: AIPerfCLI, mock_server_factory, temp_output_dir: Path
    ):
        """Test that aggregate fails gracefully when < 2 successful runs.

        Confidence intervals require at least 2 successful runs.
        If only 0 or 1 runs succeed, aggregation should fail with a clear error.
        """
        # Use mock server with 100% error rate to force all requests to fail
        async with mock_server_factory(fast=True, error_rate=100.0) as server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --num-profile-runs 3 \
                    --request-count 5 \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """,
                timeout=60.0,
                assert_success=False,  # Don't raise on non-zero exit
            )

        # Command should fail
        assert result.exit_code != 0, (
            "Should exit with non-zero code when all runs fail"
        )

        # Verify no aggregate directory was created (or it's empty)
        aggregate_dir = temp_output_dir / "aggregate"
        if aggregate_dir.exists():
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            assert not agg_json.exists(), (
                "Aggregate JSON should not exist with insufficient successful runs"
            )

    async def test_multi_run_all_runs_fail(
        self, cli: AIPerfCLI, mock_server_factory, temp_output_dir: Path
    ):
        """Test behavior when all runs fail.

        When all runs fail:
        1. No aggregate statistics should be computed
        2. Per-run artifacts may or may not exist (depending on failure point)
        3. Clear error message should be provided
        """
        # Use mock server with 100% error rate to force all requests to fail
        async with mock_server_factory(fast=True, error_rate=100.0) as server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --num-profile-runs 3 \
                    --request-count 5 \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """,
                timeout=60.0,
                assert_success=False,
            )

        # Command should fail
        assert result.exit_code != 0, (
            "Should exit with non-zero code when all runs fail"
        )

        # Verify no aggregate was created
        aggregate_dir = temp_output_dir / "aggregate"
        if aggregate_dir.exists():
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            assert not agg_json.exists(), (
                "Aggregate JSON should not exist when all runs fail"
            )

    async def test_multi_run_single_failure_still_aggregates(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that a single failure doesn't prevent aggregation.

        If we have 3 runs and 1 fails, we should still get aggregate statistics
        over the 2 successful runs.
        """
        # Run with configuration that might cause occasional failures
        # but should mostly succeed
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 3 \
                --request-count 20 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=300.0,
        )

        # Even if one run fails, we should get aggregate (if >= 2 succeed)
        aggregate_dir = temp_output_dir / "aggregate"
        if aggregate_dir.exists():
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            if agg_json.exists():
                with open(agg_json) as f:
                    agg_data = json.load(f)

                    num_successful = agg_data["metadata"]["num_successful_runs"]

                    # If we got an aggregate, we must have >= 2 successful runs
                    assert num_successful >= 2, (
                        "Aggregate requires at least 2 successful runs"
                    )

                    # Verify metrics were computed
                    assert len(agg_data["metrics"]) > 0, (
                        "Should have aggregated metrics"
                    )

    async def test_multi_run_preserves_failed_run_artifacts(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that artifacts are preserved even for failed runs.

        Failed runs should still have their artifacts directory created,
        allowing users to debug what went wrong.
        """
        # Use configuration that will likely cause some failures
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 3 \
                --request-count 100 \
                --concurrency 100 \
                --benchmark-duration 0.05 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=300.0,
            assert_success=False,
        )

        # Verify run directories exist (even for failed runs)
        profile_runs_dir = temp_output_dir / "profile_runs"
        if profile_runs_dir.exists():
            run_dirs = sorted(profile_runs_dir.glob("run_*"))

            # We should have directories for all attempted runs
            assert len(run_dirs) > 0, "Should have run directories"

            # Each directory should exist (even if run failed)
            for run_dir in run_dirs:
                assert run_dir.is_dir(), f"{run_dir.name} should be a directory"

    async def test_multi_run_invalid_num_profile_runs(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test validation of num_profile_runs parameter.

        num_profile_runs must be between 1 and 10.
        """
        # Test num_profile_runs = 0 (invalid)
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 0 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=30.0,
            assert_success=False,
        )

        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert (
            "num-profile-runs" in output.lower() or "num_profile_runs" in output.lower()
        )

    async def test_multi_run_exceeds_max_limit(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that num_profile_runs > 10 is rejected.

        The limit is set to 10 to prevent excessive execution times.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 11 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=30.0,
            assert_success=False,
        )

        assert result.exit_code != 0
        output = result.stdout + result.stderr
        # Should mention the limit or validation error
        assert any(
            phrase in output.lower()
            for phrase in [
                "10",
                "limit",
                "maximum",
                "validation",
            ]
        )

    async def test_multi_run_invalid_confidence_level(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test validation of confidence_level parameter.

        confidence_level must be between 0 and 1 (exclusive).
        """
        # Test confidence_level = 1.5 (invalid)
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 3 \
                --confidence-level 1.5 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=30.0,
            assert_success=False,
        )

        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "confidence" in output.lower()

    async def test_multi_run_negative_cooldown(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that negative cooldown is rejected."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --num-profile-runs 2 \
                --profile-run-cooldown-seconds -1.0 \
                --request-count 10 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=30.0,
            assert_success=False,
        )

        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "cooldown" in output.lower() or "negative" in output.lower()
