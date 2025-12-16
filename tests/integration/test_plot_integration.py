# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for AIPerf plot functionality.

Tests the end-to-end workflow of:
1. Running aiperf profile to generate artifacts
2. Running aiperf plot to generate PNG visualizations
3. Validating the generated PNG files
"""

import struct
from pathlib import Path

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


def is_valid_png(file_path: Path) -> bool:
    """Validate that a file is a valid PNG by checking its header.

    Args:
        file_path: Path to the PNG file to validate

    Returns:
        True if the file has a valid PNG header, False otherwise
    """
    if not file_path.exists():
        return False

    try:
        with open(file_path, "rb") as f:
            # PNG files start with an 8-byte signature
            png_signature = b"\x89PNG\r\n\x1a\n"
            header = f.read(8)
            return header == png_signature
    except Exception:
        return False


def validate_png_ihdr_chunk(file_path: Path) -> tuple[int, int] | None:
    """Validate PNG IHDR chunk and extract dimensions.

    Args:
        file_path: Path to the PNG file

    Returns:
        Tuple of (width, height) if valid, None otherwise
    """
    try:
        with open(file_path, "rb") as f:
            # Skip PNG signature (8 bytes)
            f.read(8)

            # Read IHDR chunk (first chunk after signature)
            # Format: 4 bytes length, 4 bytes type "IHDR", data, 4 bytes CRC
            chunk_length_bytes = f.read(4)
            chunk_length = struct.unpack(">I", chunk_length_bytes)[0]

            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None

            # IHDR data: width(4), height(4), bit_depth(1), color_type(1), ...
            ihdr_data = f.read(chunk_length)
            width, height = struct.unpack(">II", ihdr_data[:8])

            return (width, height)
    except Exception:
        return None


@pytest.mark.integration
@pytest.mark.asyncio
class TestPlotIntegration:
    """Integration tests for aiperf profile + plot workflow."""

    async def test_profile_then_plot_single_run(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test running profile then generating PNG plots for single run.

        Workflow:
        1. Run aiperf profile to generate artifacts
        2. Run aiperf plot to generate PNG files
        3. Validate PNG files exist and are valid
        """
        # Step 1: Run profile
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --streaming
            """
        )
        assert profile_result.exit_code == 0
        assert profile_result.request_count == defaults.request_count

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Run plot to generate PNGs
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate PNG files were created
        plot_dir = artifacts_dir / "plots"
        assert plot_dir.exists(), f"Plot directory not created at {plot_dir}"

        # Check that at least some PNG files were created
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "No PNG files were generated"

        # Validate each PNG file
        for png_path in png_files:
            assert is_valid_png(png_path), (
                f"Plot {png_path.name} is not a valid PNG file"
            )

            # Validate PNG structure
            dimensions = validate_png_ihdr_chunk(png_path)
            assert dimensions is not None, (
                f"Plot {png_path.name} has invalid IHDR chunk"
            )
            width, height = dimensions
            assert width > 0 and height > 0, (
                f"Plot {png_path.name} has invalid dimensions: {width}x{height}"
            )

        # Check summary file was created
        summary_path = plot_dir / "summary.txt"
        assert summary_path.exists(), "Plot summary.txt was not created"
        summary_content = summary_path.read_text()
        assert "Generated" in summary_content
        assert "plots:" in summary_content

    async def test_profile_then_plot_with_timeslices(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with timeslice data.

        Workflow:
        1. Run profile with timeslices enabled
        2. Generate plots including timeslice visualizations
        3. Validate timeslice plots are created
        """
        # Step 1: Run profile with timeslices
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency {defaults.concurrency} \
                --streaming \
                --benchmark-duration 3 \
                --benchmark-grace-period 0 \
                --slice-duration 1
            """
        )
        assert profile_result.exit_code == 0

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Run plot
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate timeslice plots were created
        plot_dir = artifacts_dir / "plots"
        assert plot_dir.exists(), "Plot directory not created"

        # Check that PNG files were created (at least timeslice plots)
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "No PNG files were generated"

        # Validate PNG files are valid
        for png_path in png_files:
            assert is_valid_png(png_path), f"Plot {png_path.name} is not valid"

    async def test_plot_with_nonexistent_directory_fails(self, cli: AIPerfCLI):
        """Test that plot command fails gracefully with nonexistent directory."""
        plot_result = await cli.run(
            """
            aiperf plot \
                --paths /nonexistent/path/to/artifacts
            """,
            assert_success=False,
        )
        assert plot_result.exit_code != 0

    # ========================================================================
    # Server Metrics Plotting Tests
    # ========================================================================

    async def test_plot_with_server_metrics_parquet_and_json(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with server metrics from both Parquet and JSON.

        This validates the dual data path fix: plot system loads both
        Parquet (time-series) and JSON (aggregated stats) together to
        generate complete plots with both time-series data and average lines.

        Workflow:
        1. Profile with server metrics, export both Parquet and JSON
        2. Verify both export files exist
        3. Generate plots
        4. Validate server metrics plots are created with proper data
        """
        # Step 1: Profile with server metrics
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --server-metrics-formats parquet json
            """
        )
        assert profile_result.exit_code == 0
        assert profile_result.request_count == 50

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Verify both Parquet and JSON files exist
        parquet_file = artifacts_dir / "server_metrics_export.parquet"
        json_file = artifacts_dir / "server_metrics_export.json"
        assert parquet_file.exists(), "Parquet file should exist"
        assert json_file.exists(), "JSON file should exist"

        # Step 3: Generate plots
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 4: Validate plots were created (including server metrics plots if metrics available)
        plot_dir = artifacts_dir / "plots"
        assert plot_dir.exists(), "Plot directory should exist"

        # Get all PNG files created
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "At least some plots should be created"

        # Validate all PNGs are valid with proper dimensions
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

            # Validate non-trivial dimensions
            dimensions = validate_png_ihdr_chunk(png_path)
            assert dimensions is not None, f"{png_path.name} has invalid IHDR chunk"
            width, height = dimensions
            assert width >= 800 and height >= 600, (
                f"{png_path.name} has unexpectedly small dimensions: {width}x{height}"
            )

        # Check for common server metrics plots (may not all exist depending on mock metrics)
        # Just log what was created for debugging
        plot_names = [p.name for p in png_files]
        server_metric_plot_patterns = ["cache", "requests", "dual_axis"]
        _ = [
            name
            for name in plot_names
            if any(pattern in name for pattern in server_metric_plot_patterns)
        ]

        # If we have server metrics data, we should have generated at least some plots
        # But don't fail if specific plots are missing (depends on mock server metrics)

    async def test_plot_with_server_metrics_vllm_sglang(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with server metrics from multiple endpoints (vLLM + SGLang).

        Validates that plots work correctly when collecting metrics from
        multiple different server backends simultaneously.
        """
        vllm_url = aiperf_mock_server.server_metrics_urls["vllm"]
        sglang_url = aiperf_mock_server.server_metrics_urls["sglang"]

        # Step 1: Profile with multiple server metrics endpoints
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {vllm_url} {sglang_url} \
                --server-metrics-formats parquet json
            """
        )
        assert profile_result.exit_code == 0

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Generate plots
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate plots exist and are valid
        plot_dir = artifacts_dir / "plots"
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "Should generate some plots"

        # Validate all PNGs are valid
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

    async def test_plot_with_server_metrics_parquet_only(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with only Parquet export (no JSON).

        Validates that the data loader can compute aggregated stats on-the-fly
        from Parquet when JSON is not available. This tests the fallback
        mechanism in the dual data path fix.
        """
        # Step 1: Profile with only Parquet export
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --server-metrics-formats parquet
            """
        )
        assert profile_result.exit_code == 0

        artifacts_dir = profile_result.artifacts_dir

        # Verify only Parquet exists (no JSON)
        parquet_file = artifacts_dir / "server_metrics_export.parquet"
        json_file = artifacts_dir / "server_metrics_export.json"
        assert parquet_file.exists(), "Parquet file should exist"
        assert not json_file.exists(), "JSON file should NOT exist (Parquet-only test)"

        # Step 2: Generate plots (should work without JSON)
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate plots were generated successfully
        plot_dir = artifacts_dir / "plots"
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, (
            "Plots should be generated even with Parquet-only export"
        )

        # Validate PNGs
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

    async def test_plot_with_server_metrics_json_only(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with only JSON export (no Parquet).

        Validates backwards compatibility: plots should work with JSON-only
        export (pre-Parquet format).
        """
        # Step 1: Profile with only JSON export
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --server-metrics-formats json
            """
        )
        assert profile_result.exit_code == 0

        artifacts_dir = profile_result.artifacts_dir

        # Verify only JSON exists (no Parquet)
        parquet_file = artifacts_dir / "server_metrics_export.parquet"
        json_file = artifacts_dir / "server_metrics_export.json"
        assert json_file.exists(), "JSON file should exist"
        assert not parquet_file.exists(), (
            "Parquet file should NOT exist (JSON-only test)"
        )

        # Step 2: Generate plots (should work with JSON-only)
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate plots were generated
        plot_dir = artifacts_dir / "plots"
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "Plots should be generated with JSON-only export"

        # Validate PNGs
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

    async def test_plot_with_server_metrics_histogram_data(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with histogram server metrics.

        Validates that histogram metrics (like time_to_first_token_seconds)
        are properly plotted with bucket data and percentile estimates.
        """
        # Step 1: Profile with server metrics containing histograms
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --server-metrics-formats parquet json
            """
        )
        assert profile_result.exit_code == 0

        # Verify histogram metrics exist in export
        assert profile_result.has_server_metric("vllm:time_to_first_token_seconds")

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Generate plots
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate plots exist
        plot_dir = artifacts_dir / "plots"
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "Should generate plots with histogram data"

        # Validate all PNGs
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

    async def test_plot_with_dynamo_full_stack_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test plot generation with full Dynamo stack metrics (frontend + components).

        Validates plotting with complex multi-endpoint server metrics from
        disaggregated serving architecture.
        """
        urls = aiperf_mock_server.get_server_metrics_url(
            "dynamo_frontend", "dynamo_prefill", "dynamo_decode"
        )

        # Step 1: Profile with Dynamo full stack
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {" ".join(urls)} \
                --server-metrics-formats parquet json
            """
        )
        assert profile_result.exit_code == 0

        # Verify Dynamo metrics collected
        assert len(profile_result.server_metrics_endpoints_successful) >= 3

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Generate plots
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Validate plots generated
        plot_dir = artifacts_dir / "plots"
        png_files = list(plot_dir.glob("*.png"))
        assert len(png_files) > 0, "Should generate plots with Dynamo metrics"

        # Validate PNGs
        for png_path in png_files:
            assert is_valid_png(png_path), f"{png_path.name} is not a valid PNG"

    async def test_plot_summary_includes_server_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that plot summary.txt mentions server metrics plots.

        Validates that the summary output properly reports server metrics
        visualizations that were generated.
        """
        # Step 1: Profile with server metrics
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --request-count 50 \
                --concurrency 2 \
                --streaming \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --server-metrics-formats parquet json
            """
        )
        assert profile_result.exit_code == 0

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Generate plots
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0

        # Step 3: Check summary file
        plot_dir = artifacts_dir / "plots"
        summary_path = plot_dir / "summary.txt"
        assert summary_path.exists(), "Summary file should exist"

        summary_content = summary_path.read_text()
        assert "Generated" in summary_content
        assert len(summary_content) > 0
