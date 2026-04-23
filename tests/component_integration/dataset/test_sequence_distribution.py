# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for sequence length distribution.

Tests the --sequence-distribution parameter which specifies distribution of
input/output sequence lengths using format: "isl_mean|isl_stddev,osl_mean|osl_stddev:weight;..."
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSequenceLengthDistribution:
    """Test sequence length distribution functionality."""

    def test_sequence_distribution_single_bucket(self, cli: AIPerfCLI):
        """Test that single-bucket distribution produces values within expected statistical range."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "128|50,64|25:100" \
                --num-sessions 20 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # With a single bucket (100% weight), all requests should have these lengths
        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length").value
            osl = record.metrics.get("output_sequence_length").value

            # Should be within range of mean ± 3*stddev (99.7% of values)
            # ISL: Mean 128, stddev 50 -> range [0, 278]
            # OSL: Mean 64, stddev 25 -> range [0, 139]
            assert 0 < isl <= 281, f"ISL {isl} outside expected range"
            assert 0 < osl <= 140, f"OSL {osl} outside expected range"

    def test_sequence_distribution_respects_bucket_weights(self, cli: AIPerfCLI):
        """Test that 50:50 bucket weights produce roughly equal distribution."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "100|20,50|10:50;200|40,100|20:50" \
                --num-sessions 100 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        isl_values = [
            record.metrics.get("input_sequence_length").value for record in result.jsonl
        ]

        # Count requests in each bucket based on ISL midpoint (150)
        # Bucket 1: ISL ~ 100 (values < 150)
        # Bucket 2: ISL ~ 200 (values >= 150)
        bucket1_count = sum(1 for isl in isl_values if isl < 150)
        bucket2_count = sum(1 for isl in isl_values if isl >= 150)

        # With 50:50 weights, expect roughly equal counts (allow 30% tolerance)
        expected_per_bucket = len(isl_values) / 2
        tolerance = expected_per_bucket * 0.30

        assert abs(bucket1_count - expected_per_bucket) < tolerance, (
            f"Bucket 1 count {bucket1_count} deviates too far from expected {expected_per_bucket}"
        )
        assert abs(bucket2_count - expected_per_bucket) < tolerance, (
            f"Bucket 2 count {bucket2_count} deviates too far from expected {expected_per_bucket}"
        )


@pytest.mark.component_integration
class TestRandomRangeRatio:
    """Test vllm-style --random-range-ratio uniform ISL/OSL sampling."""

    def test_random_range_ratio_float_bounds_isl_and_osl(self, cli: AIPerfCLI):
        """A single-float --random-range-ratio keeps every request's ISL/OSL in range."""
        import math

        isl_mean = 512
        osl_mean = 128
        ratio = 0.3

        isl_low = math.floor(isl_mean * (1 - ratio))
        isl_high = math.ceil(isl_mean * (1 + ratio))
        osl_low = math.floor(osl_mean * (1 - ratio))
        osl_high = math.ceil(osl_mean * (1 + ratio))

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --isl {isl_mean} \
                --osl {osl_mean} \
                --random-range-ratio {ratio} \
                --num-sessions 30 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        isls, osls = [], []
        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length").value
            osl = record.metrics.get("output_sequence_length").value
            # Allow a small tokenizer rounding slack on ISL (corpus-based generation
            # is decode-then-re-tokenize and can drift by a token or two).
            assert isl_low - 2 <= isl <= isl_high + 2, (
                f"ISL {isl} outside [{isl_low - 2}, {isl_high + 2}]"
            )
            assert osl_low <= osl <= osl_high, (
                f"OSL {osl} outside [{osl_low}, {osl_high}]"
            )
            isls.append(isl)
            osls.append(osl)

        # With 30 samples across these windows, we expect real spread (not all equal).
        assert len(set(isls)) > 1
        assert len(set(osls)) > 1

    def test_random_range_ratio_json_dict_independent_bounds(self, cli: AIPerfCLI):
        """A JSON dict lets input and output ratios differ."""
        import math

        isl_mean = 512
        osl_mean = 128
        in_r = 0.1
        out_r = 0.5

        isl_low = math.floor(isl_mean * (1 - in_r))
        isl_high = math.ceil(isl_mean * (1 + in_r))
        osl_low = math.floor(osl_mean * (1 - out_r))
        osl_high = math.ceil(osl_mean * (1 + out_r))

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --isl {isl_mean} \
                --osl {osl_mean} \
                --random-range-ratio '{{"input": {in_r}, "output": {out_r}}}' \
                --num-sessions 30 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length").value
            osl = record.metrics.get("output_sequence_length").value
            assert isl_low - 2 <= isl <= isl_high + 2, (
                f"ISL {isl} outside [{isl_low - 2}, {isl_high + 2}]"
            )
            assert osl_low <= osl <= osl_high, (
                f"OSL {osl} outside [{osl_low}, {osl_high}]"
            )
