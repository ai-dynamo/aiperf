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
