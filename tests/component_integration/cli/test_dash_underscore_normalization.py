# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dash/underscore and case normalization in CLI arguments.

These tests verify that plugin names in CLI arguments work correctly regardless of:
- Case (CONSTANT, Constant, constant)
- Dashes vs underscores (concurrency-burst vs concurrency_burst)
- Mixed variations (Concurrency-Burst, CONCURRENCY_BURST)
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestArrivalPatternNormalization:
    """Tests for arrival pattern normalization with dashes, underscores, and case."""

    @pytest.mark.parametrize(
        "arrival_pattern",
        [
            "concurrency_burst",
            "concurrency-burst",
            "CONCURRENCY_BURST",
            "CONCURRENCY-BURST",
            "Concurrency_Burst",
            "Concurrency-Burst",
        ],
    )
    def test_arrival_pattern_dash_underscore_normalization(
        self, cli: AIPerfCLI, arrival_pattern: str
    ):
        """Test that arrival pattern works with dashes or underscores."""
        result = cli.run_sync(
            f"""
            aiperf profile
                --model {defaults.model}
                --num-sessions 5
                --concurrency 2
                --arrival-pattern {arrival_pattern}
                --ui none
            """
        )
        assert result.request_count == 5


class TestEndpointTypeNormalization:
    """Tests for endpoint type normalization with dashes and underscores."""

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            "huggingface_generate",
            "huggingface-generate",
            "HUGGINGFACE_GENERATE",
            "HUGGINGFACE-GENERATE",
            "HuggingFace_Generate",
            "HuggingFace-Generate",
        ],
    )
    def test_endpoint_type_dash_underscore_normalization(
        self, cli: AIPerfCLI, endpoint_type: str
    ):
        """Test that endpoint type works with dashes or underscores."""
        result = cli.run_sync(
            f"""
            aiperf profile
                --model {defaults.model}
                --num-sessions 5
                --endpoint-type {endpoint_type}
                --ui none
            """
        )
        assert result.request_count == 5
