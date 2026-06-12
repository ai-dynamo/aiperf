# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test for tier count validation.

Feature: multi-tier-slo-search, Property 2: Tier Count Validation

Validates: Requirements 1.3

For any tier count N outside [2, 10], validate_tier_list SHALL reject with
ValueError. For any N in [2, 10], validate_tier_list SHALL accept.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.orchestrator.search_planner.multi_tier_models import SLOTier
from aiperf.orchestrator.search_planner.parsing import validate_tier_list


def _make_tier(label: str) -> SLOTier:
    """Create a minimal valid SLOTier with a unique label."""
    return SLOTier(
        label=label,
        filters=[
            SLAFilter(
                metric_tag="output_token_throughput",
                stat="avg",
                op="gt",
                threshold=100.0,
            )
        ],
    )


def _make_tier_list(count: int) -> list[SLOTier]:
    """Create a list of N valid SLOTier objects with unique labels."""
    return [_make_tier(f"tier_{i}") for i in range(count)]


class TestProperty2TierCountValidation:
    """Property 2: Tier Count Validation.

    **Validates: Requirements 1.3**
    """

    @given(count=st.integers(min_value=2, max_value=10))
    @settings(max_examples=100, deadline=None)
    def test_valid_tier_counts_accepted(self, count: int) -> None:
        """Tier counts in [2, 10] are accepted without error."""
        tiers = _make_tier_list(count)
        result = validate_tier_list(tiers)
        assert result == tiers

    @given(count=st.integers(min_value=0, max_value=1))
    @settings(max_examples=100, deadline=None)
    def test_too_few_tiers_rejected(self, count: int) -> None:
        """Tier counts below 2 raise ValueError."""
        tiers = _make_tier_list(count)
        with pytest.raises(ValueError, match="expected between"):
            validate_tier_list(tiers)

    @given(count=st.integers(min_value=11, max_value=50))
    @settings(max_examples=100, deadline=None)
    def test_too_many_tiers_rejected(self, count: int) -> None:
        """Tier counts above 10 raise ValueError."""
        tiers = _make_tier_list(count)
        with pytest.raises(ValueError, match="expected between"):
            validate_tier_list(tiers)
