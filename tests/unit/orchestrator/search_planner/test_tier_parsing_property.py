# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Tier Parsing Produces Correct Groupings.

Feature: multi-tier-slo-search, Property 1: Tier Parsing Produces Correct Groupings

Validates: Requirements 1.1, 1.5
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.orchestrator.search_planner.parsing import (
    parse_sla_tier,
    validate_tier_list,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_VALID_METRIC_TAGS = (
    "output_token_throughput",
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "request_throughput",
    "input_token_throughput",
)

_VALID_STATS = (
    "avg",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)
_VALID_OPS = ("lt", "le", "gt", "ge")


def _sla_filter_str() -> st.SearchStrategy[str]:
    """Generate a valid metric_tag:stat:op:threshold string."""
    return st.builds(
        lambda tag, stat, op, threshold: f"{tag}:{stat}:{op}:{threshold}",
        tag=st.sampled_from(_VALID_METRIC_TAGS),
        stat=st.sampled_from(_VALID_STATS),
        op=st.sampled_from(_VALID_OPS),
        threshold=st.floats(
            min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False
        ),
    )


def _tier_label() -> st.SearchStrategy[str]:
    """Generate a valid tier label (alphanumeric + underscore, no colons/commas)."""
    return st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)


def _labeled_tier_str() -> st.SearchStrategy[tuple[str, str, int]]:
    """Generate a labeled tier string: 'LABEL:FILTER[,FILTER...]'.

    Returns (tier_string, expected_label, expected_filter_count).
    """
    return st.builds(
        lambda label, filters: (
            f"{label}:{','.join(filters)}",
            label,
            len(filters),
        ),
        label=_tier_label(),
        filters=st.lists(_sla_filter_str(), min_size=1, max_size=5),
    )


def _unlabeled_tier_str() -> st.SearchStrategy[tuple[str, int]]:
    """Generate an unlabeled tier string: 'FILTER[,FILTER...]'.

    Returns (tier_string, expected_filter_count).
    """
    return st.builds(
        lambda filters: (",".join(filters), len(filters)),
        filters=st.lists(_sla_filter_str(), min_size=1, max_size=5),
    )


# ---------------------------------------------------------------------------
# Property 1: Tier Parsing Produces Correct Groupings
# ---------------------------------------------------------------------------


@given(data=_labeled_tier_str())
@settings(max_examples=100, deadline=None)
def test_labeled_tier_parsing_produces_correct_filter_count(
    data: tuple[str, str, int],
) -> None:
    """Labeled tier string produces SLOTier with correct filter count.

    **Validates: Requirements 1.1, 1.5**
    """
    tier_str, expected_label, expected_filter_count = data

    tier = parse_sla_tier(tier_str)

    assert tier.label == expected_label
    assert len(tier.filters) == expected_filter_count


@given(data=_unlabeled_tier_str())
@settings(max_examples=100, deadline=None)
def test_unlabeled_tier_parsing_produces_auto_label(
    data: tuple[str, int],
) -> None:
    """Unlabeled tier string produces SLOTier with auto-generated non-empty label.

    **Validates: Requirements 1.1, 1.5**
    """
    tier_str, expected_filter_count = data

    tier = parse_sla_tier(tier_str, _auto_index=0)

    assert tier.label == "tier_1"
    assert tier.label != ""
    assert len(tier.filters) == expected_filter_count


@given(auto_index=st.integers(min_value=0, max_value=99), data=_unlabeled_tier_str())
@settings(max_examples=100, deadline=None)
def test_unlabeled_tier_auto_index_produces_unique_labels(
    auto_index: int,
    data: tuple[str, int],
) -> None:
    """Auto-generated labels follow tier_{N+1} pattern and are non-empty.

    **Validates: Requirements 1.5**
    """
    tier_str, _ = data

    tier = parse_sla_tier(tier_str, _auto_index=auto_index)

    assert tier.label == f"tier_{auto_index + 1}"
    assert tier.label != ""


@given(
    tier_data=st.lists(
        _labeled_tier_str(),
        min_size=2,
        max_size=10,
    ).filter(lambda items: len({label for _, label, _ in items}) == len(items)),
)
@settings(max_examples=100, deadline=None)
def test_multiple_tiers_have_unique_labels(
    tier_data: list[tuple[str, str, int]],
) -> None:
    """Multiple parsed tiers with unique labels pass validate_tier_list.

    **Validates: Requirements 1.1, 1.5**
    """
    tiers = [parse_sla_tier(tier_str) for tier_str, _, _ in tier_data]

    labels = [t.label for t in tiers]
    assert len(labels) == len(set(labels))

    validated = validate_tier_list(tiers)
    assert len(validated) == len(tier_data)

    for tier, (_, expected_label, expected_count) in zip(
        validated, tier_data, strict=False
    ):
        assert tier.label == expected_label
        assert tier.label != ""
        assert len(tier.filters) == expected_count
