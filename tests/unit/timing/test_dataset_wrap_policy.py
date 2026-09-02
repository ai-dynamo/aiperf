# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset wrap policy for AGENTIC_REPLAY."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.timing.trajectory_source import validate_dataset_wrap_policy


@pytest.mark.parametrize(
    "kwargs",
    [
        param(
            {
                "distinct": 2,
                "concurrency": 3,
                "allow_dataset_wrap": True,
                "expected_num_sessions": 3,
                "total_expected_requests": None,
                "expected_duration_sec": 30.0,
            },
            id="explicit_wrap_allows_oversubscribe",
        ),
        param(
            {
                "distinct": 2,
                "concurrency": 3,
                "allow_dataset_wrap": False,
                "expected_num_sessions": 2,
                "total_expected_requests": None,
                "expected_duration_sec": 30.0,
            },
            id="sessions_within_corpus",
        ),
        param(
            {
                "distinct": 2,
                "concurrency": 3,
                "allow_dataset_wrap": False,
                "expected_num_sessions": None,
                "total_expected_requests": None,
                "expected_duration_sec": None,
            },
            id="one_pass_no_bounds",
        ),
        param(
            {
                "distinct": 8,
                "concurrency": 8,
                "allow_dataset_wrap": False,
                "expected_num_sessions": None,
                "total_expected_requests": None,
                "expected_duration_sec": 30.0,
            },
            id="concurrency_equals_distinct",
        ),
        param(
            {
                "distinct": 393,
                "concurrency": 512,
                "allow_dataset_wrap": False,
                "expected_num_sessions": None,
                "total_expected_requests": None,
                "expected_duration_sec": 1800.0,
                "cache_bust_enabled": True,
            },
            id="cache_bust_satisfies_wrap_optin",
        ),
        param(
            {
                "distinct": 2,
                "concurrency": 3,
                "allow_dataset_wrap": False,
                "expected_num_sessions": 3,
                "total_expected_requests": None,
                "expected_duration_sec": 30.0,
                "cache_bust_enabled": True,
            },
            id="cache_bust_allows_session_oversubscribe",
        ),
    ],
)  # fmt: skip
def test_validate_dataset_wrap_policy_allows(kwargs: dict) -> None:
    validate_dataset_wrap_policy(**kwargs)


def test_validate_dataset_wrap_policy_rejects_unintentional_lane_cloning() -> None:
    with pytest.raises(ValueError, match="dataset wrapping is disabled"):
        validate_dataset_wrap_policy(
            distinct=2,
            concurrency=3,
            allow_dataset_wrap=False,
            expected_num_sessions=3,
            total_expected_requests=None,
            expected_duration_sec=30.0,
        )


def test_validate_dataset_wrap_policy_rejects_duration_only_oversubscribe() -> None:
    """AgentX-style: duration set, no session budget, concurrency > pool."""
    with pytest.raises(ValueError, match="dataset wrapping is disabled"):
        validate_dataset_wrap_policy(
            distinct=5,
            concurrency=8,
            allow_dataset_wrap=False,
            expected_num_sessions=None,
            total_expected_requests=None,
            expected_duration_sec=1800.0,
        )


def test_validate_dataset_wrap_policy_rejects_when_cache_bust_disabled() -> None:
    """Explicitly-off cache-bust must not satisfy the wrap opt-in."""
    with pytest.raises(ValueError, match="dataset wrapping is disabled"):
        validate_dataset_wrap_policy(
            distinct=393,
            concurrency=512,
            allow_dataset_wrap=False,
            expected_num_sessions=None,
            total_expected_requests=None,
            expected_duration_sec=1800.0,
            cache_bust_enabled=False,
        )
