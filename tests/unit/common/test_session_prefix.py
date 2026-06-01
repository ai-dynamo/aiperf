# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-session-instance unique prefix generator."""

import pytest

from aiperf.common.session_prefix import (
    make_unique_session_prefix,
    prepend_unique_session_prefix,
)


@pytest.mark.parametrize("num_tokens", [0, -1, -100])  # fmt: skip
def test_make_unique_session_prefix_disabled_returns_empty(num_tokens: int) -> None:
    assert make_unique_session_prefix("any-correlation-id", num_tokens) == ""


def test_make_unique_session_prefix_is_deterministic_per_id() -> None:
    first = make_unique_session_prefix("corr-1", 16)
    second = make_unique_session_prefix("corr-1", 16)
    assert first == second
    assert first != ""


def test_make_unique_session_prefix_differs_across_ids() -> None:
    assert make_unique_session_prefix("corr-1", 16) != make_unique_session_prefix(
        "corr-2", 16
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 16, 64])  # fmt: skip
def test_make_unique_session_prefix_word_count_matches_request(num_tokens: int) -> None:
    assert len(make_unique_session_prefix("corr-1", num_tokens).split()) == num_tokens


@pytest.mark.parametrize("existing", [None, "", "prior context"])  # fmt: skip
def test_prepend_disabled_returns_input_unchanged(existing: str | None) -> None:
    assert prepend_unique_session_prefix(existing, "corr-1", 0) == existing


def test_prepend_without_existing_context_is_bare_prefix() -> None:
    result = prepend_unique_session_prefix(None, "corr-1", 8)
    assert result == make_unique_session_prefix("corr-1", 8)


def test_prepend_keeps_existing_context_after_prefix() -> None:
    result = prepend_unique_session_prefix("prior context", "corr-1", 8)
    prefix = make_unique_session_prefix("corr-1", 8)
    assert result == f"{prefix}\n\nprior context"


def test_prepend_is_consistent_within_session_but_unique_across() -> None:
    # Same id (a session's successive turns) => identical => within-session reuse.
    assert prepend_unique_session_prefix("ctx", "corr-1", 8) == (
        prepend_unique_session_prefix("ctx", "corr-1", 8)
    )
    # Different ids (resamples) => different => no cross-resample cache hit.
    assert prepend_unique_session_prefix("ctx", "corr-1", 8) != (
        prepend_unique_session_prefix("ctx", "corr-2", 8)
    )
