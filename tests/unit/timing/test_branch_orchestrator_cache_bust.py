# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``BranchOrchestrator._mint_child_marker``.

Covers the cache-bust marker minting contract for SPAWN/FORK child
sessions: returns ``None`` when target is NONE, deterministic for the
same input, and unique per child conversation id.
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _make_orch(
    *,
    benchmark_id: str = "bench-xyz",
    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
) -> BranchOrchestrator:
    cs = MagicMock()
    # ``BranchOrchestrator.__init__`` reads ``dataset_metadata.conversations``
    # for its prereq index; an empty list keeps the orchestrator usable for
    # pure marker-minting unit tests without needing a full dataset.
    cs.dataset_metadata = MagicMock(conversations=[])
    issuer = MagicMock()
    return BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        benchmark_id=benchmark_id,
        cache_bust_target=cache_bust_target,
    )


def test_mint_child_marker_returns_none_when_target_is_none():
    orch = _make_orch(cache_bust_target=CacheBustTarget.NONE)
    assert orch._mint_child_marker("child-conv-1") is None


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    ],
)
def test_mint_child_marker_is_deterministic_for_same_inputs(target):
    orch_a = _make_orch(benchmark_id="bench-A", cache_bust_target=target)
    orch_b = _make_orch(benchmark_id="bench-A", cache_bust_target=target)
    marker_a = orch_a._mint_child_marker("child-conv-1")
    marker_b = orch_b._mint_child_marker("child-conv-1")
    assert marker_a is not None
    assert marker_a == marker_b


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    ],
)
def test_mint_child_marker_differs_per_child_conversation_id(target):
    orch = _make_orch(benchmark_id="bench-A", cache_bust_target=target)
    markers = {orch._mint_child_marker(f"child-{i}") for i in range(8)}
    # All eight distinct child_conversation_ids should produce eight
    # distinct markers (collision probability over 48-bit digest is
    # negligible for 8 samples).
    assert len(markers) == 8
    assert None not in markers


def test_mint_child_marker_differs_per_benchmark_id():
    orch_a = _make_orch(
        benchmark_id="bench-A", cache_bust_target=CacheBustTarget.SYSTEM_PREFIX
    )
    orch_b = _make_orch(
        benchmark_id="bench-B", cache_bust_target=CacheBustTarget.SYSTEM_PREFIX
    )
    assert orch_a._mint_child_marker("child-conv-1") != orch_b._mint_child_marker(
        "child-conv-1"
    )
