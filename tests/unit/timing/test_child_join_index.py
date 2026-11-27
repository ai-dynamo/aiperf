# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChildJoinIndex must answer any_child_tracked_for_parent exactly like the full scan."""

import random

from aiperf.timing.branch_orchestrator import (
    ChildJoinEntry,
    ChildJoinIndex,
    any_child_tracked_for_parent,
)


def _entry(parent: str, gate: int | None = None) -> ChildJoinEntry:
    return ChildJoinEntry(
        parent_correlation_id=parent, gated_turn_index=gate, prereq_key=None
    )


def test_child_join_index_matches_scan_under_churn() -> None:
    rng = random.Random(3)
    plain: dict[str, list[ChildJoinEntry]] = {}
    index = ChildJoinIndex()
    parents = [f"p{i}" for i in range(12)] + ["never"]

    def check() -> None:
        assert dict(index) == plain
        for parent in parents:
            assert any_child_tracked_for_parent(
                index, parent
            ) == any_child_tracked_for_parent(plain, parent), parent

    for step in range(600):
        child = f"c{rng.randrange(80)}"
        op = rng.random()
        if op < 0.5:
            # multi-gate children name the same parent twice; some are empty
            parent = rng.choice(parents[:-1])
            entries = [_entry(parent, g) for g in range(rng.randrange(0, 3))]
            plain[child] = entries
            index[child] = entries
        elif op < 0.8:
            plain.pop(child, None)
            index.pop(child, None)
        elif op < 0.9 and child in plain:
            del plain[child]
            del index[child]
        elif op < 0.95:
            plain.clear()
            index.clear()
        if step % 25 == 0:
            check()
    check()
    assert all(n > 0 for n in index.tracked_per_parent.values())
