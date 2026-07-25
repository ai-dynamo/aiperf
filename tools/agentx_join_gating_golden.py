# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the byte-exact join-gating golden for the Rust ``agentic_replay`` parity test.

This drives the **real** Python join-gating kernel —
``aiperf.timing.trajectory_source.TrajectorySource._snapshot_for`` (reached via
``TrajectorySource`` construction / ``_build_timestamped_trajectory``) — over a
small fixed root+subagent trace, at a deterministic ``t*`` (the start-ratio
window is collapsed to a single point so no RNG is involved). It reads out, for
the profiling snapshot, which conversation states are ``waiting_on_children``
and the child ids that gate each such parent join, then emits:

    {
      "t_star_ms": <float>,
      "trace": { root + children turn timestamps + the join structure },
      "waiting_before": [[conversation_id, next_turn_index], ...],
      "gating_children": { parent_conversation_id: [child_conversation_id, ...] },
      "release_order": [child_conversation_id, ...]
    }

The ``trace`` block is the identical logical trace the Rust test reconstructs
into ``ReconstructedConversation``s, so both sides consume ONE fixture. The Rust
test then builds ``TreeSpec``s via ``build_tree_specs`` + drives a ``TreeGate``
and asserts its independent join-gating decision (which root joins are
``is_waiting`` before any child terminal, and the release order once children
terminate) equals the Python-produced ``waiting_before`` / ``gating_children`` /
``release_order`` byte-for-byte.

The Python ``waiting_on_children`` decision is: a root turn ``J`` is gated iff a
non-background branch whose join turn index is ``J`` has >=1 live child at
``t*`` (``trajectory_source.py`` ``_snapshot_for``:
``waiting = root_next_idx in pending_join_targets``). ``build_tree_specs`` +
``TreeGate`` on the Rust side compute the same rule from the reconstructed
``join_prerequisite``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

# Bootstrap the plugin registry from the in-tree manifest. When aiperf is run
# from source (not pip-installed), the `aiperf.plugins` entry point is not
# registered in distribution metadata, so `discover_plugins()` finds nothing and
# the dynamically-generated `PluginType` enum is missing its categories. Loading
# the builtin manifest directly populates the registry BEFORE `aiperf.plugin.enums`
# is imported (which builds `PluginType` from `list_categories()`).
from aiperf.plugin import plugins as _plugins

if not _plugins.list_categories():
    _plugins.load_manifest("aiperf.plugin:plugins.yaml")

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.trajectory_source import TrajectorySource

# --- The fixed trace ---------------------------------------------------------
# Root "trace": 3 turns. Turn 1 spawns branch `agent_0`; turn 2 joins on it.
# Child "trace::sa:agent_0": 3 turns, alive across t*.
ROOT_ID = "trace"
CHILD_ID = "trace::sa:agent_0"
BRANCH_ID = "trace:spawn:agent_0"
JOIN_TURN_INDEX = 2

ROOT_TS_MS = [0.0, 12000.0, 20000.0]
CHILD_TS_MS = [13000.0, 14000.0, 17000.0]


def build_dataset() -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=ROOT_ID,
                turns=[
                    TurnMetadata(timestamp_ms=ROOT_TS_MS[0]),
                    TurnMetadata(timestamp_ms=ROOT_TS_MS[1], branch_ids=[BRANCH_ID]),
                    TurnMetadata(
                        timestamp_ms=ROOT_TS_MS[2],
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id=BRANCH_ID,
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id=BRANCH_ID,
                        child_conversation_ids=[CHILD_ID],
                        mode=ConversationBranchMode.SPAWN,
                        start_timestamp_ms=13000.0,
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id=CHILD_ID,
                turns=[TurnMetadata(timestamp_ms=t) for t in CHILD_TS_MS],
                is_root=False,
                agent_depth=1,
                parent_conversation_id=ROOT_ID,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def main() -> None:
    md = build_dataset()
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = [ROOT_ID]

    # start_min_ratio == start_max_ratio collapses the t* window to a single
    # deterministic point (hi == lo, so `_build_timestamped_trajectory` takes
    # `t* = lo` with NO RNG draw). Trace bounds are [0, 20000], so
    # t* = 0 + 0.675 * 20000 = 13500.0 -- after the child's first turn (13000)
    # and before its next (14000): the child is live, and the root's next turn
    # is its join turn (index 2).
    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=123,
        start_min_ratio=0.675,
        start_max_ratio=0.675,
    )

    trajectory = src.trajectories[0]
    snapshot = trajectory.snapshot
    assert snapshot is not None, "expected a timestamped snapshot"

    # Which states are waiting_on_children (the gated parent joins), in state
    # order, as (conversation_id, next_turn_index).
    waiting_before = [
        [s.conversation_id, s.next_turn_index]
        for s in snapshot.states
        if s.waiting_on_children
    ]

    # The child ids that gate each waiting parent: read from the branch(es)
    # whose join turn matches the waiting parent's next turn index. We recover
    # this from the dataset branches (the same structure `_snapshot_for`
    # consulted via `_branch_runtimes`).
    gating_children: dict[str, list[str]] = {}
    for conv_id, turn_idx in waiting_before:
        meta = next(c for c in md.conversations if c.conversation_id == conv_id)
        # Map branch_id -> join turn index (first SPAWN_JOIN prereq turn).
        join_by_branch: dict[str, int] = {}
        for ti, turn in enumerate(meta.turns):
            for prereq in turn.prerequisites or []:
                if (
                    prereq.kind == PrerequisiteKind.SPAWN_JOIN
                    and prereq.branch_id is not None
                    and prereq.branch_id not in join_by_branch
                ):
                    join_by_branch[prereq.branch_id] = ti
        kids: list[str] = []
        for branch in meta.branches or []:
            if not branch.is_background and join_by_branch.get(branch.branch_id) == turn_idx:
                kids.extend(branch.child_conversation_ids)
        gating_children[conv_id] = kids

    # Release order: the join clears once every gating child is terminal. With a
    # single gating child the order is that child; the join releases on the last
    # child to terminate. Emit the gating children in id-sorted order (the
    # deterministic terminal order the Rust test drives).
    release_order = sorted({c for kids in gating_children.values() for c in kids})

    out = {
        "t_star_ms": snapshot.t_star_ms,
        "trace": {
            "root": {
                "conversation_id": ROOT_ID,
                "turns_ms": ROOT_TS_MS,
                "join": {
                    "turn_index": JOIN_TURN_INDEX,
                    "branch_id": "br:agent_0",
                    "child_conversation_ids": [CHILD_ID],
                },
            },
            "children": [
                {
                    "conversation_id": CHILD_ID,
                    "parent_conversation_id": ROOT_ID,
                    "turns_ms": CHILD_TS_MS,
                }
            ],
        },
        "waiting_before": waiting_before,
        "gating_children": gating_children,
        "release_order": release_order,
        "_provenance": (
            "aiperf.timing.trajectory_source.TrajectorySource._snapshot_for "
            "(via TrajectorySource construction), t* window collapsed to a point"
        ),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
