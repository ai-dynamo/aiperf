<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic-replay cross-lane spawn/join gating

## Purpose

Reach byte-exact parity with Python `AgenticReplayStrategy` for **subagent
trajectory trees** in the Rust `agentic_replay` timing mode. A WEKA trajectory is
a *tree*: a root conversation plus subagent children (`::sa:{agent_id}` and its
worker-suffixed siblings). Python gates dispatch on tree structure — a parent turn
that consumes subagent results is not issued until its children reach terminal
(`dispatchable = [s for s in states if not s.waiting_on_children]`), and a tree's
session slot is held until the **whole tree drains** before it recycles. This spec
defines that cross-lane orchestration as a layer inside the scheduled
`agentic_replay` workload, without coupling to graph-ir and without disturbing the
verbatim-Raw request bodies.

Scope: spawn/join gating + tree-gated recycle only. The gated
accelerated-cache-warmup (`--agentic-cache-warmup-duration`) is a separate,
out-of-MVP-scenario spec and is out of scope here.

## Built

The `agentic_replay` timing mode already exists and runs end-to-end
([execution anchors below](#source-anchors)):

- A first-class scheduled `Workload` (`PhaseSpec::AgenticReplay` +
  `AgenticReplayWorkload`), selected for weka runs under an agentic-replay
  scenario via `--weka-semantics` (default legacy; `graph-ir` bypasses).
- Per-lane t\* sampling (numpy PCG64), history-exclusion snapshot slice, warmup
  barrier phase (turn n-1, `max_tokens=1`, capped lead, global t\*-alignment),
  profiling offset dispatch (leading-idle cap, burst/spread), byte-exact
  cache-bust including per-recycle `recycle_pass` minting, per-stream
  continuations, trajectory recycle with a fresh-correlation double-recycle guard,
  and full engine-parity output through the shared scheduled runtime.
- Subagent children are reconstructed as separate conversations and dispatched at
  their t\*-relative offsets. This reproduces the recorded ordering but does **not**
  enforce the live cross-lane dependency: a parent join turn fires at its recorded
  offset rather than waiting for actual (possibly slow) child completion, and a
  tree recycles per round-robin rather than on whole-tree drain.

The reconstruction already detects subagent spawn/join structure (the LCP spawn +
seam-join detection in `agentx::subagent`), and the dataset `Turn` model already
carries the fields needed to express it (`TurnPrerequisite`, `ConversationBranch`,
`Turn.prerequisites`, `Turn.branch_ids`, `Conversation.dag`). The pure
tree-lifecycle logic is already ported (`agentx::session_tree::SessionTreeRegistry`
with `open_tree` / `register_descendants` / `on_descendant_done` /
`on_root_terminal` / `release_all`). Neither the metadata surfacing nor the
registry wiring is connected to dispatch yet.

## Future requirements

### Contract

For a trajectory tree replayed in the PROFILING phase:

1. A parent turn carrying a `SpawnJoin` prerequisite for branch *b* is **not
   dispatched** until every child conversation named by *b* has reached terminal
   (success or failure). This matches Python's `waiting_on_children` exclusion
   from `dispatchable` and the join resume in `_dispatch_snapshot_for_profiling`.
2. A `background` spawn does **not** gate the parent — the parent continues after
   emitting the branch; the children replay concurrently.
3. A tree's slot is held until `on_root_terminal` **and** all descendants drain;
   only then may recycle draw against that tree's lane (replacing today's
   unconditional round-robin recycle). This matches the session-tree-registry
   "root slot held until the whole tree drains" behavior.
4. Dispatch order and per-turn dispatch instants under these gates are byte-exact
   against Python for a fixed seed + trace, validated by a Python-generated golden.

### Data model (reuse existing fields)

Spawn/join structure is authored onto the dataset `Turn` model — no side channel,
no new request bytes:

- The parent's **spawn turn** carries a `ConversationBranch`
  (`branch_id → child_conversation_ids`, `mode` fork/spawn, `background`) on
  `Conversation.dag` / `Turn.branch_ids`.
- The parent's **join turn** carries a `TurnPrerequisite { kind: SpawnJoin,
  branch_id, child_conversation_ids }`.

These map 1:1 onto Python `spawn_by_branch` / `join_by_branch` /
`join_turn_index`. The join turn is the first parent turn whose recorded timestamp
is at/after the branch's child-completion time (the existing seam-join point);
`background` branches emit a `ConversationBranch` with no join prerequisite.

### Components and data flow

```
reconstruction (agentx::subagent/loader)
    └─ emit ConversationBranch on spawn turn + SpawnJoin prerequisite on join turn
composer (agentx::weka_dataset)
    └─ preserve turn.prerequisites / turn.branch_ids / Conversation.dag onto the Dataset
workload setup (agentic_replay)
    └─ SessionTreeRegistry: open_tree(root_corr) per tree; register_descendants(n)
dispatch loop
    ├─ a turn with a SpawnJoin prerequisite is deferred while waiting_on_children
    ├─ each child terminal → on_descendant_done(root_corr); when the branch is
    │  satisfied, release the deferred parent join turn
    └─ recycle draws a tree's lane only after on_root_terminal + full drain
```

The registry is the single source of truth for "is this tree's parent releasable"
and "may this tree recycle". A per-branch deferred-dispatch slot holds the parent
join turn until release.

### Error handling

- A dangling/unknown child conversation id in a prerequisite fails closed with a
  clear error at setup — the tree is never silently un-gated.
- A child **terminal failure** counts as "done" for join release (the parent
  proceeds and observes the failure), matching Python; a child that never
  terminates cannot deadlock the tree past the phase grace/drain — release is
  bounded by the phase lifecycle, not an unbounded wait.

### Testing (byte-exact bar)

- **Golden:** a Python-generated fixture for a root+subagent trace asserts the
  Rust dispatch order and the `waiting_on_children` gating decisions match the
  Python `_dispatch_snapshot_for_profiling` output byte-for-byte for a fixed seed.
- **E2e:** `aiperf profile --scenario inferencex-agentx-mvp --public-dataset
  weka_cc_traces_062126` against `aiperf-mock-server` with controlled per-child
  latency asserts the parent join turn's dispatch instant follows the *actual*
  child completion (not the recorded offset) and that a tree recycles only after
  full drain.

## Source anchors

- Timing mode + dispatch: `rust/runtime/src/agentic_replay.rs`.
- Reconstruction + subagent spawn/join detection: `rust/runtime/src/agentx/subagent.rs`,
  `rust/runtime/src/agentx/loader.rs`.
- Dataset composer: `rust/runtime/src/agentx/weka_dataset.rs`.
- Tree lifecycle (pure, to be wired): `rust/runtime/src/agentx/session_tree.rs`,
  `rust/runtime/src/agentx/replay_dependencies.rs`.
- Turn model fields (`TurnPrerequisite`, `ConversationBranch`, `prerequisites`,
  `branch_ids`, `dag`): `rust/runtime/src/dataset/model.rs`.
- Selection + lowering into the scheduled path: `rust/runtime/src/engine/online_execution.rs`
  (`lower_legacy_agentic`), `rust/runtime/src/engine/execute/dataset_build.rs`.
- Related record: [agentx-rust-port.md](agentx-rust-port.md).
