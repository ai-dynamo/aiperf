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
### Cross-lane join gating (built)

Subagent trajectory trees are gated on **live** child completion — a parent join
turn waits for the actual terminal of its children, and a tree recycles only after
the whole tree drains:

1. A parent turn that consumes subagent results (`SpawnJoin`) is **deferred** while
   any of its required children have not reached terminal; it is released and
   dispatched at `runtime.now_ns()` when the last required child terminates
   (success **or** failure). Matches Python's `waiting_on_children` exclusion from
   `dispatchable`.
2. A `background` spawn does **not** gate the parent (its join carries no
   prerequisite).
3. A tree's lane recycles only after `on_root_terminal` **and** all descendants
   drain — not per round-robin.

#### Single central driver (global-hop)

The gate, the ported `session_tree::SessionTreeRegistry`, and the recycle cursor
all live in **one** workload instance. The agentic mode therefore runs under
`--dispatch global-hop` (forced at CLI resolution and re-asserted at
`lower_legacy_agentic`): global-hop is a **single coordinator scheduling loop**
(the one central driver over the whole cell) that hops each request to a pool of
`N` worker transport threads — the Rust analog of Python's `1 strategy : 1 router :
N workers`. `--cells > 1` and non-global-hop `--dispatch` (`sharded`/`global`, which
run `N` independent per-worker pipelines) are rejected — they would split a tree's
root and children across partitions and give each partition its own gate.

#### Metadata: a side `TreeSpec` map, not the dataset DAG

The composed `Dataset` is deliberately **DAG-free**. Carrying spawn/join on the
dataset `Turn` model (`ConversationBranch` / `TurnPrerequisite` / `Conversation.dag`)
cannot satisfy the DAG validator once t\*-history-slicing separates a spawn
declaration from its join turn, and it would make subagent children non-sampleable
(breaking multi-worker partitioning). Instead the reconstruction emits
`spawn_branch`/`join_prerequisite` onto `ReconstructedTurn`, and
`agentic_replay::build_tree_specs` builds a side `Vec<TreeSpec>` (root id, the
**transitive** descendant set flattened by walking `parent_conversation_id` to the
root, and join turns keyed by the **profiling** (sliced) turn index). A join whose
children did not survive t\* is dropped (they never dispatch, so gating on them would
deadlock). The specs ride `Arc<Vec<TreeSpec>>` on `PreparedDatasetInput` (module
`agentic_tree`, always compiled) into `AgenticReplayConfig`.

#### Dispatch flow

```
reconstruction (agentx::subagent/loader)
    └─ spawn_branch / join_prerequisite on ReconstructedTurn
lower_legacy_agentic
    ├─ slice at t* (history excluded) → build_tree_specs(&sliced convs) → Vec<TreeSpec>
    └─ compose DAG-free dataset; carry Arc<Vec<TreeSpec>>; force global-hop
workload (AgenticReplayWorkload::execute, single central driver)
    ├─ TreeGate::try_new(&tree_specs) (empty specs → gate None → pass-through)
    ├─ a turn where gate.is_waiting(conv, turn_index) is DEFERRED (take_ready queue)
    ├─ each terminal → gate.on_child_terminal(conv); released parents dispatch at now
    └─ recycle draws only when gate.on_lane_terminal(root) reports whole-tree drain
```

Record-ordinal note: the warmup phase's `requests` is set to its (recycle-free)
conversation count so the striding record-ordinal issuer offsets the profiling
phase's absolute-slot base past the warmup range (otherwise both phases collide at
slot 0 under the striding issuer). `enforce_stop=false` keeps this from gating
warmup dispatch.

Byte-exact parity is validated by `tests/agentx_join_gating_parity.rs` against a
Python-generated golden (`TrajectorySource._snapshot_for` waiting/release
decisions), and the deferral ordering by `tests/agentx_join_gating_e2e.rs`.

## Future requirements

- **Accelerated cache warmup** (`--agentic-cache-warmup-duration`) + handoff
  residual delay — out of MVP-scenario scope, unimplemented (its own spec).
- Multi-child and t\*-liveness golden coverage (current golden is single-child;
  the multi-child release ordering is covered by the e2e, not the golden).

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
- Tree gate + `build_tree_specs` + `take_ready` deferral: `rust/runtime/src/agentic_replay.rs`;
  `TreeSpec` (always-compiled): `rust/runtime/src/agentic_tree.rs`.
- Tree lifecycle (pure): `rust/runtime/src/agentx/session_tree.rs`,
  `rust/runtime/src/agentx/replay_dependencies.rs`.
- Global-hop single-driver dispatch: `rust/runtime/src/engine/global_hop.rs`;
  forcing + guards: `rust/cli/src/load.rs`, `rust/runtime/src/engine/online_execution.rs`
  (`lower_legacy_agentic`).
- Parity + gating tests: `rust/runtime/tests/agentx_join_gating_parity.rs`,
  `rust/runtime/tests/agentx_join_gating_e2e.rs`,
  `tools/agentx_join_gating_golden.py`, `rust/runtime/tests/fixtures/agentx/join_gating_golden.json`.
- Turn model fields (`TurnPrerequisite`, `ConversationBranch`, `prerequisites`,
  `branch_ids`, `dag`): `rust/runtime/src/dataset/model.rs`.
- Selection + lowering into the scheduled path: `rust/runtime/src/engine/online_execution.rs`
  (`lower_legacy_agentic`), `rust/runtime/src/engine/execute/dataset_build.rs`.
- Related record: [agentx-rust-port.md](agentx-rust-port.md).
