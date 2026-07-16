<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: DAG branch orchestrator (FORK/SPAWN multi-agent branching)

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** superseded (realized by the `aiperf_runtime::graph` async dataflow) — lineage + reconciliation document, NOT a build plan
**Grounding:** end-to-end line-by-line read of the Python branch-orchestrator subsystem —
`src/aiperf/timing/branch_orchestrator.py` (518 lines), `_branch_orchestrator_state.py`,
`_branch_orchestrator_spawn.py`, `_branch_orchestrator_helpers.py`, `_branch_orchestrator_drain.py`,
`_branch_orchestrator_logging.py`; plus the collaborators it drives —
`src/aiperf/credit/sticky_router.py` (`StickyCreditRouter`), `common/models/branch.py`
(`ConversationBranchInfo`). Reconciled against the **built** Rust graph dataflow engine —
`rust/runtime/src/graph/{executor.rs,channel_store.rs,scheduler.rs,model.rs,runtime.rs,reducers.rs}`.
Companions: `2026-07-11-aiperf-rust-request-rate-multiturn-design.md` (the credit issuer; DAG children
dispatched DIRECTLY, bypassing session-slot acquisition + the continuation queue — `agent_depth>0`,
inherit parent's session slot), `2026-07-09-graph-ir-rust-port-design.md` (the dataflow port),
`2026-07-10-unified-graph-runtime-design.md` (one dispatch verb reduces every load mode).

---

## 0. Thesis — this whole subsystem collapses into the graph dataflow

**The Python `BranchOrchestrator` is superseded by the `aiperf_runtime::graph` async-dataflow engine; it is NOT a
port target.** The orchestrator is a ~1000-line credit-side driver whose entire job is to reconstruct, on
top of a fire-and-forget multiprocess credit protocol, the dependency semantics that a dataflow graph
expresses *directly*. Fan-out (spawn children on a parent turn), fan-in (a parent turn gated until N
children complete), sticky routing (keep a session's turns on one worker), and the join refcount are all
credit-protocol reconstructions of a **firing gate on an input channel** — which is exactly what
`rust/runtime/src/graph/executor.rs` + `channel_store.rs` already implement natively, running under one
single-threaded `LocalSet` per trace (`runtime.rs:5-13`), deterministic under `SimClock` (`drive_sim`,
`runtime.rs:107`).

Concretely: in Python, a parent conversation and its FORK/SPAWN children are **separate credit sessions**
(separate `x_correlation_id`s) that communicate only through the orchestrator's mutation of shared
dictionaries on the timing-manager loop; the worker processes never see the dependency. In Rust, a
parent turn, its children, and the gated continuation turn are **nodes in one `GraphRecord`**
(`model.rs:149-162`) connected by edges, and the child→parent join is a `ChannelRequirement` with a
`count` (`model.rs:83-89`) that `VersionedChannelStore::await_inputs` (`channel_store.rs:170-193`) blocks
on. The orchestrator's join bookkeeping — `PendingBranchJoin`, `PrereqState.expected`/`completed`,
`_descendant_counts`, the future/active two-level gate — is the manual re-implementation of the store's
`arrival_count` vs `producers_declared` accounting (`channel_store.rs:100-106, 207-245`). One replaces the
other wholesale.

This document (a) records **exactly what the Python orchestrator did** (the source read is still valuable —
its comments encode earned-in-blood edge cases), (b) maps each responsibility onto the graph dataflow
primitive that already expresses it, and (c) isolates the **genuine residual** — the handful of concerns
that are *policy*, not *protocol*, and therefore survive the collapse (they attach to the credit issuer /
`Workload` seam, not to a resurrected orchestrator).

---

## 1. What the Python orchestrator DOES (ground truth, cited)

The `BranchOrchestrator` is single-instance-per-phase, owned by `PhaseRunner`, constructed once the
dataset is loaded and a `CreditIssuer` exists (`branch_orchestrator.py:80-114`). Public lifecycle
(`branch_orchestrator.py:55-77`): `dispatch_pre_session_branches()` once before the first root credit;
`intercept(credit)` on EVERY credit return; `on_child_leaf_reached` / `on_child_stopped` /
`on_child_errored` from the worker when a child terminates; `has_pending_branch_work()` polled by the
strategy; `cleanup()` at teardown.

### 1.1 FORK vs SPAWN semantics

Both are `ConversationBranchInfo` (`branch.py`), discriminated by `mode` and gated by `dispatch_timing`:

- **FORK** (`branch.py:mode` doc): child **inherits the parent's context** (turn_list) and **sticky-routes
  to the parent's worker**. On spawn the orchestrator calls `sticky_router.register_child_routing(parent)`
  (`_branch_orchestrator_spawn.py:193-197`); on child-done it calls `release_child_routing(parent)`
  (`branch_orchestrator.py:387-391`). `background=True` (FORK only, `branch.py`) waives the parent's
  must-be-last-turn rule so the parent keeps running remaining turns after forking.
- **SPAWN**: child gets a **fresh context, free routing** (no sticky refcount — `_start_one_child` registers
  sticky only for FORK, `spawn.py:193-197`). Suspension of the parent is controlled by an explicit
  `DagSpawn.join_at` (a `SPAWN_JOIN` prerequisite), NOT by `background`.

### 1.2 Pre-session vs post-turn children

- **Post-turn (default `dispatch_timing="post"`)**: children fire when the parent turn that *declares* the
  branch completes. `intercept` → `get_branch_ids(credit)` (`branch_orchestrator.py:116-126`) → if
  non-empty, `_spawn_children_and_register_gates` (`:185-189`).
- **Pre-session (`dispatch_timing="pre"`, SPAWN-only, turn-0-of-root-only)**: background sub-agents
  dispatched *before* the parent's first turn. `dispatch_pre_session_branches` (`branch_orchestrator.py:128-163`)
  walks root conversations (`is_root`, `agent_depth==0`, `spawn.py` guards), fires
  `_fire_pre_session_children` (`spawn.py:45-60`), and records `(conv_id, branch_id)` in
  `_pre_dispatched_branches` so the per-turn path skips them (`spawn.py:137-139`). Pre-session branches
  **never gate the parent** (`spawn.py:141-143` forces `branch_gates=[]`).

### 1.3 Join gating (the two-level future/active gate)

A gated parent turn (one carrying a `SPAWN_JOIN` prerequisite) must not dispatch until its contributing
children complete. State is two-level (`branch_orchestrator.py:94-97`):

- **Future join** — registered at spawn time (`_ensure_future_join`, `:191-229`) keyed
  `_future_joins[parent_corr][gated_idx]`. Seeded with **every** declared `prereq_key` as an unregistered
  `PrereqState` (`:219-227`) so fan-in can't be prematurely satisfied before a sibling prereq registers.
- **Active join** — promoted when the parent actually *reaches* the turn before the gated turn
  (`_maybe_suspend_parent`, `:261-292`): if the next turn has an unsatisfied gate, set `is_blocked=True`,
  move it into `_active_joins`, bump `parents_suspended`, and return `True` (strategy must suppress its own
  next-turn dispatch). If the gate was already satisfied by fast children, pop it and return `False`.
- Satisfaction (`_satisfy_prerequisite`, `:294-336`): each child completion adds its `child_corr` to
  `PrereqState.completed`; `is_satisfied` = every prereq's `is_done` (`state.py:83-85`, `is_done` =
  `registered and len(completed) >= expected`). If satisfied **and** the parent is already blocked, pop the
  active join and `_release_blocked_join` (`:338-347`) → `issuer.dispatch_join_turn(pending)`. If satisfied
  **before** the parent arrives, pop the future entry so the parent breezes through (`:333-336`).

### 1.4 Sticky routing (parent-worker pinning) — `StickyCreditRouter`

First turn → least-loaded worker (creates a sticky session); subsequent turns → same worker; final turn →
cleanup (`sticky_router.py` class docstring). FORK children `register_child_routing(parent)` at spawn and
`release_child_routing(parent)` at done/rollback/error, refcounted so the parent's worker pin survives while
any FORK child is in flight. This exists **only** because Python fans work across worker *processes* and must
keep a multi-turn session's KV cache on one worker.

### 1.5 Drain, abort / FAIL_FAST

- **Drain** (`_branch_orchestrator_drain.py`): a sync `_drain_observer` closes the race where the
  orchestrator's *final* drain step (a cap-suppressed join, the last descendant decrement, or an
  all-children-rolled-back gate) lands AFTER the last `on_credit_return`'s deferred completion check —
  without it `all_credits_returned_event` never sets and the phase hangs (`drain.py:5-18`). Fired from
  `_handle_child_done` (`:413`), `_drain_vestigial_gates` (`spawn.py:311-315`), and error paths.
- **Descendant accounting** (`branch_orchestrator.py:400-410`): one decrement per child; when a parent's
  count hits 0 and it holds no active/future join, `_release_slot` drops the per-parent lock.
- **FAIL_FAST** (`AIPERF_DAG_FAIL_FAST=true`, `:105`): first child error →
  `_handle_child_errored_fail_fast` (`:433-480`) aborts the parent + **every orphan sibling**
  (`issuer.abort_session`), drops the parent's joins, and fires `_notify_abort` to cancel every active phase
  lifecycle so the strategy stops issuing wire credits (`:474-480`).

---

## 2. The exact state + lifecycle (cited)

`_branch_orchestrator_state.py`:

- `PrereqState` (`state.py:19-53`): `expected: int`, `completed: set[str]`, `registered: bool`. `is_done`
  requires `registered and len(completed) >= expected`. The **set** gives idempotent double-delivery
  protection; the **counter** lets multiple spawn points fan into one `prereq_key` without the orchestrator
  knowing every child id at registration.
- `PendingBranchJoin` (`state.py:56-92`): carries everything `dispatch_join_turn` needs to rebuild the
  parent's gated `TurnToSend` without re-entering the conversation source — `parent_x_correlation_id`,
  `parent_conversation_id`, `parent_num_turns`, `gated_turn_index`, `outstanding: dict[prereq_key,
  PrereqState]`, `parent_branch_mode`, `is_blocked`. `is_satisfied` = `all(s.is_done ...)`.
- `ChildJoinEntry` (`state.py:95-111`, frozen): `parent_correlation_id`, `gated_turn_index | None`,
  `prereq_key | None` (both `None` in lockstep for background/ungated children).

Orchestrator dicts (`branch_orchestrator.py:92-107`): `_child_modes`, `_future_joins`, `_active_joins`,
`_child_to_join`, `_parent_locks` (per-`x_correlation_id` `asyncio.Lock` serializing intercepts within a
session), `_descendant_counts`, `_pre_dispatched_branches`. Pre-built at init: `_prereq_index` and
`_gated_turn_prereq_keys` (`build_prereq_index`, `helpers.py:21-58`) resolve each `SPAWN_JOIN` prereq to the
spawning turn that declared its `branch_id`.

**The intercept hook** (`branch_orchestrator.py:165-189`) is how the orchestrator takes over the parent's
next-turn dispatch on credit return: under `_parent_locks[parent_corr]`, it (1) spawns any branches declared
on the completed turn (`_spawn_children_and_register_gates`) and (2) returns `_maybe_suspend_parent(credit)`
— `True` meaning "the strategy MUST suppress its default next-turn dispatch; the join will dispatch it
later." **The spawn path** (`spawn.py:62-315`): resolve branches → `_start_children` (per branch, per gate,
bump `PrereqState.expected`, register `ChildJoinEntry`, sticky-register FORK) → `asyncio.gather` the
children's `dispatch_first_turn` → `_rollback_failed_child` for any non-`True` result (three-way classify:
`BaseException`→errored, `False`→truncated, `None`→silent no-op, `spawn.py:260-282`) → `_drain_vestigial_gates`
for gates that ended zero-outstanding. **The drain path** is §1.5.

---

## 3. Reconciliation — each responsibility mapped onto `aiperf_runtime::graph`

The three columns: **Python orchestrator mechanism (path:line)** | **how the `aiperf_runtime::graph` dataflow already
expresses it** | **genuine residual, if any**.

| Responsibility | Python orchestrator mechanism (path:line) | How the graph dataflow already expresses it | Genuine residual |
|---|---|---|---|
| **Fan-out (spawn children on a turn)** | `intercept` → `_spawn_children_and_register_gates` → `_start_one_child` starts a *new credit session* per child (`branch_orchestrator.py:185-189`, `spawn.py:62-227`) | Children are **nodes** with edges from the spawning node; `schedule_successors` (`executor.rs:267-271`) + `Scheduler::successors_after` (`scheduler.rs:85-90`) schedule every successor when a node completes. Fan-out is just multiple out-edges. | **None** — pure topology. |
| **Fan-in / join gating (parent turn waits for N children)** | Two-level future/active `PendingBranchJoin` + `PrereqState.expected`/`completed`; `_satisfy_prerequisite` / `_maybe_suspend_parent` (`branch_orchestrator.py:191-336`) | The gated turn is a **node with a `ChannelRequirement { channel, count }`** (`model.rs:83-89`); `await_inputs` blocks until `arrival_count >= target` (`channel_store.rs:170-245`). `count:"all"` resolves to the static producer count (`channel_store.rs:195-205`) — the exact "wait for every declared child" semantic, computed from topology instead of a live `expected` counter. | **None** — the store's arrival/producer accounting *is* the join refcount. |
| **`expected` seeded before children register (fan-in premature-satisfaction guard)** | `_ensure_future_join` pre-seeds every `prereq_key` with an unregistered `PrereqState` (`branch_orchestrator.py:219-227`); `is_done` false until `registered` (`state.py:47-53`) | `producers_per_channel` is computed **statically** at store construction (`channel_store.rs:100-106`, `producers_declared`); `await_count` orphans a reader only when `arrival + remaining < target` (`channel_store.rs:231-236`). A gate can never be satisfied early because the producer count is known up front, not accreted. | **None** — static producer count removes the future/active two-step entirely. |
| **Sticky routing (FORK child pinned to parent worker)** | `register_child_routing` / `release_child_routing` refcount on `StickyCreditRouter` (`spawn.py:193-197`, `branch_orchestrator.py:387-391`) | **Deleted, not ported.** There are no worker processes and no KV-cache-affinity routing: a trace (parent + all descendants) runs on ONE `current_thread` `LocalSet` (`runtime.rs:5-13`); "same worker" is automatic. Parent-context inheritance (FORK) is `PromptItem::Splice` reading the parent's output channel (`model.rs:128-136`, `executor.rs:189-190`). | **None for routing.** Residual is *materialization*: FORK = child prompt splices parent turn_list; SPAWN = fresh context. That is a `PromptMaterializer` concern (dataset-segment seam), not orchestration. |
| **Pre-session spawn (background sub-agents before parent turn-0)** | `dispatch_pre_session_branches` + `_pre_dispatched_branches` dedup (`branch_orchestrator.py:128-163`, `spawn.py:137-139`) | An **entry node** (successor of `START`) with no in-edge from the parent — `Scheduler::entry_nodes` fires it at trace start (`scheduler.rs:79-82`, `executor.rs:106-110`). "Fires before the parent, never gates it" = a node on its own START-rooted branch with no edge into the parent's gated turn. | **None** — pre-session vs post-turn is edge topology (rooted at `START` vs rooted at the spawning node). The dedup set exists only because Python has two dispatch entry points; the graph has one scheduler. |
| **Gated parent turn takeover (suppress strategy dispatch, dispatch later)** | `intercept` returns `True` → strategy suppresses; `_release_blocked_join` → `dispatch_join_turn` (`branch_orchestrator.py:165-189, 338-347`) | The gated node simply **`.await`s its inputs** (`executor.rs:181-184, 273-296`); there is no "strategy" to suppress and no separate dispatch — the node fires itself when the store wakes its parked reader (`channel_store.rs:207-245`). The intercept/suppress/re-dispatch dance is a credit-protocol artifact. | **None** — the await *is* the gate. |
| **Descendant accounting / per-parent slot release** | `_descendant_counts` decrement → `_release_slot` (`branch_orchestrator.py:400-410, 482-484`) | Trace lifetime = the `TraceExecutor` future draining under `drive_sim`/`drive_real`; the in-flight counter on `Handle` tracks liveness so the driver knows when the whole trace (parent + descendants) has drained (`runtime.rs:59-61`). No per-parent lock/slot to release. | **Residual = the *session concurrency cap*** — how many traces run at once — which is a `SlotPool` / `ConcurrencyManager` policy in `aiperf_runtime::timing`, acquired per trace, not per DAG-child (children inherit; see companion spec §1.1). |
| **Drain observer (final-step-after-last-return race)** | `_drain_observer` + `_notify_drain` (`drain.py`, `branch_orchestrator.py:413`) | The race **does not exist**: completion is "the trace future resolved" (`runtime.rs:100-107`), a single well-defined edge, not a reconciliation of an async counter against a deferred callback. | **None** — deleted with the credit protocol. |
| **FAIL_FAST (abort parent + orphan siblings + whole run)** | `_handle_child_errored_fail_fast` + `_notify_abort` (`branch_orchestrator.py:433-480`) | Trace-scoped abort already exists: `ctx.set_abort(err)` short-circuits the whole trace — every node checks `ctx.is_aborted()` (`executor.rs:127-129, 178-186`), and a mid-trace dispatch failure writes a type-correct empty so successors degrade instead of orphaning (`executor.rs:199-205`). Orphan-sibling teardown = the aborted trace's other in-flight nodes seeing `is_aborted`. | **Residual = whole-*run* abort policy** (stop issuing NEW traces on first error) — a `StopChecker` / run-lifecycle concern, not per-trace. The per-trace abort + degrade is built. |
| **Malformed / rolled-back child classification (error vs truncated vs no-op)** | `_rollback_failed_child` three-way classify + `BranchStats` counters (`spawn.py:229-282`) | A child that fails to dispatch is a node whose sink returns `Err` → `empty_value` degrade (`executor.rs:199-205`) or a node that never schedules. Rollback of `expected` is unnecessary because producer counts are static. | **Residual = observability**: the `BranchStats` counters (spawned/completed/errored/truncated/suppressed) are a reporting surface; map onto the metrics accumulator, not orchestrator state. |

### 3.1 What the graph executor already does (summary)

Built and unit-tested in the `aiperf_runtime::graph` module: multi-node DAG with fan-out (multiple out-edges) and fan-in
(`ChannelRequirement.count` incl. `"all"`); firing gates honoring completion / start-anchored / first-token
/ min-start delays with compress/ignore overrides (`executor.rs:298-364`); a versioned append-only channel
store with static producer accounting and per-reader orphan propagation (`channel_store.rs`); reducers
(`overwrite` conflict-rejecting, `add_messages`) (`model.rs:27-34`, `reducers.rs`); per-trace abort +
mid-trace degrade; single-`LocalSet`-per-trace execution deterministic under `SimClock` via `drive_sim`
(`runtime.rs:107`) and live under `drive_real` (`:161`). This is the DAG conversation-branching engine —
FORK/SPAWN fan-out/fan-in/join is native.

### 3.2 What the orchestrator adds that is NOT already dataflow — the true residual

Everything genuinely residual is **policy or observability at the run/issuer boundary**, none of it a
resurrected orchestrator:

1. **Session concurrency cap** — how many *traces* (root conversations) run concurrently; per-trace acquire,
   children inherit (companion spec §1.1). Lives on `SlotPool` / `ConcurrencyManager` (`aiperf_runtime::timing`,
   built) + the `Workload`/`CreditIssuer` seam (designed).
2. **Whole-run FAIL_FAST** — "stop admitting new traces on first child error." A `StopChecker` /
   run-lifecycle flag, not per-trace (per-trace abort is built).
3. **FORK-vs-SPAWN prompt materialization** — FORK child inherits parent turn_list; SPAWN child fresh
   context. A `PromptMaterializer` / dataset-segment-seam concern (`materialize.rs`, `PromptItem::Splice`),
   not join bookkeeping.
4. **`BranchStats` observability** — spawned/completed/errored/truncated/suppressed counters map onto the
   metrics accumulator, derived from node outcomes.
5. **Pre-session `background` semantics** — a background sub-agent whose completion never gates the root and
   whose lifetime must nonetheless keep the trace alive: expressed as a START-rooted node with an edge into a
   trace-terminal sink but none into the gated turn (topology, at graph-build time).

---

## 4. Mapping onto crates — built vs designed

| Concern | Primitive | Module (crate `aiperf`) | Status |
|---|---|---|---|
| DAG fan-out / fan-in / firing gates | `TraceExecutor` + `Scheduler` | `aiperf_runtime::graph` | **built** |
| Join refcount (arrival vs static producer count) | `VersionedChannelStore` (`await_inputs`, `count:"all"`) | `aiperf_runtime::graph` | **built** |
| Parent-context splice (FORK inheritance) | `PromptItem::Splice` + `PromptMaterializer` | `aiperf_runtime::graph` | **built** (materializer trait); dataset-backed **designed** |
| Per-trace abort + mid-trace degrade (per-parent FAIL_FAST) | `ctx.set_abort` + `empty_value` | `aiperf_runtime::graph` | **built** |
| Deterministic offline / live execution | `drive_sim` / `drive_real` on `LocalSet` | `aiperf_runtime::graph` / `aiperf_runtime::clock` | **built** |
| Sticky routing | — | — | **deleted** (no worker processes; single-`LocalSet`-per-trace) |
| Future/active two-level join, `PendingBranchJoin`, `_child_to_join`, `_descendant_counts`, drain observer | — | — | **deleted** (credit-protocol artifacts subsumed by the channel store) |
| Session concurrency cap (traces in flight; children inherit) | `SlotPool` / `ConcurrencyManager` | `aiperf_runtime::timing` | **built** (pool); issuer wiring **designed** |
| Whole-run FAIL_FAST (stop admitting new traces) | `StopChecker` run flag | `aiperf_runtime::timing` | **built** (checker); DAG flag **designed** |
| Branch observability counters | metrics accumulator | `aiperf_runtime::metrics_core` | **designed** |

**No new `BranchOrchestrator` trait is needed.** The would-be seams collapse:

- A **`BranchOrchestrator`** trait would only re-wrap the executor; the executor + scheduler already own
  fan-out/fan-in. **Do not add it.**
- A **`JoinTracker`** trait is the `VersionedChannelStore`'s arrival/producer accounting — **already a
  concrete type**; if a second join policy ever appears (e.g. quorum/K-of-N joins beyond `count:N` /
  `count:"all"`), extract it behind the store's `await_count`, not a new orchestrator.
- A **`Router` / sticky-key selector** has no meaning in the single-process model. If cross-thread trace
  placement is ever added (assign whole traces to worker threads for data-plane fan-out per the request-rate
  spec §2), that is a **trace-placement** trait on the runtime (a whole trace → a thread), NOT a per-turn
  credit router — and it never splits a trace's turns across threads, so no sticky refcount is reintroduced.

---

## 5. Offline / online parity

Because DAG branching is the `aiperf_runtime::graph` module, parity is inherited for free — no orchestrator-specific parity work.
The executor runs the identical node/edge program under both drivers: `drive_real` (tokio reactor,
`runtime.rs:161`) and `drive_sim` (integer-ns DES idle-pump, `runtime.rs:107`) over the **same** `LocalSet`
and the **same** `TraceExecutor`. Firing gates read `handle.now_ns()` (`executor.rs:119-121`), so every join
delay, first-token gate, and min-start offset is on one clock timeline that `SimClock` makes deterministic
(`(at_ns, seq_no)` tie-break). A FORK/SPAWN conversation with fan-in that runs online-real reproduces
bit-for-bit under `SimClock` given the same seeds — the property the Python orchestrator could never offer,
since its joins were reconciled against live multiprocess credit returns. Parity is **code-path + report
schema**, not byte-identical real-vs-sim timings (per the port-exact ledger addendum).

---

## 6. Residual work (if any) on `aiperf_runtime::graph`

This section is deliberately NOT a port plan. The residual is small and lives at the issuer/run boundary and
the graph-build boundary — not inside a resurrected orchestrator:

1. **DAG graph-build from the dataset's branch metadata** — lower `ConversationBranchInfo` (mode /
   `dispatch_timing` / `background` / `child_conversation_ids`) + `SPAWN_JOIN` prerequisites into a
   `GraphRecord`: spawning turn → child entry nodes (edges), gated turn → node with a `ChannelRequirement`
   whose channel the children write and whose `count` = number of contributing children (or `"all"`),
   pre-session/background children → START-rooted nodes with no edge into the gated turn. This is the ONE
   real piece of work and it is a **loader/lowering** task (dataset-segment seam), not orchestration. Verify
   the branch-id shapes (`branch.py` `branch_id` doc: five forms incl. `:pre`) round-trip.
2. **FORK vs SPAWN materialization** — `PromptMaterializer` impl: FORK child splices parent turn_list; SPAWN
   child gets fresh context. Extends the existing `PromptItem::Splice` path.
3. **Session concurrency cap wiring** — acquire a session `SlotPool` slot per *trace* (root), not per
   DAG-child; children inherit (companion spec §1.1). Issuer/`Workload` glue.
4. **Whole-run FAIL_FAST flag** — a `StopChecker` predicate that stops admitting new traces after the first
   trace abort (per-trace abort + degrade already built, `executor.rs:199-205`).
5. **`BranchStats`-equivalent observability** — derive spawned/completed/errored/truncated counts from node
   outcomes into the metrics accumulator.

If none of the above surprises the graph-build, there is **no orchestrator to write**.

---

## 7. Risks / open questions — the earned-in-blood races, triaged

The Python comments flag four races. Each is triaged as **deleted (credit-protocol artifact)** or **real
dataflow concern the executor must still handle**:

- **Spawn-first race** (a graph starting directly with `SpawnNode` races child credit ahead of frame
  registration → KeyError/deadlock; user memory `gotcha_graph_spawn_first_race`). **DELETED as an
  artifact.** It exists because Python registered `_spawn_frames` on a *different* async turn than the child
  dispatch. In the dataflow, an entry node's inputs and the store's producer counts are established at
  `build_context` (`executor.rs:91-104`) *before* any node is scheduled (`schedule_entries`, `:106-110`), so
  there is no window where a child fires before its gate is registered. No leading-kickoff-LLM workaround
  needed.
- **Drain-after-final-return race** (final drain step lands after the last credit-return's deferred
  completion check → phase hangs; `drain.py:5-18`). **DELETED as an artifact.** Completion is the trace
  future resolving under `drive_sim`/`drive_real` (`runtime.rs:100-107`), a single edge — there is no
  deferred-callback-vs-counter reconciliation to lose.
- **Nested-DAG grandchild truncation** (`agent_depth+1` chains; multi-step spawn bodies with span-dedup gaps
  remain xfail in Python — user memory `gotcha_graph_loop_credit_pipeline_partial`). **REAL dataflow
  concern.** Grandchildren are just deeper nodes/edges, but the executor must correctly propagate orphaning
  through a multi-level fan-in when an intermediate producer is cancelled: `mark_producer_done(success=false)`
  poisons a channel only when it is truly dead (`channel_store.rs:339-367`), and `await_count` orphans a
  reader when `arrival + remaining < target` (`:231-236`). Add coverage for a ≥3-level DAG where a
  mid-level node degrades, asserting the leaf gate orphans (not hangs) and the trace drains.
- **Cap-refusal join deadlock** (a child refused at the concurrency cap never completes, so the parent's join
  never satisfies and the phase hangs; Python's `_drain_vestigial_gates` + `dispatch_join_turn` returning
  False handle it, `spawn.py:284-315`, `branch_orchestrator.py:343-347`). **REAL — must be handled at
  graph-build/issuer, not the store.** In the dataflow, if a child node is *never scheduled* (refused
  admission), its channel's static producer count would still expect it and the gate would hang. Resolution:
  the session cap applies to whole traces (children inherit the parent's slot, never re-acquire — companion
  spec §1.1), so a *child* is never independently cap-refused; the deadlock class is designed out. **Open
  question:** confirm the graph-build never emits a child node that could be admission-gated independently of
  its root trace — if it can, the static producer count must be decremented at build time for the refused
  child (equivalent to Python's `expected` rollback, `spawn.py:246-252`), OR the gate uses `count:N` with N
  computed post-admission. Prefer the inherit-slot invariant that removes the case entirely.

Additional open question: **`background=True` FORK parents** (parent keeps running after forking, `branch.py`
`background` doc) — model as a node with an out-edge to children AND an out-edge to its own next turn, with
NO child→next-turn gate. Verify the graph-build emits exactly that (no accidental join edge) so the parent is
not spuriously suspended.

---

## 8. One-line summary

The Python `BranchOrchestrator` (~1000 lines reconstructing fan-out / fan-in / join-refcount / sticky-routing
on top of a multiprocess credit protocol) is **superseded wholesale by the already-built `aiperf_runtime::graph`
async-dataflow engine** — FORK/SPAWN fan-out is out-edges, join gating is a `ChannelRequirement.count` the
`VersionedChannelStore` blocks on with static producer accounting, sticky routing and the future/active gate
and the drain-observer race are **deleted credit-protocol artifacts**, per-trace FAIL_FAST is `ctx.set_abort`
— leaving only a thin residual of **policy** (session-cap wiring, whole-run FAIL_FAST) and **graph-build
lowering** (branch metadata → nodes/edges, FORK/SPAWN materialization) at the issuer/loader boundary, with no
orchestrator to rebuild.
