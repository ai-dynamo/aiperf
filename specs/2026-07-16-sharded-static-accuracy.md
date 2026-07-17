<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharded static accuracy: capture on every worker, grade once on the coordinator

**Date:** 2026-07-16
**Status:** BUILT. The thread-per-core scheduled runtime shards a static-accuracy run
(`workers > 1`) exactly like any other request-bounded run. The per-record work is pure
`Send` data capture; the `!Send` pinned-Python evaluator never crosses a spawn boundary;
grading happens once on the coordinator thread over the concatenated per-shard captures.

Ground every claim below against the cited `rust/runtime/src/…:line`. Specs are intent;
code is truth.

---

## 1. Summary

Before this change, a static-accuracy run was clamped to a single worker: the accuracy
record processor held the `!Send` Python/Lighteval evaluator handle, so it could not be
built per shard, and the sharded arm was disqualified for accuracy runs. That clamp is
**gone**. The accuracy dataset now shares its response associations as an
`Arc<[ProblemAssociation]>` — plain `Send + Sync` opaque ids
(`rust/runtime/src/accuracy.rs:46`, `:77`, `:207`) — and each thread-per-core shard builds
its **own** capture processor over those same read-only associations
(`rust/runtime/src/engine/execute.rs:2457`). The per-record work each shard does is a
`problem_id` lookup that pushes a `CapturedResponse` (`rust/runtime/src/accuracy.rs:491`),
which carries only `Send` data (ids, timing, terminal status, response text —
`rust/runtime/src/accuracy.rs:59`). No Python is called per record.

At the join, the coordinator concatenates the disjoint per-shard capture vectors
(`rust/runtime/src/engine/execute.rs:2317`, `:3225`) and grades them **once** on the main
thread through the single `!Send` evaluator (`rust/runtime/src/engine/execute.rs:3437` →
`grade_accuracy_captures`, `rust/runtime/src/accuracy.rs:633`). Grading is keyed by
`problem_id`, so the merged (order-independent) capture set produces the same tally
regardless of worker count.

The design's load-bearing correctness fact — surfaced by TDD — is that the per-shard
credit `sequence` (the per-worker monotonic credit id) **collides across shards** (each
shard's issuer restarts at 0), so capture uniqueness must key on the globally-unique
per-request `correlation_id`, with `sequence` retained only as the primary sort key
(`rust/runtime/src/accuracy.rs:466`).

---

## 2. Why this is safe to shard (the seam)

The static-accuracy control plane splits cleanly into a `Send` half and a `!Send` half:

| Half | Type | Where it lives | Crosses spawn boundary? |
|---|---|---|---|
| Frozen problems + response associations | `Arc<Dataset>` + `Arc<[ProblemAssociation]>` (`accuracy.rs:77`) | Read-only, cloned into every shard | Yes — cheap `Arc` clone (`accuracy.rs:207`) |
| Per-record capture | `CapturedResponse` (`accuracy.rs:59`) | Pushed into a per-shard `RefCell<Vec<…>>` (`accuracy.rs:421`) | Only the `Vec<CapturedResponse>` merges back (`execute.rs:2303`) |
| Grading | `dyn AccuracyEvaluator` (the pinned Python/Lighteval worker) | Coordinator thread ONLY (`execute.rs:3439`) | **Never** — stays on main thread |

`ProblemAssociation` is `{problem_id, correlation_id, task}` (`accuracy.rs:46`) — all opaque
ids, hence `Send + Sync`. `CapturedResponse` is `{sequence, problem_id, correlation_id,
task, start_ns, end_ns, terminal, response_text}` (`accuracy.rs:59`) — it deliberately holds
"no `Rc`/evaluator handle, so a per-shard capture set crosses the thread-per-core spawn
boundary" (`accuracy.rs:53-58`, doc comment). Because the evaluator is confined to the
coordinator, no `!Send` value is ever required inside a worker.

The `AccuracyRecordProcessor` itself (`accuracy.rs:419`) is `!Send` only through its
`RefCell<Vec<CapturedResponse>>` interior — which never leaves its owning thread. Each shard
constructs a fresh one from the shared associations via `AccuracyRecordProcessor::new`
(`accuracy.rs:430`), which builds a `correlation_id → ProblemAssociation` `BTreeMap` lookup
table (`accuracy.rs:432-436`).

---

## 3. The per-record capture (pure `Send` data, no Python)

`AccuracyRecordProcessor` implements `TurnRecordProcessor` (`accuracy.rs:490`). On each
terminal turn, `process` (`accuracy.rs:491`):

1. Reads `credit.turn.request_correlation_id` (`accuracy.rs:496`).
2. Looks up the `ProblemAssociation` in the per-shard `BTreeMap` (`accuracy.rs:497`);
   a missing association is a hard error (`accuracy.rs:498-501`).
3. Constructs a `CapturedResponse` (`accuracy.rs:502-511`) — copying the `problem_id`/`task`
   from the association, the `start_ns`/`end_ns`/`terminal`/`response_text` from the turn
   outcome, `sequence = credit.id`, and `correlation_id = credit.turn.uuid` (the
   per-request uuid, NOT the association's correlation).
4. Pushes it into the `RefCell<Vec<…>>` (`accuracy.rs:512`).

There is no evaluator call, no network, no `Rc` in this path — only a map lookup and a
struct push. That is precisely what makes accuracy shardable
(`execute.rs:2722-2727`, comment).

---

## 4. The TDD-surfaced bug: `sequence` collides, `correlation_id` is the uniqueness key

`validate_captures` (`accuracy.rs:466`) runs once on the full merged capture set. Its
contract (`accuracy.rs:451-465`, doc comment) records the earned-in-blood detail:

- The issue `sequence` is the **per-worker** monotonic credit id. It is unique *within* a
  worker but **collides across shards** because each shard's credit issuer restarts at 0.
- The per-request `correlation_id` is a per-request uuid (`accuracy.rs:505`), so it is
  **globally unique** and is the guard against double-processing.

Therefore `validate_captures` sorts by `sequence` **then** breaks ties with `correlation_id`
(`accuracy.rs:473-477`) and asserts uniqueness on `correlation_id` alone
(`accuracy.rs:478-485`) — never on `sequence`. On the single-thread path the sequences are
already distinct, so this preserves exact issue order; on the sharded path the per-shard
runs interleave by sequence with a stable correlation tiebreak, which is
aggregate-equivalent because grading is keyed by `problem_id`
(`accuracy.rs:461-465`). Had uniqueness keyed on `sequence`, the merged sharded set would
have tripped a false "duplicate" error on every collided pair — the failing test that drove
the fix.

---

## 5. The clamp is gone; `shardable` is just `workers > 1`

The old `plan.workers = 1` clamp for accuracy no longer gates the accuracy path (the sole
surviving `plan.workers = 1` at `execute.rs:551` is unrelated dry-run/validate plumbing).
Shardability is now simply:

```text
let shardable = request.workers > 1;               // execute.rs:2750
```

`exact_fold_eligible` (`execute.rs:993`) still lists `has_accuracy` as a disqualifier
(`execute.rs:995`, field `execute.rs:1042`) — so an accuracy run stays on the **retain**
record path (records are held for post-run scoring), but retain vs. fold is orthogonal to
sharding. `shardable` is retained as an explicit `ExactFoldInputs` axis but deliberately
**not** read by the fold gate (`execute.rs:1012-1023`), a documented regression guard
against re-adding `&& !inputs.shardable`.

The routing comment at `execute.rs:2718-2730` states it directly: static accuracy shards
"INCLUDING static accuracy: its per-record capture is pure `Send` data … so each shard owns
a capture processor over the shared read-only associations and the disjoint captures
concatenate at the coordinator, which grades once on the main thread (the `!Send` Python
evaluator never crosses the spawn boundary)."

---

## 6. End-to-end flow

```text
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ COORDINATOR (main thread)  —  PREPARE                                             │
  │                                                                                    │
  │  spawn pinned Python/Lighteval evaluator  (dyn AccuracyEvaluator, !Send)          │
  │  load_evaluator_problems_with_grader(...)          accuracy.rs:299                │
  │        │  strict paginated load, dup-id + identity validation                     │
  │        ▼                                                                            │
  │  AccuracyDataset::from_evaluator_problems(...)      accuracy.rs:94                │
  │        │  lower opaque prompts → unified segment dataset                          │
  │        ▼                                                                            │
  │   ┌───────────────────────────────┐   ┌──────────────────────────────────────┐   │
  │   │ Arc<Dataset> (frozen prompts) │   │ Arc<[ProblemAssociation]>            │   │
  │   │  → normal inference dispatch  │   │   {problem_id, correlation_id, task} │   │
  │   └───────────────────────────────┘   │   Send + Sync, read-only  a.rs:207   │   │
  │                                        └──────────────────────────────────────┘   │
  │  PreparedAccuracy { evaluator, loaded, dataset, processor, tokenizer }            │
  │                                                     execute.rs:1078               │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                   │  Arc clone of associations into ShardedShared
                                   │  ShardedShared.accuracy_associations   execute.rs:2208
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ SHARD DISPATCH  —  W worker OS threads (each: current_thread rt + LocalSet, !Send) │
  │                                                                                    │
  │  thread 0                  thread 1                 ...       thread W-1           │
  │  ┌───────────────────┐     ┌───────────────────┐             ┌───────────────────┐ │
  │  │ clone Arc assoc   │     │ clone Arc assoc   │             │ clone Arc assoc   │ │
  │  │ AccuracyRecord-   │     │ AccuracyRecord-   │             │ AccuracyRecord-   │ │
  │  │ Processor::new    │     │ Processor::new    │   (each its  │ Processor::new    │ │
  │  │   a.rs:430        │     │   a.rs:430        │    own map)  │   a.rs:430        │ │
  │  │ registered on the │     │ registered on the │             │ registered on the │ │
  │  │ PROFILING phase   │     │ PROFILING phase   │             │ PROFILING phase   │ │
  │  │   execute.rs:2509 │     │   execute.rs:2509 │             │   execute.rs:2509 │ │
  │  │                   │     │                   │             │                   │ │
  │  │ per terminal turn:│     │ per terminal turn:│             │ per terminal turn:│ │
  │  │  problem_id lookup│     │  problem_id lookup│             │  problem_id lookup│ │
  │  │  push Captured-   │     │  push Captured-   │             │  push Captured-   │ │
  │  │  Response  a.rs:512│     │  Response         │             │  Response         │ │
  │  │  (NO Python)      │     │  (NO Python)      │             │  (NO Python)      │ │
  │  │                   │     │                   │             │                   │ │
  │  │ take_captures()   │     │ take_captures()   │             │ take_captures()   │ │
  │  │   a.rs:446        │     │   a.rs:446        │             │   a.rs:446        │ │
  │  │ Vec<CapturedResp> │     │ Vec<CapturedResp> │             │ Vec<CapturedResp> │ │
  │  │  seq: 0,1,2,...    │     │  seq: 0,1,2,...    │  (seqs      │  seq: 0,1,2,...    │ │
  │  │  corr: uuid-a...   │     │  corr: uuid-b...   │   COLLIDE,   │  corr: uuid-c...   │ │
  │  │                   │     │                   │   corr uniq) │                   │ │
  │  └─────────┬─────────┘     └─────────┬─────────┘             └─────────┬─────────┘ │
  │            │  ScheduledShardOutcome.accuracy_captures  execute.rs:2303 │           │
  └────────────┼───────────────────────┼──────────────────────────────────┼──────────┘
               │                        │                                   │
               └────────────────────────┴───────────────────────────────────┘
                                   │  ScheduledShardOutcome::absorb
                                   │    self.accuracy_captures.extend(other)  execute.rs:2317
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ CONCAT MERGE (order-independent — grading keyed by problem_id)                    │
  │   accuracy_captures = outcome.accuracy_captures            execute.rs:3225        │
  │   (single-thread arm instead drains its one processor      execute.rs:3100)       │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ COORDINATOR (main thread)  —  GRADE ONCE  at finalize                             │
  │                                                                                    │
  │  grade_accuracy_captures(std::mem::take(&mut accuracy_captures), evaluator, ...)  │
  │                                            execute.rs:3437 → accuracy.rs:633      │
  │   1. validate_captures  a.rs:466   → sort by (sequence, correlation_id),          │
  │                                       assert correlation_id unique (NOT sequence) │
  │   2. batch submit responses by problem_id  (GRADE_BATCH_SIZE=128)  a.rs:656       │
  │        → evaluator.grade_batch(...)   [the ONLY Python call, on main thread]      │
  │   3. join grades → AccuracyRecord[] + AccuracyFailure[]  a.rs:699                 │
  │   4. AccuracyAccumulator + AccuracyResultsAnalyzer → AccuracyAnalysis  a.rs:726   │
  │                                                                                    │
  │  → NativeReport (mode="accuracy", accuracy, accuracy_records, evaluator, errors)  │
  │                                            execute.rs:3444-3450                   │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data structures & call sites (verification index)

| Claim | Cite |
|---|---|
| `ProblemAssociation` = opaque `{problem_id, correlation_id, task}` | `accuracy.rs:46-51` |
| `CapturedResponse` holds only `Send` data, no `Rc`/evaluator handle | `accuracy.rs:53-69` |
| `AccuracyDataset` = `Arc<Dataset>` + `Arc<[ProblemAssociation]>` | `accuracy.rs:76-80` |
| `associations()` shares the `Send + Sync` handle, `#[cfg(feature = "engine")]` | `accuracy.rs:200-209` |
| `record_processor()` builds a processor over cloned associations | `accuracy.rs:217-219` |
| `AccuracyRecordProcessor::new` builds correlation→association map | `accuracy.rs:430-439` |
| `take_captures` drains the per-shard `RefCell<Vec<…>>` | `accuracy.rs:446-448` |
| `validate_captures` sorts (seq, corr), asserts corr unique | `accuracy.rs:466-487` |
| `process` = problem lookup + `CapturedResponse` push, no Python | `accuracy.rs:491-514` |
| `grade_accuracy_captures` = the single main-thread grade merge point | `accuracy.rs:625-761` |
| `grade_accuracy_responses` wraps `take_captures` + `grade_accuracy_captures` | `accuracy.rs:616-623` |
| `ShardedShared.accuracy_associations: Option<Arc<[ProblemAssociation]>>` | `execute.rs:2201-2208` |
| Each shard builds its own capture processor | `execute.rs:2457-2460` |
| Accuracy processor registered on profiling phase only | `execute.rs:2509-2513` |
| Per-shard captures drained into the shard outcome | `execute.rs:2577-2579` |
| `ScheduledShardOutcome.accuracy_captures` (disjoint per shard) | `execute.rs:2300-2303` |
| `absorb` concatenates captures across shards | `execute.rs:2317` |
| `shardable = request.workers > 1` | `execute.rs:2750` |
| Single-thread arm drains its one processor | `execute.rs:3099-3101` |
| Sharded arm concatenates per-shard captures | `execute.rs:3223-3225` |
| `accuracy_associations` populated from the prepared accuracy dataset | `execute.rs:3189-3191` |
| Single main-thread grade at finalize | `execute.rs:3432-3449` |
| `has_accuracy` still disqualifies exact-fold (retain path) | `execute.rs:995`, `:1042`, `:1043` |

---

## 8. Extension notes (design-ahead)

- Grading remains a `dyn AccuracyEvaluator` trait (`accuracy.rs:19`), so a future
  in-process Rust grader or an alternate provider is a drop-in — the shard/merge machinery
  is agnostic to what runs on the coordinator.
- The capture path is a `TurnRecordProcessor` (`accuracy.rs:490`), the same seam the metrics
  capture uses (`CapturePhaseProcessor`, `execute.rs:2497`), so accuracy adds no bespoke
  worker plumbing — it is one more record processor pushed onto the phase's processor list
  (`execute.rs:2505-2513`).
- Cross-shard uniqueness/order invariants live entirely in `validate_captures`
  (`accuracy.rs:466`); any future issuer change that makes `sequence` globally unique would
  not require changes here (the correlation tiebreak already subsumes it).
