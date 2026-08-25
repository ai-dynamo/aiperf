<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy

## Purpose

Accuracy evaluation splits the work: Rust owns dispatch, capture, metrics, and
native-v2 reporting; a pinned Python Lighteval/DeepEval worker owns canonical
prompts, tasks, private tests, and scoring. The worker is a directly injected
`AccuracyEvaluator` stdio seam, not a registry category (see
[extension-registry.md](extension-registry.md)).

## Built

### Accumulator and evaluator seam

`aiperf_runtime::accuracy_core` provides the accuracy accumulator/analyzer and the
static-accuracy evaluator worker seam (`AccuracyEvaluator`, backed by the pinned
Python worker). These modules are linked in the binary but sit off the default
product wire. `Application::stock` composes `BuiltinAIPerfRegistryFactory` with
`native_execution_factories()`. The built-in HTTP extension registers the
`scheduled` and `graph` workloads; the static-accuracy workload and its
`NativeStaticAccuracyEvaluatorFactory` are implemented behind an explicit
registration seam but are not registered by the stock composition.

The CLI's typed `BenchmarkConfig` serializes `cfg.accuracy`.
`BenchmarkConfigWireV2` deliberately deserializes only its runner-relevant
Config subset without `deny_unknown_fields` and has no accuracy field, so it
ignores that serialized value rather than rejecting it. Its projection derives
the selected workload as `scheduled` or `graph` from the dataset type.
Consequently, static accuracy is unreachable through the current product
projection (see [runner-protocol.md](runner-protocol.md)).

### Evaluator process supervision

`PythonEvaluator` frames writes through one async mutex and resolves replies in
a dedicated reader task through an `id -> oneshot sender` table. Replies may
arrive in any order; each waiter receives only the response carrying its request
id. EOF, malformed JSON, missing ids, and read failures drain the table with one
typed infrastructure error and retain that failure for later submissions, so no
request can remain parked behind a reader that failed while the table was idle.
The application-facing `AccuracyEvaluator` remains the sequential
control-plane trait; the supervisor transport itself is safe for overlapping
grade requests.

On Unix the evaluator starts a new session. Async shutdown and fault cleanup
close its stdin, wait for the leader within a finite deadline, send `SIGKILL` to
the whole process group even when the leader has already exited, and wait until
the group is absent. Synchronous drop aborts the reader tasks and signals that
same process group instead of killing only the leader. This prevents Lighteval
sandbox descendants from surviving their owned evaluator session. Non-Unix
cleanup retains child-only process handling.

### Sharded capture, single grade

A static-accuracy run shards its dispatch and capture like any other
request-bounded run. The control plane splits into a `Send` half and a `!Send`
half:

- Frozen problems and response associations are shared as `Arc<Dataset>` plus
  `Arc<[ProblemAssociation]>` (`{problem_id, correlation_id, task}`, all opaque
  ids) and cloned cheaply into every shard.
- Per-record work is a `problem_id` lookup that pushes a `CapturedResponse`
  (`{sequence, problem_id, correlation_id, task, start_ns, end_ns, terminal,
  response_text}`) — pure `Send` data; no Python is called per record.
- The `!Send` pinned-Python evaluator never crosses a spawn boundary. At the join
  the coordinator concatenates the disjoint per-shard capture vectors and grades
  them once on the main thread through the single evaluator
  (`grade_accuracy_captures`).

Grading is keyed by `problem_id`, so the merged, order-independent capture set
produces the same tally regardless of worker count. Capture uniqueness keys on the
globally unique `correlation_id` (the per-worker credit `sequence` restarts at 0
per shard and collides across shards, so it is retained only as the primary sort
key).

## Future requirements

The product projection must lower authored accuracy configuration into the
static-accuracy workload and include that workload in stock composition before
the built evaluator, sharded capture, and single-grade path are reachable from
the public CLI.

## Source anchors

- `rust/runtime/src/accuracy_core/` (`worker.rs`, `protocol.rs`, `mod.rs`),
  `rust/runtime/src/accuracy.rs` (`ProblemAssociation`, `CapturedResponse`,
  `grade_accuracy_captures`), `rust/runtime/src/metrics_core/accuracy.rs`.
- `rust/runtime/src/extensions/mod.rs` (`BuiltinAIPerfRegistryFactory`).
- `rust/runtime/src/engine/application.rs` (`Application::stock`) and
  `rust/runtime/src/engine/execution_factories.rs`
  (`native_execution_factories`).
- `rust/runtime/src/engine/online_execution.rs` (built-in workload and
  static-accuracy registration seams).
- `rust/runtime/src/engine/protocol_v2.rs` (`BenchmarkConfigWireV2` projection)
  and `rust/cli/src/model/config.rs` (`cfg.accuracy` serialization).
- `rust/runtime/src/engine/execute.rs` (shard capture and single-grade join).
- `rust/e2e-tests/tests/test_accuracy_mock.rs`.
- `rust/runtime/tests/accuracy_worker_native_path.rs` (real subprocess batch
  grading and Unix descendant reaping).
