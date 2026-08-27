<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Graph runtime

## Purpose

`aiperf_runtime::graph` is the Graph-IR runtime and dataflow plane: it compiles
graph inputs into node programs and executes them as a deterministic async
dataflow over the shared `Clock` seam. Its sole product entry point is the
`graph` workload, selected when the dataset graph source is `dag_jsonl`,
`weka_trace`, `dynamo_trace`, `agent_recording`, `conditional_graph`, or
`otlp_genai`.

## Built

### Execution

The plane runs on a tokio `current_thread` runtime plus `LocalSet` with an
injected clock, through the shared `drive_real` (online HTTP or gRPC) and
`drive_sim` (offline co-simulation) split. Graph nodes dispatch over HTTP or gRPC
through the object-safe `Rc<dyn Dispatcher>` seam, so placement never matches on
transport kind (see [execution-model.md](execution-model.md)). Phase orchestration
and Ctrl-C cancellation are the shared seam (see
[phase-orchestration.md](phase-orchestration.md)).

### Determinism

All per-trace state lives behind `Rc<RefCell<_>>` on a single thread — no `Arc`,
`Mutex`, or `Send`/`Sync` on trace state. A single monotonic `write_seq` counter
orders writes; reduction is made order-independent by sorting on
`(write_seq, writer_node_id)`. Given identical inputs, a deterministic issuer, and
an injected `SimClock`, the dispatch event stream and channel snapshots are
reproducible.

### Compilers and store

The runner selects one strict native compiler and produces `GraphTracePlan`s plus
one frozen segment store: direct authored `dag_jsonl`, recursive `weka_trace`, and
`dynamo.request.trace.v1` lowering (`graph::recorded`). `dag_jsonl` bypasses linear
`Dataset` composition (see [dataset.md](dataset.md)). Recorded content draws from
the canonical `aiperf_runtime::rng` BLAKE3/PCG64 stream (see [rng.md](rng.md)), so
equivalent WEKA and Dynamo traces have native segment and HTTP-body byte parity
under that stream.

### Trajectory snapshot and warmup priming

A trajectory-snapshot (t*) and warmup-priming subsystem is built on this plane:
`aiperf_runtime::rng::numpy_pcg64` and `aiperf_runtime::graph::{tstar, snapshot,
warmup_handoff}`, plus the graph phase runtime's per-phase t* split, warmup-abort
gating, `GraphPressureRecycle`, and handoff resume. The front end owns the
scenario config-lock; the runner consumes only the resolved v2 knobs.

## Source anchors

- `rust/runtime/src/graph/` (`lowering.rs`, `execution.rs`, `executor.rs`,
  `runtime.rs`, `scheduler.rs`, `reducers.rs`, `channels.rs`, `channel_store.rs`,
  `dag_source.rs`, `recorded/`, `segment.rs`, `snapshot.rs`, `tstar.rs`,
  `warmup_handoff.rs`, `placement.rs`, `policy.rs`, `sink.rs`, `transport_sink.rs`,
  `workload.rs`).
- `rust/runtime/src/engine/{graph_execution.rs,graph_input.rs,graph_phase_runtime.rs}`.
- `rust/e2e-tests/tests/test_graph_cellular.rs`,
  `rust/cli/tests/{recorded_graph_stdio_e2e.rs,test_graph_grpc.rs}`.
