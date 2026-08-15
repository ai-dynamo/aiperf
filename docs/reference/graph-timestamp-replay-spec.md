<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Graph timestamp replay

This page documents how Dynamo trace timestamps are preserved, transformed, and
used by the Agent Graph replay path. The implementation shares the existing trace
replay configuration with Baseten traces; Agent Graph does not add graph-specific
timing flags.

The behavior described here is implemented by the Dynamo adapter's
`from_dynamo_trace` and its `dynamo_trie_nodes` lowering, by
`AgentGraphReplayStrategy` in the schedule plane, and by `TraceExecutor` at
runtime.

## Current behavior

Dynamo lowering preserves two timing representations on each lowered `LlmNode`:

- `recorded_start_unix_ms`: the absolute request-start timestamp from the
  capture, when available;
- `arrival_offset_us`: the relative, idle-gap-warped offset used by the graph
  executor and t* snapshot logic.

The absolute timestamp is retained as provenance. The executor schedules against
the monotonic event-loop clock using relative offsets; it never sleeps against a
Unix epoch timestamp.

Agent Graph replay uses dependency-driven execution. A node can dispatch only after its
channel requirements and timing gates are ready. Open-loop replay additionally
paces trace starts and graph nodes against the recorded timeline. The graph
strategy owns trace admission, lane concurrency, replay speedup, t* planning,
and return routing.

On the open-loop path every trace starts independently at its recorded time. An
explicit `--concurrency N` bounds how many traces run concurrently: a trace whose
recorded start has arrived waits for a free slot, so execution slips while the
schedule itself stays anchored and immutable. Without an explicit
`--concurrency`, admission is not gated — the phase's inherited default of 1 is
not treated as a ceiling, so the default run is unchanged.

`fixed_schedule` remains the linear timing phase and is not an agent-graph phase
type. Graph workloads select their timing behavior through the trace replay
options described below.

## Configuration

The following existing options apply to Dynamo graph replay:

```text
--open-loop-replay
--no-open-loop-replay
--open-loop-strict
--replay-speedup FACTOR
--trace-idle-gap-cap-seconds SECONDS
--ignore-trace-delays
```

`--burst-phase-starts` can synchronize the starts of graph warmup and profiling,
but does not replace the recorded replay timeline.

The defaults are open-loop replay enabled and a replay speedup of `1.0` when no
speedup is configured. `--open-loop-strict` requires open-loop replay.

## Timestamp lowering

The Dynamo adapter derives a request start timestamp using the first available
source in this order:

1. `request_received_ms`;
2. `event_time_unix_ms - total_time_ms`;
3. `event_time_unix_ms`.

The adapter builds the per-trace timeline, applies the configured idle-gap
policy, and stamps each lowered node. A node's `arrival_offset_us` therefore
reflects the timeline that the replay will use, while
`recorded_start_unix_ms` remains the original source timestamp.

The graph store and the graph sidecar preserve both fields. Filtering and trace
selection occur before the replay strategy builds its schedule, so omitted
records do not affect the replay timeline.

## Replay timeline

For open-loop replay, the strategy finds the earliest preserved timestamp in the
selected graph corpus:

```text
schedule_zero = min(recorded_start_unix_ms)
```

For a trace, its recorded start is the earliest preserved timestamp in that
trace. The strategy waits until the trace's relative target is reached:

```text
trace_target_seconds =
    (trace_start_unix_ms - schedule_zero)
    / 1000
    / replay_speedup
```

The event-loop monotonic time at replay start is the schedule anchor.

Missing timestamps never create a synthetic schedule, but what happens next
depends on whether the corpus is wholly or partially untimestamped:

- **Wholly untimestamped** (no shipped producer emits this — the dynamo adapter
  always stamps a recorded start): the graph falls back to its relative timing
  and dependency behavior, replaying edge delays.
- **Partially timestamped** (some traces carry a recorded start, others do
  not): open-loop profiling refuses to start. The untimestamped traces cannot be
  paced, so they would all fire at once at t=0 on top of the faithful replay of
  the rest, inflating early load with nothing in the results saying so. Fix the
  trace source, or restrict the corpus to timestamped traces.

## Replay speedup

`--replay-speedup` divides normalized replay timing:

```text
1       recorded wall-clock timing
10      ten times faster
0.5     two times slower
```

The strategy scales executor-visible timing fields, including node arrival
offsets and graph edge delays. A scaled target that is already in the past is
not awaited; dependency readiness and the current monotonic clock determine
when dispatch can proceed.

Speedup does not modify request content, recorded hash IDs, KV-cache structure,
output limits, node identity, or dependency topology.

## Replay modes

### Open loop

Open-loop replay is the default. The strategy paces each trace against its
recorded start, while the executor continues to enforce graph dependencies and
channel requirements. A slow target server can therefore delay downstream
nodes; agent-graph replay does not violate causal ordering to preserve a recorded target.

### Dependency-driven replay

With `--no-open-loop-replay`, the graph is driven by dependency completion and
its recorded relative delays. This provides back-pressure when replayed service
times differ from the capture. Graph edge and channel semantics remain active.

This is not the linear `fixed_schedule` implementation: the graph executor still
owns readiness, fan-in, output publication, and successor scheduling.

### Open-loop strict

`--open-loop-strict` is supported for Dynamo graphs whose prompts are
self-contained in the unified store. The strategy creates a scheduling
projection; it does not mutate the source graph or the persisted sidecar.

The projection:

- removes channel inputs and node-level timing gates from the scheduling copy;
- replaces the graph edges with `START`-relative timestamp gates;
- retains the original graph for identity, metrics, diagnostics, and
  materialization;
- uses the earliest timestamp in the trace as that trace's zero;
- clamps negative relative targets to zero.

This mode intentionally bypasses runtime dependency gates. It is unsuitable
for graphs whose prompts require values produced by earlier nodes. Graphs
without usable node timestamps are not projected and retain normal graph
behavior.

## Idle-gap transforms

`--trace-idle-gap-cap-seconds` caps long gaps in the Dynamo per-trace timeline
before the adapter stamps `arrival_offset_us` and graph edge delays. Use it to
compress recorded dead air while retaining the order and dependency structure.

`--ignore-trace-delays` selects zero-gap behavior for supported trace loading
paths. For Dynamo graph input, the loader resolves this as an idle-gap cap of
zero unless an explicit idle-gap setting overrides the default. It does not
change request content or recorded hashes.

## t* and phase handling

`t*` is the snapshot instant: the point in a trace's own recorded timeline at
which a replay instance starts, expressed in microseconds from that trace's
start. `t* = 0` replays the trace in full; `t* > 0` resumes it mid-conversation.
Each per-trace value is drawn uniformly over
`[start_min_ratio, start_max_ratio] * trace_duration`, seeded per (trace, lane)
so it is deterministic given the run seed.

Agent Graph t* planning is performed per trace and per replay lane. Snapshot chopping
uses `arrival_offset_us` to split history from profiling nodes. The source
`recorded_start_unix_ms` values remain attached to the surviving nodes, and the
graph strategy computes replay timing after the selected graph is known.

When t* is disabled, the graph replays every trace in full. When a t*
window is enabled, the warmup graph primes the chain boundary nodes needed by the
profiling graph. `--burst-phase-starts` may collapse phase-leading offsets, but it
does not rewrite the recorded inter-node timing.

## Validation and limitations

The current Agent Graph timing contract has these boundaries:

- `fixed_schedule`, request-rate, user-centric, and adaptive-scaling phase
  behavior are not agent-graph replay modes;
- `--open-loop-strict` requires `--open-loop` behavior and self-contained Dynamo
  prompts;
- a graph with no usable timestamps cannot receive timestamp-based strict
  projection;
- open-loop pacing does not override graph channel dependencies except in the
  explicit strict scheduling projection;
- idle-gap compression changes replay wall time but does not change request
  payloads, hash IDs, or graph identity.

These are runtime validation boundaries, not planned implementation steps. For
the broader graph execution and materialization contracts, see [Graph Async
Dataflow Runtime](./graph-async-dataflow-runtime.md) and [Graph Worker
Materialization](./graph-worker-materialization.md).
