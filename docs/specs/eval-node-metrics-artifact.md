<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Eval node metrics artifact

## Purpose

`aiperf eval` already measures each NativeGraph model request through the normal
worker-local `RequestObserver` and `NativeMetricsObserver` path. This record
specifies the host-owned artifact that exposes those completed per-node records
without changing the scored reward JSON or letting an untrusted task package
choose an output destination.

The artifact is intentionally a record sidecar, not a new metrics engine or a
synthetic profile report. It uses the canonical record projection so TTFT, ITL,
token counts, latency, status, and correlation fields keep the same meanings as
other native record artifacts.

## Built

NativeGraph currently dispatches model nodes through the shared graph execution
path. `NativeMetricsObserver` produces terminal `CapturedRecord` values, but the
NativeGraph evidence sink retains only the completed-record count. `aiperf eval`
prints reward JSON and lifecycle artifacts; it does not currently expose
per-node request records.

Package-authored NativeGraph configuration is strict and does not contain host
artifact destinations. This is a trust boundary: a task package must not select
host filesystem paths or exporter endpoints.

## Future requirements

### Host-owned CLI contract

`aiperf eval` will accept `--records-output <path>` for schema-1.1 NativeGraph
evaluation. The option is absent by default and is rejected before provisioning
for modes that cannot emit NativeGraph model-node records. The destination is
opened and validated before the Docker episode starts; package content cannot
set or override it.

The initial format is canonical JSONL only. A later host-owned export policy may
add format selection or registered remote exporters, but it must remain outside
the task package and must not alter the initial JSONL contract.

### Record lifecycle

One writer spans the resolved suite. Each completed graph model node emits one
canonical record row in completion order. The coordinator, not worker token
callbacks, owns the writer so request hot paths remain IO-free and worker-local.
The existing bounded graph-evidence channel carries an owned `CapturedRecord` to
that coordinator boundary.

The output contains derived measurement fields only. It does not enable raw
model request or response capture; `ModelCapturePolicy` remains the authority
for raw exchange retention. Undefined TTFT or ITL remains absent in the
canonical row rather than being replaced with zero.

The writer is explicitly flushed before reward JSON is printed. Opening,
serialization, writing, or flushing failure makes the eval fail: an explicitly
requested local artifact is part of successful command completion. The existing
reward JSON shape and its package/lifecycle `artifacts` semantics remain
unchanged; the host-selected sidecar is not inserted into that array.

### Verification

Focused tests must prove that record events preserve their `CapturedRecord` over
the evidence channel, export-disabled execution is unchanged, and suite-level
output appends rather than truncates across episodes. A deterministic
`aiperf-mock-server` NativeGraph end-to-end test must use fixed TTFT and ITL and
assert exactly one JSONL record per model node, correlation identity, token
counts, TTFT/ITL within the existing transport tolerance, and unchanged valid
reward JSON.

## Source anchors

- `rust/cli/src/eval.rs` — eval CLI parsing and package-mode dispatch.
- `rust/cli/src/eval/native_graph.rs` — NativeGraph eval orchestration and reward
  JSON rendering.
- `rust/runtime/src/eval/native_graph/episode_runner.rs` — episode callback
  construction and coordinator-owned lifecycle.
- `rust/runtime/src/eval/native_graph/model_runtime.rs` — live trace execution
  bridge.
- `rust/runtime/src/engine/graph_execution.rs` — graph execution events and the
  NativeGraph evidence boundary.
- `rust/runtime/src/metrics.rs` — worker-local `NativeMetricsObserver` and
  terminal `CapturedRecord` construction.
- `rust/runtime/src/engine/records.rs` — canonical JSONL record projection.
- `rust/runtime/src/engine/record_lane.rs` — existing local record artifact
  writer lifecycle.
