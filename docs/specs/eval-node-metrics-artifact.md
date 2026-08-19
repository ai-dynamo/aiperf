<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Eval node metrics artifact

## Purpose

`aiperf eval` measures each NativeGraph model request through the normal
worker-local `RequestObserver` and `NativeMetricsObserver` path. This record
specifies the host-owned artifact that exposes those completed per-node records
without changing the scored reward JSON or letting an untrusted task package
choose an output destination.

The artifact is intentionally a record sidecar, not a new metrics engine or a
synthetic profile report. It uses the canonical record projection, whose
top-level fields are `metadata`, `metrics`, optional `trace_data`, and `error`,
so TTFT, ITL, token counts, latency, error classification, and correlation
fields keep the same meanings as other native record artifacts. HTTP status is
retained only by raw-record capture and is not a canonical record JSONL field.

## Built

`aiperf eval --records-output <path>` writes canonical compact JSONL for
schema-1.1 `native_graph` evaluation. The option is absent by default and is
rejected before provisioning for standard packages and externally driven
NativeGraph packages. The host opens and validates the destination before the
Docker episode starts; package content cannot set or override it.

One suite-owned writer spans all resolved episodes. Each completed model node
moves its terminal `CapturedRecord` through the bounded graph-evidence channel
to the coordinator, which appends one row in completion order. Worker token
callbacks remain IO-free and worker-local. The row uses the same default metric
projection and serialization as the native record JSONL exporter, including
canonical correlation identity, token counts, TTFT, ITL, request latency, and
error classification. Undefined TTFT or ITL remains absent rather than being
replaced with zero.

The artifact contains derived measurement fields only. It does not enable raw
model request or response capture; `ModelCapturePolicy` remains the authority
for raw exchange retention. The writer flushes each completed row and is
finished before reward JSON is printed. Opening, serialization, writing, or
flushing failure fails the evaluation. The host-selected sidecar is not inserted
into package lifecycle `artifacts`, so the existing reward JSON contract is
unchanged.

Unit and integration coverage proves record ownership across the evidence
boundary, disabled-export compatibility, suite-level append behavior,
pre-provision destination validation, and flush/write error propagation. The
product-level `aiperf-e2e-tests` coverage drives the compiled CLI against the
in-process deterministic `aiperf-mock-server` over real HTTP/SSE with two model
nodes. It verifies exactly two canonical rows, node-attributable correlation,
fixed token accounting, bounded TTFT/ITL, no terminal usage or `[DONE]` rows,
no record errors, and unchanged reward JSON. Separately, the mock-server
recorder asserts that both streamed model requests complete with HTTP 200.

## Future requirements

A later host-owned export policy may add format selection or registered remote
exporters. It must remain outside the task package and preserve the initial JSONL
and reward contracts.

## Source anchors

- `rust/cli/src/eval.rs` — eval CLI parsing and package-mode dispatch.
- `rust/cli/src/eval/native_graph.rs` — NativeGraph eval orchestration, artifact
  lifetime, final flush, and reward JSON rendering.
- `rust/runtime/src/eval/native_graph/episode_runner.rs` — episode callback
  construction and suite-owned artifact propagation.
- `rust/runtime/src/eval/native_graph/model_runtime.rs` — live trace execution
  bridge.
- `rust/runtime/src/engine/graph_execution.rs` — graph record events and the
  coordinator-owned NativeGraph evidence boundary.
- `rust/runtime/src/metrics.rs` — worker-local `NativeMetricsObserver` and
  terminal `CapturedRecord` construction.
- `rust/runtime/src/engine/records.rs` — canonical JSONL record projection.
- `rust/runtime/src/engine/record_lane.rs` — eval record writer and local record
  artifact lifecycle.
- `rust/e2e-tests/tests/test_eval_node_metrics.rs` — compiled-CLI deterministic
  HTTP/SSE contract proof.
