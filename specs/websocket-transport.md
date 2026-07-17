<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebSocket transport

## Purpose

Benchmark WebSocket inference: a persistent WSS turn stream with an HTTPS/SSE
fallback. The WebSocket transport is a future capability; this record separates
its requirements from the reusable transport, content, streaming, and
measurement prerequisites already present in the runtime.

## Built

The runtime provides the prerequisites the future transport must compose over:

- `Clock`, worker-local `WorkerSink`, `ExecutionSinkBuilder`, and `LocalSet`
  execution define transport timing and placement.
- The segment store and `BodyPlan` preserve content as handles and
  pre-serialized bytes until the selected wire materializer consumes it.
- The HTTP transport provides a byte-preserving SSE decoder and streaming
  response path that the future fallback can reuse.
- The gRPC transport provides worker-local bidirectional framed-message
  execution through endpoint bindings.
- `RequestObserver` supplies first-token, classified-token, usage,
  endpoint-metric, and terminal observations shared by transports.
- The extension registry provides open transport and endpoint registration with
  duplicate rejection and a frozen application inventory.

No WebSocket transport, WebSocket endpoint binding, or WS-frame `BodyPlan`
materializer is registered.

## Future requirements

A Clock-injected `transport_ws` module, a third transport beside HTTP and gRPC.
It reuses the segment content IR and the `BodyPlan` contract (see
[dataset.md](dataset.md) and [endpoint-body-construction.md](endpoint-body-construction.md))
and adds a third `BodyPlan` materializer (`BodyPlan → WS frames`) that splices
pre-serialized segment bytes into event envelopes as a sequence of text frames —
concat, not re-serialize, no per-request `Value`. The single-`Full<Bytes>` /
`SendCompletion` rule stays HTTP-local and does not apply.

### Dialects — separate protocols, one transport

The transport is dialect-neutral; each protocol is its own binding behind the
registry and must not be conflated:

- **Codex** — the reference dialect: a persistent WSS turn stream, WS-primary
  with an HTTPS/SSE fallback carrying the identical event stream. A turn is one
  request frame → streamed incremental output and reasoning deltas → a terminal
  completion event, with turns chaining via server-side thread continuation. It
  is not the generic OpenAI Responses API.
- **Realtime** — a distinct dialect: OpenAI's bidirectional voice/text WS with its
  own client/server event schema and base64 audio inside JSON events.

They share transport, framing mechanics, measurement, and fallback — not an event
schema. The generic Responses API, if added, is its own dialect.

### Realization onto the seams

- Transport: TLS over the shared connector; one persistent socket per worker
  driven by a single `!Send` task (unsplit `WebSocketStream`) on the worker's
  `LocalSet`, using upstream `tokio-tungstenite`.
- Fallback: first-class. On WS-upgrade/connect failure the transport latches WS
  off for the rest of the run (one-shot atomic latch) and degrades to the
  dialect's same event stream over HTTP SSE, reusing the existing HTTP streaming
  decoder — not a mid-stream resume. One shared event decoder feeds both WS and
  SSE; only framing differs.
- Content: `message`/`text`/`token-ids` segments splice into event JSON; Realtime
  audio is a `media` segment base64-encoded once at lowering and spliced by
  reference. Frames are text-only; a binary frame is a protocol violation.
- Measurement: per-frame `RequestObserver` measurement (gRPC bidi is the in-tree
  precedent for framed messages over a persistent connection). Completeness is
  gated on the terminal event, not the socket FIN, with a benign-close taxonomy.

## Source anchors

- Reusable prerequisites: `rust/runtime/src/transport/grpc/binding.rs`
  (bidirectional framed messages), `rust/runtime/src/transport/http/sse/` (SSE
  decoder), `rust/runtime/src/body_plan.rs` (materializer contract),
  `rust/runtime/src/dataset/segment.rs` (content handles), and
  `rust/loadgen-core/src/sink.rs` (`RequestObserver`).
- Placement and registration seams:
  `rust/runtime/src/engine/turn_execution.rs` (`WorkerSink`,
  `ExecutionSinkBuilder`) and `rust/runtime/src/extensions/`.
