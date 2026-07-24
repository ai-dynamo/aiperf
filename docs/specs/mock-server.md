<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mock server

## Purpose

`aiperf-mock-server` is a standalone HTTP/gRPC inference target with deterministic
response generation. It is a separately launched benchmark and test target — not
supervised by a profile run — and gives AIPerf's transports and endpoints a
controllable server with pinned latency, usage, and error behavior.

## Built

### HTTP surface

An axum + hyper server hosts OpenAI chat/completions/embeddings, TGI, rerank,
image, multimodal, and RAG endpoints with deterministic response generation,
configurable latency models, error injection, usage accounting, TLS, UDS, a
multi-process L4 balancer, accuracy fixtures, and request recording. A shared
`RequestCtx` seam owns generation, latency, and prefix-cache behavior.

### gRPC surface

An opt-in `--grpc-port` (env `MOCK_SERVER_GRPC_PORT`) starts a KServe Open
Inference Protocol v2 `GRPCInferenceService` target (and Riva bindings) on a
separate socket sharing `AppState`; HTTP on `--port` is unchanged. A hand-routed
`tower` service dispatches `ModelInfer` (unary), `ModelStreamInfer`
(server-streaming), `ModelReady`, `ServerLive`, and `ServerReady` by method path,
reusing the same checked-in prost messages AIPerf's own client encodes — wire
parity by construction, no build-time `protoc`. Requests lower to a synthetic
chat completion through the shared `RequestCtx` (`text_input` → prompt,
`text_output` ← generated text); reasoning-model streaming folds reasoning tokens
into `text_output` so a reasoning model never yields an empty gRPC stream. Under
`--processes N`, a set `--grpc-port` is warned-and-skipped (HTTP-only).

### Telemetry

A DCGM-style GPU telemetry faker and Prometheus/OpenMetrics endpoints support the
telemetry side channels (see [telemetry.md](telemetry.md)).

## Source anchors

- `rust/mock-server/src/` (`app.rs`, `handlers.rs`, `grpc.rs`, `grpc_riva.rs`,
  `latency.rs`, `prefix_cache.rs`, `scheduler.rs`, `balancer.rs`, `tls.rs`,
  `dcgm.rs`, `prom.rs`, `accuracy.rs`, `tokens.rs`, `state.rs`).
- `rust/mock-server/tests/` (`integration.rs`, `grpc_integration.rs`,
  `accuracy_integration.rs`, `tls_integration.rs`, `balancer.rs`).
- `rust/e2e/tests/test_kserve_grpc_endpoint.rs` (full-stack profile → mock).
