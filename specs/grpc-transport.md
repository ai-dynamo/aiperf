<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gRPC transport

## Purpose

The `aiperf_runtime::transport::grpc` module is the Clock-injected Tonic gRPC
stack. It carries the KServe Open Inference Protocol (OIP) v2 endpoint family and
the NVIDIA Riva ASR/TTS/NLP surface, gated behind the `grpc` Cargo feature
(default in the CLI feature set). It contributes only a `WorkerSink` and its
builder to the shared execution model; there is no gRPC worker loop.

## Built

### Binding registry

The open seams are `GrpcEndpointBinding` (endpoint-specific wire encoding, RPC
paths, readiness, response decoding — admitting unary, server-streaming, and
optional bidirectional methods plus an ordered config-first request encoder),
`GrpcEndpointBindingFactory` (startup composition), and
`GrpcBindingRegistryBuilder` / `GrpcBindingRegistry` (deterministic,
duplicate-rejecting lookup and worker-local preparation). Unary,
server-streaming, and bidirectional cardinalities are behavior objects behind the
binding trait, not a closed endpoint-kind switch.

The built-in registry binds the five gRPC-capable KServe V2 dialects
(`kserve_v2_infer`, `kserve_v2_embeddings`, `kserve_v2_rankings`,
`kserve_v2_vlm`, `kserve_v2_images`) together with the Riva bindings (`riva_asr`
with unary `Recognize` and bidirectional `StreamingRecognize`; `riva_tts` with
unary `Synthesize` and server-streaming `SynthesizeOnline`; and the seven Riva
NLP unary RPCs). See [endpoints.md](endpoints.md) for the endpoint families.

### Wire codec

Checked-in Prost DTOs (from `grpc_predict_v2.proto` and the Riva `.proto` field
numbers) avoid a host `protoc` dependency; a raw Tonic codec keeps the channel
layer independent of the schema. Canonical JSON converts to and from OIP protobuf
for BYTES, signed/unsigned integers, FP16/FP32/FP64, BOOL, request/input
parameters, typed response contents, raw numeric buffers, and length-prefixed raw
BYTES. Typed contents win over raw buffers. Streaming envelopes keep successful
messages and fail on in-band `error_message` without discarding already-received
responses. Endpoint parsing sees the same canonical JSON shape after gRPC
protobuf decoding as after HTTP JSON decoding.

### Transport lifecycle

`GrpcTransport` is worker-local and owns no global runtime or locks. It supports
`grpc://` and `grpcs://`, WebPKI roots, a 256 MiB default message limit,
lowercase-ASCII metadata, request/correlation/session metadata, unary,
server-streaming, and bidirectional inference, model readiness, and the native
gRPC-to-HTTP status mappings used by common metrics. The three reuse strategies
match the HTTP policy: pooled (one HTTP/2 channel per target), never (a fresh
channel per request), and sticky user sessions (one channel lease per correlation
ID, released on the final turn or terminal failure). All timestamps and deadlines
use the injected `Clock`; channel readiness is capped at 30 s and bounded by the
whole-request deadline. Authored cancellation arms only after the RPC future has
first been submitted, so connection establishment does not consume the
cancellation delay. Tonic does not expose Hyper's DNS/TCP/TLS sub-events, so those
HTTP-only trace fields are absent rather than fabricated.

`GrpcTransportSink` consumes only prepared endpoint bindings, merges
endpoint/request metadata, dispatches canonical prepared JSON through the dense
binding table, parses responses through the endpoint, and emits
first-token/classified-token/usage/endpoint/terminal observations. It preserves
the per-turn effective model as the OIP `model_name`, so round-robin multi-model
composition stays correct. `supports_response_streaming()` returns `false`, so
the shared worker loop never opens a live response channel for gRPC. Riva has no
model-readiness RPC, so its bindings report readiness as unsupported.

## Source anchors

- `rust/runtime/src/transport/grpc/` (`binding.rs`, `codec.rs`, `raw_codec.rs`,
  `proto.rs`, `riva_binding.rs`, `riva_codec.rs`, `riva_proto.rs`, `transport.rs`,
  `sink.rs`, `models.rs`).
- `rust/e2e-tests/tests/{test_kserve_grpc_endpoint.rs,test_riva.rs}`,
  `rust/cli/tests/riva_grpc_v2_stdio.rs`, `rust/cli/tests/grpc_v2_stdio.rs`.
