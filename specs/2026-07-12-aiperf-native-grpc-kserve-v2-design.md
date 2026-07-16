<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native KServe/Riva endpoints and gRPC transport through runner protocol v2

Status: built

## Decision

AIPerf ports PR 664's KServe endpoint family and Open Inference Protocol (OIP)
gRPC transport, plus the complete NVIDIA Riva ASR/TTS/NLP surface, into native
Rust. The product projection is runner protocol v2 only: no new `EndpointType`
variant, protocol-v1 DTO field, Python transport plugin, or `plugins.yaml` entry
is added.

“V1” has two independent meanings here. `kserve_v1_predict` is a supported
KServe endpoint dialect (`instances` / `predictions` over HTTP), but it is still
selected and executed only by an AIPerf runner protocol-v2 authored operation.
The five OIP gRPC bindings are KServe V2 dialects. This distinction is pinned by
registry and subprocess tests.

The source behavior is the complete PR 664 checkout at
`/home/anthony/tmp/pr-664-grpc-kserve` (`f9f92223`), especially:

- `src/aiperf/endpoints/kserve_v1_predict.py:16-109`
- `src/aiperf/endpoints/kserve_v2_infer.py:16-155`
- `src/aiperf/endpoints/kserve_v2_embeddings.py:17-130`
- `src/aiperf/endpoints/kserve_v2_rankings.py:12-132`
- `src/aiperf/endpoints/kserve_v2_vlm.py:17-111`
- `src/aiperf/endpoints/kserve_v2_images.py:17-146`
- `src/aiperf/transports/grpc/grpc_client.py:1-204`
- `src/aiperf/transports/grpc/grpc_transport.py:1-751`
- `src/aiperf/transports/grpc/kserve_v2_serializers.py:1-347`
- `src/aiperf/transports/grpc/status_mapping.py:1-39`
- `src/aiperf/transports/grpc/trace_data.py:1-51`

The Riva surface is grounded in reference branch `ajc/riva` at commit
`a391cfe27a333915b0f058bd05f21c932c77a898`.

Where those Python branches use runtime plugins, `grpc.aio`, wall-clock timing,
and shared mutable controller state, the Rust implementation follows this
workspace's canonical open registries, `Clock`, Tonic, worker-local ownership,
and thread-per-core placement.

## Endpoint ownership

`aiperf_runtime::endpoints::EndpointRegistry` registers nine open KServe factories:

- `kserve_chat`, `kserve_completions`, and `kserve_embeddings`
- `kserve_v1_predict`
- `kserve_v2_infer`, `kserve_v2_embeddings`, `kserve_v2_rankings`,
  `kserve_v2_vlm`, and `kserve_v2_images`

The OpenAI-compatible aliases retain the existing request/response behavior but
own KServe paths and readiness policy. The six native KServe behaviors own
selector extraction, tensor payload formation, response auto-detection,
embedding reshaping, ranking indexes, VLM media, typed image parameters, and
image decoding. Configuration remains identity-free; endpoint IDs live in the
registry and prepared references.

It also registers nine additional open, engine-v2-only Riva factories,
again without adding a closed `EndpointType` variant:

- `riva_asr`, with unary `Recognize` and bidirectional `StreamingRecognize`;
- `riva_tts`, with unary `Synthesize` and server-streaming `SynthesizeOnline`;
  and
- `riva_text_classify`, `riva_token_classify`, `riva_transform_text`,
  `riva_punctuate_text`, `riva_natural_query`, `riva_analyze_intent`, and
  `riva_analyze_entities` over unary Riva NLP RPCs.

The Riva implementations preserve the reference defaults, first-turn audio/text
extraction, ASR audio chunking, transcript concatenation, TTS PCM duration
calculation, compact JSON NLP results, and top-answer selection.
`ResponseData::Audio` carries synthesized bytes, sample rate, encoding, and
optional duration through the common prepared-endpoint seam. Riva has no model
readiness RPC, so its dialect and wire bindings explicitly report readiness as
unsupported instead of fabricating a probe.

None of these factories exposes `legacy_endpoint()`. Consequently the closed
protocol-v1 compatibility lookup returns `NoLegacyAdapter`, even for
`kserve_v1_predict`. Both HTTP and gRPC execution consume worker-local
`PreparedEndpoint` bindings.

The native HTTP binding expands PR 664's `{model_name}` placeholder for both
profile-owned custom paths and descriptor paths before applying existing `/v1`
prefix de-duplication. The real KServe V1 subprocess proof exercises the
descriptor path rather than supplying an authored override.

## Native gRPC transport module

`aiperf_runtime::transport_grpc` is the Clock-injected gRPC transport module of the
`aiperf` crate. Its open seams are:

- `GrpcEndpointBinding` for endpoint-specific wire encoding, RPC paths,
  readiness, and response decoding — admitting unary, server-streaming, and
  optional bidirectional methods, plus an ordered config-first request-message
  encoder;
- `GrpcEndpointBindingFactory` for startup composition; and
- `GrpcBindingRegistryBuilder` / `GrpcBindingRegistry` for deterministic,
  duplicate-rejecting lookup and worker-local preparation.

The built-in registry binds the five gRPC-capable KServe V2 dialects together
with the Riva ASR/TTS/NLP bindings. The checked-in `grpc_predict_v2.proto` is
byte-identical to PR 664, and the checked-in Prost DTOs for Riva are grounded in
the reference `riva_common.proto`, `riva_audio.proto`, `riva_asr.proto`,
`riva_tts.proto`, and `riva_nlp.proto` field numbers. Checked-in Prost DTOs
avoid a host `protoc` dependency, while the raw Tonic codec keeps the channel
layer independent of that schema. Unary, server-streaming, and bidirectional
cardinalities remain behavior objects behind the binding trait rather than a
closed endpoint-kind switch.

Canonical JSON converts to and from OIP protobuf for BYTES, signed and unsigned
integers, FP16/FP32/FP64, BOOL, request/input parameters, typed response
contents, raw numeric buffers, and length-prefixed raw BYTES. Typed contents
win over raw buffers exactly as in the Python source. Streaming envelopes keep
successful messages and fail on in-band `error_message` without discarding
already received responses.
One byte-exact test compares a native request with the serialized output of PR
664's Python `KServeV2GrpcSerializer`, in addition to semantic protobuf tests.

## Transport lifecycle

`GrpcTransport` is worker-local and owns no global runtime or locks. It supports
`grpc://` and `grpcs://`, WebPKI roots, 256 MiB default message limits, lowercase
ASCII metadata, request/correlation/session metadata, unary inference,
server-streaming inference, bidirectional streaming inference, model readiness,
and all 17 native gRPC-to-HTTP status mappings used by common metrics.
Bidirectional and server-streaming request messages are sent through Tonic's
raw streaming API; every unframed request message is retained, and request
message count and bytes are accounted in the existing Clock-derived trace.

The three reuse strategies match the HTTP/native policy:

- pooled: one HTTP/2 channel per target;
- never: a fresh channel per request; and
- sticky user sessions: one channel lease per correlation ID, released on the
  final turn or any terminal failure.

All application-visible timestamps and deadlines use the injected `Clock`.
Channel readiness is capped at 30 seconds and also bounded by the whole-request
deadline. Authored cancellation is armed only after the RPC future has first
been submitted, so connection establishment does not consume the cancellation
delay. Equal-time deadlines win deterministically. Traces retain connection or
reuse, send, initial metadata, response-message, byte/chunk, terminal status,
and native gRPC facts. Tonic does not expose Hyper's DNS/TCP/TLS sub-events, so
those HTTP-only fields are intentionally unavailable rather than fabricated.

## Runner protocol-v2 projection

The runner registry contains a `grpc` real-clock transport and executable
`grpc + scheduled` pair. `http + scheduled` is also registered so HTTP-only
KServe dialects, including KServe V1 Predict, are product-reachable without
falling through protocol v1. Python Config v2 accepts explicit `grpc://` /
`grpcs://` URLs only with `transport.type: grpc`, rejects mixed schemes and the
legacy Python `endpoint.transport` knob, and preserves the authored transport in
the v2 envelope.

The gRPC pair strictly rejects HTTP schemes, HTTP/2 endpoint flags, unsupported
gRPC endpoint bindings, readiness retries not yet composed by the runner, and
unregistered sidecars. This is fail-closed capability behavior. Transport
readiness itself is implemented in the leaf module; product readiness remains
disabled until the common protocol-v2 preparation lifecycle owns it.

`GrpcTransportSink` consumes only `PreparedHttpEndpoint::Prepared`; receiving a
legacy endpoint is an error. It merges endpoint/request metadata, dispatches
canonical prepared JSON through the dense gRPC binding table, parses responses
through the endpoint, emits first-token/classified-token/usage/endpoint/terminal
observations, and maps timing into the common metrics accumulator.
The scheduler-free command schema retains the effective model selected for
each turn; the sink uses that value as the OIP protobuf `model_name`. This keeps
round-robin multi-model composition correct instead of silently using the run's
first model.

One worker runs on the coordinator's current-thread runtime. Multiple workers
use one OS thread, current-thread Tokio runtime, `LocalSet`, Clock, endpoint
table, binding table, and Tonic channel set per worker. Owned commands cross
bounded queues; observer events replay on the coordinator. No channel,
endpoint, or hot-path observer is shared across worker threads.

## Compatibility facts and limits

The scheduled execution seam still names its generic command/result types
`Http*`. Until that seam is renamed, the gRPC sink projects raw gRPC exchanges
into its compatibility record shape and places `grpc-status` and
`grpc-message` in response metadata. Actual wire I/O is always native Tonic and
the aggregate metrics use native gRPC timing. This naming debt does not permit
runner protocol-v1 dispatch.

The current common prepared execution factory applies one routing/reuse/session
policy per run. The gRPC pair therefore rejects endpoint profiles whose URL,
reuse, session-header, or timeout policy differs from the default instead of
silently routing them incorrectly. Sidecars and readiness retries likewise
remain unadvertised until a complete protocol-v2 adapter is registered.

## Conformance

The implementation is guarded by:

- endpoint parity tests for all eighteen KServe and Riva IDs, payloads,
  parsing, readiness, selectors, and absence of protocol-v1 adapters;
- protobuf tests for every tensor and parameter class, every Riva RPC, exact
  method/cardinality checks, typed/raw precedence, streaming envelopes, registry
  contents, and all status mappings;
- a real loopback Tonic server covering unary, server-streaming, and
  bidirectional (ASR) RPCs, readiness, metadata, pooled/never/sticky channels,
  errors, and post-send cancellation;
- runner inventory tests distinguishing static compatibility from executable
  v2 pairs;
- a full runner subprocess proof for `grpc + scheduled`, including two
  concurrent OS-thread workers and two round-robin models whose exact OIP
  `model_name` values are captured, plus a strict runner `validate`/`execute`
  proof using an inline Rust-side WAV fixture for Riva;
- a user-facing `aiperf profile --config ...` proof that crosses Python Config
  v2, capability negotiation, orchestration, the strict runner, and a real
  mock Tonic/OIP server before asserting the native-v2 artifact; and
- a full runner subprocess proof that `kserve_v1_predict` executes over HTTP
  with `protocol_version: 2`.

No test imports or invokes the PR's Python gRPC implementation.
