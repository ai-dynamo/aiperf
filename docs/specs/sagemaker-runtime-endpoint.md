<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AWS SageMaker Runtime endpoint

## Purpose

Benchmark AWS SageMaker Runtime's native invocation APIs: `InvokeEndpoint`
(single request/response) and `InvokeEndpointWithResponseStream` (streamed
response over AWS's binary `application/vnd.amazon.eventstream` framing).

AIPerf also understands SageMaker in a separate, unrelated sense: a *dataset*
format (`sagemaker_data_capture`, see [dataset.md](dataset.md)) that replays
captured SageMaker request/response pairs recorded by SageMaker Data Capture.
That loader does not exercise the SageMaker Runtime wire protocol — it
deserializes archived JSON. This record covers the transport/endpoint dialect
that does speak `/endpoints/{EndpointName}/invocations` over the wire.

## Built

- `SageMakerFactory` (`rust/runtime/src/endpoints/sagemaker.rs`), a single
  factory/endpoint id (`sagemaker`, alias `sagemaker_invoke`) exposing both
  invocation paths on one `EndpointDescriptor`, following the same
  endpoint-path/streaming-path convention as `huggingface_generate` (TGI) in
  `rust/runtime/src/endpoints/tier2.rs` rather than KServe's separate-factory
  pattern:
  - `endpoint_path = "/endpoints/{model_name}/invocations"` — non-streaming
    `InvokeEndpoint`. `format_payload` builds an OpenAI-chat-shaped JSON body
    (reusing existing chat body-building); `parse_response` parses the
    OpenAI-chat JSON response body directly.
  - `streaming_path = "/endpoints/{model_name}/invocations-response-stream"`
    — streaming `InvokeEndpointWithResponseStream`, selected by `--streaming`
    at request-binding time (the same `supports_streaming`-gated path switch
    every other dual-path dialect uses; see
    `rust/runtime/src/endpoints/config.rs`). Response decoding goes through
    the eventstream reader, then each `PayloadPart.Bytes` frame payload is
    parsed as an OpenAI-chat-completion-chunk, reusing existing
    chunk-parsing logic from `chat.rs`/`chat_chunk.rs`.
  The factory is registered in `rust/runtime/src/endpoints/registry.rs`
  alongside the KServe registrations and re-exported from `mod.rs`.
- Mock-server routes in `rust/mock-server/src/app.rs`/`handlers.rs`, alongside
  the existing KServe/OpenAI routes:
  - `POST /endpoints/{endpoint_name}/invocations` — non-streaming
    `InvokeEndpoint`.
  - `POST /endpoints/{endpoint_name}/invocations-response-stream` — streaming
    `InvokeEndpointWithResponseStream`, encoded as AWS eventstream binary
    frames (`Content-Type: application/vnd.amazon.eventstream`) via the
    `sse_to_eventstream` adapter, which converts the shared SSE-chunk
    generation path's `data:` lines into `EventStreamMessage::payload_part`
    frames and drops the SSE `[DONE]` sentinel (AWS eventstream has no
    terminal sentinel; the stream ends at HTTP body EOF).
  - `endpoint_name` is the AWS path-segment equivalent of KServe's
    `{model_name}` and reuses the existing `{model_name}` templating
    convention on both the client (`endpoint_binding.rs`) and mock-server
    (`axum::extract::Path`) sides — no new templating token.
  - Request-body sniffing: the handler accepts either an OpenAI-chat-shaped
    body (`messages` key present) or a SageMaker JumpStart/DJL-shaped body
    (`inputs` key present, optional `parameters`), detected by key presence.
    The response is always OpenAI chat-completion shaped regardless of which
    request shape was sent, reusing the same response models as the
    `chat_completions` handler.
- AWS `application/vnd.amazon.eventstream` binary frame codec in
  `rust/runtime/src/transport/core/eventstream.rs` (sibling to `sse.rs`):
  transport-neutral, no HTTP or SSE dependency, so both the mock-server
  encoder and client decoder share it without a `core -> http` dependency.
  - `EventStreamMessage { headers, payload }`: `[4B total length][4B headers
    length][4B prelude CRC32][headers][payload][4B message CRC32]`, with a
    symmetric encoder (`encode`) and incremental decoder
    (`EventStreamDecoder`), byte-exact round-trip, independently
    unit-tested.
  - **`payload` is the raw inner chat-completion-chunk JSON bytes, with no
    base64/JSON envelope.** The AWS SageMaker `PayloadPart` event shape's
    `Bytes` member carries the Smithy `eventpayload` trait, meaning the raw
    eventstream frame payload IS the blob value directly — there is no
    `{"PayloadPart":{"Bytes":"<base64>"}}` wrapping on the wire. (An earlier
    draft of this implementation built that envelope; it was wrong and was
    caught by wire-compatibility testing against a genuine `boto3`
    `sagemaker-runtime` client — see Verification.)
  - Terminal condition: AWS SageMaker eventstream responses end at HTTP body
    EOF (no `[DONE]` sentinel like SSE); the decoder treats stream close as
    terminal.
- Client transport: `rust/runtime/src/transport/http/client/http_client.rs`
  gates on `is_sse || is_eventstream` for streaming-response box-pinning, and
  `eventstream_to_sse` adapts raw AWS eventstream binary chunks into
  synthetic `"data: <json>\n\n"` byte chunks (via `futures::stream::unfold`),
  feeding the existing, unmodified SSE reader
  (`rust/runtime/src/transport/http/sse/reader.rs`) — only frame decoding
  differs from the SSE path, not downstream token/usage/TTFT measurement.
- Out of scope, unchanged: AWS SigV4 request signing and IAM auth. This
  dialect targets AIPerf's own mock server and SageMaker-compatible test
  endpoints for load generation and measurement, not the real authenticated
  AWS API surface.

## Verification

- `rust/e2e-tests/tests/test_sagemaker_endpoint.rs` drives `aiperf profile` against
  `aiperf-mock-server` with `--endpoint-type sagemaker`, both without and
  with `--streaming`, inspecting raw per-record output: response status, parsed
  OpenAI-chat-completion body shape and content, request/response `model`,
  prompt/completion token usage (non-streaming), and reassembled streamed
  content plus ack-vs-start timing ordering (streaming).
- Unit coverage: round-trip encode→decode tests for the eventstream codec
  (`eventstream.rs`), including corrupted-CRC rejection and multi-message
  buffering across arbitrary chunk boundaries; mock-server body-shape
  sniffing tests for both `messages` and `inputs` request shapes.
- Wire-format compatibility was additionally verified manually against a
  genuine, downloaded AWS client — `boto3`'s `sagemaker-runtime` client,
  exercising `invoke_endpoint` and `invoke_endpoint_with_response_stream` for
  both accepted request-body shapes, run standalone (not wired into the
  `e2e` crate). This caught and drove the fix for the `PayloadPart.Bytes`
  double-envelope bug described above — a hand-written HTTP test harness
  would not have caught it, since it would have decoded the wire format the
  same (wrong) way the mock server encoded it.

## Source anchors

- `rust/runtime/src/endpoints/sagemaker.rs`, `registry.rs`, `mod.rs` — client
  endpoint dialect.
- `rust/runtime/src/transport/core/eventstream.rs` — frame codec.
- `rust/runtime/src/transport/http/client/http_client.rs` — client-side
  `eventstream_to_sse` adapter and streaming-response framing selection.
- `rust/mock-server/src/app.rs`, `handlers.rs`, `models.rs` — mock-server
  routes, body-shape sniffing, and the `sse_to_eventstream` adapter.
- `rust/runtime/src/dataset/loader/trace.rs` (`SageMakerDataCaptureDatasetLoader`,
  `SageMakerDataCaptureComposer`) — the separate, unrelated dataset-format
  SageMaker support.
- `rust/e2e-tests/tests/test_sagemaker_data_capture.rs` — existing e2e coverage for
  the dataset loader, not the runtime endpoint.
- `rust/e2e-tests/tests/test_sagemaker_endpoint.rs` — e2e coverage for this record.
