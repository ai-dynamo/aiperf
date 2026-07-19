<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AWS SageMaker Runtime endpoint

## Purpose

Benchmark AWS SageMaker Runtime's native invocation APIs: `InvokeEndpoint`
(single request/response) and `InvokeEndpointWithResponseStream` (streamed
response over AWS's binary `application/vnd.amazon.eventstream` framing). This
is a future capability; this record separates its requirements from the
reusable endpoint, transport, and mock-server prerequisites already present in
the runtime.

Today AIPerf only understands SageMaker in one narrow sense: a *dataset* format
(`sagemaker_data_capture`, see [dataset.md](dataset.md)) that replays captured
SageMaker request/response pairs recorded by SageMaker Data Capture. That loader
does not exercise the SageMaker Runtime wire protocol — it deserializes archived
JSON. No client transport speaks `/endpoints/{EndpointName}/invocations`, and no
mock-server route serves it.

## Built

The runtime provides the prerequisites the future dialect must compose over:

- The `EndpointFactory` / `PreparedEndpoint` registry seam
  (`rust/runtime/src/endpoints/registry.rs`) with `{model_name}` URL-path
  templating (`rust/runtime/src/transport/http/transport/endpoint_binding.rs:443-449`),
  demonstrated by the KServe dialect family
  (`rust/runtime/src/endpoints/kserve.rs`) — one factory per distinct
  path/verb, a shared prepared-endpoint wrapper, and per-behavior
  `format_payload`/`parse_response`.
- A generic, dialect-agnostic SSE reader
  (`rust/runtime/src/transport/core/sse.rs`,
  `rust/runtime/src/transport/http/sse/reader.rs`) reached through a single
  content-type check in `rust/runtime/src/transport/http/client/http_client.rs:672-680`
  (`is_sse = content_type.starts_with("text/event-stream")`).
- The mock server's shared SSE-chunk encoding helpers
  (`rust/mock-server/src/handlers.rs`: `sse_chunk`, `sse_chunk_ser`,
  `write_sse_into`) and axum dynamic-path routing
  (`rust/mock-server/src/app.rs`, `Path` extractors in `handlers.rs`), used by
  existing streaming handlers (`chat_stream`, `messages_stream`, `tgi_stream`).
- OpenAI-chat-completion request/response models and chunk-parsing logic
  (`rust/mock-server/src/models.rs`; client-side `rust/runtime/src/endpoints/chat.rs`,
  `chat_chunk.rs`) that the new dialect reuses rather than duplicates.
- The `sagemaker_data_capture` dataset loader and composer
  (`rust/runtime/src/dataset/loader/trace.rs:51-56`), which remains a
  dataset-format concern independent of this transport-level work.

No SageMaker Runtime transport path, endpoint binding, or AWS eventstream
frame codec is registered or implemented.

## Future requirements

### Mock-server routes

Two new axum routes in `rust/mock-server/src/app.rs`, alongside the existing
KServe/OpenAI routes:

- `POST /endpoints/{endpoint_name}/invocations` — non-streaming `InvokeEndpoint`.
- `POST /endpoints/{endpoint_name}/invocations-response-stream` — streaming
  `InvokeEndpointWithResponseStream`.

`endpoint_name` is the AWS path-segment equivalent of KServe's `{model_name}`
and reuses the existing `{model_name}` templating convention on both the
client (`endpoint_binding.rs`) and mock-server (`axum::extract::Path`) sides —
no new templating token.

Request-body sniffing: a handler accepts either an OpenAI-chat-shaped body
(`messages` key present) or a SageMaker JumpStart/DJL-shaped body (`inputs`
key present, optional `parameters`), detected by key presence. The response is
always OpenAI chat-completion shaped regardless of which request shape was
sent, reusing the same response models as the `chat_completions` handler — no
mirrored JumpStart response encoding path.

### AWS eventstream binary framing

`application/vnd.amazon.eventstream` is AWS's binary streaming frame format
(distinct from SSE): each message is `[4B total length][4B headers
length][4B prelude CRC32][headers][payload][4B message CRC32]`. This has no
existing seam in AIPerf — SSE parsing today is a single hardcoded
content-type branch (`http_client.rs:672-680`) with no per-dialect decoder
hook.

New transport-neutral module `rust/runtime/src/transport/core/eventstream.rs`
(sibling to `sse.rs`):

- `EventStreamMessage { headers, payload }` with a symmetric encoder and
  decoder, byte-exact round-trip, independently unit-testable.
- Client transport extends the single `is_sse` boolean at
  `http_client.rs:672-680` into a small `StreamFraming` selection
  (`Sse | EventStream | None`) chosen from response `Content-Type`. Each
  framing has its own reader, but both feed the same downstream
  token/usage/TTFT recording path — only frame decoding differs, not
  measurement semantics.
- Terminal condition: AWS SageMaker eventstream responses end at HTTP body
  EOF (no `[DONE]` sentinel like SSE); the decoder treats stream close as
  terminal.
- Mock-server encoder: a `eventstream_chunk` helper in `handlers.rs` beside
  `sse_chunk`, wrapping each streamed chat-completion chunk as
  `{"PayloadPart": {"Bytes": <base64 JSON>}}` per real SageMaker streaming
  semantics, framed through the new encoder. Response `Content-Type:
  application/vnd.amazon.eventstream`.

### Client endpoint dialect

New `rust/runtime/src/endpoints/sagemaker.rs`, following the KServe
factory/behavior pattern with **two factories** (not a `supports_streaming`
flag) so eventstream-specific decode logic stays isolated to the streaming
factory:

- `SageMakerInvokeFactory` — non-streaming.
  `EndpointDescriptor.endpoint_path = "/endpoints/{model_name}/invocations"`.
  `format_payload` builds an OpenAI-chat-shaped JSON body (reusing existing
  chat body-building); `parse_response` parses the OpenAI-chat JSON response
  body directly.
- `SageMakerInvokeStreamFactory` — streaming.
  `EndpointDescriptor.streaming_path =
  "/endpoints/{model_name}/invocations-response-stream"`. Response decoding
  goes through the new eventstream reader, then each `PayloadPart.Bytes`
  payload is parsed as an OpenAI-chat-completion-chunk, reusing existing
  chunk-parsing logic from `chat.rs`/`chat_chunk.rs`.

Both factories register in `rust/runtime/src/endpoints/registry.rs` alongside
the KServe registrations and re-export from `mod.rs`.

Out of scope: AWS SigV4 request signing and IAM auth. This dialect targets
AIPerf's own mock server and SageMaker-compatible test endpoints for load
generation and measurement, not the real authenticated AWS API surface.

### Verification

Per this repository's end-to-end testing requirement for new transport/endpoint
behavior, a new `rust/e2e/tests/test_sagemaker_endpoint.rs` drives `aiperf
profile` against `aiperf-mock-server` for both routes with fixed TTFT/ITL
jitter coefficients at zero, analytic scheduling, and pinned
tokenizer/ISL/OSL, asserting TTFT, generated-token ITL, request latency, ISL,
OSL, model, streaming mode, response content, status, and errors per raw
record — for both the non-streaming and eventstream-streaming routes, and for
both accepted request-body shapes.

Additional unit coverage:

- Round-trip encode→decode test for the new eventstream codec.
- Mock-server body-shape sniffing test for both `messages` and `inputs`
  request shapes.

## Source anchors

- `rust/runtime/src/endpoints/kserve.rs`, `registry.rs`, `mod.rs` — dialect
  pattern this record's client work extends.
- `rust/runtime/src/transport/http/transport/endpoint_binding.rs`,
  `rust/runtime/src/transport/http/client/http_client.rs`,
  `rust/runtime/src/transport/core/sse.rs`,
  `rust/runtime/src/transport/http/sse/reader.rs` — transport seams this
  record's eventstream work extends.
- `rust/mock-server/src/app.rs`, `handlers.rs`, `models.rs` — mock-server
  seams this record's route work extends.
- `rust/runtime/src/dataset/loader/trace.rs` (`SageMakerDataCaptureDatasetLoader`,
  `SageMakerDataCaptureComposer`) — the existing, unrelated dataset-format
  SageMaker support.
- `rust/e2e/tests/test_sagemaker_data_capture.rs` — existing e2e coverage for
  the dataset loader, not the runtime endpoint.
