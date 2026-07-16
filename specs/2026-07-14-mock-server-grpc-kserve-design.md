# Design: KServe gRPC target for `aiperf-mock-server`

**Date:** 2026-07-14
**Status:** Approved design, pre-implementation
**Scope:** Add a KServe Open Inference Protocol (OIP) v2 gRPC server surface to `rust/mock-server`, mirroring ai-dynamo's `GRPCInferenceService`, so AIPerf's native gRPC KServe client has a mock target.

## Motivation

`aiperf-mock-server` is today HTTP-only (axum + hyper): OpenAI chat/completions/embeddings, TGI,
rerank, image, multimodal, RAG. AIPerf's runner already ships a native gRPC **client**
(`rust/runtime/src/transport_grpc`) that dials KServe OIP v2
(`/inference.GRPCInferenceService/{ModelInfer, ModelStreamInfer, ModelReady}`), but there is no
in-repo gRPC mock to test/benchmark that client against. This adds one.

"Based on ai-dynamo" grounds the surface in
`dynamo-aiperf-native/lib/llm/src/grpc/service/kserve.rs` +
`dynamo-aiperf-native/lib/llm/src/grpc/protos/kserve.proto`, which implement exactly the
`GRPCInferenceService` (tensor requests dispatched to chat/completion flavors).

## Scope decisions (locked)

- **Surface:** KServe `GRPCInferenceService` **only** (dynamo parity). No Riva. The 9 Riva ASR/TTS/NLP
  methods AIPerf's client can also dial are explicitly out of scope (Riva is not part of ai-dynamo).
- **Proto strategy:** **hand-rolled, no build-time `protoc`.** Reuse the checked-in prost message
  types already living in `aiperf_runtime::transport_grpc::proto` — the exact structs AIPerf's own client
  encodes/decodes — guaranteeing wire parity by construction. This matches the workspace discipline:
  the runner checks in prost types by hand and hand-rolls its client with `RawBytesCodec` +
  `PathAndQuery`; no crate in the workspace uses `tonic-build`/`prost-build`/`protoc`.
- **Listener:** **opt-in** `--grpc-port` (env `MOCK_SERVER_GRPC_PORT`). HTTP is unchanged on `--port`;
  gRPC starts only when the flag is set. Zero behavior change to existing runs.
- **Balancer:** **deferred.** Ship gRPC on the single-process path. If `--processes > 1` **and**
  `--grpc-port` is set, warn and skip the gRPC listener (HTTP balancer behavior unchanged). Full
  L4-splice balancer parity for gRPC is a later follow-up.
- **RPCs:** implement only what AIPerf dials — `ModelInfer`, `ModelStreamInfer`, `ModelReady`,
  `ServerLive`, `ServerReady`. Omit `ModelMetadata` / `ModelConfig`.

## The contract (verified against code, both sides)

AIPerf's KServe binding (`rust/runtime/src/transport_grpc/binding.rs`) dials:

| RPC | Path | Kind (client) |
|---|---|---|
| `ModelInfer` | `/inference.GRPCInferenceService/ModelInfer` | unary |
| `ModelStreamInfer` | `/inference.GRPCInferenceService/ModelStreamInfer` | **server-streaming** (one request in, N responses out; `bidi_streaming_method()` is `None`) |
| `ModelReady` | `/inference.GRPCInferenceService/ModelReady` | unary |

**Request tensors** (from `rust/runtime/src/endpoints/kserve.rs::V2InferBehavior::format_payload`,
lines ~601-609, encoded by `codec.rs::encode_model_infer_request`):
- `text_input` — `BYTES` tensor, `data[0]` = the joined prompt text. (Name overridable via
  `v2_input_name`, default `text_input`.)
- `max_tokens` — optional `INT32` tensor, `data[0]` = requested output token cap, present when the
  turn sets `max_tokens`.
- plus `model_name` / `model_version` / `id` / `parameters` on the `ModelInferRequest`.

**Response tensors** (expected by `kserve.rs::parse_v2_text_response`, lines ~611-613):
- `text_output` — `BYTES` tensor, `data[0]` = generated text. (Name overridable via `v2_output_name`,
  default `text_output`.) Parser finds the tensor by name, falling back to the first tensor with data.
- For streaming, each `ModelStreamInferResponse` carries `infer_response` with a `text_output` tensor
  whose `data[0]` is the incremental chunk; `error_message` empty on success
  (`codec.rs::decode_model_stream_infer_response`).

Both sides use the **same** `aiperf_runtime::transport_grpc::proto` prost structs, so encode/decode is
symmetric and parity is structural, not test-enforced.

## Architecture

### Components (each a focused unit)

1. **`grpc` module** (`rust/mock-server/src/grpc.rs`) — new. Owns:
   - `ProstCodec<T: prost::Message>` — a minimal tonic `Codec` over prost encode/decode (the mock's
     analogue of the client's `RawBytesCodec`; avoids the `tonic-prost` dependency).
   - `KServeGrpcService` — a `tower::Service<http::Request<Body>>` that routes on `req.uri().path()`
     to the five RPC handlers, using `tonic::server::Grpc::{unary, server_streaming}` under the hood
     (this is what codegen would produce; we write it by hand to skip `protoc`).
   - Request-decode helpers: pull `text_input` / `max_tokens` out of a `ModelInferRequest` into a
     `GenRequest` for the shared generation seam.
   - Response-encode helpers: wrap generated text into a `text_output` `BYTES` `ModelInferResponse` /
     `ModelStreamInferResponse`.
   - `serve_grpc(addr, state) -> JoinHandle` — binds the gRPC socket, `TCP_NODELAY`, serves the
     service over hyper h2c on the shared runtime.

2. **`AppState` (`state.rs`)** — reused unchanged. The gRPC handlers share the one `Arc<AppState>`
   with HTTP: `recorder` (metrics), `prefix_cache`, `scheduler`, `config`, `clock_anchor`.

3. **Generation seam (`tokens.rs` + `latency.rs` + `scheduler.rs`)** — reused unchanged. gRPC is a
   new front-door onto the mock's existing brain: `tokenize_request` → usage (prompt/completion
   tokens) → `wait_for_processing` (latency/scheduler pacing) → token generation. No content logic is
   duplicated. If a small shared helper needs extracting from `handlers.rs` (the chat/text path) to
   avoid copy-paste, do that as a targeted refactor rather than re-implementing.

4. **`config.rs`** — add `grpc_port: Option<u16>` (clap `--grpc-port`, env `MOCK_SERVER_GRPC_PORT`).

5. **`main.rs`** — after building `AppState`, if `config.grpc_port` is `Some` and not skipped by
   balancer mode, `tokio::spawn(serve_grpc(...))` alongside the existing HTTP accept loop; both run to
   completion on the shared runtime.

### Data flow (ModelInfer, unary)

```
client ModelInferRequest (prost)
  -> ProstCodec decode
  -> extract text_input (BYTES) + optional max_tokens (INT32)
  -> GenRequest -> tokenize_request -> usage
  -> wait_for_processing (latency/scheduler pacing, shared clock_anchor)
  -> generate output text (tokens.rs)
  -> recorder.record_* (shared metrics)
  -> ModelInferResponse { outputs: [text_output BYTES = full text], model_name, id }
  -> ProstCodec encode -> client
```

### Data flow (ModelStreamInfer, server-streaming)

Same head; the tail streams one `ModelStreamInferResponse { infer_response: { text_output = chunk } }`
per generated token/piece, paced by the same inter-token latency the SSE path uses, then closes the
stream. `error_message` stays empty on success. This produces genuine TTFT (first response) and ITL
(inter-response) timings for the client to measure.

### Health RPCs

`ModelReady` → `ModelReadyResponse { ready: true }`. `ServerLive` → `{ live: true }`,
`ServerReady` → `{ ready: true }` (trivial prost messages; add the two tiny `ServerLive*`/`ServerReady*`
structs to the mock's codec module if not already present in `aiperf_runtime::transport_grpc::proto` — the
client only decodes `ModelReadyResponse`, so `ServerLive`/`ServerReady` messages are defined locally).

## Error handling

- Malformed `ModelInferRequest` (missing/empty `text_input`, wrong datatype) → tonic `Status`
  `invalid_argument` with a descriptive message. Mirrors HTTP 4xx behavior.
- The mock's existing error-injection knob (`state.inject_error()`) is honored on the gRPC path too →
  tonic `Status` `internal` (maps to the client's error accounting like an HTTP 5xx).
- Non-finite/oversized tensors: reuse the `CodecError`-style guards; never panic on the wire.
- gRPC `Status` codes chosen so AIPerf's `grpc_status_to_http` maps them to the HTTP-equivalent error
  buckets the metrics layer already understands.

## Testing

1. **Unit (`rust/mock-server`)** — spin the `KServeGrpcService` on an ephemeral port on the test
   runtime; hand-build a `ModelInferRequest` with a `text_input` tensor; assert:
   - unary `ModelInfer` returns a `text_output` tensor with non-empty text and honors `max_tokens`;
   - `ModelStreamInfer` yields ≥1 chunk and the concatenation matches the unary text shape;
   - `ModelReady` returns `ready: true`;
   - malformed request → `invalid_argument`.
2. **Wire round-trip (`rust/mock-server/tests/grpc_integration.rs`)** — a real tonic client drives
   `serve_grpc` over h2 using AIPerf's own `aiperf_runtime::transport_grpc` encode/decode helpers, exercising
   the framing/trailers/prost round-trip the handler unit tests bypass (unary, server-streaming,
   readiness).
3. **Full-stack e2e (`rust/e2e/tests/test_kserve_grpc_endpoint.rs`)** — the real `python -m aiperf
   profile` CLI (Python frontend → native `aiperf` binary → its production gRPC KServe client) drives
   the mock's `serve_grpc` listener, enabled via the harness's `MockServer::start_with_grpc`
   /`AIPerfHarness::new_with_grpc` (a second in-process listener sharing the mock's `AppState`),
   selected via a Config-v2 YAML with `transport.type: grpc` + `grpc://127.0.0.1:<port>` + endpoint
   `kserve_v2_infer`, for both `streaming: true` (`ModelStreamInfer`) and `streaming: false`
   (`ModelInfer`). Asserts the run succeeds and `request_count` matches. This is the layer that caught
   the reasoning-model empty-stream bug (below) — the default harness model `openai/gpt-oss-120b` is a
   reasoning model, so streaming exercises the reasoning path that lower-level tests missed. Use
   `127.0.0.1` (not `localhost`) to avoid the IPv6/IPv4 mock mismatch.

**Reasoning-model streaming (bug found by the e2e).** The mock lowers a KServe request to a synthetic
`ChatCompletionRequest`; for a reasoning model (`gpt-oss`/`qwen`) with a small `max_tokens` budget the
tokenizer spends the whole budget on reasoning tokens, leaving zero *output* tokens. The unary path
tolerates an empty `text_output`, but a server-streaming response that yields **zero** messages is a
failed request to strict gRPC clients — including AIPerf's own runner. Both handlers therefore emit the
full generation (`generated_tokens` = reasoning tokens followed by output tokens), folding reasoning
into the single KServe `text_output` (KServe text has no separate reasoning channel) so the stream is
never spuriously empty. Regression-locked by `grpc::tests::model_stream_infer_reasoning_model_is_not_empty`.

## Extensibility notes (repo non-negotiable)

- The router dispatches on method path; adding a Riva service later means registering more
  `(path -> handler)` entries and their prost messages — the request→generation→response mapping seam
  is reusable. The `ProstCodec` is generic over `prost::Message`. We are shipping KServe only, but the
  structure does not hardcode it as the sole possibility (a second service is an added route, not a
  rewrite). Documented in the module `//!` header.
- Content generation stays behind the existing `tokens`/`latency`/`scheduler` seams; the gRPC front
  door adds no new content policy.

## Files touched

- **New:** `rust/mock-server/src/grpc.rs`
- **Edit:** `rust/mock-server/src/main.rs` (spawn gRPC listener), `rust/mock-server/src/config.rs`
  (`grpc_port` flag/env), `rust/mock-server/src/lib.rs` (module export), possibly a small helper
  extraction in `rust/mock-server/src/handlers.rs` to share the generation head.
- **New tests:** unit in `rust/mock-server` + e2e in `rust/e2e/tests/`.
- **Docs:** update the `aiperf-mock-server` crate-table row in the four agent files
  (`AGENTS.md`/`CLAUDE.md`/`.github/copilot-instructions.md`/`.cursor/rules/python.mdc`) to note the
  gRPC surface, and `llms.txt`; run `python tools/check_agent_files_sync.py`.

## Non-goals

- Riva ASR/TTS/NLP gRPC services.
- `ModelMetadata` / `ModelConfig` RPCs.
- gRPC in `--processes N` balancer mode (deferred; warn+skip).
- TLS/`grpcs://` (plaintext h2c first; TLS is a later add if needed — the client supports `grpcs://`).
- `build.rs` / `protoc` / `tonic-build`.
