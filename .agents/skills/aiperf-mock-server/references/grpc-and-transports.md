<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# gRPC, UDS, and TLS transports

The mock can serve a second KServe/Riva gRPC listener, a Unix-domain socket, and TLS/HTTPS
(+ `grpcs`) alongside its cleartext TCP frontend. Source: `rust/mock-server/src/grpc.rs`,
`grpc_riva.rs`, `listener.rs`, `tls.rs`, `main.rs`, `config.rs`.

## KServe gRPC listener (`--grpc-port`)

`--grpc-port N` (env `MOCK_SERVER_GRPC_PORT`) opens a second listener serving the KServe Open
Inference Protocol v2 `GRPCInferenceService` over h2c on `--host:N`: `ModelInfer` (unary),
`ModelStreamInfer` (server-streaming), `ModelReady`, `ServerLive`, `ServerReady`. It shares
the run's `AppState` (recorder / prefix-cache / scheduler / latency) with the HTTP frontend,
reuses the same prost messages the runner's gRPC client encodes (no build-time `protoc`), and
is HTTP-only under `--processes N` (warned-and-skipped in balancer mode).

```bash
cargo run -p aiperf-mock-server -- --fast --grpc-port 8001
# target with transport.type: grpc, grpc://127.0.0.1:8001
```

### `ModelInfer` behaviors (`--grpc-behavior`, `--grpc-embedding-dim`)

`--grpc-behavior` (env `MOCK_SERVER_GRPC_BEHAVIOR`, default `auto`) picks the output tensor
for `ModelInfer` and the HTTP `/v2/.../infer` route:

| Value | Output | Auto-detect trigger (input tensors) |
|---|---|---|
| `auto` (default) | inferred | `passages` → rankings; `prompt` w/o `text_input` → images; else text |
| `text` | `text_output` BYTES (generated text) | forced |
| `rankings` | `scores` FP32 (one per passage) | forced |
| `images` | `generated_image` BYTES (base64 mock JPEG) | forced |

`--grpc-embedding-dim N` (env `MOCK_SERVER_GRPC_EMBEDDING_DIM`) overrides all of the above for
unary `ModelInfer`: it consumes the input text tensor and returns one `FP32` embedding tensor
of shape `[1, N]` (deterministic, reusing the HTTP embeddings generator), making the mock a
target for AIPerf's `kserve_v2_embeddings` gRPC endpoint (STRING-in / FP32-out, no token
semantics). Embeddings are never streamed.

### KServe gRPC e2e recipes (`test_kserve.rs`)

Driven via a Config-v2 YAML (`transport: {type: grpc}`, `endpoint.urls: ["grpc://127.0.0.1:PORT"]`,
`endpoint.type: <id>`) run as `aiperf profile --config kserve_grpc.yaml --export-level raw`:

- `kserve_v2_rankings` (unary) — output tensor `scores`, one numeric score per passage.
- `kserve_v2_images` (unary) — output `generated_image` BYTES, `data[0]` starts with `"/9j/"`
  (base64 JPEG).
- `kserve_v2_vlm` (server-streaming) — request carries `image` + `text_input` tensors,
  assembled streamed `text_output` is non-empty.

(The e2e harness enables the gRPC listener in-process; a standalone launch uses `--grpc-port`.
gRPC status 0 is mapped to HTTP `200` in the runner's records.)

## Riva ASR / TTS / NLP gRPC services

Served over the same h2c stack and prost codec as KServe (`grpc_riva.rs`), dispatched by
method path (`/nvidia.riva.*`, disjoint from KServe's `/inference.*`), so a single
`--grpc-port` serves both dialects. The mock returns deterministic canned content — the public
constants `RIVA_ASR_TRANSCRIPT`, `RIVA_NATURAL_QUERY_ANSWER`, `RIVA_INTENT_CLASS`,
`RIVA_SENTIMENT_CLASS` are the ground truth e2e assertions check.

| `--endpoint-type` | Method(s) | Returns |
|---|---|---|
| `riva_asr` | `Recognize` (unary) / `StreamingRecognize` (bidi) | canned transcript; streaming yields interim + final (`is_final`) |
| `riva_tts` | `Synthesize` (unary) / `SynthesizeOnline` (server-streaming) | deterministic PCM audio (chunked when streaming) |
| `riva_natural_query` | `NaturalQuery` | canned answer |
| `riva_text_classify` | `ClassifyText` | one sentiment result per input |
| `riva_token_classify` | `ClassifyTokens` | one labeled-token sequence per input |
| `riva_transform_text` / `riva_punctuate_text` | `TransformText` / `PunctuateText` | each input uppercased |
| `riva_analyze_intent` | `AnalyzeIntent` | canned intent classification |
| `riva_analyze_entities` | `AnalyzeEntities` | token-classification shape |

**Riva e2e recipe** (`test_riva.rs`): Config-v2 YAML with `transport: {type: grpc}`,
`endpoint.urls: ["grpc://127.0.0.1:PORT"]`, `endpoint.type: riva_asr` (etc.), `streaming: true|false`,
and — for ASR — a synthetic `audio:` block (`format: wav`, `sampleRates: [16.0]`, `depths: [16]`);
run via `aiperf profile --config riva.yaml --export-level raw`.

## Unix-domain socket (`--uds`)

`--uds <path>` (env `MOCK_SERVER_UDS`) binds a `UnixListener` and serves the **same** axum
router over it as HTTP/1.1 (the runner's UDS transport is h1-only). A stale socket file at the
path is unlinked first; a non-socket file is refused. The TCP frontend on `--port` keeps
serving in parallel. HTTP-only under `--processes N` (warned-and-skipped).

**Known runner-side limitation:** `aiperf profile` has **no UDS/`unix://` URL knob today**. The
runner transport *can* connect a `UnixStream` when `ClientConfig.uds_path` is set, but nothing
on the product path wires a URL/flag through to it (the protocol-v2 `EndpointProfileConfigV2`
has no `uds` field, and the Python frontend has no `uds`/`unix://` option). So the mock's UDS
listener is proven end-to-end by a **direct HTTP/1.1 client** over the socket
(`test_uds.rs`), not via `aiperf profile`. State this plainly when asked — a `unix://` run is
not drivable through the frontend as shipped.

## TLS / HTTPS (`--tls-cert` / `--tls-key` / `--tls-self-signed`)

A rustls frontend (ALPN `h2` + `http/1.1`) so the mock is a target for AIPerf's `https://`
(and, with `--grpc-port`, `grpcs://`) transports. Precedence: an explicit `--tls-cert`
/`--tls-key` PEM pair wins (supplying only one is an error); otherwise `--tls-self-signed`
mints a fresh in-memory cert for `127.0.0.1`/`localhost`. When TLS is on, the gRPC listener
terminates the same certificate as `grpcs` (ALPN `h2`).

| Flag | Effect |
|---|---|
| `--tls-cert <pem>` / `--tls-key <pem>` | Explicit cert chain + PKCS#8/RSA/SEC1 key (both required together) |
| `--tls-self-signed` | Fresh in-memory self-signed cert for `127.0.0.1`/`localhost` |

**HTTPS e2e recipe** (`test_tls.rs`): stands up the `--tls-self-signed` server path, then runs
`aiperf profile --config https.yaml` where the YAML sets `endpoint.url: https://127.0.0.1:PORT`,
`sslVerify: false`, `endpoint.type: chat`, `streaming: true`. `sslVerify: false` makes the
runner install a `NoCertificateVerification` verifier so the self-signed cert is accepted.
Asserts TTFT/ITL/OSL against the tuned mock (with extra TTFT tolerance for the handshake RTT).

**Known runner-side limitation (`grpcs`):** the runner's tonic `grpcs` client trusts **system
roots only** (`ClientTlsConfig::new().with_enabled_roots()` — no accept-invalid / `ssl_verify`
toggle, no custom-CA injection). A fresh self-signed cert is not in the system roots, so a
`grpcs://` `aiperf profile` run against a self-signed mock **fails the handshake by design**.
The mock's `grpcs` listener is therefore proven by a direct rustls ALPN handshake at the crate
level (`rust/mock-server/tests/tls_integration.rs`), not via `aiperf profile`. HTTPS, by
contrast, exposes `ssl_verify=false`, so the full product path is reachable over `https://`.
State both limitations plainly.
