<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoints

## Purpose

An endpoint is the API-dialect adapter between the workload (a prepared turn) and
a canonical decoded request/response shape. `aiperf_runtime::endpoints` owns
endpoint identity, the `Endpoint` trait, every native dialect, the shared
body-build skeleton, the input-ISL extractor, and each adapter's capability
descriptor. Endpoints are transport-neutral: the same implementation binds to
HTTP or gRPC.

## Built

### The `Endpoint` trait

```rust
pub trait Endpoint: Debug + Send + Sync {
    fn descriptor(&self) -> &'static EndpointDescriptor;
    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan>;
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;
}
```

`format_payload` builds the declarative `BodyPlan` at lowering (see
[endpoint-body-construction.md](endpoint-body-construction.md)). `parse_response`
parses one canonical decoded server response into a `ParsedResponse { perf_ns,
data, usage }`; the collector derives TTFT and tokens from these.
`extract_payload_inputs` is a single pass over the built body yielding tokenizable
text and media counts for input-side ISL accounting. Parse behavior preserves the
current vendor-specific response contracts and is guarded by fixtures; absent
usage fields stay absent, and non-text output-token observations are supported.

### Dialects

- OpenAI-compatible: chat, completions, embeddings, responses.
- Anthropic messages.
- KServe: `kserve_chat`, `kserve_completions`, `kserve_embeddings`,
  `kserve_v1_predict` (HTTP `instances`/`predictions`), and the five gRPC OIP v2
  dialects (`kserve_v2_infer`, `kserve_v2_embeddings`, `kserve_v2_rankings`,
  `kserve_v2_vlm`, `kserve_v2_images`).
- Riva ASR/TTS/NLP gRPC dialects; `ResponseData::Audio` carries synthesized
  bytes, sample rate, encoding, and optional duration.
- The open protocol-v2 `vllm_generate` token-in/token-out factory with exact
  raw-array accounting.
- The full tier-2 set (`tier2.rs`, `tier2/`): NIM/Cohere/Hugging Face rankings,
  image generation/edit, video generation, Hugging Face generate, NIM embeddings,
  image retrieval, Solido RAG, raw, and template.

### Identity and registry

Endpoint identity is an open string `EndpointId`, not a closed enum or central
static table. `EndpointRegistry` (frozen via `AIPerfExtension` composition) owns
open IDs, co-located descriptors, deterministic factory/alias registration,
identity-free raw/effective configuration, worker-local prepared bindings and
dense keys, readiness policy, and compile-once raw/template state. Runner
capability publication and protocol-v2 validation consume the frozen registry.
The object-safe `HttpEndpointBinding` (in `transport::http`) and
`GrpcEndpointBinding` (in `transport::grpc`) own the transport-specific wire
lowering and decoding back into the canonical response shape (see
[http-transport.md](http-transport.md) and [grpc-transport.md](grpc-transport.md)).

## Source anchors

- `rust/runtime/src/endpoints/` (`endpoints.rs` `Endpoint` trait, `registry.rs`
  `EndpointRegistry`/`EndpointId`, `chat.rs`, `kserve.rs`, `riva.rs`,
  `anthropic.rs`, `vllm_generate.rs`, `tier2.rs`, `tier2/`, `metadata.rs`,
  `usage.rs`, `extraction.rs`).
- `rust/runtime/src/transport/http/` and `rust/runtime/src/transport/grpc/binding.rs`.
- Tier-2 online endpoint tests and per-endpoint e2e tests under `rust/e2e/tests/`.
