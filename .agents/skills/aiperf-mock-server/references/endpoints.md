<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# HTTP route / endpoint catalog

Every route is registered in `rust/mock-server/src/app.rs`; handlers in
`rust/mock-server/src/handlers.rs`. For each, the `aiperf profile --endpoint-type <id>` that
targets it is listed (endpoint IDs are the runner's, from `aiperf::endpoints`). gRPC routes
and the Riva/KServe-gRPC services live in `references/grpc-and-transports.md`.

## LLM text/chat

| Route | `--endpoint-type` | Notes |
|---|---|---|
| `POST /v1/chat/completions` | `chat` | OpenAI chat; real SSE when `stream:true`. Also the target for tool-calls, accuracy, error, usage features |
| `POST /v1/completions` | `completions` | OpenAI text completions; real SSE |
| `POST /v1/messages` | (Anthropic `messages`) | Anthropic Messages shape; carries the Anthropic-only `cache_read_input_tokens`/`cache_creation_input_tokens` usage fields |
| `POST /v1/responses` | `responses` | OpenAI Responses API. Non-stream `{object:"response", status:"completed", output:[...], usage}`; streaming `response.created` / `response.output_text.delta` / `response.completed` events |
| `POST /inference/v1/generate` | `vllm_generate` | vLLM/Dynamo token-native (token-in / token-out). Consumes `token_ids`, returns integer `choices[].token_ids`. Non-streaming only |

**vllm_generate e2e recipe** (`test_new_routes.rs`):

```bash
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type vllm_generate \
  --request-count 6 --concurrency 2 --workers-max 1 \
  --synthetic-input-tokens-mean 64 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8 --output-tokens-stddev 0 \
  --export-level raw --ui simple
```

Asserts each record `status == 200`, `choices[0].token_ids` is exactly 8 integers,
`usage.prompt_tokens == 64`, `usage.completion_tokens == 8`.

**responses streaming e2e recipe**: `--endpoint-type responses --streaming` with the same
synthetic ISL/OSL knobs; asserts exactly 8 `response.output_text.delta` frames and the
terminal `response.completed` `usage.output_tokens == 8`.

## `/openai/v1/*` aliases (KServe OpenAI-compatible)

The runner's KServe chat/completions/embeddings factories default to these paths, which
dispatch to the identical OpenAI handlers above.

| Route | `--endpoint-type` |
|---|---|
| `POST /openai/v1/chat/completions` | `kserve_chat` |
| `POST /openai/v1/completions` | `kserve_completions` |
| `POST /openai/v1/embeddings` | `kserve_embeddings` |
| `GET /openai/v1/models` | (model listing) |

**kserve_chat alias e2e recipe** (`test_new_routes.rs`): `--endpoint-type kserve_chat
--streaming` with tuned latency; verifies the alias routes to the chat handler and reproduces
TTFT/ITL/OSL.

## Embeddings, rerank, TGI, image, RAG

| Route | `--endpoint-type` | Notes |
|---|---|---|
| `POST /v1/embeddings` | `embeddings` / `nim_embeddings` | Deterministic 768-dim embeddings |
| `POST /v1/ranking` | `rankings` / `nim_rankings` | NIM reranker; `rankings[].relevance_score` |
| `POST /rerank` | `hf_tei_rankings` | HF TEI reranker; `results[].score` |
| `POST /v2/rerank` | `cohere_rankings` | Cohere reranker; `results[].relevance_score` |
| `POST /generate`, `POST /generate_stream` | `huggingface_generate` | TGI |
| `POST /v1/images/generations`, `POST /v1/images/edits` | `image_generation` / `image_edit` | Base64 mock JPEG |
| `POST /v1/image/infer` **and** `POST /v1/infer` | `image_retrieval` | NIM image retrieval → bounding boxes. `/v1/infer` is the default path when `image_retrieval` is driven without an `--endpoint` override |
| `POST /v1/custom-multimodal` | (custom) | Custom multimodal echo |
| `POST /rag/api/prompt` | `solido_rag` | Solido RAG |

**image_retrieval / `/v1/infer` alias e2e recipe** (`test_kserve.rs`):

```bash
aiperf profile --model nvidia/page-elements-v2 --url http://127.0.0.1:8000 \
  --endpoint-type image_retrieval \
  --image-width-mean 64 --image-height-mean 64 \
  --request-count 6 --concurrency 2 --workers-max 1 --ui none --export-level raw
```

Driving `image_retrieval` with no `--endpoint` override resolves to the default `/v1/infer`;
asserts `status == 200` and a `data` array of bounding boxes.

## KServe Open Inference Protocol over HTTP

| Route | `--endpoint-type` | Notes |
|---|---|---|
| `POST /v2/models/{model}/infer` | `kserve_v2_infer` / `kserve_v2_vlm` / `kserve_v2_rankings` / `kserve_v2_images` | Behavior auto-detected from input tensor names (`text_input` → text; `query`+`passages` → `scores`; `prompt` → `generated_image`), overridable with `--grpc-behavior`. Text is non-streaming JSON here (streaming KServe is exercised over gRPC) |
| `POST /v1/models/{model}:predict` (same segment as `/v1/models/{id}`) | `kserve_v1_predict` | Returns `predictions[].output` |
| `GET /v2/models/{model}/ready`, `GET /v2/health/ready`, `GET /v2/models/{model}/infer`... | (readiness) | Always ready |

The runner drives KServe v2 over **either** transport; these HTTP routes mirror the gRPC
lowering so a `transport.type: http` run against a `kserve_v*` endpoint has a target.

**KServe HTTP e2e recipes** (`test_kserve.rs`) use a Config-v2 YAML (`transport: {type: http}`,
`endpoint.type: kserve_v2_infer` / `kserve_v1_predict`, `endpoint.urls: ["http://127.0.0.1:PORT"]`)
run via `aiperf profile --config kserve.yaml --export-level raw`. v2 asserts the first output
tensor is `text_output`/`BYTES` with non-empty `data[0]`; v1 asserts `predictions[0].output` is
a non-empty string.

## Model listing, accuracy, health

| Route | Purpose |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | OpenAI model listing/info. Advertises `--models` (or a default set) unioned with models seen via traffic |
| `GET /health`, `GET /` | Liveness / config echo |
| `GET /accuracy` | Live accuracy tally (`references/accuracy.md`) |
| `GET /metrics` + backend dialects + `/dcgm*/metrics` | Telemetry (`references/telemetry.md`) |
