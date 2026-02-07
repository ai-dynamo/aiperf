<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile KServe InferenceServices with AIPerf

AIPerf provides first-class support for benchmarking [KServe](https://kserve.github.io/website/) InferenceServices. KServe is the standard model inference platform on Kubernetes, supporting multiple inference runtimes (vLLM, Triton, TRT-LLM, TensorFlow Serving) through three protocol families.

## Endpoint Types

AIPerf provides five KServe-specific endpoint types:

| Endpoint Type | Protocol | URL Path | Streaming | Token Metrics | Use Case |
|---|---|---|---|---|---|
| `kserve_chat` | OpenAI-compatible | `/openai/v1/chat/completions` | Yes | Yes | LLMs via vLLM/TRT-LLM on KServe |
| `kserve_completions` | OpenAI-compatible | `/openai/v1/completions` | Yes | Yes | Text completions via vLLM/TRT-LLM on KServe |
| `kserve_embeddings` | OpenAI-compatible | `/openai/v1/embeddings` | No | No | Embedding models on KServe |
| `kserve_v2_infer` | V2 Open Inference Protocol | `/v2/models/{model_name}/infer` | No | Yes | Triton/TRT-LLM tensor inference |
| `kserve_v1_predict` | V1 TensorFlow Serving | `/v1/models/{model_name}:predict` | No | No | Legacy TF Serving-style models |

**Token Metrics**: When "Yes", AIPerf computes token-based metrics (input/output token counts, tokens per second). When "No", only request-level metrics (latency, throughput) are available.

### How KServe Endpoints Differ from Standard Endpoints

The `kserve_chat`, `kserve_completions`, and `kserve_embeddings` endpoints use the same request/response format as the standard `chat`, `completions`, and `embeddings` endpoints. The only difference is the URL path: KServe prefixes OpenAI-compatible routes with `/openai` (e.g., `/openai/v1/chat/completions` instead of `/v1/chat/completions`).

The `kserve_v2_infer` and `kserve_v1_predict` endpoints use entirely different wire formats (tensor-based and instance-based, respectively) and are implemented as separate endpoint classes.

---

## Section 1. KServe OpenAI-Compatible Endpoints

KServe InferenceServices with vLLM or TRT-LLM backends expose an OpenAI-compatible API under the `/openai` prefix. Use `kserve_chat`, `kserve_completions`, or `kserve_embeddings` to benchmark these.

### Chat Completions (Streaming)

```bash
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_chat \
    --streaming \
    --request-count 100 \
    --concurrency 10
```

### Chat Completions (Non-Streaming)

```bash
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_chat \
    --request-count 100 \
    --concurrency 10
```

### Text Completions

```bash
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_completions \
    --streaming \
    --request-count 100 \
    --concurrency 10
```

### Embeddings

```bash
aiperf profile \
    --model bge-small-en-v1.5 \
    --url http://embedding-isvc.kserve-test.svc.cluster.local \
    --endpoint-type kserve_embeddings \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --request-count 50 \
    --concurrency 8
```

### Why Not Just Use `--endpoint` (Custom Endpoint)?

You could use `--endpoint-type chat --endpoint /openai/v1/chat/completions`, but the KServe-specific endpoint types provide:

- **Discoverability**: `--endpoint-type kserve_chat` is self-documenting
- **Correct defaults**: `service_kind` is set to `kserve` for artifact naming
- **Health check paths**: Pre-configured for KServe health endpoints (e.g., `/openai/v1/models`)

---

## Section 2. KServe V2 Open Inference Protocol

The V2 protocol is used by Triton Inference Server and TRT-LLM via KServe. It wraps inputs as typed tensors (BYTES, INT32, FP32, etc.).

### Request/Response Format

**Request:**
```json
{
  "inputs": [
    {
      "name": "text_input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["What is machine learning?"]
    },
    {
      "name": "max_tokens",
      "shape": [1],
      "datatype": "INT32",
      "data": [256]
    }
  ],
  "parameters": {
    "temperature": 0.7
  }
}
```

**Response:**
```json
{
  "outputs": [
    {
      "name": "text_output",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["Machine learning is a subset of artificial intelligence..."]
    }
  ]
}
```

### Basic Usage

```bash
aiperf profile \
    --model my-trtllm-model \
    --url http://triton-isvc.default.svc.cluster.local:8000 \
    --endpoint-type kserve_v2_infer \
    --request-count 50 \
    --concurrency 4
```

The model name is automatically embedded in the URL path. For example, with `--model my-trtllm-model`, requests are sent to `/v2/models/my-trtllm-model/infer`.

### With Output Token Limit

When `--output-tokens-mean` is specified (or the dataset includes `max_tokens`), an additional `max_tokens` INT32 tensor input is included in the request:

```bash
aiperf profile \
    --model my-trtllm-model \
    --url http://triton-isvc.default.svc.cluster.local:8000 \
    --endpoint-type kserve_v2_infer \
    --output-tokens-mean 256 \
    --request-count 50
```

### Custom Tensor Names

Different model configurations may use different tensor names. Override the defaults with `--extra-inputs`:

```bash
aiperf profile \
    --model my-model \
    --url http://triton-isvc.default.svc.cluster.local:8000 \
    --endpoint-type kserve_v2_infer \
    --extra-inputs v2_input_name:INPUT_TEXT \
    --extra-inputs v2_output_name:OUTPUT_TEXT \
    --request-count 50
```

| Parameter | Default | Description |
|---|---|---|
| `v2_input_name` | `text_input` | Name of the input BYTES tensor |
| `v2_output_name` | `text_output` | Name of the output BYTES tensor to parse |

If the configured output tensor name is not found in the response, the endpoint falls back to using the first output tensor that contains data.

### Passing Model Parameters

Any `--extra-inputs` values that are not `v2_input_name` or `v2_output_name` are passed as the `"parameters"` section of the V2 request:

```bash
aiperf profile \
    --model my-model \
    --url http://triton-isvc.default.svc.cluster.local:8000 \
    --endpoint-type kserve_v2_infer \
    --extra-inputs temperature:0.7 \
    --extra-inputs top_k:40 \
    --request-count 50
```

This produces requests with:
```json
{
  "inputs": [...],
  "parameters": {"temperature": 0.7, "top_k": 40}
}
```

---

## Section 3. KServe V1 Predict (TensorFlow Serving)

The V1 protocol uses the TensorFlow Serving instance-based format. This is used by legacy KServe InferenceServices and some custom model servers.

### Request/Response Format

**Request:**
```json
{
  "instances": [
    {"text": "Classify this document"}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"output": "This document is about technology"}
  ]
}
```

### Basic Usage

```bash
aiperf profile \
    --model sklearn-iris \
    --url http://sklearn-isvc.default.svc.cluster.local \
    --endpoint-type kserve_v1_predict \
    --request-count 100 \
    --concurrency 8
```

The model name is automatically embedded in the URL path. For example, with `--model sklearn-iris`, requests are sent to `/v1/models/sklearn-iris:predict`.

### Custom Field Names

Override the default input/output field names:

```bash
aiperf profile \
    --model my-model \
    --url http://my-isvc.default.svc.cluster.local \
    --endpoint-type kserve_v1_predict \
    --extra-inputs v1_input_field:input_text \
    --extra-inputs v1_output_field:result \
    --request-count 50
```

| Parameter | Default | Description |
|---|---|---|
| `v1_input_field` | `text` | Field name for the input text in each instance |
| `v1_output_field` | `output` | Field name to extract from each prediction |

### Response Parsing

The V1 endpoint supports multiple prediction formats:

- **Dict predictions**: `{"predictions": [{"output": "text"}]}` -- uses the configured output field. If the configured field is not found in the dict, auto-detection is tried on the prediction object.
- **String predictions**: `{"predictions": ["text"]}` -- uses the string directly
- **Auto-detection fallback**: If `predictions` is missing or empty, falls back to auto-detection which tries extraction in order: embeddings format, rankings format, then text fields (`text`, `content`, `response`, `output`, `result`)

---

## Section 4. Path Template Substitution

The `kserve_v2_infer` and `kserve_v1_predict` endpoints embed the model name in the URL path using `{model_name}` templates. This substitution also works with `--endpoint` (custom endpoint):

```bash
# Using --endpoint with a template
aiperf profile \
    --model my-model \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --endpoint /v2/models/{model_name}/infer \
    --request-count 10
```

This sends requests to `http://localhost:8000/v2/models/my-model/infer`.

The template is resolved from the `--model` value (specifically the first model name when multiple are provided).

---

## Section 5. Server Metrics with KServe

KServe exposes server metrics through two channels:

1. **Runtime metrics** (vLLM, Triton) -- served on the inference port at `/metrics`, auto-discovered by AIPerf
2. **Queue proxy (qpext) sidecar** -- available on port 9088, provides aggregated Knative and runtime metrics

### Runtime Metrics (Auto-Collected)

Server metrics collection is **enabled by default**. AIPerf automatically collects Prometheus metrics from the inference endpoint's `base_url + /metrics` path without any additional flags. For most KServe + vLLM/Triton deployments, runtime metrics are collected automatically:

```bash
# Runtime metrics collected automatically from http://qwen-llm.kserve-test.svc.cluster.local/metrics
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_chat \
    --streaming \
    --request-count 100
```

Use `--no-server-metrics` to disable automatic collection. Use `--server-metrics` to specify *additional* custom Prometheus endpoints beyond the auto-discovered one.

### Collecting Queue Proxy Metrics

To collect KServe queue proxy metrics from the qpext sidecar, use `--server-metrics` to add the qpext endpoint:

```bash
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_chat \
    --streaming \
    --server-metrics http://qwen-llm.kserve-test.svc.cluster.local:9088/metrics \
    --request-count 100
```

You can specify additional queue proxy metrics alongside the auto-collected runtime metrics:

```bash
aiperf profile \
    --model qwen2.5-7b \
    --url http://qwen-llm.kserve-test.svc.cluster.local \
    --endpoint-type kserve_chat \
    --streaming \
    --server-metrics http://qwen-llm.kserve-test.svc.cluster.local/metrics \
    --server-metrics http://qwen-llm.kserve-test.svc.cluster.local:9088/metrics \
    --request-count 100
```

---

## Section 6. Health Check Paths

Each KServe endpoint type includes a `health_path` in its metadata for pre-flight server readiness validation. These paths follow the KServe standard health check endpoints:

| Endpoint Type | Health Path | Description |
|---|---|---|
| `kserve_chat` | `/openai/v1/models` | Lists available OpenAI-compatible models |
| `kserve_completions` | `/openai/v1/models` | Lists available OpenAI-compatible models |
| `kserve_embeddings` | `/openai/v1/models` | Lists available OpenAI-compatible models |
| `kserve_v2_infer` | `/v2/models/{model_name}/ready` | V2 model readiness check |
| `kserve_v1_predict` | `/v1/models/{model_name}` | V1 model metadata/status |

Health paths that contain `{model_name}` are resolved using the same template substitution as endpoint paths.

---

## Section 7. Choosing the Right Endpoint Type

| Scenario | Endpoint Type | Notes |
|---|---|---|
| KServe + vLLM (chat) | `kserve_chat` | Full streaming + multi-modal support |
| KServe + vLLM (completions) | `kserve_completions` | Text completion with streaming |
| KServe + vLLM (embeddings) | `kserve_embeddings` | Vector embeddings |
| KServe + Triton (text) | `kserve_v2_infer` | Wraps text as BYTES tensors |
| KServe + TRT-LLM via Triton | `kserve_v2_infer` | Standard Triton text pipeline |
| KServe + TF Serving model | `kserve_v1_predict` | Legacy instance-based format |
| KServe + custom model server | `kserve_v1_predict` or `template` | Depends on API format |
| Non-KServe vLLM/TRT-LLM | `chat` or `completions` | Use standard endpoints for direct deployments |

### When to Use KServe Endpoints vs Standard Endpoints

Use **KServe endpoints** when:
- Your model is deployed as a KServe InferenceService
- The inference URL routes through KServe's ingress gateway
- You need the `/openai` prefix for the OpenAI-compatible API

Use **standard endpoints** (`chat`, `completions`, `embeddings`) when:
- Your model is deployed directly (not through KServe)
- The server exposes the standard OpenAI API at `/v1/...`

---

## Troubleshooting

### Connection Refused or 404

- Verify the InferenceService is in `Ready` state: `kubectl get inferenceservice`
- Check the correct port and hostname (KServe default is port 80 via the ingress gateway)
- For V2/V1 endpoints, ensure the model name in `--model` matches the KServe InferenceService name

### V2 Tensor Name Mismatch

If you see empty or error responses with `kserve_v2_infer`, the tensor names may not match. Check your model configuration:

```bash
# Query the V2 model metadata to discover tensor names
curl http://your-isvc:8000/v2/models/your-model
```

Then use `--extra-inputs v2_input_name:ACTUAL_NAME --extra-inputs v2_output_name:ACTUAL_NAME`.

### V1 Field Name Mismatch

If predictions are not parsed correctly with `kserve_v1_predict`, check the response format:

```bash
curl -X POST http://your-isvc/v1/models/your-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"text": "test"}]}'
```

Then adjust with `--extra-inputs v1_input_field:ACTUAL_FIELD --extra-inputs v1_output_field:ACTUAL_FIELD`.

### Streaming Not Supported

The `kserve_v2_infer` and `kserve_v1_predict` endpoints do not support streaming (`--streaming` will be automatically disabled with a warning). For streaming LLM inference on KServe, use `kserve_chat` or `kserve_completions` which route through the OpenAI-compatible API.
