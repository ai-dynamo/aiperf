---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile KServe Endpoints
---

# Profile KServe Endpoints

AIPerf supports nine KServe endpoint dialects through Config v2 and the native
Rust runner. KServe V1 Predict remains supported, but “V1” here names the
KServe wire dialect; every endpoint on this page is selected through AIPerf
runner protocol v2.

## Endpoint matrix

| Endpoint type | Shape | HTTP | Native gRPC | Streaming |
|---|---|---:|---:|---:|
| `kserve_chat` | OpenAI Chat Completions | Yes | No | HTTP SSE |
| `kserve_completions` | OpenAI Completions | Yes | No | HTTP SSE |
| `kserve_embeddings` | OpenAI Embeddings | Yes | No | No |
| `kserve_v1_predict` | `instances` / `predictions` | Yes | No | No |
| `kserve_v2_infer` | OIP typed tensors | Yes | Yes | gRPC server stream |
| `kserve_v2_embeddings` | OIP typed tensors | Yes | Yes | No |
| `kserve_v2_rankings` | OIP typed tensors | Yes | Yes | No |
| `kserve_v2_vlm` | OIP text and image tensors | Yes | Yes | gRPC server stream |
| `kserve_v2_images` | OIP image-generation tensors | Yes | Yes | No |

The three OpenAI-compatible endpoints use `/openai/v1/...`. KServe V1 uses
`/v1/models/{model_name}:predict`, and KServe V2 HTTP uses
`/v2/models/{model_name}/infer`. AIPerf expands `{model_name}` from the
effective model selected for each request.

## Native gRPC example

Native gRPC is available only through a Config v2 file. Select the
`online_grpc` backend and use an explicit `grpc://` or `grpcs://` URL:

```yaml
# kserve-grpc.yaml
schemaVersion: "2.0"

benchmark:
  models: [my-triton-model]
  backend:
    type: online_grpc
    config: {}
  endpoint:
    urls: ["grpc://triton.default.svc.cluster.local:8001"]
    type: kserve_v2_infer
    streaming: true
    waitForModelTimeout: 0
    headers:
      authorization: ${INFERENCE_AUTHORIZATION:Bearer local-token}
  dataset:
    type: synthetic
    entries: 100
    prompts: {isl: 512, osl: 128}
  phases:
    - name: profiling
      type: concurrency
      requests: 100
      concurrency: 8
  tokenizer:
    name: cl100k_base
    trustRemoteCode: false
    applyChatTemplate: false
  gpuTelemetry: {enabled: false}
  serverMetrics: {enabled: false}
  artifacts:
    dir: ./artifacts/kserve-grpc
```

Run the benchmark through the only human-facing CLI:

```bash
aiperf profile --config kserve-grpc.yaml
```

With `streaming: false`, the runner calls `ModelInfer`. With `streaming: true`,
`kserve_v2_infer` and `kserve_v2_vlm` call `ModelStreamInfer`, enabling TTFT
and inter-token measurements.

## KServe V1 Predict over HTTP

KServe V1 Predict is an HTTP dialect, still carried by runner protocol v2:

```yaml
schemaVersion: "2.0"

benchmark:
  models: [tensorflow-model]
  backend:
    type: online_http
    config: {}
  endpoint:
    urls: ["http://tensorflow-serving.default.svc.cluster.local:8501"]
    type: kserve_v1_predict
    streaming: false
    extra:
      v1_input_field: sentence
      v1_output_field: answer
  dataset:
    type: synthetic
    entries: 50
    prompts: {isl: 32, osl: 1}
  phases:
    - {name: profiling, type: concurrency, requests: 50, concurrency: 4}
  gpuTelemetry: {enabled: false}
  serverMetrics: {enabled: false}
```

The default input and output fields are `text` and `output`. AIPerf sends
`{"instances": [{"text": "..."}]}` and accepts dictionary or scalar entries
under `predictions`, with normal endpoint auto-detection as a fallback.

## KServe V2 selectors

Use endpoint `extra` fields when a model configuration uses non-default tensor
names. Unconsumed values become request-level OIP parameters.

```yaml
endpoint:
  urls: ["grpc://triton:8001"]
  type: kserve_v2_infer
  streaming: true
  extra:
    v2_input_name: INPUT_TEXT
    v2_output_name: OUTPUT_TEXT
    temperature: 0.7
    top_k: 40
```

The other KServe V2 dialects expose corresponding selectors for embeddings,
rankings, VLM media, and image tensors. The selected endpoint validates these
values during side-effect-free runner-v2 preparation.

## Current fail-closed limits

- `online_grpc` accepts only `grpc://` or `grpcs://`; do not set the legacy
  `endpoint.transport` field.
- All URLs in one run must use the same gRPC security scheme.
- Positive `waitForModelTimeout` is not yet composed into the gRPC runner
  lifecycle, so leave it at its default `0`. The transport-level `ModelReady`
  implementation is reserved for that future composition.
- GPU/server/network/live-streaming sidecars are not advertised for the gRPC
  pair yet. AIPerf rejects them instead of silently dropping them.

See [Native gRPC Transport](./grpc-transport.md) for channel reuse, TLS,
cancellation, status mapping, and trace behavior.
