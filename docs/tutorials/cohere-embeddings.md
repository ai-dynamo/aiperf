---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Cohere Embed Endpoints with AIPerf
---

# Profile Cohere Embed Endpoints with AIPerf

AIPerf supports benchmarking Cohere Embed v2-compatible endpoints using the
`cohere_embeddings` endpoint type. Use this endpoint type for servers that expose
`POST /v2/embed`, including
[vLLM pooling servers](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#cohere-embed-api).

This guide uses
[`nvidia/llama-nemotron-embed-vl-1b-v2`](https://docs.vllm.ai/en/stable/models/pooling_models/specific_models/#embedding-model),
a multimodal vLLM embedding model that accepts text and image inputs.

The Cohere Embed API accepts text inputs through `texts` and multimodal inputs
through `inputs`. AIPerf chooses the correct request shape from the dataset: text-only
requests use `texts`, while image or mixed text-image requests use `inputs`.

---

## Start a vLLM Embedding Server

Launch a vLLM server with Llama Nemotron VL Embed:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model nvidia/llama-nemotron-embed-vl-1b-v2 \
  --trust-remote-code \
  --chat-template examples/pooling/embed/template/nemotron_embed_vl.jinja
```

Verify the Cohere-compatible endpoint is ready:

```bash
IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
BASE64_IMAGE=$(curl -sL "$IMAGE_URL" | base64 -w 0)

curl -s http://localhost:8000/v2/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "inputs": [
      {
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,'"$BASE64_IMAGE"'"
            }
          },
          {
            "type": "text",
            "text": "A wooden boardwalk through green wetlands."
          }
        ]
      }
    ],
    "embedding_types": ["float"]
  }' | jq
```

---

## Profile with Synthetic Inputs

Run AIPerf against `/v2/embed` with generated text inputs:

```bash
aiperf profile \
    --model nvidia/llama-nemotron-embed-vl-1b-v2 \
    --endpoint-type cohere_embeddings \
    --endpoint /v2/embed \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

Embeddings endpoints report request latency, input sequence length, request throughput,
and embedding throughput. Token-generation metrics such as TTFT and ITL do not apply
because embedding endpoints return vectors instead of streamed output tokens.

---

## Profile with Custom Text Inputs

Create a JSONL input file:

```bash
cat <<EOF > cohere-embedding-inputs.jsonl
{"texts": ["What is artificial intelligence?"]}
{"texts": ["How do embedding models support semantic search?"]}
{"texts": ["What is retrieval augmented generation?"]}
{"texts": ["How do vector databases index embeddings?"]}
EOF
```

Run AIPerf with the custom dataset:

```bash
aiperf profile \
    --model nvidia/llama-nemotron-embed-vl-1b-v2 \
    --endpoint-type cohere_embeddings \
    --endpoint /v2/embed \
    --input-file cohere-embedding-inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 4 \
    --concurrency 2
```

When using custom inputs, AIPerf sends the text values from each JSONL row in the
Cohere `texts` field.

---

## Profile Multimodal Embeddings

For multimodal embeddings, include both `texts` and `images` in the input file. Local
image paths are encoded to base64 data URLs before sending, while remote URLs are
passed through.

Download a sample image:

```bash
curl -L \
  -o nemotron-vl-boardwalk.jpg \
  https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
```

Create a multimodal JSONL input file:

```bash
cat <<EOF > cohere-multimodal-inputs.jsonl
{"texts": ["A wooden boardwalk through green wetlands."], "images": ["nemotron-vl-boardwalk.jpg"]}
{"texts": ["A nature trail image with a raised boardwalk."], "images": ["nemotron-vl-boardwalk.jpg"]}
EOF
```

Run AIPerf with the same Cohere-compatible endpoint:

```bash
aiperf profile \
    --model nvidia/llama-nemotron-embed-vl-1b-v2 \
    --endpoint-type cohere_embeddings \
    --endpoint /v2/embed \
    --input-file cohere-multimodal-inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 2
```

Mixed text-image requests are paired by position. Each row must contain the same number
of text items and image items when both fields are present. AIPerf sends these rows
using the Cohere `inputs` field with `text` and `image_url` content parts.
