<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile OpenAI-Compatible Text APIs Using AIPerf

This guide demonstrates how to profile OpenAI-compatible text generation endpoints (Chat Completions, Completions, and the Responses API) using vLLM and AIPerf.

## Start a vLLM server

```bash
# Pull and run vLLM Docker container:
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
```

```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

## Profile Chat Completions API
The Chat Completions API uses the `/v1/chat/completions` endpoint.

### Profile with synthetic inputs

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --url localhost:8000 \
    --request-count 20
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

### Profile with custom input file
Create a JSONL input file:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->

```bash
cat <<EOF > inputs.jsonl
{"texts": ["Hello!", "How are you?"]}
{"texts": ["Tell me a joke.", "Tell me a story."]}
EOF
```

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 10
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

## Profile Completions API
The Completions API uses the `/v1/completions` endpoint.

### Profile with synthetic inputs

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type completions \
    --endpoint /v1/completions \
    --synthetic-input-tokens-mean 64 \
    --synthetic-input-tokens-stddev 4 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 4 \
    --url localhost:8000 \
    --request-count 32
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

### Profile with custom input file
Create a JSONL input file:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat <<EOF > inputs.jsonl
{"texts": ["Hello!", "What’s up?"]}
{"texts": ["Tell me a joke.", "Give me a poem."]}
EOF
```

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type completions \
    --endpoint /v1/completions \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 10

```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->