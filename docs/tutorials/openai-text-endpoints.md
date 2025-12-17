<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile OpenAI-Compatible Text APIs Using AIPerf

This guide covers profiling OpenAI-compatible Chat Completions and Completions endpoints with vLLM and AIPerf.

## Start a vLLM server

Pull and start a vLLM server using Docker:
```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3
```

Verify the server is ready:
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

## Profile Chat Completions API
The Chat Completions API uses the `/v1/chat/completions` endpoint.

### Profile with synthetic inputs

Run AIPerf against the Chat Completions endpoint using synthetic inputs:
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
{"texts": ["Hello!"]}
{"texts": ["Tell me a joke."]}
EOF
```

Run AIPerf against the Chat Completions endpoint using the custom input file:
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

Run AIPerf against the Completions endpoint using synthetic inputs:
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
{"texts": ["How are you?"]}
{"texts": ["Give me a poem."]}
EOF

```
Run AIPerf against the Completions endpoint using the custom input file:
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