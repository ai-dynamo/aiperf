<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Request Cancellation Testing

AIPerf supports request timeout and cancellation scenarios, which are important for calculating the impact of user cancellation on performance.

## Setting Up the Server

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

## Basic Request Cancellation

Test with a small percentage of cancelled requests:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with 10% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 10 \
    --request-cancellation-delay 0.5 \
    --synthetic-input-tokens-mean 800 \
    --synthetic-input-tokens-stddev 80 \
    --output-tokens-mean 400 \
    --output-tokens-stddev 40 \
    --concurrency 8 \
    --request-count 50 \
    --warmup-request-count 5
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Parameters Explained:**
- `--request-cancellation-rate 10`: Cancel 10% of requests (value between 0.0 and 100.0)
- `--request-cancellation-delay 0.5`: Wait .5 seconds before cancelling selected requests

### High Cancellation Rate Testing

Test service resilience under frequent cancellations:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with 50% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 50 \
    --request-cancellation-delay 1.0 \
    --synthetic-input-tokens-mean 1200 \
    --output-tokens-mean 600 \
    --concurrency 10 \
    --request-count 40
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

### Immediate Cancellation Testing

Test rapid cancellation scenarios:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with immediate cancellation (0 delay)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 30 \
    --request-cancellation-delay 0.0 \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 100 \
    --concurrency 15 \
    --request-count 60
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Expected Results:**
- Tests how quickly the server can handle connection terminations
- Useful for testing resource cleanup and connection pooling
