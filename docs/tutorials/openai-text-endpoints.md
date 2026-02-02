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

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:35<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1678.90 │ 1456.34 │ 1923.45 │ 1923.45 │ 1667.23 │
│    Time to First Token (ms) │  234.56 │  198.34 │  289.12 │  289.12 │  231.45 │
│    Inter Token Latency (ms) │   13.89 │   11.23 │   17.45 │   17.45 │   13.67 │
│ Output Token Count (tokens) │  200.00 │  200.00 │  200.00 │  200.00 │  200.00 │
│  Request Throughput (req/s) │    5.67 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency1/profile_export_aiperf.json
```

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

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using /v1/completions endpoint
INFO     AIPerf System is PROFILING

Profiling: 32/32 |████████████████████████| 100% [00:28<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-completions-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┃                      Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│        Request Latency (ms) │ 876.45 │ 789.34 │ 987.12 │ 978.90 │ 871.23 │
│    Time to First Token (ms) │ 156.78 │ 134.56 │ 189.23 │ 185.67 │ 155.12 │
│    Inter Token Latency (ms) │  12.34 │  10.23 │  15.67 │  15.34 │  12.12 │
│ Output Token Count (tokens) │ 128.00 │ 120.00 │ 136.00 │ 135.00 │ 128.00 │
│  Request Throughput (req/s) │  10.89 │      - │      - │      - │      - │
└─────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-completions-concurrency1/profile_export_aiperf.json
```

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