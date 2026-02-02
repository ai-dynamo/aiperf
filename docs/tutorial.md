<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling with AIPerf

This tutorial will demonstrate how you can use AIPerf to measure the performance of
models using various inference solutions.

## Profile Qwen3-0.6B using vllm <a id="vllm-qwen3-0.6B">
<!-- setup-vllm-default-openai-endpoint-server -->
```bash
# Pull and run vLLM Docker container:
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
```
<!-- /setup-vllm-default-openai-endpoint-server -->

<!-- health-check-vllm-default-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-default-openai-endpoint-server -->


<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile the model
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 32 \
    --request-count 64 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 64/64 |████████████████████████| 100% [00:42<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-request_rate32/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1234.56 │ 987.34 │ 1567.89 │ 1534.23 │ 1223.45 │
│    Time to First Token (ms) │  234.56 │ 189.23 │  298.45 │  289.34 │  231.12 │
│    Inter Token Latency (ms) │   15.67 │  12.34 │   19.45 │   19.01 │   15.45 │
│ Output Token Count (tokens) │  150.00 │ 120.00 │  180.00 │  178.90 │  149.00 │
│  Request Throughput (req/s) │   31.45 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-request_rate32/profile_export_aiperf.json
```