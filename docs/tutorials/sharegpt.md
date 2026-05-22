---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with ShareGPT Dataset
---

# Profile with ShareGPT Dataset

AIPerf supports benchmarking using the ShareGPT dataset, which contains real conversational data from user interactions.

This guide covers profiling OpenAI-compatible chat completions endpoints using the ShareGPT public dataset.

---

## Start a vLLM Server

Launch a vLLM server with a chat model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

Verify the server is ready:
```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with ShareGPT Dataset

AIPerf automatically downloads and caches the ShareGPT dataset from HuggingFace.

<!-- aiperf-run-vllm-default-openai-endpoint-server weight=200 -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset sharegpt \
    --request-count 20 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Downloading ShareGPT dataset from HuggingFace
INFO     Cached ShareGPT dataset loaded
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:45<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency4/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1456.78 │ 1089.34 │ 1978.90 │ 1898.45 │ 1423.67 │
│    Time to First Token (ms) │  267.89 │  198.34 │  389.12 │  367.45 │  262.12 │
│    Inter Token Latency (ms) │   13.45 │   10.67 │   18.90 │   17.89 │   13.12 │
│ Output Token Count (tokens) │  187.00 │  142.00 │  245.00 │  239.00 │  184.00 │
│  Request Throughput (req/s) │    8.45 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency4/profile_export_aiperf.json
```
