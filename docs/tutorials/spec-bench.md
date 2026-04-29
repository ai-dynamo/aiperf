---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with SpecBench Dataset
---

# Profile with SpecBench Dataset

AIPerf supports benchmarking using the SpecBench dataset, which contains diverse questions
across writing, reasoning, math, and coding categories. This dataset is commonly used for
evaluating speculative decoding methods.

This guide covers profiling OpenAI-compatible chat completions endpoints using the SpecBench
public dataset.

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

## Profile with SpecBench Dataset

AIPerf downloads the SpecBench JSONL file from GitHub and uses the first turn of each
question as a single-turn prompt.

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset spec_bench \
    --request-count 10 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```

                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃              Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Time to First Token │  1,184.13 │    385.39 │  2,252.54 │  2,252.54 │  2,252.52 │    570.29 │    862.86 │
│                (ms) │           │           │           │           │           │           │           │
│      Time to Second │     60.70 │     50.20 │     73.47 │     73.21 │     70.93 │     59.97 │      7.68 │
│          Token (ms) │           │           │           │           │           │           │           │
│       Time to First │ 21,668.12 │ 12,749.53 │ 36,041.69 │ 35,432.43 │ 29,949.09 │ 19,877.32 │  6,284.90 │
│   Output Token (ms) │           │           │           │           │           │           │           │
│     Request Latency │ 36,715.17 │ 22,633.27 │ 69,707.26 │ 68,331.92 │ 55,953.83 │ 30,128.68 │ 14,438.18 │
│                (ms) │           │           │           │           │           │           │           │
│ Inter Token Latency │     62.47 │     51.53 │     69.41 │     69.26 │     67.89 │     64.96 │      5.98 │
│                (ms) │           │           │           │           │           │           │           │
│        Output Token │     16.17 │     14.41 │     19.40 │     19.32 │     18.58 │     15.40 │      1.67 │
│ Throughput Per User │           │           │           │           │           │           │           │
│   (tokens/sec/user) │           │           │           │           │           │           │           │
│     Output Sequence │    572.20 │    326.00 │  1,004.00 │  1,000.31 │    967.10 │    501.50 │    221.81 │
│     Length (tokens) │           │           │           │           │           │           │           │
│      Input Sequence │     41.50 │     22.00 │     96.00 │     92.49 │     60.90 │     35.50 │     20.86 │
│     Length (tokens) │           │           │           │           │           │           │           │
│        Output Token │     58.14 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│          Throughput │           │           │           │           │           │           │           │
│        (tokens/sec) │           │           │           │           │           │           │           │
│  Request Throughput │      0.10 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│      (requests/sec) │           │           │           │           │           │           │           │
│       Request Count │     10.00 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│          (requests) │           │           │           │           │           │           │           │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```