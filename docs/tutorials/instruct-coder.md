---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with InstructCoder Dataset
---

# Profile with InstructCoder Dataset

AIPerf supports benchmarking using the InstructCoder dataset (`likaixin/InstructCoder`), which
contains code editing and generation instructions. This dataset is useful for measuring model
throughput and latency under code generation workloads.

This guide covers profiling OpenAI-compatible chat completions endpoints using the InstructCoder
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

## Profile with InstructCoder Dataset

AIPerf loads the InstructCoder dataset from HuggingFace and uses each instruction as a
single-turn prompt.

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset instruct_coder \
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
│ Time to First Token │    467.66 │    276.26 │    727.41 │    727.40 │    727.36 │    320.26 │    202.34 │
│                (ms) │           │           │           │           │           │           │           │
│      Time to Second │     58.66 │     51.91 │     70.72 │     70.37 │     67.24 │     54.80 │      7.30 │
│          Token (ms) │           │           │           │           │           │           │           │
│       Time to First │ 35,962.41 │ 18,843.60 │ 82,092.30 │ 79,669.08 │ 57,860.07 │ 30,302.17 │ 18,641.22 │
│   Output Token (ms) │           │           │           │           │           │           │           │
│     Request Latency │ 49,546.08 │ 23,752.73 │ 99,184.28 │ 95,994.62 │ 67,287.65 │ 47,369.35 │ 20,292.88 │
│                (ms) │           │           │           │           │           │           │           │
│ Inter Token Latency │     60.88 │     43.09 │     67.81 │     67.80 │     67.68 │     64.09 │      7.40 │
│                (ms) │           │           │           │           │           │           │           │
│        Output Token │     16.73 │     14.75 │     23.21 │     22.85 │     19.64 │     15.60 │      2.49 │
│ Throughput Per User │           │           │           │           │           │           │           │
│   (tokens/sec/user) │           │           │           │           │           │           │           │
│     Output Sequence │    849.60 │    383.00 │  2,295.00 │  2,184.12 │  1,186.20 │    709.50 │    512.81 │
│     Length (tokens) │           │           │           │           │           │           │           │
│      Input Sequence │     15.20 │     11.00 │     21.00 │     20.82 │     19.20 │     14.50 │      3.31 │
│     Length (tokens) │           │           │           │           │           │           │           │
│        Output Token │     47.46 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│          Throughput │           │           │           │           │           │           │           │
│        (tokens/sec) │           │           │           │           │           │           │           │
│  Request Throughput │      0.06 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│      (requests/sec) │           │           │           │           │           │           │           │
│       Request Count │     10.00 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│          (requests) │           │           │           │           │           │           │           │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```