---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with AIMO Dataset
---

# Profile with AIMO Dataset

AIPerf supports benchmarking using the AIMO NuminaMath-TIR dataset, which contains competition
mathematics problems requiring chain-of-thought reasoning. This dataset is useful for measuring
model throughput and latency under long-context, reasoning-heavy workloads.

This guide covers profiling OpenAI-compatible chat completions endpoints using the AIMO public
dataset.

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

## Profile with AIMO Dataset

AIPerf loads the AIMO dataset from HuggingFace and uses each problem as a single-turn prompt.

AIMO problems elicit long chain-of-thought responses. Use `--prompt-output-tokens-mean` to cap
output length and reduce benchmark duration:

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset aimo \
    --request-count 10 \
    --concurrency 4 \
    --prompt-output-tokens-mean 512
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

**Sample Output (Successful Run):**

```

                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃               Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│  Time to First Token │  1,509.04 │    687.93 │  1,879.53 │  1,879.53 │  1,879.51 │  1,563.66 │   430.29 │
│                 (ms) │           │           │           │           │           │           │          │
│ Time to Second Token │     56.44 │     32.27 │     86.48 │     84.59 │     67.58 │     53.16 │    14.62 │
│                 (ms) │           │           │           │           │           │           │          │
│ Request Latency (ms) │ 32,823.02 │ 18,943.44 │ 38,431.19 │ 38,431.18 │ 38,431.07 │ 34,163.15 │ 7,192.79 │
│  Inter Token Latency │     61.40 │     35.80 │     71.72 │     71.72 │     71.67 │     63.92 │    13.26 │
│                 (ms) │           │           │           │           │           │           │          │
│         Output Token │     17.42 │     13.94 │     27.94 │     27.94 │     27.93 │     15.64 │     5.31 │
│  Throughput Per User │           │           │           │           │           │           │          │
│    (tokens/sec/user) │           │           │           │           │           │           │          │
│      Output Sequence │    511.00 │    511.00 │    511.00 │    511.00 │    511.00 │    511.00 │     0.00 │
│      Length (tokens) │           │           │           │           │           │           │          │
│       Input Sequence │     56.90 │     31.00 │     84.00 │     83.46 │     78.60 │     53.00 │    17.59 │
│      Length (tokens) │           │           │           │           │           │           │          │
│         Output Token │     55.82 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│           Throughput │           │           │           │           │           │           │          │
│         (tokens/sec) │           │           │           │           │           │           │          │
│   Request Throughput │      0.11 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│       (requests/sec) │           │           │           │           │           │           │          │
│        Request Count │     10.00 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│           (requests) │           │           │           │           │           │           │          │
└──────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘
```


> Higher request latency compared to conversational datasets is expected — AIMO problems require
> extended chain-of-thought reasoning and produce significantly longer outputs.
