---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with VisionArena Dataset
---

# Profile with VisionArena Dataset

AIPerf supports benchmarking using the VisionArena dataset, a collection of real-world
conversations between users and vision language models gathered from Chatbot Arena. Each sample
contains a real user image and question, covering tasks like captioning, OCR, diagram
interpretation, and visual reasoning.

This guide covers profiling OpenAI-compatible vision language models using the VisionArena public
dataset.

> **Note:** VisionArena requires HuggingFace authentication. Set your `HF_TOKEN` environment
> variable before running.

---

## Start a vLLM Server

Launch a vLLM server with a vision language model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

Verify the server is ready:
```bash
curl -s 127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-VL-2B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with VisionArena Dataset

{/* aiperf-run-vllm-vision-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url 127.0.0.1:8000 \
    --public-dataset vision_arena \
    --request-count 10 \
    --concurrency 4
```
{/* /aiperf-run-vllm-vision-openai-endpoint-server */}

**Sample Output (Successful Run):**

```
                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                   Metric ┃      avg ┃      min ┃       max ┃       p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to First Token (ms) │ 1,230.82 │   480.86 │  2,375.71 │  2,375.70 │ 2,375.62 │   712.68 │   793.32 │
│     Time to Second Token │   286.66 │    83.72 │  1,894.74 │  1,735.21 │   299.41 │   108.29 │   536.11 │
│                     (ms) │          │          │           │           │          │          │          │
│     Time to First Output │ 1,230.82 │   480.86 │  2,375.71 │  2,375.70 │ 2,375.62 │   712.68 │   793.32 │
│               Token (ms) │          │          │           │           │          │          │          │
│     Request Latency (ms) │ 5,847.57 │ 2,735.34 │ 10,756.76 │ 10,502.44 │ 8,213.58 │ 5,800.62 │ 2,337.90 │
│ Inter Token Latency (ms) │   134.88 │    62.38 │    186.61 │    185.08 │   171.30 │   138.40 │    34.96 │
│  Output Token Throughput │     8.13 │     5.36 │     16.03 │     15.44 │    10.09 │     7.23 │     2.95 │
│                 Per User │          │          │           │           │          │          │          │
│        (tokens/sec/user) │          │          │           │           │          │          │          │
│   Output Sequence Length │    37.50 │     9.00 │     96.00 │     91.68 │    52.80 │    35.50 │    23.56 │
│                 (tokens) │          │          │           │           │          │          │          │
│    Input Sequence Length │    28.90 │     4.00 │    167.00 │    157.55 │    72.50 │     8.50 │    48.88 │
│                 (tokens) │          │          │           │           │          │          │          │
│  Output Token Throughput │    22.63 │      N/A │       N/A │       N/A │      N/A │      N/A │      N/A │
│             (tokens/sec) │          │          │           │           │          │          │          │
│       Request Throughput │     0.60 │      N/A │       N/A │       N/A │      N/A │      N/A │      N/A │
│           (requests/sec) │          │          │           │           │          │          │          │
│ Request Count (requests) │    10.00 │      N/A │       N/A │       N/A │      N/A │      N/A │      N/A │
└──────────────────────────┴──────────┴──────────┴───────────┴───────────┴──────────┴──────────┴──────────┘
```

> Higher input sequence length compared to text-only datasets is expected — each request includes an encoded image alongside the question text.
