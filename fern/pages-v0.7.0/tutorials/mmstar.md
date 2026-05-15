---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with MMStar Dataset
---

# Profile with MMStar Dataset

AIPerf supports benchmarking using the MMStar dataset, a multimodal visual question answering
benchmark that tests fine-grained visual perception and reasoning. Each sample contains an image
and a question that requires understanding the image to answer.

This guide covers profiling OpenAI-compatible vision language models using the MMStar public
dataset.

---

## Start a vLLM Server

Launch a vLLM server with a vision language model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct
```

Verify the server is ready:
```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-VL-2B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with MMStar Dataset

AIPerf loads the MMStar dataset from HuggingFace, attaches the image from each row to the
question, and sends each pair as a single-turn vision request.

{/* aiperf-run-vllm-vision-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset mmstar \
    --request-count 10 \
    --concurrency 4
```
{/* /aiperf-run-vllm-vision-openai-endpoint-server */}

**Sample Output (Successful Run):**

```
                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃           Metric ┃       avg ┃       min ┃        max ┃        p99 ┃        p90 ┃       p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│    Time to First │ 36,158.25 │ 10,145.84 │  68,830.92 │  68,830.92 │  68,830.88 │ 28,015.68 │ 22,647.12 │
│       Token (ms) │           │           │            │            │            │           │           │
│   Time to Second │  8,736.44 │     89.13 │  53,118.68 │  49,851.97 │  20,451.58 │    113.97 │ 16,181.79 │
│       Token (ms) │           │           │            │            │            │           │           │
│    Time to First │ 36,158.25 │ 10,145.84 │  68,830.92 │  68,830.92 │  68,830.88 │ 28,015.68 │ 22,647.12 │
│     Output Token │           │           │            │            │            │           │           │
│             (ms) │           │           │            │            │            │           │           │
│  Request Latency │ 70,665.03 │ 10,972.01 │ 164,879.38 │ 158,399.81 │ 100,083.68 │ 65,599.64 │ 40,227.69 │
│             (ms) │           │           │            │            │            │           │           │
│      Inter Token │  1,445.22 │     82.62 │   4,143.54 │   3,985.99 │   2,568.02 │  1,356.06 │  1,277.29 │
│     Latency (ms) │           │           │            │            │            │           │           │
│     Output Token │      3.50 │      0.24 │      12.10 │      11.81 │       9.18 │      0.98 │      4.30 │
│   Throughput Per │           │           │            │            │            │           │           │
│             User │           │           │            │            │            │           │           │
│ (tokens/sec/use… │           │           │            │            │            │           │           │
│  Output Sequence │     26.40 │      7.00 │     118.00 │     110.71 │      45.10 │     15.00 │     31.52 │
│  Length (tokens) │           │           │            │            │            │           │           │
│   Input Sequence │     41.60 │     25.00 │      59.00 │      58.46 │      53.60 │     41.50 │     11.32 │
│  Length (tokens) │           │           │            │            │            │           │           │
│     Output Token │      1.47 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       Throughput │           │           │            │            │            │           │           │
│     (tokens/sec) │           │           │            │            │            │           │           │
│ Image Throughput │      0.02 │      0.01 │       0.09 │       0.09 │       0.03 │      0.02 │      0.02 │
│     (images/sec) │           │           │            │            │            │           │           │
│    Image Latency │ 70,665.03 │ 10,972.01 │ 164,879.38 │ 158,399.81 │ 100,083.68 │ 65,599.64 │ 40,227.69 │
│       (ms/image) │           │           │            │            │            │           │           │
│          Request │      0.06 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       Throughput │           │           │            │            │            │           │           │
│   (requests/sec) │           │           │            │            │            │           │           │
│    Request Count │     10.00 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       (requests) │           │           │            │            │            │           │           │
└──────────────────┴───────────┴───────────┴────────────┴────────────┴────────────┴───────────┴───────────┘
```
