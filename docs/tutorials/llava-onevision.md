---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with LLaVA-OneVision Dataset
---

# Profile with LLaVA-OneVision Dataset

AIPerf supports benchmarking using the LLaVA-OneVision dataset, which contains a large
multimodal collection of instruction-tuning examples covering charts, diagrams, scientific
figures, natural photos, and more.

This guide uses the `sharegpt4o` subset — GPT-4o annotated ShareGPT conversations with natural
scene images.

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

## Profile with LLaVA-OneVision Dataset

AIPerf loads the `sharegpt4o` subset from HuggingFace, extracts the first user message and image
from each row, and sends each as a single-turn vision request.

<!-- aiperf-run-vllm-vision-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url 127.0.0.1:8000 \
    --public-dataset llava_onevision \
    --request-count 10 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-vision-openai-endpoint-server -->

**Sample Output (Successful Run):**

```
                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃         Metric ┃        avg ┃       min ┃        max ┃        p99 ┃        p90 ┃        p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│  Time to First │  42,612.16 │  2,865.14 │ 124,091.22 │ 118,453.32 │  67,712.19 │  39,609.67 │ 36,226.35 │
│     Token (ms) │            │           │            │            │            │            │           │
│ Time to Second │     117.54 │     92.84 │     149.27 │     147.27 │     129.28 │     115.31 │     13.31 │
│     Token (ms) │            │           │            │            │            │            │           │
│  Time to First │  42,612.16 │  2,865.14 │ 124,091.22 │ 118,453.32 │  67,712.19 │  39,609.67 │ 36,226.35 │
│   Output Token │            │           │            │            │            │            │           │
│           (ms) │            │           │            │            │            │            │           │
│        Request │ 121,911.45 │ 16,030.77 │ 225,154.30 │ 220,635.78 │ 179,969.13 │ 123,267.35 │ 61,477.79 │
│   Latency (ms) │            │           │            │            │            │            │           │
│    Inter Token │     462.52 │     95.96 │   1,865.22 │   1,803.95 │   1,252.60 │     168.13 │    564.71 │
│   Latency (ms) │            │           │            │            │            │            │           │
│   Output Token │       5.31 │      0.54 │      10.42 │      10.25 │       8.69 │       5.95 │      3.16 │
│ Throughput Per │            │           │            │            │            │            │           │
│           User │            │           │            │            │            │            │           │
│ (tokens/sec/u… │            │           │            │            │            │            │           │
│         Output │     228.60 │     82.00 │     421.00 │     412.99 │     340.90 │     228.00 │    109.17 │
│       Sequence │            │           │            │            │            │            │           │
│         Length │            │           │            │            │            │            │           │
│       (tokens) │            │           │            │            │            │            │           │
│ Input Sequence │       9.70 │      4.00 │      13.00 │      13.00 │      13.00 │      10.50 │      3.03 │
│         Length │            │           │            │            │            │            │           │
│       (tokens) │            │           │            │            │            │            │           │
│   Output Token │       7.41 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     Throughput │            │           │            │            │            │            │           │
│   (tokens/sec) │            │           │            │            │            │            │           │
│          Image │       0.01 │      0.00 │       0.06 │       0.06 │       0.03 │       0.01 │      0.02 │
│     Throughput │            │           │            │            │            │            │           │
│   (images/sec) │            │           │            │            │            │            │           │
│  Image Latency │ 121,911.45 │ 16,030.77 │ 225,154.30 │ 220,635.78 │ 179,969.13 │ 123,267.35 │ 61,477.79 │
│     (ms/image) │            │           │            │            │            │            │           │
│        Request │       0.03 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     Throughput │            │           │            │            │            │            │           │
│ (requests/sec) │            │           │            │            │            │            │           │
│  Request Count │      10.00 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     (requests) │            │           │            │            │            │            │           │
└────────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴───────────┘
```

> LLaVA-OneVision's GPT-4o annotated responses are detailed and verbose, producing longer output
> sequences than typical VQA datasets. Use `--prompt-output-tokens-mean` to cap output length if
> needed.
