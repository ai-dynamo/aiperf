---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with MMVU Dataset
---

# Profile with MMVU Dataset

AIPerf supports benchmarking using the MMVU dataset, an expert-level video understanding
benchmark that tests multi-discipline reasoning over video content. Each sample contains a
video URL and a question (multiple-choice or open-ended) that requires watching the video
to answer.

This guide covers profiling OpenAI-compatible video language models using the MMVU public
dataset.

---

## Start a vLLM Server

Launch a vLLM server with a video-capable vision language model:

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

## Profile with MMVU Dataset

AIPerf loads the MMVU dataset from HuggingFace, combines each question with its
multiple-choice options, attaches the video URL, and sends each pair as a single-turn
video request. The prompt format matches vLLM's own MMVU benchmark format.

{/* aiperf-run-vllm-video-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset mmvu \
    --request-count 5 \
    --concurrency 2 \
    --output-tokens-mean 128
```
{/* /aiperf-run-vllm-video-openai-endpoint-server */}

**Sample Output (Successful Run):**

```
                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃           Metric ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p50 ┃        std ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│    Time to First │ 236,267.00 │   2,967.98 │ 535,809.00 │ 528,246.99 │ 460,180.00 │ 292,846.13 │ 206,874.00 │
│       Token (ms) │            │            │            │            │            │            │            │
│   Time to Second │ 157,173.00 │     113.08 │ 473,750.00 │ 467,270.27 │ 408,951.00 │     127.74 │ 199,053.00 │
│       Token (ms) │            │            │            │            │            │            │            │
│  Request Latency │ 476,346.00 │ 297,204.97 │ 841,020.00 │ 829,081.39 │ 721,631.00 │ 350,652.38 │ 200,572.00 │
│             (ms) │            │            │            │            │            │            │            │
│      Inter Token │   3,631.07 │     106.31 │  11,204.46 │  11,020.23 │   9,362.19 │     127.14 │   4,543.17 │
│     Latency (ms) │            │            │            │            │            │            │            │
│     Output Token │       5.19 │       0.09 │       9.41 │       9.37 │       9.01 │       7.87 │       4.17 │
│   Throughput Per │            │            │            │            │            │            │            │
│             User │            │            │            │            │            │            │            │
│     (tokens/sec) │            │            │            │            │            │            │            │
│  Output Sequence │      58.00 │      32.00 │     128.00 │     125.04 │      98.40 │      42.00 │      35.84 │
│  Length (tokens) │            │            │            │            │            │            │            │
│   Input Sequence │      26.00 │       9.00 │      67.00 │      65.72 │      54.20 │      10.00 │      22.79 │
│  Length (tokens) │            │            │            │            │            │            │            │
│     Output Token │       0.24 │        N/A │        N/A │        N/A │        N/A │        N/A │        N/A │
│       Throughput │            │            │            │            │            │            │            │
│     (tokens/sec) │            │            │            │            │            │            │            │
│    Request Count │       5.00 │        N/A │        N/A │        N/A │        N/A │        N/A │        N/A │
│       (requests) │            │            │            │            │            │            │            │
└──────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

> **Note:** High TTFT variance (3s min, 536s max) is expected — the model server fetches
> each video URL from HuggingFace during inference, and fetch time varies with video size
> and network conditions.

---

## Notes

- The `video` column in MMVU contains HTTPS URLs pointing to `.mp4` files hosted on
  HuggingFace. AIPerf passes these URLs directly to the model server, which fetches
  the video during inference.
- For multiple-choice questions, choices are appended to the question in the format
  `A.option B.option ...`. Open-ended questions use the question text only.
- The dataset has a `validation` split with samples spanning multiple academic disciplines
  (Art, Science, Engineering, etc.).
