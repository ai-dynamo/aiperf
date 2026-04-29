---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with Blazedit Dataset
---

# Profile with Blazedit Dataset

AIPerf supports benchmarking using the Blazedit datasets (`vdaita/edit_5k_char` and
`vdaita/edit_10k_char`), which contain code change requests paired with code files of varying
lengths. These datasets are useful for measuring model throughput and latency under long-context
code editing workloads.

Two variants are available:

- `blazedit_5k` — ~5k character code contexts, lower token count per request
- `blazedit_10k` — ~10k character code contexts, higher memory pressure

This guide covers profiling OpenAI-compatible chat completions endpoints using the Blazedit
public datasets.

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

## Profile with Blazedit Dataset

AIPerf loads the Blazedit dataset from HuggingFace and constructs prompts that include the
full code file alongside the change request, matching the evaluation approach used by vLLM's
benchmark suite. Each prompt averages ~1,500 input tokens for the 5k variant.

Use `--prompt-output-tokens-mean` to cap output length — without it the model regenerates the
entire modified file, producing thousands of output tokens per request.

**5k character variant:**

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset blazedit_5k \
    --request-count 5 \
    --concurrency 4 \
    --prompt-output-tokens-mean 512
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**

```
                          NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃           Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│    Time to First │  2,563.25 │    344.17 │  3,144.87 │  3,144.87 │  3,144.84 │  3,144.55 │  1,110.31 │
│       Token (ms) │           │           │           │           │           │           │           │
│   Time to Second │     94.15 │     40.08 │    107.95 │    107.95 │    107.95 │    107.93 │     27.04 │
│       Token (ms) │           │           │           │           │           │           │           │
│    Time to First │ 54,707.53 │ 49,364.22 │ 60,050.84 │ 59,943.97 │ 58,982.18 │ 54,707.53 │  5,343.31 │
│     Output Token │           │           │           │           │           │           │           │
│             (ms) │           │           │           │           │           │           │           │
│  Request Latency │ 54,396.21 │ 20,093.94 │ 62,997.77 │ 62,997.77 │ 62,997.75 │ 62,997.54 │ 17,151.18 │
│             (ms) │           │           │           │           │           │           │           │
│      Inter Token │    101.77 │     38.73 │    117.60 │    117.60 │    117.59 │    117.59 │     31.52 │
│     Latency (ms) │           │           │           │           │           │           │           │
│     Output Token │     11.97 │      8.50 │     25.82 │     25.13 │     18.90 │      8.50 │      6.93 │
│   Throughput Per │           │           │           │           │           │           │           │
│             User │           │           │           │           │           │           │           │
│ (tokens/sec/use… │           │           │           │           │           │           │           │
│  Output Sequence │    510.40 │    510.00 │    511.00 │    511.00 │    511.00 │    510.00 │      0.49 │
│  Length (tokens) │           │           │           │           │           │           │           │
│   Input Sequence │  1,485.60 │  1,161.00 │  1,739.00 │  1,733.08 │  1,679.80 │  1,569.00 │    200.73 │
│  Length (tokens) │           │           │           │           │           │           │           │
│     Output Token │     30.75 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│       Throughput │           │           │           │           │           │           │           │
│     (tokens/sec) │           │           │           │           │           │           │           │
│          Request │      0.06 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│       Throughput │           │           │           │           │           │           │           │
│   (requests/sec) │           │           │           │           │           │           │           │
│    Request Count │      5.00 │       N/A │       N/A │       N/A │       N/A │       N/A │       N/A │
│       (requests) │           │           │           │           │           │           │           │
└──────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

**10k character variant:**

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset blazedit_10k \
    --request-count 5 \
    --concurrency 4 \
    --prompt-output-tokens-mean 512
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->