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

AIPerf loads the Blazedit dataset from HuggingFace and uses each change request as a
single-turn prompt.

**5k character variant:**

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset blazedit_5k \
    --request-count 10 \
    --concurrency 4
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

**Sample Output (Successful Run):**

```

                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃           Metric ┃       avg ┃       min ┃        max ┃        p99 ┃        p90 ┃       p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│    Time to First │    732.60 │    396.55 │   1,141.02 │   1,141.01 │   1,140.93 │    555.79 │    328.18 │
│       Token (ms) │           │           │            │            │            │           │           │
│   Time to Second │     65.57 │     54.31 │      83.25 │      82.63 │      77.06 │     65.06 │     10.31 │
│       Token (ms) │           │           │            │            │            │           │           │
│    Time to First │ 58,962.62 │ 22,013.22 │ 125,567.46 │ 122,218.68 │  92,079.71 │ 48,100.15 │ 30,413.87 │
│     Output Token │           │           │            │            │            │           │           │
│             (ms) │           │           │            │            │            │           │           │
│  Request Latency │ 86,958.47 │ 49,516.99 │ 155,572.57 │ 151,516.69 │ 115,013.81 │ 83,652.11 │ 30,423.22 │
│             (ms) │           │           │            │            │            │           │           │
│      Inter Token │     71.55 │     45.91 │      77.97 │      77.96 │      77.93 │     74.36 │      9.19 │
│     Latency (ms) │           │           │            │            │            │           │           │
│     Output Token │     14.30 │     12.83 │      21.78 │      21.17 │      15.65 │     13.45 │      2.57 │
│   Throughput Per │           │           │            │            │            │           │           │
│             User │           │           │            │            │            │           │           │
│ (tokens/sec/use… │           │           │            │            │            │           │           │
│  Output Sequence │  1,250.10 │    703.00 │   2,399.00 │   2,367.32 │   2,082.20 │  1,146.00 │    543.01 │
│  Length (tokens) │           │           │            │            │            │           │           │
│   Input Sequence │     35.50 │     21.00 │      57.00 │      55.83 │      45.30 │     33.00 │     10.04 │
│  Length (tokens) │           │           │            │            │            │           │           │
│     Output Token │     44.98 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       Throughput │           │           │            │            │            │           │           │
│     (tokens/sec) │           │           │            │            │            │           │           │
│          Request │      0.04 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       Throughput │           │           │            │            │            │           │           │
│   (requests/sec) │           │           │            │            │            │           │           │
│    Request Count │     10.00 │       N/A │        N/A │        N/A │        N/A │       N/A │       N/A │
│       (requests) │           │           │            │            │            │           │           │
└──────────────────┴───────────┴───────────┴────────────┴────────────┴────────────┴───────────┴───────────┘
```

**10k character variant:**

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset blazedit_10k \
    --request-count 10 \
    --concurrency 4
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

**Sample Output (Successful Run):**

```

                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃         Metric ┃        avg ┃       min ┃        max ┃        p99 ┃        p90 ┃        p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│  Time to First │     950.96 │    350.77 │   1,598.25 │   1,598.22 │   1,597.92 │     645.16 │    525.26 │
│     Token (ms) │            │           │            │            │            │            │           │
│ Time to Second │      65.78 │     49.93 │      92.18 │      90.85 │      78.82 │      63.63 │     13.91 │
│     Token (ms) │            │           │            │            │            │            │           │
│  Time to First │  69,605.83 │ 29,908.82 │ 112,177.39 │ 111,778.85 │ 108,192.05 │  62,257.42 │ 27,027.18 │
│   Output Token │            │           │            │            │            │            │           │
│           (ms) │            │           │            │            │            │            │           │
│        Request │ 107,469.40 │ 54,653.07 │ 154,730.83 │ 154,133.85 │ 148,761.09 │ 104,468.66 │ 30,179.95 │
│   Latency (ms) │            │           │            │            │            │            │           │
│    Inter Token │      75.85 │     65.36 │      87.14 │      87.08 │      86.52 │      74.72 │      7.05 │
│   Latency (ms) │            │           │            │            │            │            │           │
│   Output Token │      13.30 │     11.48 │      15.30 │      15.23 │      14.63 │      13.38 │      1.22 │
│ Throughput Per │            │           │            │            │            │            │           │
│           User │            │           │            │            │            │            │           │
│ (tokens/sec/u… │            │           │            │            │            │            │           │
│         Output │   1,414.00 │    773.00 │   2,106.00 │   2,088.54 │   1,931.40 │   1,428.00 │    417.27 │
│       Sequence │            │           │            │            │            │            │           │
│         Length │            │           │            │            │            │            │           │
│       (tokens) │            │           │            │            │            │            │           │
│ Input Sequence │      46.20 │     27.00 │      69.00 │      67.56 │      54.60 │      44.50 │     10.13 │
│         Length │            │           │            │            │            │            │           │
│       (tokens) │            │           │            │            │            │            │           │
│   Output Token │      48.24 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     Throughput │            │           │            │            │            │            │           │
│   (tokens/sec) │            │           │            │            │            │            │           │
│        Request │       0.03 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     Throughput │            │           │            │            │            │            │           │
│ (requests/sec) │            │           │            │            │            │            │           │
│  Request Count │      10.00 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│     (requests) │            │           │            │            │            │            │           │
└────────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴───────────┘
```