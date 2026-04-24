<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile with BurstGPT Traces

AIPerf supports benchmarking using [BurstGPT](https://github.com/HPMLL/BurstGPT), a real-world LLM traffic trace dataset from Microsoft Research. The dataset captures bursty request patterns with per-request token counts.

This guide covers replaying BurstGPT traces to reproduce real-world traffic patterns against your inference server.

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

## BurstGPT Trace Format

BurstGPT traces are CSV files where each row represents a single independent request.

| Column | Description |
|---|---|
| `Timestamp` | Request arrival time in seconds (converted to milliseconds internally) |
| `Model` | Model name from the original trace |
| `Request tokens` | Input token count |
| `Response tokens` | Output token count |
| `Total tokens` | Sum of request and response tokens |
| `Log Type` | Request type (e.g. `chat`) |

Example rows:

```csv
Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
0.123,gpt-4,512,128,640,chat
0.456,gpt-4,300,80,380,chat
1.200,gpt-4,200,60,260,chat
```

Each row is treated as an independent single-turn request. AIPerf synthesizes prompts of the prescribed token lengths — no actual prompt text is stored in the trace.

---

## Download and Profile

Download a trace file from the BurstGPT repository and run a benchmark:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Download a trace file
curl -Lo burst_gpt.csv \
  https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv

# Create a small subset for a quick test
head -n 11 burst_gpt.csv > burst_gpt_short.csv  # 11 = header + 10 rows

# Run trace replay
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file burst_gpt_short.csv \
    --custom-dataset-type burst_gpt_trace \
    --fixed-schedule
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**

```
                                        NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃               Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│  Time to First Token │  3,476.33 │    277.72 │  9,816.51 │  9,680.42 │  8,455.61 │  2,323.72 │ 3,315.33 │
│                 (ms) │           │           │           │           │           │           │          │
│ Time to Second Token │     34.01 │     26.42 │     45.75 │     45.38 │     42.10 │     32.55 │     6.13 │
│                 (ms) │           │           │           │           │           │           │          │
│ Time to First Output │ 17,296.54 │ 12,208.92 │ 22,384.15 │ 22,282.40 │ 21,366.63 │ 17,296.54 │ 5,087.62 │
│           Token (ms) │           │           │           │           │           │           │          │
│ Request Latency (ms) │ 17,257.32 │  2,917.77 │ 35,580.78 │ 35,145.07 │ 31,223.72 │ 13,758.60 │ 9,919.75 │
│  Inter Token Latency │     34.86 │     28.29 │     42.35 │     42.16 │     40.39 │     35.23 │     4.74 │
│                 (ms) │           │           │           │           │           │           │          │
│         Output Token │     29.23 │     23.61 │     35.35 │     35.26 │     34.48 │     28.43 │     4.02 │
│  Throughput Per User │           │           │           │           │           │           │          │
│    (tokens/sec/user) │           │           │           │           │           │           │          │
│      Output Sequence │    385.60 │     17.00 │    900.00 │    877.05 │    670.50 │    330.50 │   236.07 │
│      Length (tokens) │           │           │           │           │           │           │          │
│       Input Sequence │    581.50 │     37.00 │  1,528.00 │  1,512.88 │  1,376.80 │    444.50 │   526.41 │
│      Length (tokens) │           │           │           │           │           │           │          │
│         Output Token │      7.13 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│           Throughput │           │           │           │           │           │           │          │
│         (tokens/sec) │           │           │           │           │           │           │          │
│   Request Throughput │      0.02 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│       (requests/sec) │           │           │           │           │           │           │          │
│        Request Count │     10.00 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│           (requests) │           │           │           │           │           │           │          │
└──────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘
```
---

## Related Tutorials

- [Bailian Traces](bailian-trace.md) - Bailian production trace replay
- [Fixed Schedule](fixed-schedule.md) - Precise timestamp-based execution for any dataset
- [Prefix Synthesis](prefix-synthesis.md) - KV cache testing with hash-based prefix data
- [Multi-Turn Conversations](multi-turn.md) - Multi-turn conversation benchmarking
