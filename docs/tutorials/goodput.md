<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Goodput with AIPerf

## Context

Goodput is defined as the number of completed requests per second
that meet specified metric constraints, also called service level
objectives.

For example, perhaps you want to measure the user experience of your service
by considering throughput only including requests where the time to first token
is under 50ms and inter-token latency is under 10ms.

AIPerf provides this value as goodput.

## Tutorial

Below you can find a tutorial on how to benchmark a model
using goodput.

### Setting Up the Server

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

#### Run AIPerf with Goodput Constraints

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    -m Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --goodput "time_to_first_token:100 inter_token_latency:3.40"
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

Example output:

```
                                             NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                               Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│             Time to First Token (ms) │    50.04 │    30.65 │    60.12 │    59.53 │    55.68 │    51.80 │     6.52 │
│            Time to Second Token (ms) │     3.19 │     2.89 │     3.80 │     3.80 │     3.44 │     3.13 │     0.24 │
│                 Request Latency (ms) │ 3,651.23 │ 2,273.54 │ 5,807.03 │ 5,707.95 │ 5,027.05 │ 3,261.42 │ 1,033.79 │
│             Inter Token Latency (ms) │     3.40 │     3.36 │     3.45 │     3.45 │     3.44 │     3.40 │     0.03 │
│     Output Token Throughput Per User │   294.11 │   289.70 │   297.60 │   297.57 │   297.17 │   293.85 │     2.31 │
│                    (tokens/sec/user) │          │          │          │          │          │          │          │
│      Output Sequence Length (tokens) │ 1,058.45 │   660.00 │ 1,676.00 │ 1,649.78 │ 1,457.90 │   949.00 │   297.94 │
│       Input Sequence Length (tokens) │   550.00 │   550.00 │   550.00 │   550.00 │   550.00 │   550.00 │     0.00 │
│ Output Token Throughput (tokens/sec) │   287.44 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│    Request Throughput (requests/sec) │     0.27 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│             Request Count (requests) │    20.00 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│               Goodput (requests/sec) │     0.14 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
└──────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```
