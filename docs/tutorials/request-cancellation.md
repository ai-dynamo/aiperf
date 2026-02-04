<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Request Cancellation Testing

AIPerf supports request timeout and cancellation scenarios, which are important for calculating the impact of user cancellation on performance.

## How Request Cancellation Works

Request cancellation tests how inference servers handle client disconnections. A percentage of requests are sent completely, then the client disconnects before receiving the full response.

### Timing Flow

```
T0: Request scheduled
     │
     │← Worker processing, connection acquired from pool
     ▼
T1: Start writing request to socket
     │
     │← HTTP headers + body transmitted
     ▼
T2: Request fully sent (cancellation timer starts here)
     │
     │← --request-cancellation-delay
     ▼
T3: Request cancelled if still waiting for response
```

The cancellation timer starts at **T2** ("request fully sent") for two reasons:

1. **Realistic simulation**: The server always receives the complete request before cancellation, just like when a real user closes their browser tab.

2. **Reproducibility**: The delay is measured from a fixed point (request fully sent) rather than being affected by variable queue times or connection setup. This means running the same benchmark twice with `--request-cancellation-delay 0.5` will cancel requests at the same point in their lifecycle, regardless of system load.

> [!NOTE]
> If the server responds before the delay expires, the request completes normally and is **not** cancelled. Only requests still waiting for a response when the timer expires are cancelled.

### Understanding the Delay Parameter

| Delay | Behavior |
|-------|----------|
| `0` | Disconnect immediately after request is fully sent |
| `0.5` | Wait 0.5 seconds after sending, then disconnect |
| `5` | Wait 5 seconds after sending, then disconnect |

> [!TIP]
> A delay of **0 means "send the full request, then immediately disconnect"**. The server receives the complete request but the client closes the connection before receiving any response. Longer delays allow partial responses to be received before disconnection.

### Testing Disaggregated Inference Systems

The delay parameter can be used to target different inference phases:

| Delay | Likely Cancelled During | Tests |
|-------|------------------------|-------|
| `0` or very small | **Prefill phase** | Prefill worker cancellation, KV cache allocation cleanup |
| Longer delays | **Generation phase** | Decode worker cancellation, partial KV cache cleanup |

This is useful for testing how disaggregated architectures (separate prefill and decode workers) handle cancellations at different stages of request processing.

## Setting Up the Server

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

## Basic Request Cancellation

Test with a small percentage of cancelled requests:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with 10% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 10 \
    --request-cancellation-delay 0.5 \
    --synthetic-input-tokens-mean 800 \
    --synthetic-input-tokens-stddev 80 \
    --output-tokens-mean 400 \
    --output-tokens-stddev 40 \
    --concurrency 8 \
    --request-count 50 \
    --warmup-request-count 5
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Request cancellation enabled: 10.0% rate, 0.5s delay
INFO     AIPerf System is WARMING UP

Warming Up: 5/5 |████████████████████████| 100% [00:04<00:00]

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: 50/50 |████████████████████████| 100% [01:15<00:00]

INFO     Benchmark completed successfully
INFO     Cancelled requests: 5 (10.0%)
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency8/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1234.56 │ 987.34 │ 1678.90 │ 1598.23 │ 1198.45 │
│    Time to First Token (ms) │  234.56 │ 187.90 │  298.34 │  289.67 │  228.12 │
│    Inter Token Latency (ms) │   14.23 │  11.45 │   19.67 │   18.90 │   13.89 │
│ Output Token Count (tokens) │  400.00 │ 360.00 │  440.00 │  438.00 │  398.00 │
│  Request Throughput (req/s) │   12.34 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency8/profile_export_aiperf.json
```

**Parameters Explained:**
- `--request-cancellation-rate 10`: Cancel 10% of requests (value between 0.0 and 100.0)
- `--request-cancellation-delay 0.5`: Wait .5 seconds before cancelling selected requests

### High Cancellation Rate Testing

Test service resilience under frequent cancellations:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with 50% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 50 \
    --request-cancellation-delay 1.0 \
    --synthetic-input-tokens-mean 1200 \
    --output-tokens-mean 600 \
    --concurrency 10 \
    --request-count 40
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Request cancellation enabled: 50.0% rate, 1.0s delay
INFO     AIPerf System is PROFILING

Profiling: 40/40 |████████████████████████| 100% [01:30<00:00]

INFO     Benchmark completed successfully
INFO     Cancelled requests: 20 (50.0%)
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1890.45 │ 1456.78 │ 2456.89 │ 2398.12 │ 1867.34 │
│    Time to First Token (ms) │  345.67 │  278.90 │  456.23 │  445.67 │  338.45 │
│    Inter Token Latency (ms) │   16.78 │   13.45 │   22.34 │   21.56 │   16.45 │
│ Output Token Count (tokens) │  600.00 │  540.00 │  660.00 │  658.00 │  598.00 │
│  Request Throughput (req/s) │    8.90 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/profile_export_aiperf.json
```

### Immediate Cancellation Testing (Delay = 0)

Test immediate disconnection where the client closes the connection right after sending the request:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile with immediate cancellation (0 delay)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 30 \
    --request-cancellation-delay 0.0 \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 100 \
    --concurrency 15 \
    --request-count 60
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Request cancellation enabled: 30.0% rate, 0.0s delay (immediate)
INFO     AIPerf System is PROFILING

Profiling: 60/60 |████████████████████████| 100% [00:45<00:00]

INFO     Benchmark completed successfully
INFO     Cancelled requests: 18 (30.0%)
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency15/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                      Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│        Request Latency (ms) │ 678.90 │ 534.56 │ 898.12 │ 876.34 │ 667.89 │
│    Time to First Token (ms) │ 156.78 │ 123.45 │ 198.90 │ 192.34 │ 154.23 │
│    Inter Token Latency (ms) │  12.45 │   9.89 │  16.78 │  16.12 │  12.23 │
│ Output Token Count (tokens) │ 100.00 │  90.00 │ 110.00 │ 109.00 │  99.00 │
│  Request Throughput (req/s) │  23.45 │      - │      - │      - │      - │
└─────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency15/profile_export_aiperf.json
```

**What happens with delay=0:**
- The full request (headers + body) is sent to the server
- The client immediately disconnects after sending
- The server receives the complete request but the client won't read any response
- Tests how the server handles abandoned requests and cleans up resources
