<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling with AIPerf

This tutorial demonstrates how to use AIPerf to measure the performance of models using various inference solutions and benchmark scenarios.

## Table of Contents

- [Basic Chat Benchmark (vLLM)](#basic-chat-benchmark-vllm)
- [Embeddings Benchmark](#embeddings-benchmark)
- [Multi-Turn Conversations](#multi-turn-conversations)
- [Custom Dataset Benchmark](#custom-dataset-benchmark)
- [Load Testing with Goodput](#load-testing-with-goodput)
- [Understanding the Output](#understanding-the-output)
- [Next Steps](#next-steps)

## Basic Chat Benchmark (vLLM) <a id="basic-chat-benchmark-vllm">
<!-- setup-vllm-default-openai-endpoint-server -->
```bash
# Pull and run vLLM Docker container:
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
```
<!-- /setup-vllm-default-openai-endpoint-server -->

<!-- health-check-vllm-default-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-default-openai-endpoint-server -->


<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile the model
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 32 \
    --request-count 64 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 64/64 |████████████████████████| 100% [00:42<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-request_rate32/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1234.56 │ 987.34 │ 1567.89 │ 1534.23 │ 1223.45 │
│    Time to First Token (ms) │  234.56 │ 189.23 │  298.45 │  289.34 │  231.12 │
│    Inter Token Latency (ms) │   15.67 │  12.34 │   19.45 │   19.01 │   15.45 │
│ Output Token Count (tokens) │  150.00 │ 120.00 │  180.00 │  178.90 │  149.00 │
│  Request Throughput (req/s) │   31.45 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-request_rate32/profile_export_aiperf.json
```

**What this shows:**
- **Request Latency**: Total end-to-end time per request
- **Time to First Token (TTFT)**: How quickly the first token arrives (critical for interactive UX)
- **Inter Token Latency (ITL)**: Average time between tokens (affects streaming smoothness)
- **Request Throughput**: Completed requests per second

See [Metrics Reference](metrics_reference.md) for complete metric definitions.

---

## Embeddings Benchmark

Benchmark an embeddings endpoint for vector database ingestion:

```bash
# Start an embeddings server (example with HuggingFace TEI)
docker run -p 8000:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-small-en-v1.5

# Wait for server to be ready
sleep 30

# Benchmark embeddings with batch processing
aiperf profile \
  --model BAAI/bge-small-en-v1.5 \
  --endpoint-type embeddings \
  --url http://localhost:8000 \
  --prompt-batch-size 10 \
  --concurrency 50 \
  --request-count 500 \
  --isl 128
```

**Key differences from chat:**
- No `--streaming` flag (embeddings return complete vectors)
- `--prompt-batch-size 10` sends 10 texts per request (efficient batch processing)
- Higher concurrency (embeddings are typically faster than generation)

**Sample Output:**
```
NVIDIA AIPerf | Embeddings Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │   45.23 │  32.15 │   78.90 │
│  Request Throughput (req/s) │  342.50 │      - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┘
```

See [Embeddings Tutorial](tutorials/embeddings.md) for advanced features.

---

## Multi-Turn Conversations

Simulate realistic chat application usage with multi-turn conversations:

```bash
# Benchmark with conversation patterns
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  \
  --num-sessions 50 \
  --session-turns-mean 5 \
  --session-turns-stddev 2 \
  --session-turn-delay-mean 2000 \
  --session-turn-delay-stddev 500 \
  \
  --warmup-request-count 25
```

**Configuration breakdown:**
- `--num-sessions 50`: 50 concurrent conversation sessions
- `--session-turns-mean 5`: Average 5 exchanges per conversation (±2)
- `--session-turn-delay-mean 2000`: 2 seconds average "think time" between turns (±0.5s)

**What this simulates:**
```
User → "Hello" → Server → Response
  ↓ (2 seconds think time)
User → "Follow-up question" → Server → Response
  ↓ (2 seconds think time)
User → "Another question" → Server → Response
  ... (continues for ~5 turns)
```

See [Multi-Turn Tutorial](tutorials/multi-turn.md) for advanced conversation patterns.

---

## Custom Dataset Benchmark

Use your own prompts instead of synthetic data:

**Step 1: Prepare your dataset**

Create a JSONL file with your prompts:

```bash
# Create prompts.jsonl
cat > prompts.jsonl <<EOF
{"prompt": "Explain quantum computing in simple terms"}
{"prompt": "Write a haiku about artificial intelligence"}
{"prompt": "What are the benefits of serverless architecture?"}
{"prompt": "Summarize the main points of climate change"}
EOF
```

**Step 2: Run benchmark with custom dataset**

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  \
  --input-file prompts.jsonl \
  --custom-dataset-type single_turn \
  --dataset-sampling-strategy sequential \
  \
  --concurrency 10 \
  --warmup-request-count 10
```

**Sampling strategies:**
- `sequential`: Use prompts in order (good for deterministic testing)
- `shuffle`: Randomize order but use each prompt once per cycle
- `random`: Random sampling with replacement

**For multi-turn conversations:**

```jsonl
{"conversation": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing well, thank you!"},
  {"role": "user", "content": "Can you help me with Python?"}
]}
```

```bash
aiperf profile \
  --input-file conversations.jsonl \
  --custom-dataset-type multi_turn \
  --url http://localhost:8000 \
  --endpoint-type chat
```

See [Custom Prompt Benchmarking Tutorial](tutorials/custom-prompt-benchmarking.md) for format details.

---

## Load Testing with Goodput

Measure throughput while enforcing Service Level Objectives (SLOs):

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  \
  --concurrency 50 \
  --benchmark-duration 300 \
  --warmup-duration 30 \
  \
  --goodput "time_to_first_token:50 inter_token_latency:10" \
  \
  --random-seed 42
```

**What this measures:**
- **Request Throughput**: Total requests/sec (all requests)
- **Goodput**: Requests/sec meeting SLOs (only "good" requests)
- **SLO Compliance %**: Percentage of requests within targets

**Sample Output:**
```
NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│    Time to First Token (ms) │   48.26 │  68.82 │   45.12 │
│    Inter Token Latency (ms) │    9.23 │  12.48 │    8.95 │
│  Request Throughput (req/s) │   87.50 │      - │       - │
│               Goodput (req/s) │   74.30 │      - │       - │
│                               │         │        │         │
│        SLO Compliance: 84.9% │         │        │         │
└─────────────────────────────┴─────────┴────────┴─────────┘
```

**Interpretation:**
- 87.5 requests/sec total throughput
- 74.3 requests/sec meet SLOs (84.9% compliance)
- 15.1% of requests exceed TTFT or ITL targets

See [Goodput Tutorial](tutorials/goodput.md) for advanced SLO configuration.

---

## Understanding the Output

### Console Metrics Table

AIPerf displays key metrics in a table during and after the benchmark:

| Metric | What It Means | Good Values |
|--------|---------------|-------------|
| **Time to First Token (TTFT)** | Latency until first token arrives | < 50ms for interactive chat |
| **Inter Token Latency (ITL)** | Time between tokens during generation | < 20ms for smooth streaming |
| **Request Latency** | Total end-to-end time per request | Depends on output length |
| **Output Token Throughput** | Total tokens generated per second | Higher is better |
| **Request Throughput** | Requests completed per second | Higher is better |

### Output Files

All results are saved to the artifacts directory:

```
artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
├── profile_export_aiperf.csv      # Summary metrics (spreadsheet-ready)
├── profile_export_aiperf.json     # Summary metrics with metadata
├── profile_export.jsonl           # Per-request detailed metrics
└── profile_export_raw.jsonl       # Raw request/response data (with --export-level raw)
```

**For analysis:**
```bash
# View CSV in Excel/Google Sheets
open artifacts/*/profile_export_aiperf.csv

# Parse JSON programmatically
python analyze_results.py artifacts/*/profile_export_aiperf.json

# Generate plots
aiperf plot --paths artifacts/*/
```

See [Working with Profile Exports Tutorial](tutorials/working-with-profile-exports.md) for programmatic parsing.

### Percentiles Explained

Don't just look at averages! Percentiles show the full performance picture:

- **P50 (Median)**: Typical user experience (50% of requests are faster)
- **P90**: 90% of users see this or better (capacity planning)
- **P99**: Catches outliers (1% of requests slower than this)
- **Max**: Worst-case scenario

**Example:**
```
TTFT avg: 50ms, p50: 45ms, p99: 120ms
```
- Most users see ~45ms (good!)
- But 1% experience >120ms (investigate outliers)

---

## Next Steps

### Learn More Features

- **[Examples Gallery](examples.md)** - Real-world benchmark scenarios
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
- **[Metrics Reference](metrics_reference.md)** - Complete metrics guide

### Optimize Performance

- **[Performance Tuning](performance-tuning.md)** - Maximize throughput and minimize latency
- **[Deployment Guide](deployment.md)** - Docker, Kubernetes, production setups

### Advanced Features

- **[Arrival Patterns](tutorials/arrival-patterns.md)** - Realistic traffic simulation
- **[Trace Replay](tutorials/trace-benchmarking.md)** - Deterministic workload replay
- **[GPU Telemetry](tutorials/gpu-telemetry.md)** - Monitor GPU metrics
- **[Server Metrics](server_metrics/server-metrics.md)** - Collect Prometheus metrics
- **[Plotting](tutorials/plot.md)** - Visualize results

### Get Help

- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[FAQ](faq.md)** - Frequently asked questions
- **[Discord](https://discord.gg/D92uqZRjCZ)** - Join our community