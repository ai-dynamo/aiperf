<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Dataset Guide

Benchmark LLMs with your own data using single-turn requests, multi-turn conversations, or random sampling.

## Overview

AIPerf supports three custom dataset types for benchmarking with your own data:

| Dataset Type | Best For | Multi-Turn | Timing Control | Random Sampling |
|-------------|----------|-----------|---------------|-----------------|
| **single_turn** | Independent single requests | No | Yes | No |
| **multi_turn** | Conversations with context | Yes | Yes (per turn) | No |
| **random_pool** | Load testing with variety | No | No | Yes |

**All three support:**
- Multi-modal data (text, images, audio, video)
- Client-side batching
- Local file encoding (base64)
- URL pass-through

---

## Server Setup

Start a vLLM server for testing:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

Verify the server is ready:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }' | jq
```

---

## Single-Turn Datasets

Each line represents one independent single-turn request.

### When to Use

- Testing individual prompts with known inputs
- Debugging specific request configurations
- Load testing with predetermined patterns
- Request timing control needed

### Basic Text Example

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > prompts.jsonl << 'EOF'
{"text": "What is machine learning?"}
{"text": "Explain neural networks."}
{"text": "How does backpropagation work?"}
{"text": "What are transformers?"}
{"text": "Define reinforcement learning."}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file prompts.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output:**
```
INFO     Starting AIPerf System
INFO     Loaded 5 entries from prompts.jsonl
INFO     Using single_turn dataset type

Profiling: 5/5 |████████████████████████| 100% [00:08<00:00]

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1123.45 │ 789.23 │ 1456.78 │ 1456.78 │ 1098.34 │
│    Time to First Token (ms) │   42.34 │  31.23 │   58.90 │   58.90 │   40.12 │
│    Inter Token Latency (ms) │   11.23 │   8.90 │   14.56 │   14.56 │   10.89 │
│ Output Token Count (tokens) │   95.20 │  78.00 │  120.00 │  120.00 │   92.00 │
│  Request Throughput (req/s) │    4.56 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘
```

---

## Multi-Turn Datasets

Each entry represents a complete conversation with multiple turns.

### When to Use

- Testing conversational AI with context
- Each turn builds on previous turns
- Simulating realistic chat interactions
- Benchmarking multi-turn task completion

### Basic Conversation

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > conversations.jsonl << 'EOF'
{
  "session_id": "chat_1",
  "turns": [
    {"text": "What is machine learning?"},
    {"text": "Can you give me an example?"}
  ]
}
{
  "session_id": "chat_2",
  "turns": [
    {"text": "Explain neural networks."},
    {"text": "How do they differ from traditional algorithms?"},
    {"text": "Which architecture for image classification?"}
  ]
}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file conversations.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output:**
```
INFO     Loaded 2 conversations from conversations.jsonl
INFO     Total turns: 5

Profiling: 5/5 |████████████████████████| 100% [00:15<00:00]

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1234.56 │ 890.12 │ 1678.90 │ 1678.90 │ 1189.45 │
│    Time to First Token (ms) │   45.67 │  32.34 │   62.89 │   62.89 │   43.21 │
│    Inter Token Latency (ms) │   12.34 │   9.45 │   15.67 │   15.67 │   11.89 │
│ Output Token Count (tokens) │  102.30 │  85.00 │  135.00 │  135.00 │   98.00 │
│  Request Throughput (req/s) │    3.45 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘
```

**Key Points:**
- Each turn includes full conversation history
- Turns execute sequentially within each conversation
- Multiple conversations run concurrently (up to `--concurrency`)

---

## Random Pool Datasets

Randomly sample from one or more data pools for varied request patterns.

### When to Use

- Load testing with unpredictable patterns
- Simulating production workloads
- Combining data sources (query + passage)
- Reranking or embedding benchmarks
- You don't need conversation context or timing

### Basic Single-File Sampling

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > pool.jsonl << 'EOF'
{"text": "What is machine learning?"}
{"text": "Explain neural networks."}
{"text": "How does backpropagation work?"}
{"text": "What are transformers?"}
{"text": "Define reinforcement learning."}
{"text": "What is transfer learning?"}
{"text": "Explain gradient descent."}
{"text": "What are GANs?"}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file pool.jsonl \
    --custom-dataset-type random_pool \
    --request-count 50 \
    --concurrency 4 \
    --random-seed 42 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output:**
```
INFO     Loaded 8 entries into random pool
INFO     Will generate 50 requests by sampling with replacement

Profiling: 50/50 |████████████████████████| 100% [00:28<00:00]

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1156.78 │ 845.67 │ 1598.34 │ 1578.90 │ 1123.45 │
│    Time to First Token (ms) │   43.56 │  29.87 │   64.23 │   62.34 │   41.89 │
│    Inter Token Latency (ms) │   11.89 │   8.56 │   15.78 │   15.23 │   11.45 │
│ Output Token Count (tokens) │   98.50 │  76.00 │  128.00 │  125.00 │   96.00 │
│  Request Throughput (req/s) │   17.86 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘
```

**Behavior:**
- Randomly samples 50 requests from 8-entry pool
- Sampling with replacement (entries can repeat)
- Use `--random-seed` for reproducibility
