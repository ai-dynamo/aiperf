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
| **Single Turn** | Independent single requests | No | Yes | No |
| **Multi Turn** | Conversations with context | Yes | Yes (per turn) | No |
| **Random Pool** | Load testing with variety | No | No | Yes |

**All three support:**
- Client-side batching
- Automatic media handling: local files are converted to base64 format, while remote URLs are sent directly to the API

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

Use single_turn when you need **deterministic, sequential execution** where requests always run in the exact order they appear in the file:

- **Debugging**: Test specific prompts in a known sequence
- **Regression testing**: Same input file → same output order every time
- **Timing control**: Schedule requests with precise timestamps or delays
- **Predictable testing**: Know exactly which request runs when

**Execution:** Sequential by default (request 1, then 2, then 3, etc.)
**Input:** Single JSONL file only

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

**Output:**
```


                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃               Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│  Time to First Token │    19.99 │    12.53 │    49.62 │    48.89 │    42.24 │    13.93 │    12.92 │
│                 (ms) │          │          │          │          │          │          │          │
│ Time to Second Token │     3.81 │     2.01 │     8.25 │     7.94 │     5.15 │     3.36 │     1.62 │
│                 (ms) │          │          │          │          │          │          │          │
│ Time to First Output │    19.99 │    12.53 │    49.62 │    48.89 │    42.24 │    13.93 │    12.92 │
│           Token (ms) │          │          │          │          │          │          │          │
│ Request Latency (ms) │ 2,940.39 │ 1,536.67 │ 7,319.35 │ 7,034.86 │ 4,474.42 │ 2,239.67 │ 1,611.04 │
│  Inter Token Latency │     3.52 │     3.47 │     3.64 │     3.63 │     3.56 │     3.50 │     0.05 │
│                 (ms) │          │          │          │          │          │          │          │
│         Output Token │   284.54 │   274.60 │   288.35 │   288.33 │   288.13 │   285.38 │     3.98 │
│  Throughput Per User │          │          │          │          │          │          │          │
│    (tokens/sec/user) │          │          │          │          │          │          │          │
│      Output Sequence │   833.40 │   438.00 │ 2,106.00 │ 2,022.21 │ 1,268.10 │   626.50 │   465.81 │
│      Length (tokens) │          │          │          │          │          │          │          │
│       Input Sequence │     5.00 │     4.00 │     7.00 │     7.00 │     7.00 │     5.00 │     1.10 │
│      Length (tokens) │          │          │          │          │          │          │          │
│         Output Token │   527.06 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│           Throughput │          │          │          │          │          │          │          │
│         (tokens/sec) │          │          │          │          │          │          │          │
│   Request Throughput │     0.63 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│       (requests/sec) │          │          │          │          │          │          │          │
│        Request Count │    10.00 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│           (requests) │          │          │          │          │          │          │          │
└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'prompts.jsonl' --custom-dataset-type 'single_turn' --streaming --url 'localhost:8000' --concurrency
2
Benchmark Duration: 15.81 sec
CSV Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.csv
JSON Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.json
Log File: /home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/logs/aiperf.log
```

---

## Multi-Turn Datasets

Each entry represents a complete conversation with multiple turns.

### When to Use

Use multi_turn when you need **conversations with context** where each turn builds on previous turns in the conversation:

- **Chat testing**: Test conversational AI that maintains context across turns
- **Realistic interactions**: Simulate real user conversations with follow-up questions
- **Task completion**: Test multi-step tasks that require conversation history

**Execution:** Sequential within each conversation (turn 1, then 2, then 3, etc.), but multiple conversations run concurrently
**Input:** Single JSONL file only

### Basic Conversation

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > conversations.jsonl << 'EOF'
{"session_id": "chat_1", "turns": [{"text": "What is machine learning?"}, {"text": "Can you give me an example?"}]}
{"session_id": "chat_2", "turns": [{"text": "Explain neural networks."}, {"text": "How do they differ from traditional algorithms?"}, {"text": "Which architecture for image classification?"}]}
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

**Output:**
```
                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃                 Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│    Time to First Token │    23.17 │    11.83 │    56.70 │    55.34 │    43.06 │    18.00 │  13.66 │
│                   (ms) │          │          │          │          │          │          │        │
│   Time to Second Token │     4.77 │     2.29 │    15.41 │    14.65 │     7.73 │     3.44 │   3.74 │
│                   (ms) │          │          │          │          │          │          │        │
│   Time to First Output │    23.17 │    11.83 │    56.70 │    55.34 │    43.06 │    18.00 │  13.66 │
│             Token (ms) │          │          │          │          │          │          │        │
│   Request Latency (ms) │ 2,008.84 │ 1,348.13 │ 3,045.04 │ 3,007.53 │ 2,669.92 │ 2,082.32 │ 572.34 │
│    Inter Token Latency │     3.50 │     3.13 │     3.67 │     3.67 │     3.62 │     3.52 │   0.14 │
│                   (ms) │          │          │          │          │          │          │        │
│           Output Token │   286.03 │   272.35 │   319.58 │   316.89 │   292.60 │   283.77 │  12.33 │
│    Throughput Per User │          │          │          │          │          │          │        │
│      (tokens/sec/user) │          │          │          │          │          │          │        │
│ Output Sequence Length │   565.60 │   380.00 │   838.00 │   826.57 │   723.70 │   581.50 │ 150.96 │
│               (tokens) │          │          │          │          │          │          │        │
│  Input Sequence Length │   379.80 │     5.00 │ 1,331.00 │ 1,287.80 │   899.00 │   203.00 │ 438.88 │
│               (tokens) │          │          │          │          │          │          │        │
│           Output Token │   533.83 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             Throughput │          │          │          │          │          │          │        │
│           (tokens/sec) │          │          │          │          │          │          │        │
│     Request Throughput │     0.94 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│         (requests/sec) │          │          │          │          │          │          │        │
│          Request Count │    10.00 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             (requests) │          │          │          │          │          │          │        │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'conversations.jsonl' --custom-dataset-type 'multi_turn' --streaming --url 'localhost:8000'
--concurrency 2
Benchmark Duration: 10.60 sec
CSV Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.csv
JSON Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.json
Log File: /home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/logs/aiperf.log
```

**Key Points:**
- Each turn includes full conversation history
- Turns execute sequentially within each conversation
- Multiple conversations run concurrently (up to `--concurrency`)

---

## Random Pool Datasets

Randomly sample from one or more data pools for varied request patterns.

### When to Use

Use random_pool when you need **random sampling with replacement** for unpredictable, varied request patterns:

- **Load testing**: Generate diverse request patterns with variety
- **Production simulation**: Model real-world workloads where requests vary
- **Stress testing**: Test system behavior under mixed input patterns
- **Multiple data sources**: Combine files from a directory (each file becomes a pool)

**Execution:** Random sampling with replacement (same entry can be selected multiple times)
**Input:** Single JSONL file OR directory of multiple JSONL files
**Note:** Does NOT support timing control or multi-turn conversations

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
    --num-conversations 50 \
    --streaming \
    --concurrency 4 \
    --random-seed 42 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Output:**
```
                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃              Metric ┃      avg ┃      min ┃       max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to First Token │    17.73 │    12.25 │     53.21 │    53.17 │    19.85 │    14.63 │     9.90 │
│                (ms) │          │          │           │          │          │          │          │
│      Time to Second │     3.73 │     2.20 │     10.38 │     7.68 │     4.08 │     3.66 │     1.10 │
│          Token (ms) │          │          │           │          │          │          │          │
│       Time to First │    17.73 │    12.25 │     53.21 │    53.17 │    19.85 │    14.63 │     9.90 │
│   Output Token (ms) │          │          │           │          │          │          │          │
│     Request Latency │ 3,321.54 │ 1,356.57 │ 10,393.82 │ 9,063.81 │ 5,372.92 │ 2,917.73 │ 1,644.46 │
│                (ms) │          │          │           │          │          │          │          │
│ Inter Token Latency │     3.81 │     3.53 │      4.17 │     4.15 │     3.97 │     3.79 │     0.12 │
│                (ms) │          │          │           │          │          │          │          │
│        Output Token │   262.66 │   239.55 │    283.24 │   279.36 │   270.36 │   264.13 │     8.25 │
│ Throughput Per User │          │          │           │          │          │          │          │
│   (tokens/sec/user) │          │          │           │          │          │          │          │
│     Output Sequence │   861.02 │   369.00 │  2,615.00 │ 2,255.83 │ 1,306.40 │   766.00 │   404.28 │
│     Length (tokens) │          │          │           │          │          │          │          │
│      Input Sequence │     5.00 │     4.00 │      7.00 │     7.00 │     6.10 │     5.00 │     0.96 │
│     Length (tokens) │          │          │           │          │          │          │          │
│        Output Token │ 1,007.36 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│          Throughput │          │          │           │          │          │          │          │
│        (tokens/sec) │          │          │           │          │          │          │          │
│  Request Throughput │     1.17 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│      (requests/sec) │          │          │           │          │          │          │          │
│       Request Count │    50.00 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│          (requests) │          │          │           │          │          │          │          │
└─────────────────────┴──────────┴──────────┴───────────┴──────────┴──────────┴──────────┴──────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'pool.jsonl' --custom-dataset-type 'random_pool' --num-conversations 50 --streaming --concurrency 4
--random-seed 42 --url 'localhost:8000'
Benchmark Duration: 42.74 sec
CSV Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/profile_export_aiperf.csv
JSON Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/profile_export_aiperf.json
Log File: /home/lkomali/aiperf/artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/logs/aiperf.log
```

**Behavior:**
- Randomly samples 50 requests from 8-entry pool
- Sampling with replacement (entries can repeat)
- Use `--random-seed` for reproducibility
