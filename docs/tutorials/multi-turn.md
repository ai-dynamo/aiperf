<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-Turn Conversations

Multi-turn conversations allow you to benchmark chat-based models with realistic back-and-forth dialogue patterns. This feature simulates real-world scenarios where users engage in extended conversations with multiple exchanges, rather than isolated single-turn queries.

## Overview

Multi-turn benchmarking provides several advantages:

- **Realistic Chat Simulation**: Model actual user interactions with conversational AI systems
- **Context Window Testing**: Evaluate performance as conversation history grows
- **Session-based Load**: Test how servers handle sustained multi-turn sessions
- **Memory and State Management**: Identify issues with conversation state handling
- **Conversation Flow Analysis**: Measure performance degradation over multiple turns

> [!IMPORTANT]
> **Understanding Request Control Options**
>
> AIPerf provides different options for controlling the number of requests depending on whether you're running single-turn or multi-turn benchmarks:
>
> - **`--request-count`**: Controls the total number of **single-turn requests** to send. Use this for traditional single-turn benchmarks.
> - **`--conversation-num`**: Controls the total number of **conversations (sessions)** to send in multi-turn scenarios. Each conversation may contain multiple turns (requests).
>
> These options are mutually exclusive in their intent - use `--request-count` for single-turn benchmarking and `--conversation-num` for multi-turn benchmarking to avoid confusion.

> [!NOTE]
> **Dataset Generation vs Request Execution**
>
> The `--num-dataset-entries` option controls how many **unique prompts** are generated in the dataset. This is separate from the number of requests or conversations:
>
> - `--num-dataset-entries`: Number of unique prompt entries to generate in the dataset
> - `--request-count`: Number of single-turn requests to send (for single-turn benchmarks)
> - `--conversation-num`: Number of conversations to send (for multi-turn benchmarks)
>
> The dataset entries are reused/sampled as needed to fulfill the total request or conversation count. For example, you might generate 100 unique prompts (`--num-dataset-entries 100`) but send 1000 requests that sample from those prompts. `--dataset-sampling-strategy` determines how the pool of prompts is sampled when building payloads.

## Core Parameters

### Conversation Control

- **`--conversation-num <N>`**: Total number of unique conversation sessions to execute
  - Aliases: `--num-conversations`, `--num-sessions`
  - Each conversation represents a complete multi-turn dialogue session

### Turn Configuration

- **`--conversation-turn-mean <N>`**: Average number of turns per conversation
  - Default: 1 (single-turn)
  - Aliases: `--session-turns-mean`

- **`--conversation-turn-stddev <N>`**: Standard deviation for number of turns
  - Default: 0 (fixed number of turns)
  - Aliases: `--session-turns-stddev`

### Turn Delays

- **`--conversation-turn-delay-mean <MS>`**: Average delay between turns in milliseconds
  - Default: 0ms
  - Simulates realistic user "think time" between messages
  - Aliases: `--session-turn-delay-mean`

- **`--conversation-turn-delay-stddev <MS>`**: Standard deviation for turn delays
  - Default: 0ms
  - Adds natural variance to delays
  - Aliases: `--session-turn-delay-stddev`

## Setting Up the Server

```bash
# Start vLLM server for multi-turn benchmarking
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```


## Basic Multi-Turn Examples

### Fixed-Length Conversations

Run a simple multi-turn benchmark with a fixed number of turns per conversation:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Run 10 conversations, each with exactly 3 turns
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 10 \
    --conversation-turn-mean 3 \
    --conversation-turn-stddev 0 \
    --synthetic-input-tokens-mean 200 \
    --output-tokens-mean 150 \
    --concurrency 2 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Multi-turn mode: 10 conversations, 3 turns each (30 total requests)
INFO     AIPerf System is PROFILING

Profiling: 30/30 |████████████████████████| 100% [00:45<00:00]
  Conversations: 10/10 complete

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency2/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                      Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│        Request Latency (ms) │ 234.56 │ 198.23 │ 287.45 │ 276.12 │ 232.45 │
│    Time to First Token (ms) │  52.34 │  45.12 │  68.23 │  65.45 │  51.89 │
│ Output Token Count (tokens) │ 150.00 │ 150.00 │ 150.00 │ 150.00 │ 150.00 │
└─────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency2/profile_export_aiperf.json
```

This command will:
- Execute 10 separate conversation sessions
- Each conversation will have exactly 3 turns (requests)
- Total requests sent: 10 conversations × 3 turns = 30 requests
- 2 conversations will run concurrently

### Variable-Length Conversations

Add variance to the number of turns per conversation for more realistic patterns:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Run conversations with variable lengths (mean: 5, stddev: 2)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 20 \
    --conversation-turn-mean 5 \
    --conversation-turn-stddev 2 \
    --synthetic-input-tokens-mean 150 \
    --output-tokens-mean 100 \
    --concurrency 4 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

This creates conversations with varying lengths (typically 3-7 turns), simulating natural conversation patterns where some users ask quick questions and others engage in deeper discussions.

## Advanced Multi-Turn Scenarios

### Realistic User Behavior with Turn Delays

Simulate real user "think time" between turns to model actual human interaction patterns:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Add realistic delays between turns (mean: 2000ms, stddev: 500ms)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 15 \
    --conversation-turn-mean 4 \
    --conversation-turn-stddev 1 \
    --conversation-turn-delay-mean 2000 \
    --conversation-turn-delay-stddev 500 \
    --synthetic-input-tokens-mean 180 \
    --output-tokens-mean 120 \
    --concurrency 3 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

The turn delays simulate realistic pauses as users read responses and formulate follow-up questions. This is critical for:
- Testing connection keep-alive mechanisms
- Evaluating server-side session state management
- Measuring sustained performance under realistic load

### High-Concurrency Multi-Turn Sessions

Test how your server handles many simultaneous multi-turn conversations:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Run 50 concurrent conversations with variable lengths
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 100 \
    --conversation-turn-mean 6 \
    --conversation-turn-stddev 2 \
    --synthetic-input-tokens-mean 250 \
    --output-tokens-mean 200 \
    --concurrency 50 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

This benchmark:
- Maintains 50 active conversations simultaneously
- Tests session isolation and resource management
- Identifies scalability bottlenecks with multiple concurrent sessions

### Request Rate with Multi-Turn Conversations

Combine request rate control with multi-turn conversations for controlled, sustained load:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Start new conversations at 5 conversations/second
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 30 \
    --conversation-turn-mean 4 \
    --request-rate 5 \
    --request-rate-mode poisson \
    --synthetic-input-tokens-mean 200 \
    --output-tokens-mean 150 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

This approach is ideal for:
- Modeling steady conversation arrival patterns
- Avoiding thundering herd problems during testing
- Measuring performance under controlled, sustained multi-turn load

## Use Cases

### Customer Support Chatbot Testing

Simulate realistic customer support interactions with varying conversation lengths:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Model customer support conversations:
# - Average 6-8 turns per conversation
# - Natural delays between user messages
# - Mix of short and long input/output sequences

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 50 \
    --conversation-turn-mean 7 \
    --conversation-turn-stddev 2 \
    --conversation-turn-delay-mean 3000 \
    --conversation-turn-delay-stddev 1000 \
    --synthetic-input-tokens-mean 150 \
    --synthetic-input-tokens-stddev 50 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 80 \
    --concurrency 10 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

### Context Window Stress Testing

Test model performance with long conversations that accumulate substantial context:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Test long conversations with growing context
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 10 \
    --conversation-turn-mean 15 \
    --conversation-turn-stddev 3 \
    --synthetic-input-tokens-mean 300 \
    --output-tokens-mean 250 \
    --concurrency 2 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

Each turn in a conversation includes the full conversation history, so:
- Turn 1: ~300 tokens input
- Turn 5: ~300 + (4 × 250) = ~1300 tokens input
- Turn 15: ~300 + (14 × 250) = ~3800 tokens input

This helps identify performance degradation as context grows.

### Burst Traffic Simulation

Simulate sudden spikes in conversation activity:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Simulate burst of conversation starts
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 100 \
    --conversation-turn-mean 3 \
    --concurrency 50 \
    --synthetic-input-tokens-mean 180 \
    --output-tokens-mean 120 \
    --random-seed 42
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

## How Multi-Turn Works

### Message History Accumulation

In multi-turn conversations, each subsequent turn includes the complete conversation history:

**Turn 1:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ]
}
```

**Turn 2:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give an example?"}
  ]
}
```

**Turn 3:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give an example?"},
    {"role": "assistant", "content": "Sure! One example is..."},
    {"role": "user", "content": "How does it differ from traditional programming?"}
  ]
}
```

This accumulation means:
- Input token count grows with each turn
- Later turns have increasingly large context to process

### Real-World Conversation Flow

AIPerf simulates realistic multi-turn conversations by modeling natural user behavior patterns. Here's how a typical multi-turn conversation flows:

**Turn 0 (First Turn):**
- User sends initial message → AI responds
- **No delay** before this turn (users don't wait to start conversations)

**Turn 1 (Second Turn):**
- **DELAY**: User reads AI's response and thinks about next message (configurable delay applied)
- User sends follow-up message → AI responds

**Turn 2 (Third Turn):**
- **DELAY**: User reads AI's response and thinks about next message (configurable delay applied)
- User sends next message → AI responds

**...and so on for subsequent turns**

This flow pattern ensures benchmarks reflect real-world usage where:
- Users need time to read and process AI responses
- There's natural thinking/typing time between messages
- The first message is sent immediately when starting a conversation
- Delays are applied **before** sending each subsequent turn

The delays between turns are controlled by:
- `--conversation-turn-delay-mean`: Average delay in milliseconds (e.g., 2000ms = 2 seconds)
- `--conversation-turn-delay-stddev`: Variation in delays to simulate natural human behavior
- `--conversation-turn-delay-ratio`: Scaling factor for all delays

### Execution Flow

1. **Dataset Generation**: AIPerf generates the specified number of conversations, each with a random number of turns based on your mean and stddev
2. **Conversation Distribution**: Conversations are distributed to workers according to concurrency and rate limits
3. **Turn Execution**: For each conversation:
   - Execute turn 1 (first turn, no delay), wait for response
   - Append assistant's response to conversation history
   - Apply turn delay (simulating user reading/thinking time)
   - Execute turn 2 with accumulated history, wait for response
   - Apply turn delay
   - Repeat for all remaining turns in the conversation
4. **Metrics Collection**: Metrics are collected per-turn and aggregated across all conversations

## Quick Reference

**Conversation Control:**
- `--conversation-num <N>` — Number of conversation sessions (for multi-turn)
- `--request-count <N>` — Number of requests (for single-turn)
- `--num-dataset-entries <N>` — Number of unique prompts to generate

**Turn Configuration:**
- `--conversation-turn-mean <N>` — Average turns per conversation (default: 1)
- `--conversation-turn-stddev <N>` — Standard deviation of turns (default: 0)

**Turn Delays:**
- `--conversation-turn-delay-mean <MS>` — Average delay between turns in ms (default: 0)
- `--conversation-turn-delay-stddev <MS>` — Standard deviation of delays in ms (default: 0)

**Best Practices:**
- Start with lower concurrency when testing multi-turn (2-5) to understand baseline behavior
- Use turn delays to model realistic user interaction patterns
- Monitor context window growth in long conversations (turns × output tokens)
- Consider using `--request-rate` to control conversation start rate for more predictable load
- Use `--random-seed` for reproducible conversation patterns

