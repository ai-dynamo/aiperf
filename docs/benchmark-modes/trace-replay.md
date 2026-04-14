---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Trace Replay with Mooncake Traces
---

# Trace Replay with Mooncake Traces

This tutorial covers replaying production traces using the Mooncake trace format. Trace replay benchmarking reproduces real-world traffic patterns with precise timing control, enabling performance validation and capacity planning under realistic load.

## When to Use This Tutorial

Use this approach when you need to:
- Replay production traffic patterns captured from real systems
- Validate performance with industry-standard Mooncake FAST'25 traces
- Test system behavior under specific temporal load patterns
- Reproduce benchmark results for regression testing

For other use cases:
- **Custom prompts without timing**: See [Custom Prompt Benchmarking](../tutorials/custom-prompt-benchmarking.md)
- **Precise timestamp control for any dataset**: See [Fixed Schedule](../tutorials/fixed-schedule.md)
- **Multi-turn conversations from files**: See [Multi-Turn Conversations](../tutorials/multi-turn.md)

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

## Mooncake Trace Format

Mooncake provides a specification and sample datasets for [trace replay](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) that can be replayed for performance benchmarking.

Mooncake traces use a JSONL file where each line represents a request with timing information.

Each trace entry requires exactly one input mode:
- `input_length`: Number of input tokens (synthetic prompt generated from token count)
- `text_input`: Literal text string sent as the prompt
- `messages`: List of OpenAI-compatible message dicts sent directly to the API
- `payload`: Complete API request dict sent verbatim (bypasses all endpoint formatting)

Optional fields:
- `timestamp`: Request arrival time in milliseconds
- `delay`: Milliseconds to wait before sending (alternative to `timestamp`)
- `output_length`: Number of output tokens
- `hash_ids`: List of block hashes (only with `input_length`)
- `tools`: List of OpenAI-compatible tool definitions (only with `messages`)
- `session_id`: Unique identifier for multi-turn conversation grouping

Example entry:

```json
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [0, 1, 2]}
```

## Profile using a Custom Trace File

Create a trace file with timing information:

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
cat > custom_trace.jsonl << 'EOF'
{"timestamp": 0, "input_length": 1200, "output_length": 52, "hash_ids": [0, 1, 2]}
{"timestamp": 105, "input_length": 1800, "output_length": 26, "hash_ids": [0, 3, 4, 5]}
{"timestamp": 274, "input_length": 1300, "output_length": 52, "hash_ids": [1, 4, 6]}
EOF
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}
Run AIPerf with the trace file:

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file custom_trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

The `--fixed-schedule` flag tells AIPerf to send requests at the exact timestamps specified in the trace. This reproduces the original timing pattern.

## Using Pre-formatted Messages

Instead of synthetic prompts generated from `input_length` and `hash_ids`, you can provide an OpenAI-compatible `messages` array directly per trace entry. This is useful for replaying captured conversations (e.g., coding agent sessions) with exact prompt content.

Each entry's `messages` field contains the full conversation history up to that point. In multi-turn sessions, later entries include prior turns so the server receives the complete context:

```json
{"session_id": "sess-1", "messages": [{"role": "user", "content": "Hello"}], "output_length": 50, "timestamp": 0}
{"session_id": "sess-1", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How are you?"}], "output_length": 30, "timestamp": 2000}
```

The `messages` field is mutually exclusive with `input_length` and `text_input`. When set, the messages array is sent directly to the API payload, bypassing prompt synthesis entirely. The model's actual response is not carried forward between turns -- each turn uses its pre-defined messages.

### Tool Definitions

When replaying conversations that involve tool use (function calling), include the `tools` field alongside `messages` to provide the tool definitions the model needs:

```json
{"messages": [{"role": "user", "content": "What's the weather?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}], "output_length": 50, "timestamp": 0}
```

The `tools` field is only valid when `messages` is provided. It is injected directly into the API payload as the `tools` parameter.

## Using Raw Payloads (Verbatim Replay)

For the most precise replay, you can provide complete API request payloads that are sent verbatim to the server with zero formatting. This bypasses all endpoint payload construction, giving you full control over every field in the request body while still using Mooncake's timestamp/delay scheduling.

Each entry's `payload` field contains the exact JSON body to send:

```json
{"payload": {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4", "stream": true, "max_tokens": 100}, "timestamp": 0}
{"payload": {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How?"}], "model": "gpt-4", "stream": true}, "timestamp": 2000}
```

The `payload` field is mutually exclusive with `input_length`, `text_input`, and `messages`. When set, the payload dict is sent directly to the transport without any endpoint formatting. Any endpoint type can be used -- the endpoint controls response parsing and URL path, while payload formatting is bypassed automatically:

```bash
aiperf profile \
    --url localhost:8000 \
    --input-file payloads.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```

Multi-turn sessions work with `session_id` and `delay`:

```json
{"session_id": "s1", "payload": {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}, "timestamp": 0}
{"session_id": "s1", "payload": {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "Continue"}], "model": "gpt-4"}, "delay": 500}
```

## Profile using real Mooncake Trace

For real-world benchmarking, use the FAST25 production trace data from the Mooncake research paper:

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Download the Mooncake trace data
curl -Lo mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# Create a subset for quick testing
head -n 10 mooncake_trace.jsonl > mooncake_trace_short.jsonl

# Run the trace replay
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file mooncake_trace_short.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}