---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Anonymize Trace
---
# Privacy-Preserving Trace Anonymization

Share realistic LLM workload traces without exposing sensitive prompt content.

## Overview

Production LLM traces are valuable for benchmarking because they capture real-world
patterns: input/output length distributions, request timing, and prefix sharing from
repeated system prompts or multi-turn conversations. However, sharing these traces
directly would expose user data, proprietary prompts, and PII.

`aiperf anonymize-trace` solves this by converting raw chat logs into Mooncake traces
where all text is replaced with block hash IDs. The hash sequences preserve prefix
overlap patterns (enabling KV cache-aware benchmarking) while making it impossible
to recover the original text.

## Preparing Your Input

Create a JSONL file where each line is a conversation record with OpenAI-compatible messages:

### Single-Turn Example

```jsonl
{"timestamp": 0, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "output": "The capital of France is Paris."}
{"timestamp": 100, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain machine learning in simple terms."}], "output": "Machine learning is a type of AI that allows computers to learn from data."}
```

Note how both requests share the same system prompt. The anonymized trace will reflect this shared prefix through matching hash IDs.

### Multi-Turn Example

Use `session_id` to group turns within a conversation:

```jsonl
{"timestamp": 0, "session_id": "user_42", "messages": [{"role": "user", "content": "What is Python?"}], "output": "Python is a programming language."}
{"timestamp": 5000, "session_id": "user_42", "messages": [{"role": "user", "content": "Show me a hello world example"}], "output": "print('Hello, World!')"}
```

Each turn only needs its own new messages. The anonymizer automatically accumulates the full conversation history (including prior assistant responses) when computing hash IDs for later turns.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | OpenAI-compatible messages with `role` and `content` |
| `output` | string | Assistant response text (used only for token counting) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | number | Milliseconds since trace start (for `--fixed-schedule` replay) |
| `session_id` | string | Groups turns into multi-turn conversations |

## Choosing Your Target Model

The `--model` argument specifies the model you intend to **benchmark against**, not the model that generated the original logs.

This matters because:

- **Chat template**: Different models use different chat formats (ChatML, Llama, Mistral, etc.). The template tokens are part of what gets cached, so prefix patterns depend on which template is applied.
- **Tokenization**: Token counts and block boundaries vary by tokenizer. A trace anonymized for Llama will have different `input_length` values than one for Mistral.

### Example: Migrating from a Proprietary API

If you have production logs from Claude or GPT-4 and want to evaluate switching to a self-hosted model:

```bash
# Anonymize for benchmarking against Llama 3.1 70B
aiperf anonymize-trace \
  --input-file production_logs.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --output-file llama_trace.jsonl

# Anonymize for benchmarking against Mistral
aiperf anonymize-trace \
  --input-file production_logs.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-file mistral_trace.jsonl
```

## Running the Command

```bash
aiperf anonymize-trace \
  --input-file raw_logs.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --output-file anonymized_trace.jsonl \
  --block-size 512
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-file` | Yes | | Path to input JSONL |
| `--model` | Yes | | HuggingFace model name (target model) |
| `--output-file` | No | `<input>_anonymized.jsonl` | Output path |
| `--block-size` | No | 512 | Tokens per hash block |

The default `--block-size` of 512 matches common KV cache page sizes. Smaller values increase hash granularity but produce larger `hash_ids` arrays.

## Verifying the Output

Inspect the anonymized trace with `aiperf analyze-trace`:

```bash
aiperf analyze-trace --input-file anonymized_trace.jsonl --block-size 512
```

This shows ISL/OSL distributions, prefix reuse ratios, and theoretical cache hit rates,
letting you verify the trace captures meaningful prefix sharing patterns.

## Replaying the Trace

Use the anonymized trace as a benchmark workload:

```bash
# With timestamps (fixed schedule replay)
aiperf profile \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file anonymized_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule

# Without timestamps (use request rate instead)
aiperf profile \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file anonymized_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --request-rate 10 \
  --concurrency 50
```

## What Gets Shared vs. What Stays Protected

| Shared | Protected |
|--------|-----------|
| Request timestamps | Actual prompt text |
| Input/output token counts | Token IDs |
| Block hash ID sequences | Assistant responses |
| Prefix cache hit patterns | User information |
| Session grouping | Proprietary system prompts |
