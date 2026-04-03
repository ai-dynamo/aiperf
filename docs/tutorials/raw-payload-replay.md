---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Raw Payload Replay
---

# Raw Payload Replay

Benchmark LLM servers by replaying pre-built API request bodies verbatim.

## Overview

The `raw_payload` dataset type replays complete API request bodies exactly as written in your JSONL files. Unlike other dataset types where AIPerf constructs the request payload from structured fields, raw payload replay sends each JSON object directly to the server with no transformation.

This is useful when you:

- **Have captured production traffic** and want to replay it exactly
- **Need full control** over every field in the request body (model, temperature, tools, system prompts, etc.)
- **Are testing non-standard APIs** where AIPerf's built-in endpoint formatters do not apply
- **Want to benchmark with pre-built payloads** exported from another tool or logging pipeline

| Property | Value |
|----------|-------|
| Default sampling | Sequential |
| Multi-turn support | Yes (directory mode) |
| Context mode | `message_array_with_responses` |
| Timing control | No |

---

## Input Modes

The loader supports two input modes, selected automatically based on whether `--input-file` points to a file or a directory.

### Single File Mode

Each line in the JSONL file is a complete API request payload. Each line becomes a separate single-turn conversation.

```
payloads.jsonl
  line 1 -> conversation 1 (single turn)
  line 2 -> conversation 2 (single turn)
  line 3 -> conversation 3 (single turn)
```

### Directory Mode

Each `.jsonl` file in the directory is one multi-turn conversation. Lines within a file are ordered turns. Files are processed in sorted alphabetical order.

```
payloads/
  session_001.jsonl -> conversation 1 (lines = turns)
  session_002.jsonl -> conversation 2 (lines = turns)
  session_003.jsonl -> conversation 3 (lines = turns)
```

---

## File Format

Each line must be a valid JSON object containing at minimum a `messages` key with a list value. Any additional fields (model, temperature, max_tokens, tools, stream, etc.) are preserved and sent verbatim.

### Single-Turn Example

```jsonl
{"messages": [{"role": "user", "content": "What is machine learning?"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 100}
{"messages": [{"role": "user", "content": "Explain neural networks."}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 200}
{"messages": [{"role": "user", "content": "How does backpropagation work?"}], "model": "Qwen/Qwen3-0.6B", "temperature": 0.7}
```

### Multi-Turn Example (Directory Mode)

Each file represents a conversation. Each line carries the full message history for that point in the conversation:

**`session_001.jsonl`:**
```jsonl
{"messages": [{"role": "user", "content": "Hello"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 100}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}, {"role": "user", "content": "How are you?"}], "model": "Qwen/Qwen3-0.6B", "temperature": 0.7}
```

### Auto-Detection

When `--custom-dataset-type` is not specified, AIPerf auto-detects raw payload format by checking the first non-empty line for a `messages` key with a list value. In directory mode, it checks the first `.jsonl` file found.

Auto-detection rejects records that contain a `conversation_id` key or a `data` key with a list value (to avoid conflicts with other dataset formats). If your payloads include these keys, use `--custom-dataset-type raw_payload` explicitly.

---

## Basic Usage

### Single File

```bash
cat > payloads.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "What is machine learning?"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 100}
{"messages": [{"role": "user", "content": "Explain neural networks."}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 200}
{"messages": [{"role": "user", "content": "How does backpropagation work?"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 150}
EOF

aiperf profile \
    --input-file payloads.jsonl \
    --model Qwen/Qwen3-0.6B \
    --custom-dataset-type raw_payload \
    --endpoint-type raw \
    --streaming \
    --url localhost:8000/v1/chat/completions \
    --concurrency 2
```

Since auto-detection recognizes files with `messages` arrays, you can omit `--custom-dataset-type`:

```bash
aiperf profile \
    --input-file payloads.jsonl \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type raw \
    --streaming \
    --url localhost:8000/v1/chat/completions \
    --concurrency 2
```

### Directory for Multi-Turn Conversations

```bash
mkdir -p conversations/

cat > conversations/session_001.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "What is Python?"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 200}
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language."}, {"role": "user", "content": "Show me a hello world example."}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 200}
EOF

cat > conversations/session_002.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "Explain REST APIs."}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 300}
{"messages": [{"role": "user", "content": "Explain REST APIs."}, {"role": "assistant", "content": "REST is an architectural style..."}, {"role": "user", "content": "What about GraphQL?"}], "model": "Qwen/Qwen3-0.6B", "max_tokens": 300}
EOF

aiperf profile \
    --input-file conversations/ \
    --model Qwen/Qwen3-0.6B \
    --custom-dataset-type raw_payload \
    --endpoint-type raw \
    --streaming \
    --url localhost:8000/v1/chat/completions \
    --concurrency 2
```

---

## The Raw Endpoint

The `raw` endpoint type (`--endpoint-type raw`) is designed for verbatim replay. It differs from other endpoint types in two ways:

1. **No payload formatting**: Payloads from the dataset are sent directly to the server without modification.
2. **No endpoint path**: The URL you provide via `--url` is used as-is. You must include the full API path (e.g., `--url localhost:8000/v1/chat/completions`).

Response parsing uses auto-detection (works with OpenAI-compatible and other common formats). For non-standard response formats, you can specify a [JMESPath](https://jmespath.org/) expression via `--extra-inputs response_field:<expression>` to extract the relevant field.

---

## Configuration Reference

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input-file` | Yes | -- | Path to a JSONL file or directory of JSONL files |
| `--model` | Yes | -- | Model name (e.g., `Qwen/Qwen3-0.6B`) |
| `--endpoint-type` | No | `chat` | Set to `raw` for verbatim replay (no formatting, no path appended) |
| `--custom-dataset-type` | No | Auto-detected | Set to `raw_payload` to force this loader |
| `--streaming` | No | `false` | Enable streaming responses |
| `--url` | No | `localhost:8000` | Server URL including the full API path |
| `--concurrency` | No | -- | Number of concurrent users |
| `--dataset-sampling-strategy` | No | `sequential` | `sequential`, `random`, or `shuffle` |

---

## Context Mode

Raw payload conversations use `message_array_with_responses` [context mode](../reference/conversation-context-mode.md) by default. Each turn is sent exactly as written -- AIPerf does not accumulate prior turns or inject server responses into subsequent requests.

This is the correct behavior because raw payloads already contain the complete message history for each turn. In directory mode, each line in a session file should include all prior context needed for that point in the conversation (see the multi-turn examples above).

---

## Tips

- **Include the full API path in `--url`** when using `--endpoint-type raw`. No path is appended automatically.
- **Every line must have a `messages` key** with a list value.
- **Empty lines are skipped** in both modes.
- **Directory files are sorted alphabetically**. Name files with zero-padded numbers (e.g., `session_001.jsonl`) for predictable ordering.
- **Non-`.jsonl` files are ignored** in directory mode.
- **Payloads are sent verbatim** -- AIPerf does not modify, validate, or reformat them.
- **Default sampling is `sequential`**. Use `--dataset-sampling-strategy shuffle` or `random` for varied ordering.
