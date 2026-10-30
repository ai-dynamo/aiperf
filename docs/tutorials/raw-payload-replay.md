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
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```

Since auto-detection recognizes files with `messages` arrays, you can omit `--custom-dataset-type`:

```bash
aiperf profile \
    --input-file payloads.jsonl \
    --model Qwen/Qwen3-0.6B \
    --streaming \
    --url localhost:8000 \
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
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```

---

## Endpoint Type

Raw payloads work with any endpoint type. The endpoint controls only **response parsing** and **URL path** -- payload formatting is always bypassed when raw payloads are present.

Using a regular endpoint type (e.g., the default `chat`) is recommended because it provides structured response parsing (token counts, finish reasons, choices) instead of generic auto-detection.

For non-standard APIs where no built-in endpoint matches, use `--endpoint-type raw`. The raw endpoint does not append a URL path (you must include the full path in `--url`) and parses responses using auto-detection. For non-standard response formats, you can specify a [JMESPath](https://jmespath.org/) expression via `--extra-inputs response_field:<expression>` to extract the relevant field.

---

## Configuration Reference

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input-file` | Yes | -- | Path to a JSONL file or directory of JSONL files |
| `--model` | Yes | -- | Model name (e.g., `Qwen/Qwen3-0.6B`) |
| `--endpoint-type` | No | `chat` | Any endpoint type works; `raw` available for non-standard APIs |
| `--custom-dataset-type` | No | Auto-detected | Set to `raw_payload` to force this loader |
| `--streaming` | No | `false` | Enable streaming responses |
| `--url` | No | `localhost:8000` | Server base URL (endpoint type appends the API path) |
| `--concurrency` | No | -- | Number of concurrent users |
| `--dataset-sampling-strategy` | No | `sequential` | `sequential`, `random`, or `shuffle` |

---

## Context Mode

Raw payload conversations use `message_array_with_responses` [context mode](../reference/conversation-context-mode.md) by default. Each turn is sent exactly as written -- AIPerf does not accumulate prior turns or inject server responses into subsequent requests.

This is the correct behavior because raw payloads already contain the complete message history for each turn. In directory mode, each line in a session file should include all prior context needed for that point in the conversation (see the multi-turn examples above).

To retain live model responses (including tool calls) between structured-message turns, use `--custom-dataset-type mooncake_trace` with `"assistant_responses": "live"` on every row in the session. The first row carries the initial history, which may include earlier recorded assistant messages; later rows carry only new messages. See [Live Responses for Structured Messages](../benchmark-modes/trace-replay.md#live-responses-for-structured-messages).

---

## Structured decision APIs

Use `--endpoint-type decisions` for OpenAI-schema `POST /v1/decisions` or `--endpoint-type systemone` for Jev `POST /v1/systemone`. Both adapters are text-only and non-streaming, preserve structured responses rather than converting JSON into generated text, and reject `nvext` and SGLang-native Decisions formats. Engine selection is independent: qualify each SGLang or vLLM serving profile separately.

The `raw_payload` JSONL loader requires a chat `messages` array. For decisions, use the existing `inputs_json` loader instead: it accepts complete payloads without inventing chat fields. Save this example as `decisions.json`:

```json
{"data":[{"session_id":"department","payloads":[{"model":"qualified-model","input":"Please correct my invoice.","questions":[{"type":"choice","name":"department","instructions":"Which department should handle this?","choices":[{"value":"billing"},{"value":"support"}]}]}]}]}
```

```bash
aiperf profile --model qualified-model --endpoint-type decisions \
    --url http://localhost:8000 --custom-dataset-type inputs_json \
    --input-file decisions.json --request-count 20 --concurrency 1 \
    --no-gpu-telemetry --export-level raw --artifact-dir artifacts/decisions
```

For System One, save `systemone.json` and use `--endpoint-type systemone --input-file systemone.json` with a different artifact directory:

```json
{"data":[{"session_id":"department","payloads":[{"model":"qualified-model","state":"Please correct my invoice.","questions":{"department":{"type":"choice","criteria":{"billing":"Billing requests","support":"Technical support"}}}}]}]}
```

The configured model must be registered by the server, and each payload must contain its actual model alias. The adapters do not merge generation settings, replace the model, coerce choice values, or rewrite questions. OpenAI string and boolean choices remain distinct. Answers are checked against question count, order, names and typed candidate identities. Malformed HTTP-200 responses become record errors; infrastructure errors and genuine OpenAI refusals remain distinct. Confidence is checked as a finite probability, not against an invented universal calibration formula. Missing usage remains unknown, and actual nonzero output-token usage is retained when a backend performs an internal decode.

These adapters provide structural validation and per-record question/refusal metadata, not complete official SDK/schema conformance or an aggregate question-throughput report. Request latency and HTTP throughput are not question throughput. Generated-text TTFT, inter-token latency and output-token-throughput metrics are not applicable. Raw exports retain the wire evidence for auditing. Do not force `AIPERF_RECORD_STRIP_PAYLOAD_BYTES=true`: response reconciliation needs the original payload, and stripping it causes validation failure. Model quality, numerical parity, physical GPU cancellation and performance qualification need separate campaigns. See [enhancement #1536](https://github.com/ai-dynamo/aiperf/issues/1536) and the [Dynamo benchmark workstream](https://github.com/ai-dynamo/dynamo/pull/15886) for remaining scope.

## Tips

- **Include the full API path in `--url`** only when using `--endpoint-type raw`. Other endpoint types append the path automatically.
- **Every line must have a `messages` key** with a list value.
- **Empty lines are skipped** in both modes.
- **Directory files are sorted alphabetically**. Name files with zero-padded numbers (e.g., `session_001.jsonl`) for predictable ordering.
- **Non-`.jsonl` files are ignored** in directory mode.
- **Payloads are sent verbatim** -- AIPerf does not modify, validate, or reformat them.
- **Default sampling is `sequential`**. Use `--dataset-sampling-strategy shuffle` or `random` for varied ordering.
