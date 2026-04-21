---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile the OpenAI Responses API with AIPerf
---

# Profile the OpenAI Responses API with AIPerf

This guide covers benchmarking servers that implement the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) (`POST /v1/responses`) using AIPerf.

The Responses API is OpenAI's newer API primitive that replaces Chat Completions for new projects. It supports text, images, audio, streaming, and reasoning output.

---

## Overview

AIPerf's `responses` endpoint type handles the key differences between the Responses API and Chat Completions:

| Chat Completions | Responses API |
|---|---|
| `messages` array | `input` array |
| `system` role message | Top-level `instructions` field |
| `max_completion_tokens` | `max_output_tokens` |
| `{"type": "text", ...}` content | `{"type": "input_text", ...}` content |
| `{"type": "image_url", ...}` content | `{"type": "input_image", ...}` content |
| `choices[0].delta.content` (streaming) | `response.output_text.delta` event (streaming) |
| `choices[0].message.content` (non-streaming) | `output[].content[].text` (non-streaming) |

---

## Start a Server

Launch an OpenAI Responses API-compatible server. For example, using a vLLM server:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B --reasoning-parser qwen3
```

Verify the server is ready:

```bash
curl -s http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "input": [{"role": "user", "content": "Hello"}],
    "max_output_tokens": 10
  }' | jq
```

---

## Profile with Synthetic Inputs

Run AIPerf against the Responses API endpoint using synthetic inputs:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --url localhost:8000 \
    --request-count 20
```

**Sample Output:**

```text
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:35<00:00]

INFO     Benchmark completed successfully

            NVIDIA AIPerf | LLM Metrics
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1678.90 │ 1456.34 │ 1923.45 │ 1923.45 │ 1667.23 │
│    Time to First Token (ms) │  234.56 │  198.34 │  289.12 │  289.12 │  231.45 │
│    Inter Token Latency (ms) │   13.89 │   11.23 │   17.45 │   17.45 │   13.67 │
│ Output Token Count (tokens) │  200.00 │  200.00 │  200.00 │  200.00 │  200.00 │
│  Request Throughput (req/s) │    5.67 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

---

## Profile with Custom Input Files

Create a JSONL input file:

```bash
cat <<EOF > inputs.jsonl
{"texts": ["Explain quantum computing in simple terms."]}
{"texts": ["Write a haiku about machine learning."]}
EOF
```

Run AIPerf:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 10
```

---

## System Instructions

In the Responses API, system instructions use a top-level `instructions` field rather than a system role message. AIPerf handles this mapping automatically when you use `--shared-system-prompt-length` to generate a synthetic system prompt:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --shared-system-prompt-length 50 \
    --synthetic-input-tokens-mean 100 \
    --output-tokens-mean 200 \
    --url localhost:8000 \
    --request-count 20
```

This generates a synthetic system prompt of approximately 50 tokens and places it in the `"instructions"` field of the Responses API payload, rather than adding a system message to the input array. The same prompt is shared across all requests in the session.

---

## Vision (Image Inputs)

Profile vision-capable models with synthetic images:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --image-width-mean 512 \
    --image-height-mean 512 \
    --synthetic-input-tokens-mean 100 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

Image inputs are formatted as `{"type": "input_image", "image_url": "<url>"}` in the Responses API (compared to `{"type": "image_url", "image_url": {"url": "<url>"}}` in Chat Completions).

---

## Audio Inputs

Profile audio-capable models with the Responses API:

```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --url localhost:8000 \
    --request-count 20
```

Audio inputs are formatted as `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "<fmt>"}}`, the same structure used by Chat Completions.

See the [Audio](audio.md) tutorial for details on audio input configuration and supported formats.

---

## Non-Streaming Mode

Run without streaming to get full responses:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --synthetic-input-tokens-mean 100 \
    --output-tokens-mean 200 \
    --url localhost:8000 \
    --request-count 20
```

> [!NOTE]
> Without `--streaming`, time-to-first-token (TTFT) and inter-token latency (ITL) metrics are not available. Use streaming mode for the most detailed latency breakdown.

---

## Concurrency and Rate Control

Control load generation the same way as other endpoint types:

```bash
# Concurrency-based
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --concurrency 10 \
    --url localhost:8000 \
    --request-count 100

# Request-rate-based
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --request-rate 5 \
    --url localhost:8000 \
    --request-count 100
```

---

## Multi-Turn Conversations

Benchmark multi-turn conversations using the Responses API:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --conversation-num 10 \
    --conversation-turn-mean 3 \
    --synthetic-input-tokens-mean 100 \
    --output-tokens-mean 200 \
    --url localhost:8000
```

See the [Multi-Turn Conversations](multi-turn.md) tutorial for details on conversation control parameters.

---

## Server Token Counts

Use server-reported token counts instead of client-side tokenization:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --use-server-token-count \
    --url localhost:8000 \
    --request-count 20
```

When `--use-server-token-count` is enabled with streaming, AIPerf automatically sets `stream_options.include_usage` in the request payload to receive usage data in the `response.completed` event.

---

## Extra Parameters

Pass additional API parameters using `--extra-inputs`:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --extra-inputs temperature:0.7 \
    --extra-inputs top_p:0.9 \
    --url localhost:8000 \
    --request-count 20
```

---

## Key Differences from Chat Completions

When migrating AIPerf benchmarks from `--endpoint-type chat` to `--endpoint-type responses`:

1. Change `--endpoint-type chat` to `--endpoint-type responses`
2. Change `--endpoint /v1/chat/completions` to `--endpoint /v1/responses`
3. The `--use-legacy-max-tokens` flag is not applicable (the Responses API always uses `max_output_tokens`)
4. All other AIPerf flags (`--streaming`, `--concurrency`, `--extra-inputs`, etc.) work the same way

---

## Streaming Event Handling

For reference, AIPerf processes these Responses API streaming events:

| Event Type | Data Extracted |
|---|---|
| `response.output_text.delta` | Text content delta |
| `response.reasoning_text.delta` | Reasoning content delta |
| `response.output_text.done` | Final text (fallback for providers that emit text only in done events) |
| `response.completed` | Usage statistics |
| All other events | Skipped |

This enables accurate measurement of TTFT, ITL, and token throughput metrics when streaming is enabled.
