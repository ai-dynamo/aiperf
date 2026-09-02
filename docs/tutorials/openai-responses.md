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
    --model Qwen/Qwen2.5-Omni-3B \
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

## Multi-Turn Conversations & Stateful Chaining

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

### Stateful Chaining with `previous_response_id`

Stateful chaining is **opt-in and driven by requesting storage**. Enable it with:

```bash
--extra-inputs '{"store": true}'
```

`store` is a *request* parameter, not a standard field of the Responses object,
so AIPerf keys chaining off the storage you requested rather than off an echoed
response field — the OpenAI spec does not include `store` on the response, and
servers such as vLLM's agentic-api accept it on the request but never serialize
it back. (If a server *does* echo `store: true` on the response object, that is
honored too.)

Some backends also require a server-side flag (e.g. vLLM with
`VLLM_ENABLE_RESPONSES_API_STORE=1`) to actually persist responses. If the
server does not persist a requested response, the next chained request fails
against the missing `previous_response_id`; use a storing backend when enabling
this feature.

> **Startup requirement:** requesting `store: true` on `--endpoint-type responses`
> is rejected at startup unless [`--use-server-token-count`](#server-token-counts)
> is also set. Chaining sends only the newest turn on the wire, so client-side ISL
> would undercount the server-side prompt; server-reported token counts are the
> only accurate source. Add `--use-server-token-count`, or drop `store: true` to
> keep sending the full history client-side.

When chaining is active with `--endpoint-type responses`:
- On **Turn 0**, AIPerf sends the initial prompt and captures the server-generated `response.id` (e.g. `resp_<hash>`) from the response object — because `store: true` was requested for the run.
- On **Turn 1+**, AIPerf sets `previous_response_id: <resp_id>` and sends only the single newest turn in the `input` array rather than re-sending the entire accumulated conversation history.

**Scope:** chaining is applied only in the default delta context mode
(`deltas_without_responses`), where the newest turn is a genuine delta. The
`*_with_responses` context modes carry full per-turn history and are left
unchained to avoid sending the conversation twice.

Chaining is also limited to the session-driven request path. Pre-encoded
datasets (`--input-file` payloads sent verbatim) bypass session tracking, so
their requests are sent exactly as authored and are never chained — author
`previous_response_id` into those payloads yourself if you need it.

The startup requirement above inspects the endpoint-level `--extra-inputs`. If
`store: true` is instead supplied per turn (via a dataset row's `extra`),
it cannot be seen at startup, but chaining still triggers the one-time runtime
warning that client-side ISL undercounts the server-side prompt — enable
`--use-server-token-count` in that case too.

> **Input Sequence Length note:** because a chained turn only puts the newest turn
> on the wire (the prior history lives server-side), the default client-side ISL
> reflects just that turn and undercounts the prompt the server actually prefills.
> Use [`--use-server-token-count`](#server-token-counts) for accurate multi-turn
> ISL when chaining is enabled. AIPerf emits a one-time warning if chaining runs
> without it.

See the [Multi-Turn Conversations](multi-turn.md) tutorial for details on conversation control parameters.

---

## WebSocket Mode

The Responses API also supports [WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode):
instead of one HTTP request per turn, AIPerf keeps a persistent socket open and
sends each turn as a `response.create` event, reading back the same `response.*`
lifecycle events the HTTP SSE path streams. Select it by giving the endpoint a
`ws://` or `wss://` URL — the WebSocket transport is auto-detected from the
scheme:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --streaming \
    --use-server-token-count \
    --url ws://localhost:8000 \
    --conversation-num 10 \
    --conversation-turn-mean 3 \
    --concurrency 4
```

Key behaviors:

- **One dedicated socket per conversation.** Each conversation holds a socket for
  its whole lifetime (a hard lease keyed on the conversation, like the HTTP
  sticky-user-sessions strategy), opened on the first turn and closed on the
  final turn. Every turn of a conversation therefore reuses the same connection,
  so `previous_response_id` chaining stays inside the server's connection-local
  cache even when turns from other conversations interleave.
- **Open sockets track concurrently active conversations, not in-flight requests.**
  `--concurrency` caps concurrent *sessions* (a session slot is held from a
  conversation's first turn through its final turn), so at saturation the socket
  count equals `--concurrency`. Under request-rate load, where session
  concurrency is unbounded, the socket count instead tracks how many
  conversations are active at once — the same way the HTTP sticky strategy
  behaves.
- **Chaining is automatic and does not require `store: true`.** The WebSocket
  contract resolves `previous_response_id` from the connection's response cache,
  so multi-turn conversations chain in the default `deltas_without_responses`
  context mode without requesting server-side storage. As with HTTP chaining,
  only the newest turn is put on the wire, so pair WebSocket runs with
  [`--use-server-token-count`](#server-token-counts) for accurate multi-turn ISL.
- **A mid-conversation reconnect ends a chained turn.** Because non-stored
  chaining lives only in the connection-local cache, if the peer drops a
  conversation's socket between turns the cached `previous_response_id` is no
  longer resolvable on the fresh socket. AIPerf fails that turn explicitly
  (`ChainingContextLost`) rather than emitting a confusing
  `previous_response_not_found` from the server. Add `--extra-inputs
  '{"store": true}'` to persist responses server-side so a reconnect still
  resolves the id.
- **HTTP-only fields are stripped.** `stream`, `stream_options`, and `background`
  are HTTP-transport concepts; the socket always streams events, so AIPerf drops
  them from the `response.create` envelope automatically.
- **Requests carry a `stream_id`** derived from the conversation so forked
  conversations replaying against the same server remain addressable.
- **WebSocket mode requires `--endpoint-type responses`.** The transport only
  speaks the Responses API contract; a `ws://`/`wss://` URL (or `--transport
  websocket`) with any other endpoint type is rejected at config validation.
- **Credentials require `wss://`.** An API key or authentication header sent over
  unencrypted `ws://` would travel in cleartext, so AIPerf rejects that
  combination — use `wss://` for credential-bearing WebSocket runs.

The endpoint layer is transport-agnostic: metrics, parsing, and multi-turn
control behave identically to the HTTP path. You can therefore compare stateless
HTTP, stateful HTTP (`store: true` chaining), and WebSocket runs against the same
server by only changing the URL scheme and the chaining flags.

> **Note:** WebSocket mode is specific to the Responses endpoint. Other endpoint
> types continue to use the HTTP transport.

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

## Verifying ISL/OSL Distribution with the Mock Server

Use the mock server's `--record-requests` flag to capture the exact token lengths
AIPerf sends on the wire before running against a real server:

```bash
aiperf-mock-server --record-requests /tmp/responses-req.jsonl --fast &
mock_server_pid=$!

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type responses \
    --endpoint /v1/responses \
    --synthetic-input-tokens-mean 512 \
    --output-tokens-mean 256 \
    --url http://localhost:8000 \
    --request-count 100

kill $mock_server_pid   # graceful shutdown flushes the JSONL and prints a summary
```

The recorder writes one JSON line per request. For `/v1/responses` requests the
`request_id` carries a `resp-` prefix and `max_output_tokens` is canonicalized
into the `max_completion_tokens` column so the schema stays uniform with chat
and completions rows:

```json
{"ts": 1714000000.123, "request_id": "resp-3b8568cb-...", "endpoint": "/v1/responses",
 "model": "Qwen/Qwen3-0.6B", "isl": 512,
 "requested_osl": 256, "max_tokens": null, "max_completion_tokens": 256,
 "min_tokens": null, "ignore_eos": false, "reasoning_effort": null,
 "stream": false, "tokenization_mode": "tokenizer_call"}
```

See the [mock server README](https://github.com/ai-dynamo/aiperf/blob/main/tests/aiperf_mock_server/README.md#request-recording)
for the full output format and summary schema.

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
| `response.created` | Server-generated response ID (`resp_<hash>`) for stateful session chaining |
| `response.output_text.delta` | Text content delta |
| `response.reasoning_text.delta` | Reasoning content delta |
| `response.function_call_arguments.delta` | Tool call arguments delta |
| `response.output_text.done` | Final text (fallback for providers that emit text only in done events) |
| `response.completed` | Usage statistics and response ID |
| All other events | Skipped |

This enables accurate measurement of TTFT, ITL, and token throughput metrics when streaming is enabled.

