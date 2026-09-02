---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Interactive Chat Sanity Checks
---

# Interactive Chat Sanity Checks

`aiperf chat` lets you talk to an OpenAI-compatible endpoint one message at a
time and see per-turn speed stats — time to first token, throughput,
inter-token latency, and prompt-cache hit rate — after every reply. It is a
lightweight sanity-check companion to `aiperf profile`: handy for confirming an
endpoint is wired up correctly and getting a ballpark feel for performance
before committing to a full benchmark run, and for eyeballing
speculative-decoding or reasoning-model behavior where speedups are
prompt-dependent.

It is deliberately minimal. For real benchmarking — concurrency, request rates,
datasets, sampling knobs, artifacts, and statistical aggregation — use
`aiperf profile`. Every number `aiperf chat` prints is produced by the same
metric classes `profile` uses, so they stay consistent (including
reasoning-token accounting).

---

## Prerequisites

Before you begin, ensure you have:

- AIPerf installed
- An OpenAI-compatible inference server running (e.g. vLLM, SGLang) and
  reachable, exposing `/v1/chat/completions`

---

## Single-Shot Sanity Check

Send one message and exit with `--quick`:

```bash
aiperf chat \
    --model Qwen/Qwen3-0.6B \
    --url http://localhost:8000 \
    --quick "say hello in one short sentence"
```

**Sample Output:**

```
Using model: Qwen/Qwen3-0.6B
Hello! How can I help you today?
TTFT: 20.06 ms
TPS:  154.11 tokens/s (9 tokens in 0.06s)
ITL:  6.31 ms/token (decode 158.40 tokens/s)
Cache: 14/18 prompt tokens cached (77.8%)
```

Every reply is followed by the same per-turn stats block, explained next.

---

## Reading the per-turn stats

The block is the same in every mode (single-shot, multi-turn, or
`--no-history`). All four values come from the same metric classes `aiperf
profile` uses:

- **TTFT** — time to first token (client-observed).
- **TPS** — generated tokens divided by end-to-end latency. The vLLM-familiar
  rate; it *includes* TTFT, so it blends prefill and decode into one number.
- **ITL** (inter-token latency) / **decode** — a decode-focused rate that
  excludes prefill. Because it ignores TTFT, it stays comparable across turns
  even when a growing history inflates prefill (where TPS sags). Omitted for
  single-token replies, where it is undefined.
- **Cache** — prompt-cache hit rate (`cached / prompt tokens`). Prefix caches
  are **server-side and shared across requests and sessions**, so a hit does
  not require a prior turn in this conversation — a shared system prompt or any
  prefix the server already cached (from earlier requests, other sessions, or
  other clients) can hit, even on a first or single-shot message. It is read
  from the server's `usage`, so it appears only when the server reports prompt
  caching (see below), and reflects the same counts `aiperf profile` records.

> **Don't see the `Cache:` line?** The server has to *report* cached tokens in
> its `usage`, which is separate from merely *enabling* caching. Launch the
> server accordingly:
>
> - **vLLM:** `--enable-prefix-caching` **and** `--enable-prompt-tokens-details`.
>   The reporting flag is off by default and independent of prefix caching —
>   without it vLLM sends `prompt_tokens_details: null` and no `Cache:` line
>   appears.
> - **SGLang:** `--enable-cache-report` (off by default).
> - **TRT-LLM:** reports `cached_tokens` by default; no extra flag.
>
> A genuinely novel prefix is a cold miss (`Cache: 0/N … (0.0%)`); the rate
> rises as prefixes are reused — within this session or already resident in the
> server's cache. See
> [Vendor Usage Field Reference](../reference/vendor-usage-fields.md) for the
> full matrix.

---

## Interactive Multi-Turn Chat

Omit `--quick` to start an interactive session. History is retained and resent
each turn, so the model has full context (press `Ctrl-D` to exit). Watching the
same stats evolve turn over turn is the point: TTFT and the cache hit rate climb
as the resent prefix grows, while ITL — the decode-only rate — stays roughly
flat even as the headline TPS sags.

```bash
aiperf chat --model Qwen/Qwen3-0.6B --url http://localhost:8000
```

```
Please enter a message for the chat model (Ctrl-D to exit):
> what's the capital of France?
The capital of France is Paris.
TTFT: 21.483 ms
TPS:  148.20 tokens/s (8 tokens in 0.05s)
ITL:  6.05 ms/token (decode 165.29 tokens/s)
Cache: 12/40 prompt tokens cached (30.0%)
> and its population?
As of recent estimates, the population of Paris is about 2.1 million.
TTFT: 34.92 ms
TPS:  121.66 tokens/s (15 tokens in 0.12s)
ITL:  6.11 ms/token (decode 163.67 tokens/s)
Cache: 56/72 prompt tokens cached (77.8%)
```

---

## Stateless Turns (`--no-history`)

By default each turn resends the full conversation. Pass `--no-history` to make
every message an independent, stateless request instead — the completion-style
flow, where the model has no memory of prior turns:

```bash
aiperf chat --model Qwen/Qwen3-0.6B --no-history
```

This is useful for measuring single-turn performance repeatedly without the
prefill cost of a growing history. The same stats block still prints; the cache
hit rate won't climb from *this* session's history (none is resent), but it can
still be non-zero when the server already holds a matching prefix — e.g. a
shared system prompt or a prefix cached from earlier requests. `--no-history` is
ignored with `--quick`, which is already a single request.

---

## Reasoning Models

When the endpoint serves a reasoning model, `aiperf chat` counts reasoning
tokens the same way `aiperf profile` does. Whether the server emits reasoning in
a dedicated `reasoning_content` field (via a reasoning parser) or inline as
`<think>...</think>` in the content, the generated reasoning tokens are included
in the token total, and a separate reasoning count is shown in the TPS line:

```
TTFT: 20.06 ms
TPS:  154.11 tokens/s (186 tokens, 142 reasoning in 1.20s)
ITL:  6.12 ms/token (decode 163.40 tokens/s)
```

---

## Options

| Option | Description |
|---|---|
| `--model`, `-m` | Model name served by the endpoint (required). |
| `--url`, `-u` | Base URL of the OpenAI-compatible server (default `http://localhost:8000`). |
| `--system-prompt` | Optional system prompt prepended to the conversation. |
| `--quick`, `-q` | Send a single message and exit instead of looping. |
| `--no-history` | Treat each interactive message as a stateless request (no history retained). Ignored with `--quick`. |
| `--api-key` | API key sent as a Bearer token (defaults to `OPENAI_API_KEY`). |
| `--tokenizer` | Tokenizer for client-side token counts (defaults to the model name). Pass `builtin` for a zero-network tokenizer. |

---

## Tuning request timeouts

`aiperf chat` fails fast on an unreachable endpoint but never caps overall
generation time, so long replies are not truncated. Both timeouts are tunable
via environment variables (no total timeout is applied):

- `AIPERF_CHAT_CONNECT_TIMEOUT` (default `10.0`) — seconds to establish a
  connection before a turn fails.
- `AIPERF_CHAT_READ_TIMEOUT` (default `300.0`) — seconds to wait for the next
  streamed chunk before a turn fails (only fires if the server stalls).

See [Environment Variables](../environment-variables.md) for the full list.

## See Also

- [Multi-Turn Conversations](multi-turn.md) - Benchmark multi-turn sessions at scale
- [Using Local Tokenizers Without HuggingFace](local-tokenizer.md)
- [Metrics Reference](../metrics-reference.md) - Definitions of TTFT, ITL, OSL, and more
