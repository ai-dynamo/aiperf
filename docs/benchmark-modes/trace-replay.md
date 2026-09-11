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

Required fields for trace replay:
- `timestamp`: Request arrival time in milliseconds
- `input_length`: Number of input tokens
- `output_length`: Number of output tokens
- `hash_ids`: List of block hashes (optional)
- `tools`: List of OpenAI-compatible tool definitions (optional, requires `messages`)
- `extra`: Dict of vendor extras (optional). Shallow-merged into the top of the request body at dispatch; user-supplied keys win over `--extra-inputs`.

Example entry:

```json
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [0, 1, 2]}
```

## Profile using a Custom Trace File

Create a trace file with timing information:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > custom_trace.jsonl << 'EOF'
{"timestamp": 0, "input_length": 1200, "output_length": 52, "hash_ids": [0, 1, 2]}
{"timestamp": 105, "input_length": 1800, "output_length": 26, "hash_ids": [0, 3, 4, 5]}
{"timestamp": 274, "input_length": 1300, "output_length": 52, "hash_ids": [1, 4, 6]}
EOF
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->
Run AIPerf with the trace file:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
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
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

The `--fixed-schedule` flag tells AIPerf to send requests at the exact timestamps specified in the trace. This reproduces the original timing pattern.

### Automatic Fixed-Schedule Promotion

When you supply a trace dataset (`--custom-dataset-type mooncake_trace`, `bailian_trace`, `burst_gpt_trace`, ...) and the file's first record carries a `timestamp` field, AIPerf automatically switches the profiling phase to fixed-schedule mode and fills `--request-count` from the number of trace entries. You can pass `--fixed-schedule` explicitly for clarity, but it's no longer required.

A few formats carry timing somewhere other than a top-level `timestamp` and are recognized by type instead: `sagemaker_data_capture` nests it under `eventMetadata`, `burst_gpt_trace` enforces a `Timestamp` column, and `tracelab` derives a submission time per round from its `timing_events[]` array.

To override the auto-promotion — for example, to replay the same trace under a fresh `--concurrency` or `--request-rate` setting and ignore the captured timestamps — pass `--no-fixed-schedule`:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file custom_trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --no-fixed-schedule \
    --concurrency 4 \
    --request-count 100
```

AIPerf refuses parameter sweeps (e.g. `--concurrency 1,2,4`) against an auto-promoted trace; either pin a single value or pass `--no-fixed-schedule` to keep your sweep semantics.

AIPerf sends the stable replay-session ID in both `X-Correlation-ID` and
`X-Session-Affinity` so session-based traffic can be routed consistently.

### Large Baseten Parquet Traces

The `baseten_trace` loader reads trace rows and sampled-session metadata in
bounded batches and projects only the columns required by the selected replay
options. It also skips `inputs.json` generation, which would otherwise
duplicate large recorded prompts. This is independent of the mmap dataset
cache; Baseten traces remain uncached.

The same schema can be stored as one uncompressed Arrow IPC file with an
`.arrow` or `.ipc` suffix. AIPerf memory-maps that file and materializes only
rows selected for replay. Use `--custom-dataset-type baseten_trace`; no schema
or row-order conversion beyond the storage format is required.

## Using Pre-formatted Messages

Instead of synthetic prompts generated from `input_length` and `hash_ids`, you can provide an OpenAI-compatible `messages` array directly per trace entry. This is useful for replaying captured conversations (e.g., coding agent sessions) with exact prompt content.

Each entry's `messages` field contains the full conversation history up to that point. In multi-turn sessions, later entries include prior turns so the server receives the complete context:

```json
{"session_id": "sess-1", "messages": [{"role": "user", "content": "Hello"}], "output_length": 50, "timestamp": 0}
{"session_id": "sess-1", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How are you?"}], "output_length": 30, "timestamp": 2000}
```

The `messages` field is mutually exclusive with `input_length` and `text_input`. When set, the messages array is sent directly to the API payload, bypassing prompt synthesis entirely. By default (`"message_mode": "history"`), the model's actual response is not carried forward between turns -- each turn uses its pre-defined messages.

### Replaying Message Deltas with Live Responses

Set `"message_mode": "delta"` on every `messages` entry in a session to opt in to incremental replay ([`deltas_without_responses` context mode](../reference/conversation-context-mode.md)). The first entry carries the initial history (e.g. system prompt + first user message); each later entry carries only the NEW messages for that turn. Between turns, AIPerf captures the live assistant response -- text and streamed `tool_calls` (reassembled `id`/`function.name`/`function.arguments` deltas) -- and threads it into the history, so each request's context grows with what the model actually generated instead of a recorded transcript:

```json
{"session_id": "sess-1", "message_mode": "delta", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}], "output_length": 512, "timestamp": 0}
{"session_id": "sess-1", "message_mode": "delta", "messages": [{"role": "user", "content": "How are you?"}], "output_length": 512, "delay": 2000}
```

Request 2 is then `[system, user "Hello", assistant <live reply to request 1>, user "How are you?"]`.

Behavior and constraints:

- **Opt-in only.** Omitting `message_mode` (or setting `"history"`) keeps the existing verbatim replay. A session mixing `"delta"` and `"history"` entries -- including entries that omit `message_mode` and silently default to `"history"` -- is rejected at load time. Sessions in the same file are independent: one session can be delta while another is history.
- **`messages` only.** `"message_mode": "delta"` is rejected for `payload`, `text_input`, and synthesized (`input_length`) entries.
- **Timing is unchanged.** `timestamp`/`delay` values are preserved with millisecond scheduling accuracy, exactly as in history mode.
- **No offset cropping.** `--fixed-schedule-start-offset` / `--fixed-schedule-end-offset` drop individual rows before sessions are assembled, which would silently remove the initial history or preceding turns that later delta rows depend on. If the trace file contains any `"message_mode": "delta"` entry, configuring either offset fails at load time -- before any delta-session row can be dropped. History-mode and synthesized traces keep the existing offset filtering. To replay a subset, construct a trace file containing only the complete delta sessions you want instead of cropping with offsets.
- **`output_length` is a ceiling, not a forced length.** It maps to the request's max-completion-tokens cap (e.g. keep your usual 8192 ceiling); the model stops naturally at EOS. Because the live replies are threaded into later requests, the actual context length grows with whatever the model generated -- it will diverge from the recorded transcript's token counts.
- **Tool calls use the live IDs.** Assistant `tool_calls` in the accumulated history carry the IDs the server actually generated during the run. A `role: "tool"` delta must reference a `tool_call_id` the live model emitted for the conversation to remain coherent; AIPerf performs no remapping from recorded IDs to live IDs, so recorded tool-result deltas can diverge from what the live model called. AIPerf never executes tools -- tool-result content comes only from your dataset deltas.
- **`extra` cannot override the assembled history.** A per-entry `extra` containing a `messages` key, or an endpoint-global `--extra-inputs messages:...`, is rejected for delta sessions because the shallow merge would silently replace the accumulated conversation.
- **Sticky transport is not KV affinity.** Multi-turn sessions reuse a sticky connection/correlation ID so all turns hit the same backend, but that does not guarantee the server keeps the session's KV cache on the same GPU -- prefix-cache hit rates depend on the serving stack.

### Tool Definitions

When replaying conversations that involve tool use (function calling), include the `tools` field alongside `messages` to provide the tool definitions the model needs:

```json
{"messages": [{"role": "user", "content": "What's the weather?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}], "output_length": 50, "timestamp": 0}
```

The `tools` field is only valid when `messages` is provided. It is injected directly into the API payload as the `tools` parameter. In delta sessions, later turns inherit the most recent `tools` declaration (typically the first turn's); a turn that redeclares `tools` replaces them for that turn onward.

## Per-Request Extra Inputs

Use the `extra` field to inject arbitrary key-value pairs into the HTTP payload for individual trace entries. This works alongside (and after) the global `--extra-inputs` flag, so per-entry values override global defaults for the same top-level key.

```json
{"input_length": 100, "output_length": 50, "timestamp": 0, "extra": {"nvext": {"priority": 99}}}
{"input_length": 200, "output_length": 30, "timestamp": 500}
{"messages": [{"role": "user", "content": "Hello"}], "output_length": 50, "timestamp": 1000, "extra": {"routing": "fast"}}
```

**Merge semantics:** Merging is shallow — a per-entry `{"nvext": {...}}` replaces the entire global `nvext` key. Deep merge is not performed.

## Profile using real Mooncake Trace

For real-world benchmarking, use the FAST25 production trace data from the Mooncake research paper:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
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
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->
