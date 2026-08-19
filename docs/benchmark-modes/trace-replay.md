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
- `endpoint_type`: Name of a registered endpoint plugin (optional). Routes this row to a different endpoint of the same server. See [Per-Row Endpoint Routing](#per-row-endpoint-routing).

Example entry:

```json
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [0, 1, 2]}
```

> **Note:** `endpoint_type` is an AIPerf extension. The published Mooncake FAST'25 traces carry only `timestamp`, `input_length`, `output_length`, and `hash_ids`; a trace using this field will not be portable to other Mooncake replay tools.

## Per-Row Endpoint Routing

Production traffic is rarely homogeneous: one timeline can interleave chat completions with embeddings requests against the same server. Add `endpoint_type` to a row to send it to a different endpoint than the run-level `--endpoint-type`:

```json
{"timestamp": 0,   "input_length": 300, "output_length": 40}
{"timestamp": 100, "input_length": 300, "endpoint_type": "embeddings"}
{"timestamp": 200, "input_length": 300, "output_length": 40}
```

The named endpoint determines the URL path, the request body format, and the response parser for that row. Rows without the field use `--endpoint-type` as before. Run `aiperf plugins endpoint` to see the registered names.

### Metrics for Routed Requests

Requests routed to an endpoint other than `--endpoint-type` are **issued but not measured**. Their latency metrics do not appear in the results; instead AIPerf reports a per-endpoint count:

```
                     Requests Not Measured
  1 request(s) were sent to an endpoint other than chat by
  per-row dataset routing.
    - embeddings: 1 request
```

The count is also written to `profile_export_aiperf.json` under `metadata.unmeasured_request_counts`.

This exclusion is deliberate. Metric applicability is a property of the endpoint — the embeddings endpoint declares `produces_tokens: false`, so time-to-first-token, inter-token latency, and output sequence length are undefined for its requests. Averaging them into the chat results would silently corrupt those statistics. Routed requests still:

- dispatch on their scheduled timestamp, so they exert real load and affect the measured endpoint's numbers,
- count toward the error totals and `--failed-request-threshold` if they fail.

Per-endpoint metric sections are planned; until then, benchmark an endpoint's own performance in a run where it is the `--endpoint-type`.

### Limitations

`endpoint_type` is rejected at load time in these combinations, so a mistake fails before the benchmark starts rather than producing misleading results:

- **With `payload`.** A `payload` row is sent verbatim and bypasses endpoint formatting entirely, so naming an endpoint would apply only its URL path while ignoring its request format — the row would not really be going to that endpoint. Build routed rows with `input_length`, `text_input`, or `messages`.
- **With `session_id`.** Per-row routing is single-turn only, because multi-turn dispatch replays the whole conversation history on every turn and endpoints such as `embeddings` accept exactly one turn.
- **Naming an endpoint with no path**, such as `raw`. Those are passthroughs for non-standard APIs, configured run-wide via `--endpoint-type` with `--custom-endpoint`, not per row.

`output_length` is accepted but ignored when the row targets an endpoint that generates no tokens (embeddings, rankings, image and video generation, audio transcription). Native Mooncake rows always carry `output_length`, so a real trace routes without edits; AIPerf reports the affected rows once when the trace loads:

```
WARNING  412 row(s) routed to 'embeddings' carry 'output_length'; that endpoint
         generates no tokens, so the value does not apply and is ignored.
```

Two further constraints apply at runtime:

- Authentication (`--api-key`, `--header`) is applied per run, not per endpoint. Every row is sent with the same credentials.
- `--custom-endpoint` applies only to the run-level endpoint. A routed row always uses its own endpoint's registered path.

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

The `messages` field is mutually exclusive with `input_length` and `text_input`. When set, the messages array is sent directly to the API payload, bypassing prompt synthesis entirely. The model's actual response is not carried forward between turns -- each turn uses its pre-defined messages.

### Tool Definitions

When replaying conversations that involve tool use (function calling), include the `tools` field alongside `messages` to provide the tool definitions the model needs:

```json
{"messages": [{"role": "user", "content": "What's the weather?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}], "output_length": 50, "timestamp": 0}
```

The `tools` field is only valid when `messages` is provided. It is injected directly into the API payload as the `tools` parameter.

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
