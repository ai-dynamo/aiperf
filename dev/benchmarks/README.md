# Memory Benchmarks

Standalone scripts for profiling memory usage of AIPerf's hot-path data structures. All use `tracemalloc` snapshot diffing to measure net allocations.

## Scripts

### `worker_memory_profile.py`

Measures memory of core Worker data structures under load:

- **Single objects**: `Turn`, `Conversation`, `RequestRecord`, `Credit`/`CreditContext` at various payload sizes
- **Session cache**: `UserSessionManager` with N cached sessions x M turns x varied prompt sizes
- **Multi-turn growth**: How `UserSession.turn_list` grows as assistant responses accumulate
- **Concurrent load simulation**: Steady-state Worker memory with N concurrent slots, each holding a session + credit context + in-flight request record
- **RequestInfo growth**: How `RequestInfo.turns` scales with conversation history length

### `streaming_response_memory_profile.py`

Measures memory of the full SSE streaming pipeline:

- **SSE chunk memory**: Individual `SSEMessage` and `SSEField` sizes at various content lengths
- **Stream accumulation**: `RequestRecord` memory after accumulating all SSE chunks from a streaming response
- **Endpoint parsing**: `ParsedResponse` list memory after `ChatEndpoint.extract_response_data()`
- **Full record pipeline**: Complete `ParsedResponseRecord` (raw SSE + parsed + request info + token counts)
- **Concurrent streaming**: N concurrent in-flight parsed records at various token counts
- **Parsing overhead ratio**: Compares raw SSE byte size vs in-memory object representation

### `msgspec_vs_pydantic_memory.py`

Three-way memory comparison: **Pydantic** vs **`@dataclass(slots=True)`** vs **msgspec Struct**.

Defines equivalent model hierarchies for all hot-path types (SSEMessage, ParsedResponse, RequestRecord, RequestInfo, Turn, Text, TokenCounts, etc.) and measures:

- **SSE message lists**: Per-message overhead across serialization approaches
- **Parsed response lists**: Post-parse object memory comparison
- **Full record pipeline**: Nested RequestInfo + turns + SSE + parsed responses
- **Concurrent load at scale**: N x full records to show aggregate savings
- **Per-object overhead**: Bulk allocation (200-1000 instances) for stable per-item cost
- **Three-way full record**: Single and concurrent comparisons across all three approaches
- **Three-way per-object**: Individual type overhead (SSEMessage, ParsedResponse, Turn, RequestInfo, TokenCounts)

## Running

```bash
uv run python dev/benchmarks/worker_memory_profile.py
uv run python dev/benchmarks/streaming_response_memory_profile.py
uv run python dev/benchmarks/msgspec_vs_pydantic_memory.py
```

## Cross-branch comparison: metrics-accumulator pipeline

### `metrics_pipeline_worker.py`

Single-pipeline measurement probe. Auto-detects which pipeline is importable
in the current worktree and reports a single-line JSON record:

- New pipeline (`ajc/k8s-metrics`): `MetricsAccumulator` + `ColumnStore` + `RaggedSeries`
- Old pipeline (`../new-config-kube`, `ajc/k8s`): `MetricResultsProcessor` +
  `MetricArray` + `TDigestListMetricAggregator`

Two record sources:

- **Synthetic** (`--n-records N --avg-icl-chunks K`): deterministic stream
  (seeded `Random(42)`, log-normal ICL distribution, 5 registered metric tags:
  `time_to_first_token`, `request_latency`, `output_token_count`,
  `output_sequence_length`, `input_sequence_length`, `inter_chunk_latency`).
- **File replay** (`--records-file PATH [--repeat N]`): loads a real
  `profile_export.jsonl` artifact, strips the `{"value": …, "unit": …}`
  envelope, and replays each record `repeat` times with monotonically-bumped
  `session_num` so the ColumnStore doesn't overwrite slots. Captures all 25+
  metric tags the real export carries (HTTP timing, per-chunk latencies,
  throughput-per-user metrics) — far more realistic per-tag scalar-array
  pressure than the synthetic generator can fake.

Either source ingests through `process_record` / `process_result`, then runs
`summarize()`. Reports per-stage `tracemalloc` peak, post-stage RSS, wall time,
and a `pympler.asizeof` breakdown attributed to the pipeline-owned container,
including a per-metric-tag scalar-array breakdown for apples-to-apples
comparison (`per_tag_numeric` on the new side, `per_tag_metric_array` on the
old side).

```bash
# Synthetic
uv run python dev/benchmarks/metrics_pipeline_worker.py \
    --n-records 100000 --avg-icl-chunks 100 --slice-duration 5

# Replay a real artifact
uv run python dev/benchmarks/metrics_pipeline_worker.py \
    --records-file /path/to/profile_export.jsonl --repeat 16
```

### `metrics_pipeline_compare.py`

Driver that runs the worker against both worktrees, sweeps either the
synthetic grid or a file-replay multiplier, writes `summary.md` +
`results.json` + matplotlib PNGs under
`dev/benchmarks/results/metrics_pipeline_<timestamp>/`.

```bash
# Synthetic sweep: n_records ∈ {10k, 50k, 100k, 500k, 1M} at icl=100,
#                 plus avg_icl_chunks ∈ {10, 50, 100, 200, 500} at n=100k
uv run python dev/benchmarks/metrics_pipeline_compare.py
uv run python dev/benchmarks/metrics_pipeline_compare.py --quick

# Real-artifact replay sweep
uv run python dev/benchmarks/metrics_pipeline_compare.py \
    --records-file /path/to/profile_export.jsonl \
    --repeats 1,4,16,64,256

# Custom old-worktree path
uv run python dev/benchmarks/metrics_pipeline_compare.py \
    --old-worktree /path/to/new-config-kube
```

The driver copies the worker into the old worktree at start so both sides run
identical loader + measurement code against their own pipeline imports. Each
run is a separate `uv run` subprocess so the two pipelines never share a
Python process.
