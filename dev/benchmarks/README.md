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

## Other benchmarks (throughput / latency / serialization)

These do not use `tracemalloc`; they profile CPU, latency, or serialization
cost. Run any of them with `uv run python dev/benchmarks/<script>`.

- **`enum_comparison_benchmark.py`** — Microbenchmark of `CaseInsensitiveStrEnum` comparison (`==`) vs identity/plain-string checks on the per-packet SSE streaming hot path.
- **`serialization_benchmark.py`** — Compares JSON (orjson) vs MessagePack (msgspec) serialization to test whether it bottlenecks the receive path.
- **`zmq_credit_bench.py`** — Open-loop ZMQ credit microbenchmark measuring throughput and round-trip latency percentiles across ZMQ patterns (`await` drive vs `FdEdgeReader`/sync-NOBLOCK), multi-process rig.
- **`per_worker_credit_latency.py`** — Derives per-worker percentiles of credit-to-request-start latency (`request_start_ns - credit_issued_ns`) from the column-store filter + timestamp/numeric-metadata columns.
- **`filter_demo.py`** — Demonstrates the column-store filter + summarize primitives against `MetricsAccumulator` (categorical/bool/time-range masks, numpy boolean ops, `compute_results_for_mask`).
- **`pipeline_profile.py`** — cProfile probe of the metrics-accumulator ingest and summarize path over a synthetic export-schema batch (ingest and `summarize()` timed separately).
- **`record_processing_benchmark.py`** — Microbenchmarks the record-processing drain path; select stages with `--scenario {parser,rp,rm-ingest,export}` and emit machine-readable results with `--json`.
- **`tdigest_batch_benchmark.py`** — Measures per-record vs cross-record batching (batch size K) for the t-digest wrapper `TDigestListMetricAggregator`.

## Running

```bash
uv run python dev/benchmarks/worker_memory_profile.py
uv run python dev/benchmarks/streaming_response_memory_profile.py
uv run python dev/benchmarks/msgspec_vs_pydantic_memory.py
```
