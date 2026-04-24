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
