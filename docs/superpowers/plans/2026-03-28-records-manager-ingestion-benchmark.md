# Records Manager Ingestion Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `rm-ingest` scenario to `dev/benchmarks/record_processing_benchmark.py` that isolates and reports the `RecordProcessor -> RecordsManager` ingestion hotspot.

**Architecture:** Extend the existing benchmark script with one new scenario, a small set of synthetic RM-ingest helpers, and five focused sub-scenarios that reuse the script's timing/reporting helpers. Drive the real hot-path functions in-process using lightweight benchmark doubles for side effects so the measurements stay stable and attributable.

**Tech Stack:** Python 3.10+, asyncio, existing benchmark helpers in `dev/benchmarks/record_processing_benchmark.py`, AIPerf `MetricRecordsMessage` / `MetricRecordsData` / `RecordsTracker` / `MetricResultsProcessor`

---

## File structure

- Modify: `dev/benchmarks/record_processing_benchmark.py`
  - Add synthetic RM-ingest builders
  - Add lightweight doubles for RM-only benchmarking
  - Add `benchmark_records_manager_ingestion()` async scenario
  - Add CLI flags for `rm-ingest`
  - Register the new scenario in `_run_async_scenarios()` and `parse_args()`

No new files are required.

### Task 1: Add benchmark knobs and scenario wiring

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add the failing scenario wiring in `parse_args()` and `_run_async_scenarios()`**

```python
parser.add_argument(
    "--scenario",
    choices=[
        "all",
        "core",
        "parse-variants",
        "full-path",
        "parser",
        "rp",
        "rm-ingest",
        "zmq",
        "tcp-connect",
        "tcp-zmq",
        "sticky-credit",
        "worker-start",
        "mmap",
        "export",
    ],
    default="all",
    help="Benchmark scenario to run.",
)

parser.add_argument(
    "--producer-tasks",
    type=int,
    default=4,
    help="Concurrent producer tasks for rm-ingest scenarios.",
)
parser.add_argument(
    "--rm-include-exports",
    action="store_true",
    help="Include export-style downstream processors in rm-ingest benchmark.",
)
```

```python
if args.scenario in {"all", "rm-ingest"}:
    results.extend(await benchmark_records_manager_ingestion(args))
```

- [ ] **Step 2: Run the benchmark command to verify it fails because the new function does not exist yet**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 100
```

Expected: Python exits with `NameError: name 'benchmark_records_manager_ingestion' is not defined` or equivalent missing-scenario failure.

- [ ] **Step 3: Add the minimal stub so the CLI path resolves**

```python
async def benchmark_records_manager_ingestion(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    return []
```

- [ ] **Step 4: Run the command again to verify the new scenario is wired**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 100
```

Expected: command succeeds and prints no rm-ingest rows yet.

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "chore: wire rm-ingest benchmark scenario"
```

### Task 2: Build synthetic RM-ingest benchmark fixtures

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add helper builders for synthetic RM-ingest payloads**

Add focused helpers near the other benchmark builders:

```python
def _make_rm_metric_results(
    *,
    processor_count: int,
    metrics_per_processor: int,
    request_index: int,
) -> list[dict[str, int | float]]:
    results: list[dict[str, int | float]] = []
    for processor_index in range(processor_count):
        result: dict[str, int | float] = {
            "request_latency": 10.0 + request_index,
            "output_tokens": 800,
        }
        for metric_index in range(metrics_per_processor):
            result[f"bench_metric_{processor_index}_{metric_index}"] = (
                processor_index * 1000 + metric_index + request_index
            )
        results.append(result)
    return results


def _make_rm_metric_message(
    *,
    request_index: int,
    processor_count: int,
    metrics_per_processor: int,
) -> MetricRecordsMessage:
    return MetricRecordsMessage(
        service_id="record-processor-bench",
        metadata=_make_metric_metadata(request_index),
        results=_make_rm_metric_results(
            processor_count=processor_count,
            metrics_per_processor=metrics_per_processor,
            request_index=request_index,
        ),
        trace_data=None,
        error=None,
    )


def _make_rm_metric_messages(args: argparse.Namespace) -> list[MetricRecordsMessage]:
    return [
        _make_rm_metric_message(
            request_index=request_index,
            processor_count=args.processors,
            metrics_per_processor=args.metrics_per_processor,
        )
        for request_index in range(args.records)
    ]
```

- [ ] **Step 2: Run the existing RP benchmark as a safety check before using the new helpers**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rp --records 100 --repeats 1 --warmup-runs 0
```

Expected: one `record_processor_batch_throughput` row prints successfully.

- [ ] **Step 3: Add pre-flattened data builders for tracker/processor-only scenarios**

```python
def _make_rm_metric_data_batch(
    messages: list[MetricRecordsMessage],
) -> list[MetricRecordsData]:
    return [message.to_data() for message in messages]
```

- [ ] **Step 4: Run the new scenario again to verify helpers import and construct cleanly**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 100 --repeats 1 --warmup-runs 0
```

Expected: command still succeeds with no benchmark rows or only stub output.

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "chore: add rm-ingest benchmark fixtures"
```

### Task 3: Add lightweight doubles for RM bookkeeping and dispatch

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add lightweight doubles that exercise the real hot-path methods without full service bootstrap**

```python
class BenchmarkRecordsManager:
    def __init__(self, metric_processors: list[MetricResultsProcessor]) -> None:
        self._records_tracker = RecordsTracker()
        self._metric_results_processors = metric_processors
        self._error_tracker = SimpleNamespace(
            increment_error_count_for_phase=_noop,
        )
        self._handle_all_records_received = AsyncMock()
        self.is_trace_enabled = False
        self.trace = _noop

    async def _send_results_to_results_processors(
        self,
        record_data: MetricRecordsData,
    ) -> None:
        await RecordsManager._send_results_to_results_processors(self, record_data)

    async def _on_metric_records(self, message: MetricRecordsMessage) -> None:
        await RecordsManager._on_metric_records(self, message)
```

If `AsyncMock` is too heavyweight for timing-sensitive paths, replace it with a tiny async no-op function:

```python
async def _noop_async(*args: Any, **kwargs: Any) -> None:
    return None
```

and set:

```python
self._handle_all_records_received = _noop_async
```

- [ ] **Step 2: Add a helper that constructs the metric processor list once per benchmark run**

```python
def _build_rm_metric_processors() -> list[MetricResultsProcessor]:
    return [_build_metric_results_processor(use_server_token_count=True)]
```

- [ ] **Step 3: Run the script to verify the doubles import and instantiate cleanly**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 10 --repeats 1 --warmup-runs 0
```

Expected: no attribute errors from the benchmark doubles.

- [ ] **Step 4: Keep the benchmark double minimal**

Use this exact rule while editing:

```python
# No BaseComponentService bootstrap, no comms setup, no publish/control clients.
# Only fields required by:
# - RecordsManager._on_metric_records
# - RecordsManager._send_results_to_results_processors
# - RecordsTracker methods
```

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "chore: add records manager benchmark doubles"
```

### Task 4: Implement `to_data_merge` and `tracker_only`

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add the `to_data_merge` sub-scenario**

```python
async def _benchmark_rm_to_data_merge(
    args: argparse.Namespace,
    messages: list[MetricRecordsMessage],
) -> BenchmarkSample:
    async def operation(_: int) -> None:
        for message in messages:
            _ = message.to_data()

    return await _time_async_operation(
        name="rm_ingest::to_data_merge",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "processors": args.processors,
            "metrics_per_processor": args.metrics_per_processor,
        },
        operation=operation,
    )
```

- [ ] **Step 2: Add the `tracker_only` sub-scenario**

```python
async def _benchmark_rm_tracker_only(
    args: argparse.Namespace,
    record_data_batch: list[MetricRecordsData],
) -> BenchmarkSample:
    async def operation(_: int) -> None:
        tracker = RecordsTracker()
        for record_data in record_data_batch:
            tracker.update_from_record_data(record_data)
            tracker.check_and_set_all_records_received_for_phase(
                record_data.metadata.benchmark_phase
            )

    return await _time_async_operation(
        name="rm_ingest::tracker_only",
        items=len(record_data_batch),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={"phase": "profiling"},
        operation=operation,
    )
```

- [ ] **Step 3: Call both sub-scenarios from `benchmark_records_manager_ingestion()`**

```python
async def benchmark_records_manager_ingestion(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    messages = _make_rm_metric_messages(args)
    record_data_batch = _make_rm_metric_data_batch(messages)
    return [
        await _benchmark_rm_to_data_merge(args, messages),
        await _benchmark_rm_tracker_only(args, record_data_batch),
    ]
```

- [ ] **Step 4: Run the new scenario and verify both rows appear**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 1000 --repeats 3 --warmup-runs 1
```

Expected: output includes:
- `rm_ingest::to_data_merge`
- `rm_ingest::tracker_only`

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "feat: benchmark records manager merge and tracker costs"
```

### Task 5: Implement `metric_processor_only` and `on_metric_records_total`

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add the `metric_processor_only` sub-scenario using the real metric processor**

```python
async def _benchmark_rm_metric_processor_only(
    args: argparse.Namespace,
    record_data_batch: list[MetricRecordsData],
) -> BenchmarkSample:
    async def operation(_: int) -> None:
        processor = _build_metric_results_processor(use_server_token_count=True)
        for record_data in record_data_batch:
            await processor.process_result(record_data)

    return await _time_async_operation(
        name="rm_ingest::metric_processor_only",
        items=len(record_data_batch),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={"downstream": "MetricResultsProcessor"},
        operation=operation,
    )
```

- [ ] **Step 2: Add the full `on_metric_records_total` sub-scenario**

```python
async def _benchmark_rm_on_metric_records_total(
    args: argparse.Namespace,
    messages: list[MetricRecordsMessage],
) -> BenchmarkSample:
    async def operation(_: int) -> None:
        benchmark_rm = BenchmarkRecordsManager(_build_rm_metric_processors())
        for message in messages:
            await benchmark_rm._on_metric_records(message)

    return await _time_async_operation(
        name="rm_ingest::on_metric_records_total",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "includes_merge": True,
            "includes_tracker": True,
            "includes_metric_processor": True,
        },
        operation=operation,
    )
```

- [ ] **Step 3: Add both sub-scenarios to the main rm-ingest scenario**

```python
return [
    await _benchmark_rm_to_data_merge(args, messages),
    await _benchmark_rm_tracker_only(args, record_data_batch),
    await _benchmark_rm_metric_processor_only(args, record_data_batch),
    await _benchmark_rm_on_metric_records_total(args, messages),
]
```

- [ ] **Step 4: Run the scenario and verify the main comparison row exists**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 1000 --repeats 3 --warmup-runs 1 --json
```

Expected: JSON output contains objects named:
- `rm_ingest::metric_processor_only`
- `rm_ingest::on_metric_records_total`

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "feat: benchmark full records manager ingestion path"
```

### Task 6: Implement optional export-inclusive mode

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Add a lightweight export-style benchmark processor instead of bootstrapping real file writers**

```python
class BenchmarkExportProcessor:
    async def process_result(self, record_data: MetricRecordsData) -> None:
        metric_record = MetricRecordInfo(
            metadata=record_data.metadata,
            metrics=MetricRecordDict(record_data.metrics).to_display_dict(
                MetricRegistry,
                False,
                False,
            ),
            trace_data=record_data.trace_data,
            error=record_data.error,
        )
        orjson.dumps(metric_record.model_dump(exclude_none=True, mode="json"))
```

- [ ] **Step 2: Add `full_with_exports` as an opt-in sub-scenario**

```python
async def _benchmark_rm_full_with_exports(
    args: argparse.Namespace,
    messages: list[MetricRecordsMessage],
) -> BenchmarkSample:
    async def operation(_: int) -> None:
        benchmark_rm = BenchmarkRecordsManager(
            _build_rm_metric_processors() + [BenchmarkExportProcessor()]
        )
        for message in messages:
            await benchmark_rm._on_metric_records(message)

    return await _time_async_operation(
        name="rm_ingest::full_with_exports",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={"includes_export_serialization": True},
        operation=operation,
    )
```

- [ ] **Step 3: Gate the export scenario behind `--rm-include-exports`**

```python
results = [
    await _benchmark_rm_to_data_merge(args, messages),
    await _benchmark_rm_tracker_only(args, record_data_batch),
    await _benchmark_rm_metric_processor_only(args, record_data_batch),
    await _benchmark_rm_on_metric_records_total(args, messages),
]
if args.rm_include_exports:
    results.append(await _benchmark_rm_full_with_exports(args, messages))
return results
```

- [ ] **Step 4: Run both forms and verify the optional row only appears when requested**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 1000 --repeats 2 --warmup-runs 1
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 1000 --repeats 2 --warmup-runs 1 --rm-include-exports
```

Expected:
- first command does **not** print `rm_ingest::full_with_exports`
- second command **does** print `rm_ingest::full_with_exports`

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "feat: add export-inclusive records manager benchmark mode"
```

### Task 7: Add producer concurrency and finalize benchmark ergonomics

**Files:**
- Modify: `dev/benchmarks/record_processing_benchmark.py`

- [ ] **Step 1: Update the rm-ingest operations to shard work across producer tasks**

```python
def _chunked[T](items: list[T], parts: int) -> list[list[T]]:
    if parts <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / parts))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]
```

```python
async def _run_concurrent_batches(
    batches: list[list[Any]],
    worker: Callable[[list[Any]], Awaitable[None]],
) -> None:
    await asyncio.gather(*(worker(batch) for batch in batches if batch))
```

Apply this pattern inside rm-ingest sub-scenarios so `args.producer_tasks` affects the measured scheduling pressure without changing benchmark semantics.

- [ ] **Step 2: Make the details payload describe the synthetic workload clearly**

Use this exact details shape for the main row:

```python
details={
    "processors": args.processors,
    "metrics_per_processor": args.metrics_per_processor,
    "producer_tasks": args.producer_tasks,
    "streaming_shape": True,
    "target_output_tokens": 800,
}
```

- [ ] **Step 3: Run the benchmark with the intended investigation shape**

Run:
```bash
uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest --records 10000 --processors 3 --metrics-per-processor 12 --producer-tasks 8 --repeats 5 --warmup-runs 1
```

Expected: table output includes stable rm-ingest rows with non-zero throughput and the details block shows the concurrency knobs.

- [ ] **Step 4: Run pre-commit on the modified file**

Run:
```bash
pre-commit run --files dev/benchmarks/record_processing_benchmark.py
```

Expected: all hooks pass for the benchmark file.

- [ ] **Step 5: Commit**

```bash
git add dev/benchmarks/record_processing_benchmark.py
git commit -m "feat: finish records manager ingestion benchmark"
```

## Spec coverage check

- New standalone `rm-ingest` scenario: covered by Tasks 1, 4, 5, 6, and 7.
- Real hot-path measurement of `to_data()`, tracker work, processor dispatch, and `_on_metric_records()`: covered by Tasks 3, 4, and 5.
- Optional exporter-inclusive comparison: covered by Task 6.
- Existing benchmark-script UX and CLI shape: covered by Tasks 1 and 7.
- Producer concurrency knob and streaming-shaped details: covered by Task 7.

## Placeholder scan

- No `TODO`, `TBD`, or deferred implementation markers remain.
- Every code-changing step includes the concrete code shape to add.
- Every verification step includes an exact command and expected result.

## Type consistency check

- Scenario function name is consistently `benchmark_records_manager_ingestion`.
- Scenario names are consistently:
  - `rm_ingest::to_data_merge`
  - `rm_ingest::tracker_only`
  - `rm_ingest::metric_processor_only`
  - `rm_ingest::on_metric_records_total`
  - `rm_ingest::full_with_exports`
- The benchmark double consistently exposes `_records_tracker`, `_metric_results_processors`, `_handle_all_records_received`, and `_send_results_to_results_processors`, which are the fields/methods used by the real `RecordsManager` methods being called.
