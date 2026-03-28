# Records manager ingestion benchmark design

Date: 2026-03-28
Status: Proposed
Scope: Standalone synthetic benchmark for the `RecordProcessor -> RecordsManager` ingestion hotspot
Compatibility: No runtime behavior changes. Adds benchmark coverage only.

## Summary

Records ingestion becomes the steady-state throughput limit at large streaming scale. In the observed workload shape (`~250k` concurrency, streaming, `~800` OSL), the system currently tops out around `600-700` ingested records per second, but there is no synthetic benchmark that isolates the `RecordsManager` ingestion path well enough to measure before/after improvements.

The design adds a new standalone benchmark scenario to `dev/benchmarks/record_processing_benchmark.py` that measures the records ingestion hotspot in the same style as the existing dev benchmarks. The benchmark isolates the main layers in the current handoff path so future optimization work can answer three questions quickly:

1. how much time is spent flattening `MetricRecordsMessage.results` into `MetricRecordsData.metrics`
2. how much time is spent in `RecordsManager` bookkeeping
3. how much time is spent dispatching into downstream results processors

The primary output is sustained items/sec and microseconds/item for the current `RecordsManager` path, with additional sub-scenarios that localize where the ingestion ceiling comes from.

## Problem statement

Current state:
- `RecordProcessor` emits one `MetricRecordsMessage` per request
- `RecordsManager._on_metric_records()` immediately calls `message.to_data()`
- `message.to_data()` merges processor result dicts into one metrics dict
- `RecordsManager` updates phase tracking and then dispatches each record into downstream results processors
- the repo already has standalone dev benchmarks for adjacent record-processing stages, but not one that isolates the `RecordsManager` ingestion hotspot directly

This makes performance work slower than it should be:
- real benchmark numbers are noisy and expensive to iterate on
- there is no direct before/after signal for changes to ingress bookkeeping
- there is no direct before/after signal for changes to the RP -> RM handoff format
- it is difficult to tell whether the ceiling is caused mostly by merge overhead, tracker overhead, or downstream processor dispatch

## Design goals

1. Measure the current `RecordsManager` ingestion hotspot with a fast standalone benchmark script.
2. Follow the existing `dev/benchmarks/record_processing_benchmark.py` style and CLI shape.
3. Isolate the main ingestion layers so optimizations can be targeted precisely.
4. Approximate the observed real workload shape well enough to be directionally useful for the streaming bottleneck.
5. Keep the benchmark code-only and non-invasive: no product behavior changes are required to add the benchmark.

## Non-goals

1. Perfectly reproduce the full 250k-concurrency production environment.
2. Replace end-to-end benchmark validation.
3. Add pytest-based perf harnesses.
4. Predict exact real-world throughput from synthetic numbers.

## Proposed design

### 1. Extend the existing standalone benchmark script

Add a new scenario to `dev/benchmarks/record_processing_benchmark.py`:
- `--scenario rm-ingest`

This keeps benchmark execution aligned with the repo's existing microbenchmark workflow:
- `uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest`
- optional `--json` output remains supported
- existing timing/reporting helpers remain the shared implementation pattern

The benchmark should produce one `BenchmarkSample` per sub-scenario so the output fits naturally into the script's existing table and JSON formats.

### 2. Benchmark the real hotspot in-process

The benchmark should exercise the actual current code paths directly in-process rather than simulating them loosely.

Hot-path functions to measure:
- `MetricRecordsMessage.to_data()` in `src/aiperf/common/messages/inference_messages.py`
- `RecordsTracker.update_from_record_data()` in `src/aiperf/records/records_tracker.py`
- `RecordsTracker.check_and_set_all_records_received_for_phase()` in `src/aiperf/records/records_tracker.py`
- `RecordsManager._send_results_to_results_processors()` in `src/aiperf/records/records_manager.py`
- `RecordsManager._on_metric_records()` in `src/aiperf/records/records_manager.py`

The benchmark should avoid unrelated runtime concerns such as inference I/O, worker networking, or controller messaging. It should build synthetic metric-record messages and then drive the `RecordsManager` ingestion logic directly.

### 3. Add focused sub-scenarios

The `rm-ingest` scenario should emit these sub-scenarios.

#### `rm_ingest::to_data_merge`

Measure only the cost of:
- constructing realistic `MetricRecordsMessage` payloads once
- calling `message.to_data()` repeatedly during the timed section

Purpose:
- isolate the current merge/flattening overhead from `results: list[dict]` to `metrics: dict`
- provide a direct before/after signal if the RP -> RM handoff format changes later

#### `rm_ingest::tracker_only`

Measure only the cost of:
- `RecordsTracker.update_from_record_data()`
- `RecordsTracker.check_and_set_all_records_received_for_phase()`

Use prebuilt `MetricRecordsData` inputs so this scenario excludes merge cost.

Purpose:
- quantify the bookkeeping overhead inside the current ingestion path

#### `rm_ingest::metric_processor_only`

Measure only the cost of dispatching prebuilt `MetricRecordsData` into a real `MetricResultsProcessor`.

Purpose:
- isolate the downstream aggregation path from ingress bookkeeping
- determine whether ingestion is limited primarily by downstream metric aggregation rather than message handling

#### `rm_ingest::on_metric_records_total`

Measure the full current `RecordsManager._on_metric_records()` path with minimal stubs for unrelated side effects.

This includes:
- `message.to_data()`
- tracker updates
- completion checks
- dispatch to downstream metric results processors

This is the main comparison number for future optimization work.

#### `rm_ingest::full_with_exports`

Optional heavier sub-scenario that includes export-style downstream processors in addition to the metric processor.

Purpose:
- distinguish ingestion-path limits from exporter-path limits
- show whether the perceived ingestion ceiling actually comes from export-related work in the same downstream path

This mode should stay opt-in inside the benchmark implementation and be clearly labeled in sample details.

### 4. Use synthetic payloads shaped like the real bottleneck

The benchmark should not attempt perfect realism, but it should avoid toy payloads.

Default workload shape should approximate:
- streaming-style records
- enough metric fields to resemble the observed `~800` OSL streaming case
- multiple synthetic producer tasks to exercise steady-state async scheduling pressure
- representative success-path records by default, with optional error/trace knobs available for sensitivity testing

Recommended benchmark knobs:
- `--records`
- `--repeats`
- `--warmup-runs`
- `--processors`
- `--metrics-per-processor`
- `--producer-tasks`
- optional flags for trace/error variants if needed

The first version should keep defaults simple and tuned for the main bottleneck investigation rather than exposing every possible dimension.

### 5. Preserve the existing benchmark UX

The new scenario should reuse the current benchmark script conventions:
- CLI-driven scenario selection
- `BenchmarkSample` outputs
- table output and `--json`
- warmup followed by measured repeats
- GC disabled during timed repeats where appropriate, matching the current script style

This keeps the new benchmark easy to compare with existing record-processing benchmark results and avoids introducing a second benchmark framework.

## Benchmark execution model

1. Build synthetic `MetricRecordsMessage` inputs once per run configuration.
2. Build pre-flattened `MetricRecordsData` inputs once for scenarios that should exclude merge overhead.
3. For each sub-scenario:
   - run warmup iterations
   - run measured repeats
   - report items/sec, microseconds/item, mean time, best time, and scenario details
4. For `on_metric_records_total`, instantiate a lightweight `RecordsManager`-like object or minimally configured real object that exercises the actual ingestion methods without unrelated runtime setup.

## Files to change

- `dev/benchmarks/record_processing_benchmark.py`
  - add `rm-ingest` scenario
  - add synthetic message/data builders needed for records-manager ingestion benchmarking
  - add new benchmark sub-scenarios and CLI options

No production runtime files are required for the initial benchmark-only change.

## Measurement requirements

### Primary metric

- sustained items/sec for `rm_ingest::on_metric_records_total`

### Secondary metrics

- microseconds/item for each sub-scenario
- relative contribution of:
  - merge cost
  - tracker cost
  - downstream metric processor cost
- optional exporter-inclusive comparison from `rm_ingest::full_with_exports`

### Benchmark quality bar

The benchmark is considered useful if it can answer all of the following reliably:
- is the current ceiling dominated by `to_data()` merge cost?
- is the current ceiling dominated by tracker bookkeeping?
- is the current ceiling dominated by metric/export processor dispatch?
- does a code change measurably improve `rm_ingest::on_metric_records_total`?

## Trade-offs

Pros:
- fast local iteration loop for hotspot work
- directly aligned with existing repo benchmark workflow
- isolates the main layers in the current ingestion path
- creates a durable performance signal before changing product code

Cons:
- synthetic throughput will not equal end-to-end throughput exactly
- benchmark realism depends on how well synthetic metric payloads approximate the real workload
- keeping the benchmark focused means some downstream runtime effects remain intentionally out of scope

## Final recommendation

Add a new `rm-ingest` scenario to `dev/benchmarks/record_processing_benchmark.py` with sub-scenarios for merge-only, tracker-only, metric-processor-only, full current ingestion, and optional export-inclusive ingestion. Use that benchmark as the baseline for future records-ingestion optimization work, and judge code changes primarily against `rm_ingest::on_metric_records_total` while using the narrower sub-scenarios to identify where wins come from.
