# Streaming Aggregation Design

**Status:** research / scoping (no code in this branch)
**Author:** investigation 2026-04-29
**Triggering incident:** controller pod `8/9 OOMKilled` at 64 min during a 1 h run with 20,854 records (5.81 RPS, ISL=512, OSL=200, concurrency=32). Workaround `bf9f4c212` flips default `resourceMode` from `guaranteed` to `burstable`. This doc scopes the real fix.

## Goal

Let `guaranteed` mode (cgroup-bounded controller pod) survive an arbitrarily long benchmark by replacing full-history retention in the records manager with single-pass aggregation, without breaking any of the existing artifacts in `profile_export_aiperf.{json,csv}` or the per-record JSONL/CSV files.

---

## 1. Where does memory grow?

End-to-end data path from a worker pod's HTTP response to `profile_export_aiperf.json`:

```mermaid
flowchart LR
    W[Worker pod<br/>InferenceService] -->|RawInference ZMQ push| RP[RecordProcessor pod<br/>parses + computes per-record metrics]
    RP -->|MetricRecordsWireMessage / Batch<br/>over CommAddress.RECORDS PUSH/PULL| RM[RecordsManager<br/>controller pod]
    RM --> MRP[MetricResultsProcessor<br/>_results: MetricResultsDict]
    RM --> RER[RecordExportResultsProcessor<br/>buffered_write JSONL]
    RM --> RCSV[RecordExportCSVProcessor<br/>buffered_csv_write]
    RM --> TS[TimesliceMetricResultsProcessor<br/>defaultdict by slice index]
    RM -. CHECKPOINT_INTERVAL=30s .-> CKPT[partial checkpoint<br/>JsonExportData snapshot]
    MRP -->|summarize once at end| PRR[ProcessRecordsResultMessage<br/>list[MetricResult] scalars]
    PRR -->|ZMQ pub| SC[SystemController<br/>_profile_results]
    SC --> EM[ExporterManager.export_data]
    EM --> JSON[(profile_export_aiperf.json)]
    EM --> CSV[(profile_export_aiperf.csv)]
```

Holders inspected, ranked by per-record cost on this branch:

### A. `MetricResultsProcessor._results` (the big one)
File: `src/aiperf/post_processors/metric_results_processor.py:61`
Shape: `MetricResultsDict = dict[MetricTagT, MetricArray | ListMetricAggregator | scalar]` (`src/aiperf/metrics/metric_dicts.py:121`).

For each record, `_process_record_metric` (lines 129-159) appends one float per RECORD-typed tag into the per-tag accumulator. RECORD-metric class count on this branch is 27 (`grep -rln BaseRecordMetric src/aiperf/metrics/`), of which after endpoint filtering (`get_filters` in `base_metrics_processor.py:27`) maybe 12-18 are active for a chat/completions streaming run — **unverified** without instrumentation, but bounded.

Two backing stores:

- **Scalar RECORD metrics** — `MetricArray` wrapping `GrowableArray(np.float64, track_sum=True)` (`src/aiperf/common/growable_array.py:15`). Doubles capacity on overflow (`_grow_to`, line 110). Steady-state cost: 8 bytes × N_records × N_scalar_tags. At N=21k × ~15 tags ≈ **2.5 MB** plus 2× overhead from doubling = **~5 MB** worst case.
- **List-valued RECORD metrics** — handled via `ListMetricAggregator` (`src/aiperf/metrics/list_metric_aggregation.py:27`). Two implementations:
  - `ExactListMetricAggregator` (default — `ListMetricAggregationMode.EXACT` per `src/aiperf/config/metrics.py:24`) — backed by `MetricArray`, retains every per-chunk sample. ICL escapes this via `MetricFlags.AGGREGATE_TDIGEST` on `InterChunkLatencyMetric` (`src/aiperf/metrics/types/inter_chunk_latency_metric.py:39`). **Other list-valued RECORD metrics that lack the flag fall back to exact retention** — list of those is unverified on this branch, but the build-time selection is in `build_list_metric_aggregator_for_tag`, line 195.
  - `TDigestListMetricAggregator` — bounded ~few KB.

For OSL=200 streaming, an exact list metric is 200 samples × 21 k records × 8 B = **34 MB per tag**. Two such tags ⇒ ~70 MB. Three ⇒ ~100 MB. This is the most likely single contributor; cgroup limit of ~2-4 GiB still survives this on its own, so the next item matters.

### B. `summarize()` numpy temp buffers
File: `src/aiperf/metrics/metric_dicts.py:203` — `MetricArray.to_result` calls `np.percentile(arr, [1,5,10,25,50,75,90,95,99])` plus `np.std` and `np.mean`.
- `np.percentile` makes an internal sorted copy of the array (numpy implementation detail, **unverified** that it's not in-place; either way at least one copy is produced for partition).
- `np.std` allocates a deviations array same size as input.

This means `summarize()` peak ≈ 2-3× the in-memory array footprint. At end-of-run on 100 MB of list metrics, the peak briefly hits 200-300 MB.

Crucially: **`write_partial_checkpoint` calls `generate_realtime_metrics(processors)` every `RECORD.CHECKPOINT_INTERVAL` (default 30 s)** (`src/aiperf/records/records_manager.py:429-447`, `records_manager_processing.py:79`). Each call does a full `summarize()` on every processor, which means **120 full sort+percentile passes over the growing arrays** in a 1 h run. If any of those passes leaks memory through the numpy/Python allocator (HWM retention — see `gotcha_python_native_allocator_hwm_retention.md` in user memory), it accumulates across calls. This is the most plausible OOM mechanism on top of the steady-state list-metric retention.

### C. `TimesliceMetricResultsProcessor`
File: `src/aiperf/post_processors/timeslice_metric_results_processor.py:51`. Creates a fresh `MetricResultsDict` per timeslice index via `defaultdict`. With `slice_duration=10s` over 3607 s = 360 slices × per-slice arrays of avg 60 records = roughly the same total cost as a single non-timeslice processor, but split. Only active if `artifacts.slice_duration` is set (default `None`, `src/aiperf/config/artifacts.py:117`). **Off in this incident.**

### D. `RawRecordWriterProcessor` and `RecordExportResultsProcessor` and `RecordExportCSVProcessor`
Files: `src/aiperf/post_processors/raw_record_writer_processor.py`, `record_export_results_processor.py`, `record_export_csv_processor.py`.
All three already extend `BufferedJSONLWriterMixin` / `BufferedCSVWriterMixin` (`src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:23`). Buffer is bounded by:
- `RECORD.RAW_EXPORT_BATCH_SIZE = 10` (`src/aiperf/common/_env_services.py:68`)
- `RECORD.EXPORT_BATCH_SIZE = 100` (line 56)
- `RECORD.EXPORT_FLUSH_INTERVAL = 2.0` s (line 62)
Plus a periodic background-task flush (`buffered_jsonl_writer_mixin.py:145`). **Already streaming. Skip.**

### E. `RecordsManager._records_tracker`
File: `src/aiperf/records/records_tracker.py:17` — pure counters per phase + per-worker stats `defaultdict(WorkerProcessingStats)`. Bounded by worker count, not record count. **O(1).**

### F. `RecordsManager._error_tracker`
File: `src/aiperf/records/error_tracker.py`. Counts errors by phase + tracks unique error fingerprints. Bounded by unique-error cardinality. **Effectively O(1).**

### G. SystemController `_profile_results`
Files: `src/aiperf/controller/system_controller.py:1046`, `metric_result_models.py:127`. Holds `ProcessRecordsResult.results.records: list[MetricResult]` — already-summarized scalar dataclasses (`metric_result_models.py:19`), one entry per metric tag. **Bounded by tag count, not record count.** This is what crosses the ZMQ wire from RecordsManager and what the `ExporterManager` reads. Not a growth source.

### H. ExporterManager / MetricsJsonExporter / MetricsCsvExporter
Files: `src/aiperf/exporters/exporter_manager.py:34`, `metrics_json_exporter.py:36`, `metrics_csv_exporter.py:40`. Both materialize the entire JSON/CSV string in memory (`_generate_content` returns `str`, `metrics_base_exporter.py:70`) before a single `aiofiles` write. For a fixed-size summary this is fine. **Not a growth source per record.**

### Summary of growth attribution

| Source | Per-record cost | 21k records | OOM contribution |
|---|---|---|---|
| Scalar `MetricArray` ×15 tags | 8 B × 15 = 120 B | ~5 MB | low |
| Exact list aggregators (non-tdigest list metrics) | ~OSL × 8 B × N_tags | 30-100 MB | medium |
| `summarize()` numpy temp buffers (called 120×/run via checkpoint) | transient 2-3× | up to 200-300 MB peak | **high** (driver) |
| Buffered JSONL/CSV writers | bounded buffer | <1 MB | none |
| RecordsTracker / ErrorTracker | constant | <1 MB | none |
| SystemController.\_profile\_results | bounded by tag count | <1 MB | none |
| Exporter content-string materialization | bounded by tag count | <10 MB | none |

The **single biggest fix** is getting summarize() out of the hot path during the run (currently called every 30 s by checkpoint and on every realtime UI tick by `_report_realtime_inference_metrics_task`).

## 2. What's already incremental?

- `BufferedJSONLWriterMixin` (`src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:23`) — batch + interval flush, lazy file open, on-stop drain. Used by `RecordExportResultsProcessor`, `RawRecordWriterProcessor`, GPU-telemetry / server-metrics jsonl writers. **O(batch_size) memory.**
- `BufferedCSVWriterMixin` (`src/aiperf/common/mixins/buffered_csv_writer_mixin.py:18`) — same shape. Used by `RecordExportCSVProcessor`. **O(batch_size) memory.**
- `RawRecordAggregator.export()` (`src/aiperf/post_processors/raw_record_writer_processor.py:149`) — concatenates per-RP files line-by-line via `aiofiles`, never loads all records. **O(line) memory.**
- `MetricResultsProcessor` aggregate-typed metrics (`metric_results_processor.py:122-126`) — `BaseAggregateMetric.aggregate_value(value)` updates a single accumulator (e.g. running counter, max). **O(1) per metric.**
- `GrowableArray.track_sum=True` (`growable_array.py:49`) — running sum, used by `np.mean` shortcut via `MetricArray.sum`. Already O(1) for sum/avg.
- `TDigestListMetricAggregator` (`list_metric_aggregation.py:139`) — bounded sketch. Already used by `inter_chunk_latency` metric.
- `RecordsTracker`, `ErrorTracker` — counter-only.
- `ServerMetricsAccumulator` writes Parquet incrementally; out of scope here (covered by §7).

## 3. What forces full-history retention?

Three things, in decreasing severity:

### 3a. Exact percentiles via numpy on a retained array
File: `src/aiperf/metrics/metric_dicts.py:203-229`.
`np.percentile(arr, [1,5,10,25,50,75,90,95,99])` requires the full sorted array. There's no streaming exact percentile. The only options are (1) a sketch (t-digest, HDR histogram, KLL) which gives approximate percentiles in O(log N) memory, or (2) accepting an external secondary pass over the streamed JSONL.

### 3b. ExactListMetricAggregator backing for list-valued metrics
File: `src/aiperf/metrics/list_metric_aggregation.py:105-137`. Default `ListMetricAggregationMode = EXACT` (`src/aiperf/config/metrics.py:21-24`). Only metrics flagged `MetricFlags.AGGREGATE_TDIGEST` (currently only `inter_chunk_latency`) escape exact retention regardless of mode. Per-class flag override is in `build_list_metric_aggregator_for_tag` line 195.

### 3c. Repeated `summarize()` invocation by `_write_partial_checkpoint_task`
File: `src/aiperf/records/records_manager.py:429-447`. The checkpoint exists for crash recovery (`docs/dev/kubernetes-flow.md` references it as the partial-results signal the operator can read while a job is mid-run). It calls `generate_realtime_metrics(processors)` every 30 s, which in turn calls `summarize()` on every processor — full sort + percentile pass over the growing arrays. Across a 1 h run that is 120 full passes. If numpy / glibc allocator HWM retention applies (it does, per the user-memory gotcha), this can fragment up to GBs of RSS that `gc.collect()` can't return.

The realtime-UI path `_report_realtime_inference_metrics_task` (line 386) does the same on `Environment.UI.REALTIME_METRICS_INTERVAL` cadence, but only when an interactive UI or API port is enabled. **Both paths recompute, neither caches.**

## 4. Where's the natural cut line?

Minimal change set, in order of impact:

### Cut 1: Replace exact-percentile path with a streaming sketch *for in-process aggregation only*
Keep the full per-record JSONL on disk (it already streams). The aggregate `profile_export_aiperf.json` only needs p1, p5, p10, p25, p50, p75, p90, p95, p99 per metric tag, plus min/max/avg/sum/std. All of these except percentiles are O(1) running. Percentiles via t-digest are within ~1% of exact at p99 with a few KB sketch.

- Replace `MetricArray` (in `metric_dicts.py:159`) with a `MetricSketch` class that maintains: count, sum, sum-of-squares (for std), min, max, and a t-digest. Implements the same `MetricSeriesProtocol` (`metric_dicts.py:144`) so callers don't see the difference.
- Reuse `TDigestListMetricAggregator` shape; the `tdigest` pip dep already exists (line 9 of `list_metric_aggregation.py` — gated import).
- Default `ListMetricAggregationMode` flips from `EXACT` to `TDIGEST` for list-valued metrics. Keep `EXACT` reachable as opt-in (debugging / accuracy validation).

**LOC ballpark: ~150 LOC.** New class + swap two construction sites + delete `MetricArray.to_result`'s numpy block. Tests for percentile-tolerance against current exact path: ~100 LOC.

### Cut 2: Cache the `summarize()` result during the run
The checkpoint and realtime paths re-run `summarize()` from scratch every tick. Add a `_last_summary` cache on `MetricResultsProcessor` that is invalidated when `process_result` runs. Only recompute when at least N records have arrived since the last cached snapshot, or M seconds have passed.

This decouples "background snapshot for crash recovery" from "live aggregation cost" — sketch updates remain O(1) per record, and snapshot turns into a serialization of already-finalized values rather than a re-aggregation.

**LOC ballpark: ~50 LOC.** Add `_summary_cache`, `_records_since_summary` counter, gate in `summarize()`. Touches `metric_results_processor.py:212`, `timeslice_metric_results_processor.py:97`, `records_manager_export.py:116`.

### Cut 3 (optional, lower impact): Drop the running list-of-records assumption from checkpoint
`write_partial_checkpoint` in `records_manager_export.py:116` currently builds a `JsonExportData` with all metric results inline; that part is already fine since the output is bounded by tag count. **No change needed once Cut 1 lands**, because the upstream sketches are already small.

After Cuts 1 + 2, in-memory cost is independent of N_records. Estimated controller RSS at end of a 1 h × 21 k run drops from "OOM at 64 min on 1.5 GiB" to "~300 MB steady-state" (back-of-envelope, unverified).

### What we explicitly do not change
- `BufferedJSONLWriterMixin` / `BufferedCSVWriterMixin` — already correct.
- `RecordsTracker` / `ErrorTracker` — already counter-only.
- `ProcessRecordsResultMessage` payload — already summarized.
- `MetricsJsonExporter` / `MetricsCsvExporter` content-string materialization — bounded by tag count, no need to chunk.

## 5. What's the API impact?

Public protocols:

- `RecordProcessorProtocol.process_record` and `ResultsProcessorProtocol.process_result` / `summarize` (`src/aiperf/post_processors/protocols.py:24-36`) — **unchanged.** The streaming sketch replacement is hidden behind `MetricSeriesProtocol`.
- `MetricSeriesProtocol` (`src/aiperf/metrics/metric_dicts.py:144`) — `sum`, `__len__`, `to_result(tag, header, unit)`. The new `MetricSketch` already satisfies this contract, so callers (`MetricResultsProcessor._create_metric_result` at line 237) need no change.
- `MetricResult` (`src/aiperf/common/models/metric_result_models.py:19`) — **unchanged.** Same fields, same JSON wire shape, same Pydantic compatibility.
- `JsonExportData` schema — unchanged.
- `profile_export_aiperf.{json,csv}` byte schema — unchanged at field level. Only changes value: `p99` etc. become approximate. **Schema-level change zero**, value-level acceptance criterion: documented tolerance vs exact mode.

Exporters that break the streaming contract today: **none.** The aggregate exporters (`metrics_csv_exporter.py`, `metrics_json_exporter.py`) consume the already-summarized `ProfileResults.records` — they don't see the in-process state at all. The only callers of `MetricArray.to_result` are inside `MetricResultsProcessor` (`metric_results_processor.py:244`) and `ExactListMetricAggregator.to_result` (`list_metric_aggregation.py:128`); both go through `MetricSeriesProtocol`.

### Sweep aggregator — separate
`SweepAnalyzer.compute()` in `src/aiperf/orchestrator/aggregation/sweep.py` and `aggregate_sweep_and_export` (cited in `CLAUDE.md` Parameter Sweeping section) operate on per-variation `BenchmarkRun` outputs *after* each variation has produced its summarized `MetricResult` list. They don't see per-record data. **No rework needed.**

The k8s `sweep_controller` walks children-manifest by index → reads each child's already-exported aggregate JSON → does the same post-hoc aggregation. **Unaffected.**

## 6. Migration strategy

Three phases, all opt-in via an env var until confidence is high:

### Phase 1 — env-gated streaming sketches
Add `Environment.METRICS.STREAMING_AGGREGATION` boolean in `src/aiperf/common/environment.py` (per CLAUDE.md `feedback_constants_in_environment_py`, all tunables live as `Field` on `_XxxSettings`).
Default `False` initially. When `True`:
- `build_list_metric_aggregator_for_tag` returns `TDigestListMetricAggregator` regardless of mode.
- `MetricArray` construction in `MetricResultsProcessor._process_record_metric` (line 152) routes through a new `_build_scalar_aggregator()` factory that returns either `MetricArray` (legacy) or `MetricSketch` (streaming) based on the flag.
- The summarize-cache (Cut 2) is unconditional — it's a pure perf optimization.

CI: add a parameterized variant of the existing metrics test suite running once per mode; assert percentile tolerance ≤1% relative on a fixed-seed dataset.

### Phase 2 — auto-enable under operator
In `cli_runner._reject_in_process_sweep_under_operator`-adjacent logic (or earlier in `cli_runner.py`), when `AIPERF_OPERATOR_MANAGED=1` and `resourceMode=guaranteed`, force `STREAMING_AGGREGATION=True` regardless of user setting. Keeps the cgroup-bounded path safe by default while leaving local CLI runs on the exact path. Document this in `docs/kubernetes/configuration.md` and (per the table in CLAUDE.md) `docs/environment-variables.md` (auto-generated).

### Phase 3 — flip the default
After two release cycles with no percentile-quality regressions, flip `STREAMING_AGGREGATION` default to `True` and reclassify the legacy path as opt-in for byte-exact reproducibility against historical runs. Mark the legacy path for removal one release later.

The migration **must** be opt-in first, not all-at-once: there are accuracy-sensitive consumers that compare run-over-run percentile drift, and any sketch will perturb the absolute values.

## 7. What stays out of scope

- **Server-metrics streaming.** Already incrementalized — `ServerMetricsAccumulator` writes Parquet via `ServerMetricsParquetExporter` (`plugins.yaml:820`) on a separate sidecar accumulator path. The `parquet_exporter` is even explicitly skipped in `ExporterManager.export_data` (line 67) because it's owned by the records-manager-side accumulator, not the ExporterManager.
- **`inputs.json`.** One-shot dataset materialization at job start (artifact written by the dataset stage before workers spin up). Bounded by dataset size, not run length.
- **Sweep aggregation across variations.** `SweepAnalyzer` consumes already-summarized per-variation outputs; per-record data never reaches it. The k8s-side sweep_controller walks the children-manifest → reads each child's exported aggregate JSON; same shape.
- **GPU telemetry.** `GPUTelemetryAccumulator` + `GPUTelemetryJSONLWriter` already stream (`plugins.yaml:751-762`), batch-flushed via the same buffered-JSONL pattern.
- **Raw records JSONL.** Already streamed by `RawRecordWriterProcessor`; `RawRecordAggregator.export` line-by-line concatenates the per-RP files. Memory-bounded.
- **Per-record JSONL/CSV.** Already streamed via `BufferedJSONLWriterMixin` / `BufferedCSVWriterMixin`.

## Implementation phases

A linearized checklist if work splits cleanly:

- [ ] **P1.1** Land `MetricSketch` class implementing `MetricSeriesProtocol` (sum, count, min, max, sum_of_squares for std, t-digest for percentiles). Tests: percentile tolerance vs `MetricArray.to_result()` on fixed seeds.
- [ ] **P1.2** Add `Environment.METRICS.STREAMING_AGGREGATION` field; route construction in `MetricResultsProcessor._process_record_metric` and `build_list_metric_aggregator_for_tag` through a factory gated by the flag.
- [ ] **P1.3** Add `MetricResultsProcessor._summary_cache` + invalidation on `process_result`; gate by record-count delta + time delta in `summarize()`. Same shape on `TimesliceMetricResultsProcessor`.
- [ ] **P1.4** Document new env var in `docs/environment-variables.md` (auto-generated by `make generate-env-vars-docs`) and add a migration note to `docs/kubernetes/configuration.md`.
- [ ] **P2.1** Auto-enable streaming aggregation when `AIPERF_OPERATOR_MANAGED=1` and `resourceMode=guaranteed` at controller startup. Log the policy decision once at INFO.
- [ ] **P2.2** Audit suite (`tests/kubernetes/audit/`, per CLAUDE.md K8s Patterns) — add a streaming-vs-exact tolerance bucket so the existing operator-vs-local audit suite keeps passing.
- [ ] **P2.3** Run the same workload that triggered the OOM (1 h, ISL=512, OSL=200, conc=32, sticky-sessions, real LLM) under `guaranteed` + streaming aggregation. Capture controller RSS curve. Acceptance: peak RSS < 1.5× steady-state, no OOM, exported aggregate matches exact-mode within 1% relative on every percentile.
- [ ] **P3.1** Flip `STREAMING_AGGREGATION` default to `True` after one release cycle clean.
- [ ] **P3.2** Mark `MetricArray.to_result` numpy path as deprecated; remove one release after default flip if no objections.

## Open questions / unverified

- **Number of list-valued RECORD metrics that today retain exact arrays without `MetricFlags.AGGREGATE_TDIGEST`** — only `inter_chunk_latency` is verified to escape exact retention. A grep for `BaseRecordMetric[list[...]]` would enumerate the rest; **unverified** in this doc, would need a mechanical sweep across `src/aiperf/metrics/types/`.
- **Whether the actual OOM was driven by allocator-HWM retention vs raw record retention** — inferred from the mismatch between back-of-envelope per-record cost (~5-100 MB) and a 1.5+ GB pod limit being hit at 64 min. Confirming would need either (a) an `objgraph` / `tracemalloc` snapshot from a reproducer, or (b) running the same workload with the checkpoint task disabled (`AIPERF_RECORD_CHECKPOINT_INTERVAL` raised to >3600) and seeing whether the OOM still happens. The fix above addresses both root causes regardless.
- **Whether `np.percentile` allocates a full sorted copy or operates partition-in-place** — inspected docs imply at least one full-size temp array; **unverified at code level** in numpy's C path.
