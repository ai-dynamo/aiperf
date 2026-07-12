# PR 1036: native AIPerf metrics versus Tachometer

Status: complete source comparison with runtime validation; GitHub comments drafted but not posted.

## Scope and revisions

- Native AIPerf baseline: `19b07fcf587d480fb58f7bbb23d16ccc8170ec08` in this workspace. The shared worktree had unrelated user changes, which this review left untouched.
- Tachometer: PR [ai-dynamo/aiperf#1036](https://github.com/ai-dynamo/aiperf/pull/1036), head `2ea84c88984ac1761d1f7f324570ca9089cb4ec5`, checked out at `/home/anthony/tmp/aiperf-watch`.
- PR merge base: `35d6e5a3c1b1e5b16b1661e9426fb67ea828d3b6`.
- Full revision receipt: `artifacts/repro-runtime-20260711/revisions.txt`.

## Bottom line

Tachometer is not an alternative implementation of our metrics pipeline. It is an independent, long-running Prometheus snapshot recorder: scrape arbitrary endpoints, optionally rename/filter series, buffer rows, and compact Arrow/Parquet to local or S3 storage. That is useful operational telemetry functionality which our benchmark-owned pipeline does not try to replace.

Our native pipeline is the authoritative benchmark measurement engine. It correlates transport-neutral request events, token timing, endpoint usage, HTTP traces, GPU telemetry, server Prometheus data, and network calibration on the benchmark Clock; derives record/aggregate/SLO/sweep metrics; applies exact warmup/profiling boundaries; and emits a typed native-v2 report. Tachometer has none of the request, phase, counter-delta, semantic aggregation, SLO, or cross-domain join layers.

PR 1036 should therefore be evaluated as a new raw telemetry product, not as a replacement for native AIPerf metrics. In its current form I would not merge the recorder: four paths silently corrupt or duplicate stored data, and non-2xx HTTP responses are accepted as successful scrapes.

## End-to-end pipeline comparison

| Stage | Native AIPerf | Tachometer PR 1036 |
|---|---|---|
| Lifecycle | Owned by one benchmark run and its warmup/profiling phase barriers. | Independent `aiperf watch` process, runs until SIGINT/SIGTERM. |
| Inputs | `RequestObserver` arrival/admit/classified-token/usage/terminal events; HTTP trace; server Prometheus; DCGM/Python GPU telemetry; TCP RTT probes. | Prometheus text from HTTP or `file://` endpoints only. |
| Time authority | One injected `Clock` and integer-ns request/phase timestamps (`loadgen-core/src/sink.rs:63-104`, `aiperf-server-metrics/src/source.rs:129-160`). | Tokio `Instant`/`sleep`; rows receive seconds since writer creation (`tachometer-writer/src/writer.rs:185-211`). No phase or benchmark clock. |
| Scrape cadence | Clock-paced sidecars plus forced start/end boundary scrapes (`aiperf-runner/src/server_metrics.rs:236-368`, `gpu_telemetry.rs:248-373`). | Scrape completion followed by a full interval sleep, so effective period is scrape latency plus `1/frequency` (`tachometer-scraper/src/runner.rs:600-618`). |
| Prometheus model | Typed families with structured, escaped labels and per-label-set histograms in `f64` (`aiperf-server-metrics/src/parser.rs:188-240,381-516`). | Custom two-pass parser; labels are re-encoded into metric-name strings; malformed lines are silently skipped (`tachometer-scraper/src/parse.rs:28-65,400-425`). |
| Request facts | Correlation/session/turn/model/endpoint, ns timestamps, token arrivals, errors/cancellation, usage, modality, HTTP phases (`aiperf-metrics/src/ingest.rs:113-164`). | None. |
| Storage | Append-only `f64` sparse columns, categorical dimensions, exact ragged per-token replay; worker-local merge (`aiperf-metrics/src/store.rs:21-27,146-170,348-420`). | Shared `Arc<Mutex>` row buffer; Float32 metric/histogram columns; periodic full-buffer Arrow checkpoint; threshold Parquet files (`tachometer-writer/src/writer.rs:14-47,108-149,210-320`). |
| Metric semantics | Validated 119-entry catalog (103 source identities + 16 native sweep identities), record formulas, aggregates, derived rates, SLO goodput, error-adjusted tails (`aiperf-metrics/src/catalog.rs:2081-2092`, `accumulator.rs:426-515,582-949`). | Raw snapshots plus four filter families. Node CPU counters are pre-aggregated across CPUs; no general gauge/counter/rate semantics after storage. |
| Histograms/counters | Exact phase boundary deltas, reset clamping, gauge distributions/timeslices, histogram percentile estimation, backend atlas (`aiperf-server-metrics/src/accumulator.rs:179-301`, `histogram.rs:102-259`). | Cumulative bucket rows with copied sum/count. No boundary delta, reset handling, rate, or percentile calculation after scrape. |
| Load curves | Exact ICL-aware concurrency, prefill/decode throughput, active/effective curves, duration-weighted stats (`aiperf-metrics/src/sweepline/mod.rs:154-220,348-446`). | None. |
| Cross-domain joins | GPU power/energy, tokens/J, energy/user, RTT-adjusted request/TTFT, accuracy/performance joins (`aiperf-gpu-telemetry/src/accumulator.rs:292-350`, `aiperf-metrics/src/accumulator.rs:906-949`). | None; endpoint metadata becomes extra string columns. |
| Windowing | Authoritative phase masks, half-open time windows, per-model/endpoint series, non-empty timeslices (`aiperf-metrics/src/accumulator.rs:245-300,483-570,1009-1075`). | Relative scrape timestamp only; final sort by metric-name string and time. |
| Output | Typed native-v2 scalar/counter/distribution/histogram series, units, labels, timeslices, run metadata; compatibility raw/record/server/GPU artifacts (`aiperf-metrics/src/report.rs:22-164`). | One flat `final.parquet`, local or S3, plus intermediate Arrow/Parquet files. |
| Extension seams | `RequestSink`, `RequestObserver`, `Accumulator`, `ListMetricBackend`, `MetricsTextParser`, `ServerMetricAtlas`, GPU source/decoder, network probe, reporter. | `MetricFilter` is the only domain seam; HTTP client, clock, row schema, writer, and compactor are concrete. |

### What Tachometer does better

- It can watch arbitrary Prometheus endpoints outside a benchmark and retain queryable raw history for a long-running operational workflow.
- It has a direct local/S3 Parquet archive and periodic durability mechanism.
- Per-endpoint frequencies and simple deployment metadata filters are convenient for fleet observation.

### What should remain authoritative

- Native AIPerf must remain authoritative for all benchmark metrics and telemetry attribution. Replacing it with Tachometer would discard per-request identity, token timing, phase boundaries, counter deltas, units, SLO/goodput, sweep lines, accuracy joins, and deterministic `{transport, clock}` parity.
- If `watch` is wanted, keep it explicitly separate as a raw operational recorder. Before sharing code, first make its parser/schema lossless (`f64`, structured labels, semantic types), inject an HTTP client/clock, and expose benchmark boundary hooks. Our existing `PrometheusTextParser` and server/GPU accumulators are the stronger semantic substrate.

## Confirmed findings

### F1 — P1 — histogram label sets contaminate one another

- Status: **Confirmed**.
- Source evidence: `rust/tachometer-scraper/src/parse.rs:526-590`. Histogram stats are keyed by metric plus non-`le` labels, but bucket rows are grouped only by the metric-name prefix at lines 535-548. Lines 568-589 derive one stats key from the first bucket and apply its bounds/sum/count to every label set in that family.
- Impact: any histogram family with two label sets stores incorrect lower bounds and assigns the first series' sum/count to the others. Downstream rates and quantiles cannot be repaired from this artifact.
- Runtime: the real PR binary parsed `route=a` (`sum=1.5,count=3`) and `route=b` (`sum=70,count=10`); every `route=b` bucket was stored with `sum=1.5,count=3`, and its first bucket lower bound became `1` instead of `0`.
- Receipts: `artifacts/repro-runtime-20260711/histogram-multiple-labelsets.prom`, `runtime-results.txt`; generated Parquet at `/tmp/aiperf-watch-hist-intermediate-20260711/final.parquet`.
- Conclusion: group and sort buckets by the same `(family, labels_without_le)` identity used for histogram stats.

### F2 — P1 — a stale Arrow checkpoint duplicates rows during final compaction

- Status: **Confirmed**.
- Source evidence: periodic saves overwrite `current.arrow` with the current in-memory buffer (`rust/tachometer-writer/src/writer.rs:290-320`). A threshold flush persists that buffer to `out-N.parquet` and clears memory but does not invalidate `current.arrow` (`writer.rs:246-258`). If shutdown sees an empty buffer it leaves the old checkpoint untouched (`writer.rs:277-284`), and final compaction concatenates both sources (`compaction.rs:508-527`).
- Impact: stopping after a threshold flush duplicates an already persisted prefix. Counts, rates, and every query over the final dataset can be wrong.
- Runtime: two real scrapes produced `out-1.parquet` with two rows plus a one-row stale checkpoint. `final.parquet` contained three rows; the first timestamp appeared twice.
- Receipts: `artifacts/repro-runtime-20260711/checkpoint-duplication.prom`, `runtime-results.txt`; generated Parquet at `/tmp/aiperf-watch-dup-intermediate-20260711/final.parquet`.
- Conclusion: checkpoint identity must track the current buffer generation. Remove/replace it atomically when its rows enter a numbered Parquet file, or version committed offsets and exclude superseded checkpoints during compaction.

### F3 — P1 — Float32 storage destroys Prometheus counter precision

- Status: **Confirmed**.
- Source evidence: samples are parsed as `f64` (`rust/tachometer-scraper/src/parse.rs:378-395`) and then cast to `f32` for values, bounds, sums, and counts (`parse.rs:159-210,233-281`). `Row` and the Arrow schema permanently use Float32 (`rust/tachometer-writer/src/writer.rs:14-23,115-141`).
- Impact: Float32 cannot represent every integer above 16,777,216. Common request/token/energy counters lose small increments, so downstream deltas and rates become incorrect; histogram counts and sums are affected too.
- Runtime: source value `100000001` was stored with Arrow type Float32 as `100000000.0`.
- Receipts: `artifacts/repro-runtime-20260711/f32-precision.prom`, `runtime-results.txt`; generated Parquet at `/tmp/aiperf-watch-f32-intermediate-20260711/final.parquet`.
- Conclusion: keep all Prometheus numeric fields as Float64 end to end.

### F4 — P2 — valid commas in quoted label values are silently truncated

- Status: **Confirmed**.
- Source evidence: `rust/tachometer-scraper/src/parse.rs:400-425` splits the label block on every comma without tracking quotes or escapes; label formatting at lines 499-507 also does not escape values.
- Impact: valid Prometheus labels such as error messages, routes, or argument lists change identity. Separate time series can collapse, and the original label cannot be recovered from Parquet.
- Runtime: `message="left,right"` was stored as `message="left"`; the `right` fragment disappeared without an error.
- Receipts: `artifacts/repro-runtime-20260711/escaped-label.prom`, `runtime-results.txt`; generated Parquet at `/tmp/aiperf-watch-label-intermediate-20260711/final.parquet`.
- Conclusion: use a Prometheus/OpenMetrics parser or a quote/escape-aware state machine, and escape labels when serializing them.

### F5 — P1 — non-2xx HTTP bodies are archived as successful metrics

- Status: **Confirmed**.
- Source evidence: `rust/tachometer-scraper/src/lib.rs:25-31` calls `reqwest::get` and immediately reads the body without `error_for_status` or an explicit status check. By contrast, the native source rejects statuses outside 200-299 at `crates/aiperf-server-metrics/src/source.rs:175-190`.
- Impact: endpoint failures are invisible to the watch process. A 500 response containing stale or diagnostic Prometheus text becomes ordinary data; an HTML error body is silently reduced to an empty successful scrape because parser errors are skipped.
- Runtime: a temporary server returned HTTP 500 with `error_metric 42`; Tachometer wrote `error_metric=42.0` to `final.parquet` without logging a scrape error.
- Receipts: `artifacts/repro-runtime-20260711/runtime-results.txt`; generated Parquet at `/tmp/aiperf-watch-status-intermediate-20260711/final.parquet`.
- Conclusion: validate the status before reading/parsing the body and configure a reusable client with connection/request timeouts.

## Additional gaps, not filed as blocking findings

- Unknown filter names silently select `NoOpFilter` (`tachometer-scraper/src/filters.rs:383-399`) instead of failing configuration validation.
- The `file://` branch performs blocking filesystem I/O inside async code (`tachometer-scraper/src/lib.rs:19-23`).
- The compactor's reported row count is always the one-row aggregation-frame length rather than the count value (`tachometer-writer/src/compaction.rs:212-216`); this is misleading logging, not stored-data corruption.
- The filter layer intentionally collapses labels and DCGM/node metric names. That may suit a specific dashboard schema, but it prevents Tachometer's output from being called a lossless Prometheus archive.

## Validation

- PR Rust workspace: **43 passed**, no failures.
- Native metrics/server/GPU/network crates: **124 passed**. Two server-source tests initially hit sandbox listener restrictions, then passed outside the sandbox.
- PR Python wrapper tests: blocked during shared conftest import by missing test-environment packages before the watch tests collected; see `test-results.txt`.
- Full receipts: `artifacts/repro-runtime-20260711/`.

## Planned GitHub review (not posted)

PR files are entirely added in one hunk, so the proposed diff positions equal their new-file line numbers.

1. `rust/tachometer-scraper/src/parse.rs`, line/position 548 — F1 histogram label-set contamination.
2. `rust/tachometer-writer/src/writer.rs`, line/position 279 — F2 stale checkpoint duplication.
3. `rust/tachometer-writer/src/writer.rs`, line/position 18 — F3 Float32 precision loss.
4. `rust/tachometer-scraper/src/parse.rs`, line/position 417 — F4 quoted-label parsing.
5. `rust/tachometer-scraper/src/lib.rs`, line/position 28 — F5 HTTP status validation.

The full proposed summary and inline bodies are in `artifacts/planned-github-review.md`. They will be shown to the user for approval before any GitHub API write.
