<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Exporters — Overhaul (v2 native report core + full native export plane)

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — the typed IO-free native-v2 report core and the full static
`Exporter` plane (`aiperf_runtime::export`, nine sinks behind one trait) are
implemented and are the **default sole emitter** on the native path.
**Grounding:** `rust/runtime/src/export/` (`mod.rs`, `genai_perf.rs`, `console_txt.rs`,
`timeslice.rs`, `parquet.rs`, `per_record_parquet.rs`, `parquet_util.rs`, `mlflow.rs`,
`wandb.rs`, `otel.rs`, `accuracy_csv.rs`, `server_metrics/`) + the typed report core
`rust/runtime/src/metrics_core/report.rs` and the runner commit site
`rust/runtime/src/report.rs`. Ported line-by-line from the legacy Python
`exporters/*` plane and `common/{constants,finite}.py` / `common/models/*`.
**Companion / parent:** `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`
(the `Reporter` seam + the typed `NativeReport` this consumes; the `MetricValue`
finite/absent discipline; the metric `type` = RECORD/AGGREGATE/DERIVED), the
telemetry + accuracy specs (the result types the report unifies).

---

## 0. Thesis — overhaul the mechanism AND break the format (v2 native default, v1 as a sink)

Two independent decisions:

1. **Mechanism (redo-cleaner):** the Python exporter plane was a thick
   multiprocess/plugin machine — a plugins.yaml registry, exception-as-disable
   constructors, `asyncio` fan-out, a fragment-glob-and-merge, subprocess uploaders.
   All of it was accidental complexity of the N-record-processor-processes model.
   Single-process Rust deletes it (§4): one typed `NativeReport` (from the metrics
   `Reporter`) → a **static set of `Exporter` impls behind one trait** with an
   explicit `enabled(cfg) -> bool`.
2. **Format (Tech Lead decision):** **break genai-perf as the default.** The
   genai-perf JSON/CSV format has real warts (below); the Rust tool ships a clean
   **v2 native report** as the typed core (§1), and keeps genai-perf **v1 as one
   `Exporter` sink** (`genai_perf`) so anyone depending on it isn't broken. Because
   both are just `Exporter` impls behind the same trait, v1 costs one translation
   function — no architectural weight.

**The v2 format is not invented from scratch — it generalizes the existing
server-metrics export format** (`server_metrics/{export_stats,json_exporter}.py`),
which already got this right: type-specific series models (Gauge/Counter/Histogram),
metrics-keyed-by-name, labeled `series[]`, per-type `timeslices`. §1 lifts that
shape to *every* metric (inference + telemetry + server).

The warts v2 fixes (the ones that drove this):
- **`avg` for scalars is a lie.** genai-perf puts a scalar metric's single value
  (throughput, duration, count) into the `avg` field and nulls
  min/max/percentiles/std. A scalar is not an average. v2 keys off metric **type**:
  distribution metrics get `avg/…/percentiles`; scalar metrics get a plain `value`.
- **Flat metric namespace on the JSON root**, mixed with `run_info`/`schema_version`,
  plus arbitrary injected top-level keys via `extra="allow"`. v2 uses a proper nested
  `metrics: {}`.
- **File sprawl** — 5+ files (`profile_export_aiperf.json` + `.csv` + `outputs.json` +
  `console.txt` + separate telemetry/server-metrics files). The v2 core is **one
  unified report** (+ the console `.txt`).
- (Also fixed for free by nesting: the three inconsistent CSV/JSON/console stat
  orderings; the flat `p1/p5/…/p99` keys → a structured `percentiles` map; missing
  per-metric metadata.)

The genai-perf v1 artifacts are still produced — as an explicit compat sink (§2) —
but they no longer *constrain* the native core.

---

## 1. The v2 native report — the type-specific-series model (adopted from the server-metrics design)

**The v2 report generalizes the server-metrics export format** to *every* metric.
That design never forces a metric into an `avg` shape it doesn't have; each metric
is a **type-specific `MetricData`** whose stats match its nature; and a metric is a
**set of labeled `series`**, not a flat scalar. We use it for inference metrics, GPU
telemetry, and server metrics uniformly, in one file.

**One report.** Metrics keyed by name (O(1) lookup); each metric type-tagged; each
holds a `series[]` (one per label-set); type-appropriate `stats` + per-type
`timeslices` at the leaf:

```jsonc
{
  "schema_version": "2.0",
  "aiperf_version": "…",
  "run": { /* RunInfo — shared with GET /api/run */ },
  "summary": { "start_time": …, "end_time": …, "duration_s": …, "was_cancelled": false,
               "endpoints_configured": […], "endpoints_successful": […] },

  "metrics": {                                        // metrics-first, keyed by name
    "request_latency": {
      "type": "distribution", "unit": "ms", "group": "default", "higher_is_better": false,
      "series": [                                     // inference metric ⇒ one unlabeled series
        { "labels": null,
          "stats": { "count": 1000, "avg": 42.1, "min": 8.0, "max": 910.4, "std": 33.7,
                     "percentiles": { "p50": 38.0, "p90": 71.2, "p95": 88.0, "p99": 210.5 } },
          "timeslices": [ { "start_ns": …, "end_ns": …, "complete": true,
                            "stats": { /* same distribution shape */ } } ] } ]
    },
    "request_throughput": {                            // rate ⇒ SCALAR stats (no `avg` lie)
      "type": "scalar", "unit": "requests/sec", "group": "default", "higher_is_better": true,
      "series": [ { "labels": null, "stats": { "value": 512.4 } } ]
    },
    "adj_request_latency": {                           // non-finite tail under failure
      "type": "distribution", "unit": "ms", "group": "default", "higher_is_better": false,
      "series": [ { "labels": null,
        "stats": { "count": 1000, "avg": null, "min": 8.0, "max": null, "std": null,
                   "percentiles": { "p50": 40.0, "p99": null } } } ]  // +inf → null
    },

    "vllm:num_requests_running": {                     // server GAUGE ⇒ many labeled series
      "type": "gauge", "unit": "requests", "group": "none",
      "series": [ { "labels": { "model": "…" }, "endpoint_url": "http://…/metrics",
                    "stats": { "avg": …, "min": …, "max": …, "std": …, "percentiles": {…} },
                    "timeslices": […] } ]
    },
    "vllm:generation_tokens": {                        // server COUNTER ⇒ total + rate
      "type": "counter", "unit": "tokens", "group": "none",
      "series": [ { "labels": {…}, "endpoint_url": "…",
                    "stats": { "total": 1.2e6, "rate": 20000.0,
                               "rate_avg": …, "rate_min": …, "rate_max": …, "rate_std": … },
                    "timeslices": [ { …, "total": …, "rate": … } ] } ]
    },
    "vllm:e2e_request_latency_seconds": {              // server HISTOGRAM ⇒ count/sum/rates/pct/buckets
      "type": "histogram", "unit": "s", "group": "none",
      "series": [ { "labels": {…}, "endpoint_url": "…",
                    "stats": { "count": …, "count_rate": …, "sum": …, "sum_rate": …, "avg": …,
                               "percentiles": {…} },     // polynomial-estimated
                    "buckets": { "0.1": …, "1.0": …, "+Inf": … },
                    "timeslices": […] } ]
    }
  },

  "warmup_metrics": { /* same map, or omitted when empty */ },
  "accuracy":  { /* accuracy summary + analyzer joins */ },
  "errors":    [ { "code": 429, "type": "RateLimit", "message": "…", "count": 12 } ],
  "per_record": [ /* optional: session_num, conversation_id, turn_index, x_request_id,
                     request_start_ns, request_end_ns, metrics{allowlist}, response_text */ ]
}
```

The four `stats` shapes, chosen by metric **type** (this is the whole point — no
`avg`-for-scalars):

| type | AIPerf source | `stats` shape |
|---|---|---|
| **distribution** | RECORD inference (latency/ISL/ITL); GAUGE telemetry | `{ count?, avg, min, max, std, percentiles{} }` |
| **scalar** | DERIVED (throughput, duration); AGGREGATE MIN/MAX (timestamps) | `{ value }` |
| **counter** | AGGREGATE-counter (request_count, total_*); server COUNTER | `{ total, rate, rate_avg/min/max/std? }` |
| **histogram** | server HISTOGRAM | `{ count, count_rate, sum, sum_rate, avg, percentiles{} }` + `buckets{}` |

Design rules (each a wart fixed, all inherited from the server-metrics design):

- **Type-specific stats, driven by metric type** — a counter reports `total`+`rate`
  (not a fake `avg`), a scalar reports `value`, a distribution reports
  `avg/…/percentiles`, a histogram reports count/sum/rates/percentiles/buckets. The
  `type` field tells the consumer which shape to read.
- **`series[]` per metric**, each with `labels` (null for unlabeled inference
  metrics; `{k:v}` for server/GPU dimensions) + optional `endpoint_url` — a metric is
  a set of labeled series, giving per-endpoint/per-model breakdowns for free and a
  natural home for future per-worker/per-model inference splits. Series sorted by
  `(endpoint_url, labels)`; metrics sorted by name (deterministic).
- **Nested `metrics: {}` keyed by name** — no flat root keys, no `extra="allow"`
  injection; run/summary metadata in separate namespaces; telemetry + server +
  inference all live here uniformly.
- **`percentiles` is a map** (`{p50, p90, p99, …}`) — one representation everywhere,
  extensible to `p999` with no schema change.
- **Per-metric metadata inline** — `unit`, `type`, `group`, `higher_is_better` —
  rank/plot without the registry.
- **`timeslices` per-series**, each a `{start_ns, end_ns, complete, stats}` with the
  space-efficient **`complete` three-state** (`true` complete / `false` partial),
  stats in the same type shape. Reset-detection + clamp are baked into the stats
  computation (a counter reset → clamp to 0 + a logged warning; a bucket reset → omit
  buckets), carried from `export_stats.py`.
- **Absent vs non-finite unambiguous** — a metric/stat with no value is **omitted**;
  a present-but-non-finite value is **`null`** (`+inf`/NaN scrubbed at the sink;
  `MetricValue = finite | absent` makes absent structural). So `null` means exactly
  "present but non-finite" (an `adj_*` tail).
- **One report** — inference metrics + GPU telemetry + server metrics + accuracy +
  per-record + run, all under one report.

The Rust `MetricEntry` is `{ type, unit, group, higher_is_better, series: Vec<Series> }`
where `Series { labels: Option<Labels>, endpoint_url: Option<Url>, stats: Stats,
timeslices: Vec<Slice> }` and `Stats` is an enum
`Scalar{value} | Distribution{…} | Counter{…} | Histogram{…}` — the type-tagged
shape serializes to exactly the leaf above.

**Built.** This IO-free native-v2 core lives in `aiperf_runtime::metrics_core::report`
(`rust/runtime/src/metrics_core/report.rs`): `NativeReporter` behind the `Reporter`
trait produces a typed `NativeReport` with run/summary/error, metric, series,
timeslice, and distribution/scalar/counter stats plus warmup and accuracy joins.
Metrics are name-keyed with typed series; per-metric metadata and per-series
timeslices are inline; structurally absent values are omitted and present-but-
non-finite adjusted tails encode as JSON `null`. A deterministic exact-JSON golden
pins the shape. The `NativeReport` is the single typed value every `Exporter`
consumes.

---

## 2. The genai-perf v1 compat sink (`genai_perf`) — reproduce the frozen contract

The `genai_perf` `Exporter` translates the `NativeReport` into the **exact** legacy
artifacts, for downstream tooling that still consumes them. It reproduces the frozen
v1 contract byte-for-byte (guarded by golden fixtures and verified **byte-identical**
to the legacy Python `metrics_json/csv_exporter` on live runs, empty `cmp` diff):

- **`profile_export_aiperf.json`**, `SCHEMA_VERSION = "1.4"`, `extra="allow"`, metrics
  as **flat top-level keys** on the root; each metric object in the frozen **JSON
  per-metric key order** (`unit, avg, p1, p5, p10, p25, p50, p75, p90, p95, p99, min,
  max, std, count, sum`); **scalars put their value in `avg`** (v1's lie, reproduced
  faithfully in the compat sink); `count` forced absent for AGGREGATE/DERIVED; the
  NaN/null rule (`exclude_none` + `scrub_non_finite`, never `model_dump_json`).
- **`profile_export_aiperf.csv`**, the frozen **CSV `STAT_KEYS`** order (`avg, min,
  max, sum, p1, p5, p10, p25, p50, p75, p90, p95, p99, std`), the two-section
  `Metric,<stats>` + `Metric,Value` layout (split by has-percentiles), the
  GPU-telemetry section, `None → ""`.
- The frontend projects `MetricRegistry` header/filter/scalar maps +
  `input_config`/`run_info` into `ExportConfig`; Rust assembles in exact
  `JsonExportData` order.

The v1 sink is a *translation from `NativeReport`*, so all the v1
field-name/ordering/NaN details are its acceptance contract; they no longer constrain
the v2 core.

Per-record outputs (`outputs.json` / per-record JSONL) are built from the in-RAM
per-record list, **not** a fragment glob (§4). The wide columnar per-record sidecars
(`profile_export.parquet` / `profile_export_records.csv`) are a **runner-owned
artifact, not an `Exporter` over `NativeReport`** — the per-record data lives only at
the `CapturedRecord` callsites, so `engine/records.rs::write_records_{parquet,csv}`
drives the writers directly (parquet via `export::per_record_parquet`, gated on the
`parquet` feature; CSV is stdlib and always available). See `export/mod.rs` for the
explicit note that `ParquetExporter` (server-metrics parquet) and the per-record
parquet writer are distinct.

---

## 3. What stays earned-in-blood in BOTH formats — built in the console sink

The warning/insight intelligence and the console record-then-replay behavior are
built natively in the `console_txt` `Exporter` (`export/console_txt.rs`), byte-exact
in geometry and detector text vs the legacy Python:

- **OSL-mismatch warning** (`osl_mismatch_count.avg > 0`; threshold `min(requested·5%,
  50 tokens)`) with the fix-text verbatim (`--extra-inputs ignore_eos:true` /
  `min_tokens:<N>` per backend, `--use-server-token-count`, the `osl_mismatch_diff_pct`
  + `AIPERF_METRICS_*` pointers).
- **Usage-discrepancy warning** (`usage_discrepancy_count.avg > 0`; threshold 10%)
  with the tokenizer-mismatch causes + `usage_*_diff_pct` pointer +
  `--use-server-token-count`.
- **API-error insights** (`ErrorInsight { problem, causes, investigation, fixes }`) —
  both detectors verbatim: **MaxCompletionTokens** (`extra_forbidden` +
  `max_completion_tokens` → `--use-legacy-max-tokens`) and **DynamoSessionControl**
  (serde `unknown variant` + `bind` → Dynamo `session_control action='bind'`
  unsupported pre-v1.3.0-dev, commit d97c889ba; upgrade /
  `--use-legacy-dynamo-session-control` / disable `--use-dynamo-conv-aware-routing`).
  This version lore is irreplaceable and is ported exactly.
- **Error-summary table** (Code / Type / Message / Count, `N/A` handling, grouped
  counts).
- The **INTERNAL/EXPERIMENTAL filter** lives at the sink (both formats drop them
  unless dev-mode).

The console table renders **twice**: once to a **fixed-width** buffer
(`CONSOLE_EXPORT_WIDTH`) for the width-pinned `profile_export_console.txt` (stable
CI-log artifact, decoupled from terminal width), then to the **live terminal** if
attached (replay the recorded fixed-width text if non-tty). Content grouping/headers
are projected from Python's console metadata to match the `MetricRegistry` view.

---

## 4. Accidental complexity DELETED (mechanism — independent of format)

1. **Plugin registry for exporters** (`plugins.iter_all` + plugins.yaml
   `data_exporter:`/`console_exporter:`) → a static Rust registry
   (`ExporterRegistry::with_builtin_exporters`). Alphabetical YAML order was
   meaningless; sinks carry an explicit ascending `order` key.
2. **Exception-as-disable** (`DataExporterDisabled`/`ConsoleExporterDisabled` from
   `__init__`) → an explicit `Exporter::enabled(&ExportConfig) -> bool`.
3. **`asyncio.Task` fan-out + `gather`** over synchronous file writes → a sequential
   loop; short-lived timed runtimes only for network uploaders.
4. **`outputs_json` fragment glob-merge-cleanup** (`output_fragments/*.jsonl` + the
   `session_num:turn_index` cross-file join + rmdir) — a multi-process shard artifact.
   Accumulate in RAM, write once.
5. **The two-file join** (`profile_export.jsonl` metrics-map re-read) — in-process the
   metrics are already on each record.
6. **Double-instantiation** for path metadata → paths as data (`FileExportInfo`).
7. **`SERVER_METRICS_PARQUET` cross-process skip TODO** → gone in-process.
8. **`is_deferred` getattr probe** → encode the local-writers→uploaders order
   structurally via the `order` key (§5).
9. **The mlflow/wandb `spawn`/`mp.Queue`/`repr(exc)`/exitcode apparatus** → a
   short-lived `current_thread` runtime under `tokio::time::timeout` (§6).

---

## 5. The redesign — one `NativeReport` → static `Exporter`s (built)

**Built.** A thin sink layer above `aiperf_runtime::metrics_core` (it does
file/console/network IO). Every output format/destination is an object-safe
`Exporter` behind one trait, registered in an `ExporterRegistry`
(`rust/runtime/src/export/mod.rs`) with an ascending `order` key (ties break on
`name`):

```rust
pub trait Exporter {
    fn name(&self) -> &'static str;
    fn enabled(&self, cfg: &ExportConfig) -> bool;     // replaces exception-as-disable
    fn file_info(&self) -> Option<FileExportInfo>;      // path as DATA
    fn export(&self, report: &NativeReport, artifact_dir: &Path, cfg: &ExportConfig) -> Result<()>;
}
```

`ExporterRegistry::with_builtin_exporters()` registers the **nine builtin sinks**
(`register_builtins`), filtered by `enabled(cfg)` and run in `order`:

| order band | sink | output |
|---|---|---|
| file writer | `genai_perf` | `profile_export_aiperf.{json,csv}` (schema 1.4, §2) |
| file writer | `server_metrics` | `server_metrics.{json,csv}` |
| file writer | `timeslice` | timeslice JSON |
| file writer | `accuracy_csv` | `accuracy_results.csv` |
| file writer +4 | `parquet` | `server_metrics_export.parquet` |
| console | `console_txt` | `profile_export_console.txt` (§3) |
| uploader | `otel` | OTLP/HTTP GenAI-semconv metrics |
| uploader +1 | `mlflow` | REST + `file://` FileStore |
| uploader +2 | `wandb` | offline `.wandb` transaction log |

`ExporterRegistry::run(report, artifact_dir, cfg)` iterates in emit order — local
writers, then the console sink, then the timed uploaders — logging each failure and
continuing (**best-effort**: one sink's failure never aborts the others). The
convenience `run_exporters(...)` composes `with_builtin_exporters().run(...)`. The
runner calls this at its report-commit site (`coordinator::persist_prepared_report`).
`ExportConfig` carries a per-sink sub-config
(`genai_perf`/`otel`/`mlflow`/`server_metrics`/`timeslice`/`accuracy_csv`/`console_txt`/`wandb`/`parquet`),
projected from the frontend via `rust_wire._export`. Serialization is `serde` with
fields in fixed order + `skip_serializing_if = Option::is_none` for absent + a
non-finite→`null` scrub.

**Shared exporter helpers** (`export/mod.rs`, `export/parquet_util.rs`) keep the sinks
DRY:
- `finite_guarded` / `finite_passthrough` — the finite-value guards (guarded scrub vs
  trusted passthrough).
- `summary_series` — the summary-series selection (`native_report._summary_series`:
  the sole series a scalar/inference metric exposes).
- `crlf_csv_writer` — the CRLF `csv::Writer` used by every CSV sink.
- `normalize_endpoint_display` — unified endpoint-display normalization (strips query/
  fragment), ported from `exporters/utils.py::normalize_endpoint_display`.
- `parquet_util::{string_column, float_column, writer_properties, write_parquet_table}`
  — shared by the two parquet sinks (`ParquetExporter` and `per_record_parquet`).

---

## 6. Uploaders (mlflow / wandb / otel) — timed task, not a subprocess (built)

Python used a subprocess for a hard timeout over uncancellable blocking SDK network
I/O + fork-state isolation. Carried forward is only the **durable requirement: a hard
wall-clock timeout so an unreachable tracking server can't hang shutdown**. In Rust
each uploader runs under a short-lived `current_thread` runtime + `tokio::time::timeout`
— no `spawn`/`Queue`/pickle/exitcode apparatus.

- **`otel`** — OTLP/HTTP GenAI-semconv metrics. Per-record histograms with populated
  `bucket_counts` from a runner per-record accumulator (`OtelRecordAccumulator`);
  count/sum match the native-v2 report exactly (decoded via `opentelemetry-proto`).
  Residual: a single terminal cumulative export vs Python's periodic (final aggregate
  identical); `aiperf.*` per-record attribute fragmentation omitted; online-scheduled
  + profiling-phase only.
- **`mlflow`** — REST + `file://` FileStore; live filestore matches the
  `metric.tag[.stat]` scheme.
- **`wandb`** — offline `.wandb` transaction log (leveldb framing + protobuf),
  SDK-decodable.

---

## 7. Scope + testing

- **Built (Rust, default sole emitter):** the typed IO-free native-v2 `NativeReport`
  core in `aiperf_runtime::metrics_core::report`; the single native-v2 JSON writer in
  `aiperf_runtime::report`; and the full static `Exporter` plane
  (`aiperf_runtime::export`, nine sinks) wired at
  `coordinator::persist_prepared_report`.
- **Default cutover (zero-Python):** `Environment.RUNTIME.NATIVE_EXPORT` (default
  true) makes the native plane authoritative; `export_python_compatibility_reports` is
  a no-op on the native path. `AIPERF_RUNTIME_NATIVE_EXPORT=0` restores the legacy
  Python emitters for A/B (mirroring `AIPERF_RUNTIME_ENGINE=python`). The Python
  exporter modules are retained only for that legacy path and unit tests, not invoked
  natively. Live-verified: a default run emits the native compat files with zero
  `ExporterManager` invocation.
- **Parity evidence (all live-run, byte-diffed):**
  - `genai_perf`, `timeslice`, `server_metrics` (json/csv) — **byte-identical** to the
    legacy Python (server-metrics driven against a live Prometheus endpoint).
  - `parquet` — `server_metrics_export.parquet` **schema+data equal** (`pyarrow
    Table.equals`); byte-identity not targeted (arrow-rs vs pyarrow encoding
    defaults). Consumes the runner's `.aiperf-server-metrics-parquet-wire.jsonl` and
    deletes it after (the Python round-trip is retired).
  - `accuracy_csv` — **byte-equal** (offline oracle; live enable→emit path confirmed).
  - `console_txt` — **byte-exact geometry**; warning/insight detectors + error table
    **byte-exact**.
  - `otel`/`mlflow`/`wandb` — decoded and matched as in §6.
  - Parity harness: `AIPERF_EXPORT_SUBDIR=<dir>` redirects the native sinks under
    `<artifact_dir>/<dir>/` so they coexist with the legacy Python files for a
    same-`native-v2.json` byte-diff.
- **Testing golden gates:** a fixed `NativeReport` → the exact native-v2 JSON (nested
  `metrics`, scalar `value` vs distribution stats by type, `percentiles` map,
  per-metric metadata, absent-omitted/non-finite-null) plus per-sink byte-parity
  goldens (`export/golden/`); a field/order drift in any fails its gate.
- **Deferred / not native-owned:** aggregate/sweep exporters (outer-loop coordinator;
  the `aiperf-cli` sweep aggregate is native there), dashboards. The server-HISTOGRAM
  leaf in the v2 core carries the histogram shape; the v2 **native CSV** projection of
  the whole report (flat one-row-per-`(metric, series)`) is not yet a distinct sink —
  the `genai_perf` CSV covers the summary-CSV consumer today.

## 8. Open questions (resolved)

1. **v2 filename + `schema_version`** — the typed core carries `schema_version: "2.0"`
   (`NATIVE_REPORT_SCHEMA_VERSION`); the runner commits it as `native-v2.json`.
2. **v2 default on, v1 as a sink** — shipped: v2 core is authoritative, `genai_perf`
   v1 artifacts always emit as a compat sink (no `--export-genai-perf` gate needed —
   downstream tooling still depends on them).
3. **Per-metric `type` vocabulary** — emit the *shape* the consumer reads
   (`distribution` / `scalar` / `counter` / `histogram`), not the internal
   `record`/`aggregate`/`derived` compute detail; mapping in the §1 table. Inference
   series emit `labels:null` until a per-model/per-worker breakdown is wired (the
   `series[]` structure supports it for free).
4. **Crate extraction** — the writer stays a runner-facing module
   (`aiperf_runtime::report` + `aiperf_runtime::export`); no separate `aiperf-report`
   crate was needed.
