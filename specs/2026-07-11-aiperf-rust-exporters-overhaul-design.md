<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Exporters — Overhaul (v2 native report + genai-perf v1 compat sink)

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** partially built — the typed, IO-free native-v2 report core and the runner's JSON writer
are implemented; native CSV, the genai-perf-v1 compat sink, warning/insight and error-table
renderers, console record/replay, and timed uploaders remain unbuilt in Rust (the Python `aiperf`
parent owns human presentation and compatibility/export generation).
**Grounding:** line-by-line read of `exporters/{metrics_json,metrics_csv,console_metrics,metrics_base}_exporter.py`,
`exporters/{protocols,exporter_config,exporter_manager,outputs_json_exporter}.py`, the four
console-warning exporters, the mlflow/wandb subprocess uploaders, plus
`common/{constants,finite}.py` and `common/models/{export_models,record_models}.py`.
**Companion / parent:** `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`
(the `Reporter` seam + the typed `Report` this consumes; the `MetricValue` finite/absent
discipline; the metric `type` = RECORD/AGGREGATE/DERIVED), the telemetry + accuracy specs
(the result types the report unifies).

---

## 0. Thesis — overhaul the mechanism AND break the format (v2 native default, v1 opt-in)

Two independent decisions:

1. **Mechanism (redo-cleaner):** the Python exporter plane is a thick multiprocess/plugin
   machine — a plugins.yaml registry, exception-as-disable constructors, `asyncio` fan-out, a
   fragment-glob-and-merge, subprocess uploaders. All of it is accidental complexity of the
   N-record-processor-processes model. Single-process Rust deletes it (§4): one typed `Report`
   (from the metrics `Reporter`) → a **static set of `Exporter` impls behind one trait** with an
   explicit `enabled(cfg) -> bool`.
2. **Format (Tech Lead decision):** **break genai-perf as the default.** The genai-perf JSON/CSV
   format has real warts (below); the Rust tool ships a clean **v2 native report** as the primary
   output (§1), and keeps genai-perf **v1 as an opt-in compat sink** (§2, e.g. `--export-genai-perf`)
   so anyone depending on it isn't broken. Because both are just `Exporter` impls behind the same
   trait, v1 costs one translation function — no architectural weight.

**The v2 format is not invented from scratch — it generalizes the existing server-metrics export
design** (`server_metrics/{export_stats,json_exporter}.py`), which already got this right:
type-specific series models (Gauge/Counter/Histogram), metrics-keyed-by-name, labeled `series[]`,
per-type `timeslices`. §1 lifts that shape to *every* metric (inference + telemetry + server).

The warts v2 fixes (the ones that drove this):
- **`avg` for scalars is a lie.** genai-perf puts a scalar metric's single value (throughput,
  duration, count) into the `avg` field and nulls min/max/percentiles/std. A scalar is not an
  average. v2 keys off metric **type**: distribution metrics get `avg/…/percentiles`; scalar
  metrics get a plain `value`.
- **Flat metric namespace on the JSON root**, mixed with `run_info`/`schema_version`, plus
  arbitrary injected top-level keys via `extra="allow"`. v2 uses a proper nested `metrics: {}`.
- **File sprawl** — 5+ files (`profile_export_aiperf.json` + `.csv` + `outputs.json` + `console.txt`
  + separate telemetry/server-metrics files). v2 is **one unified report** (+ the console `.txt`).
- (Also fixed for free by nesting: the three inconsistent CSV/JSON/console stat orderings; the
  flat `p1/p5/…/p99` keys → a structured `percentiles` map; missing per-metric metadata.)

---

## 1. The v2 native report — the type-specific-series model (adopted from the server-metrics design)

**The v2 report generalizes the server-metrics export format** (`server_metrics/{export_stats,
json_exporter}.py` — the "hybrid, metrics-first-keyed, type-specific, labeled-series" shape) to
*every* metric. That design already solves the warts: it never forces a metric into an `avg`
shape it doesn't have; each metric is a **type-specific `MetricData`** whose stats match its
nature; and a metric is a **set of labeled `series`**, not a flat scalar. We use it for inference
metrics, GPU telemetry, and server metrics uniformly, in one file.

**One file, `aiperf_report.json`.** Metrics keyed by name (O(1) lookup); each metric type-tagged;
each holds a `series[]` (one per label-set); type-appropriate `stats` + per-type `timeslices` at
the leaf:

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
  "per_record": [ /* optional (--export-per-record): session_num, conversation_id, turn_index,
                     x_request_id, request_start_ns, request_end_ns, metrics{allowlist}, response_text */ ]
}
```

The four `stats` shapes, chosen by metric **type** (this is the whole point — no `avg`-for-scalars):

| type | AIPerf source | `stats` shape |
|---|---|---|
| **distribution** | RECORD inference (latency/ISL/ITL); GAUGE telemetry | `{ count?, avg, min, max, std, percentiles{} }` |
| **scalar** | DERIVED (throughput, duration); AGGREGATE MIN/MAX (timestamps) | `{ value }` |
| **counter** | AGGREGATE-counter (request_count, total_*); server COUNTER | `{ total, rate, rate_avg/min/max/std? }` |
| **histogram** | server HISTOGRAM | `{ count, count_rate, sum, sum_rate, avg, percentiles{} }` + `buckets{}` |

Design rules (each a wart fixed, all inherited from the server-metrics design):

- **Type-specific stats, driven by metric type** — a counter reports `total`+`rate` (not a fake
  `avg`), a scalar reports `value`, a distribution reports `avg/…/percentiles`, a histogram reports
  count/sum/rates/percentiles/buckets. The `type` field tells the consumer which shape to read.
- **`series[]` per metric**, each with `labels` (null for unlabeled inference metrics; `{k:v}` for
  server/GPU dimensions) + optional `endpoint_url` — a metric is a set of labeled series, giving
  per-endpoint/per-model breakdowns for free and a natural home for future per-worker/per-model
  inference splits. Series sorted by `(endpoint_url, labels)`; metrics sorted by name (deterministic).
- **Nested `metrics: {}` keyed by name** — no flat root keys, no `extra="allow"` injection; run/
  summary metadata in separate namespaces; telemetry + server + inference all live here uniformly.
- **`percentiles` is a map** (`{p50, p90, p99, …}`) — one representation everywhere, extensible to
  `p999` with no schema change.
- **Per-metric metadata inline** — `unit`, `type`, `group`, `higher_is_better` — rank/plot without
  the registry.
- **`timeslices` per-series**, each a `{start_ns, end_ns, complete, stats}` with the space-efficient
  **`complete` three-state** (`true` complete / `false` partial; the Python `is_complete` uses
  `None`=complete for space — in v2 use an explicit `complete: bool`), stats in the same type shape.
  Reset-detection + clamp are baked into the stats computation (a counter reset → clamp to 0 + a
  logged warning; a bucket reset → omit buckets), carried from `export_stats.py`.
- **Absent vs non-finite unambiguous** — a metric/stat with no value is **omitted**; a present-but-
  non-finite value is **`null`** (`+inf`/NaN scrubbed at the sink; `MetricValue = finite | absent`
  makes absent structural). So `null` means exactly "present but non-finite" (an `adj_*` tail).
- **One file** — inference metrics + GPU telemetry + server metrics + accuracy + per-record + run,
  all under one report. A v2 **CSV** is a flat projection (one row per `(metric, series)`, columns
  by type) — the JSON is the source of truth.

The Rust `MetricEntry` is `{ type, unit, group, higher_is_better, series: Vec<Series> }` where
`Series { labels: Option<Labels>, endpoint_url: Option<Url>, stats: Stats, timeslices: Vec<Slice> }`
and `Stats` is an enum `Scalar{value} | Distribution{…} | Counter{…} | Histogram{…}` — the
type-tagged shape serializes to exactly the leaf above.

**Built.** This IO-free native-v2 core lives in `aiperf_runtime::metrics_core::report` (`rust/runtime/src/
metrics_core/report.rs`): `NativeReporter` behind the `Reporter` trait produces a typed
`NativeReport` with run/summary/error, metric, series, timeslice, and distribution/scalar/counter
stats plus warmup and accuracy joins. Metrics are name-keyed with typed series; per-metric metadata
and per-series timeslices are inline; structurally absent values are omitted and present-but-non-
finite adjusted tails encode as JSON `null`. A deterministic exact-JSON golden pins the shape. The
server-HISTOGRAM leaf and a native CSV projection remain unbuilt.

---

## 2. The genai-perf v1 compat sink (opt-in — reproduce the frozen contract)

**Unbuilt in Rust — owned by the Python parent.** The runner emits only the native-v2 report; the
Python `aiperf` frontend validates it and generates genai-perf-v1 compatibility artifacts. The
frozen v1 contract below is recorded as the acceptance target for whichever layer produces it, and
does not constrain the Rust native-v2 default.

An opt-in `Exporter` (`--export-genai-perf`) that translates the same `Report` into the **exact**
legacy artifacts, for downstream tooling that still consumes them. It must reproduce the frozen v1
contract byte-for-byte (guarded by golden fixtures):

- **`profile_export_aiperf.json`**, `SCHEMA_VERSION = "1.4"`, `extra="allow"`, metrics as **flat
  top-level keys** on the root; each metric object in the frozen **JSON per-metric key order**
  (`unit, avg, p1, p5, p10, p25, p50, p75, p90, p95, p99, min, max, std, count, sum`); **scalars
  put their value in `avg`** (v1's lie, reproduced faithfully in the compat sink); `count` forced
  absent for AGGREGATE/DERIVED; the NaN/null rule (`exclude_none` + `scrub_non_finite`, never
  `model_dump_json`).
- **`profile_export_aiperf.csv`**, the frozen **CSV `STAT_KEYS`** order (`avg, min, max, sum, p1,
  p5, p10, p25, p50, p75, p90, p95, p99, std`), the two-section `Metric,<stats>` + `Metric,Value`
  layout (split by has-percentiles), the GPU-telemetry section, `None → ""`.
- **`outputs.json`** (schema `"1.0"`, the 8-field record + the `output_token_count /
  output_sequence_length / request_latency` allowlist + PROFILING-only + `(session_num, turn_index)`
  sort) — but built from the in-RAM per-record list, **not** the fragment glob (§4).
- **Console `DEFAULT_STAT_KEYS`** (`avg, min, max, p99, p90, p50, std`) + the 8-group order — the
  console renderer is shared between v1 and v2 (same table code, different stat set).

The v1 compat sink is a *translation from `Report`*, so all the v1 field-name/ordering/NaN details
from the source read are its acceptance contract; they no longer constrain the v2 default.

---

## 3. What stays earned-in-blood in BOTH formats

**Unbuilt in Rust — owned by the Python parent.** The warning/insight intelligence and the console
behavior are format-independent domain value rendered by the Python presentation layer, not the
Rust runtime library. The lore below is the porting contract for that layer:

- **OSL-mismatch warning** (`osl_mismatch_count.avg > 0`; threshold `min(requested·5%, 50 tokens)`)
  with the fix-text verbatim (`--extra-inputs ignore_eos:true` / `min_tokens:<N>` per backend,
  `--use-server-token-count`, the `osl_mismatch_diff_pct` + `AIPERF_METRICS_*` pointers).
- **Usage-discrepancy warning** (`usage_discrepancy_count.avg > 0`; threshold 10%) with the
  tokenizer-mismatch causes + `usage_*_diff_pct` pointer + `--use-server-token-count`.
- **API-error insights** (`ErrorInsight { problem, causes, investigation, fixes }`) — keep both
  detectors verbatim: **MaxCompletionTokens** (`extra_forbidden` + `max_completion_tokens` →
  `--use-legacy-max-tokens`) and **DynamoSessionControl** (serde `unknown variant` + `bind` →
  Dynamo `session_control action='bind'` unsupported pre-v1.3.0-dev, commit d97c889ba; upgrade /
  `--use-legacy-dynamo-session-control` / disable `--use-dynamo-conv-aware-routing`). **This
  version lore is irreplaceable — port it exactly**; factor one `warning_panel(title, insight)`
  helper + `fn detect(&Report) -> Option<Warning>` functions instead of a class-per-detector.
- **Error-summary table** (Code / Type / Message / Count, `N/A` handling, grouped counts).
- The **INTERNAL/EXPERIMENTAL filter** lives at the sink (both formats drop them unless dev-mode);
  v2 additionally can *include* them behind `--dev` since its metadata makes them self-describing.

---

## 4. Accidental complexity DELETED (mechanism — independent of format)

1. **Plugin registry for exporters** (`plugins.iter_all` + plugins.yaml `data_exporter:`/
   `console_exporter:`) → a static Rust list. Alphabetical YAML order is meaningless.
2. **Exception-as-disable** (`DataExporterDisabled`/`ConsoleExporterDisabled` from `__init__`) →
   an explicit `enabled(&Cfg) -> bool`.
3. **`asyncio.Task` fan-out + `gather`** over synchronous file writes → a sequential loop; a small
   `JoinSet` only for network uploaders.
4. **`outputs_json` fragment glob-merge-cleanup** (`output_fragments/*.jsonl` + the
   `session_num:turn_index` cross-file join + rmdir) — a multi-process shard artifact. Accumulate
   in RAM, write once.
5. **The two-file join** (`profile_export.jsonl` metrics-map re-read) — in-process the metrics are
   already on each record.
6. **Double-instantiation** for path metadata → paths as data.
7. **`SERVER_METRICS_PARQUET` cross-process skip TODO** → gone in-process.
8. **`is_deferred` getattr probe** → encode the local-writers→uploaders order structurally (§6).
9. **The mlflow/wandb `spawn`/`mp.Queue`/`repr(exc)`/exitcode apparatus** → a cancellable timed
   async task (§6).

---

## 5. The redesign — one `Report` → static `Exporter`s

**Built state.** The typed IO-free `NativeReport` model lives in `aiperf_runtime::metrics_core` and the sole
native-v2 JSON write lives in `aiperf_runtime::report` (`rust/runtime/src/report.rs`), which
`aiperf-cli` calls to atomically commit the unified report. There is no plugin registry and no
shard-glob path. The broader static-`Exporter`-list design below (multiple sink impls behind one
trait, `enabled(cfg)` gating, console/uploader stages) is **the target shape once a second IO sink
exists in Rust**; today only the single JSON writer is implemented, so no `aiperf-report` crate has
been extracted — the writer stays in the runner-facing `aiperf_runtime::report` module. The type-driven
`MetricEntry` enum described here is realized in `aiperf_runtime::metrics_core::report`.

The design (for when a second sink lands): a thin sink layer above `aiperf_runtime::metrics_core` (it does
file/console/network IO). Consumes the typed `Report` (metrics + telemetry + accuracy + `RunInfo` +
errors + timeslices + per-record).

```rust
pub trait Exporter {
    fn enabled(&self, cfg: &Cfg) -> bool;               // replaces exception-as-disable
    fn file_info(&self) -> Option<FileExportInfo>;      // path as DATA
    async fn export(&self, report: &Report, cfg: &Cfg) -> Result<()>;
}
```

The bin assembles a **static list** filtered by `enabled(cfg)`:
`[V2NativeExporter (default on), GenaiPerfV1Exporter (--export-genai-perf), OutputsJsonExporter
(--export-per-record), TimesliceExporter]` + the console renderer. Run: local writers sequentially
→ console record-then-replay (§7) → timed uploaders (§6, after local files exist). Serialization is
`serde` with fields in a fixed order + `skip_serializing_if = Option::is_none` for absent + a custom
non-finite→`null` scrub; the v2 `metrics` map is an `IndexMap<Tag, MetricEntry>` where `MetricEntry`
is an enum `Scalar { value, … } | Distribution { avg, …, percentiles }` (the type-driven shape).

---

## 6. Uploaders (mlflow / wandb) — timed async task, not a subprocess

Python used a subprocess for a hard timeout over uncancellable blocking SDK network I/O (a thread
wrapper releases the awaiter but the SDK keeps the socket/run open for minutes against an
unreachable server) + fork-state isolation. Carry forward only the **durable requirement: a hard
wall-clock timeout so an unreachable tracking server can't hang shutdown**. In Rust: run each
uploader under `tokio::time::timeout` (or on a `spawn_blocking` worker the run stops awaiting) —
one generic `TimedUploader`, no `spawn`/`Queue`/pickle/exitcode apparatus. Per the coverage-gap
ledger, mlflow/wandb are **deferred / side-car**; the trait leaves the seam, impls come later.

---

## 7. Console record-then-replay (keep the behavior)

**Unbuilt in Rust — owned by the Python parent.** The native CLI console tables, logger, and
accuracy-summary CSV writer were deleted with the native binary; console presentation is a Python
concern. The behavior below is the contract for that layer.

Render the console table **twice**: once to a **fixed-width** buffer (`CONSOLE_EXPORT_WIDTH`) for the
width-pinned `profile_export_console.txt` (stable CI-log artifact, decoupled from terminal width),
then to the **live terminal** if attached (replay the recorded fixed-width text if non-tty). Same
table code, two targets. The console groups/order + `display_order` sort are shared by v1/v2; v2's
table can show the `type`/`higher_is_better` metadata inline.

---

## 8. Scope + testing

- **Built (Rust):** the typed IO-free native-v2 `Report` core in `aiperf_runtime::metrics_core::report`
  (`NativeReporter` behind the `Reporter` trait; typed run/summary/error, metric, series, timeslice,
  distribution/scalar/counter stats; warmup + accuracy joins; absent-omitted / non-finite-null
  discipline) and the single native-v2 JSON writer in `aiperf_runtime::report` that `aiperf-cli` invokes,
  pinned by a deterministic exact-JSON golden.
- **Unbuilt in Rust (Python-owned or deferred):** the native **v2 CSV** projection, the
  genai-perf-**v1** JSON/CSV + `outputs.json` compatibility sink, the warning/insight and error-table
  renderers, console record-then-replay, and timed uploaders. The native CLI console tables, logger,
  accuracy-summary CSV, and legacy aggregate/timing JSON helpers were **deleted** with the binary;
  the Python `aiperf` parent owns human presentation and compatibility/export generation. The
  static-`Exporter`-list, `enabled(cfg)` gating, and `TimedUploader` seam remain the target shape
  for when a second runner-owned Rust IO sink has a demonstrated hot-path reason to exist.
- **Deferred:** mlflow/wandb impls (side-car), parquet server-metrics, aggregate/sweep exporters
  (outer-loop coordinator), dashboards.
- **Deleted:** everything in §4.
- **Testing (two golden gates):** (a) **v2 golden — built** — a fixed `Report` → the exact native-v2
  JSON (nested `metrics`, scalar `value` vs distribution stats by type, `percentiles` map, per-metric
  metadata, absent-omitted/non-finite-null); a deterministic exact-JSON golden already pins this
  shape. (b) **v1 compat golden — future/Python** — the exact `profile_export_aiperf.{json,csv}` +
  `outputs.json` (SCHEMA_VERSION 1.4, the three frozen orderings, avg-for-scalars, extra=allow),
  gating whichever layer produces the compat artifacts. A field/order drift in either fails its gate.

## 9. Open questions

1. **v2 filename + `schema_version`.** `aiperf_report.json` / `"2.0"` proposed. Confirm the name and
   whether v2 CSV ships alongside v2 JSON or JSON-only (CSV as a v1-only compat concern).
2. **v2 default on, v1 default off — migration.** Ship v2 as default immediately (this decision), or
   a release where both emit and v1 warns "deprecated"? Lean: v2 default now, v1 behind
   `--export-genai-perf`, with a one-line note in release docs.
3. **Per-metric `type` vocabulary — RESOLVED (§1):** emit the *shape* the consumer reads —
   `distribution` / `scalar` / `counter` / `histogram` (the server-metrics vocabulary), not the
   internal `record`/`aggregate`/`derived` compute detail. The mapping (RECORD→distribution,
   DERIVED/AGGREGATE-MIN/MAX→scalar, AGGREGATE-counter→counter, server-HISTOGRAM→histogram) is in
   the §1 table. Open sub-question: do inference-metric series ever need `labels` (per-model in
   multi-model runs, per-worker)? The `series[]` structure supports it for free; emit `labels:null`
   until a breakdown is wired.
4. **Extract an `aiperf-report` crate vs keep the writer as a runner-facing module** — today the
   single JSON writer stays as the `aiperf_runtime::report` module; extract a lean crate when a second IO
   sink lands, testable on a synthetic `Report`.

---

## Addendum — 2026-07-14 (the full native-Rust exporter plane is BUILT and is the default)

Supersedes the "unbuilt in Rust / Python-owned" status throughout §2, §3, §5, §6, §7, §8. The
static `Exporter` plane (`aiperf_runtime::export`) is now built, wired into the runner's report-commit
site (`coordinator::persist_prepared_report`), and is the **default sole emitter** on the native
path. Nine sinks behind one object-safe `Exporter` trait with explicit `enabled(&ExportConfig)`
gating, projected from the frontend via `rust_wire._export`:

- **`genai_perf`** — aiperf v1 summary `profile_export_aiperf.{json,csv}` (schema 1.4). **Byte-identical**
  to the legacy Python `metrics_json/csv_exporter` on a live run (empty `cmp` diff), via the
  frontend projecting `MetricRegistry` header/filter/scalar maps + the `input_config`/`run_info`
  envelope; Rust assembles in exact `JsonExportData` order.
- **`timeslice`**, **`server_metrics`** (json/csv) — **byte-identical** on live runs (server-metrics
  driven against a live Prometheus endpoint).
- **`parquet`** — `server_metrics_export.parquet`, **schema+data equal** (`pyarrow Table.equals`);
  byte-identity not targeted (arrow-rs vs pyarrow encoding defaults). Consumes the runner's
  `.aiperf-server-metrics-parquet-wire.jsonl` and deletes it after (the Python round-trip is retired).
- **`accuracy_csv`** — **byte-equal** (offline oracle; live enable→emit path confirmed).
- **`otel`** — OTLP/HTTP GenAI-semconv metrics. Per-record histograms with **populated
  `bucket_counts`** from a runner per-record accumulator (`OtelRecordAccumulator`); count/sum match
  the native-v2 report exactly (decoded via `opentelemetry-proto`). Residual: single terminal
  cumulative export vs Python periodic (final aggregate identical); `aiperf.*` per-record attribute
  fragmentation omitted; online-scheduled + profiling-phase only.
- **`mlflow`** — REST + `file://` FileStore; live filestore matches the `metric.tag[.stat]` scheme.
- **`wandb`** — offline `.wandb` transaction log (leveldb framing + protobuf), SDK-decodable.
- **`console_txt`** — `profile_export_console.txt`. The Rich table+panel renderer is **byte-exact
  geometry**; the warning/insight detectors (OSL-mismatch, usage-discrepancy, MaxCompletionTokens,
  DynamoSessionControl) + error table are **byte-exact**. Content grouping/headers projected from
  Python's console metadata to match the `MetricRegistry` view.

**Default cutover (zero-Python):** `Environment.RUNTIME.NATIVE_EXPORT` (default true) makes the
native plane authoritative; `export_python_compatibility_reports` is a no-op on the native path.
`AIPERF_RUNTIME_NATIVE_EXPORT=0` restores the legacy Python emitters for A/B (mirroring
`AIPERF_RUNTIME_ENGINE=python`). The Python exporter modules are retained for the legacy path and
unit tests, not invoked natively. Live-verified: a default run emits the native compat files with
zero `ExporterManager` invocation.

**Parity harness:** `AIPERF_EXPORT_SUBDIR=<dir>` redirects the native sinks under
`<artifact_dir>/<dir>/` so they coexist with the legacy Python files for a same-`native-v2.json`
byte-diff (the verification vehicle for every "byte-identical" claim above).

The §5 `TimedUploader` seam is realized inside the otel/mlflow sinks (short-lived `current_thread`
runtime + `tokio::time::timeout`). The v2-native-report core (§1) remains as described.
