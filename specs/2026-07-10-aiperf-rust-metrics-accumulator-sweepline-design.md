<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Metrics Accumulator + Sweep-Line Engine (`aiperf_runtime::metrics_core`)

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — the IO-free performance-metrics engine ships in `aiperf_runtime::metrics_core`
(NaN-sparse column store, exact ragged replay, record/aggregate/derived metrics,
authoritative completion-usage reconciliation, SLO goodput, all effective/active +
ICL-aware sweep curves, duration-weighted stats, phase windows/timeslices, deterministic
worker-local merge, per-model/per-endpoint native-v2 series, typed `Reporter`). The
frozen genai-perf-v1 compatibility export (§14) is the one piece **not built** here — it
remains a compatibility sink in the separate exporter-overhaul spec.
**Grounding:** line-by-line read of `analysis/sweepline{,_stats,_kv_cache}.py`,
`metrics/{accumulator,column_store,_column_store_handlers,accumulator_models,accumulator_sweeps,metric_dicts,derived_latency,metric_registry}.py`,
`common/enums/metric_enums.py`, `common/accumulator_protocols.py`, the summary path
in `records/records_manager.py`, and the current Rust seam
`rust/loadgen-core/src/collector.rs`.
**Companions:** `2026-07-10-unified-graph-runtime-design.md` (§11.5 `Collector` /
`Reporter` seams — this IS "the columnar accumulator from the coverage-gap ledger
§1" they point at), `2026-07-10-aiperf-rust-coverage-gap-ledger.md` §1,
`2026-07-10-aiperf-rust-accuracy-accumulator-design.md` (the sibling accumulator +
the real-toposort-deps fix), `2026-07-10-aiperf-rust-rng-derive-system-design.md`
(the finite-float discipline pattern).

---

## 0. Thesis — new code, built from the scars

This is **redo-cleaner, not port-exact** (per the master ledger). The Python metrics
plane is two things fused: a set of **earned-in-blood algorithms** (sweep-line
time-weighted curves, the percentile kernel, the failure-inflated `adj_*`
distribution, phase-authoritative masking) wrapped in a thick layer of
**multiprocess/ZMQ accidental complexity** (a fan-in service, SUB/PULL ordering
hacks, a legacy per-instance replay processor kept alive beside the real one, a
plugin registry with reverse-lookup, dead dependency metadata). The
**from-scratch** `aiperf_runtime::metrics_core` engine (a module of `aiperf-runtime`, formerly the
standalone `aiperf-metrics` leaf crate):

1. **Carries the scars exactly** — the numeric algorithms and boundary contracts
   below are ported *behavior-for-behavior* and guarded with parity fixtures. These
   were paid for in wrong-number bugs; a "cleaner" reimplementation that changes the
   `−1`, the tie-break, the ddof, or the `nextafter` is a regression, not an
   improvement.
2. **Throws away the accidental complexity** — none of the process-boundary
   machinery survives; it collapses to owned Rust state behind clean traits.
3. **Fits the unified runtime** — it provides the columnar accumulator plus the typed
   `Reporter` (columns → `Report`) that the unified-graph-runtime spec names but leaves
   to "the columnar accumulator." Runtime translation lives one layer up, in
   `aiperf_runtime::metrics::NativeMetricsObserver`, which joins the Clock-stamped observer
   `Event` stream into per-record columns; the `metrics_core` engine itself stays
   IO-free. Same summary math runs on ONLINE and OFFLINE because both feed the same
   observer stream — online, fixed-schedule, user-centric, adaptive, accuracy, and
   graph execution all feed one accumulator.

The single most important thing this subsystem gets right: **the genai-perf output
contract** (field names, STAT_KEYS ordering, NaN-vs-null semantics) is the reason
AIPerf's numbers are trusted downstream. That contract does not move.

---

## 1. Ground truth (Python source)

| File | What it owns |
|---|---|
| `analysis/sweepline.py` | vectorized sweep-line curves: `_sweep_line_cumsum` (tie-break + FP-snap), concurrency / throughput / prefill / total / per-user / active variants, `SweepLineCurves.compute_metrics`. |
| `analysis/sweepline_stats.py` | duration-weighted stats over a step function: `compute_time_weighted_stats`, `compute_active_weighted_stats`, `_build_clipped_segments`. |
| `analysis/sweepline_kv_cache.py` | tokens-in-flight + ICL-aware throughput/tokens sweeps; the `np.nextafter` chunk clamp. |
| `metrics/accumulator.py` | `MetricsAccumulator` — the SOLE summary engine: columnar ingest, phase mask, RECORD/AGGREGATE/DERIVED dispatch, sweep injection, timeslicing. |
| `metrics/column_store.py` | `ColumnStore` — NaN-sparse columns, sentinels, categorical interning, O(1) sum/count, doubling grow. |
| `metrics/metric_dicts.py` | the percentile kernel (linear + ddof split) + `observation_duration` window override. |
| `metrics/derived_latency.py` | `effective_latency`, `credit_to_start_latency`, `adj_*` (`+inf`/nearest/`std=None`), the `_delta_ms` NaN-drop+clamp. |
| `metrics/metric_registry.py` | metric catalog, type-tier dependency rule, graphlib topo-sort. |
| `common/enums/metric_enums.py` | the unit algebra, `MetricType` / `AggregationKind` / `MetricConsoleGroup` / `MetricValueType` / `MetricFlags`. |
| `common/accumulator_protocols.py` | the `Accumulator` / result / stream-exporter protocol seam + `ExportContext` / `SummaryContext`. |
| `records/records_manager.py` | the summary *orchestration*: phase windowing, network-RTT delivery, the finalize order. |

The current Rust `loadgen-core::collector` is a *fixed struct* (nearest-rank
percentiles, population std, a hand-written `Serialize`). It is the walking-skeleton
seam; this spec is what it grew into (and largely moved out of `loadgen-core` into
`aiperf_runtime::metrics_core`).

---

## 2. The scars — port these behavior-exact (guard with parity fixtures)

Each row is an algorithm that looks simple and is not. Cite the Python `path:line`
in the Rust `///` docs and pin a golden fixture.

| Scar | Rule (must reproduce) | Source |
|---|---|---|
| **Sweep event tie-break** | lexsort `(event_type, ts)` with `event_type = (delta>0)`: **ends (−) sort before starts (+)** at equal timestamps. | `sweepline.py:246-249` |
| **FP-roundoff snap** | after cumsum, `where(|v| < 1e-9·max_abs, 0, v)` — physical non-negativity; a real bug is orders larger. | `sweepline.py:257-261` |
| **Decode-rate `−1`** | throughput rate `= (output_tokens − 1)/gen_dur`; valid iff `gen_dur>0 ∧ osl≥1`. The `−1`: the first token is not a decode step. | `sweepline.py:461` |
| **Prefill rate (no `−1`)** | `= input_tokens/prefill_dur` over `[start, gen_start)`; all input tokens processed. | `sweepline.py:500` |
| **Active-mask stats** | throughput/per-user time-weighted **only over segments where the phase has ≥1 in flight** (mask>0), else biased by idle gaps. | `sweepline.py:162-238`, `sweepline_stats.py:117-188` |
| **Duration-weighted percentile** | argsort value → cumulative-duration-fraction CDF → `searchsorted([.5,.9,.95,.99])`, clip to `n−1`. NOT count-weighted. | `sweepline_stats.py:80-88` |
| **ICL chunk clamp** | chunk arrival past `end_ns` → `nextafter(end_ns, −∞)` (NOT `−ε`: at epoch-ns, ULP ≈256 ns, so any sub-ULP subtract round-trips to a no-op and the `−end` free mis-orders before the `+chunk`, leaving a permanent negative offset). | `sweepline_kv_cache.py:246-261` |
| **ICL `≥0` vs `>0`** | tokens-in-flight allows `icl≥0` (back-to-back same-packet chunks are legit); throughput requires `icl>0` (zero interval can't carry a rate). Not a typo. | `sweepline_kv_cache.py:235` vs `:313` |
| **`(osl−1)` chunk spread** | the TTFT chunk is 1 token emitted separately at `gen_start`; the remaining `osl−1` spread across the K ICL intervals; total = osl. | `sweepline_kv_cache.py:222,329` |
| **`+`/`−` balance** | every `+contribution` needs an exactly-matching `−` gated by the same record-level `has_icl`/`pf_valid` mask, or the cumsum drifts a permanent offset. | `sweepline_kv_cache.py:163-181` |
| **Percentile band** | `[1,5,10,25,50,75,90,95,99]`; report kernel = **manual linear** `virtual_idx = q/100·(n−1)`, `lo=trunc`, `frac`, `clean[lo]+frac·(clean[hi]−clean[lo])`. | `metric_dicts.py:53,82-86` |
| **ddof split** | std **ddof=0 (population)** for inference metrics; **ddof=1 (sample)** for telemetry time-series; guard `std=0 if n≤ddof`. A forgotten `ddof=1` is a silent wrong-std. | `metric_dicts.py:65,88` |
| **`adj_*` percentiles (#688)** | flag-gated `PERCENTILE_INCLUDES_FAILED_REQUESTS`; `error_count≤0` no-op; append `error_count × +inf`; `method="nearest"` (linear across the `inf` boundary → NaN); `std=None`; avg/min/max/sum legitimately `+inf`. | `derived_latency.py:170-259` |
| **effective / credit-queue latency** | `effective_latency = end_ns − credit_issued_ns` (CO-aware); `credit_to_start = start_ns − credit_issued_ns`; identity `effective = credit_to_start + request_latency`. Absent (None) when no `credit_issued_ns` (fixed-schedule). | `derived_latency.py:92-151` |
| **`_delta_ms` NaN-drop + clamp** | drop NaN deltas, then `max(delta, 0)` before `/1e6` — epoch-ns ULP + cross-process skew can make a near-zero delta slightly negative. | `derived_latency.py:74-89` |
| **observation_duration override** | throughput/goodput denominators use `end−start` when BOTH window bounds set (timeslice/analyzer window), else `BenchmarkDuration`; zero → no value (not ÷0). | `metric_dicts.py:221-242` |
| **Phase-tag-authoritative mask** | when a phase is scoped, select by the `benchmark_phase` categorical tag **alone** — do NOT intersect the wall-clock window (a straggler's `start_ns == phase_end_ns` on a coarse clock would be dropped). Phase-less realtime windows keep half-open `[start,end)`. | `accumulator.py:179-209` |
| **Append-only record index** | records index by a monotonic internal counter, NOT `session_num` (which restarts per phase and would collide warmup-rec-0 with profiling-rec-0). | `accumulator.py:70-73` |
| **avg from running sum** | RECORD `avg = arr_sum/n` using the O(1) running sum, not a recomputed mean (float-order stability). | `metric_dicts.py:96` |

---

## 3. Accidental complexity to delete (do NOT transcribe)

All exist only because Python ran N record-processor *processes* over a ZMQ bus:

- **The fan-in service + SUB/PULL ordering hacks** (`records_manager.py` `await_dataset_configured`, dual-bind TCP, PULL handlers) → in-process the `Collector` receives the `Event` stream directly; no bus, no ordering barrier.
- **The legacy `MetricResultsProcessor` dual path** — the accumulator is already the *sole* summary producer; Python keeps the per-instance replay processor alive only for distributed shards and gates it off (`type(p) is not MetricResultsProcessor`). We build **one** engine, no gate.
- **The plugin registry + reverse-lookup + auto-discovery import side effects** (`metric_registry._discover_metrics`, `plugins.get_class`) → a static Rust catalog built once.
- **The `_grow` cached-closure invalidation dance** (`_tag_handlers.clear()` because reallocation invalidates captured array refs) → a Rust `Vec<f64>` grows in place; monomorphized column writers need no closure cache.
- **`required_accumulators` / `summary_dependencies` as dead ClassVars** re-checked imperatively → **real deps**: a `petgraph` toposort enforced at build (same fix as the accuracy spec).
- **The realtime log-block rendering** (`_render_realtime_block`) is console cosmetics over the *same* `MetricResult`s — it belongs to the presentation plane (coverage-gap §6), not here.
- **Per-processor JSONL/CSV shard writers + concat** → a single in-process `StreamExporter` writes once.

---

## 4. The engine: `aiperf_runtime::metrics_core`

The engine ships as the `metrics_core` module of `aiperf-runtime` (it began life as the
standalone `aiperf-metrics` leaf crate and was inlined when the sixteen former
`aiperf-*` library crates became modules of `aiperf-runtime`). **No `ndarray`** (the
sweep-line is `Vec<f64>` + `sort_by` + manual cumsum/searchsorted — full control over
the tie-break and determinism, no hidden allocation). It depends only on a shared
finite-float type, `serde`, and `rustc-hash` for categorical interning — **no
`blake3`**: metric categorical interning is dense first-appearance code assignment via
`FxHashMap` + a reverse vector, not content addressing (BLAKE3 stays relevant to
segment content hashes and RNG seed derivation, not metric category codes). The engine
stays the core metrics seam: telemetry domains reuse its accumulator/reporting seam and
may contribute side-channel accumulators or summaries, but `metrics_core` never depends
on telemetry-specific modules and forms no dependency cycle.

```
rust/runtime/src/metrics_core/
  mod.rs                # module root + Accumulator / Analyzer / Reporter seam
  value.rs              # MetricValue (finite | +inf | absent), FiniteFloat scrub
  catalog.rs            # MetricSpec table + MetricType/AggregationKind/Flags/ConsoleGroup + toposort
  units.rs              # the unit algebra (convert_to), MetricValueType
  store.rs              # ColumnStore: typed sparse columns, sentinels, interning, O(1) sum/count, worker-store merge
  ingest.rs             # RecordIngest — the per-record shape the observer fills
  kernel.rs             # percentile/distribution kernels (linear + nearest + ddof)
  derived.rs            # effective / credit_to_start / adj_* / network_adjusted
  counter.rs            # counter-delta + histogram side-channel accumulator seam
  sidecar.rs            # side-channel accumulator injection points
  accuracy.rs           # accuracy-join seam over this Accumulator
  sweepline/
    mod.rs              # _sweep_line_cumsum, step-fn ops (add/divide/lookup), curves
    stats.rs            # duration-weighted + active-weighted stats
    kv_cache.rs         # tokens-in-flight + ICL variants + the nextafter clamp
  accumulator.rs        # MetricsAccumulator: ingest → mask → compute → summarize + per-worker merge
  window.rs             # ExportContext / phase mask / observation_duration / timeslices
  report.rs             # NativeReporter: AccumulatorSummary → native-v2 Report
```

Dependency direction: `aiperf_runtime::run`/`aiperf_runtime::graph` (owning the observer →
`RecordIngest` translation in `aiperf_runtime::metrics`) → `aiperf_runtime::metrics_core`.
`metrics_core` is a leaf module beside `aiperf_runtime::clock` / `aiperf_runtime::rng`.

---

## 5. The columnar store (redesigned, scars kept)

Python's `ColumnStore` is NaN-sparse float64 columns + uint8/int32 sentinel
metadata + a doubling `_grow` + cached setter closures. Rust makes the *sparsity
explicit at the type level* and drops the closure cache:

```rust
/// A metric column. Absent slots are explicit, not NaN — but NaN-drop semantics
/// are preserved: `finite_values()` yields only present, finite entries in index
/// order (the input to the percentile kernel and running sum).
pub struct NumericColumn {
    vals: Vec<f64>,          // grows in place; f64::NAN marks absent (kept for O(1) masked math)
    sum: f64,                // O(1) running sum over PRESENT values (Python `_sums`)
    count: usize,            // O(1) present count (Python `_counts`)
}
```

Design decisions (each maps to a scar):

- **Append-only index.** The store has one monotonic `next_idx`; `record_count =
  count of present rows`. `session_num` is stored as *categorical metadata*, never
  the index (scar: warmup-rec-0 vs profiling-rec-0 collision).
- **NaN stays the in-column "absent" marker for float columns** — it is what makes
  the masked vectorized math (`col[mask]`, `col[~isnan]`) O(1) and keeps the sweep
  inputs index-aligned. `finite_values()` / a `present` bitset expose the drop.
  (An `Option<f64>` column would break the branch-free masked ops the sweep relies
  on; keep NaN, expose it behind a typed accessor.)
- **Metadata columns** are separate so the metric-compute loop never picks them up.
  Booleans: a `Vec<Option<bool>>` (Python's uint8 sentinel 255 is a memory trick we
  don't need — Rust `Option<bool>` is 1 byte). Categoricals: **dense first-appearance
  interning** — `FxHashMap<Box<str>, u32>` + a reverse `Vec<Box<str>>`, code assigned
  in insertion order (scar: the ordering guarantee; and the int32→u32 width is
  because `x_correlation_id` cardinality == n_records, overflowing int16).
- **`mask_for_categorical(tag, value)`** returns a bitset; a value never interned
  returns all-false (scar: the missing-sentinel must never false-positive).
- **Two `query_time_range` semantics, kept distinct** (both are real, don't merge):
  the accumulator's is **half-open on `start_ns` only** (`~nan ∧ start≥lo ∧ start<hi`);
  the store's is **inclusive overlap** (`start≤hi ∧ end≥lo`). Name them
  `mask_started_in` and `mask_overlaps`.
- **No `_grow` closure cache** — columns are `Vec`, grow in place, no captured refs
  to invalidate. The Python 30%-ingest-speedup was recovering what Rust has for free.

`RecordIngest` (`ingest.rs`) is the clean per-record shape the `Collector` fills
from the `Event` stream + trace metadata — replacing the `MetricRecordsData` +
`MetricRecordsMetadata` wire structs:

```rust
pub struct RecordIngest {
    // backbone timestamps (ns since run origin, from the Clock)
    pub start_ns: i64,
    pub end_ns: i64,
    pub ttft_ns: Option<i64>,          // generation_start = start + ttft
    pub credit_issued_ns: Option<i64>, // None ⇒ fixed-schedule; effective/credit-queue absent
    // per-tag record metric values (RECORD/AGGREGATE inputs)
    pub metrics: SmallVec<[(MetricTag, MetricValue); 16]>,
    pub inter_chunk_latency: Option<Vec<i64>>,  // ragged; drives the ICL sweeps
    // metadata (categorical/bool)
    pub phase: Phase,                  // Warmup | Profiling — the authoritative mask key
    pub was_cancelled: bool,
    pub has_error: bool,
    pub session_num: u64,
    pub turn_index: u32,
    pub worker_id: Option<Box<str>>,
    pub conversation_id: Option<Box<str>>,
    pub correlation_id: Option<Box<str>>,
}
```

---

## 6. Metric catalog + taxonomy (resolves ledger §8.1)

**Decision: a static `MetricSpec` catalog + `AggregationKind` fold for RECORD/
AGGREGATE, a typed `Derive` fn-table for DERIVED, and a real `petgraph` toposort.**
Not a heavyweight per-metric `Metric` trait — the metric set is fixed and known; a
trait-object-per-metric buys extensibility no one needs and costs a vtable per value.

```rust
pub struct MetricSpec {
    pub tag: MetricTag,
    pub header: &'static str,
    pub unit: Unit,
    pub display_unit: Option<Unit>,     // convert at export; None ⇒ unit
    pub kind: MetricType,               // Record | Aggregate | Derived
    pub aggregation: Option<AggregationKind>, // Aggregate only: Sum | Max | Min
    pub flags: MetricFlags,             // bitflags (§13)
    pub console_group: MetricConsoleGroup,
    pub plot_direction: PlotMetricDirection,
    pub required: &'static [MetricTag], // real deps
}

/// DERIVED metrics only. A pure function of the already-computed scalar dict.
/// Small table keyed by tag — the "trait only where extensibility is real".
pub type DeriveFn = fn(&ScalarDict) -> Option<MetricValue>;
```

- **Type-tier rule (enforced at build):** RECORD→{RECORD}, AGGREGATE→{RECORD,
  AGGREGATE}, DERIVED→{any}. A `petgraph` DAG over `required`; `toposort` gives the
  DERIVED evaluation order; a cycle is a *compile-time-adjacent* startup panic (the
  catalog is static, so a `#[test]` asserts acyclic + tier-valid — the port's
  equivalent of Python's import-time fail-fast).
- **RECORD** → a column; summarized by the percentile kernel (full distribution).
- **AGGREGATE** → folded to one scalar by `AggregationKind` (`Sum`/`Max`/`Min` —
  there is deliberately **no Mean** kind; means are DERIVED). Default `Sum`.
- **DERIVED** → `DeriveFn` over the `ScalarDict`, in topo order; one failing derive
  logs and is skipped (scar: "one bad derive must not abort the summary").

The catalog holds the **103 source-grounded Python metric identities plus 16 native
sweep identities** (ITL `osl<2` guard, TTFO first-non-reasoning-token, osl_mismatch
`min()` cap, thinking-efficiency, `good_request_count` per-metric `LARGER_IS_BETTER`,
`cache_reporting_hint` absent-vs-0). Telemetry-owned rows remain explicit injected
`NoValue` seams until their producer specs are implemented. This spec fixes the
*engine*; the catalog is data over it.

---

## 7. Percentile & distribution kernel (`kernel.rs`)

Two kernels, both intentional, chosen per output path — do NOT unify them:

- **Report kernel (genai-perf contract): manual linear interpolation.** Band
  `[1,5,10,25,50,75,90,95,99]`; sort ascending in place; `virtual_idx =
  q/100·(n−1)`; `lo = floor`, `hi = min(lo+1, n−1)`, `frac = virtual_idx − lo`;
  `pct = v[lo] + frac·(v[hi] − v[lo])`. `avg = sum/n` (running sum), `min = v[0]`,
  `max = v[n−1]`. **std takes `ddof`**: `sqrt(Σ(v−avg)²/(n−ddof))`, `0.0` if `n≤ddof`
  (scar). Default `ddof=0` (inference); telemetry passes `ddof=1`.
- **Mocker-conformance kernel: nearest-rank** `round((n−1)·p/100).min(n−1)` +
  population std. This is what the *current* Rust collector and the mocker's own
  `TraceCollector` use; it stays only on the OFFLINE-vs-mocker byte-conformance path
  (unified spec §13.5(b): the two legitimately differ — linear for genai-perf, nearest
  for mocker parity). The report never uses it.

`MetricResult` carries `{tag, header, unit, avg, min, max, std: Option<f64>, sum,
count, p1..p99}` where percentile and avg/min/max are `MetricValue` (finite | +inf |
absent) so `adj_*` can legitimately carry `+inf` and `std=None` (§13 NaN/Inf).

---

## 8. Derived latencies (`derived.rs`)

Reconstructed at summarize-time from stored timestamp columns, emitted in **display
units (ms)** (so they bypass the unit registry — injected *after* display
conversion, scar):

- **`credit_to_start_latency = start_ns − credit_issued_ns`** (queue wait; console
  group `NONE`). **`effective_latency = end_ns − credit_issued_ns`** (CO-aware;
  console `EFFECTIVE`). Both via `_delta_ms` = drop-NaN → `max(·,0)` → `/1e6`. Both
  **absent** (return None, omit the metric) when the `credit_issued_ns` column is
  empty (fixed-schedule). Identity holds: `effective = credit_to_start +
  request_latency`.
- **`adj_*` (issue #688):** for each RECORD metric flagged
  `PERCENTILE_INCLUDES_FAILED_REQUESTS`, if `error_count > 0`, build a second result
  over `concat(sorted_values, [+inf; error_count])` using the **nearest-rank**
  method (linear would compute `inf − inf = NaN` across the boundary); `std = None`;
  `avg/min/max/sum` become `+inf` (the intended "unbounded tail under failure"
  reading). tag `adj_{tag}`, header `"{header} (error-adjusted)"`, parent's native
  unit. `error_count ≤ 0` ⇒ emit nothing.
- **`network_adjusted_*`:** a constant-RTT shift, clamp-0, applied only to
  request-start-anchored metrics; ITL/ICL are deliberately excluded (RTT cancels in
  `latency − ttft`). Delivered via `set_network_rtt_ns` **before** summarize; RTT of
  0/None is a no-op (scar: avoid duplicate injection).

---

## 9. Sweep-line engine (`sweepline/`)

The heart. All curves are `(timestamp, delta)` events fed through one primitive:

```rust
/// Sort events (ends before starts at ties) and cumsum. THE tie-break + FP-snap.
fn sweep_line_cumsum(mut events: Vec<(f64 /*ts*/, f64 /*delta*/)>) -> StepFn {
    // stable sort by (ts, delta>0): end(0) before start(1) at equal ts.
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1 > 0.0).cmp(&(b.1 > 0.0))));
    let mut acc = 0.0; let mut vals = Vec::with_capacity(events.len());
    let mut ts = Vec::with_capacity(events.len());
    for (t, d) in &events { acc += d; ts.push(*t); vals.push(acc); }
    // snap: physical non-negativity; residual ~1e-12 renders "-0.00".
    if let Some(&max_abs) = vals.iter().map(|v| v.abs())... {
        for v in &mut vals { if v.abs() < 1e-9 * max_abs { *v = 0.0; } }
    }
    StepFn { ts, vals }
}
```

Curves (all with NaN-validity masks matching the Python `valid = ...` predicates):

- **concurrency** `+1@start / −1@end`; **weighted concurrency** (`+w/−w`) →
  tokens-in-flight when `w = token count`.
- **throughput** `+r/−r` over `[gen_start, end)`, `r = (osl−1)/gen_dur` (the `−1`).
- **prefill throughput** over `[start, gen_start)`, `r = isl/prefill_dur` (no `−1`).
- **total throughput** — prefill + generation events in one sweep pass.
- **per-user** — `divide_step_functions(throughput, concurrency)` (decode→generation
  concurrency, prefill→prefill concurrency; safe-divide → 0 where denom=0).
- **active variants** — `compute_active_weighted_stats` over the mask>0 grid, so the
  average reflects intensity *while the phase runs*, not diluted by idle gaps
  (`ConsoleGroup::ACTIVE`).
- **ICL variants** (`kv_cache.rs`) — output tokens ramp at each SSE chunk boundary:
  grouped-cumsum over the flat `(icl_values, record_indices, offsets)` CSR to get
  per-chunk wall-clock arrivals, `(osl−1)/chunk_count` tokens per chunk, the TTFT
  chunk emitted separately as `+1@gen_start`. **The `nextafter(end, −∞)` clamp**
  (scar §2) and the **`icl≥0` (tokens) vs `icl>0` (throughput)** split are
  load-bearing. Rust: `f64::next_down()` (or `libm::nextafter(x, f64::NEG_INFINITY)`),
  **never `x − ε`** — at epoch-ns magnitude a sub-ULP subtract is a silent no-op.

`SweepLineCurves` bundles all 9 families (ts+vals each) + 5 active variants;
`compute_metrics(window_start, window_end)` produces one `MetricResult` per curve
via the duration-weighted stats (§10). The bundle is computed **once** per summarize
and **re-windowed** per timeslice.

---

## 10. Duration-weighted stats (`sweepline/stats.rs`)

Over a clipped step function `[window_start, window_end]`:

- `avg = Σ(val·dur) / total_dur` — **total_dur = window span** (idle time counts as
  zero-value weight); the active variant divides by `active_dur` instead.
- `std = sqrt(Σ(dur·(val−avg)²) / total_dur)` — duration-weighted population.
- **duration-weighted percentiles:** argsort by value; `cum_dur = cumsum(dur[order])`;
  `cum_frac = cum_dur / cum_dur[−1]`; `idx = searchsorted(cum_frac, [.5,.9,.95,.99])`,
  clip to `n−1`; `p = sorted_val[idx]`. (The percentile is the value at which the
  cumulative *duration* fraction crosses p — not the count fraction.)
- `_build_clipped_segments`: `lo = max(0, searchsorted_right(start)−1)`, `hi =
  min(len, searchsorted_left(end)+1)`; seg value before the window = `v[lo−1]` or 0;
  clamp seg starts/ends to the window; keep `dur>0`.

Scale to output units at the boundary (`tokens/ns → tokens/sec` via
`NANOS_PER_SECOND`), then `scrub_non_finite` (finite-or-absent).

---

## 11. Windowing, masking, timeslicing (`window.rs`)

```rust
pub struct ExportContext {
    pub start_ns: Option<i64>,   // inclusive; None ⇒ unbounded
    pub end_ns: Option<i64>,     // exclusive; None ⇒ unbounded
    pub phase: Option<Phase>,    // None ⇒ phase-agnostic
    pub error_summary: Option<Vec<ErrorCount>>,
    pub cancelled: bool,
}
```

- **`mask_for(ctx)` — phase-tag-authoritative (the crown-jewel correctness fix):**
  base mask = present rows; **if `ctx.phase` is set → AND with
  `mask_for_categorical("phase", phase)` and RETURN** (no wall-clock intersection —
  the coarse-clock straggler scar). Phase-*less* (realtime rolling) contexts apply
  half-open `[start,end)` on `start_ns` so adjacent windows never overlap.
- **observation_duration:** the window bounds ride into the compute as
  `window_start_ns/window_end_ns`; a rate metric's denominator is `end−start` when
  both are set, else `BenchmarkDuration`; zero span ⇒ the metric is absent, never ÷0.
- **timeslices:** grid `[min_start, max(max_start, max_end)]` stepped by
  `slice_duration`; `n_slices` computed before building edges; `digitize` assigns
  bins; **empty bins dropped** (list stays dense = chronological index); the **last
  slice's end is clipped to run-end** (`is_complete` three-state: absent when
  complete, `false` when clipped) so sweep metrics aren't diluted by phantom idle
  padding. Each slice re-windows the one pre-computed sweep bundle (O(T·log M)).

---

## 12. Seams: `Accumulator` / `Analyzer` / `Reporter` (`mod.rs`)

The engine sits behind an observer feed and a typed `Reporter`. Runtime translation
lives above the leaf module: `aiperf_runtime::metrics::NativeMetricsObserver` joins
Clock-stamped observer events, terminal state, token classification, endpoint usage,
fine-grained HTTP trace facts, and workload dimensions into `RecordIngest`. Fixed
schedules explicitly omit the credit-relative latency family because they have no
credit-issuance phase. Graph workers accumulate without a per-token cross-thread lock,
return their local (lean) stores through scoped thread joins, and merge in
deterministic worker order:

```rust
/// Owns exactly one record type; ingest → mask → summarize. `!Send` single-thread.
pub trait Accumulator {
    type Summary: AccumulatorResult;
    fn record_type(&self) -> RecordType;
    fn process_record(&mut self, rec: &RecordIngest);
    fn query_time_range(&self, lo: i64, hi: i64) -> BitVec;      // half-open, started-in
    fn export_results(&self, ctx: &ExportContext) -> Self::Summary;
}

/// Reads OTHER accumulators' outputs in dependency (toposort) order — this is where
/// accuracy-per-watt / quality-at-goodput joins live (accuracy spec).
pub trait Analyzer {
    const REQUIRED: &'static [AccumulatorType];   // REAL deps, toposorted — not dead metadata
    const OPTIONAL: &'static [AccumulatorType];
    fn summarize(&self, ctx: &SummaryContext) -> AnalyzerOutput;
}

/// Columns/summary → the genai-perf-schema Report (§14).
pub trait Reporter { fn report(&self, summary: &AccumulatorSummary, run: &RunOutcome) -> Report; }
```

- **`MetricsAccumulator` is the sole summary engine** — no legacy replay processor.
- **Real deps, enforced:** a `petgraph` toposort over `Analyzer::REQUIRED` (and the
  metric catalog's `required`) — an analyzer whose required accumulators are absent
  is *skipped*, deps run in order (accuracy spec's fix; kills "works by luck of
  insertion order").
- **Finalize order (from `records_manager`, kept):** deliver network-RTT → summarize
  profiling → summarize warmup → finalize stream exporters → snapshot completed-count
  **before** appending derived aggregates (they must not inflate the request count).
- **No completion barrier / ZMQ** — the Collector owns the record stream; summarize
  runs when the run drains.

---

## 13. Vocabulary + NaN/Inf discipline (`units.rs`, `value.rs`, `catalog.rs`)

- **Units** — enumerate the families with `convert_to` exactly: size (**1024-based**),
  time (`per_second` ratio `other/self`), power (W/mW), energy (J/mJ/MJ), frequency
  (Hz/MHz/GHz), **temperature (affine: `(v+off)·celsius` then `/other.celsius −
  other.off`** — the one non-linear family), and composite over-time (`primary/time`,
  `inverted` → `time/primary`, optional `third`). **Same-unit conversion is an
  identity no-op across families; a distinct cross-family conversion errors.**
- **`MetricValueType`** = `Float | Int | FloatList | IntList` (default `Float`).
- **`MetricConsoleGroup`** = `None | Default | Usage | Cache | Prediction | Audio |
  Reasoning | Effective | Active` — the console-filtering contract.
- **`MetricFlags`** = a `bitflags!` set preserving the **exact bit positions** incl.
  the **reserved gap at bit 3** (`1<<3` owned by nothing, between
  `PRODUCES_TOKENS_ONLY` and `LARGER_IS_BETTER`). Predicates: `has_all`, `has_any`,
  `missing` (with `missing(NONE) == true`). The flags gate applicability
  (`STREAMING_ONLY`, `ERROR_ONLY`, `GOODPUT`, `HTTP_TRACE_ONLY`, …), console/export
  visibility (`INTERNAL`, `EXPERIMENTAL`, `NO_INDIVIDUAL_RECORDS`), direction
  (`LARGER_IS_BETTER`), and the `adj_*` opt-in (`PERCENTILE_INCLUDES_FAILED_REQUESTS`).
- **`MetricValue` = finite(f64) | pos_inf | absent.** NaN never crosses the boundary.
  **Inf is meaningful only for `adj_*`** (unbounded tail under failure) and is
  preserved there; every other path `scrub_non_finite` maps non-finite → absent. std
  is `Option<f64>` (absent for `adj_*`). This is the `FiniteFloat` newtype discipline
  at the type level (RNG-spec pattern).

---

## 14. The report contract (`report.rs`) — do not drift

**Built:** `NativeReporter` implements the IO-free, metrics-first native-v2 report
model (per-model / per-endpoint series included); the application layer writes it for
`--json` in every CLI mode.

**Unbuilt:** the frozen genai-perf-v1 compatibility export described below is **not**
part of the accumulator/sweep-line engine — it remains a compatibility sink in the
separate exporter-overhaul spec. Its contract still governs any future genai-perf-v1
sink and must not drift:

The one frozen external contract. Three distinct STAT_KEY orderings (CSV / JSON /
console), `profile_export_aiperf.json` (`SCHEMA_VERSION`, `extra="allow"`) +
`.csv` field names, and the **NaN/Inf round-trip rule**: serialize via
`scrub_non_finite` + a direct encoder, **never** a coercing `model_dump_json`
equivalent (which turns non-finite → `null` and collides with explicit-absent
"metric not present"). Explicit-absent (metric omitted) and `null` (present but
non-finite) are **distinct** and must stay distinct. The two console warning
exporters (OSL-mismatch, usage-discrepancy) carry earned thresholds + actionable
text — keep them.

---

## 15. Testing / parity

1. **Scar fixtures (byte-behavior):** a Python twin harness emits, per algorithm, a
   fixed input → golden output (sweep curves at known event sets incl. tie-break and
   `nextafter` cases; percentile kernel incl. the ddof split and the `+inf` adj band;
   duration-weighted percentile; phase-mask boundary straggler; observation_duration
   override). The Rust unit test asserts equality (exact for integer/counted values,
   tight ULP tolerance for float sums where documented).
2. **Property tests:** sweep `+`/`−` balance (final cumsum returns to 0 when all
   intervals close); concurrency ≥ 0 everywhere; percentile monotonic in p.
3. **End-to-end regression canary (Rust-internal):** a fixed workload → snapshot the
   full report → commit → assert reproduced (catches accidental metric drift), the
   same shape as the RNG canary.
4. **genai-perf schema golden:** a report JSON/CSV field-name + ordering fixture,
   so the export contract can't silently drift.

---

## 16. Scope boundaries

- **In:** the columnar `MetricsAccumulator`, the sweep-line engine (all curves +
  ICL), the percentile/derived kernels, duration-weighted stats, phase windowing +
  timeslicing, the metric catalog + taxonomy (struct+fold+toposort), the units/flags
  vocabulary, the `Reporter` + genai-perf schema, `RaggedSeries` (CSR flat-values +
  offsets, grouped-cumsum resetting at record boundaries) as the list backend.
- **Deferred (own specs, but the seams are here):** the ~120-metric **catalog
  appendix** (each metric's tag/unit/flags/formula); the **t-digest** list backend
  (only if ramp-scale forces the sketch — keep exact `RaggedSeries` until then);
  the **telemetry accumulators** (gpu/server counter-delta + histogram estimator,
  coverage-gap §5 — they reuse this `Accumulator`/`query_time_range` seam and the
  `ddof=1` kernel path); the **accuracy accumulator** (its own spec).
- **Thrown away:** everything in §3.

---

## 17. Open questions

1. **`ndarray` vs `Vec<f64>`** for the sweep-line hot path. **Resolved: `Vec` + manual
   ops** (control over the tie-break/FP-snap, deterministic, no hidden alloc); revisit
   only if a >1M-record summarize profiles hot.
2. **Where does the record translation live** — **Resolved: above the leaf.**
   `aiperf_runtime::metrics::NativeMetricsObserver` (in the runtime layer) owns the observer →
   `RecordIngest` translation and knows the `Event` stream; `metrics_core` stays IO-free
   and testable on synthetic `RecordIngest`.
3. **Report kernel ddof for the mixed console table** — inference metrics are ddof=0,
   telemetry ddof=1; when both appear in one table the kernel must be told per-metric
   (carry `ddof` on `MetricSpec`, default 0). Confirm no metric needs a third rule.
4. **Catalog encoding** — a `const` array of `MetricSpec` vs a build-time macro. Lean
   `const` array + a `#[test]` that asserts the toposort is acyclic and tier-valid
   (the port's equivalent of Python's import-time fail-fast).
