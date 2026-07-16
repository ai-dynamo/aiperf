<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Telemetry Accumulators (GPU · server-metrics · network-RTT)

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — `aiperf_runtime::gpu_telemetry`, `aiperf_runtime::server_metrics` (owns its self-contained
Prometheus/OpenMetrics parser), and `aiperf_runtime::network_latency`, driven by Clock-paced runner
sidecars that snapshot at the phase boundaries the runtime owns.
**Grounding:** line-by-line read of `gpu_telemetry/{accumulator,constants,metrics_config,dcgm_collector}.py`
+ `common/models/telemetry_models.py`; `server_metrics/{accumulator,data_collector,units,histogram_percentiles}.py`
+ `common/mixins/base_metrics_collector_mixin.py`; `network_latency/{accumulator,probe,manager}.py`;
the relevant `environment.py` settings and the `records_manager` RTT delivery.
**Companion / parent:** `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`
(the `Accumulator` / `query_time_range` / `ExportContext` seam + the **ddof=1** telemetry
kernel path this reuses), `2026-07-10-aiperf-rust-accuracy-accumulator-design.md`
(the sibling accumulator + the accuracy-per-watt / EnergyEfficiencyAnalyzer JOIN), and the
catalog appendix (the injected `total_gpu_*` / `network_adjusted_*` `MetricSpec` rows).

---

## 0. Thesis

Three telemetry planes — **GPU** (DCGM/NVML/AMDSMI scrape), **server-metrics** (the
inference server's own `/metrics`), and **network-RTT** (TCP-connect calibration) — are
each a Python ZMQ-service manager wrapping a small, sharp domain algorithm. The managers
collapse to plain Clock-paced scrape tasks writing into in-process accumulators behind the
parent spec's `Accumulator` trait; we carry the domain algorithms exactly and delete the
multiprocess scaffolding. Two things bind all three to the metrics engine:

1. **The `query_time_range(start,end) → mask` hook is identical across every accumulator** —
   it is the JOIN contract the EnergyEfficiencyAnalyzer / accuracy-per-watt rides for the
   *gauge* (distribution-over-phase) series.
2. **GPU energy and server counters are the SAME counter engine** — a phase-boundary
   snapshot delta with a reset-clamp. Build it once, share it.

Because single-process AIPerf-Rust owns the clock **and** the phase lifecycle, it queries
telemetry *at the moments it controls* — it snapshots counters at phase boundaries instead
of scraping on a blind timer and reconstructing phase membership from timestamps after the
fact. That single decision deletes a large windowing layer the Python path carried only as an
artifact of its multiprocess/async-timer model (see §1). The load-bearing scars that remain
are process-independent facts about what the hardware/server *emit*: the **counter
reset-clamp**, the server **routing/fallback/auto-disable state machine**, the
**vLLM/SGLang metric atlas**, the **polynomial histogram estimator**, and the **network-RTT
calibration** (fresh-unpooled/TLS-skip/DNS-once/MIN_SAMPLES top-up). One thing to explicitly
**NOT** port: the **NaN-taint-whole-histogram** behavior (a ZMQ/orjson artifact, §6).

---

## 1. The shared counter engine (`aiperf_runtime::metrics_core::counter`)

GPU energy and every server COUNTER use one algorithm, and it rides the phase boundaries the
runtime owns rather than reconstructing windows from an async sample series.

### 1.1 Phase-boundary snapshots (the windowing layer)
At each transition it owns (warmup→profiling start; profiling end), the runtime issues a
synchronous **telemetry barrier**: force one scrape of every collector and capture the counter
values as that phase's **baseline** (at start) / **final** (at end).

```
per-phase delta = max(final_snapshot − baseline_snapshot, 0.0)   # reset-clamp stays
```

- **Reset-clamp `max(·,0)`** absorbs a monotonic-counter reset (DCGM/server restart drops the
  counter below its prior value); without it the realtime row emits negative rates/hit-rates.
  This is process-independent — a GPU/server can still restart mid-run — so it is kept exactly.
- **`avg` carries the delta** — the counter `MetricResult` has no distribution/percentiles;
  downstream sums read `result.avg`.
- **Rates use the same baseline and the authoritative phase duration.** `rate = (final −
  baseline) / (phase_end − phase_start)`, where the phase bounds are the exact timestamps the
  runtime owns. There is **one** baseline (the phase-start snapshot); the Python path's
  asymmetric *delta* (last-before-start) vs *rate* (first-at/after-start) pickers were different
  only because it reconstructed windows from a blind async series — with a phase-start snapshot
  both use the same exact baseline, so they unify.

What the boundary-snapshot model **deletes** from the Python path (all of it multiprocess/timer
accidental complexity, none of it earned-in-blood):
- **`FINAL_SCRAPE_GRACE_NS`** window-widening — the grace window existed only to catch a
  trailing async scrape landing after `requests_end_ns`; a forced scrape *at* phase end captures
  the exact end counter.
- **Pre-window baseline reconstruction** (`searchsorted` for the sample before `start_ns`) — the
  phase-start snapshot *is* the baseline, known exactly, not guessed.
- **The two masking conventions + `query_time_range` for counters** — the phase→samples mapping
  is exact by construction (the phase told the collector when to snapshot).

### 1.2 Gauges over authoritative windows
GAUGE series (non-counter: power/utilization/temperature) need a *distribution* over the phase
(avg/percentiles), so they are still sampled on a cadence *during* the phase — but the window
bounds are the **authoritative phase-boundary timestamps the runtime owns**: samples with
`t ∈ [phase_start, phase_end]`, exact bounds, no grace/baseline/coarse-clock fudge. Over that
window the parent kernel applies: `nanmin/max/mean/std(ddof=1, only if ≥2 non-NaN else 0)/
percentile[1,5,10,25,50,75,90,95,99]`. This is exactly the parent spec's **`ddof=1` telemetry
path** — the reason the percentile kernel takes `ddof` as a parameter. `query_time_range` for
gauges is **half-open `[start, end)`** (the analyzer JOIN hook).

**Design driver, generalized:** in the single-process, clock-owning architecture, *the
benchmark decides WHEN telemetry is queried relative to phases.* Telemetry is snapshotted at the
boundaries the runtime controls, not tagged-and-reconstructed. (The metrics-engine spec's
phase-tag-authoritative mask for *request records* is unchanged and stays correct — records ARE
phase-tagged at dispatch; only *telemetry* samples, which are not phase-tagged at their source,
move to boundary snapshots.)

---

## 2. GPU telemetry (`aiperf_runtime::gpu_telemetry`)

### 2.1 Ingest + storage
Gauge samples land in a `TelemetryHierarchy` (`dcgm_url → gpu_uuid → per-signal columnar
NaN-padded series`) tagged with `record.timestamp_ns`; `query_time_range` = half-open
`[start, end)` over that series for the gauge distributions (§1.2). Counter series need no
read-time reconstruction — their per-phase values are the boundary snapshots (§1.1).

### 2.2 The DCGM field / unit table (port as a static table)

DCGM Prometheus field → internal name → unit → collector scale:

| DCGM field | internal | unit | collector scale (why) |
|---|---|---|---|
| `DCGM_FI_DEV_POWER_USAGE` | gpu_power_usage | WATT | — |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | energy_consumption | **MEGAJOULE** | ×`1e-9` (mJ→MJ) |
| `DCGM_FI_DEV_GPU_UTIL` | gpu_utilization | PERCENT | — |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | mem_utilization | PERCENT | — |
| `DCGM_FI_DEV_FB_USED` | gpu_memory_used | GIGABYTES | ×`1.048576e-3` (MiB→GB) |
| `DCGM_FI_DEV_GPU_TEMP` | gpu_temperature | CELSIUS | — |
| `DCGM_FI_DEV_ENC/DEC_UTIL` | encoder/decoder_utilization | PERCENT | — |
| `DCGM_FI_PROF_SM_ACTIVE` | sm_utilization | PERCENT | ×`100` (ratio→%) |
| `DCGM_FI_DEV_XID_ERRORS` | xid_errors | COUNT | — |
| `DCGM_FI_DEV_POWER_VIOLATION` | power_violation | MICROSECONDS | ×`1e-3` (ns→µs) |

**Counter set** (delta-not-stats): `{energy_consumption, xid_errors, power_violation,
amd_energy_consumption, amd_ecc_uncorrectable}`. The AMD/ROCm mirror table (`amd_power`,
`amd_energy_consumption`, `amd_gfx/umc/mm_activity`, `amd_memory_used`, `amd_temperature`,
`amd_ecc_uncorrectable`, `amd_throttle_status`) is a parallel static table.

Collector (`dcgm_collector`): scrape `/metrics` with a Prometheus text parser; **one
`Clock`-derived stamp per scrape** (all samples share it); skip non-finite; require a
parseable `gpu` label; **strip `_total`** before matching (DCGM counters expose `..._total`;
`_created` samples simply never match after the strip — no special-case needed); dedup identical
scrape bodies (`is_duplicate`). **Note the double energy scaling:** collector mJ→MJ (`1e-9`),
rollup MJ→J (`×1e6`) — net `1e-3`, in two files.

### 2.3 GPU energy at the phase boundary
Energy is a monotonic counter on a ~3 Hz cadence. Under the boundary-snapshot model its
per-phase value is a clean `max(final_snapshot − baseline_snapshot, 0.0)` at the forced
phase-end scrape — no window-widening heuristic. (The Python path carried a
`FINAL_SCRAPE_GRACE_NS = 666_000_000` ns grace window, ≈2× the 333 ms collection interval,
only to catch a closing async scrape landing a few hundred ms after `requests_end_ns`; the
forced boundary scrape makes it unnecessary and it is deleted.)

### 2.4 Cross-GPU efficiency rollups (the injected `total_gpu_*`)
Once per profiling phase, Σ over GPUs:
- **total_gpu_power** = Σ per-GPU mean watts (over the authoritative `[phase_start, phase_end]`
  gauge window). guard `power_count>0`.
- **total_gpu_energy** = Σ per-GPU counter-delta (MJ) × `1e6` → J, from the phase-boundary
  snapshots. guard `energy_count>0`.
- **output_tokens_per_joule** = `total_output_tokens / total_gpu_energy` (guard energy>0).
- **energy_per_user** = `total_gpu_energy / concurrency` — **omitted** when concurrency is
  unset (request-rate runs) or 0; concurrency must be a positive `int` and **not a bool** (the
  bool-is-int subclass trap).

These four are written into the metric results as the injected `total_gpu_power/energy`,
`output_tokens_per_joule`, `energy_per_user` catalog rows (§0 of the catalog appendix — their
`DeriveFn` returns `NoValue`; this injector supplies the scalar). This is the
EnergyEfficiencyAnalyzer feedstock and the accuracy-per-watt JOIN.

**Deferred collectors:** PyNVML / AMDSMI local collectors need `nvml-wrapper`-style bindings;
only DCGM-Prometheus is in the first cut. Custom-metrics CSV (`--custom-metrics`) with the
`(in <unit>)` regex inference is a small add behind the same table.

---

## 3. Server metrics (`aiperf_runtime::server_metrics`) — the routing/fallback/auto-disable state machine

This is the highest-risk domain logic, and it is the part that survives the boundary-snapshot
rethink untouched — it is server-compatibility behavior, not scrape-transport bookkeeping. It
spans the scrape engine + the collector:

```
scrape /metrics:
  Content-Type application/json?           → reject up front (IncompatibleMetricsEndpointError)  [TRT-LLM serves JSON stats here]
  else read body; parse by Content-Type:
     application/openmetrics-text*          → strict OpenMetrics parser, fall back to classic on ValueError  [vLLM Rust frontend quirk]
     else                                   → classic Prometheus parser
  parse ValueError / non-Prometheus body   → IncompatibleMetricsEndpointError (with a 200-char body preview)

on IncompatibleMetricsEndpointError:
  first time, URL ends with /metrics (not already /prometheus/metrics)?
     → probe-once: swap URL to <base>/prometheus/metrics, RESET last-response-hash, re-fetch   [TRT-LLM return_perf_metrics mount]
        success → keep the swapped URL permanently
        failure → restore URL; WRAP any failure back into IncompatibleMetricsEndpointError      [else it escapes auto-disable]
  else
     → TERMINAL AUTO-DISABLE: set endpoint_disabled = true, log once, notify once;
        every future scrape short-circuits to a no-op.
```

Earned-in-blood details the port keeps (all process-independent):
- **`IncompatibleMetricsEndpointError` is the single funnel.** Anything that reaches the loop
  as a different error type (a raw 404, a `ClientResponseError`) **escapes auto-disable** and
  spirals ("30-min benchmark → 8-hr parse-error loop"). The fallback wraps *every* non-success
  into it deliberately.
- **`_created` double-skip** (family-level: names ending `_created`/`_uptime`; AND sample-level:
  OpenMetrics `_created` samples share the `_total` label set and would overwrite the real value
  in label-dedup). **SUMMARY families skipped** (server-lifetime cumulative quantiles, not
  per-benchmark). **HISTOGRAM** → the estimator (§4).
- **Type-guards everywhere** (COUNTER vs GAUGE): after the parser strips `_total`, a counter
  `num_retracted_reqs_total` collides with the gauge `num_retracted_reqs`; every counter/gauge
  reader filters by metric type to avoid cross-contamination.

### 3.1 Sequential Clock-paced scrape loop
`aiperf_runtime::server_metrics` owns a **self-contained** Prometheus/OpenMetrics exposition parser
(`server_metrics/parser.rs`) — there is no separate parsing leaf; the `_total` strip,
`_created`/`_uptime` skip, OpenMetrics-vs-classic routing, and label-dedup are all custom, so
they live here.

The scrape is driven as a Clock-paced task that **awaits each scrape before scheduling the
next**: at most one scrape in flight. This deletes the concurrency gymnastics the Python
fire-and-forget loop needed — Python fired scrape coroutines every interval without awaiting, so
multiple could be in-flight and all reach the terminal-disable block, which is why it had to
**check-and-set `endpoint_disabled` before any `await`** and dedup on `_last_response_hash` as a
race-safety requirement. With one scrape in flight, the race disappears: the auto-disable
*decision* (`IncompatibleMetricsEndpointError` → stop scraping) is a plain set, and
`_last_response_hash` dedup drops to a mere parse-skip optimization (still reset on URL swap so
the alt endpoint's first body isn't mistaken for a duplicate of the prior `/metrics` body).

Constant: `COLLECTION_INTERVAL 0.333s`; `REACHABILITY_TIMEOUT 10s` (no total timeout — long
scrapes not cut off); hit-rate cap `100%`. There is **no** `COLLECTION_FLUSH_PERIOD` and **no**
flush/settle wait: force one final scrape at the phase boundary, snapshot, done. The Python
`COLLECTION_FLUSH_PERIOD` (~2s) only ever existed to widen/settle a reconstruction window, and
waiting for the server to "catch up" is not worth buying:
- **AIPerf's own metrics are authoritative.** AIPerf owns every request's dispatch and
  completion, so per-phase tokens/requests/latencies come from the metrics engine. The server
  `/metrics` counters are a **secondary cross-check** (server-side throughput, KV-cache,
  preemptions) — their phase-exactness is not worth a settle wait.
- **The residual boundary smear is bounded and lands next door.** A few requests whose
  server-side counter increment arrives just after the boundary snapshot fall into the
  neighboring phase's delta — a one-server-lag smear, not a systematic error, negligible against
  a phase total.
- **A settle wait's cost is real and one-directional:** a fixed dead-time on every phase
  transition, *and* it pollutes the boundary — during an idle settle window the server cools, so
  any gauge sampled then is garbage.

So GPU and server counters are identical: **snapshot at the boundary → delta → reset-clamp**, no
grace, no settle. Server counters (`prompt_tokens`, `generation_tokens`,
`prefix_cache_hits`/`_queries`, `num_preemptions`, request-latency histogram buckets) take
`max(end_snapshot − start_snapshot, 0.0)`, and rates use the exact phase duration
`(end − start) / (phase_end − phase_start)`.

### 3.2 The vLLM/SGLang realtime metric atlas (vLLM-first, SGLang-fallback)
A static mapping table; every name stored **without `_total`** (the parser strips it). Each row
picks the first present source:

| output | vLLM | SGLang | handling |
|---|---|---|---|
| prefix_cache_hit_rate | `prefix_cache_hits`/`prefix_cache_queries` | `cached_tokens`/`prompt_tokens` (last-resort gauge `cache_hit_rate`) | counter-delta pair, `100·min(hits,queries)/queries` (cap ≤100% — independent latching) |
| unique_input_tokens_srv | queries−hits | prompt−cached | `max(delta,0)` |
| external_prefix_cache_hit_rate | `external_prefix_cache_hits`/`_queries` | — | only when ext_queries>0 |
| kv_cache_usage_pct | `kv_cache_usage_perc` (v0 `gpu_cache_usage_perc`) | `token_usage` | first-gauge, `_to_pct` |
| cpu_kv_cache_usage_pct | `cpu_cache_usage_perc` | `hicache_host_used/total_tokens` | gauge-max / within-endpoint ratio |
| num_running | `num_requests_running` | `num_running_reqs` | first-gauge |
| num_waiting | `num_requests_waiting` | `num_queue_reqs` | first-gauge |
| num_preemptions | `num_preemptions` | `num_retracted_reqs` | counter-delta (COUNTER type-guard critical) |
| input_token_throughput_srv | `prompt_tokens` | `prompt_tokens` | counter-rate (tok/s) |
| output_token_throughput_srv | `generation_tokens` | `generation_tokens` | counter-rate |

`_to_pct(frac) = frac·100 if frac≤1 else frac` (normalize 0–1 gauges without double-scaling
already-percent values). SGLang `cache_hit_rate` gauge is the *last* fallback (per-batch, reads
0 between requests → misleading in idle low-concurrency windows).

### 3.3 Unit inference (`units.rs`, priority-ordered, cached)
`infer_unit(name, description)`, memoized (lru 2048):
1. **description scale** (most authoritative): `RATIO` if the description matches a 0–1 range
   (`(0-1)`, `0.0-1.0`, `0 to 1`, "1 means 100%"), else `PERCENT` if a 0–100 range — this can
   *override* a `_percent` name suffix.
2. **description unit**: the DCGM `(in <unit>)` parenthetical (case-sensitive map, `mJ`≠`MJ`,
   `MiB`, `°C`, `µs`), else a case-insensitive unit-phrase regex.
3. **name suffix** (longest-first so `_milliseconds`>`_seconds`, `_tokens_total`>`_total`), then
   a `num_requests_*` fast-path → REQUESTS.

### 3.4 Gauges and histograms: the split
- **Gauges** (`num_running`, `num_waiting`, `kv_cache_usage_pct`) need a distribution over the
  phase → continuous sampling over the authoritative `[phase_start, phase_end]` window, `ddof=1`
  stats. Same as GPU §1.2.
- **Histograms are a hybrid** (the one case that still wants the intra-phase series): the
  polynomial estimator's Phase-1 bucket-mean learning (§4) derives exact per-bucket means from
  intervals *between consecutive scrapes* where a single bucket is active — so it **needs the
  continuous intra-phase scrape series**, not just the two boundary snapshots. Keep continuous
  intra-phase scrapes feeding the learner; take the phase's **total** bucket counts/sum from the
  boundary delta (`end_buckets − start_buckets`). Boundary snapshots give the totals; the series
  gives the shape.

---

## 4. The polynomial histogram percentile estimator (`aiperf_runtime::server_metrics::histogram`)

Server histograms (Prometheus cumulative buckets) need percentiles; standard linear
interpolation is ~950% P99 error when data lands in `+Inf`. Port the estimator from
**HistogramTools (arXiv 2504.00001)** — ~20% P99 error (2.5×–47× better on the tail).

**Two public methods; the caller selects (no runtime auto-select):**
- **A — Prometheus linear (baseline):** `compute_prometheus_percentiles(cumulative_buckets)`.
  `target_rank = q·total`; find the bucket crossing it; `prev_bound + (bound−prev_bound)·
  (target−prev_count)/bucket_count`; **`+Inf` bucket → returns the last finite upper bound**
  (why the tail is wrong). Fixed band `[1,5,10,25,50,75,90,95,99]`.
- **B — polynomial (accurate):** `compute_estimated_percentiles(bucket_deltas, bucket_stats,
  total_sum, total_count)`. Needs learned per-bucket means/variance + the exact `total_sum`.

**B is not closed-form** — it *materializes a synthetic observation array* (capped at
`_MAX_OBSERVATIONS = 100_000`, downsampling counts+sum by the same ratio so within-bucket
averages are preserved) and takes `np.percentile` of it. **A faithful Rust port must reproduce
the observation-placement arithmetic AND numpy's linear percentile interpolation.**

**Phase 1 — learn bucket stats** (`accumulate_bucket_statistics` over the intra-phase scrape
series): between consecutive scrapes, take count/sum/bucket deltas (clamp ≥0 for resets); an
interval where **exactly one bucket is active** has an exact mean `sum_delta/count_delta` →
record it. `estimated_mean` = count-weighted mean of recorded interval means; `estimated_variance`
= `var(observed_means, ddof=1)`, only trusted with ≥ `MIN_VARIANCE_OBSERVATIONS = 3` observations.
The phase's total counts/sum come from the boundary snapshot delta (§3.4).

**Phases 2–4 (the driver):**
- guards: `total_count≤0` or empty → None; `total_sum` non-finite/<0 → None; `total_sum==0,
  count>0` → all percentiles 0.
- **finite bucket sums:** per bucket, `mean` = learned mean *iff* strictly inside `(lower,
  upper)` else the **midpoint**; `sum = count·mean`.
- **`+Inf` back-calc:** `inf_sum = total_sum − finite_sum`; `inf_avg = inf_sum/inf_count` (or
  `max_finite·1.5` fallback, and again if `inf_avg ≤ max_finite`); spread as a **uniform** with
  that mean over `[max_finite, 2·inf_avg − max_finite]`; **`inf_count==1` → return `[inf_avg]`**
  (linspace(…,1) would drop the whole tail sum). `inf_count = ceil(raw)` to keep ≥1.
- **finite observations with a sum constraint** (`_generate_observations_with_sum_constraint`):
  per bucket pick a shape by variance:
  - **F3 two-point mass** when `spread_coverage = 4σ/width < 0.01` (matches first two moments
    exactly: masses at `lower` and `a = mean + var/(mean−lower)`, prob `p = var/(var+(mean−lower)²)`);
  - **Blended** (50/50 shifted-uniform + variance-aware) when `spread<0.2 ∧ mean_offset<0.3`;
  - **Variance-aware** (truncated-normal-like, ±3σ clamp) otherwise;
  - **Shifted/pure uniform** fallback (center = learned mean, or the overall `avg` if one bucket
    holds ≥95% of observations — the narrow-distribution dominance case).
  Then a **Pass-2 sum fine-tune**: if `|target−generated|/target > 0.1%`, distribute the residual
  proportionally to each bucket's sum contribution, per-observation shift **capped at ±40% of
  bucket width** (keeps observations inside their bucket).
- final: concat finite+inf observations, `percentile([1,5,10,25,50,75,90,95,99])`.

All thresholds (`MIN_VARIANCE_OBSERVATIONS=3`, `_MAX_OBSERVATIONS=100k`, dominance `0.95`, F3
`0.01`, blended `0.2`/`0.3`, `spread=4σ/width`, ±3σ clamp, sum-tol `0.001`, shift-cap `0.4`, +Inf
`1.5×`) are load-bearing — pinned in a Rust `histogram` module with the paper cite. It is a
large, self-contained numeric module; guard it with a golden-vector parity test against the
Python.

---

## 5. Network-RTT calibration (`aiperf_runtime::network_latency`)

Feeds the `network_adjusted_*` catalog metrics: a constant mean RTT subtracted from
request-start-anchored latencies (§4.8 of the catalog scars).

### 5.1 The TCP-connect probe (domain invariants, port exactly)
- **Fresh unpooled connection per probe** — raw TCP connect timed with the `Clock`; a pooled
  HTTP client measures ~0 on reuse.
- **TLS skipped** — a plain TCP connect to the port even for `https://`, so `rtt_ns` is one
  uniform round-trip across http/https.
- **DNS resolved once at configure time & cached** — probes time pure TCP connect, not DNS.
- **Failures never crash the run** — timeout/refused → a failed sample (`success=false,
  rtt_ns=None`); only `OSError`-class caught.
- **Fire-and-forget interval loop** (`DEFAULT_PROBE_INTERVAL = 1.0s`) so a slow handshake near
  the timeout doesn't delay the next probe.

### 5.2 MIN_SAMPLES top-up + the manual bypass
- **`MIN_SAMPLES = 5`**: at `PROFILE_COMPLETE`, if a short run didn't reach 5 successful samples,
  fire synchronous back-to-back `probe_once()` up to `min_samples·2` attempts, bounded by
  **`COMPLETE_TOPUP_TIMEOUT = 3.0s`** (kept under the command-response budget so a slow endpoint
  can't stall completion). `CONNECT_TIMEOUT = 5.0s` per probe.
- **`--network-latency-mean` bypass**: a fixed mean RTT (ms→ns) supplied by the user; no probing
  service, no accumulator. Mutually exclusive with `--network-latency-automatic`.

### 5.3 Aggregation + delivery
The accumulator (a plain `Accumulator` impl, but delivered specially): `mean_rtt_ns` = a **flat
per-sample mean over every successful RTT across every target** (unweighted); per-target
percentile summaries use population std (ddof=0). Delivered via `set_network_rtt_ns` to the
metrics accumulator **before** summarize (so `network_adjusted_*` inject). RTT of 0/None → a
no-op (adjusted == raw), skip injection. In Rust this is a direct call in the finalize order the
parent spec fixes ("deliver network-RTT → summarize"), not a bus message.

---

## 6. What we do NOT port (the ZMQ/orjson artifact)

The server-metrics **NaN-taint-whole-histogram** + **drop-non-finite-and-warn** behavior exists
*only* because the Python pipeline ships each scrape over ZMQ where orjson coerces NaN→`null` and
invalidates the whole receiver batch, and because `HistogramTimeSeries` latches a truncated bucket
schema from a partial first sample. **The single-process Rust tool has no ZMQ boundary and no
orjson coercion** — an `f64` NaN survives in-process. So: drop individual non-finite samples where
it matters, but **do not taint the whole histogram** and do not port the warn-once machinery
verbatim; let the Rust histogram store accept per-scrape bucket sets. (This is the one place the
"carry the scars" rule is *overridden* — the scar was self-inflicted by the transport we deleted.)

---

## 7. Rust shape + scope

- **Reuse the parent `Accumulator` trait**: `GpuTelemetryAccumulator`, `ServerMetricsAccumulator`,
  `NetworkLatencyAccumulator` each `impl Accumulator` with `query_time_range` (half-open) for the
  gauge series + the shared `counter` module (phase-boundary snapshot + reset-clamp) + the parent
  kernel's `ddof=1` path for gauges. They are **side-channel accumulators** (record_type ≠
  `metric_records`), summarized separately in the finalize order, injecting `total_gpu_*` /
  `network_adjusted_*` into the main results.
- **Three producer modules** (`aiperf_runtime::gpu_telemetry`, `aiperf_runtime::server_metrics` — which owns its
  self-contained Prometheus/OpenMetrics parser and the `histogram` estimator —
  `aiperf_runtime::network_latency`) implement the shared side-channel accumulator seam. The dependency
  direction runs from the telemetry producers **toward the IO-free `aiperf_runtime::metrics_core` seam**;
  the core metrics engine is runtime-neutral and never depends back on a telemetry-specific
  collector. The scrape *managers* are plain Clock-paced tasks in the runner, not services, and
  they snapshot at the phase boundaries the runtime owns.
- **In:** the phase-boundary counter engine, GPU DCGM table + boundary energy + rollups, the
  server routing/fallback/auto-disable SM + vLLM/SGLang atlas + unit inference, the polynomial
  histogram estimator, the network-RTT probe + calibration + delivery.
- **Deferred:** PyNVML/AMDSMI local collectors (need bindings); custom-metrics CSV; the
  parquet/csv telemetry exporters (presentation plane); the realtime dashboard block.
- **Testing:** golden-vector parity for the counter delta (boundary snapshot + reset-clamp), the
  histogram estimator (the whole synthetic-observation pipeline vs numpy), and the unit-inference
  table; a fixture that replays a TRT-LLM `/metrics` JSON + a vLLM OpenMetrics body to exercise
  the routing/fallback/auto-disable SM.

## 8. Open questions

1. **Histogram estimator: exact numpy-percentile parity or distributional?** The estimator
   materializes samples then calls `np.percentile` (linear). Byte-exact for the estimator's own
   unit tests (golden vectors), distributional tolerance end-to-end.
