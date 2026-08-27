<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Telemetry side-channel parity audit

Domain: GPU telemetry (DCGM/pynvml/amdsmi), inference-server Prometheus scraping,
network-latency probing. Read against `docs/dev/python-rust-parity-gaps.md`
(2026-07-17); relevant pre-existing entries are P1.38 (telemetry config lowering),
P1.39 (server-metrics phase/histogram rules), P1.40 (GPU telemetry windows and
failure observability).

**Python baseline:** `origin/main` @ `bc359bf8fd`, read from the clean worktree at
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/`. Rust side is the working tree at
`/home/anthony/nvidia/projects/aiperf/ajc/rust/rust/`.

**Baseline-correction scope note.** The audit was first written against the local
feature branch's `src/aiperf/`, which is ahead of `origin/main`. In this domain only
two files differ from the baseline: `gpu_telemetry/dcgm_collector.py` (+23/−3) and
`gpu_telemetry/worker.py` (+245/−0, branch-only — the file does not exist on
`origin/main`). `worker.py` was cited by no finding, and the `dcgm_collector.py`
delta only *adds* the `collect_records_once` boundary seam without changing the
cadence-path deduplication that finding 7 rests on. Every other Python file in the
evidence set — `metrics/energy_efficiency_analyzer.py`,
`metrics/network_adjusted_analyzer.py`, all of `server_metrics/`, all of
`network_latency/`, `config/{gpu_telemetry,server_metrics,network_latency}.py`,
`common/mixins/base_metrics_collector_mixin.py`, and both telemetry console
exporters — is byte-identical between branch and baseline, so their citations and
line numbers are valid as written. Citations touching `dcgm_collector.py` have been
re-pointed at baseline line numbers below. No finding was withdrawn.

## Summary

The raw per-GPU telemetry plane is in good shape: every DCGM/AMD source field
both sides collect maps to the same output field name, the same unit, the same
scale factor, and the same gauge/counter classification, and the distribution
statistics (percentile set, interpolation, `ddof=1`) agree. The risk is
concentrated one layer up, in the *derived* energy plane and in failure
handling. Python emits twelve vendor-namespaced energy-efficiency metrics per
vendor (`nvidia_*`/`amd_*`); Rust emits four vendor-neutral ones, so eight
user-visible rows — including `Average GPU Power`, `Energy per Output Token`,
`Performance per Watt` and `Output Tokens per Second per Watt` — silently
disappear and the four survivors are renamed. Python also derives total energy by
integrating the power gauge when no energy counter exists; Rust does not, so
pynvml/amdsmi runs lose the entire energy family. Failure handling diverges in the
dangerous direction: one failed GPU phase-boundary scrape drops *all* GPU
telemetry in Rust (Python keeps the continuous series), an unreachable configured
DCGM URL is not distinguishable from a healthy one in the Rust export, and a
missing server-metrics phase-start snapshot makes Rust report a counter's entire
since-boot value as the phase delta. Nothing in this domain showed a wrong unit
or a wrong sign.

## Telemetry field mapping diff

Source → output mapping (`src/aiperf/gpu_telemetry/constants.py:58-68`,
`gpu_telemetry/dcgm_collector.py:23-27` (baseline) vs
`rust/runtime/src/gpu_telemetry/fields.rs:97-270`).

| Source metric | Python output field + unit (scale) | Rust output field + unit (scale) | Match? |
| --- | --- | --- | --- |
| `DCGM_FI_DEV_POWER_USAGE` | `nvidia_power_usage` W (×1) | `nvidia_power_usage` W (×1) | yes |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | `nvidia_energy_consumption` MJ (×1e-9), counter | `nvidia_energy_consumption` MJ (×1e-9), counter | yes |
| `DCGM_FI_DEV_GPU_UTIL` | `nvidia_gpu_utilization` % (×1) | same | yes |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | `nvidia_memory_utilization` % (×1) | same | yes |
| `DCGM_FI_DEV_FB_USED` | `nvidia_memory_used` GB (×1.048576e-3) | same | yes |
| `DCGM_FI_DEV_GPU_TEMP` | `nvidia_temperature` °C (×1) | same | yes |
| `DCGM_FI_DEV_ENC_UTIL` | `nvidia_encoder_utilization` % (×1) | same | yes |
| `DCGM_FI_DEV_DEC_UTIL` | `nvidia_decoder_utilization` % (×1) | same | yes |
| `DCGM_FI_PROF_SM_ACTIVE` | `nvidia_sm_utilization` % (×100) | same | yes |
| `DCGM_FI_DEV_XID_ERRORS` | `nvidia_xid_errors` count (×1), counter | same | yes |
| `DCGM_FI_DEV_POWER_VIOLATION` | `nvidia_power_violation` µs (×1e-3), counter | same | yes |
| `DCGM_FI_DEV_JPG_UTIL` | not mapped from DCGM (the field name exists for pynvml, `constants.py` config list) | `nvidia_jpg_utilization` % (×1) | Rust-only, out of scope |
| amdsmi `amd_power` | `amd_power` W (×1) | `amd_power` W (×1) | yes |
| amdsmi `amd_energy_consumption` | `amd_energy_consumption` MJ (×1e-12), counter | same | yes |
| amdsmi `amd_gfx/umc/mm_activity` | `%` (×1) | same | yes |
| amdsmi `amd_memory_used` | GB (×1e-9) | GB (×1e-9) | yes |
| amdsmi `amd_temperature` | °C (×1) | °C (×1) | yes |
| amdsmi `amd_ecc_uncorrectable` | count, counter | count, counter | yes |
| amdsmi `amd_throttle_status` | count, gauge | count, gauge | yes |

Derived energy plane (`src/aiperf/metrics/types/power_efficiency_metrics.py`
vs `rust/runtime/src/metrics_core/catalog.rs:250-253,1736-1771`):

| Derived quantity | Python tag + unit | Rust tag + unit | Match? |
| --- | --- | --- | --- |
| fleet power | `nvidia_total_gpu_power` / `amd_total_gpu_power`, W | `total_gpu_power`, W | renamed, vendor fan-out lost |
| fleet energy | `nvidia_total_gpu_energy` / `amd_…`, J | `total_gpu_energy`, J | renamed, vendor fan-out lost |
| tokens per joule | `nvidia_output_tokens_per_joule` / `amd_…`, tokens/J | `output_tokens_per_joule`, tokens/J | renamed |
| energy per user | `nvidia_energy_per_user` / `amd_…`, J/user | `energy_per_user`, J/user | renamed |
| average power | `nvidia_average_gpu_power` / `amd_…`, W | absent | **dropped** |
| energy per output token | `nvidia_energy_per_output_token`, mJ/token | absent | **dropped** |
| energy per total token | `nvidia_energy_per_total_token`, mJ/token | absent | **dropped** |
| energy per request | `nvidia_energy_per_request`, J/request | absent | **dropped** |
| energy-delay product | `nvidia_energy_delay_product`, J·s | absent | **dropped** |
| performance per watt | `nvidia_performance_per_watt`, req/s/W | absent | **dropped** |
| output TPS per watt | `nvidia_output_tps_per_watt`, tokens/s/W | absent | **dropped** |
| goodput per watt | `nvidia_goodput_per_watt`, goodput/W | absent | **dropped** |

## Findings

### 1. Eight of twelve GPU energy-efficiency metrics are silently absent in Rust, and the four survivors are renamed

**Severity:** P0
**Status:** NEW (P1.38 covers telemetry *config* lowering, P1.40 covers windows and
failure observability; neither states the derived-metric set or tag rename)

**Python evidence:** `src/aiperf/metrics/energy_efficiency_analyzer.py:70-99` maps
twelve metric keys per vendor, and `_analyze_vendor` /
`_energy_ratio_metrics` / `_per_watt_metrics` emit all of them
(`:236-278`, `:303-351`, `:353-383`):

```python
    if avg_power_w > 0:
        out.append(_result(vendor_metrics["average_power"], avg_power_w))
    out.append(_result(vendor_metrics["total_energy"], total_energy_j))
    out += self._energy_ratio_metrics(total_energy_j, metric, concurrency, vendor_metrics)
    out += self._per_watt_metrics(avg_power_w, metric, vendor_metrics)
```

Tags are vendor-prefixed, e.g. `src/aiperf/metrics/types/power_efficiency_metrics.py:111-141`:

```python
class NvidiaAverageGpuPowerMetric(_InjectedEnergyMetric):
    tag = "nvidia_average_gpu_power"
    header = "Average GPU Power"
    unit = PowerMetricUnit.WATT
```

**Rust evidence:** `rust/runtime/src/gpu_telemetry/accumulator.rs:338-367` injects
exactly four tags and nothing else:

```rust
        if power_gpu_count > 0 {
            summary.injections.insert(MetricTag::TotalGpuPower, ...);
        }
        if energy_gpu_count > 0 {
            summary.injections.insert(MetricTag::TotalGpuEnergy, ...);
            if total_energy > 0.0 {
                ... MetricTag::OutputTokensPerJoule ...
                ... MetricTag::EnergyPerUser ...
```

The catalog has no other efficiency tag at all
(`rust/runtime/src/metrics_core/catalog.rs:94-97`, `:250-253`), so the missing
eight are not "declared but unpopulated" — they do not exist:

```rust
    TotalGpuPower,
    TotalGpuEnergy,
    OutputTokensPerJoule,
    EnergyPerUser,
```

**Observable user impact:** a run with GPU telemetry that printed twelve energy
rows under Python prints four under Rust, and the four have different tags in
`profile_export_aiperf.json` / CSV (`total_gpu_power` instead of
`nvidia_total_gpu_power`). Any dashboard, script, or regression baseline keyed on
the Python tags reads "metric missing" for every one of the twelve. No error or
warning is emitted. `Average GPU Power` is the most consequential loss: it was
the denominator-bearing power number users compare across runs.

**Confidence:** high (both sides read directly; the Rust tag enum is exhaustive).

### 2. Rust drops Python's power-integration energy fallback, so collectors without an energy counter lose the whole energy family

**Severity:** P1
**Status:** NEW

**Python evidence:** `src/aiperf/metrics/energy_efficiency_analyzer.py:280-301`:

```python
        if energy_count > 0 and energy_j > 0:
            avg = energy_j / duration_s if duration_s > 0 else 0.0
            return energy_j, avg, EnergySource.DCGM_COUNTER
        if power_count > 0 and power_w > 0 and duration_s > 0:
            return power_w * duration_s, power_w, EnergySource.POWER_INTEGRATION
        return 0.0, 0.0, EnergySource.UNAVAILABLE
```

**Rust evidence:** `rust/runtime/src/gpu_telemetry/accumulator.rs:428-443` — energy
comes only from the counter delta, with no gauge-integration branch, and
`:348-367` gates every energy-derived injection on `energy_gpu_count > 0`:

```rust
    fn total_energy(&self, boundary: &GpuPhaseBoundary) -> (f64, usize) {
        self.series.keys().filter_map(|key| {
            ["nvidia_energy_consumption", "amd_energy_consumption"].into_iter()
                .find_map(|name| boundary_counter_delta(
                    boundary.start.counter(key, name),
                    boundary.end.counter(key, name),
                ).map(|delta| delta.delta * MEGAJOULE_TO_JOULE))
        })
        .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
    }
```

**Observable user impact:** on a collector that exposes power but no cumulative
energy (`--gpu-telemetry-collector pynvml`/`amdsmi`, or a DCGM exporter whose
field list omits `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`), Python reports
`total_gpu_energy` and `output_tokens_per_joule` from `power × duration`; Rust
reports neither. `total_gpu_power` still appears, so the user sees a
half-populated family rather than an explicit "no energy source" signal.

**Confidence:** high.

### 3. One failed GPU phase-boundary scrape silently discards the entire GPU telemetry section

**Severity:** P1
**Status:** KNOWN(still-true) — specific consequence of P1.40, which states only
that "Rust uses exact boundaries and often logs warnings before producing empty
telemetry"

**Python evidence:** gauges are computed from the continuous scrape history with a
time filter, independent of any boundary scrape
(`src/aiperf/gpu_telemetry/accumulator.py:403-438`); a failed final scrape costs
at most the trailing sample. The boundary scrape itself is best-effort and only
raises for the caller to log (`src/aiperf/gpu_telemetry/manager.py:343-354`):

```python
        for telemetry_source_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
            except Exception as exc:  # one failed endpoint should not skip others
                errors.append(f"{telemetry_source_url}: {type(exc).__name__}: {exc}")
```

**Rust evidence:** `rust/runtime/src/engine/gpu_telemetry.rs:389-437` — a failed
final scrape only warns, and the boundary is installed *only* if both combined
snapshots exist; `accumulator.rs:487-501` returns an empty summary when no
boundary is set:

```rust
        if let (Some(start), Some(end)) = (
            combine_snapshots(&start_snapshots, start_ns),
            combine_snapshots(&end_snapshots, end_ns),
        ) {
            ... set_phase_boundary(boundary.clone());
        }
```

```rust
    fn export_results(&self, context: &ExportContext) -> Self::Summary {
        let Some(boundary) = self.phase_boundary.as_ref() else {
            return GpuTelemetrySummary::default();
        };
```

**Observable user impact:** with the common single-DCGM-endpoint setup, a
transient failure of the one closing scrape (exporter restart, 503, connection
reset) removes every per-GPU power/utilization/temperature series and every
efficiency metric from the report, even though hundreds of cadence scrapes
succeeded. Python would still print the full per-GPU tables. The only signal is a
`warn!` line in the log.

**Confidence:** high.

### 4. Rust reports every configured GPU telemetry endpoint as successful, hiding unreachable ones

**Severity:** P1
**Status:** KNOWN(still-true) — the "structured status/errors" half of P1.40; the
`endpoints_configured == endpoints_successful` identity is the concrete mechanism

**Python evidence:** the controller overwrites the accumulator's optimistic
summary with the manager's real configured/reachable split
(`src/aiperf/controller/system_controller.py:684-690`):

```python
                telemetry_results.summary.endpoints_configured = (
                    self._telemetry_endpoints_configured
                )
                telemetry_results.summary.endpoints_successful = (
                    self._telemetry_endpoints_reachable
                )
```

and the console exporter renders the ratio plus a per-endpoint failure list
(`src/aiperf/exporters/gpu_telemetry_console_exporter.py:117-140`, `:219-222`).
The configured set includes user-supplied URLs whether or not they are reachable
(`src/aiperf/gpu_telemetry/manager.py:161-167`).

**Rust evidence:** `rust/runtime/src/export/genai_perf.rs:466-475` builds both
fields from the same list — the endpoints that actually produced series:

```rust
    let raw_urls: Vec<Value> = endpoint_order.iter().map(...).collect();
    let mut summary = Map::new();
    summary.insert("endpoints_configured".to_owned(), Value::Array(raw_urls.clone()));
    summary.insert("endpoints_successful".to_owned(), Value::Array(raw_urls));
```

**Observable user impact:** `aiperf --gpu-telemetry-url dcgm-a:9400,dcgm-b:9400`
with `dcgm-b` down reports `endpoints_configured == endpoints_successful ==
[dcgm-a]` in `telemetry_data.summary`; the user believes both GPUs' hosts were
measured and that the fleet power number covers the whole fleet. Python reports
"1/2 endpoints reachable" and names the failure. Note the *server-metrics* path
keeps the distinction (`rust/runtime/src/engine/server_metrics.rs:161-162`), so
this is GPU-telemetry-specific.

**Confidence:** high.

### 5. A missing server-metrics phase-start snapshot makes Rust report a counter's whole since-boot value as the phase delta

**Severity:** P1
**Status:** KNOWN(still-true) — P1.39 says Python uses "continuous scrape history
plus pre-phase references" while Rust uses "forced phase-boundary snapshots"; the
zero-baseline consequence is not stated

**Python evidence:** `src/aiperf/server_metrics/export_stats.py:314-337` falls back
to the first in-window sample when no pre-window reference exists, so the delta is
always relative to an observed value:

```python
    reference_value = (
        float(time_series.values[reference_idx])
        if reference_idx is not None
        else float(filtered_values[0])
    )
    ...
    raw_delta = float(filtered_values[-1]) - reference_value
```

(`get_reference_idx` is the last sample strictly before `start_ns`,
`src/aiperf/server_metrics/storage.py:449-475`.)

**Rust evidence:** `rust/runtime/src/server_metrics/accumulator.rs:765-772`
substitutes zero for an absent start value:

```rust
    let end = scalar_value(end)?;
    // A counter absent from the start boundary scrape appeared during the phase
    // ... so its value at phase start was zero. Only a missing *end* value omits the series.
    let start = scalar_value(start).unwrap_or(0.0);
    let total = (end - start).max(0.0);
```

and the start snapshot map only contains endpoints whose *start-barrier* scrape
succeeded (`rust/runtime/src/engine/server_metrics.rs:240-264`): a transport error
there is warned and skipped, yet the phase still runs and the closing snapshot
still lands.

**Observable user impact:** if the phase-start scrape of a long-running server
fails (503 during model warmup, brief connection reset) but later scrapes
succeed, every counter in `server_metrics_export` reports its full since-boot
total for the phase — e.g. `vllm:prompt_tokens` totalling millions of tokens for a
60-second run — and the derived `*_rate` is inflated by the same factor. Python
reports the correct in-window delta. The comment's assumption (absent at start ⇒
created during the phase) is sound for lazily-created families but silently
misfires on a failed boundary scrape.

**Confidence:** high for the mechanism; the frequency depends on how often a
boundary scrape fails in practice.

### 6. Rust collapses mixed-vendor fleets into one blended power/energy number

**Severity:** P2
**Status:** NEW

**Python evidence:** `src/aiperf/metrics/energy_efficiency_analyzer.py:193-216`
fans out over `gpu.available_platforms()` and emits an independent metric family
per vendor; `src/aiperf/gpu_telemetry/accumulator.py:503-523` and `:525-552` sum
power/energy only over GPUs of the selected platform.

**Rust evidence:** `rust/runtime/src/gpu_telemetry/accumulator.rs:411-426` picks
whichever of the two vendor field names a series carries and folds everything into
a single scalar:

```rust
                ["nvidia_power_usage", "amd_power"]
                    .into_iter()
                    .find_map(|name| { ... })
            })
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
```

**Observable user impact:** on a mixed NVIDIA+AMD host, Python prints
`nvidia_total_gpu_power` and `amd_total_gpu_power` separately; Rust prints one
`total_gpu_power` equal to their sum, and `output_tokens_per_joule` is computed
against the combined energy. The unit is right, the attribution is not, and the
name does not disclose the mixing. Single-vendor runs (the overwhelmingly common
case) are unaffected, hence P2.

**Confidence:** high.

### 7. Python deduplicates unchanged DCGM scrape bodies; Rust keeps every scrape, changing gauge sample counts and dispersion

**Severity:** P2
**Status:** NEW

**Python evidence:** the shared collector mixin hashes each body
(`common/mixins/base_metrics_collector_mixin.py:602-611`, byte-identical on branch
and baseline):

```python
            response_hash = hash(text)
            is_duplicate = response_hash == self._last_response_hash
            self._last_response_hash = response_hash
```

and the DCGM collector drops duplicates on the cadence path
(baseline `gpu_telemetry/dcgm_collector.py:96-100`):

```python
        fetch_result = await self._fetch_metrics_text()
        if fetch_result.is_duplicate:
            return
        records = self._parse_metrics_to_records(fetch_result.text)
        await self._send_records_via_callback(records)
```

The branch refactors this same early return behind a `collect_records_once`
helper that adds a `bypass_dedup` boundary seam; the cadence path still returns on
a duplicate body, so the finding holds identically on the baseline.

**Rust evidence:** `rust/runtime/src/gpu_telemetry/source.rs:161-182` performs no
body comparison; every 200 response is decoded and ingested, and
`rust/runtime/src/gpu_telemetry/accumulator.rs:279-299` appends each one as a
distinct sample:

```rust
    async fn scrape(&self, _mode: GpuScrapeMode) -> Result<Option<GpuScrape>, GpuTelemetryError> {
        ...
        let timestamp_ns = self.clock.now_ns();
        self.decoder.decode(&self.display_url, timestamp_ns, &body).map(Some)
    }
```

**Observable user impact:** DCGM exporters typically refresh slower than the
333 ms scrape cadence, so the same body is served repeatedly. Python's per-GPU
gauge statistics are computed over *changed* samples; Rust's over *all* scrapes.
`count` differs outright, and `std`/percentiles shift because repeated values are
weighted differently. `avg`, `min`, and `max` move only slightly. (This affects
GPU telemetry only: the server-metrics path stores duplicates on both sides —
`src/aiperf/server_metrics/storage.py:145-174` vs
`rust/runtime/src/server_metrics/accumulator.rs:216-219`.)

**Confidence:** high for the mechanism; magnitude depends on the exporter's
refresh interval.

### 8. Histogram bucket regressions: Python withholds buckets and percentiles, Rust clamps and publishes them

**Severity:** P2
**Status:** KNOWN(still-true) — same family as P1.39's non-finite-bucket clause,
now confirmed for the reset path too

**Python evidence:** `src/aiperf/server_metrics/export_stats.py:796-831` — any
negative bucket delta discards the whole bucket map, which also suppresses the
estimated percentiles:

```python
        if bucket_delta < 0:
            bucket_reset_detected = True
            bucket_deltas = None
            break
    ...
    estimated = None
    if bucket_deltas:
```

**Rust evidence:** `rust/runtime/src/server_metrics/accumulator.rs:815-847` clamps
each bucket independently and always computes percentiles:

```rust
            let delta = (end.buckets[&name] - start_bucket).max(0.0) as u64;
    ...
    let percentiles = compute_estimated_percentiles(&cumulative, &learned, sum, count)
```

**Observable user impact:** behind a load balancer or across a server restart,
Python omits the histogram's `buckets` and percentile estimates (a visible gap
plus a warning); Rust publishes clamped buckets and percentile estimates derived
from an inconsistent baseline, with no warning. The user gets plausible-looking
latency percentiles that are wrong.

**Confidence:** high.

### 9. Counter-suffix unit inference is coarser in Rust for error/block counters

**Severity:** P2
**Status:** NEW

**Python evidence:** `src/aiperf/server_metrics/units.py:43-56` distinguishes
counter flavors:

```python
    "_errors": GenericMetricUnit.ERRORS,
    "_errors_total": GenericMetricUnit.ERRORS,
    "_error_count": GenericMetricUnit.ERRORS,
    "_error_count_total": GenericMetricUnit.ERRORS,
    "_blocks": GenericMetricUnit.BLOCKS,
    "_blocks_total": GenericMetricUnit.BLOCKS,
    "_block_count": GenericMetricUnit.BLOCKS,
```

**Rust evidence:** `rust/runtime/src/server_metrics/units.rs:62-96` keeps
tokens/requests but maps all seven error/block suffixes to `Unit::Count`:

```rust
        ("_error_count_total", Unit::Count),
        ("_blocks_total", Unit::Count),
        ("_errors_total", Unit::Count),
        ("_block_count", Unit::Count),
        ("_errors", Unit::Count),
        ("_blocks", Unit::Count),
```

(`_error_count` has no Rust entry at all and falls through to `_count` → `Count`.)

**Observable user impact:** a scraped `*_errors_total` / `*_blocks_total` metric is
labelled `count` in the Rust `server_metrics_export` unit metadata instead of
`errors` / `blocks`. Values are identical; only the unit string a reader or
downstream renderer sees changes.

**Confidence:** high.

## Checked and consistent

- **Metrics URL normalization.** `src/aiperf/common/metric_utils.py`
  (`normalize_metrics_endpoint_url`) and
  `rust/runtime/src/config/model/telemetry.rs:411-430` /
  `rust/runtime/src/server_metrics/source.rs:345-362` agree: prepend `http://` when
  no scheme, strip trailing slashes, append `/metrics` when absent — including the
  path-preserving behavior (`http://host/v1` → `http://host/v1/metrics`). Rust
  additionally translates `grpc://`/`grpcs://` (Rust-only, out of scope). Python
  raises on an empty URL where Rust would build `http:///metrics`; unreachable in
  practice because both surfaces reject empty list entries earlier.
- **Default DCGM endpoints.** Python `["http://localhost:9400/metrics",
  "http://localhost:9401/metrics"]` (`src/aiperf/common/environment.py`) vs Rust
  `["localhost:9400", "localhost:9401"]` normalized at
  `rust/runtime/src/config/model/telemetry.rs:358-372` — identical after
  normalization, same first-seen dedup with user-supplied extras appended.
- **Default intervals and timeouts.** 0.333 s collection interval and 10 s
  reachability timeout for both GPU and server metrics
  (`src/aiperf/common/environment.py` vs
  `rust/runtime/src/config/model/telemetry.rs` `COLLECTION_INTERVAL_NS =
  333_000_000`, `REACHABILITY_TIMEOUT_NS = 10_000_000_000`).
- **Enabled-by-default.** GPU telemetry and server metrics default enabled;
  network latency defaults disabled with a 1.0 s probe interval, on both sides
  (`src/aiperf/config/{gpu_telemetry,server_metrics,network_latency}.py` vs
  `rust/runtime/src/config/model/telemetry.rs`).
- **Server-metrics target derivation.** Both derive from `endpoint.urls` plus
  user-specified `server_metrics.urls`, normalized and deduped in first-seen order
  (`src/aiperf/server_metrics/manager.py:91-108` vs
  `rust/runtime/src/config/resolve.rs:1562-1573`).
- **Prometheus family filtering.** Both skip `_created`, `_uptime`, `*_uptime_*`,
  and `summary` families, and both dedupe scalar samples by label set with
  last-value-wins (`src/aiperf/server_metrics/data_collector.py:286-322,401-456`
  vs `rust/runtime/src/server_metrics/parser.rs:198-243,362-364`). Both strip the
  `_total` suffix from declared counter names.
- **TRT-LLM `/prometheus/metrics` fallback.** Same one-shot probe, same
  precondition (URL ends with `/metrics`, not already the fallback path), same
  terminal disablement afterwards
  (`src/aiperf/server_metrics/data_collector.py:186-244` vs
  `rust/runtime/src/server_metrics/source.rs:252-341`).
- **Distribution statistics.** Percentile set `[1,5,10,25,50,75,90,95,99]`,
  linear interpolation, sample standard deviation (`ddof=1`) for gauges on both
  sides (`src/aiperf/server_metrics/export_stats.py:533-547` and
  `src/aiperf/common/models/telemetry_models.py` vs
  `rust/runtime/src/metrics_core/kernel.rs:17,97-157`).
- **Counter reset clamping.** `max(delta, 0)` on both sides for server counters
  (`export_stats.py:354-355` vs `accumulator.rs:771`).
- **Per-GPU reporting and labels.** Both key series by endpoint then GPU UUID and
  carry `gpu` index, `gpu_uuid`, `model_name`, `platform`, plus optional
  `hostname`/`namespace`/`pod`; the genai-perf export nests
  `endpoints → gpus → gpu_<index> → metrics` identically
  (`src/aiperf/gpu_telemetry/accumulator.py:315-345` vs
  `rust/runtime/src/gpu_telemetry/accumulator.rs:508-527`,
  `rust/runtime/src/export/genai_perf.rs:363-464`).
- **Fleet power semantics.** Both sum the per-GPU *mean* power over the window
  (not the max, not the last value) and report the contributing GPU count
  (`accumulator.py:403-438` vs `accumulator.rs:411-426`).
- **Energy unit chain.** DCGM mJ → MJ at scrape (×1e-9) then MJ → J at
  aggregation (×1e6) on both sides
  (baseline `dcgm_collector.py:24`, `accumulator.py:479` vs `fields.rs:106-113`,
  `accumulator.rs:27,439`).
- **Network-latency probe method and formula.** Fresh TCP connect timed to
  handshake completion, DNS pre-resolved once, immediate first probe then interval
  cadence, `MIN_SAMPLES` top-up bounded by a shared deadline and `2 × floor`
  attempts per target (`src/aiperf/network_latency/probe.py`,
  `manager.py:144-188` vs `rust/runtime/src/network_latency/probe.rs`,
  `rust/runtime/src/engine/network_latency.rs:301-348`). The adjustment formula is
  `max(latency − mean_rtt, 0)` with nanosecond internal storage and millisecond
  display on both sides (`src/aiperf/metrics/network_adjusted_analyzer.py` vs
  `rust/runtime/src/metrics_core/accumulator.rs`,
  `rust/runtime/src/metrics_core/catalog.rs`), and network-latency aggregate stats
  use population standard deviation on both sides.
- **Sidecar phase scope.** GPU telemetry and network-latency probes attach only to
  phases that are not `exclude_from_results`
  (`rust/runtime/src/engine/execute/compose_sidecars.rs:22-30`), matching Python's
  PROFILE_START/PROFILE_COMPLETE scoping.
- **Server-metrics configured-vs-successful split** is preserved in Rust
  (`rust/runtime/src/engine/server_metrics.rs:161-162`) — unlike the GPU path in
  finding 4.

## Unverified / needs runtime check

- **Percent-vs-ratio unit inference on parenthesis-free ranges.** Rust's
  `PERCENT_RANGE` (`rust/runtime/src/server_metrics/units.rs:115-117`) makes the
  parentheses optional where Python's `_PERCENT_RANGE_PATTERN`
  (`src/aiperf/server_metrics/units.py:311-315`) requires them for a bare numeric
  range. A HELP string like `... range 0-100` would infer `percent` in Rust and
  fall through in Python. UNVERIFIED: I did not find a real vLLM/SGLang/DCGM HELP
  string that takes this branch, so the divergence may be unreachable. Needs a
  scan of actual exposition text from the supported backends (or a table-driven
  test over both regex sets) to confirm it is user-observable.
- **Energy window offset magnitude.** Python widens the energy counter window by
  `FINAL_SCRAPE_GRACE_NS` (666 ms) while dividing by the exact profiling duration
  (`src/aiperf/gpu_telemetry/accumulator.py:539-544` and the comment at
  `energy_efficiency_analyzer.py:172-177`); Rust uses exact boundary snapshots
  (`rust/runtime/src/gpu_telemetry/accumulator.rs:392-409`). This is KNOWN
  (P1.40). The direction is clear — Python's total energy is biased slightly high
  on short runs — but quantifying the resulting `total_gpu_energy` /
  `output_tokens_per_joule` gap needs a side-by-side run against a deterministic
  DCGM fixture at a few run lengths.
- **Server-metrics counter-rate denominator.** Python divides the delta by the
  span between its reference sample and the last in-window sample
  (`export_stats.py:337-359`); Rust divides by the exact phase duration
  (`accumulator.rs:772`). Each is internally consistent with its own numerator, so
  the difference should be bounded by roughly one scrape interval, but confirming
  it stays within that bound needs a run with a known constant request rate.
