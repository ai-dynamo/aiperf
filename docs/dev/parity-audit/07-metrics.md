<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Metrics definition and math parity audit

**Python baseline:** `/mnt/4tb/aiperf-parity-py-main/src/aiperf/`, git rev
`bc359bf8fd` (`origin/main`). All `src/aiperf/...` paths below are relative to
that tree. Rust citations are unchanged (`origin/main` has no `rust/` tree).

**Baseline-correction scope note.** An earlier pass of this audit read the
in-repo `src/aiperf/` feature branch. Of the Python files this report relies on,
four differ from the baseline and were re-verified in full:
`metrics/types/inter_token_latency_metric.py` (branch removed the whole
first-content-chunk divisor block), `records/inference_result_parser.py` (branch
removed the `first_content_chunk_tokens` plumbing),
`metrics/types/osl_mismatch_metrics.py`, and
`metrics/types/usage_diff_metrics.py` (both header-string only). A fifth file
outside the four, `common/models/record_models.py`, also differs (branch removed
`first_content_chunk_completion_tokens` and the `TokenCounts` field) and was
re-verified with them. Every other file cited here is byte-identical to the
baseline, so its line numbers stand. Re-verification changed one verdict
(inter-token latency, now confirmed at parity for a different reason than
originally stated) and added one finding (#8).

## Summary

The core inference math is in far better shape than the metric *inventory*. I
diffed the Python `MetricRegistry` (139 registered tags, dumped programmatically)
against the Rust static catalog (147 definitions, dumped through
`aiperf metrics list`) plus both sides' non-registry injected families
(sweep-line, derived-latency, energy, replay-lag). Percentile band, interpolation
method, population standard deviation, duration-weighted sweep-line statistics,
the error-adjusted `adj_*` construction, and every throughput/latency formula I
checked are byte-for-byte equivalent designs — that part of the port is faithful.

The headline risks are all inventory and boundary risks. First, the GPU
energy-efficiency family is silently *renamed and truncated*: Python emits 12
vendor-scoped metrics per vendor (`nvidia_*`/`amd_*`), Rust emits 4 unprefixed
tags and drops the other 8 entirely, so a user's `nvidia_energy_per_request` or
`nvidia_performance_per_watt` key simply vanishes from the summary with no error.
Second, the fixed-schedule replay-lag family (`replay_sched_lag_p50/p90/p99`,
`replay_sched_degraded`) has no Rust counterpart at all, and with it goes the
run-level "schedule degraded" warning — a trace-replay user loses the one signal
that tells them the load generator, not the server, was late. Third, the
request-latency terminal boundary (already logged as P1.29 and still true) is the
one *numeric* divergence with wide blast radius: Python stops at the last content
chunk, Rust stops when the transport stream closes, and that difference
propagates into `benchmark_duration` and therefore every rate denominator.

Lower-severity inventory slips round it out: `context_overflow_count` is
implemented in Rust but never wired into any report, the HTTP byte metrics
changed their unit spelling from `B` to `bytes`, and six display headers changed
string while keeping their tag (findings 6 and 8).

Inter-token latency, the metric most at risk here, is at parity. Both sides
divide the `request_latency - ttft` decode window by
`OSL - first_content_chunk_tokens`, with the same `--per-chunk-usage` opt-in
gate, the same three config validations, the same inconsistent-count fallback to
`OSL - 1`, and the same warn-once behaviour — see "Checked and consistent". Its
per-request *value* still moves, but only through finding 3's numerator.

## Metric tag inventory diff

Method: Python side enumerated from `MetricRegistry` via a throwaway dump script
(139 tags with header/unit/display_unit/flags); Rust side from
`aiperf metrics list` (147 definitions, `aiperf.` prefix stripped), which renders
the same `Definition` structs the report emits (`spec!` in
`rust/runtime/src/metrics_core/catalog.rs:1032` builds `def:` inline, and
`rust/runtime/src/metrics_core/definition.rs:124` indexes exactly those). 109
tags are present on both sides. Injected (non-registry) families were then
reconciled by hand, because both sides emit metrics that never appear in a
registry.

Shared-tag summary — of the 109 tags present in both registries, six differ in
unit or header spelling, and none differ in formula:

| tag | in Python | in Rust | same unit? | same formula? |
| --- | --- | --- | --- | --- |
| `http_req_data_sent` | yes | yes | **no** (`B` vs `bytes`) | yes |
| `http_req_data_received` | yes | yes | **no** (`B` vs `bytes`) | yes |
| `osl_mismatch_diff_pct` | yes | yes | yes; **header differs** (`OSL Mismatch Diff %` vs `OSL Mismatch Diff`) | yes |
| `usage_prompt_tokens_diff_pct` | yes | yes | yes; **header differs** (`Usage Prompt Diff %` vs `Usage Prompt Diff`) | yes |
| `usage_completion_tokens_diff_pct` | yes | yes | yes; **header differs** (`Usage Completion Diff %` vs `Usage Completion Diff`) | yes |
| `usage_reasoning_tokens_diff_pct` | yes | yes | yes; **header differs** (`Usage Reasoning Diff %` vs `Usage Reasoning Diff`) | yes |
| other 103 shared tags | yes | yes | yes | yes (spot-checked; see "Checked and consistent") |

The four `*_diff_pct` header rows were missed on the first pass because the
branch checkout had already dropped the trailing `%` from all four; they are
finding 8.

Python-registry tags with no Rust counterpart (30):

| tag group | tags | in Rust |
| --- | --- | --- |
| NVIDIA energy | `nvidia_average_gpu_power`, `nvidia_energy_delay_product`, `nvidia_energy_per_output_token`, `nvidia_energy_per_request`, `nvidia_energy_per_total_token`, `nvidia_energy_per_user`, `nvidia_goodput_per_watt`, `nvidia_output_tokens_per_joule`, `nvidia_output_tps_per_watt`, `nvidia_performance_per_watt`, `nvidia_total_gpu_energy`, `nvidia_total_gpu_power` | partially, renamed (finding 1) |
| AMD energy | same 12 with `amd_` prefix | partially, renamed (finding 1) |
| replay lag | `replay_sched_lag_p50`, `replay_sched_lag_p90`, `replay_sched_lag_p99`, `replay_sched_degraded`, `replay_send_schedule_offset` | **no** (finding 2) |
| agent scenario | `context_overflow_count` | computed but unreported (finding 4) |

Rust-catalog tags with no Python *registry* counterpart (38) — most are not
actually Rust-only, because Python emits them outside the registry:

| tag group | Python counterpart |
| --- | --- |
| `effective_concurrency`, `effective_decode_throughput`, `effective_prefill_throughput`, `effective_decode_concurrency`, `effective_prefill_concurrency`, `effective_total_throughput`, `effective_decode_throughput_per_user`, `effective_prefill_throughput_per_user`, `tokens_in_flight` | `src/aiperf/analysis/sweepline.py:53` `SWEEP_LINE_METRIC_SPECS` — identical tag, header, unit, scale |
| `active_decode_throughput`, `active_prefill_throughput`, `active_decode_throughput_per_user`, `active_prefill_throughput_per_user`, `active_total_throughput` | `src/aiperf/analysis/sweepline.py:162` `_compute_active_variants` — identical tag, header, unit |
| `credit_to_start_latency`, `effective_latency` | `src/aiperf/metrics/derived_latency.py:92,122` — same tag and unit, **different header string** (finding 6) |
| `total_gpu_power`, `total_gpu_energy`, `output_tokens_per_joule`, `energy_per_user` | Python's vendor-prefixed variants (finding 1) |
| `accuracy.*`, `analyzer.*`, `avg_round_trip_time`, `time_to_last_round_trip`, `image_samples_per_second`, `total_num_images`, `effective_image_samples_per_second*`, `active_image_samples_per_second` | genuinely Rust-only — out of scope |

## Findings

### 1. GPU energy-efficiency metrics lose their vendor prefix and 8 of 12 metrics disappear

**Severity:** P1
**Status:** NEW (the pre-existing backlog covers GPU telemetry *windows and
failure observability* in P1.40, `docs/dev/python-rust-parity-gaps.md:918`, not
the tag set)

**Python evidence:** `src/aiperf/metrics/types/power_efficiency_metrics.py:14`
documents the scheme, and each of 24 classes carries a vendor-scoped tag:

```
# Each metric exists in an NVIDIA variant (``nvidia_`` prefix,
# ... GPU_POWER_EFFICIENCY_AMD console group). The analyzer emits only
# the variants whose vendor actually reported data during the run.
```

The 12 NVIDIA tags (headers and units taken from the registry dump) are
`nvidia_average_gpu_power` (W), `nvidia_energy_delay_product` (J*s),
`nvidia_energy_per_output_token` (mJ/token), `nvidia_energy_per_request`
(joules/request), `nvidia_energy_per_total_token` (mJ/token),
`nvidia_energy_per_user` (joules/user), `nvidia_goodput_per_watt`
(good-req/s/W), `nvidia_output_tokens_per_joule` (tokens/J),
`nvidia_output_tps_per_watt` (tokens/sec/W), `nvidia_performance_per_watt`
(requests/sec/W), `nvidia_total_gpu_energy` (J), `nvidia_total_gpu_power` (W) —
declared at `power_efficiency_metrics.py:64,111,115,123,127,135,139,147,151,159,163,171,175,183,187,195,202`,
with the AMD mirror at `power_efficiency_metrics.py:213-355`.

**Rust evidence:** the catalog has four unprefixed energy tags and nothing else.
`rust/runtime/src/metrics_core/catalog.rs:250`:

```rust
Self::TotalGpuPower => "total_gpu_power",
Self::TotalGpuEnergy => "total_gpu_energy",
Self::OutputTokensPerJoule => "output_tokens_per_joule",
Self::EnergyPerUser => "energy_per_user",
```

with their specs at `rust/runtime/src/metrics_core/catalog.rs:1736-1771`
(`"Total GPU Power"`/Watt, `"Total GPU Energy"`/Joule,
`"Output Tokens per Joule"`/TokensPerJoule, `"Energy per User"`/JoulesPerUser).
`aiperf metrics list` confirms no `nvidia_*` or `amd_*` id exists.

**Observable user impact:** any dashboard, `jq` filter, or regression gate keyed
on `nvidia_total_gpu_power` reads a missing key on a native run — the value is
now under `total_gpu_power`. Eight metrics have no new home at all:
`average_gpu_power`, `energy_delay_product`, `energy_per_output_token`,
`energy_per_request`, `energy_per_total_token`, `goodput_per_watt`,
`output_tps_per_watt`, `performance_per_watt`. Separately, Python's per-vendor
scoping means a mixed NVIDIA+AMD host reports two independent families; Rust has
one unscoped set, so on such a host the single `total_gpu_power` cannot mean the
same thing as either Python family. No error or warning is emitted in any case.

**Confidence:** high for the rename and the 8 missing tags (direct enumeration
of both sides). Medium for the mixed-vendor aggregation claim — I read the tag
set, not the Rust telemetry fold, so I did not confirm *how* Rust combines
multi-vendor samples.

### 2. Fixed-schedule replay send-lag metrics and the degraded warning have no Rust counterpart

**Severity:** P1
**Status:** NEW

**Python evidence:**
`src/aiperf/metrics/replay_sched_lag_analyzer.py:51` injects the family, anchored
at the run-global least-late request:

```python
anchored_ms = (offsets - offsets.min()) / NANOS_PER_MILLIS
for cls in _PERCENTILE_METRICS:
    value = float(np.percentile(anchored_ms, cls.percentile))
    lag[cls.tag] = value
    results[cls.tag] = MetricResult(tag=cls.tag, ...)
```

and at `replay_sched_lag_analyzer.py:93` derives the degraded flag plus a
run-level warning:

```python
degraded = lag[ReplaySchedLagP99Metric.tag] > REPLAY_SCHED_DEGRADED_THRESHOLD_MS
results[ReplaySchedDegradedMetric.tag] = MetricResult(...)
if degraded and warn_degraded is not None:
    warn_degraded(...)
```

It is wired unconditionally into the summarize path at
`src/aiperf/metrics/accumulator.py:682`. Registry dump: `replay_sched_lag_p50/p90/p99`
(ms), `replay_sched_degraded` (count), `replay_send_schedule_offset` (ns), all
flagged `FIXED_SCHEDULE_ONLY`.

**Rust evidence:** no counterpart exists. `rg -n 'replay_sched|ReplaySched|replay_send_schedule'`
over `rust/` matches only `rust/runtime/src/agentx/trajectory_source.rs:264`
(`replay_schedule`, a turn-timestamp scheduler, unrelated) and its callers in
`rust/runtime/src/agentx/replay.rs` and `rust/runtime/src/agentx/export.rs`. The
`MetricTag` enum (`rust/runtime/src/metrics_core/catalog.rs:20-400`) has no lag
tag, and `aiperf metrics list` emits none.

**Observable user impact:** on a fixed-schedule / trace-replay run, Python tells
the user how far behind its own schedule the generator fell (p50/p90/p99 in ms)
and warns when p99 crosses the degraded threshold. Rust reports neither, so a
run where AIPerf itself was the bottleneck is indistinguishable from a clean one:
the server's latency numbers look valid while the arrival process was actually
wrong. This is the failure mode the metric exists to catch.

**Confidence:** high — exhaustive symbol search on both sides.

### 3. Request latency ends at the last content chunk in Python and at transport stream close in Rust, and the difference propagates into every rate denominator

**Severity:** P1
**Status:** KNOWN (still true) — `docs/dev/python-rust-parity-gaps.md:765` P1.29

**Python evidence:** `src/aiperf/metrics/types/request_latency_metric.py:36-45`:

```python
        Note: Uses the last content response (with actual data), not usage-only chunks.
        """
        request_ts: int = record.start_perf_ns

        # Use content_responses to get last response with actual content
        if not record.content_responses:
            raise NoMetricValue(
                "Request latency requires at least 1 non-empty content response."
            )
        final_response_ts = record.content_responses[-1].perf_ns
```

`content_responses` filters out usage-only and `[DONE]` frames
(`src/aiperf/common/models/record_models.py:1589`: `[response for response in
self.responses if response.data]`). The run window is built from that same
latency: `src/aiperf/metrics/types/max_response_metric.py:48` computes
`final_response_ts = record.timestamp_ns + request_latency`, and
`src/aiperf/metrics/types/benchmark_duration_metric.py:46` returns
`max_res_time - min_req_time`.

**Rust evidence:** the transport stamps `end_ns` after the whole response body is
consumed — `rust/runtime/src/transport/http/transport/http_transport.rs:696`:

```rust
record.end_ns = Some(self.clock.now_ns());
```

which is carried verbatim into the record
(`rust/runtime/src/metrics.rs:488`: `let end_ns = self.response.end_ns.or(self.terminal_ns).unwrap_or(finish_ns);`)
and used for both latency and the run window
(`rust/runtime/src/metrics_core/store.rs:1580`):

```rust
self.set_metric_f64(row, MetricTag::MaxResponseTimestamp, record.end_ns as f64);
self.set_metric_f64(row, MetricTag::RequestLatency, record.latency_ns() as f64);
```

with `latency_ns()` = `end_ns - start_ns`
(`rust/runtime/src/metrics_core/ingest.rs:264`). There is no "last content chunk"
timestamp on the metrics path; the only such concept in Rust is the WebSocket
round-trip fact at `rust/runtime/src/transport/ws.rs:82`, which feeds
`TimeToLastRoundTrip`, not latency.

**Observable user impact:** on streaming runs Rust's `request_latency` includes
the trailing usage chunk and the `data: [DONE]` frame plus the final read/close;
Python excludes them. The extra term is typically sub-millisecond to a few ms per
request, so `request_latency` avg/percentiles shift up by that amount. Because
`benchmark_duration` is derived from the same boundary, every rate that divides
by it (`request_throughput`, `input_token_throughput`,
`output_token_throughput`, `total_token_throughput`, `goodput`,
`image_samples_per_second`) shifts *down* by the same relative amount. It also
feeds `inter_token_latency` (`(latency - ttft) / (osl - 1)`), and from there
`output_token_throughput_per_user`, `decode_duration`, and
`e2e_output_token_throughput`. Direction is deterministic (Rust latency ≥ Python
latency), so it is a systematic bias, not noise.

**Confidence:** high on mechanism and direction. The magnitude is
UNVERIFIED — quantifying it needs a paired run against a fixed-latency mock
server with `stream_options.include_usage` on and off.

### 4. `context_overflow_count` is computed in Rust but never reaches any report

**Severity:** P2
**Status:** NEW

**Python evidence:** `src/aiperf/metrics/types/context_overflow_count_metric.py:33`
registers the tag and it is emitted through the ordinary error-metric path:

```python
tag = "context_overflow_count"
header = "Context Overflow Count"
unit = GenericMetricUnit.REQUESTS
flags = MetricFlags.ERROR_ONLY | MetricFlags.NO_INDIVIDUAL_RECORDS
```

**Rust evidence:** the counter exists as a standalone struct —
`rust/runtime/src/agentx/metrics.rs:65`:

```rust
/// `ContextOverflowCountMetric`): counts records with `context_overflow == true`.
pub struct ContextOverflowCount {
```

but `rg -n 'ContextOverflowCount|context_overflow_count' rust/` outside
`rust/runtime/src/agentx/metrics.rs` returns nothing, and no `MetricTag`
variant exists for it (`rust/runtime/src/metrics_core/catalog.rs:20-400`). It is
never summarized, exported, or rendered.

**Observable user impact:** a run whose failures are context-length overflows
shows those requests only in the generic `error_request_count`; the dedicated
`context_overflow_count` key is absent from the native summary. Python's AgentX
scenario uses this count to flip `submission_valid` when the overflow rate
exceeds 1%, so a consumer implementing that rule against native output silently
sees zero overflows.

**Confidence:** high.

### 5. HTTP byte metrics changed their unit spelling from `B` to `bytes`

**Severity:** P2
**Status:** NEW

**Python evidence:** `src/aiperf/common/enums/metric_enums.py:84`:

```python
BYTES = MetricSizeUnitInfo(
    tag="B",
    long_name="bytes",
    num_bytes=1,
)
```

The registry dump shows `http_req_data_sent` and `http_req_data_received` with
`unit = "B"` and no `display_unit` override, so `B` is what serializes.

**Rust evidence:** `rust/runtime/src/metrics_core/units.rs:247`:

```rust
Self::Byte => "bytes",
```

`aiperf metrics list` renders both HTTP metrics with unit `bytes`. Headers are
identical (`HTTP Data Sent` / `HTTP Data Received`) and the values are the same
quantity.

**Observable user impact:** the `unit` field of these two metrics changes string
in the summary JSON, CSV, and console; any consumer matching on `"unit": "B"`
(or rendering the unit as a label) shows `bytes` instead. Numbers do not move.

**Confidence:** high.

### 6. `credit_to_start_latency` and `effective_latency` keep their tags but change their display headers

**Severity:** P2
**Status:** NEW

**Python evidence:** `src/aiperf/metrics/derived_latency.py:113` and
`derived_latency.py:146`:

```python
tag="credit_to_start_latency",
header="Credit-to-Start Latency",
unit="ms",
...
tag="effective_latency",
header="Effective Latency (CO-aware)",
unit="ms",
```

**Rust evidence:** `rust/runtime/src/metrics_core/catalog.rs:2050` and
`catalog.rs:2059`:

```rust
    spec!(
        EffectiveLatency,
        "Effective Latency",
        Millisecond,
        ...
    spec!(
        CreditToStartLatency,
        "Credit To Start",
        Millisecond,
```

**Observable user impact:** tags, units, and values match, but the human-readable
header — which is what the console table and CSV column label use, and which is
carried in the summary JSON `header` field — changes from
`Credit-to-Start Latency` to `Credit To Start` and from
`Effective Latency (CO-aware)` to `Effective Latency`. The second one drops the
"(CO-aware)" qualifier that tells the reader this latency includes credit-queue
wait, which is the whole point of the metric.

**Confidence:** high.

### 7. `credit_drop_latency` includes failed and cancelled requests in Rust and only successful ones in Python

**Severity:** P2
**Status:** KNOWN (still true, and now precisely located) —
`docs/dev/python-rust-parity-gaps.md:780` P1.30 states this vaguely
("Credit/effective latency can be populated outside Rust's normal valid-record
gate")

**Python evidence:** `credit_drop_latency` is an ordinary `BaseRecordMetric`
without `MetricFlags.ERROR_ONLY`
(`src/aiperf/metrics/types/credit_drop_latency_metric.py:24-31`), and the record
processor parses non-error metrics only for records that pass the validity gate
— `src/aiperf/post_processors/metric_record_processor.py:69`:

```python
parse_funcs = self.valid_parse_funcs if record.valid else self.error_parse_funcs
```

where `valid_parse_funcs` is built with `exclude_error_metrics=True`
(`metric_record_processor.py:43`), which sets
`disallowed_flags |= MetricFlags.ERROR_ONLY`
(`src/aiperf/post_processors/base_metrics_processor.py:111`). A cancelled request
is an error record in Python (`src/aiperf/transports/aiohttp_client.py:291` sets
`ErrorDetails(type="RequestCancellationError", ... code=499)`), so both failures
and cancellations are excluded.

**Rust evidence:** `rust/runtime/src/metrics_core/store.rs:1574` gates the normal
metrics on validity, but the credit block at `store.rs:1675` sits *after* the
`else` branch closes and therefore runs for every row:

```rust
let valid = !record.errored && !record.canceled;
if valid { /* ... */ } else { /* ErrorRequestCount ... */ }

if let Some(credit_ns) = record.admit_ns {
    let queue_ns = (record.start_ns - credit_ns).max(0);
    self.set_metric_f64(row, MetricTag::CreditDropLatency, queue_ns as f64);
    self.set_metric_f64(row, MetricTag::CreditToStartLatency, queue_ms);
    self.set_metric_f64(row, MetricTag::EffectiveLatency, effective_ms);
}
```

**Observable user impact:** on a run with failures or cancellations,
`credit_drop_latency`'s `count`, `avg`, and percentiles include the failed
attempts in Rust and exclude them in Python. Direction depends on the workload:
failures that were queued behind a saturated issuer raise the Rust numbers, and
fast-rejected requests lower them. The metric is `INTERNAL` on both sides, which
caps user impact.

Note the sibling case is now *consistent*, contrary to the backlog's framing:
Python's injected `credit_to_start_latency` / `effective_latency` are computed
over a phase/window mask that never filters errors
(`src/aiperf/metrics/accumulator.py:626` → `_mask_for_export_context` at
`accumulator.py:309-343`, which only ANDs `~isnan(start_ns)`, phase, and window
bounds), so both implementations include failed rows for those two tags.

**Confidence:** high.

### 8. The four `*_diff_pct` metrics drop the trailing `%` from their display header

**Severity:** P2
**Status:** NEW (found only after the baseline correction — the branch checkout
had already dropped the `%` on all four)

**Python evidence:** `src/aiperf/metrics/types/osl_mismatch_metrics.py:89-92`
and `src/aiperf/metrics/types/usage_diff_metrics.py:50-53,112-115,177-180`:

```python
    tag = "osl_mismatch_diff_pct"
    header = "OSL Mismatch Diff %"
    short_header = "OSL Diff"
    short_header_hide_unit = True
...
    tag = "usage_prompt_tokens_diff_pct"
    header = "Usage Prompt Diff %"
...
    tag = "usage_completion_tokens_diff_pct"
    header = "Usage Completion Diff %"
...
    tag = "usage_reasoning_tokens_diff_pct"
    header = "Usage Reasoning Diff %"
```

**Rust evidence:** `rust/runtime/src/metrics_core/catalog.rs:1647,1656,1665,1694`:

```rust
        "Usage Prompt Diff",
        "Usage Completion Diff",
        "Usage Reasoning Diff",
        "OSL Mismatch Diff",
```

`aiperf metrics list` confirms all four render with the `%` unit and the
shortened header.

**Observable user impact:** tags, units, and values are identical. The console
row label — built as `f"{record.header} ({record.unit})"` at
`src/aiperf/exporters/console_metrics_exporter.py:218` — changes from
`OSL Mismatch Diff % (%)` to `OSL Mismatch Diff (%)`, and the `header` field in
the summary JSON changes with it, on all four metrics. Any consumer keyed on the
header string (or a diffed golden console/CSV output) breaks. Rust's spelling is
arguably the better one; the change is simply undocumented.

**Confidence:** high.

## Checked and consistent

Verified equivalent by reading both implementations; listed so a future audit
does not re-derive them.

- **Percentile band.** Both report exactly p1/p5/p10/p25/p50/p75/p90/p95/p99:
  `src/aiperf/metrics/metric_dicts.py:49` (`_PERCENTILE_QS`) and
  `rust/runtime/src/metrics_core/kernel.rs:17` (`PERCENTILES`).
- **Percentile interpolation.** Both use manual linear interpolation on the
  sorted vector with the identical `virtual_idx = q/100*(n-1)`, `lo = floor`,
  `hi = min(lo+1, n-1)` construction: `metric_dicts.py:78-82` vs
  `kernel.rs:126-133`. Rust's low-cardinality fast path
  (`kernel.rs:189-245`) reproduces the same sorted-position lookup and replays
  repetitions individually to preserve addition order.
- **Standard deviation.** Population (ddof=0) on both sides:
  `metric_dicts.py:84` (`np.std(clean, ddof=ddof)`, `ddof: int = 0`) vs
  `kernel.rs:134-147` (`denom = count - ddof`). Both return 0.0 when the
  denominator would be non-positive.
- **Error-adjusted `adj_*` band.** Same trigger flag on the same four metrics
  (`request_latency`, `time_to_first_token`, `inter_token_latency`,
  `decode_duration`): Python at `request_latency_metric.py:25`,
  `ttft_metric.py:26`, `inter_token_latency_metric.py:42-45` (baseline line
  numbers; the flag pair is unchanged in content),
  `decode_duration_metric.py:23`; Rust at `catalog.rs:1158,1167,1194,1212`. Same
  construction (append `error_count` copies of `+inf` to the success-only
  sample), same nearest-rank method to avoid `inf - inf`, same `std = None`
  clamp, same `adj_` tag prefix: `derived_latency.py:229-249` vs
  `accumulator.rs:1205-1224` and `kernel.rs:259-322`. NumPy's
  `method="nearest"` rounds halves to even, matching Rust's
  `round_ties_even()` at `kernel.rs:285`.
- **Sweep-line (`effective_*`, `active_*`, `tokens_in_flight`).** Identical
  tags, headers, units, and scales
  (`src/aiperf/analysis/sweepline.py:53-105,162-236` vs
  `rust/runtime/src/metrics_core/sweepline/mod.rs:1028-1151`); identical
  duration-CDF percentile selection (first cumulative fraction ≥ q, no
  interpolation) at `src/aiperf/analysis/sweepline_stats.py:86-88` vs
  `sweepline/stats.rs:510-519`; identical duration-weighted mean and population
  variance over the full window span for `effective_*` and over active duration
  only for `active_*` (`sweepline_stats.py:75-78,172-175` vs
  `stats.rs:480-503`); both emit only avg/min/max/p50/p90/p95/p99/std for this
  family (`sweepline_stats.py:205-217` vs `sweepline/mod.rs:1183-1201`).
- **Rate denominators.** Both resolve one shared observation duration —
  explicit window bounds when present, else `benchmark_duration`, and both refuse
  a zero duration: `src/aiperf/metrics/metric_dicts.py:217-238` vs
  `rust/runtime/src/metrics_core/accumulator.rs:1484-1501`.
  `request_throughput`, `input_token_throughput`, `output_token_throughput`, and
  `total_token_throughput` all divide by it on both sides
  (`request_throughput_metric.py:37`, `input_token_throughput_metric.py:50`,
  `output_token_throughput_metrics.py:42`, `total_token_throughput.py:46` vs
  `accumulator.rs:1557-1562`).
- **`benchmark_duration` guard.** Both require `min < max`
  (`benchmark_duration_metric.py:41` raises; `accumulator.rs:1537` returns
  `None`). Python's raise is caught and logged, Rust's `None` omits the tag, so
  the observable result — the metric is absent — is the same.
- **Inter-token latency divisor and chunk inclusion** (re-derived against the
  baseline after the correction; this replaces an earlier, wrongly-reasoned
  dismissal that called the first-chunk divisor Rust-only). Baseline Python is
  `(request_latency - ttft) / decode_tokens` where `decode_tokens` is
  `OSL - first_content_chunk_tokens`, *not* a fixed `OSL - 1`:
  `src/aiperf/metrics/types/inter_token_latency_metric.py:63-102`. Rust is the
  same function, `rust/runtime/src/metrics_core/itl.rs:15-41`. Every branch
  matches:
  - **Guard.** Python raises `NoMetricValue` when `osl < 2`
    (`inter_token_latency_metric.py:64-65`); Rust returns `None` for
    `output_sequence_length < 2` (`itl.rs:19-21`). Both omit the metric.
  - **Divisor.** Absent first-chunk count → `OSL - 1` on both
    (`inter_token_latency_metric.py:78-81` vs `itl.rs:23`). Present and
    consistent → `OSL - count` on both (`:83` vs `itl.rs:24-29`). Python's
    fallback trigger is `first_chunk_tokens <= 0 or decode_tokens < 1`, i.e.
    accept iff `0 < count < OSL`, which is exactly Rust's guard
    `count > 0 && count < OSL`. Both then fall back to `OSL - 1` and log a
    warning **once per process** (Python's `_mismatch_warned` class flag at
    `:53,88-97`; Rust's `HAS_WARNED_FIRST_CHUNK_MISMATCH` at `itl.rs:8,31-37`).
  - **Which chunk supplies the count.** Both walk forward to the first
    *content-bearing* chunk that carries usage and read that chunk's cumulative
    `completion_tokens`: `src/aiperf/common/models/record_models.py:1398-1418`
    (`first_content_chunk_completion_tokens`, `if response.data and
    response.usage`) vs `rust/runtime/src/transport/reduce.rs:103-123`
    (`capture_first_content_chunk_usage`, which re-tries on the next content
    chunk because a content chunk without usage leaves the field `None`). Same
    unit choice too — raw `completion_tokens`, reasoning included, matching OSL.
  - **Opt-in gate.** Both populate the count only under `--per-chunk-usage`,
    default `false`: `src/aiperf/records/inference_result_parser.py:518-534`
    (`if self.run.cfg.endpoint.per_chunk_usage else None`) and
    `src/aiperf/config/endpoint.py:252-255` vs
    `rust/runtime/src/config/resolve.rs:243` /
    `rust/cli/src/yaml.rs:2474` (`unwrap_or(false)`). Both enforce the same
    three validations — requires `--use-server-token-count`, requires endpoint
    type `chat`, requires `--streaming`:
    `src/aiperf/config/endpoint.py:552-576` vs
    `rust/runtime/tests/per_chunk_usage_config.rs:22,36`. The Rust tree also
    carries a dedicated cross-implementation test,
    `rust/e2e-tests/tests/test_per_chunk_usage_parity.rs:383`.
  - **Excluded chunks.** The terminal usage-only chunk is excluded from both the
    numerator and the divisor selection on both sides, because both key off
    content presence (`record_models.py:1583` `content_responses` filters
    `if response.data`; `reduce.rs:110-113` requires
    `ResponseData::has_token_output`). `[DONE]` never becomes a parsed response
    on either side. **Caveat:** the *numerator* is still subject to finding 3 —
    Python's `request_latency` ends at the last content chunk, Rust's at
    transport stream close — so per-request ITL values do diverge, through that
    boundary and not through the divisor.
- **Per-record latency and token formulas.** `time_to_first_token` (first token
  minus start), `time_to_second_token` (second minus first),
  `inter_chunk_latency` (adjacent content-chunk deltas, suppressed on
  non-monotonic arrivals), `decode_duration` (`latency - ttft`),
  `output_token_throughput_per_user` (`1 / itl`),
  `e2e_output_token_throughput` (`osl / latency`),
  `prefill_throughput_per_user` (`isl / ttft`):
  `ttft_metric.py`, `ttst_metric.py:46-52`,
  `inter_chunk_latency_metric.py:56-73`,
  `decode_duration_metric.py:30-40`, `output_token_throughput_metrics.py:64-78`,
  `e2e_output_throughput_metric.py:38-52`, `prefill_throughput_per_user.py:37-53`
  vs `rust/runtime/src/metrics_core/store.rs:1583-1642` and
  `rust/runtime/src/metrics_core/accumulator.rs:885-932`.
- **`output_sequence_length` = output + reasoning, absent when both are absent.**
  `output_sequence_length_metric.py:43-50` vs
  `rust/runtime/src/metrics_core/ingest.rs:56-59`.
- **`output_token_count` omits zero.** Python's `not record.token_counts.output`
  treats 0 as missing (`output_token_count.py:43`); Rust guards with
  `is_some_and(|tokens| tokens > 0)` (`store.rs:1619`). Same absence semantics,
  so `total_output_tokens` and the OSL/OTC distributions agree.
- **Cancellation classification.** Python turns a cancellation into a 499 error
  record (`src/aiperf/transports/aiohttp_client.py:291`), so it lands in
  `error_request_count` and is excluded from latency/token distributions; Rust's
  `valid = !errored && !canceled` (`store.rs:1574`) with the `else` branch
  setting `ErrorRequestCount` (`store.rs:1667`) produces the same partition.
- **Error/goodput derivations.** `completed_request_count` (successes + errors,
  errors treated as 0 when absent), `request_error_rate`
  (`100 * errors / total`), `good_request_fraction` (`good / attempted`, 0.0 when
  attempted is 0), and the per-record SLO direction test (`>=` for
  `LARGER_IS_BETTER`, else `<=`, with a missing metric failing the SLO) all
  match: `completed_request_count_metric.py:40-43`,
  `request_error_rate_metric.py:38-44`, `good_request_fraction_metric.py:53-65`,
  `good_request_count_metric.py:76-94` vs `accumulator.rs:1514-1533` and
  `rust/runtime/src/metrics_core/definition.rs:101-107` /
  `accumulator.rs:1066-1081`.
- **List-metric percentiles are exact on both sides by default.** Python's
  `inter_chunk_latency` backend defaults to `ragged`, not the t-digest
  (`src/aiperf/common/environment.py:1018`, `default="ragged"`;
  `src/aiperf/metrics/column_store.py:45`), matching Rust's exact vector path.
  Python's t-digest aggregator (`list_metric_aggregation.py:132-162`) keeps
  count/sum/min/max/avg exact and Welford `std` exact with only percentiles
  approximate — the same contract Rust's `--sketch-metrics` advertises
  (`kernel.rs:44-79`, which keeps count/sum/avg/min/max exact and returns `None`
  for an empty sketch, preserving the exact path's absence semantics).
- **Empty distributions.** Both omit rather than emit zeros:
  `kernel.rs:104` returns `None` for an empty value vector and
  `kernel.rs:82-93` builds an all-`Absent` `DistributionStats`; Python's
  `metric_result_from_array` is only reached for non-empty arrays and the
  injected families return `None` when their arrays are empty
  (`derived_latency.py:111,145`).
- **Zero-valid-record runs.** Both omit the whole derived chain rather than
  reporting zeros: Python's `request_count` is not an `ERROR_ONLY` metric so it
  is absent, and `completed_request_count` / `request_error_rate` both
  `get_or_raise(RequestCountMetric)`; Rust's `derive_scalar` uses
  `get(MetricTag::RequestCount)?` for the same two tags
  (`accumulator.rs:1515,1518`). Same (arguably unfortunate) outcome: an
  all-failed run reports no error rate on either side.

## Unverified / needs runtime check

- **Magnitude of finding 3.** The mechanism is proven from code but the size of
  the shift is not. Needed: one streaming run against
  `aiperf-mock-server` with fixed TTFT/ITL and jitter at zero, executed by both
  implementations, comparing `request_latency` avg and `benchmark_duration`, once
  with `stream_options.include_usage` enabled and once without.
- **Mixed-vendor GPU energy aggregation (finding 1).** Whether Rust's single
  `total_gpu_power` sums across vendors or picks one is not determined by the tag
  set. Needed: read `rust/runtime/src/gpu_telemetry/accumulator.rs` against
  `src/aiperf/metrics/energy_efficiency_analyzer.py`, or run against a host with
  both vendors present.
- **`reasoning_token_count` zero-vs-absent.** Python emits 0 when
  `token_counts.reasoning is 0` and omits only on `None`
  (`reasoning_token_count.py:49`); Rust passes the `Option` straight through
  (`store.rs:1622`). The two agree *given the same `Option`*, but I did not trace
  whether the Rust reasoning parser produces `Some(0)` in exactly the cases the
  Python parser produces `0`. Needed: a reasoning-endpoint run with a response
  containing no reasoning tokens, comparing whether the tag is present.
- **Rounding before serialization.** I confirmed neither kernel rounds
  (`MetricValue::from_f64` and `metric_result_from_array` both store raw f64),
  but I did not compare the console/CSV formatters' significant-figure handling
  (`Definition::format_value` at
  `rust/runtime/src/metrics_core/definition.rs:110-115` uses `{value:.2}` for
  non-integer types). Needed: a side-by-side console and CSV export diff.
- **Derived-metric dependency ordering.** Both sides topologically order derived
  metrics (Python `MetricRegistry.create_dependency_order_for`, Rust
  `DERIVED_TOPO_ORDER` at `accumulator.rs:1190`) and both treat a missing
  dependency as "omit this metric". I verified the individual guards but not that
  the two orderings are identical for the full 109-tag shared set.
