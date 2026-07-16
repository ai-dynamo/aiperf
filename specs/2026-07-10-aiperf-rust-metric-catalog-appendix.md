<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Metric Catalog Appendix (the `MetricSpec` table)

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — `aiperf_runtime::metrics_core::catalog` (`rust/runtime/src/metrics*` / `metrics_core`).
The `CATALOG` holds 103 inherited Python metric identities plus 16 native sweep-result
identities, with exact metadata/dependencies and record/aggregate/derived implementations
for every row whose source data exists; validation + a deterministic metadata fingerprint
pin the graph; telemetry-owned injected rows stay catalogued-but-absent until their producer
supplies values.
**Grounding:** line-by-line read of `metrics/base_*.py`, `metrics/derived_sum_metric.py`,
all `metrics/types/*.py` (44 files), and `common/enums/metric_enums.py`.
**Companion / parent:** `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`
(the engine — columnar accumulator, kernels, sweep-line, `Reporter`). **This appendix
is the *data* over that engine**: the ~120 concrete metrics as declarative `MetricSpec`
rows + the earned-in-blood per-metric edge cases. It resolves the parent's "deferred:
the ~120-metric catalog appendix."

---

## 0. What this is

The parent spec fixed the *engine* and made the taxonomy call: a static `MetricSpec`
catalog + `AggregationKind` fold for RECORD/AGGREGATE, a typed `DeriveFn` table for
DERIVED, real `petgraph` toposort deps. This appendix is that catalog — every metric's
identity fields + compute logic — so the engine has data to run.

The catalog and its compute layer are built in `aiperf_runtime::metrics_core`. `CATALOG` holds all
103 inherited Python metric rows with source-faithful headers, short headers, units/display
units, flags, console groups, display order, value types, aggregation kinds, and dependency
edges, followed by 16 native sweep-result rows. Startup/test validation rejects duplicate
tags, missing dependencies, illegal type-tier edges, and cycles; a deterministic metadata
fingerprint pins the full catalog identity. The record, aggregate, and derived
implementations preserve every §4 scar.

**Identity is fully declarative.** Every metric is exactly:

```rust
MetricSpec {
    tag, header, short_header: Option<&str>, short_header_hide_unit: bool,
    unit, display_unit: Option<Unit>, display_order: Option<u32>,
    flags: MetricFlags, console_group: MetricConsoleGroup,
    required: &[MetricTag], value_type: MetricValueType,
    kind: MetricType, aggregation: Option<AggregationKind>,
}
```
plus **one compute closure** keyed by `kind`. There is **no** `long_header` and **no**
`missing_value` field (verified — they don't exist); "missing" is signalled by the
compute returning **`NoMetricValue`** (Rust: `Result<MetricValue, NoValue>` / `Option`).

---

## 1. Base-class → compute-shape mapping (the machinery)

| Python base | Rust shape | contract |
|---|---|---|
| `BaseRecordMetric[V]` | RECORD | `fn(record, &RecordDict) -> V` — one value per record; validity-gated (§4.1). |
| `BaseAggregateMetric[V]` | AGGREGATE | per-record extractor `fn(record,&RecordDict)->V` **+ fold `AggregationKind` (SUM/MAX/MIN) + default_value**. The columnar engine folds the column directly (never replays the per-record accumulate). |
| `BaseAggregateCounterMetric` | AGGREGATE[SUM] | extractor is a constant `1`; the **flag gate** (`ERROR_ONLY` etc.) picks which records feed it. |
| `BaseDerivedMetric[V]` | DERIVED | `fn(&ScalarDict) -> V` — reads other metrics' final scalars, no per-record state. |
| `DerivedSumMetric[V, Src]` | DERIVED | macro: `required = {Src.tag}`, `unit = Src.unit`, `flags = Src.flags` **iff self left flags NONE**, value = `Src`'s column sum. |

`default_value` for AGGREGATE folds: **MIN uses `i64::MAX`** (Python `sys.maxsize`), **MAX
uses 0**, SUM uses the type's zero. Value types: `Float`(0.0) `Int`(0) `FloatList`([])
`IntList`([]); a metric emitting a list *per record* (e.g. `inter_chunk_latency`) is
`IntList` and the column is list-of-lists, pooled for stats.

---

## 2. The core catalog (counts, latency, tokens, throughput, goodput)

Format: **tag** · header/short · unit→display · Kind[agg] · order · flags · group(if≠DEFAULT) ·
required · formula. `NoMV` = `NoMetricValue`. All latency ns→ms.

**Counts & rate**
- **request_count** · Requests · `requests` · AGG[SUM]·counter · 1100 · `LARGER_IS_BETTER|NO_INDIVIDUAL_RECORDS` · — · counts valid records (→1). No `ERROR_ONLY` so valid-only.
- **error_request_count** · Error Count · `requests` · AGG[SUM]·counter · — · `ERROR_ONLY|NO_INDIVIDUAL_RECORDS` · — · `ERROR_ONLY` **inverts** the validity gate → counts only invalid records. **Absent on zero-error runs** (the zero-error trap, §4.2).
- **completed_request_count** · Completed · `requests` · DERIVED · 1075 · `NO_INDIVIDUAL_RECORDS` · {request_count, error_request_count} · `request_count + (error_request_count or 0)`.
- **request_error_rate** · Err % · `%` · DERIVED · 1080 · `NO_INDIVIDUAL_RECORDS` · {request_count, error_request_count} · `total=succ+(err or 0)`; `total<=0`→NoMV; `100·err/total`.
- **good_request_count** · — · `requests` · AGG[SUM]·counter · — · `GOODPUT` · group NONE · (deps set by `set_slos`) · per-record `1` iff every SLO passes. `set_slos` converts each threshold from the target metric's *display_unit* to its *unit*, and picks direction per the target's `LARGER_IS_BETTER` flag (`>=` else `<=`). Missing target value → not good.
- **goodput** · Goodput · `requests/sec` · DERIVED · 1000 · `GOODPUT` · {good_request_count, benchmark_duration} · `good_request_count / obs_duration_s`; good_count absent→NoMV.
- **good_request_fraction** · GoodReqFrac · `ratio` · DERIVED · — · `GOODPUT|LARGER_IS_BETTER` · group NONE · **{good_request_count, request_count}** (deliberately NOT error_request_count) · `attempted = valid + (err or 0)`; `attempted==0`→0.0; `good/attempted`.

**Timestamps (internal, feed benchmark duration)** — use **wall-clock** `record.timestamp_ns`, not perf:
- **min_request_timestamp** · Min Req · `ns` · AGG[**MIN**] (default `i64::MAX`) · `NO_INDIVIDUAL_RECORDS|INTERNAL` · group NONE · — · `record.timestamp_ns`.
- **max_response_timestamp** · Max Resp · `ns` · AGG[**MAX**] (default 0) · `NO_INDIVIDUAL_RECORDS|INTERNAL` · group NONE · {request_latency} · `record.timestamp_ns + request_latency` (reconstruct wall-clock end).
- **benchmark_duration** · Duration · `ns`→`sec` · DERIVED · — · group NONE · {min_request_timestamp, max_response_timestamp} · `min>=max`→**ValueError**; else `max−min`.

**Latencies (RECORD, `PERCENTILE_INCLUDES_FAILED_REQUESTS` where noted → adj_*)** — use **perf-clock** `start_perf_ns` / `content_responses[i].perf_ns`:
- **request_latency** · Req Latency · `ns`→`ms` · RECORD · 300 · `PERCENTILE_INCLUDES_FAILED_REQUESTS` · — · needs ≥1 *content* response; `last_content.perf_ns − start_perf_ns`; `<0`→ValueError.
- **time_to_first_token** (TTFT) · TTFT · `ns`→`ms` · RECORD · 100 · `STREAMING_TOKENS_ONLY|PERCENTILE_INCLUDES_FAILED_REQUESTS` · — · `content[0].perf_ns − start_perf_ns`.
- **time_to_second_token** (TTST) · TTST · `ns`→`ms` · RECORD · 200 · `STREAMING_TOKENS_ONLY` · — · need ≥2 content; `content[1]−content[0]`.
- **time_to_first_output_token** (TTFO) · TTFO · `ns`→`ms` · RECORD · 210 · `STREAMING_TOKENS_ONLY|SUPPORTS_REASONING` · — · first content whose data is non-empty Text **or Reasoning.content (not the reasoning field)** or ToolCall text; `−start_perf_ns`. ==TTFT for non-reasoning.
- **inter_token_latency** (ITL) · ITL · `ns`→`ms` (float) · RECORD · 400 · `STREAMING_TOKENS_ONLY|PERCENTILE_INCLUDES_FAILED_REQUESTS` · {request_latency, time_to_first_token, output_sequence_length} · **`osl<2`→NoMV**; `(request_latency − ttft)/(osl−1)`. The `osl−1` = gap count. **The only `−1` in the family.**
- **inter_chunk_latency** (ICL) · ICL · `ns`→`ms` (**IntList**) · RECORD · — · `STREAMING_TOKENS_ONLY` · group NONE · — · consecutive `content[i]−content[i−1]` gaps; `<0`→ValueError; pooled for stats + drives the ICL sweeps.
- **credit_drop_latency** · Credit Latency · `ns`→`ms` · RECORD · — · `INTERNAL` · group NONE · — · `record.request.credit_drop_latency` or NoMV.

**Sequence lengths & token counts** (RECORD + `DerivedSum` totals):
- **output_sequence_length** (OSL) · OSL · `tokens` · RECORD · 600 · `PRODUCES_TOKENS_ONLY|LARGER_IS_BETTER` · `(output or 0)+(reasoning or 0)` (includes reasoning). → **total_output_sequence_length** DerivedSum.
- **input_sequence_length** (ISL) · ISL · `tokens` · RECORD · 700 · `TOKENIZES_INPUT_ONLY|LARGER_IS_BETTER` · `token_counts.input`. → **total_input_sequence_length**; also **error_isl** (adds `ERROR_ONLY`) → **total_error_isl**.
- **output_token_count** · Output Tokens · `tokens` · RECORD · `PRODUCES_TOKENS_ONLY|LARGER_IS_BETTER` · group NONE · `token_counts.output` (**excludes** reasoning; differs from OSL). → **total_output_tokens**.
- **reasoning_token_count** · Reasoning Tokens · `tokens` · RECORD · `…|SUPPORTS_REASONING` · group NONE · `token_counts.reasoning`. → **total_reasoning_tokens**.

**Throughput** — **every one is `count / observation_duration_s`; there is NO `−1`.** `observation_duration` = window span (both bounds set) else `benchmark_duration`, ns→s; zero→NoMV.
- **request_throughput** · Req/sec · `requests/sec` · DERIVED · 900 · `LARGER_IS_BETTER` · {request_count, benchmark_duration}.
- **input_token_throughput** · Input TPS · `tokens/sec` · DERIVED · 805 · `LARGER_IS_BETTER` · group DEFAULT (deliberate — shows in realtime) · {total_isl, benchmark_duration}.
- **output_token_throughput** · Output TPS · `tokens/sec` · DERIVED · 800 · `PRODUCES_TOKENS_ONLY|LARGER_IS_BETTER` · {total_osl, benchmark_duration}.
- **total_token_throughput** · Total TPS · `tokens/sec` · DERIVED · — · group NONE · {total_isl, total_osl, benchmark_duration} · `(total_isl+total_osl)/dur`.
- **output_token_throughput_per_user** · Output TPS/User · `tokens/sec/user` · **RECORD** (float) · 500 · `STREAMING_TOKENS_ONLY|LARGER_IS_BETTER` · {inter_token_latency} · `1 / itl_seconds` (itl==0→NoMV).
- **e2e_output_token_throughput** · E2E Output TPS/User · `tokens/sec/user` · **RECORD** (float) · 510 · `PRODUCES_TOKENS_ONLY|LARGER_IS_BETTER` · {output_sequence_length, request_latency} · `osl / request_latency_s` (includes TTFT+queue, unlike `1/itl`).
- **prefill_throughput_per_user** · Prefill TPS/User · `tokens/sec/user` · RECORD (float) · — · `STREAMING_TOKENS_ONLY|TOKENIZES_INPUT_ONLY|LARGER_IS_BETTER` · group NONE · {input_sequence_length, time_to_first_token} · `isl / ttft_s`.

**ASR**
- **rtfx** · RTFx · `ratio` · RECORD (float) · 850 · `SUPPORTS_AUDIO_ONLY|LARGER_IS_BETTER` · {audio_duration, request_latency} · `audio_duration_s / request_latency_s` (latency≤0→NoMV).

---

## 3. The extended families (usage / cache / diff / mismatch / efficiency / http-trace / media)

Full identity tables live in the reader briefs; the load-bearing shape per family:

- **Usage token metrics** (`usage_metrics.py`, `usage_cache_metrics.py`, `usage_extras_metrics.py`):
  all `BaseUsageRecordMetric[int]`, `unit=tokens`, group **USAGE**. A subclass declares
  `usage_field` (a property on the merged `final_usage`) + `missing_message`; the reader
  raises `NoMV` on `None`, returns present `0` as real zero (**the absent-vs-0
  distinction — the cache_reporting_hint contract**, §4.3). Tags: `usage_prompt_tokens`,
  `usage_completion_tokens`, `usage_total_tokens`, `usage_reasoning_tokens`,
  `usage_prompt/completion_audio_tokens`, `usage_accepted/rejected_prediction_tokens`
  (note `rejected` omits `LARGER_IS_BETTER`), `usage_prompt_cache_read/write/miss_tokens`
  (`write` omits `LARGER_IS_BETTER` — writes cost more), `usage_tool_use_prompt_tokens`,
  `usage_prompt_audio_seconds` (unit=`sec`, not tokens).
- **Usage totals** (`usage_total_metrics.py`): `DerivedSumMetric` over each usage metric →
  `total_usage_*`. Plus **overall_usage_prompt_cache_read_pct** (DERIVED %, **token-volume-weighted**
  `total_cache_read/total_prompt·100`, not an average of per-request percents).
- **Usage diff** (`usage_diff_metrics.py`): three `%` RECORD metrics
  `usage_{prompt,completion,reasoning}_tokens_diff_pct` = `abs((server−client)/client)·100`,
  flag `USAGE_DIFF_ONLY`, group NONE, client source = ISL/OSL/reasoning_count. Plus
  **usage_discrepancy_count** (AGG-counter): `1` if any diff > `USAGE_PCT_DIFF_THRESHOLD`
  (default 10%); reasoning checked opportunistically (`.get`, not required).
- **OSL mismatch** (`osl_mismatch_metrics.py`): **requested_osl** (RECORD, `INTERNAL`,
  reads hoisted `max_tokens`), **osl_mismatch_diff_pct** (RECORD %, **signed**
  `(actual−requested)/requested·100`), **osl_mismatch_count** (AGG-counter): increment iff
  `abs(actual−requested) > min(requested·pct_threshold, max_token_threshold=50)` — **the
  `min()` cap** tightens the bound for large OSL.
- **Thinking efficiency** (`thinking_efficiency_metrics.py`): **thinking_efficiency**
  (RECORD ratio, per-record `reasoning/output`, `EXPERIMENTAL`) vs
  **overall_thinking_efficiency** (DERIVED, **token-volume-weighted** `total_reasoning/total_output`).
- **Power efficiency** (`power_efficiency_metrics.py`) + **network_adjusted** + **total_gpu_***:
  **the injected-metric pattern** — `_derive_value` deliberately raises `NoMV`; the value is
  written externally (by the GPU accumulator / the network-RTT shift) and the derivation walk
  *catches and skips*. `total_gpu_power` (W), `total_gpu_energy` (J), `output_tokens_per_joule`,
  `energy_per_user` (÷concurrency, omitted when unset/0); `network_adjusted_{request_latency,ttft,ttfo}`
  + `network_rtt`.
- **Stream latency** (`stream_latency_metrics.py`): `stream_setup_latency` (recv_start−start),
  `stream_prefill_latency` (ttft−setup) — both `EXPERIMENTAL`.
- **Accuracy** (`accuracy_metrics.py`): `accuracy.correct` / `accuracy.unparsed` (dotted tags),
  `INTERNAL` running sums re-read from the record dict (the real display is the accuracy
  accumulator — see the accuracy spec).
- **Media**: `audio_duration` (sec), `num_images`/`image_throughput`/`image_latency`,
  `video_inference_time`/`video_peak_memory` (SGLang-reported).
- **HTTP-trace** (`http_trace_metrics.py`, 14 metrics): all `HTTP_TRACE_ONLY`, group NONE,
  ns→ms; the k6/HAR duration table (which two `perf_ns` subtract):

  | tag | end − start |
  |---|---|
  | `http_req_blocked` | pool_wait_end − pool_wait_start (start None→0) |
  | `http_req_dns_lookup` | dns_end − dns_start (cache-hit/reused/start-None→0) |
  | `http_req_connecting` (TCP+TLS) | tcp_end − tcp_start (reused→0) |
  | `http_req_sending` | send_end − send_start |
  | `http_req_waiting` (TTFB) | recv_start − send_end |
  | `http_req_receiving` | recv_end − recv_start (1 chunk→0) |
  | `http_req_duration` | recv_end − send_start |

  plus `http_req_connection_reused` (0/1), `http_req_data_sent/received` (bytes→KB),
  `http_req_chunks_sent/received` (count), and two composites summing siblings via
  `record_metrics.get(tag,0)`: `http_req_connection_overhead` (blocked+dns+connecting),
  `http_req_total` (all six phases). **Two-tier errors:** `NoMV` = soft-suppress
  (incomplete/wrong-type trace); `ValueError` = hard-fail (end-before-start).

---

## 4. The scars (per-metric earned-in-blood — port behavior-exact)

1. **`ERROR_ONLY` inverts the validity gate.** `_require_valid_record` normally raises
   `NoMV` on an invalid record; a metric with `ERROR_ONLY` computes *only* on invalid
   records (error_request_count, error_isl). Rust: the gate is `if record.valid ==
   flags.contains(ERROR_ONLY) { skip }`.
2. **The zero-error trap.** `error_request_count` emits *no value* on clean runs.
   `completed_request_count` / `request_error_rate` list it as required only for **ordering**
   and read it as `.get(tag) or 0`; `good_request_fraction` must **not** require it. A Rust
   port that treats "required ⇒ must exist" will kill these metrics on every clean run.
3. **absent-vs-0 (cache_reporting_hint).** A usage field that is `None` → metric omitted
   (`NoMV`); present `0` → real zero. `Option<i64>`: `None` skips, `Some(0)` emits. This is
   how "cache off" (absent) differs from "cache on, 0 hits" (zero).
4. **ITL `osl<2` guard + `osl−1`.** No inter-token interval exists below 2 output tokens;
   the divisor is the gap count `osl−1`, not `osl`.
5. **TTFO = first *non-reasoning* token.** Skip reasoning-only chunks; the first chunk with
   real Text/Reasoning-*content*/ToolCall text. ==TTFT when the model doesn't reason.
6. **osl_mismatch `min()` cap** (`min(requested·pct, 50)`) and the **signed** diff-pct.
7. **Volume-weighted overalls.** `overall_thinking_efficiency`, `overall_usage_prompt_cache_read_pct`
   weight by token volume (`Σnum/Σden`), *not* a mean of per-request ratios.
8. **network_adjusted exclusions.** Only request-start-anchored metrics get the constant-RTT
   shift (clamp-0, stddev unchanged); **ITL/ICL/TTST are excluded** — RTT cancels in
   `latency − ttft`.
9. **The injected-metric pattern.** `power_efficiency`, `network_adjusted`, `total_gpu_*`
   `_derive_value` → raise `NoMV`; value written by an external accumulator; the DERIVED walk
   catches-and-skips. Rust: these `DeriveFn`s return `NoValue`, and the injector writes the
   scalar before the report is built.
10. **Wall-clock vs perf-clock.** min/max timestamps + benchmark_duration use epoch
    `record.timestamp_ns`; every latency uses monotonic `*_perf_ns`. Do not mix.

**Two assumptions corrected by the read:** there is **no `−1` in any throughput** (throughput
is `count/duration`; the only `−1` is ITL's `osl−1`); and there is **no `long_header`/
`missing_value`** attribute (missing = `NoMV`).

---

## 5. Rust encoding

- A `const` array of `MetricSpec` (grouped by family) + a `DeriveFn`/`AggregateFn`/`RecordFn`
  table keyed by tag. A `#[test]` asserts: tags unique + non-empty; `required` all resolve;
  the type-tier rule holds (RECORD→RECORD, AGGREGATE→RECORD·AGGREGATE, DERIVED→any); the dep
  graph is acyclic (the port's equivalent of Python's import-time fail-fast).
- **`set_slos` for goodput** is a runtime step: at config time, resolve each SLO target's
  spec, convert the threshold display_unit→unit, set `good_request_count.required` +
  per-target direction from `LARGER_IS_BETTER`.
- **DerivedSum** is a small generator: given `Src`, emit a DERIVED spec inheriting
  `unit`/`required={Src}`/`flags`(if self NONE) that sums `Src`'s column.
- Percentile layout `[1,5,10,25,50,75,90,95,99]` + linear interp + population std (ddof=0)
  for inference — as the parent engine spec fixes.

## 6. Scope

- **In:** every metric above ships as a `MetricSpec` row + its compute closure + the §4 scars.
- **Producer-gated (catalogued, absent until fed):** the telemetry-injected metrics
  (`total_gpu_*`, power-efficiency) receive their values from the **telemetry spec**;
  `accuracy.*` display comes from the **accuracy spec**; `network_adjusted_*` needs the
  network-RTT calibration (telemetry spec). Their `MetricSpec` rows + injected-`NoValue`
  `DeriveFn`s live here and are always catalogued; the *injectors* live there. These rows
  intentionally remain absent unless a producer supplies an override — the deferred-producer
  boundary, not an unimplemented catalog row.
- **Testing:** a Python twin emits `{record → per-metric value}` goldens over a fixture corpus
  exercising each edge case (ITL osl<2, TTFO reasoning-skip, zero-error absence, absent-vs-0,
  osl min-cap, adj_* +inf, http-trace zero-cases); Rust asserts equality.
