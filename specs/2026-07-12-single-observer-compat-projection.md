# Single measurement observer on the request hot path — derive the legacy `TraceSimulationReport` by projection

**Date:** 2026-07-12
**Status:** Proposed (verified against code; compile-checked projection skeleton attached)
**Scope:** `rust/runtime` (scheduled/run/phase_runtime/metrics), `rust/loadgen-core` (collector), `rust/runtime/src/metrics_core` (accumulator/report), `rust/loadgen-core` (observer). Repo is READ-ONLY for this spec; no repo files were modified.
**Verification artifact:** `~/tmp/a2-spec/` — a throwaway Cargo crate with path deps on `loadgen-core` + `aiperf` (for the `aiperf_runtime::metrics_core` module) that compiles `fn project_trace_report(summary: &AccumulatorSummary, records: &[RecordIngest], wall_ms) -> TraceSimulationReport`. `cargo build` is green (see "Verification evidence").

---

## 1. Motivation — the duplicate per-request store

On the paths where the dispatcher forwards the runtime's observer, **every request is measured twice**:

- `rust/runtime/src/scheduled.rs:482-486` builds
  `ObserverTee::new(vec![CollectorObserver, NativeMetricsObserver])` and stores it as the runtime `observer`.
- `rust/runtime/src/metrics.rs:688-736` (`ObserverTee`) fans **every** callback (`on_arrival`/`on_admit`/`on_token`/`on_classified_token`/`on_output_tokens`/`on_usage`/`on_terminal`) to **both** delegates in order.
- Delegate 1 — `CollectorObserver` (`rust/loadgen-core/src/observer.rs:19-68`) → `TraceCollector` (`rust/loadgen-core/src/collector.rs:466-746`): a `FxHashMap<Uuid, TraceRequestStats>`; `on_token` does one map lookup + one `Vec::push` per token (`collector.rs:740-746`).
- Delegate 2 — `NativeMetricsObserver` (`rust/runtime/src/metrics.rs:190-673`): a second `FxHashMap<Uuid, usize>` slot lookup + `Vec<i64>` push per token (`metrics.rs:599-635`).

So each token pays **two hashmap lookups + two vec pushes**, and each request pays **two per-request allocations** that are reduced by two independent finalizers at `finish_at` (`scheduled.rs:1115-1127`: `collector.finish()` **and** `native_metrics.finish()`, optionally in `rayon::join`).

### Where this actually costs (verified, and it is NOT uniform)

| Path | Dispatcher feeds runtime observer? | Per-token double cost? |
|---|---|---|
| **Library online** (`run.rs` `run_scheduled_online`/`run_request_rate_online_*`/`run_paced_*`, and `accuracy.rs` `run_single_turn_dataset_online`) | Yes — TransportSink-backed dispatcher forwards `observer` | **Yes** |
| **Offline Dynamo** (`dynosim.rs:2878-2885` `run_trace_offline`, `dynosim.rs:3265-3266` graph sink) | Yes — engine/graph sink forwards `observer` | **Yes** |
| **Runner product online** (`aiperf-cli` `engine/execute.rs:3262-3311` `ConfiguredDispatcher`) | **No** | **No** — see below |

The runner's product dispatcher deliberately **ignores** the `ScheduledRuntime` observer and feeds its own single `RunCapture` observer instead:

> `execute.rs:3271` `_observer: &dyn RequestObserver` (unused) …
> `execute.rs:3276-3286`: "the ScheduledRuntime's runtime observer (CollectorObserver + NativeMetricsObserver) is computed and then discarded … Feed only the capture observer so the single dispatch thread does not replay every token into a discarded accumulator."

`RunCapture.observer` is a **single** `Rc<NativeMetricsObserver>` (`execute.rs:3067-3080`). Because tokens never reach the runtime `ObserverTee`, the runner's `TraceCollector` sees only `on_arrival`/`on_terminal` (`scheduled.rs:896,955,983`) and produces a **degenerate, discarded** `report.performance` (no tokens ⇒ `accumulate_requests` skips every request at `collector.rs:956-961`). The runner consumes only `report.turns` from the recorder.

**Consequence:** in the runner the second observer graph is idle dead-weight (allocated + finalized + thrown away), and in the library/offline paths it is a live per-token duplicate. Both are removable.

---

## 2. Target

**One measurement observer on the hot path — `NativeMetricsObserver` — and derive `TraceSimulationReport` by projection when a caller still needs the legacy compat shape.**

Concretely:
1. Stop adding `CollectorObserver` to the runtime delegate list on the online/offline paths (reuse the **existing** `PhaseScheduledPlan::collect_performance_summary` seam — `phase_runtime.rs:287,775-786` — which already drops the collector and collapses the `ObserverTee` to a single delegate when only native metrics remain).
2. Add one shared `project_trace_report(&AccumulatorSummary, &[RecordIngest], wall_ms) -> TraceSimulationReport` in `rust/runtime` (consumed by both online and offline), replacing the live `CollectorObserver` finalization with a projection off the native accumulator that the run already computes.
3. Keep the offline Dynamo `compatibility_report_from_dynamo` projection (`dynosim.rs:986-1050`) for the SimClock byte-parity gate — it is a **different** projection source (see §4).

Net: the online/offline hot path loses `CollectorObserver`/`TraceCollector`/`ObserverTee`; the compat report becomes a pure function of the native accumulator plus its retained records.

---

## 3. The legacy `TraceSimulationReport` — shape + downstream (verified)

Defined at `rust/loadgen-core/src/collector.rs:10-27`; **custom** `Serialize` at `collector.rs:191-283` (this is the byte surface the offline parity gate compares). Fields, in serialized order:

- **Counts** (`TraceRequestCounts`, `collector.rs:29-35`): `num_requests`, `completed_requests`, `total_input_tokens`, `total_output_tokens`.
- **Throughput** (`TraceThroughputStats`, `collector.rs:37-68`): `duration_ms`, `wall_time_ms`, `request_throughput_rps`, `input_throughput_tok_s`, `output_throughput_tok_s`, `total_throughput_tok_s`, `prefill_worker_seconds`, `decode_worker_seconds`, `prefill_gpus_per_worker`, `decode_gpus_per_worker`, `gpu_hours`.
- **Derived scalars** (methods, `collector.rs:119-135`): `processed_tokens`, `processed_tokens_per_s`, `processed_output_tokens_per_s`.
- **Cache ratios**: `prefix_cache_reused_ratio`, `first_admission_prefix_cache_reused_ratio`.
- **Latency distributions** (`TraceLatencyStats`, `collector.rs:97-105`), each a 9-field `TraceDistributionStats` (`collector.rs:84-95`: `mean/min/max/median/p75/p90/p95/p99/std`): `ttft`, `ttst`, `tpot`, `itl` (+ scalar `max_itl_ms`, `collector.rs:107-111,274`), `e2e_latency`, `output_token_throughput_per_user`.
- **Goodput** (`TraceGoodputStats`, conditional — only when an SLA was set; `collector.rs:70-82,248-258`): `goodput_completed_requests`, `goodput_request_throughput_rps`, `goodput_output_throughput_tok_s`.

That is the "74 base fields (+3 goodput)" the CLAUDE.md parity gate references.

**Downstream consumers (verified):**
- **Runner product path:** discards it (`execute.rs:3276-3282`; the runner emits the native-v2 report from `RunCapture`).
- **Library:** `run.rs` returns it inside `OnlineRunReport.performance` (`run.rs:63`); asserted by library tests (e.g. `run.rs:1693` `report.performance.request_counts.num_requests > 0`) and real-mock coverage.
- **Offline Dynamo:** `offline_execution.rs` reads `report.performance.throughput` for Dynamo capacity facts (`offline_execution.rs:788-806`) and runs the byte-parity gate on it (`offline_execution.rs:1871-1954`, `verify_parity`/`verify_parity_online`).

---

## 4. The offline projection is NOT a drop-in precedent for online

The task framing ("offline already reconstructs the compat report by projection") is **true but off-source**. The offline single-phase projection (`dynosim.rs:986-1050`, gated by `finish_shared_metrics_enforcing`'s `independently_accumulated` check at `dynosim.rs:924-932`) projects from **`DynamoSimulationReport`** — the Dynamo engine's *own* independently-computed replay report — **not** from AIPerf's `AccumulatorSummary`. It is byte-equal because the SimClock conformance gate (`dynosim.rs:957-968`) proves *AIPerf-collector == Dynamo-report*; when the AIPerf collector is skipped (`num_requests == 0`), Dynamo's already-proven-equal numbers are substituted verbatim.

**Online has no Dynamo.** The only second measurement online is `AccumulatorSummary`, which is computed by a **different kernel** than the collector. So online must project from `AccumulatorSummary` (+records), and that projection is **not** byte-identical to today's collector output. This is the crux of the risk analysis (§6).

---

## 5. Field-by-field mapping: native `AccumulatorSummary` → compat `TraceSimulationReport`

Native units for latency metrics are nanoseconds with `display_unit = Millisecond`, so `MetricResult::distribution()` already returns **ms** (proven by `metrics.rs:855-863`, TTFOT avg `19.0` for a token at 20 ms minus admit at 1 ms). Percentile band `[1,5,10,25,50,75,90,95,99]` (`kernel.rs:18`) contains every percentile the compat report needs.

| Compat field | Native source (`MetricTag` / API) | Status |
|---|---|---|
| `completed_requests` | `finite_value(RequestCount)` (Aggregate Sum of successful records, `catalog.rs:431-441`) | OK |
| `num_requests` | `RequestCount + ErrorRequestCount` | **GAP-1**: collector counts *all* arrivals incl. **canceled**; native has no single "all arrivals" scalar (canceled records return early, `metrics.rs:591-597` never reach a count). Approximate, or add a native arrivals scalar. |
| `total_input_tokens` | `finite_value(TotalInputSequenceLength)` (`catalog.rs`) | OK |
| `total_output_tokens` | `finite_value(TotalOutputSequenceLength)` | **NOT OK — real delta.** Native `TotalOSL` sums `usage.completion_tokens.or(output count)` (store.rs:1029–1033; test accumulator.rs:1552–1560 shows OSL following `completion_tokens=20`), whereas the collector sums `actual_output_length() = token_times_ms.len()` (collector.rs:501–503,965–966) — the **streamed-token count**. When authoritative server usage ≠ streamed count (the norm under CLAUDE.md's authoritative-usage design) these differ by whole tokens, not sub-ULP. **Records-first: recompute from the streamed `token_arrival_ns` count** to preserve collector parity, OR accept the native usage-based total as the new semantics (and update the golden + `run.rs`/`scheduled_real_mock.rs` pins). |
| `duration_ms` | `finite_value(BenchmarkDuration)` / 1e6 (ns→ms) | OK |
| `wall_time_ms` | caller-supplied (same `end_ns - start_ns` as today, `scheduled.rs:1109`) | OK |
| `request_throughput_rps` | `finite_value(RequestThroughput)` | OK |
| `input_throughput_tok_s` | `finite_value(InputTokenThroughput)` | OK |
| `output_throughput_tok_s` | `finite_value(OutputTokenThroughput)` | OK |
| `total_throughput_tok_s` | `finite_value(TotalTokenThroughput)` | OK |
| `prefill/decode_worker_seconds`, `*_gpus_per_worker`, `gpu_hours` | 0 online; imported from `DynamoSimulationReport` in the offline adapter (already done at `dynosim.rs:938-942`) | OK (engine-owned) |
| `processed_tokens*` | methods on the compat struct (input+output), unchanged | OK |
| `prefix_cache_reused_ratio`, `first_admission_prefix_cache_reused_ratio` | 0.0 online (online `on_admit` always passes `reused_input_tokens = 0`; native drops it anyway, `metrics.rs:591`); Dynamo-owned offline | OK |
| `ttft` distribution | `result(TimeToFirstToken).distribution()` | **RISK** (percentile algo + time base, §6) |
| `ttst` distribution | `result(TimeToSecondToken).distribution()` | RISK (percentile algo) |
| `tpot` distribution | `result(InterTokenLatency).distribution()` — aiperf ITL = `(latency−ttft)/(osl−1)`, `latency=end_ns−start_ns`, `osl=usage.completion_tokens.or(output count)` (accumulator.rs:670–682,996–998; store.rs:1029–1033) | **NOT an OK match.** Both numerator and denominator differ from the collector's `mean_tpot_ms = (last_token − first_token)/(observed_count − 1)` (collector.rs:505–514): native numerator uses terminal `end_ns` (≠ `last_token`, it is response/terminal completion **after** the final content token), and native denominator uses `completion_tokens` (≠ streamed count). Real (not sub-ULP) divergence whenever `end_ns ≠ last_token` or `completion_tokens ≠ streamed count`, which is the norm. **Records-first: recompute `(last_token − first_token)/(count − 1)` from `token_arrival_ns` per request** to preserve collector parity; the summary `InterTokenLatency` is NOT parity-safe. |
| `itl` distribution + `max_itl_ms` | **not in the summary** — collector `itl` is a per-token-**GAP** series (`collector.rs:988-993`); the native summary only holds per-request `InterTokenLatency`. Recompute from `RecordIngest.token_arrival_ns` (retained by `NativeMetricsCollection.records`, `metrics.rs:207-212,489-492`). | **GAP-2** (recoverable from records) |
| `e2e` distribution | `result(RequestLatency).distribution()` | RISK (time base + percentile algo) |
| `output_token_throughput_per_user` distribution | per-token-**GAP** `1000/itl` series (`collector.rs:990`); native `OutputTokenThroughputPerUser` is per-request (`catalog.rs:723-730`). Recompute from records. | **GAP-2** (recoverable from records) |
| `goodput_completed_requests` | `finite_value(GoodRequestCount)` | OK |
| `goodput_request_throughput_rps` | `finite_value(Goodput)` | OK |
| `goodput_output_throughput_tok_s` | — no good-only OTPS scalar in the catalog | **GAP-3**: minor; either add a native good-only OTPS scalar or recompute from records filtered by SLA. |

**Decision:** the projection must take **`&NativeMetricsCollection` (summary + records)**, not just the summary, so GAP-2 fields (`itl`, `max_itl`, `output_token_throughput_per_user`) keep the collector's per-gap semantics exactly (recompute with nearest-rank from `token_arrival_ns`). The runner already produces records (`RunCapture` uses `finish_with_records`); the library/offline finalizers expose `finish_with_records` too (`metrics.rs:375-378,423-442`).

---

## 6. Byte-parity analysis + risk

**What actually gates bytes today (verified):**
- **Offline SimClock** enforces byte-exact AIPerf==Dynamo (`dynosim.rs:963`, `offline_execution.rs:2266-2292` `verify_parity`). This path already prefers the Dynamo projection and imports engine-owned fields — **unaffected** by dropping the online collector, since it projects from Dynamo, not the collector.
- **Offline online-clock** (`replay_mode=online`) uses `verify_parity_online` (`offline_execution.rs:2299-2317`) which **does not byte-compare** — only field-accounting invariants. Its `independently_accumulated` logic already substitutes Dynamo when the collector is unfed (`dynosim.rs:924-932`), so setting `collect_performance_summary=false` makes it project from Dynamo cleanly.
- **Pure online product path:** the runner **discards** `report.performance` (`execute.rs:3276-3282`). There is **no product byte gate** on the online compat report.

**Therefore the projection changes only library-visible values, not any product byte gate.** The three real behavior deltas vs today's collector:

1. **Percentile algorithm (biggest risk).** Native record distributions use `linear_distribution` — interpolated percentiles (`kernel.rs:62-123`, called at `accumulator.rs:907,935`, `ddof=0`). The collector uses **nearest-rank** (`collector.rs:1119-1132`: `((len-1)*p/100).round()`). `median/p75/p90/p95/p99` of `ttft/ttst/tpot/e2e` will differ whenever a percentile falls between samples. `mean/min/max/std` are algorithmically identical (both arithmetic mean, both population std / `ddof=0`). *Mitigation options:* (a) accept native (interpolated) percentiles as the new compat semantics — the native report is already the authoritative product surface; or (b) if exact legacy percentiles are required for a specific consumer, recompute those distributions from `records` with the collector's nearest-rank helper (as the skeleton does for GAP-2). GAP-2 fields already use (b).

2. **Latency time base — TWO independent deltas (front AND back), not one.** **(a) Front / arrival base (ttft and e2e):** collector `ttft = first_token − arrival` and `e2e = last_token − arrival`, where `arrival` is the **credit-issued** time (`scheduled.rs:896-901`, `collector.rs:577-604,971-972`). Native `TimeToFirstToken = first_token − start` / `RequestLatency = end − start`, where `start` is the transport dispatch start / admit (`metrics.rs:447-456`), i.e. both **exclude** the credit→dispatch queue latency; request-rate/adaptive runs with admission backpressure diverge (closed-loop ~0). **(b) Back / terminal base (e2e only — previously missed):** even after fixing the front, native `RequestLatency = end_ns − start_ns` (accumulator.rs:996-998) ends at the **response/terminal completion** `end_ns` (after the final content token / usage frame), whereas the collector's `e2e` ends at the **last token arrival** `last_token` (collector.rs:972). `end_ns ≠ last_token` whenever a terminal/usage/`[DONE]`/trailing frame follows the last content token — the streaming norm — so an arrival-base-only fix still leaves a real back-tail offset. *Both are semantic changes, not rounding.* Preserve collector parity by projecting `ttft`/`e2e` **records-first** with the collector's exact endpoints: `first_token − arrival` and `token_arrival_ns.last() − arrival`, from retained `admit_ns`/`start_ns`/`token_arrival_ns` (`metrics.rs:485-492`) — do **not** substitute native `end_ns` for `last_token`.

3. **`num_requests` / goodput-OTPS gaps** (GAP-1, GAP-3): small; fill with a native arrivals scalar and a good-only OTPS scalar, or recompute from records.

**Recommendation:** make `project_trace_report` **records-first** for every distribution and for `num_requests` (recompute with the collector's exact arithmetic from `RecordIngest`), and use `AccumulatorSummary` scalars only for the throughput/count/duration fields that are already algorithm-identical. This yields a projection that is **byte-identical to today's collector** on every path (because it replays the same per-request math over the same retained facts), at zero per-token duplicate cost — the records already exist in the single native observer. The summary-only projection (skeleton default) is the cheaper fallback if a consumer accepts native percentile/time-base semantics.

---

## 7. Removal touch points (file:line)

Drop `CollectorObserver` from the delegate list / stop feeding it, then project at finalize:

- `rust/runtime/src/scheduled.rs:482-486` — `new_with_metrics_config` builds the 2-delegate tee. Change to native-only; keep `collector` field removal or make it optional. `finish_at` at `scheduled.rs:1115-1127` replaces `collector.finish()` with `project_trace_report(&native_collection)`.
- `rust/runtime/src/phase_runtime.rs:762-786` — already conditional on `plan.collect_performance_summary`; default it `false` for the online/offline plans (`phase_runtime.rs:324`) and route `finish` through the projection. The single-delegate collapse (`phase_runtime.rs:782-786`) already removes the `ObserverTee` allocation when only native remains.
- `rust/runtime/src/run.rs:434-441`, `run.rs:897-904`, `run.rs:1151-1158` — three online entry points build the tee; switch to native-only + projection at `run.rs:667` (`performance: collector.finish(wall_ms)` → `project_trace_report(...)`).
- `rust/runtime/src/dynosim.rs:2878-2885`, `dynosim.rs:3265-3266`, `dynosim.rs:3647-3648`, `dynosim.rs:3791-3792` — offline tees; native-only, and keep `compatibility_report_from_dynamo` (`dynosim.rs:986-1050`) for the SimClock byte gate. The `independently_accumulated` branch (`dynosim.rs:924-932`) already handles an unfed collector.
- `rust/runtime/src/engine/execute.rs:3067-3080,3262-3311` — already single-observer for tokens; the win here is removing the idle runtime collector+native allocation and their discarded finalization (or setting `collect_performance_summary=false` on the runner plan at `execute.rs:1963`).
- After no caller feeds it: `CollectorObserver` (`rust/loadgen-core/src/observer.rs`) and `ObserverTee` (`rust/runtime/src/metrics.rs:677-736`) lose their online/offline users; `TraceCollector` (`rust/loadgen-core/src/collector.rs`) survives only as the projection **target type** + the Dynamo parity path.

---

## 8. Enabling relationship to A1

A1 (retiring / relocating `loadgen-core`'s live `TraceCollector`/`CollectorObserver` off the request hot path) is **blocked** as long as the online and offline runtimes wire `CollectorObserver` into their observer tee. This change removes the **last live token-fed consumers** of `CollectorObserver`: after it, `TraceCollector` is only a *serialization target* (populated by projection, or by the Dynamo engine's own report). That collapses A1's remaining coupling to a data-type dependency, not a hot-path observer dependency, and unblocks A1's merge. Ship this before A1.

---

## 9. Verification evidence

**Compile-checked skeleton:** `~/tmp/a2-spec/` (`Cargo.toml` path-deps `loadgen-core` + `aiperf` for the `aiperf_runtime::metrics_core` module; `src/lib.rs` implements `project_trace_report`). `cargo build` → `Finished dev` (green). It proves against the **real** public types that:

- `AccumulatorSummary::result(tag).distribution()` yields `DistributionStats { avg, min, max, std: Option<f64>, percentiles: BTreeMap<u32, MetricValue> }` (`kernel.rs:24-43`) — exactly the `mean/min/max/std` + `p50/p75/p90/p95/p99` the compat `TraceDistributionStats` needs.
- `AccumulatorSummary::finite_value(MetricTag)` supplies every scalar count/throughput/duration/goodput field.
- `RecordIngest.token_arrival_ns: Vec<i64>` is public and lets the projection recompute the per-gap `itl` / `max_itl` / `output_token_throughput_per_user` with the collector's **nearest-rank** arithmetic (GAP-2), byte-preserving.
- The projected `TraceSimulationReport` re-serializes through the collector's custom `Serialize` (`projected_bytes`), i.e. the exact byte surface the parity gate compares.

The skeleton also encodes, in comments at each field, the two hard mismatches (linear-interp vs nearest-rank percentiles; transport-start vs credit-issued TTFT/e2e base) so the implementer chooses summary-cheap vs records-exact per field.

**Static proof re: offline precedent:** `dynosim.rs:986-1050` projects from `DynamoSimulationReport`, not `AccumulatorSummary` — it is **not** liftable to online; the online projection is genuinely new and is the skeleton above.

---

## 10. Open decisions for the implementer

1. **Percentile/time-base fidelity:** records-exact (byte-identical to legacy, recommended) vs summary-cheap (adopt native semantics). Pick per-consumer; library tests (`run.rs:1693`, `tests/scheduled_real_mock.rs`) pin some values and will need updating if summary-cheap is chosen.
2. **Where `project_trace_report` lives:** `rust/runtime/src/report.rs` (shared by `scheduled`/`run`/`phase_runtime`/`dynosim`), taking `&NativeMetricsCollection`.
3. **GAP-1 / GAP-3:** add a native "arrivals" scalar and a good-only OTPS scalar to the catalog, or recompute both from records (records-first path already has the data).
