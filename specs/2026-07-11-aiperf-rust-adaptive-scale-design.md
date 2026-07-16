<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: adaptive-scale — SLA-driven concurrency/rate autoscaling

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — `aiperf_runtime::adaptive_core` (formerly the `aiperf-adaptive` leaf crate,
now a module of `aiperf-runtime`) plus the online/offline composition in `aiperf_runtime::run` and the
CLI surface. All four control variables ramp online and offline over an injected backend.
**Grounding:** line-by-line read of the Python adaptive-scale subsystem —
`src/aiperf/timing/strategies/adaptive_scale.py`,
`adaptive_scale_controller.py`, `adaptive_scale_sla.py`,
`adaptive_scale_backends.py`, `adaptive_scale_runtime.py`,
`adaptive_scale_artifacts.py`, `adaptive_scale_types.py`,
`src/aiperf/timing/adaptive_config.py`, `adaptive_types.py`, and the
`SLAFilter` model in `src/aiperf/config/sweep/adaptive.py`.
**Companions (read first, do not duplicate):**
`2026-07-11-aiperf-rust-request-rate-multiturn-design.md` (the single-loop credit
issuer / `SlotPool` / `StopChecker` / two-plane framing that adaptive-scale drives on
top of), `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` +
`…-telemetry-accumulators-design.md` (the measured-metrics seam the window sampler
reads — referenced, not re-designed), `2026-07-11-aiperf-rust-exporters-overhaul-design.md`
(where the JSONL/summary artifacts land). `aiperf_runtime::timing` built primitives verified in
`rust/runtime/src/timing/{intervals,slots,stop}.rs`.

---

## 0. What this is

Adaptive-scale is a **closed-loop SLA controller** layered *over* an already-running
load phase. It does not issue credits itself — it keeps the existing
request-rate/concurrency issuance path running (`adaptive_scale.py:47`
subclasses `RequestRateStrategy`, and delegates to `UserCentricStrategy` when the
control variable is `users`, `:79-83`, `:233-253`) and runs a **background
assessment task** beside it (`_assessment_loop`, `:300-316`). Every assessment window
it: samples the SLA metrics over the requests that returned in that window, decides
pass/fail, and **mutates one control knob** — session concurrency, prefill
concurrency, request rate, or target users — to ramp load up until the SLA breaks,
then holds at the last-good level.

The only strategy type in this release is **`ramp_until_fail`** (`adaptive_config.py:97`,
enforced at `adaptive_scale.py:110`). The search is **not** binary search and **not**
Bayesian (that lives in the separate `AdaptiveSearchSweep`/BO planner, out of scope
here). It is a **monotone upward ramp with an SLA-margin-scaled step**, a
boundary-discovery step-back, and a **single-recovery sustain hold**. Three controller
phases (`adaptive_scale_types.py:10`): `discover` → `sustain` → `complete`.

The control knob is abstracted behind an `AdaptiveControlBackend` (`adaptive_scale_backends.py:15-27`)
— four concrete backends today (`concurrency`, `prefill_concurrency`, `request_rate`,
`users`, `:51-107`). These map **directly** onto AIPerf-Rust's already-built ramp
actuators: `SlotPool::set_limit` and `IntervalGenerator::set_rate`.

---

## 1. What runs, and the two clocks it touches

There are **two concurrent time-driven activities** during an adaptive phase:

1. **The credit issuer** (the request-rate/user-centric loop) — the single-loop
   dispatcher from the request-rate multi-turn design. Adaptive-scale never touches its
   inner loop; it only resizes the pools/rate it obeys.
2. **The assessment task** — `_assessment_loop` (`adaptive_scale.py:300-307`):

   ```
   while controller_phase != "complete" and stop_checker.can_send_any_turn():
       await sleep(assessment_period)     # :306
       await assess_window()              # :307
   ```

   `assessment_period` is `adaptive_assessment_period_sec` (default 30 s, floor 1 s;
   `adaptive_config.py:78-82`, floor re-checked at `adaptive_scale.py:103-107` against
   `MIN_ASSESSMENT_PERIOD_SEC = 1.0`, `adaptive_scale_types.py:12`).

Window accumulation happens on the credit-return path, guarded by an `asyncio.Lock`
(`adaptive_scale.py:122`, `:273`, `:297`, `:331`):

- `handle_first_token` records `ttft_ns` keyed by credit id (`:296-298`).
- `handle_credit_result` (`:272-294`): on error/no-latency → bump `errors` or (if
  cancelled) `cancelled`; else append `request_latency_ns`, optional
  `inter_token_latency_ns`, and a `WindowRequestSample`
  (`request_latency_ns`, `ttft_ns` popped by credit id, `itl_ns`, `output_sequence_length`).

`_take_window` (`:330-357`) snapshots the accumulators into a `WindowStats` and **resets
them to empty** (a tumbling, non-overlapping window). `elapsed_sec` = wall delta since
last take (`time.perf_counter()`), `start_ns`/`end_ns` = `time.time_ns()`.

> **Rust note (a simplification win):** on the `!Send` single-loop model the
> credit-return handler and the assessment task run on the **same** `LocalSet`, so the
> `asyncio.Lock` collapses to a plain `Rc<RefCell<WindowState>>` — no lock, no `Arc`.
> Both the `sleep(assessment_period)` and the `perf_counter`/`time_ns` window timing
> **must** go through `Clock` (`aiperf_runtime::clock`), never `tokio::time`/`Instant::now`,
> so the loop runs unchanged under `SimClock` (see §5).

---

## 2. The control loop + SLA math (ground truth, cited)

### 2.1 Window triage (`adaptive_scale_controller.py:20-91`, `assess_window`)

Per window, in order:

1. **No successful samples but errors/cancels** (`:23`): emit `adaptive_window`
   (`passed=False`), record a rejected candidate (`rejection_reason="error_threshold"`),
   route to `assess_failed_window`. (An *empty* idle window — no samples, no errors —
   falls through to the next check.)
2. **`len(samples) < adaptive_min_completed_requests`** (`:42`, default 1,
   `adaptive_config.py:120-124`): emit `adaptive_window` with `passed=None`
   (**inconclusive**), record a rejected candidate
   (`rejection_reason="insufficient_samples"`), and **return without a state
   transition** — the knob does not move.
3. Otherwise compute `sla_values = evaluator.values(filters, stats)` (`:60`),
   `primary_value = sla_values[key(primary_sla)]` (`:61`, primary = `filters[0]`,
   `adaptive_scale.py:99`), `passing = passes(all filters)` (`:62`). Emit
   `adaptive_window`, record the candidate (`accepted=passing`), and dispatch by phase:
   `assess_discover` (`:81`) or `assess_sustain` (`:89`).

`finally: strategy._advance_adaptive_iteration()` (`:91`) increments the iteration
counter every window regardless of outcome.

### 2.2 SLA evaluation (`adaptive_scale_sla.py`)

Each `SLAFilter` (`sweep/adaptive.py:23-59`) = `(metric_tag, stat, op, threshold)`.
`stat` ∈ latency stats (`avg/min/max/p1/p5/p10/p25/p50/p75/p90/p95/p99`) for
latency-family metrics, or `{avg,min,max}` for rate-family (`adaptive_scale_sla.py:19-33`,
validated `:289-309`). `op` ∈ `{lt,le,gt,ge}` (`:320-331`). A filter **passes** iff
`stat(metric) op threshold` (`passes_single`, `:320-331`); the window passes iff **all**
filters pass (`passes`, `:315-318`).

Metric families (`value`, `:234-264`):

| Metric tags | Source | Value |
|---|---|---|
| `request_latency` | latency-ns samples | percentile/agg in **ms** (`:82-96`) |
| `time_to_first_token`, `ttft` | ttft-ns samples | ms; **`inf` if no samples** (`:98-102`) |
| `inter_token_latency`, `itl`, `tpot` | itl-ns samples | ms; **`inf` if no samples** (`:104-118`) |
| `throughput`, `request_throughput`, `completed_request_throughput` | window | `len(samples)/elapsed_sec` (`types.py:72-76`) |
| `output_token_throughput` | window | `sum(osl)/elapsed_sec` (`types.py:78-82`) |
| `goodput` | per-request | `good_count/elapsed_sec` (`:188-200`) |
| `goodput_ratio` | per-request | `good_count/total` (`:202-214`) |
| `success_rate`, `request_success_rate` | window | `len(samples)/total` (`:139-146`) |
| `error_rate`, `request_error_rate` | window | `errors/total` (`:216-223`) |
| `cancellation_rate`, `request_cancellation_rate` | window | `cancelled/total` (`:225-232`) |

**Scars to carry byte-for-byte:**
- **ns→ms** = `/1_000_000` everywhere latency crosses into an SLA value (`:85`, `:153`, etc.).
- **Empty ttft/itl ⇒ `math.inf`** (`:100`, `:107`) — so a latency-upper-bound SLA
  *fails* when no tokens arrived, rather than dividing by zero.
- **Percentile kernel** (`percentile_value`, `:334-349`): sort; `rank=(p/100)*(n-1)`;
  linear interpolation between `floor`/`ceil`; single-sample returns that sample.
  This is the **same interpolation the `aiperf_runtime::metrics_core` spec pins** — reuse that kernel,
  do not re-derive.
- **Goodput is per-request quality-gated** (`_good_request_count`, `:164-186`): a
  request is "good" iff it passes **every** quality filter (`request_latency`/`ttft`/`itl`,
  `QUALITY_METRICS`, `:36`) on its own per-request value; missing per-request value ⇒
  not good. `validate_filters` (`:273-287`) **rejects** a goodput/goodput_ratio SLA that
  has no accompanying quality filter.
- **`total = len(samples)+errors+cancelled`** (`types.py:48-50`) — the rate
  denominators include failures/cancels; `success_rate` etc. are over *all* returns.
- **`throughput` uses `elapsed_sec ≤ 0 ⇒ 0.0`** (`types.py:73-76`) — the window wall
  duration, which under `SimClock` is virtual-ns elapsed (§5).

### 2.3 The `discover` ramp (`assess_discover`, `controller.py:113-173`)

- **Passing** (`:122`): `last_good = current`. If `current >= maximum` → complete with
  `adaptive_incomplete` / reason `max_control_value_reached_without_saturation`
  (`:124-135`) and stop sending. Otherwise **step up**:
  `next = _next_up(sla_values) = min(maximum, current + step_size(current, sla_values))`
  (`adaptive_scale.py:392-397`), `set_control(next)`, emit `adaptive_decision`.
- **Failing** with **no `last_good` yet** (`:154`): complete `adaptive_failed` /
  `no_sustainable_concurrency_found` and stop — the SLA broke at the very first
  (minimum) level.
- **Failing** with a `last_good` (`:167`): record `first_failing = current`, then
  `enter_sustain(...)`.

An **all-failed** window in discover (`assess_failed_window`, `:93-109`) is the same
fork: no `last_good` ⇒ `no_sustainable_concurrency_found`; else `enter_sustain(None, …)`.

### 2.4 The step policy (`_step_size`, `adaptive_scale.py:399-435`)

- **`fixed_percent_step`**: `max(1, ceil(current * step_percent/100))`
  (`:402-404`; `step_percent` default 25, `adaptive_config.py:115-119`).
- **`sla_margin`** (default): `base_step * multiplier` where
  - per-filter **margin** `= (threshold - observed)/|threshold|` for `lt/le`,
    `(observed - threshold)/|threshold|` for `gt/ge` (`_sla_margin`, `:198-205`;
    `None` if `threshold==0` or observed missing) — normalized head-room, **positive =
    passing with slack, → 0 as it approaches violation**;
  - `effective_margin = max(0, min(margins over all filters))` (`:427`) — the
    **binding** (tightest) filter governs;
  - `multiplier = clamp(1, max_step_multiplier, int(effective_margin * max_step_multiplier))`
    (`:428-434`; `base_step`=10, `max_step_multiplier`=4 defaults,
    `adaptive_config.py:105-114`).

  So a window with lots of head-room takes a big step (up to 4× base); a window near the
  edge takes the base step. This is the "exponential-ish when far, linear when close"
  behavior — **not** binary search.

### 2.5 `enter_sustain` + the `sustain` hold (`controller.py:175-328`)

`enter_sustain` (`:294-328`): requires a `last_good` (raises otherwise, `:297-298`);
`boundary = max(minimum, last_good)`; `set_control(boundary)` (**steps the knob back
down** from the failing level to the last passing one); `phase = "sustain"`;
`sustain_started_at = perf_counter()`; emits `sustain_started` + `boundary_discovered`.

`assess_sustain` (`:175-191`): `sustain_windows += 1`; then:

- **Passing** (`_assess_passing_sustain`, `:193-208`): `sustain_passed_windows += 1`;
  `last_good = current`; **reset recovery** (`sustain_recovery_used = False`); emit
  `adaptive_decision`; then `_complete_sustain_if_elapsed` (`:275-292`): if
  `perf_counter() - sustain_started_at >= sustain_duration` → complete
  `adaptive_complete` / `sustain_duration_completed` and stop.
- **Failing** (`_assess_failing_sustain`, `:210-245`): **one** recovery attempt only.
  - If `sustain_recovery_used` already ⇒ `_fail_sustain` (`sustain_failed_after_recovery`,
    `adaptive_failed`).
  - Else set `recovery_used = True`; `target = _sustain_recovery_target` (`:247-259`) =
    `max(minimum, last_good)` if that's `< before`, else `max(minimum, before - step_size)`;
    if `target == before == minimum` ⇒ `_fail_sustain` (`sustain_failed_sla_unrecoverable`);
    else `set_control(target)`, emit `adaptive_decision`.

`sustain_duration = adaptive_sustain_duration_sec` is **required** (validated
`adaptive_scale.py:108-109`; `adaptive_config.py:73-77`).

### 2.6 Terminal + artifacts (`adaptive_scale_runtime.py`, `…_artifacts.py`)

`_complete_controller` (`runtime.py:96-128`) is **idempotent** (`:107-108`): set
`phase="complete"`, record `completed_reason`, emit the terminal event, write the
summary once (`:142-179`). Status map (`_status_for_terminal_reason`, `:130-140`):
`max_…_without_saturation` → `incomplete`; `assessment_failed:*` /
`no_sustainable_concurrency_found` / `sustain_failed_*` → `failed`; else `completed`.
`_stop_sending` (`adaptive_scale.py:318-322`) marks the phase sending-complete + freezes
progress counts so the credit issuer winds down.

Two artifacts (`SCHEMA_VERSION=2`, `artifacts.py:22`), written through an async queue +
`to_thread` writer (`:31-81`; orjson, sorted keys):
- **`adaptive_scale_events.jsonl`** — one line per event: `adaptive_phase_started`,
  `adaptive_window`, `adaptive_decision`, `sustain_started`, `boundary_discovered`,
  terminal (`adaptive_complete`/`adaptive_failed`/`adaptive_incomplete`). Rich payload
  (`event_payload`, `:177-239`): control before/after, boundary/last-passing/first-failing,
  primary + all `sla_values`, the **binding SLA key** (`_binding_sla_key` = filter with
  smallest margin, `adaptive_scale.py:183-196`), throughput, counts, `sla_passed`,
  correlation block (`run_id`/`phase_id`/`adaptive_iteration`/candidate/accepted).
- **`adaptive_scale_summary.json`** — final roll-up (`summary_payload`, `:241-313`):
  status, boundary/last-passing/first-failing `result`, sustain windows/passed,
  primary SLA, per-window `candidates[]` (`candidate_payload`, `:114-175`: p50/p95/p99
  latency+ttft+itl in ms, throughput, success rate, accept + reason).

---

## 3. The seams (every extension point a trait)

The Python code is already trait-shaped (Protocol backend + evaluator/controller split);
in Rust it becomes five explicit traits, each object-safe at the `dyn` boundary.

| Seam | Python origin | Abstracts | Impls |
|---|---|---|---|
| **`ControlActuator`** | `AdaptiveControlBackend` (`backends.py:15-27`) | one knob: `min`/`max`/`current`/`set(value)`/`snapshot()` | `ConcurrencySlot`, `PrefillSlot`, `Rate`, `Users` — thin adapters over built `SlotPool::set_limit` / `IntervalGenerator::set_rate` / user-target hook |
| **`SlaEvaluator`** | `AdaptiveScaleSLAEvaluator` (`sla.py:78`) | `WindowStats → {key: value}`, `passes`, `margin`, `binding` | one impl carrying the metric-family + percentile scars |
| **`StepPolicy`** | `_step_size` (`adaptive_scale.py:399`) | `(current, margins) → step` | `SlaMargin`, `FixedPercent` |
| **`WindowSampler`** | window accumulators + `_take_window` (`adaptive_scale.py:132-140`,`:330-357`) | accumulate returns → tumbling `WindowStats`, snapshot+reset | one impl tapping the observer/metrics seam |
| **`Controller`** | `AdaptiveScaleController` (`controller.py:17`) | the discover/sustain/complete state machine + transition decisions | one impl (`RampUntilFail`); the `strategy_type` enum is the seam for a future search |

Supporting: **`ArtifactSink`** (events + summary) is not adaptive-specific — it is an
`Exporter` in the exporters-overhaul family (`2026-07-11-…-exporters-overhaul-design.md`);
adaptive-scale emits typed `AdaptiveEvent`/`AdaptiveSummary` records into it.

`ControlActuator` is the load-bearing abstraction: because the controller only ever
calls `actuator.set(value)`, the *identical* discover/sustain logic ramps concurrency,
prefill, rate, or users with zero branching — exactly the Python `build_adaptive_control_backend`
dispatch (`backends.py:127-145`).

---

## 4. Mapping onto the crates — built vs designed

| Concern | Primitive | Crate | Status |
|---|---|---|---|
| Concurrency ramp actuator (`set_session_limit`) | `SlotPool::set_limit` (debt-drain) | `aiperf_runtime::timing` | **built** (`slots.rs:151`) |
| Prefill ramp actuator (`set_prefill_limit`) | `SlotPool::set_limit` (prefill pool) | `aiperf_runtime::timing` | **built** (`slots.rs`, `ConcurrencyManager`) |
| Rate ramp actuator (`set_request_rate`) | `IntervalGenerator::set_rate` | `aiperf_runtime::timing` | **built** (`intervals.rs:50`) |
| Assessment-period sleep + window/sustain timing | `Clock::sleep` / `now_ns` | `aiperf_runtime::clock` | **built** |
| Loop gate (`can_send_any_turn`) | `StopChecker` | `aiperf_runtime::timing` | **built** (`stop.rs:167`) |
| Underlying credit issuance path | `CreditIssuer` / `RateWorkload` | request-rate multiturn spec | **designed** (its own spec) |
| Percentile / agg kernel over window samples | `aiperf_runtime::metrics_core` percentile kernel (`linear_distribution`) | `aiperf_runtime::metrics_core` | **built** (reused, not re-derived) |
| Measured-return stream feeding the window | `RequestObserver` → `WindowSampler` (`AdaptiveObserver` tee) | `loadgen-core` / `aiperf_runtime::adaptive_core` | **built** |
| **`ControlActuator` trait + 4 actuators** | — | `aiperf_runtime::adaptive_core` | **built** |
| **`SlaEvaluator`** | — | `aiperf_runtime::adaptive_core` | **built** |
| **`StepPolicy` (SlaMargin/FixedPercent)** | — | `aiperf_runtime::adaptive_core` | **built** |
| **`WindowSampler` (tumbling, reset)** | — | `aiperf_runtime::adaptive_core` | **built** |
| **`Controller` (RampUntilFail state machine)** | — | `aiperf_runtime::adaptive_core` | **built** |
| **Adaptive events/summary artifacts** | — | `aiperf_runtime::adaptive_core` (`AdaptiveArtifactSink`/`FileArtifactSink`) | **built** |

**Home module:** the control logic lives in `aiperf_runtime::adaptive_core` (the former
`aiperf-adaptive` leaf crate, now a module of `aiperf-runtime`), depending on the timing
primitives (actuators/stop), `aiperf_runtime::clock` (time), and the metrics seam (percentile
kernel). It is pure control logic with no HTTP/engine deps. All four actuators
(`SessionConcurrencyActuator`, `PrefillConcurrencyActuator`, `RequestRateActuator`,
`UsersActuator`) live together in `adaptive_core`, superseding the earlier proposal to
split them across their target crates.

The **actuator row** rides the ramp knobs that already existed and are debt-drain-graceful
(a downward `set_limit` in `enter_sustain`/recovery drains in-flight rather than
hard-cancelling, `slots.rs:151`, exactly the Python `DynamicConcurrencyLimit` semantics the
request-rate spec §1.1 cites). The **controller + evaluator + sampler + step policy** are
pure logic with no I/O, unit-tested against synthetic windows; the Python `_percentile`
kernel and the pass/fail forks are ported as parity fixtures.

---

## 5. Online / mock / offline parity — the two-plane point

Adaptive-scale is a **closed-loop controller that requires live measured metrics
mid-run** — it reads a `WindowSampler` over *already-returned* requests every
`assessment_period` and feeds the decision back into the actuator **while the phase is
still running**. This is the sharpest instance of the two-plane framing: the control
plane (issuer + controller) must observe the data plane's *completions* in near-real
time, not just an end-of-run roll-up.

That said, nothing in the loop needs a wall clock:

- **ONLINE-real / ONLINE-mock** (`RealClock`): assessment task and issuer share the
  `LocalSet`; `Clock::sleep(assessment_period)` paces windows; the observer records
  returns into the `Rc<RefCell>` window state; identical code, only the base URL differs.
- **OFFLINE** (`SimClock`): adaptive-scale **runs unchanged and deterministically**,
  *provided* window timing and the assessment sleep go through `Clock` (not
  `tokio::time`). Under `drive_sim`, the assessment `sleep` is a heap event; the window
  `elapsed_sec` is virtual-ns elapsed; the returns are produced by the feature-gated
  in-process engine sink. Because throughput/goodput denominators are `elapsed_sec`
  (§2.2), they are well-defined on the virtual timeline. The ramp is then a fully
  reproducible DES: same seed ⇒ same boundary discovered.

**The one hard requirement offline:** the engine sink must deliver completions
*incrementally as virtual time advances*, so a window mid-run has samples. A sink that
only reports at the end starves every window (`insufficient_samples`, §2.1) and the
controller never ramps. The feature-gated in-process Dynamo sink (`DynosimSink`) satisfies
this — it feeds real engine completion events through the same paced issuer,
`AdaptiveObserver`, and sampler for all four offline control variables — the offline
analogue of the request-rate spec's live TTFT/return hooks (the same "measurements must
flow during the run" contract).

Parity is **code-path + report-schema, not byte-identical boundary values**: simulated
vs real latencies differ by construction, so the *discovered* concurrency/rate will
differ; the event/summary schema and the decision logic are identical.

---

## 6. Composition (built)

All of the following are built in `aiperf_runtime::adaptive_core` plus the composition functions
in `aiperf_runtime::run` and the CLI surface:

1. **`ControlActuator` trait + concurrency/prefill/rate/users actuators** over
   `SlotPool::set_limit` / `IntervalGenerator::set_rate` and a live `UserTarget` gate,
   with clamp + `snapshot`. Bounds validation follows `backends.py:110-226` (int-≥1 for
   concurrency/prefill/users, `max>min`, prefill `max ≤ concurrency`, rate rejects
   `CONCURRENCY_BURST`).
2. **`WindowStats` + `DefaultSlaEvaluator`** — the metric-family table, aliases, and
   statistics with `passes`/`margin`/`binding`, reusing the `aiperf_runtime::metrics_core`
   `linear_distribution` percentile kernel. The Python SLA unit tests are ported as
   parity fixtures (empty→`inf`, ns→ms, goodput quality-gate).
3. **`SlaMarginStep` + `FixedPercentStep`** — the margin→multiplier clamp exactly.
4. **`TumblingWindowSampler`** — tumbling accumulate/snapshot/reset over the observer's
   returns (`Rc<RefCell>`, no lock); TTFT-by-credit-id join. `ObservedUsage` carries
   authoritative `completion_tokens` into output sequence length and the
   `(last−first)/(osl−1)` ITL denominator; missing usage leaves both absent.
5. **`RampUntilFailController`** — the discover/sustain/complete state machine and every
   terminal reason, driven by an `AdaptiveScale` assessment task paced exclusively through
   `Clock::sleep` (so `SimClock` tests drive the same controller deterministically). A
   local waker-backed stop future interrupts a long issuer arrival sleep as soon as the
   controller becomes terminal.
6. **Artifacts** — typed `AdaptiveEvent`/`AdaptiveSummary` records written through
   `AdaptiveArtifactSink` / `FileArtifactSink` as schema-v2 `adaptive_scale_events.jsonl`
   + `adaptive_scale_summary.json` (binding-SLA + candidates, recursively sorted keys).

**Backend-neutral online/offline composition.** `aiperf_runtime::run` exposes backend-neutral
paced, request-rate, user-centric, and adaptive composition functions. Online wrappers
inject `RealClock + TransportSink`; offline (feature-gated) wrappers inject
`SimClock + DynosimSink` and run the *same* futures, observers, actuators, issuance gates,
and artifact sinks. The support matrix is complete for all four control variables in their
owning workloads:

- paced concurrency: `concurrency` and `prefill_concurrency`;
- continuation-priority request rate: `request_rate`;
- user-centric scheduling: `users`.

The `AdaptiveObserver` tees returned-request events into the sampler and the ordinary
collector on the one-thread `LocalSet`; to keep that hot path lock-free, `RequestObserver`
is a local-loop trait without `Send`/`Sync` supertraits, its optional
`on_usage(ObservedUsage)` callback carries endpoint counts, and `CollectorObserver` stores
its collector in `RefCell`. Prefill slots release at the first meaningful parsed SSE
token — not a role/usage frame — with terminal fallback
(`HttpTransport::send_request_with_first_token_filter` retries the callback until the chat
parser accepts a delta). Successful request latency runs from admission/dispatch to the
last meaningful token; a terminal response with no meaningful token is an error, matching
the Python credit-return record semantics.

The CLI exposes `--adaptive-scale`, all four control variables, tumbling and sustain
durations, the explicit `ramp_until_fail` strategy selector, repeatable SLA filters, both
step policies, control bounds, minimum completions, and an artifact directory. Unit and
integration coverage exercises SLA math and aliases, error/cancel and sparse windows, both
step policies, every controller terminal path, recovery reset, all four live actuators,
`SimClock` pacing, schema-v2 artifacts, early online stop, and executable CLI acceptance
of terminal adaptive windows and failure/summary artifacts for offline concurrency,
request rate, and users.

---

## 7. Risks / open questions

- **Inconclusive-window starvation (convergence).** If `assessment_period` is short
  relative to the achieved rate, every window has `< min_completed_requests` samples and
  the controller **neither ramps nor fails** (`controller.py:42-58`) — it can idle
  forever until `StopChecker` ends the run. The floor is 1 s; the Python default 30 s is
  a deliberate hedge. Document that assessment period must be ≫ round-trip latency, and
  consider a guard that surfaces "persistently inconclusive."
- **Warmup / settling inside a window.** A ramp step changes load, but the *next* window
  includes the transient — there is **no intra-window warmup discard** (the tumbling
  boundary *is* the settling assumption). A too-short window measures a not-yet-settled
  system and can prematurely trip the SLA. Open question: whether to skip the first
  window after each `set_control` (the Python code does not).
- **Single recovery only.** Sustain tolerates exactly one failing window (recovery step
  down), then the next failure is terminal (`_assess_failing_sustain`, `:219-224`). A
  flapping server near the boundary fails the run rather than oscillating — intended, but
  worth surfacing in the summary (`sla_passed_during_sustain`, `artifacts.py:285-287`).
- **Boundary is `last_good`, not `first_failing`.** `enter_sustain` holds
  `max(min, last_good)` (`controller.py:299-305`) — it steps *back down* to the last
  passing level. The reported boundary is conservative by one step; don't confuse
  `boundary_value` with `first_failing_value` in the report.
- **Phase interaction with the issuer.** Adaptive-scale mutates the *same* `SlotPool` /
  `IntervalGenerator` the credit issuer is actively draining. On the single loop this is
  serialized and safe (no lock), but the debt-drain semantics of a downward `set_limit`
  mid-flight must be exercised: a recovery step-down during sustain should drain, not
  cancel, in-flight requests (`slots.rs:151` debt path) so the *next* window isn't
  polluted by cancellations counted as `cancelled` (which inflate `cancellation_rate` and
  deflate `success_rate`).
- **`throughput`/`goodput` depend on `elapsed_sec`** — under `SimClock` this must be
  virtual-ns elapsed via `Clock`, or the rate SLAs are meaningless offline (§5).
- **Users variable rides the user-centric workload.** `UsersControlBackend.set`
  (`backends.py:94-99`) hard-requires a `set_target_users` hook; the Rust `UsersActuator`
  mutates a live `UserTarget` gate on the user-centric workload and starts from the
  configured adaptive minimum, online and offline.

---

## 8. One-line summary

Adaptive-scale is a **closed-loop `ramp_until_fail` controller** that, every
`Clock`-paced assessment window, evaluates SLA filters over the requests returned in
that window and **ramps one knob** — session/prefill concurrency (`SlotPool::set_limit`),
request rate (`IntervalGenerator::set_rate`), or target users — **upward with an
SLA-margin-scaled step** until the SLA breaks (`discover`), then **holds at the last
passing level** for a sustain duration with a single recovery step-down (`sustain`),
emitting per-window decision events + a boundary summary; the ramp actuators are already
built over the timing ramp knobs, the controller/evaluator/step-policy/window-sampler are
pure-logic seams in `aiperf_runtime::adaptive_core`, and the whole loop runs deterministically
offline under `SimClock` **iff** completions flow during the run — which the feature-gated
`DynosimSink` supplies for all four control variables.
