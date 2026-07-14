<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: ancillary timing policy — ramping, cancellation, URL sampling

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design (not built)
**Grounding:** end-to-end line-by-line reads of the Python timing subsystem —
`src/aiperf/timing/ramping.py` (full: `Ramper` loop + `BaseRampStrategy` +
`Linear`/`Exponential`/`Poisson` strategies), `src/aiperf/timing/request_cancellation.py`
(full: `RequestCancellationConfig` + `RequestCancellationSimulator`),
`src/aiperf/timing/url_samplers.py` (full: `URLSelectionStrategyProtocol` +
`RoundRobinURLSampler`). Caller wiring read in full where it matters:
`src/aiperf/timing/phase/runner.py:457-534` (`_create_rampers`),
`src/aiperf/credit/issuer.py:197-238` (cancel + url_index at credit-issuance),
`src/aiperf/timing/intervals.py:55-234` (the `set_rate` actuators),
`src/aiperf/timing/concurrency.py:97-164` (the `set_limit` debt-drain actuator),
`src/aiperf/workers/worker.py:490-501,744-748` + `src/aiperf/workers/session_manager.py:144-206`
(sticky per-session url_index), `src/aiperf/transports/aiohttp_transport.py:206-223`
(url_index → resolved base URL), `src/aiperf/common/environment.py:875-879`
(`RATE_RAMP_UPDATE_INTERVAL` default 0.1s).
**Companions (read in full):** `specs/2026-07-11-aiperf-rust-request-rate-multiturn-design.md`
(the `IntervalGenerator.set_rate` / `SlotPool.set_limit` / `StopChecker` / `Clock`
context; `cancel_after_ns` on the credit = simulated disconnect delay),
`specs/2026-07-10-aiperf-transport-rust-port-design.md` (cancellation-after-send is
a named design target), `specs/README.md` (status conventions).

---

## 0. What this is

Three **ancillary** timing-policy knobs that ride on top of the core credit-issuer
loop (the companion request-rate spec). None of them is a workload of its own; each
perturbs an already-running phase:

1. **Ramping** — smoothly walk the target **rate** (via `IntervalGenerator::set_rate`)
   or **concurrency limit** (via `SlotPool::set_limit`) from a start value to a
   target value over a fixed duration, on a pluggable curve.
2. **Request cancellation** — probabilistically arm a **client-disconnect timer** on a
   fraction of requests, to test how the server handles mid-response disconnects.
3. **URL sampling** — when several `--url` endpoints are given, pick which one each
   *conversation* hits, load-balancing across backends.

All three are **policy** we keep; all three reduce to a tiny trait + an actuator that
**already exists** in `aiperf-timing`/`aiperf-transport-http`. The Python code is small and
was read whole; the earned-in-blood details are the curve math (`ramping.py`), the
"start the cancel timer at send-complete, not at issuance" invariant
(`request_cancellation.py:62-81`), and the "advance round-robin on turn-0 only, then
pin per-session" invariant (`issuer.py:212-220` + `worker.py:496-501`).

---

## 1. Ramping

### 1.1 The config surface (`ramping.py:65-94`)

`RamperConfig` is `frozen`, floats throughout (`ramping.py:65-70`). Fields:
`ramp_type` (curve selector), `start` (`>0`), `target` (`>0`), `duration_sec` (`>0`),
`update_interval` (optional; **its presence flips discrete→continuous mode**,
`ramping.py:80-84`), `step_size` (optional, LINEAR discrete step, default `1.0`,
`ramping.py:85-89`+`362`), `exponent` (optional, `>1.0`, EXP curve, default `2.0`,
`ramping.py:90-94`+`381`).

### 1.2 Two drive modes (`Ramper`, `ramping.py:133-246`)

The `Ramper` owns a setter (`Callable[[float], None]`) + a strategy, and runs one of
two loops chosen by `update_interval is None` (`ramping.py:171-175`):

- **Discrete** (`_run_discrete`, `:185-216`): sets `start`, then repeatedly asks
  `strategy.next_step(current, elapsed) -> (delay, next_value) | None`, `sleep(delay)`,
  apply `next_value`. `None` ⇒ done ⇒ **force-set `target`** (`:204-205`, covers
  timing drift / `start==target`). Used for **concurrency** (integer +1/+step_size).
- **Continuous** (`_run_continuous`, `:218-246`): sets `start`, then loops
  `sleep(update_interval)` → `value = strategy.value_at(elapsed)` → apply; `None` ⇒
  set `target` and stop (`:237-242`). Used for **rate** (smooth float interpolation).

On completion, both modes guarantee the exact `target` is applied. On cancel
(`stop()` → `Task.cancel()`, `:177-183`+`:214-216`+`:244-246`) the ramp **freezes at
the current value** — the caller decides what happens next. `is_running` tracks the
task (`:160-163`).

**Rust note (mandatory):** the Python loops use `time.perf_counter()`
(`:192,199,225,234`) and `asyncio.sleep`. In Rust `elapsed` and every sleep MUST go
through `Clock` (`Clock::now_ns` + `Clock::sleep`), so a ramp under `SimClock` steps in
virtual time deterministically. **Never `tokio::time`** (its 1 ms wheel would smear the
`update_interval`).

### 1.3 The curve math (`BaseRampStrategy`, `ramping.py:254-349`)

Common state (`:262-269`): `start`, `target`, `duration`, `range = |target-start|`,
`direction ∈ {-1,0,+1}`. Two hooks a curve overrides: `_apply_curve(progress)` maps
**value-progress → time-fraction**; `_time_to_value_progress(time)` maps
**time-progress → value-progress** (must be the exact inverse of `_apply_curve`).

- **`next_step`** (discrete, `:281-303`): terminate if `current==target`, `range==0`,
  or overshoot (`:289-297`). Else `next_val = _compute_next_value(current)`;
  `progress = clamp01(|next_val-start| / range)`;
  `time_at_next = duration * _apply_curve(progress)`;
  `delay = max(0, time_at_next - elapsed)`. Return `(delay, next_val)`.
- **`value_at`** (continuous, `:305-331`): `None` if `range==0` or `elapsed>=duration`;
  else `time_progress = clamp01(elapsed/duration)`,
  `value_progress = _time_to_value_progress(time_progress)`,
  return `start + range*direction*value_progress`.

Three shipped curves (the curve **family** to port):

| Curve (`RampType`) | `_compute_next_value` | `_apply_curve(p)` | `_time_to_value_progress(t)` |
|---|---|---|---|
| **Linear** (`:357-368`) | `current + step_size*direction`, clamped to target | `p` (identity) | `t` (identity) |
| **Exponential** ease-in (`:376-396`) | `current + direction` (unit step) | `p ** (1/exponent)` | `t ** exponent` |
| **Poisson** (`:404-487`) | *n/a* — pre-computed trajectory | *n/a* | *n/a* |

**Exponential** is a slow-start-accelerate ease-in: value rises as `t^exponent`
(`:394-396`); the discrete `_apply_curve` inverts that (`p^(1/exponent)`, `:390-392`)
so `next_step` schedules the same shape.

**Poisson** (`:404-487`) pre-generates a step-function trajectory in `__init__`
(`:419-460`), not per-step:
1. RNG derived `rng.derive("timing.ramp.poisson")` (`:421`) — **reproducible**.
2. Rate `λ = range/duration` (`:431`); draw `expovariate(λ)` intervals until their
   cumulative sum exceeds `duration` (`:435-438`) → stochastic event count.
3. **Time-normalize:** `time_scale = duration/cumulative` so events fit exactly
   (`:441`). **Value-normalize:** `step_size = range/num_events` (`:445`); values step
   linearly toward target, the **final event pinned to exact `target`** to avoid FP
   drift (`:453-458`).
4. `next_step` walks the trajectory by index (`:466-478`);
   `value_at` is a `bisect_right` over `event_times` (step function, `:480-487`).

### 1.4 What gets ramped, and the schedule (`runner.py:457-534`)

`_create_rampers` builds up to three rampers per phase from phase config:

- **Session concurrency** (`:467-485`): `LINEAR`, **discrete**, `start=1`,
  `target=concurrency`, `duration=concurrency_ramp_duration_sec`; setter =
  `concurrency_manager.set_session_limit(phase, int(limit))`.
- **Prefill concurrency** (`:487-505`): same shape, `target=prefill_concurrency`,
  `duration=prefill_concurrency_ramp_duration_sec`; setter =
  `set_prefill_limit(phase, int(limit))`.
- **Request rate** (`:507-534`): `LINEAR`, **continuous**
  (`update_interval = RATE_RAMP_UPDATE_INTERVAL`, default **0.1s**,
  `environment.py:875-879`), `target=request_rate`, and a **proportional start**
  `start_rate = request_rate * (update_interval / ramp_duration_sec)` (`:512-514`) —
  deliberately *not* a fixed 1 QPS, so a sub-1-QPS target doesn't ramp *up*
  (`:509-510`). Setter = `strategy.set_request_rate` (guarded by `RateSettableProtocol`,
  `:526-533`), which forwards to `IntervalGenerator::set_rate`
  (`request_rate.py:302` → `intervals.py`).

### 1.5 The two actuators (already built in Rust)

- **`IntervalGenerator::set_rate(new_rate)`** — `intervals.py:69-234`. Poisson updates
  `λ` (`:115-119`), Gamma re-derives shape/scale keeping mean `1/rate`
  (`:165-174`), Constant recomputes `period=1/rate` (`:195-200`), Burst is a **no-op**
  (rate is meaningless under concurrency-drive, `:227-229`). "Takes effect on the next
  `next_interval()`" (`:76`).
- **`SlotPool::set_limit(new_limit)`** — `concurrency.py:97-140`. **Increase:** cancel
  debt first, then `release()` extra slots (`:121-127`). **Decrease:** drain available
  slots immediately, remainder tracked as **debt** absorbed by future `release()`
  (`:128-138`+`:154-164`). This debt-drain is exactly the `SlotPool` already ported into
  `aiperf-timing` (companion spec §1.1).

### 1.6 Phase interaction

Rampers are created **per phase** and set **phase-scoped** limits
(`set_session_limit(phase, …)`, `runner.py:481-483`). Concurrency ramps start at `1`
regardless of the phase's steady-state limit; the ramp **is** how the phase reaches its
limit. Cancelling a ramp mid-flight leaves the actuator at the last-applied value (no
snap to target). In Rust the rampers are `spawn_local` tasks on the same `!Send`
`LocalSet` as the credit issuer, all sharing the one `Clock`.

---

## 2. Request cancellation (simulated client disconnect)

### 2.1 The policy (`request_cancellation.py:17-141`)

`RequestCancellationConfig` (`frozen`, `:17-37`): `rate` — **percentage** `0-100` of
requests to cancel (`None`/`0` disables, `:27-32`); `delay` — seconds to wait **after
the request is fully sent** before cancelling (default `0.0`, `:33-37`).

`RequestCancellationSimulator.__init__` (`:93-108`): derives
`rng.derive("timing.request.cancellation")` (`:102`), precomputes
`enabled = bool(rate)` (`:104`), `cancellation_rate = rate/100` (`:105-107`),
`cancellation_delay_ns = int(delay * NANOS_PER_SECOND)` (`:108`).

`next_cancellation_delay_ns(turn, phase)` (`:110-135`) — the whole decision:
```
if not enabled:              return None      # :125-126
if phase == WARMUP:          return None      # no cancel during warmup :128-129
if rng.random() < rate:      return delay_ns  # Bernoulli hit :132-133
else:                        return None
```

**Distribution reality (do not embellish):** the *delay* is a **fixed constant**
(`cancellation_delay_ns`), identical for every cancelled request. The **only**
randomness is the per-request Bernoulli draw `rng.random() < cancellation_rate`
(`:132`). There is no per-request delay distribution. `delay=0` means "send the full
request, then immediately disconnect" (`:80-81`). Cancellation is **off during warmup**
by construction (`:128-129`).

### 2.2 The timing invariant (`request_cancellation.py:53-82`)

The cancel timer starts at **T2 = request fully sent** (headers+body on the socket),
**not** at credit issuance (T0) and **not** at connection acquire (T1). This guarantees
the server always receives the *complete* request before the disconnect
(`:78-81`). The delay measures from send-complete to abort. If the timer fires while
still awaiting the response, the request is cancelled and its terminal record carries
`error="RequestCancellationError"` (`:59-60`). This is **distinct from credit
cancellation** (the `CancelCredits` clean-shutdown path, `:88-90`) — different concept,
do not merge.

### 2.3 The plumbing (`issuer.py:205-231`)

At credit issuance the issuer calls `next_cancellation_delay_ns(turn, phase)` (`:205`)
and stores the result in `Credit.cancel_after_ns` (`:230`; struct field
`structs.py:38-63`). The worker forwards it to the transport, which arms the timer at
send-complete. So `cancel_after_ns` is a **per-request scalar computed at issuance,
consumed at send-complete**.

### 2.4 Rust mapping

The companion request-rate spec already puts `cancel_after_ns` on the credit/turn.
Here it is produced by a `CancellationPolicy` (§4) and consumed by `aiperf-transport-http`:

- The transport already has a **real send-complete hook** (workspace commit
  `e960752c5`, "real send-complete hook") and cancellation support (transport spec's
  "cancellation-after-send" target). On send-complete, if `cancel_after_ns` is
  `Some(d)`, arm a **`Clock`-scheduled abort**: `clock.sleep(d)` racing the response
  future (`tokio::select!` on the `!Send` loop), and on timer-win abort the hyper
  request + emit `on_terminal` with a `RequestCancellationError` cause.
- Under `SimClock` the sleep is virtual, so the cancel fires at a deterministic
  integer-ns instant relative to the (virtual) send-complete time.

---

## 3. URL sampling (multi-URL load balancing)

### 3.1 The strategy surface (`url_samplers.py:14-81`)

`URLSelectionStrategyProtocol` (`:14-39`): constructed with `urls: list[str]`, exposes
`next_url_index() -> int` (index into `urls`, `:28-34`) and a `urls` property. **One
concrete impl exists:** `RoundRobinURLSampler` (`:42-81`) — `_index` starts at 0,
`next_url_index` returns the current index then advances `(_index+1) % len(urls)`
(`:73-81`); empty `urls` raises `ValueError` (`:63-64`). Default strategy is
`URLSelectionStrategy.ROUND_ROBIN` (`timing/config.py:81-82`, enum
`enums.py:391`).

### 3.2 The "advance on turn-0 only, then pin per-session" invariant

This is the earned-in-blood detail and lives **outside** the sampler, split across the
issuer and the worker:

- **Issuer** (`issuer.py:212-220`): advance the round-robin **only on the first turn**
  of a conversation (`is_first_turn = turn.turn_index == 0`); for later turns
  `url_index = None`. So one round-robin *tick* per conversation, not per request.
- **Worker/session** (`worker.py:490-501` + `session_manager.py:144-206`): on turn-0 the
  worker stores `credit.url_index` in the `UserSession` (`worker.py:496-501`,
  "so all turns hit the same backend"); on later turns it reads back `session.url_index`
  (`worker.py:744-748`). This makes URL selection **sticky per conversation**: a
  multi-turn session always hits the same backend, while different sessions spread
  round-robin across backends.
- **Transport** (`aiohttp_transport.py:206-223`): resolves the chosen index via
  `endpoint_info.get_url(request_info.url_index)` to the concrete base URL.

So "sticky per-session" is **not** a second sampler — it is round-robin + the turn-0
gate + the worker-session pin. Any future sticky/hash/weighted policy *would* be a new
sampler impl.

### 3.3 Rust mapping

- A small **`UrlSelector`** trait (§4) with the one `RoundRobin` impl. It lives on the
  single-loop issuer (`Rc<RefCell>`, `!Send`); no locking needed.
- The turn-0 gate stays in the **issuer** (it knows `turn_index`); `url_index: Option<u32>`
  rides on the turn/credit into the `RequestSink`. The sink resolves it to a base URL
  when building the `HttpRequest`.
- Sticky-per-session state belongs wherever conversation/session state lives (the
  companion spec's `ConversationSource` / worker-session equivalent): store the turn-0
  `url_index` on the session, reuse for continuation turns.

---

## 4. Mapping onto the crates — built vs designed

| Concern | Primitive | Crate | Status |
|---|---|---|---|
| Rate actuator (`set_rate`; Poisson/Gamma/Const/Burst) | `IntervalGenerator::set_rate` | `aiperf-timing` | **built** |
| Concurrency actuator (`set_limit`, debt-drain) | `SlotPool::set_limit` | `aiperf-timing` | **built** |
| Stop bounds / phase scope | `StopChecker` / `ConcurrencyManager` | `aiperf-timing` | **built** |
| All time (ramp elapsed + sleeps, cancel timer) | `Clock` (`now_ns`/`sleep`) | `aiperf-clock` | **built** |
| Transport send-complete hook + request abort | `aiperf-transport-http` client | `aiperf-transport-http` | **built** (send-complete hook `e960752c5`; abort-after-send wiring **designed**) |
| RNG for Poisson trajectory + Bernoulli draw | `aiperf-rng` (BLAKE3-derived) | `aiperf-rng` | **designed** (rng-derive spec) |
| **Ramp curve family** (`Linear`/`Exp`/`Poisson`, `next_step`/`value_at`) | **`RampStrategy`** trait | new / `aiperf-timing` | **designed** |
| **Ramp driver loop** (discrete/continuous, force-target, stop) | **`RampDriver`** + `RamperConfig` | new / `aiperf-timing` | **designed** |
| **Cancellation decision** (rate %, warmup-off, fixed delay) | **`CancellationPolicy`** trait | new / `aiperf-timing` | **designed** |
| **Cancel timer** (arm at send-complete, race response) | `Clock::sleep` + `select!` in sink | `aiperf-transport-http` | **designed** |
| **URL selection** (round-robin, turn-0 gate) | **`UrlSelector`** trait | new / `aiperf-timing` | **designed** |

### 4.1 The new trait seams (every extension point a trait)

- **`RampStrategy`** — object-safe:
  `fn next_step(&mut self, current: f64, elapsed_ns: u64) -> Option<(u64 /*delay_ns*/, f64)>`
  and `fn value_at(&self, elapsed_ns: u64) -> Option<f64>`, plus `start()`/`target()`.
  Impls: `LinearRamp` (`step_size`), `ExponentialRamp` (`exponent`, ease-in via
  `t^e` / `p^(1/e)`), `PoissonRamp` (pre-computed trajectory, `aiperf-rng`-seeded). New
  curves (log, s-curve, cosine) drop in as impls — never a `match RampType`.
- **`RampDriver`** — wraps a `RampStrategy` + a setter closure
  (`FnMut(f64)`; the actuator's `set_rate`/`set_limit`), runs the discrete-or-continuous
  loop on `Clock`, force-sets `target` on completion, freezes on cancel. One driver
  spawned per active ramp (session/prefill/rate).
- **`CancellationPolicy`** — `fn next_cancel_delay_ns(&mut self, phase: Phase) -> Option<u64>`.
  Impl `BernoulliFixedDelay` (rate% + constant delay + warmup-off, `aiperf-rng`-seeded).
  A future per-request-distributed delay is just another impl.
- **`UrlSelector`** — `fn next_index(&mut self) -> usize` + `fn len(&self) -> usize`.
  Impl `RoundRobin`. The turn-0 gate + session-pin stay in the issuer/session, not the
  trait. Future weighted/least-loaded/hash selectors are new impls.

---

## 5. Offline / online parity (determinism)

All three are `Clock`-driven, so `SimClock` makes them virtual-time and deterministic:

- **Ramp** — `elapsed_ns` = `clock.now_ns() - ramp_start_ns`; both discrete `sleep(delay)`
  and continuous `sleep(update_interval)` are `Clock::sleep`. Under `SimClock` a ramp
  advances in discrete-event steps (`advance_to`), so the exact `(instant, value)`
  sequence is reproducible. The `PoissonRamp` trajectory is drawn once from an
  `aiperf-rng` BLAKE3-derived stream → bit-reproducible for a given seed.
- **Cancellation** — the Bernoulli draw is a seeded `aiperf-rng` stream (order-independent
  per the rng-derive spec); the delay is a constant. The abort timer is `Clock::sleep`
  armed at (virtual) send-complete, so the cancel instant is deterministic under
  `SimClock`.
- **URL sampling** — pure integer round-robin, no clock, no RNG; identical online and
  offline.

**Parity is code-path + report-schema, not byte-identical metric values** (per the
port-exact ledger addendum): the same `RampDriver`/`CancellationPolicy`/`UrlSelector`
run under both clocks; only `{Clock, RequestSink}` are injected.

---

## 6. Build order (increments)

1. **`UrlSelector` + `RoundRobin`** — smallest, no clock/RNG. Add `url_index: Option<u32>`
   to the turn/credit; issuer advances on turn-0 only; sink resolves index → base URL;
   session pins the turn-0 index for continuation turns.
2. **`CancellationPolicy` + `BernoulliFixedDelay`** — needs `aiperf-rng`. Produce
   `cancel_after_ns` at issuance; store on the turn/credit (field already in the
   companion design).
3. **Cancel timer in the sink** — arm `Clock::sleep(cancel_after_ns)` at the transport
   send-complete hook, race the response with `select!`, emit terminal
   `RequestCancellationError` on timer-win. Validate against the in-repo mock (delay=0
   and delay>0 cases).
4. **`RampStrategy` (Linear) + `RampDriver`** — discrete mode first (concurrency ramp,
   `SlotPool::set_limit`), then continuous mode (rate ramp, `IntervalGenerator::set_rate`,
   proportional `start_rate`). All time via `Clock`.
5. **`ExponentialRamp`** — the ease-in inverse-curve pair; unit-test that
   `_time_to_value_progress` is the exact inverse of `_apply_curve`.
6. **`PoissonRamp`** — pre-computed `aiperf-rng` trajectory (normalize time to duration,
   pin final value to target); `value_at` = binary search.

1–3 deliver cancellation + multi-URL (independent of the credit-issuer core); 4–6 add
the ramp curve family once the issuer/actuator wiring from the companion spec lands.

---

## 7. Risks / open questions

- **Ramp ↔ debt-drain interaction.** A *down* concurrency ramp calls `set_limit` with a
  smaller value, which becomes **debt** absorbed by future `release()`
  (`concurrency.py:128-138`). A driver that steps down fast can pile debt; verify the
  ramp's step cadence vs. the release rate so the effective limit tracks the intended
  curve and doesn't lag arbitrarily. (The shipped rampers only ever ramp *up* from 1,
  `runner.py:476,496`; a general down-ramp is a new capability — flag it.)
- **Cancel-vs-complete race.** The abort timer and the response terminal can fire
  near-simultaneously. The terminal path must be **idempotent** — exactly one terminal
  record per request, whether it's `RequestCancellationError` (timer won) or a normal
  completion (response won). `select!` on the `!Send` loop makes this a clean either/or,
  but the observer/collector must reject a second terminal for the same request key.
- **Timer anchor = send-complete, not issuance.** Easy to get wrong: `cancel_after_ns`
  is relative to **T2** (`request_cancellation.py:62-81`), so it must be armed inside the
  sink at the send-complete hook, never at dispatch. Arming at issuance would cancel
  before/while sending and violate the "server sees the whole request" guarantee.
- **Sticky-URL per-session state.** The turn-0 gate (`issuer.py:215`) only works if the
  session stores the turn-0 `url_index` and continuation turns read it back
  (`worker.py:496-501,744-748`). If the Rust `ConversationSource`/session doesn't carry
  this field, multi-turn sessions would round-robin *within* a conversation and hit
  different backends — a correctness bug that tests on single-turn workloads won't catch.
- **Continuous-rate proportional start.** `start_rate = target*(update_interval/duration)`
  (`runner.py:512-514`) is deliberate to avoid ramping *up* a sub-1-QPS target; port the
  formula exactly rather than defaulting to 1 QPS.
- **Ramp under `SimClock`.** A continuous rate ramp sleeps `update_interval` (0.1s) and
  reads `value_at(elapsed)`; under `SimClock` this is fine, but the `IntervalGenerator`
  it drives must *also* be on the same virtual clock so the new rate takes effect on the
  next virtual `next_interval()`, not a stale wall-clock tick.
- **Exponent inverse exactness.** `ExponentialRamp` relies on `p^(1/e)` and `t^e` being
  exact inverses (`ramping.py:390-396`); FP round-trip error at the endpoints is why the
  driver force-sets `target` on completion — preserve that safety net.

---

## 8. One-line summary

Three ancillary knobs — a **`RampStrategy`/`RampDriver`** that walks rate
(`IntervalGenerator::set_rate`) or concurrency (`SlotPool::set_limit`) along a
Linear/Exponential-ease-in/Poisson curve over a fixed duration, a
**`CancellationPolicy`** that arms a fixed-delay client-disconnect timer on a Bernoulli
fraction of non-warmup requests (fired as a `Clock`-scheduled transport abort at
send-complete), and a **`UrlSelector`** round-robin that ticks once per conversation
(turn-0) and pins sticky per-session — every actuator already built in
`aiperf-timing`/`aiperf-transport-http`, every new piece a small trait, all time through
`Clock` so they run bit-deterministically under `SimClock`.

## Addendum — 2026-07-11: implemented in the native Rust workspace

This design is now built. This addendum supersedes the original `Status: design
(not built)`, the designed rows in §4, and the prospective build order in §6.

- `aiperf-timing::{RampStrategy, RampDriver}` is implemented in
  `rust/aiperf-timing/src/ramping.rs`, with `LinearRamp`,
  `ExponentialRamp`, and the precomputed, normalized `PoissonRamp`. Both driver
  modes use injected `Clock` time, natural completion force-applies the exact
  target, and abort freezes the last applied value. Poisson and cancellation
  streams use the canonical `aiperf-rng` namespaces.
- `CancellationPolicy` / `BernoulliFixedDelay` and `UrlSelector` /
  `RoundRobinUrlSelector` are implemented in `cancellation.rs` and
  `url_selection.rs`. Warmup returns before the RNG draw. The implementation
  uses signed nanoseconds (`Option<i64>`) because that is the native `Clock` and
  transport representation; validation prevents negative configured delays.
- `aiperf-transport-http` now shares a `SendCompletion` signal between `TimedBody`
  and the cancellation race. The deadline is anchored to the captured
  send-complete timestamp, not to issuance, connection acquisition, or the
  later wakeup time. A timer win drops the request future and produces HTTP 499
  `RequestCancellationError`; response is biased at an exact tie, so only one
  terminal result wins. Real-HTTP tests verify that the server received and
  parsed the complete JSON body before a zero-delay disconnect.
- `aiperf::scheduled::ScheduledRuntime` makes the cancellation decision for
  every issued turn, advances endpoint selection only on turn zero, retains the
  turn-zero selector result on the issued credit, and stores the effective
  endpoint in per-session state for continuations. `TransportSink` resolves the
  pinned index over a prebuilt ordered URL list. Deterministic three-turn tests
  and real two-endpoint HTTP tests pin both invariants.
- The ordinary online issuer wires linear session/prefill ramps to the live
  `SlotPool`s and the proportional-start 100ms continuous rate ramp to the live
  `IntervalGenerator`. User-centric mode additionally wires its owned session
  `SlotPool`; its request cadence is schedule-authored and it has no prefill
  pool, so rate and prefill ramps are rejected rather than accepted inertly.
  Fixed-schedule replay likewise rejects actuator ramps because its authored
  timestamps are the complete run plan. Both scheduled strategies still use
  cancellation and sticky multi-URL selection. The separate warmup/phase
  orchestrator remains the companion phase-runner spec's unbuilt scope; the
  shared policy already accepts and tests `Phase::Warmup` for that future caller.
- The online CLI exposes `--concurrency-ramp-duration`,
  `--prefill-concurrency-ramp-duration`, `--request-rate-ramp-duration`,
  `--request-cancellation-rate`, and `--request-cancellation-delay`.
  Comma-separated positional base URLs select the ordered endpoint list. Graph
  mode rejects these online issuer flags explicitly; the transport-neutral
  traits and `SimClock` tests remain reusable by its future arrival/slot policy
  consumer.

Tests cover configuration bounds, both ramp directions, exact exponential
inverse math, deterministic normalized Poisson trajectories, force-target and
freeze-on-stop behavior, live actuator wiring under `SimClock`, warmup RNG
non-consumption, Bernoulli reproducibility, round-robin wraparound, turn-zero
session pinning, real endpoint resolution, positive and zero post-send delays,
complete-body delivery, cancellation classification, and CLI validation.

## Addendum — 2026-07-12: offline runtime consumption

The shared policies are now consumed by the in-process Dynamo backend as well
as HTTP. Backend-neutral run functions receive `{Clock, HttpRequestDispatcher}`;
offline composition supplies `SimClock + DynosimSink` and merges ramp,
cancellation, issuer, and engine deadlines in the same DES pump.

- paced offline workloads support session- and prefill-concurrency ramps;
- continuation-priority request-rate workloads support all three Clock-native
  rate curves;
- user-centric workloads support their session-concurrency ramp;
- cancellation calls the steppable engine's real terminal operation in single,
  aggregate, and disaggregate topologies, including terminal cleanup and report
  classification.

Fixed authored schedules still reject ramps because their timestamps are the
workload, and Graph-IR still owns no arrival/slot actuator. Multi-URL selection
is intentionally invalid for the one in-process endpoint; that is a topology
constraint, not an offline no-op.
