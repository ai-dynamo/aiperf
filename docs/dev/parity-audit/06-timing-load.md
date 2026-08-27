<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Load generation and timing parity audit

**Python baseline:** `/mnt/4tb/aiperf-parity-py-main/src/aiperf/` at git rev
`bc359bf8fd` (`origin/main`). All Python `path:line` citations below are against
that tree, not the local feature branch. Rust citations are against
`rust/` in this working tree.

**Baseline correction applied.** An earlier revision of this report cited the
local feature branch (4345 commits ahead, 132 modified Python files). Two of that
revision's Python claims were branch artifacts; the affected finding is recorded
in [Withdrawn after baseline correction](#withdrawn-after-baseline-correction).

## Summary

The largest silent divergence is the **warmup phase**: upstream Python derives
warmup's rate, arrival pattern, gamma smoothness, prefill concurrency, and all
three ramps from the corresponding profiling flags whenever no `--warmup-*`
override was given, while Rust reads only the `--warmup-*` flags and inherits
concurrency alone. `--request-rate 10 --warmup-request-count 50` therefore warms
up open-loop at 10 rps upstream and closed-loop at concurrency 1 in Rust. The
same converter also disagrees about what *triggers* a warmup: upstream requires a
stop condition and ignores stray `--warmup-*` flags, while Rust builds an
**unbounded** warmup phase from any one of nine. Second, on Rust's default
execution path (`workers` = core count, `dispatch=global`) the Poisson/Gamma
renewal process is replaced by a fixed rate grid plus mean-zero jitter, and that
same path performs unbounded post-stall catch-up bursts — notable because
baseline Python and Rust otherwise implement *identical* bounded re-anchoring,
down to a shared `AIPERF_TIMING_MAX_CATCHUP_SECONDS` env var and 0.01 s default,
and upstream's bound exists expressly to prevent "a burst storm". Third, a
user-centric run that authors only a rate and a user count is capped at 10
requests upstream and is **unbounded** in Rust. Beyond that, several authored
inputs are accepted and silently dropped in Rust's user-centric projection
(`--request-cancellation-rate`, all three ramps, `--prefill-concurrency`),
`--num-users` inflates to the worker count when it is below it, and
`--fixed-schedule-end-offset` alone flips the replay anchor. Baseline Python
turns out to be much closer to Rust on pacing mechanics than the branch
suggested: both sleep on a `CLOCK_MONOTONIC` timerfd and share the bounded
catch-up contract.

## Findings

### 1. Warmup phase does not inherit the profiling rate, arrival pattern, smoothness, prefill concurrency, or ramps

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **STILL VALID** at baseline
(adjacent to KNOWN P1.11, which covers phase-list collapse, not warmup field
inheritance)

The branch added 86 lines to this file, but the diff is entirely an orthogonal
`--agentic-cache-warmup-duration` path; the inheritance helpers are byte-identical
to baseline at the same line numbers.

**Python evidence** — `_converter_warmup.py:37-48` picks the profiling flag when
the `warmup_*` variant is absent from `model_fields_set`:

```37:48:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_warmup.py
def _warmup_pattern_type(w: dict[str, Any], cli: CLIConfig, s: set[str]) -> None:
    warmup_rate = (
        cli.warmup_request_rate if "warmup_request_rate" in s else cli.request_rate
    )
    warmup_pattern = (
        cli.warmup_arrival_pattern
        if "warmup_arrival_pattern" in s
        else cli.arrival_pattern
    )
    warmup_concurrency = (
        cli.warmup_concurrency if "warmup_concurrency" in s else cli.concurrency
    )
```

and `:56-65` turns an inherited rate into a rate-shaped warmup phase carrying the
profiling smoothness:

```56:65:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_warmup.py
    if warmup_rate is not None:
        w["rate"] = warmup_rate
        match warmup_pattern:
            case ArrivalPattern.GAMMA:
                w["type"] = PhaseType.GAMMA
                w["smoothness"] = cli.arrival_smoothness
            case ArrivalPattern.CONSTANT:
                w["type"] = PhaseType.CONSTANT
            case _:
                w["type"] = PhaseType.POISSON
```

The same fallback covers all three ramps (`_converter_warmup.py:79-92`, the
`_pick(warmup_field, fallback_field)` helper) and prefill concurrency
(`_converter_warmup.py:147-150`, `elif "prefill_concurrency" in s`).

**Rust evidence** — `rust/cli/src/load.rs:346-358` populates `Warmup` only from
`warmup_*` flags:

```346:358:rust/cli/src/load.rs
        Some(Warmup {
            concurrency: flags.warmup_concurrency,
            rate: flags.warmup_request_rate,
            requests: flags.warmup_request_count,
            sessions: flags.num_warmup_sessions,
            prefill_concurrency: flags.warmup_prefill_concurrency,
            rate_mode: flags.warmup_arrival_pattern.clone(),
            concurrency_ramp: flags.warmup_concurrency_ramp_duration,
            rate_ramp: flags.warmup_request_rate_ramp_duration,
            prefill_ramp: flags.warmup_prefill_concurrency_ramp_duration,
            duration: flags.warmup_duration,
            grace_period: flags.warmup_grace_period,
        })
```

and `rust/runtime/src/config/resolve.rs:1466-1484` inherits **only** concurrency,
passing a literal `None` where `build_phase` takes smoothness:

```1466:1484:rust/runtime/src/config/resolve.rs
            if let Some(warmup) = inputs.warmup.as_ref() {
                let concurrency = warmup.concurrency.or(inputs.concurrency);
                let mut wp = build_phase(
                    "warmup",
                    true,
                    concurrency.unwrap_or(1),
                    warmup.rate,
                    warmup.rate_mode.as_deref(),
                    None,
                    concurrency,
                    warmup.requests,
                    ...
                );
                wp.common.prefill_concurrency = warmup.prefill_concurrency;
                wp.common.concurrency_ramp = warmup.concurrency_ramp.map(linear_ramp);
                wp.common.rate_ramp = warmup.rate_ramp.map(linear_ramp);
```

With `rate = None`, `build_phase` selects
`PhaseKind::Concurrency { concurrency: concurrency.unwrap_or(default) }`
(`rust/runtime/src/config/resolve.rs:2083-2087`).

**Observable user impact:** `aiperf profile --request-rate 10 --request-count 500
--warmup-request-count 50` warms up as 50 requests paced at 10 rps (open loop,
~5 s) upstream and as 50 requests at concurrency **1** (closed loop, serialized)
in Rust. Gamma smoothness never reaches a Rust warmup phase at all. A
`--prefill-concurrency N` run applies the prefill cap to warmup upstream and not
in Rust. Warmup wall time, warmup concurrency, and cache-priming pressure all
differ — and because warmup shapes the KV cache that the profiling phase
measures, profiling numbers move with it.

**Confidence:** High — all sites read directly on both sides.

### 2. Default `global` dispatch replaces the Poisson/Gamma renewal process with a jittered fixed grid

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **STILL VALID** at baseline
(`intervals.py` is byte-identical to baseline; `request_rate.py` diverged but the
cited accumulation line is unchanged in substance)

**Python evidence** — inter-arrival times are drawn from the distribution, so
successive gaps are i.i.d. exponential
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/intervals.py:121-123`):

```121:123:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/intervals.py
    def next_interval(self) -> float:
        """Generate exponentially distributed inter-arrival time."""
        return self._rng.expovariate(self._request_rate)
```

and the loop accumulates them onto an absolute target
(`request_rate.py:255`, `:282-284`):

```282:284:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/request_rate.py
                # Schedule next interval BEFORE issuing credit. This way, variable
                # credit issuance latency doesn't affect the timing of the next interval.
                next_target_perf += self._rate_generator.next_interval()
```

Upstream runs one `RequestRateStrategy` credit issuer per phase regardless of
`--workers`, so this is always the arrival process on the wire.

**Rust evidence** — `rust/runtime/src/request_rate.rs:619-646`: with a
`GlobalRateGate` attached, the target becomes `start + interval + k·interval +
(draw − interval)`:

```619:646:rust/runtime/src/request_rate.rs
                let jitter_ns = self
                    .intervals
                    .borrow_mut()
                    .next_interval_ns()
                    .saturating_sub(gate.interval_ns());
                ...
                runtime
                    .start_ns()
                    .saturating_add(gate.interval_ns())
                    .saturating_add(base_offset_ns)
                    .saturating_add(jitter_ns)
```

`base_offset_ns` is `index · interval_ns`
(`rust/runtime/src/timing/rate_gate.rs:108-116`), and the module doc states the
consequence outright (`rust/runtime/src/timing/rate_gate.rs:15-24`): "**not** a
reproduction of Poisson/Gamma arrival-process statistics … the resulting
inter-arrival times are not exponentially distributed".

The gate is attached whenever `dispatch == Global` and `workers > 1`
(`rust/runtime/src/engine/execute/compose_sidecars.rs:612-668`), and both
defaults land there: `runtime.workers` defaults to the machine's core count
(`rust/runtime/src/engine/protocol_v2.rs:243-247`) and `runtime.dispatch`
defaults to `Global` for a single-process run
(`rust/runtime/src/engine/protocol_v2.rs:265-274`). On a one-core host or with an
explicit `--workers 1`, Rust uses the true renewal process instead — so the
arrival distribution depends on the host's core count.

**Observable user impact:** With `--request-rate R --arrival-pattern poisson`
(Poisson is the default once a rate is set), arrival `k` lands at
`start + k/R + Exp_k` instead of `start + Σ Exp_i`. Successive gaps become
`1/R + (Exp_{k+1} − Exp_k)`: same mean rate, materially lower burstiness, no
queue build-up from a long exponential run. Every queueing-sensitive metric
(TTFT tail, p99 latency, observed peak concurrency) shifts, and the same command
yields different burstiness on a 1-core box than on a 64-core box.

**Confidence:** High for the mechanism and the default resolution. The magnitude
of the metric shift is workload-dependent and unmeasured here.

### 3. Default `global` dispatch bursts to catch up after a stall, bypassing the shared bounded-catch-up contract

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **CHANGED** — narrowed to the
gate path only. The earlier revision's claim that Python re-anchors with *zero*
catch-up was a branch artifact; see
[Withdrawn after baseline correction](#withdrawn-after-baseline-correction).

Baseline Python and Rust implement the *same* bounded re-anchor, with the same
env var, default, and range. The divergence is that Rust's default
global-dispatch path opts out of it entirely.

**Python evidence** — `request_rate.py:265-268` re-anchors only past a bounded
window:

```265:268:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/request_rate.py
                # Behind schedule: re-anchor to now only past the bounded catch-up
                # window, so sub-ms oversleeps stay on the original schedule.
                if next_target_perf < now - Environment.TIMING.MAX_CATCHUP_SECONDS:
                    next_target_perf = now
```

The window's stated purpose is exactly this failure mode
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/environment.py:1249-1257`):

```1249:1257:/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/environment.py
    MAX_CATCHUP_SECONDS: float = Field(
        ge=0.0,
        le=10.0,
        default=0.01,
        description="Maximum schedule backlog in seconds the rate loop is allowed to catch up on "
        ...
        "a genuine multi-second stall still re-anchors instead of firing a burst storm.",
    )
```

**Rust evidence** — the non-gate path matches this exactly, including the env var
name, the 0.01 default, and the `0..=10` range
(`rust/runtime/src/request_rate.rs:40-56`, `:648-649`;
`rust/runtime/src/timing/arrival.rs:88-95`):

```88:95:rust/runtime/src/timing/arrival.rs
pub(crate) fn bounded_reanchor_target(target_ns: i64, now_ns: i64, max_catchup_ns: i64) -> i64 {
    debug_assert!(max_catchup_ns >= 0);
    if target_ns < now_ns.saturating_sub(max_catchup_ns) {
        now_ns
    } else {
        target_ns
    }
}
```

The **gate** path deliberately skips it
(`rust/runtime/src/request_rate.rs:616-618`, `:652-662`):

```616:618:rust/runtime/src/request_rate.rs
                // for jittered phases is `global-hop`'s job. No per-thread
                // re-anchor: a claimed slot already in the past pages through
                // via the `scheduled_ns <= now` yield path below.
```

```652:662:rust/runtime/src/request_rate.rs
            if scheduled_ns > now_ns {
                if !runtime.wait_until_or_stop(scheduled_ns).await {
                    break;
                }
            } else {
                ...
                tokio::task::yield_now().await;
            }
```

Slot indices advance one per loop iteration, so after a `D`-second stall the next
`D/interval` claimed slots are all in the past and are issued back-to-back at
loop speed rather than at the configured rate.

**Observable user impact:** On a rate-controlled run that hits a transient stall
(GC pause, endpoint hiccup, slow tokenizer batch), upstream re-anchors once the
backlog exceeds 10 ms and never fires a burst; Rust's default path issues the
whole backlog as fast as the loop can turn. Instantaneous offered load right
after any stall differs by up to the full stall backlog, and on duration-bounded
phases the delivered totals differ. Setting
`AIPERF_TIMING_MAX_CATCHUP_SECONDS` — which both engines read — has no effect on
Rust's default path, since the gate branch never consults it.

**Confidence:** High.

### 4. Any stray `--warmup-*` flag creates an unbounded warmup phase in Rust; upstream ignores it or errors

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **STILL VALID** at baseline (the
trigger gate is byte-identical to baseline; only its line number moved)

**Python evidence** — `_converter_warmup.py:127-141`: a warmup phase exists only
when a *stop condition* was authored, and the one secondary flag that could
otherwise be swallowed is rejected explicitly.

```127:141:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_warmup.py
    if not ({"warmup_request_count", "warmup_num_sessions", "warmup_duration"} & s):
        # No warmup trigger -> no warmup phase. Refuse to silently drop
        # secondary warmup-only flags the user supplied — except under a
        # --scenario, where the auto-synthesized agentic warmup consumes
        # --warmup-grace-period as its barrier grace (v1 parity; see
        # _apply_agentic_replay_fields in _converter_profiling).
        if cli.warmup_grace_period is not None and cli.scenario is None:
            raise ValueError(
                "--warmup-grace-period was supplied without any warmup "
                "trigger; warmup runs only when --warmup-request-count, "
                "--warmup-num-sessions, or --warmup-duration is set. Pass "
                "--warmup-duration to enable a duration-bounded warmup with "
                "the grace period, or drop --warmup-grace-period."
            )
        return None
```

The module docstring makes the intent explicit
(`_converter_warmup.py:112-115`): "Other warmup_* fields without a trigger are
intentionally ignored." A stop condition is separately mandatory for every
non-fixed-schedule phase
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:394`,
`_stop_condition_required: ClassVar[bool] = True`).

**Rust evidence** — `rust/cli/src/load.rs:334-345` treats nine flags as triggers,
six of which carry no stop bound:

```334:345:rust/cli/src/load.rs
    let warmup = if flags.warmup_request_count.is_none()
        && flags.warmup_concurrency.is_none()
        && flags.warmup_request_rate.is_none()
        && flags.num_warmup_sessions.is_none()
        && flags.warmup_prefill_concurrency.is_none()
        && flags.warmup_concurrency_ramp_duration.is_none()
        && flags.warmup_request_rate_ramp_duration.is_none()
        && flags.warmup_duration.is_none()
        && flags.warmup_grace_period.is_none()
    {
        None
    } else {
```

The resulting phase carries `requests: None, sessions: None, duration: None`
(`rust/runtime/src/config/resolve.rs:1468-1480`), and
`rust/runtime/src/timing/stop.rs:171-190` then installs only the `Lifecycle`
condition, whose sole predicate is
`!state.cancelled && !state.sending_complete`
(`rust/runtime/src/timing/stop.rs:92-94`).

**Observable user impact:** `aiperf profile --concurrency 8 --request-count 100
--warmup-concurrency 4` runs no warmup upstream and, in Rust, prepends a warmup
phase with no request, session, or duration bound. `--warmup-grace-period 5`
alone is a hard config error upstream and a warmup trigger in Rust. Warmup
records are excluded from results, so the symptom is a run that appears to hang
before profiling rather than a wrong number.

**Confidence:** High on the projection and the absence of a stop condition.
Whether the phase self-terminates depends on conversation-source exhaustion,
which is dataset-dependent — see *Unverified / needs runtime check*.

### 5. `--user-centric-rate` without `--num-users` is silently ignored and the phase falls back to a different shape

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **STILL VALID** at baseline

**Python evidence** — `--user-centric-rate` alone still selects the user-centric
phase type
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py:201-202`,
`:221-222`):

```201:202:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py
    user_centric_needed = "users" in search_dims
    user_centric_selected = cli.user_centric_rate is not None or user_centric_needed
```

`users` is a **required** field on that phase, so the missing flag surfaces as a
validation error rather than a shape change
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:622-629`):

```622:629:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py
    users: Annotated[
        int,
        Field(
            ge=1,
            description="Number of simulated concurrent users (must be >= 1). "
            "Requests distributed across users to achieve global rate.",
        ),
    ]
```

The strategy enforces the pairing again at runtime
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/user_centric_rate.py:147-153`).

**Rust evidence** — `rust/cli/src/load.rs:224-227`:

```224:227:rust/cli/src/load.rs
    let user_centric_cli = match (flags.user_centric_rate, flags.num_users) {
        (Some(rate), Some(users)) => Some((rate, users)),
        _ => None,
    };
```

`None` leaves `inputs.user_centric` unset, so `resolve.rs:1365` falls through to
the ordinary rate/concurrency construction at `resolve.rs:1423-1448`. Because
`inputs.request_rate` is a separate field from `user_centric_rate`, the rate is
dropped too, leaving `PhaseKind::Concurrency { concurrency: 1 }`. The YAML
frontend *does* validate the pair (`rust/cli/src/yaml.rs:2206-2208`,
`"user_centric phase requires rate and users"`); only the CLI path is silent.

**Observable user impact:** `aiperf profile --user-centric-rate 5` errors
upstream and, in Rust, runs a closed-loop concurrency-1 phase of 10 requests
(`DEFAULT_REQUEST_COUNT` applies here precisely because `user_centric` is
`None`). Entirely different experiment, no diagnostic.

**Confidence:** High.

### 6. A user-centric run authoring only rate and users is bounded at 10 requests upstream and unbounded in Rust

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW — surfaced by the baseline
correction. The branch narrowed the default-10 rule with exclusions that upstream
does not have.

**Python evidence** — the unbounded-run default applies to every phase type
except fixed-schedule, user-centric included
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py:552-560`):

```552:560:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py
    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        # Why: when no bound is given for an unbounded run, default to
        # 10 requests so the run terminates in a reasonable time.
        # Deliberate override of the PhaseConfig default (which would
        # leave it unbounded).
        prof.setdefault("requests", 10)
```

**Rust evidence** — `rust/runtime/src/config/resolve.rs:1353-1359` excludes
`user_centric` from the default:

```1353:1359:rust/runtime/src/config/resolve.rs
    let effective_requests = inputs.request_count.or_else(|| {
        (inputs.benchmark_duration.is_none()
            && inputs.fixed_schedule.is_none()
            && inputs.user_centric.is_none()
            && inputs.sessions.is_none())
        .then_some(DEFAULT_REQUEST_COUNT)
    });
```

and the user-centric arm reads the raw flag rather than `effective_requests`
(`rust/runtime/src/config/resolve.rs:1372`):

```1372:1374:rust/runtime/src/config/resolve.rs
                requests: inputs.request_count,
                sessions: inputs.sessions,
                duration: inputs.benchmark_duration,
```

With all three `None`, `rust/runtime/src/timing/stop.rs:171-190` installs only
the `Lifecycle` condition.

**Observable user impact:** `aiperf profile --user-centric-rate 5 --num-users 5`
sends exactly 10 requests and exits upstream; in Rust it runs with no request,
session, or duration bound. A user probing user-centric mode for the first time
gets a 2-second run in one engine and an open-ended one in the other. (At
`--num-users 20` upstream instead errors, because the same phase validator
requires `requests >= users` —
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:643-647` — so the
silent case is `num_users <= 10`.)

**Confidence:** High on both projections. The Rust run's actual termination
depends on dataset exhaustion, same caveat as finding 4.

### 7. Rust's user-centric phase silently drops cancellation, all three ramps, and prefill concurrency

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW for cancellation and ramps;
KNOWN(still-true) for prefill (P1.13). **CHANGED** — `rate_series` removed from
the list: upstream rejects it for user-centric phases too
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:634-635`), so Rust's
`None` matches.

**Python evidence** — `UserCentricPhase` extends `RatePhaseConfig` →
`BasePhaseConfig` (`phases.py:610`, `:534`, `:68`), inheriting
`concurrency_ramp` (`phases.py:179`), `prefill_concurrency` (`:188`),
`prefill_ramp` (`:198`), `cancellation` (`:221`), and `rate_ramp` (`:546`). The
converter writes `cancellation` onto the profiling phase unconditionally, before
any phase-type gate
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py:561-566`):

```561:566:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py
    delay_set = "request_cancellation_delay" in cli.model_fields_set
    if cli.request_cancellation_rate:
        cancel: dict[str, Any] = {"rate": cli.request_cancellation_rate}
        if delay_set:
            cancel["delay"] = cli.request_cancellation_delay
        prof["cancellation"] = cancel
```

**Rust evidence** — `rust/runtime/src/config/resolve.rs:1376-1386` hardcodes each
of those to `None` in the user-centric arm:

```1376:1386:rust/runtime/src/config/resolve.rs
                prefill_concurrency: None,
                grace_period: profiling_grace_period,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
                agentic_warmup_grace_period: inputs.agentic_warmup_grace_period,
                failed_request_threshold: inputs.failed_request_threshold,
                adaptive_scale: None,
                rate_series: None,
```

The non-user-centric arm at `resolve.rs:1449-1459` carries all of them, so this
is specific to the user-centric projection.

**Observable user impact:** `--user-centric-rate 5 --num-users 20
--request-cancellation-rate 10` cancels ~10% of requests upstream and zero in
Rust; the run reports no cancellations and no cancellation-induced errors.
`--request-rate-ramp-duration`, `--concurrency-ramp-duration`, and
`--prefill-concurrency` are likewise accepted and inert. No warning.

**Confidence:** High.

### 8. `--num-users` is inflated to the worker count when it is smaller than the worker count

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW (P1.14 describes the same
floor-to-one mechanism for `concurrency`, which no longer applies on the default
path — see *Checked and consistent*)

**Python evidence** — `num_users` is one global value consumed by the single
credit issuer, and the turn gap is derived from it directly
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/user_centric_rate.py:261-263`):

```261:263:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/user_centric_rate.py
    def _recompute_turn_gap(self, num_users: int) -> None:
        # num_users firing once per turn_gap gives: qps = num_users / turn_gap.
        self._turn_gap = num_users / self._request_rate
```

`setup_phase` seeds exactly `num_users` virtual-history users
(`user_centric_rate.py:204-209`).

**Rust evidence** — `rust/runtime/src/engine/sharded_scheduled.rs:148` and
`:228-233`:

```148:148:rust/runtime/src/engine/sharded_scheduled.rs
    let owned_cap = |value: usize| owned_positions(value as u64, t, workers).max(1) as usize;
```

```228:233:rust/runtime/src/engine/sharded_scheduled.rs
            *rate = scaled_rate(*rate);
            *users = owned_cap(*users);
            if let Some(cap) = concurrency {
                *cap = owned_cap(*cap);
            }
```

`users` is sliced with the floored `owned_cap` in **every** dispatch mode — the
`global_admits_concurrency_and_rate` guard that exempts `concurrency` and `rate`
under `Global` is not applied to it, as the function doc states
(`sharded_scheduled.rs:131-136`). With `workers` defaulting to the core count
(`rust/runtime/src/engine/protocol_v2.rs:243-247`), any `--num-users` below the
core count is rounded up per thread.

**Observable user impact:** `--user-centric-rate 8 --num-users 4` on a 16-core
host simulates 16 concurrent users in Rust (16 threads × 1 each) and 4 upstream.
Aggregate rate stays at 8 rps because `rate` is divided by `workers`, but each
thread computes `turn_gap = 1 / (8/16) = 2 s` instead of `4 / 8 = 0.5 s`, so
per-user think time, session lifetime mix, and the number of distinct in-flight
conversations all change. When `num_users >= workers` the split tiles exactly and
the totals agree.

**Confidence:** High.

### 9. `--fixed-schedule-end-offset` alone flips auto-offset off in Rust, changing the replay anchor

**Severity:** P1 &nbsp;·&nbsp; **Status:** NEW, **STILL VALID** at baseline

**Python evidence** — `auto_offset` defaults to `True`
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:666-673`) and the
converter forces it off only for `start_offset`
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py:803-804`):

```803:804:/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/flags/_converter_profiling.py
    if prof["type"] == PhaseType.FIXED_SCHEDULE and "start_offset" in prof:
        prof.setdefault("auto_offset", False)
```

`_FIXED_SCHEDULE_ONLY_ROUTES` copies the CLI value only when the flag was
explicitly set (`_converter_profiling.py:41-44`), so an unset
`--fixed-schedule-auto-offset` leaves the model default `True`. The zero is then
the first entry's timestamp
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/fixed_schedule.py:112-117`):

```112:117:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/fixed_schedule.py
        if self._config.auto_offset_timestamps:
            self._schedule_zero_ms = self._absolute_schedule[0].timestamp_ms
        elif self._config.fixed_schedule_start_offset is not None:
            self._schedule_zero_ms = float(self._config.fixed_schedule_start_offset)
        else:
            self._schedule_zero_ms = 0.0
```

**Rust evidence** — `rust/cli/src/load.rs:307-311` lets *either* offset flip the
default:

```307:311:rust/cli/src/load.rs
        let default_auto = flags.fixed_schedule_start_offset.is_none()
            && flags.fixed_schedule_end_offset.is_none();
        (
            Some(flags.fixed_schedule_auto_offset.unwrap_or(default_auto)),
            Some(count),
        )
```

The YAML frontend applies the same rule (`rust/cli/src/yaml.rs:2216-2220`,
`:2946-2948`). With `auto_offset == false` and no start offset,
`schedule_zero_ms` becomes `0.0` (`rust/runtime/src/fixed_schedule.rs:115-119`)
and each target becomes `anchor + timestamp_ms`
(`rust/runtime/src/fixed_schedule.rs:126-133`).

**Observable user impact:** `--fixed-schedule --input-file trace.jsonl
--fixed-schedule-end-offset 60000` replays the trace's first 60 s starting
immediately upstream. In Rust the same command anchors at trace time 0, so a
trace of absolute epoch milliseconds schedules every request decades out (the
phase issues nothing), and a trace whose first timestamp is 30 000 ms idles 30 s
before the first request. No error either way. The converse combination is also
asymmetric: upstream *rejects* `auto_offset=True` together with `start_offset`
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/phases.py:693-695`) while Rust
accepts an explicit `--fixed-schedule-auto-offset` alongside
`--fixed-schedule-start-offset`.

**Confidence:** High on the projection and the anchor arithmetic. The "issues
nothing" consequence for epoch-millisecond traces follows from `timestamp_to_ns`
but was not executed.

### 10. Three `AIPERF_TIMING_*` environment variables have no Rust reader

**Severity:** P2 &nbsp;·&nbsp; **Status:** NEW — surfaced by baseline's
`high_res_timer.py`. An instance of the KNOWN P1.6 class ("environment-variable
contracts are split"), but none of these three appear in that entry's examples.

**Python evidence** — baseline exposes four timing settings under the
`AIPERF_TIMING_` prefix
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/environment.py:1225-1258`):
`CANCEL_DRAIN_TIMEOUT` (default 10.0, `:1229`), `RATE_RAMP_UPDATE_INTERVAL`
(default 0.1, `:1235`), `HIGH_RES_TIMER` (default `True`, `:1241`), and
`MAX_CATCHUP_SECONDS` (default 0.01, `:1249`). `HIGH_RES_TIMER` gates pacer
selection at `request_rate.py:139-140`:

```139:140:/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/request_rate.py
        if not Environment.TIMING.HIGH_RES_TIMER:
            return None
```

**Rust evidence** — the only `AIPERF_TIMING_*` name anywhere in `rust/` is
`MAX_CATCHUP_SECONDS` (`rust/runtime/src/request_rate.rs:40`); a repository-wide
scan of `rust/` returns no other. The drain timeout is a hardcoded constant with
no environment read (`rust/runtime/src/timing/phase/runner.rs:27`):

```27:27:rust/runtime/src/timing/phase/runner.rs
const DEFAULT_CANCEL_DRAIN_TIMEOUT_NS: i64 = 10_000_000_000;
```

**Observable user impact:** `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=60` extends the
post-cancellation drain wait upstream and is silently inert in Rust, which forces
phase completion at 10 s and releases stuck slots — changing which late responses
land inside the phase. `AIPERF_TIMING_HIGH_RES_TIMER=false`, an upstream escape
hatch for environments where timerfd/thread pacing misbehaves, has no Rust
equivalent. Defaults agree in all three cases, so only users who set these are
affected.

**Confidence:** High for the absence of the readers and for the defaults matching.

## Withdrawn after baseline correction

### W1. "Python re-anchors the rate schedule to now with zero catch-up"

Claimed in the earlier revision as the `workers == 1` / `--dispatch sharded` half
of finding 3, citing the branch's
`src/aiperf/timing/strategies/request_rate.py:152-155`:

```python
            # Behind schedule: reset to now instead of sending a burst to catch up.
            # This sacrifices inter-arrival distribution accuracy for stable throughput.
            if next_target_perf < now:
                next_target_perf = now
```

**Branch artifact.** Upstream re-anchors only past a bounded window
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/strategies/request_rate.py:267`)
and Rust implements the identical rule
(`rust/runtime/src/timing/arrival.rs:88-95`) — same predicate
(`target < now − window`), same re-anchor value (`now`), same environment
variable `AIPERF_TIMING_MAX_CATCHUP_SECONDS`, same 0.01 s default, same `0..=10`
range (`/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/environment.py:1249-1252`
vs `rust/runtime/src/request_rate.rs:40-56`). This is deliberate, exact parity;
the local branch removed it. Withdrawn in full. The gate-path divergence remains
as finding 3.

A related suspicion — that upstream Python paces on coarse ~1 ms event-loop
timers while Rust uses a `timerfd` — is also withdrawn. Upstream's
`high_res_timer.py` (deleted on the branch, so never examined in the earlier
revision) sleeps on an absolute-deadline `CLOCK_MONOTONIC` timerfd awaited
through the event loop's fd reader
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/timing/high_res_timer.py:3-15`,
`:75`, `:107-109`), with a `clock_nanosleep`-backed `ThreadPacer` fallback for
non-Linux (`:125-134`). Rust does the same thing
(`rust/runtime/src/clock/real_clock.rs:116-143`, `libc::timerfd_create(
libc::CLOCK_MONOTONIC, …)`). Pacing resolution is at parity; the residual
difference is only the missing `AIPERF_TIMING_HIGH_RES_TIMER` toggle, recorded as
finding 10.

## Checked and consistent

All Python paths in this section are under
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/`.

- **Interval generators.** `Constant` = `1/rate`, `Poisson` =
  `expovariate(rate)`, `Gamma` = `gammavariate(shape=smoothness,
  scale=1/(rate·smoothness))`, `ConcurrencyBurst` = 0, smoothness defaulting to
  1.0, on both sides (`timing/intervals.py:95-233` vs
  `rust/runtime/src/timing/intervals.rs:51-204`). `intervals.py` is byte-identical
  to baseline. Rust quantizes to integer nanoseconds; Python keeps float seconds.
- **Rate-loop pacing resolution.** Both sleep to absolute deadlines on a
  `CLOCK_MONOTONIC` timerfd on Linux with a portable fallback elsewhere
  (`timing/high_res_timer.py:51-122` and `:125-234` vs
  `rust/runtime/src/clock/real_clock.rs:94-180`).
- **Bounded catch-up on the non-gate path.** Identical predicate, re-anchor
  value, env var, default, and range — see W1.
- **First arrival.** Both place arrival 0 one drawn interval after phase start,
  not at `t = 0` (`timing/strategies/request_rate.py:255` vs
  `rust/runtime/src/request_rate.rs:593-595`, `FirstArrival::AfterInterval` at
  `rust/runtime/src/timing/arrival.rs:36-39`).
- **Continuation-turn priority.** Both drain a queued continuation turn before
  starting a new session, and both spend a rate tick on it
  (`timing/strategies/request_rate.py:286-293` vs
  `rust/runtime/src/request_rate.rs:672-681`).
- **Grace / cancel / drain / force escalation.** Same four-stage ladder
  (`timing/phase/runner.py:1093-1097` vs
  `rust/runtime/src/timing/phase/runner.rs:616-640`), same 10 s drain default
  (`common/environment.py:1229-1233` and `timing/phase/runner.py:1172` vs
  `rust/runtime/src/timing/phase/runner.rs:27`), and the same deadline basis
  (`phase start + duration + grace`, not send-complete + grace;
  `rust/runtime/src/timing/phase/lifecycle.rs:171-186`). Only the env override is
  missing in Rust (finding 10).
- **Grace-period defaults.** Warmup waits indefinitely and profiling gets 30 s
  when — and only when — a duration is set (`timing/config.py:541-543` and
  `config/flags/cli_config.py:2113-2125` (`= 30.0`) vs
  `rust/runtime/src/timing/phase/config.rs:86-89` and
  `rust/runtime/src/config/resolve.rs:1360-1364`).
- **`request_count` vs `duration`.** Independent stop conditions that AND-combine
  (first to fire wins); neither overrides the other. Both suppress the
  unbounded-run default for fixed-schedule phases and for explicitly
  session-bounded runs (`config/flags/_converter_profiling.py:552-560` vs
  `rust/runtime/src/config/resolve.rs:1353-1359`). The user-centric exclusion is
  Rust-only — finding 6. Warmup counts are separate from the profiling count on
  both sides.
- **Request timeout.** 21 600 s (6 h) on both (`config/endpoint.py:58`,
  `TIMEOUT = 6 * 60 * 60` vs
  `rust/runtime/src/config/model/endpoint.rs:31-33`), and a timeout is recorded
  as a failed request in both.
- **Post-send cancellation policy.** Same units (`rate` = percent 0–100, `delay`
  = seconds from send-complete), same Bernoulli-per-request draw, same
  `delay = 0` meaning, and warmup excluded from cancellation without consuming
  RNG draws (`timing/request_cancellation.py:104-135`, byte-identical to
  baseline, vs `rust/runtime/src/timing/cancellation.rs:114-160`).
- **Request-budget remainder handling.** Rust's `owned_positions` tiles exactly,
  so `requests` and `sessions` sum to the authored total for any worker or cell
  count (`rust/runtime/src/engine/cell_launcher.rs:331-338`, asserted by
  `cell_launcher.rs:471-478`); this matches upstream's single-issuer total.
- **P1.14 (worker-local cap slicing) — KNOWN(now-fixed on the default path).**
  `concurrency` and `prefill_concurrency` are no longer sliced per thread under
  any non-`Sharded` mode; they are enforced through one shared `GlobalSlotPool`
  per phase (`rust/runtime/src/engine/sharded_scheduled.rs:154`, `:166-170`,
  `rust/runtime/src/engine/execute/sharding.rs:48-65`). The original defect still
  reproduces under an explicit `--dispatch sharded`, and the `users` axis is
  unfixed (finding 8).
- **P1.12 (fixed-schedule admission and stop behavior) — KNOWN(still-true).**
  Rust's fixed-schedule projection hardcodes `duration: None` and
  `cancellation: None` (`rust/runtime/src/config/resolve.rs:1402-1415`), replaces
  the authored `--request-count` with the trace's line count
  (`rust/cli/src/load.rs:301-315`), and the workload "intentionally ignores stop
  bounds" (`rust/runtime/src/fixed_schedule.rs:9-10`), while upstream applies the
  full stop chain — `config/phases.py:659` only relaxes the *requirement*, not
  the enforcement.
- **User-centric turn budgets and pacing.** The fresh `order = 0` user is bound
  to its concrete sample's turn count on both sides, not the dataset average
  (`timing/strategies/user_centric_rate.py:189` and `:253` vs
  `rust/runtime/src/user_centric.rs:415-427`, which overrides the plan's
  `avg_session_turns` from `rust/runtime/src/timing/user_centric.rs:152-157`).
  Continuation pacing is `max(now, previous + turn_gap)` on both
  (`timing/strategies/user_centric_rate.py:427` vs
  `rust/runtime/src/user_centric.rs:203-210`).
- **Fixed-schedule continuation precedence.** `timestamp_ms` → `delay_ms` →
  immediate, with `delay_ms` measured from the previous turn's response end
  (`timing/strategies/fixed_schedule.py:152-178` vs
  `rust/runtime/src/fixed_schedule.rs:279-301`).
- **Low-severity undocumented refusal.** Rust rejects user-centric mode when the
  dataset's average turn count rounds below 2
  (`rust/runtime/src/user_centric.rs:400-403`); upstream accepts it and floors
  `session_lifetime` at 1 (`timing/strategies/user_centric_rate.py:213`,
  `session_lifetime = max(1, session_turns - 1)`). Loud, so out of scope beyond
  this note.

## Unverified / needs runtime check

- **Findings 4 and 6 — terminal behavior.** A stop-condition-free Rust phase
  installs only `Lifecycle`, which trips on external cancellation or
  `sending_complete`. Whether the phase ends depends on whether the conversation
  source raises on exhaustion or recycles (`allow_dataset_wrap`). Needs
  `aiperf profile --concurrency 8 --request-count 20 --warmup-concurrency 4` and
  `aiperf profile --user-centric-rate 5 --num-users 5` against
  `aiperf-mock-server` under a wall-clock bound, checking whether each phase
  reaches `COMPLETE`.
- **Magnitude of finding 2.** The metric delta between a true renewal process and
  a jittered grid at equal mean rate needs a paired run (`--dispatch global` vs
  `--dispatch global-hop`, or `--workers 1`) at fixed TTFT/ITL with jitter
  coefficients zeroed, comparing per-record admission-time gaps and the TTFT
  tail. `global-hop` is documented as preserving arrival statistics
  (`rust/runtime/src/engine/global_hop.rs:17-20`), so it is the natural control.
- **Cross-engine RNG stream equality.** Python derives arrival RNG by namespace
  (`rng.derive("timing.request.poisson_interval")`) and Rust from an integer seed
  (`ConfiguredRandomGenerator::from_seed_or_entropy(Some(seed))`,
  `rust/runtime/src/timing/intervals.rs:63`). Whether the same `--random-seed`
  yields the same interval sequence in both engines is unverified; the existing
  `rust/e2e-tests/tests/test_seeded_poisson_parity.rs` compares Python against a
  Rust *reference generator*, not the product path (KNOWN P0.7).
- **Measurement-window definition.** Both engines derive throughput from
  per-record timestamps rather than an explicit window on the paths read here,
  and Rust adds an opt-in `--steady-state` window with no Python counterpart
  (Rust-only, out of scope). A byte-level comparison of the denominator used for
  `request_throughput` was not completed and should be audited alongside the
  metrics domain.
