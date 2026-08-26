# Native High-Resolution Request-Rate Pacing

## Purpose

Port the user-visible timing semantics of origin/main commit
`21f8ad7b3e621285a1682b336df16607e7d3bb9f` onto the native clock and
request-rate scheduler without duplicating Python's event-loop-specific pacer
architecture.

## Existing native foundation

`RealClock` already bypasses coarse event-loop timers on Linux by creating a
one-shot `CLOCK_MONOTONIC` timerfd and awaiting it through Tokio `AsyncFd`.
`OwnedFd` and future drop protect the descriptor on success, failure, and
cancellation, and syscall/reactor failures fall back to the remaining duration
on Tokio's timer. `SimClock` supplies exact integer-nanosecond deadlines for
deterministic tests.

The missing semantic is schedule retention. The local/sharded
`RequestRateWorkload` currently moves any past target to `now`. At intervals
below ordinary scheduling jitter, each small oversleep therefore resets the
grid and permanently forfeits offered load.

## Approved design

### Bounded local catch-up

Capture `AIPERF_TIMING_MAX_CATCHUP_SECONDS` once while constructing a
`RequestRateWorkload`. Omission selects `0.01` seconds. A provided value must
parse as finite `f64` in the inclusive range `0.0..=10.0`, then convert once to
rounded integer nanoseconds. Invalid values fail construction with the variable
name, rejected value, and range in the error.

At the start of each local/sharded tick:

1. Read `now_ns` once.
2. If `next_target_ns < now_ns - max_catchup_ns`, re-anchor the target to
   `now_ns`.
3. Otherwise retain the absolute target, including a target slightly in the
   past; yield once before admission so continuations and returns can progress.
4. Draw and add the following interval before dispatch, preserving current ramp
   and sampler ordering.

A zero catch-up window preserves the old no-burst policy. Saturating arithmetic
must make extreme clock/target values safe. A lag exactly equal to the window is
retained; only a strictly larger lag re-anchors, matching upstream.

### Global dispatch boundary

`global`, `global-hop`, and `global-push` use `GlobalRateGate`: every claimed
slot is both an aggregate fire-grid position and, for supported sampling, the
corpus position. Those modes already retain late slots and therefore do not
under-deliver due to per-tick re-anchoring. Shifting or dropping a shared claim
would violate their dense-slot and corpus-order guarantees, so this port leaves
the global gate unchanged. Its existing yield-on-past-target path continues to
provide callback progress.

### High-resolution timer selection

Do not add a Python-style pacer factory, background thread, or
`AIPERF_TIMING_HIGH_RES_TIMER` bypass. Native timing is injected through
`Clock`; routing a request-rate wait around that seam would break virtual time
and introduce a second timing authority. The upstream diagnostic toggle is
specific to choosing between Python's event-loop timer and its new pacer, while
native already has one high-resolution real-clock implementation.

Do not add synchronization to the per-request loop. Reusing one timerfd across
all `RealClock` callers is out of scope because a clock supports concurrent
transport, scheduler, cancellation, and workload sleepers; any reusable timer
design requires its own benchmark and cancellation protocol.

## Errors and resource safety

Environment parsing occurs before issuance. No environment lookup, allocation,
logging, descriptor creation policy, or synchronization is added to the tick
arithmetic. Existing `RealClock` fallback remains non-fatal and measures elapsed
time before sleeping the remainder. All integer calculations are saturating.

## Validation

Deterministic tests use literal nanosecond targets to prove:

- a 1 ns lag inside a 10 ms window stays on the authored grid;
- a lag beyond 10 ms re-anchors to the single captured `now`;
- zero window retains the previous no-catch-up behavior;
- exact-boundary and saturated arithmetic do not re-anchor or wrap;
- environment omission/default, valid endpoints, non-numeric, non-finite, and
  out-of-range values produce the stated policy or error.

A real-clock Rust integration characterization drives the actual
`RequestRateWorkload` with a constant sub-millisecond interval and an immediate
in-process dispatcher. It asserts the exact requested completion count and a
broad elapsed-rate floor/ceiling that detects timer quantization while tolerating
shared-host scheduling. A release-mode benchmark receipt records exact count,
elapsed time, and achieved requests/second; it is evidence, not a flaky
microsecond wakeup assertion.

## Source anchors

- `rust/runtime/src/clock/real_clock.rs`
- `rust/runtime/src/timing/arrival.rs`
- `rust/runtime/src/request_rate.rs`
- `rust/runtime/tests/request_rate_sim.rs`
- `rust/runtime/tests/request_rate_real.rs`
