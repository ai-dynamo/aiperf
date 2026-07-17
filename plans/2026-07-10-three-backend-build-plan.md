# Build Plan: every aiperf-rust command over {real-http · online-mock · offline-mock}

**Date:** 2026-07-10
**Owner:** Anthony Casagrande
**Design:** `specs/execution-model.md` (the unified execution model) +
`specs/offline-cosimulation.md` (the engine boundary).
**Grounded in:** `src/aiperf/timing/**` and dynamo `lib/mocker/**` read line-by-line.

## Goal (the acceptance bar)

Every command should run unchanged against the current `RequestSink`-based HTTP path
(real | online-mock) and the designed offline sink/engine path, emitting **aiperf's
own report** on all three. ONLINE-REAL and ONLINE-MOCK are the same code (a base URL
apart); the real work is OFFLINE-MOCK covering the full command surface + producing
aiperf's report (Level B).

## The single most de-risking finding (grounded)

`execute_pass` in the vLLM/SGLang cores calls **only 3 collector methods** —
`on_arrival`, `on_admit`, `on_token` (verified: `vllm/core.rs:1345,1469,2052,2305`;
`sglang/core.rs:595,640`). `on_terminal` and every disagg-detail method
(`on_prefill_admit`, `on_decode_assigned`, `on_source_held`, `on_destination_*`,
`on_*_route_overlap`) are called by the **runtime** layer (`agg.rs`/`disagg.rs`),
NOT inside `execute_pass`. So the observer-generic change (Level B #1) is a **4-method
`&mut` trait + a signature swap at ~5 sites + one blanket impl** — not a rewrite. The
disagg extras stay on the concrete `TraceCollector` at the runtime layer as dynosim-
native co-observed data.

---

## Ownership split

- **aiperf side** (`rust/`): Phases 0, 1, 4, 5, 6.
- **dynamo side** (`lib/mocker`, base `f553c46a`): Phases 2, 3 — these are "conform
  the mocker to the engine-boundary spec you already wrote"; #2/#3 surface primitives
  that already exist, #1 is the one real signature change.

Phases are dependency-ordered; each is independently shippable behind its gate.

---

## Phase 0 — Current sink/clock seam + both ONLINE modes (aiperf only)

**Deliver:** keep the built `{RequestSink, Clock}` seam as the online foundation, so
every existing command runs on ONLINE-REAL and ONLINE-MOCK before sim work starts.

- Use the current contract types and traits: `loadgen-core::{Dispatchable,
  RequestSink<R>, RequestObserver}` plus `aiperf-clock::Clock`. Do not reintroduce
  the obsolete north-star backend/sink/advance trait shapes from the
  north-star sketches.
- `aiperf::http::TransportSink` wraps the `aiperf-transport-http` crate behind
  `RequestSink<HttpRequest>`, emitting observer events from SSE into the collector.
- The future offline sink should adapt the engine into the same observer contract.
- Bin: the positional/base URL selects real vs online-mock; both stay the same code
  path.

**Gate:** every existing workload runs against a real server AND against the mocker's
HTTP server (`ONLINE-MOCK` = URL swap), producing aiperf's report. No sim yet.

---

## Phase 1 — Clock-thread the entire timing layer (aiperf only)

**Deliver:** virtual clock becomes possible — nothing reads wall time.

- Every seam that needs `now` takes it from the injected `Clock`: `Workload`, `Gate`,
  `RatePool`/`IntervalGenerator`, `SlotPool` (interval only; the semaphore is fine),
  `Controller`/`WindowSampler`, `RampStrategy`, phase lifecycle, `CancellationPolicy`.
- Port the wall-clock offenders from the Python read: `request_rate.py:149`,
  `user_centric_rate.py:373,404`, `adaptive_scale.py:139,332`, `ramping.py:192`,
  `lifecycle.py:87,151` → `clock.now()`.
- Add a **CI grep-gate** scoped to product source: reject `Instant::now` /
  `SystemTime::now` / `Instant::elapsed` outside `RealClock` and explicitly allowed
  test/benchmark wall-time measurement. Do not grep docs/specs or reject legitimate
  throughput benchmark timers.

**Gate:** sim-safety audit (spec §11) green; suite passes on `RealClock` with the
grep-gate enforced. Unblocks `SimClock`.

---

## Phase 2 — dynamo: observer-generic `execute_pass` (Level B #1, behavior-preserving)

**Deliver:** the cores emit through an injected `&mut` observer instead of a concrete
`TraceCollector`. This is what lets aiperf's collector receive offline events live.

- Define a 4-method `&mut` observer trait beside the existing `RequestObserver`
  (which is `&self`/Send+Sync — keep it for the online path). Call it e.g.
  `PassSink`:
  ```rust
  pub trait PassSink {
      fn on_arrival(&mut self, uuid: Uuid, ms: f64, isl: usize, osl: usize);
      fn on_admit(&mut self, uuid: Uuid, ms: f64, reused_input_tokens: usize);
      fn on_token(&mut self, uuid: Uuid, ms: f64);
      fn on_terminal(&mut self, uuid: Uuid, status: ReplayTerminalStatus);
  }
  ```
- Swap `collector: &mut TraceCollector` → `obs: &mut dyn PassSink` at the pass sites:
  `scheduler/vllm/core.rs:1030` + `execute_hidden_pass:1038` + `execute_pass_internal:1304`;
  `scheduler/sglang/core.rs:553` + `:561` + `:565`; `scheduler/mod.rs:290,301`
  (EngineCore); `replay/offline/core.rs:69` (ReplayWorkerCore); `replay/offline/state.rs:379,387`.
  The cores only call `on_arrival/on_admit/on_token`, so the edits are mechanical.
- `impl PassSink for TraceCollector` (delegate to its existing inherent `&mut`
  methods). Every self-driving runtime keeps passing `&mut self.collector`
  unchanged — it now coerces to `&mut dyn PassSink`.
- The runtime-level `on_terminal` + disagg-detail calls stay on the concrete
  `TraceCollector` (they're outside `execute_pass`) — no change.

**Gate:** all 141 mocker offline tests + `run_offline_handoff_conformance` +
`test_trace_replay_matches_manual_steps` pass **unchanged** (default impl is still
`TraceCollector`). Pure refactor; zero behavior change. Ships on its own.

---

## Phase 3 — dynamo: expose `next_event_ms` + `step_to` + make runtimes `pub` (Level B #2,#3)

**Deliver:** the DES event source and the caller-clocked step your spec specified.

- Surface `next_event_ms(&self) -> Option<f64>` on `SteppableEngine`/`SteppableAgg`/
  `SteppableDisagg`. For agg/disagg it's the existing private `next_timestamp()`
  (`agg.rs:444`, `disagg.rs:1320`) exposed via a non-mut peek; for single-worker it's
  derived (`in_flight()>0 ? Some(now) : None`, or a pass-deadline peek if cheap).
- Promote `advance_to(until_ms)` (`agg.rs:919`, `disagg.rs:2066`) out of
  `#[cfg(test)]` into the production trait as
  `step_to(&mut self, until_ms: f64, obs: &mut dyn PassSink) -> f64` — it already
  clamps to `until_ms` (no overshoot). The shipped engine-clocked `step_dynamic`
  stays as the graph-offline fast path.
- Make `EngineCore` / `ReplayWorkerCore` / `EnginePassResult` `pub` (or curate a
  `pub` facade) so the aiperf `MockerEngine` can construct + drive them.

**Gate:** conformance byte-exact still holds; new unit test asserts `step_to(until_ms)`
advances to `min(next_event, until_ms)` and never past `until_ms` when the next event
is beyond it. Ships on its own.

---

## Phase 4 — aiperf: `MockerEngine` + `ObsAdapter` + `EngineHost` + general `SimBackend`

**Deliver:** the offline backend, general (not graph-only), producing aiperf events.

- `MockerEngine` (impl `Engine`) wraps `Box<dyn SteppableReplay>` (single/agg/disagg
  from flags): `admit` = `set_now_ms + submit`; `step(now, out)` =
  `set_now_ms(ns_to_ms(now)); step_to(ns_to_ms(now), &mut ObsAdapter{out})`;
  `next_wake` = `next_event_ms().map(ms_to_ns)`; `is_idle`.
- `ObsAdapter` (impl `PassSink`): `on_arrival/on_admit/on_token/on_terminal(uuid, ms)`
  → `out(reqid, Event{ at: ms_to_ns(ms) })`. First `on_token` → `FirstToken`;
  terminal → `Done{terminal}` (`rejected`→`Rejected`).
- `EngineHost` owns the engine + `ReqId→CompletionSlot`; routes events to the
  per-request observer/sink adapter, resolves the slot on `Done` (wakes via the
  virtual-clock wait primitive).
- The offline sink should implement the same `RequestSink` contract as the online
  transport sink: admit → register slot → drive/await terminal events. The current
  graph runtime already has `drive_sim`; general offline work should extend that
  architecture rather than retire a nonexistent graph-only co-sim driver.
- Optional co-observer: `Tee(ObsAdapter→aiperf, TraceCollector)` when the run wants
  dynosim-native extras (prefix-reuse, GPU-hours).

**Gate:** OFFLINE-MOCK runs the SIMPLE modes (concurrency, request-rate) end-to-end,
producing aiperf's report; conformance vs batch on the fixture corpus.

---

## Phase 5 — Runtime DES pump + every workload offline (aiperf)

**Deliver:** the virtual-clock pump; every command works offline.

- `Runtime::run_sim`: drain ready-queue → `t = min(clock.next_parked(),
  engine.next_wake())` → `clock.advance_to(t)` → `engine.step(t, &mut route_events)`
  (step_to, clamped) → route Events + resolve slots (WaitQueue FIFO). `run_real` stays
  the reactor branch. Branch on `clock.is_virtual()`.
- Run **every** workload offline: user-centric, fixed-schedule/trace, agentic/DAG,
  adaptive-scale. Adaptive's `WindowSampler`/`Controller` consume the live aiperf
  event stream (Level B) — no post-hoc bridge.
- Wire future offline flags with `clap` alongside the current positional/base URL,
  `--mode`, `--http2`, and graph/online options. Avoid spike-era env toggles such as
  environment-variable HTTP/2 toggles or hand-rolled argv scans.

**Gate:** spec §14 acceptance — every command × 3 backends, aiperf's report; Level B
live (adaptive windows + streaming metrics + dashboard work offline); equal-instant
ordering matches `SingleRuntime` (arrivals-before-pass).

---

## Phase 6 — parity + determinism harness

**Deliver:** the proof.

- Conformance corpus: `step_to`-driven == dynamo batch `run()` byte-for-byte.
- Offline bit-reproducibility (10× + parallel), CLI byte-identical.
- "Same workload file compiles + runs under `{Real,Virtual}Clock` × `{Http,Sim}`" test.
- Determinism: assert `SlotPool`/`RatePool`/`await_inputs` admission order == FIFO
  ready-queue order (Invariant W).

**Gate:** full acceptance checklist (spec §14) green.

---

## Dependency graph

```
0 ──▶ 1 ──▶ 4 ──▶ 5 ──▶ 6
        ▲     ▲
2 ──▶ 3 ┘─────┘        (2,3 are dynamo; 4 needs 3's pub surface + step_to/next_event_ms
                        and 2's observer-generic pass)
```

- 0,1 (aiperf) and 2,3 (dynamo) proceed in parallel.
- 4 needs 1 (clock-threaded) + 2 (observer-generic) + 3 (next_event_ms/step_to/pub).
- 5 needs 4. 6 needs 5.

## Risks

- **R1 — `PassSink` `&mut` vs `RequestObserver` `&self`.** Don't reuse the online
  `RequestObserver` (Send+Sync, `&self`); the offline step wants exclusive `&mut`.
  Define `PassSink` separately; `TraceCollector` impls both trivially. (Phase 2.)
- **R2 — single-worker `next_event_ms`.** Agg/disagg have `next_timestamp`; single
  may need a pass-deadline peek. Fallback: `in_flight()>0 ? Some(now) : None` (coarse
  but correct for the graph/closed-loop cases; only open-loop rate offline needs the
  precise value). (Phase 3.)
- **R3 — determinism when threading the slot pool through the DES executor.** Slot
  admission order must ride `WaitQueue`. Covered by Phase 6's Invariant-W test.
- **R4 — disagg sub-ms tail nondeterminism** (noted in commit `f018d27`): 24
  simultaneous arrivals hit a pre-existing batch-scheduler tie. Assert count/token
  invariants for disagg, byte-for-byte only for single/agg. (Phase 6.)
