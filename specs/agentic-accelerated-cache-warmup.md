<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic-replay accelerated cache-pressure warmup + handoff

## Purpose

Close the final byte-exact-parity gap with Python `AgenticReplayStrategy` for the
Rust legacy `agentic_replay` timing mode: the **gated accelerated cache-pressure
warmup** and the **warmup→profiling residual-delay handoff**. The legacy path
(`rust/runtime/src/agentic_replay.rs`, lowered by `lower_legacy_agentic`)
implements this: `AgenticReplayWorkload::execute` runs the accelerated substage
(`execute_accelerated_warmup`) and the carrier-driven profiling resume
(`execute_profiling_resume`), and `lower_legacy_agentic` threads the cross-phase
[`WarmupHandoffCarrier`](../rust/runtime/src/agentic_replay.rs) between the two
agentic phase instances.

When `--agentic-cache-warmup-duration <sec>` is set, the WARMUP phase does not stop
after the static turn-(n-1) prime. Instead it **continues replaying the sampled
live trajectories under compressed traffic** (zero idle delay, `max_tokens=1`) for
the configured wall-duration to build KV-cache pressure, then **hands the drained
live stream positions to PROFILING** so profiling resumes each lane at its true
next turn — each carrying a *residual* next-turn delay so the handoff ramps into
the recorded cadence instead of firing every lane at once.

## Reference implementation already in-repo

The **graph path already implements this exact feature**, Clock-derived and
byte-exact under `SimClock`. The legacy port mines it rather than reinventing:

- Drain wall + per-lane return wall on the `Clock`: `graph_phase_runtime.rs`
  (`drain_end_wall_us = clock.now_ns()/1_000.0`, `GraphLaneLedger::observe_return`
  / `return_wall_us`).
- Residual formula: `graph/snapshot.rs::chop_trie_at_frontier` — `recorded_delay −
  max(0, drain_end_wall_us − return_wall_us[pred])`, AND-fan-in takes the max,
  floored at 0, clamped to a cap.
- Cross-phase carrier: `GraphWarmupHandoff` / `LaneHandoff`
  (`graph/warmup_handoff.rs`), `build_warmup_handoff` +
  `build_profiling_resume_lane_plans`.
- Pressure-recycle: `PreparedPressureRecycle` / `build_pressure_recycle`.

**Divergence to honor for byte-exactness:** the graph path caps the residual with a
fixed `HANDOFF_RESIDUAL_CAP_SEC = 60.0`; Python caps with the **trace idle-gap cap**
(`_handoff_residual_delay_ms` → `_phase_offset_cap_ms`, sourced from the dataset
idle-gap cap). The legacy port MUST use the idle-gap cap (the same cap already
threaded as `idle_gap_cap_ms` in `AgenticReplayConfig`), not 60s.

## Byte-exactness bar and the Clock

Python captures `_handoff_returned_at_ns` and `finalized_at_ns` with
`time.perf_counter_ns()`; the residual subtracts real drain wall-time. This is
inherently wall-clock-dependent, so:

- **Under `SimClock`: byte-exact and reproducible** — both captures route through
  `Clock::now_ns()`, drain elapsed is deterministic virtual time. This is the
  parity bar the golden fixture asserts.
- **Under `RealClock`: statistical parity only** — wall-jittered exactly as Python
  is. This is not a regression; it is the same property Python has.

The port MUST route every time capture (per-return wall, drain-finalize wall,
duration timer) through the injected `Clock`, never `Instant::now`. Matches the
CLAUDE.md clock-seam mandate.

## Config gating

Port `validate_agentic_cache_warmup` (`src/aiperf/config/config.py:689`):
`--agentic-cache-warmup-duration` is rejected unless the resolved profiling timing
mode is `agentic_replay` (scenario-declared or phase-resolved). The field already
threads CLI→`PhaseCommonSpec.agentic_cache_warmup_duration` (`protocol.rs:453`);
`lower_legacy_agentic` reads it onto the synthesized WARMUP phase and installs a
live [`WarmupHandoffCarrier`](../rust/runtime/src/agentic_replay.rs) when present.
The guard belongs in `cli/src/load.rs`/`phase_validate.rs` alongside the existing
weka-semantics guard.

## Architecture

Six units, dependency-sorted. Each is independently testable.

### 1. Config → workload plumbing

`AgenticReplayConfig` gains `cache_warmup_duration_s: Option<f64>` and a
`max_tokens_override: Option<u32>`. `lower_legacy_agentic` reads
`common.agentic_cache_warmup_duration`; when `Some`, the warmup phase is lowered as
an **accelerated** warmup (duration-bearing) rather than the static prime, and a
handoff carrier is threaded warmup→profiling (see unit 6).

### 2. `max_tokens=1` pressure override

A per-phase `max_tokens_override` applied in the first-turn/continuation builder
(`build_first_turn`), forcing single-token generation during pressure. No
issuer-global mutable state; the override rides the config.

### 3. ReplayBarrierCoordinator + gate

Port `ReplayIssueGate` / `ReplayBarrierCoordinator`
(`src/aiperf/timing/replay_dependencies.py:156-435`) as a worker-local (single
central driver) struct: `activate`/`pause_releases`/`complete`/
`seed_completed_prefixes`/`completed_prefixes`/`pending_turns`/
`pending_turns_by_root`/`close_root`. It releases a turn only once its recorded
`replay_predecessors` (from the existing `infer_cross_stream_predecessors`,
`agentx/replay_dependencies.rs:67`) have completed; retains pending turns on pause;
exposes retained turns and completed prefixes for the handoff. This is the largest
component; it is pure logic (no I/O) and golden-testable against Python in
isolation.

### 4. Return-observation in the central driver

The single global-hop driver already owns each request's terminal via the
`schedule_agentic_turn` continuation closure. Extend that closure (no new
`Workload` trait method) to, during accelerated warmup: `gate.complete(credit)`,
and record `handoff_credits[x_corr]` + `return_wall_ns[x_corr] = clock.now_ns()`
for non-final turns (pop on final). Mirrors `observe_credit_return`.

### 5. Duration timer → drain → finalize

`clock`-scheduled timer at `+cache_warmup_duration_s` fires `finish_accelerated`:
`gate.pause_releases()` + stop new issuance (drain in-flight without freezing
counts). On phase end, `finalize`: capture `finalized_ns = clock.now_ns()`, build
handoff states (returned mid-flight + barrier-pending), compute each
`next_dispatch_offset_ms` via the residual formula (unit 6), build replay-resume
boundaries, and populate the handoff carrier.

### 6. Handoff carrier + trajectory rebuild

A `LegacyWarmupHandoff` object (analogue of `GraphWarmupHandoff`) carrying, per
lane, the surviving `ConversationState`s (sorted `(agent_depth, x_correlation_id)`)
with residual `next_dispatch_offset_ms` and merged replay-resume boundaries
(sorted by `conversation_id`). Empty lanes draw a fresh recycle root
(`next_recycle_conversation_id`). PROFILING reads the carrier instead of the
load-time `::warmup`-split trajectory list: it re-seeds prefixes, activates the
gate, and dispatches each lane at its residual offset. Port targets:
`_build_handoff_states`, `_add_returned_handoff_states`,
`_returned_credit_handoff_state`, `_add_pending_handoff_states`,
`_pending_turn_handoff_state`, `_handoff_lane_for_turn`,
`_handoff_residual_delay_ms`, `_handoff_base_delay_ms`,
`_build_handoff_replay_boundaries`, `_build_handoff_trajectories`.

### Residual-delay formula (exact)

```
base = next_turn.delay_ms                        if present & finite (>=0)
     else next_ts_ms - prev_ts_ms - max(0, prev.api_time_ms or 0)   (>=0)
     else 0
returned_ns present? elapsed_ms = max(0, (finalized_ns - returned_ns)/1e6)
                     delay = max(0, base - elapsed_ms)
                else delay = base
cap present?         delay = min(delay, idle_gap_cap_ms)   # idle-gap cap, NOT 60s
```

## Error handling

- A child continuation refused mid-pressure (e.g. `--request-count` cap) drains its
  parent's join via the existing `TreeGate::on_child_terminal` — never deadlocks.
- A lane with no dispatchable root and no baseline root acquires a holding lane
  credit for concurrency accounting (port `acquire_lane_credit`), released on tree
  drain.
- Duration `<= 0` or absent → the static prime path (today's behavior) unchanged.

## Testing (byte-exact bar)

- **Unit goldens (SimClock, deterministic):**
  - ReplayBarrierCoordinator release/pending/completed-prefix decisions vs a
    Python-generated fixture over a fixed predecessor graph.
  - Residual-delay formula vs Python `_handoff_residual_delay_ms` over fixed
    `(base, returned_ns, finalized_ns, cap)` rows, including the idle-gap-cap clamp
    and the elapsed-subtraction floor.
  - Handoff trajectory rebuild: state ordering `(agent_depth, x_correlation_id)`,
    boundary ordering by `conversation_id`, empty-lane recycle draw. Correlation
    ids are injected from a seeded source on both sides (Python uses `uuid4`).
- **E2e (SimClock):** `aiperf profile --scenario inferencex-agentx-mvp
  --public-dataset weka_cc_traces_062126 --agentic-cache-warmup-duration <sec>`
  against `aiperf-mock-server`: assert (a) warmup records carry `max_tokens=1` and
  fire at zero idle delay; (b) profiling resumes each lane at its true next turn
  index (not turn 0) with the residual offset; (c) a lane whose tree drained during
  warmup recycles a fresh root in profiling; (d) aggregate metrics present.
- **Config guard:** `--agentic-cache-warmup-duration` on a non-agentic scenario is
  rejected with the ported message.

## Out of scope

Nothing further — this is the terminal parity subsystem. The legacy `agentic_replay`
mode reaches full byte-exact parity (under `SimClock`) with Python
`AgenticReplayStrategy`. The linear-lane MVP populates the DAG fields
(`agent_depth`, `parent_correlation_id`, `root_correlation_id`, `branch_mode`) with
their tree-root defaults; full subagent-depth population across the handoff waits on
the WEKA loader emitting those onto each turn (a separate loader-seam task).

## Source anchors

- Python subsystem: `src/aiperf/timing/strategies/agentic_replay.py`
  (accelerated/handoff methods ~540-1120), collaborators
  `src/aiperf/timing/replay_dependencies.py`,
  `src/aiperf/timing/trajectory_source.py`, `src/aiperf/credit/issuer.py`,
  `src/aiperf/credit/callback_handler.py`, guard
  `src/aiperf/config/config.py:689`.
- Rust graph reference: `rust/runtime/src/engine/graph_phase_runtime.rs`,
  `rust/runtime/src/graph/snapshot.rs` (`chop_trie_at_frontier:201`),
  `rust/runtime/src/graph/warmup_handoff.rs`.
- Rust legacy target: `rust/runtime/src/agentic_replay.rs`,
  `rust/runtime/src/engine/online_execution.rs` (`lower_legacy_agentic`),
  `rust/runtime/src/agentx/replay_dependencies.rs`,
  `rust/runtime/src/agentx/session_tree.rs`,
  `rust/runtime/src/scheduled.rs` (`Workload`, `ScheduledRuntime`, `ClockTaskScheduler`).
- Related records: [agentic-replay-join-gating.md](agentic-replay-join-gating.md),
  [agentx-rust-port.md](agentx-rust-port.md).
