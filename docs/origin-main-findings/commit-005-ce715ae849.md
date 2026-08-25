# Commit 005 — `ce715ae849` — preserve think time with a global idle guard

**Status: applicable; replay mechanics are already covered, but the native
scenario lock is not.** The upstream change replaces AgentX MVP's per-trace
10-second compression with a 10-second *global* idle guard: pending timers are
shifted together only when no request is running or ready. It also forbids the
per-trace and per-turn compression knobs for that scenario.

## Evidence

- `ce715ae849` is already an ancestor of the current branch, through actual
  merge commit `4fe700caef` (second parent `ce715ae849`). Its Python payload is
  therefore present, but that does not by itself establish native parity.
- The native CLI accepts `--system-idle-gap-cap-seconds` in
  `rust/cli/src/flags.rs`, validates/projects it through the v2 configuration
  path, and documents that it applies to both Weka semantics arms.
- Legacy AgentX replay implements the operational rule in
  `rust/runtime/src/agentic_replay.rs`: `cap_system_idle_offsets_ms` preserves
  relative spacing when phase-start work is shifted, and
  `system_idle_continuation_delay_ms` caps a continuation only when the
  scheduler has no other pending work. The accompanying
  `system_idle_gap_tests` cover both cases.
- Graph-IR carries the same authored cap to `ExecutorFlags` through
  `rust/runtime/src/engine/protocol_v2.rs` and
  `rust/runtime/src/engine/graph_execution.rs`. The end-to-end guard in
  `rust/e2e-tests/tests/test_system_idle_gap_cap.rs` proves capped waits while
  preserving the trace's turn order.
- The scenario layer is stale relative to this upstream commit:
  `rust/runtime/src/agentx/scenario.rs::inferencex_agentx_mvp` still sets
  `trace_idle_gap_cap_seconds: Some(10.0)` and has no system-cap or
  forbid-per-trace/forbid-per-turn fields. Consequently
  `rust/runtime/src/config/resolve.rs::apply_scenario_synthesis` injects that
  old per-trace cap into graph reconstruction; `apply_scenario_locks` has no
  corresponding conflict checks.

## Port implications

The implementation scope is confined to native scenario configuration and
resolver projection, not the replay scheduler. Add a system-idle-cap field and
the two forbiddance fields to `ScenarioSpec`; make the AgentX MVP lock default
the global cap to 10 seconds, leave trace timing unwarped, and reject explicit
`--trace-idle-gap-cap-seconds` and `--inter-turn-delay-cap-seconds` under that
scenario. Ensure legacy and graph-IR receive the same resolved cap.

Tests should cover: scenario default projection to the global cap; both
forbidden flags yielding scenario violations; unmodified recorded spacing under
the scenario; and the existing global-idle E2E on each Weka semantic arm. The
feature merits a focused spec and Sol implementation plan before code changes;
no new scheduler design is required.
