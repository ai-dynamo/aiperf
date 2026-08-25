# Native AgentX Global Idle Guard

## Problem

Upstream commit `ce715ae849` changes the AgentX MVP policy from per-trace
idle-gap compression to a ten-second global idle guard. Native replay engines
already implement the global guard for legacy and Graph-IR Weka paths, but
`inferencex_agentx_mvp()` still synthesizes
`trace_idle_gap_cap_seconds: Some(10.0)`. That rewrites the recorded timeline,
and scenario resolution does not reject either legacy compression override.

## Decision

Extend `ScenarioSpec` with an optional default
`system_idle_gap_cap_seconds` and explicit forbiddance switches for
`trace_idle_gap_cap_seconds` and `inter_turn_delay_cap_seconds`. The AgentX
MVP scenario defaults only the global cap to 10 seconds, leaves the two legacy
caps unset, and forbids users from supplying either legacy cap. Other
scenarios retain their existing values and no scheduler algorithm changes.

`apply_scenario_synthesis` must project the scenario default when the user did
not author a global cap. Scenario locking must emit normal, bypassable
`ScenarioViolation` records for authored legacy-cap values, preserving existing
`--unsafe-override` handling. The resolved global cap continues through the
existing protocol-v2 projection to both Weka semantics; no wire schema changes
are needed.

## Acceptance criteria

1. AgentX MVP resolves `system_idle_gap_cap_seconds` to 10 seconds without
   synthesizing a trace or inter-turn cap.
2. Authored `--trace-idle-gap-cap-seconds` and
   `--inter-turn-delay-cap-seconds` each fail scenario validation, or surface
   as recorded invalidity under `--unsafe-override`.
3. Both legacy and Graph-IR Weka replay receive the same resolved global cap
   and retain recorded relative timing except when the whole system is idle.
4. Existing non-AgentX scenario defaults are unchanged.
5. Unit resolver tests and the established system-idle-gap E2E coverage pass;
   Graham review approves the final port diff.

## Scope boundaries

The change is limited to native scenario specification and resolution. It does
not alter trace parsing, the scheduler, the Rust/Python compatibility schema,
or any unrelated shim work present in the shared checkout.
