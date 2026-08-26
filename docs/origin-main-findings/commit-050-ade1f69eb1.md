# Commit 050 — `ade1f69eb1`

## Upstream intent

Upstream fixes the direction of the public `seamless` phase flag. The flag is
authored on the phase which may begin without waiting for its predecessor's
in-flight requests. Therefore phase `i` must hand off after sending precisely
when phase `i + 1` has `seamless: true`; phase `i`'s own flag describes its
incoming transition and must not make it hand off to phase `i + 1`.

The Python repair computes `seamless_to_next` in the orchestrator, passes that
outbound fact explicitly into `PhaseRunner.run`, and uses the same fact for
detached return waiting, active-runner cleanup, fatal-error callback wiring,
and deferred server-profiler stop. A final phase never hands off, even if its
incoming flag is true.

## Native applicability and current gap

The native public protocol has the same incoming authored meaning:
`PhaseCommonSpec.seamless` belongs to the phase carrying it. The internal timing
runtime instead represents an outbound handoff on `PhaseConfig.seamless`.
`phase_seamless_to_next(phases, phase_index)` already bridges those directions
at the authoring-to-runtime boundary by reading only the next authored phase.
All native product execution families call that helper before constructing a
phase plan: unsharded scheduled execution, worker sharding, offline Dynamo
execution, and Graph-IR execution.

The runtime already supplies the rest of the upstream behavior. A non-final
outbound handoff starts a background return wait, retains the runner in the
active set, propagates detached terminal failure through
`SeamlessFailureSignal`, and waits at the final barrier. Phase sidecars,
including the server profiler, finish during phase finalization after returns
drain rather than at the issuance handoff.

The behavior is therefore already implemented, but the evidence is incomplete.
The existing adapter unit test proves only the positive two-phase case, while
the runtime overlap tests construct internal `PhaseConfig` values directly and
do not prove the public authored direction. This port adds explicit negative,
middle-transition, and final-phase mapping tests and keeps the real HTTP overlap
and detached-failure tests as composed runtime evidence. No production refactor
is justified because it would only move the already-correct adapter contract.

## Upstream-to-native test map

| Upstream behavior | Native evidence |
| --- | --- |
| Successor `seamless: true` makes its predecessor hand off | Authored-phase lowering test checks `[false, true] -> [true, false]`. |
| A phase's own incoming flag does not make it hand off to a non-seamless successor | New inverse lowering test checks `[true, false] -> [false, false]`. |
| Only the predecessor of each seamless phase hands off in a longer workflow | New three-phase table test checks exact outbound flags for every index. |
| A final phase never creates an outbound handoff | New final-phase test checks authored `seamless: true` lowers to `false`. |
| Handoff overlaps real in-flight HTTP work | Existing `phase_runtime_online::seamless_phases_overlap_over_the_real_http_dispatcher`. |
| Non-seamless transitions wait for predecessor drain | Existing simulated orchestrator/runtime transition tests. |
| Detached fatal failure cancels the active successor and fails the run | Existing `seamless_predecessor_failure_cancels_active_phase_before_advancing`. |
| Server-profiler stop follows return drain | Phase sidecar lifecycle tests plus the shared runner finalization boundary. |

## Ancestry constraint

The isolated branch is based on `482f859241`, whose previously integrated
upstream `c2889280a6` ancestry already contains `ade1f69eb1`. Porcelain Git
therefore reports the exact target as already merged and cannot create another
`--no-ff` commit. The port must still record an explicit two-parent merge whose
first parent is the native review state, whose second parent is exact upstream
`ade1f69eb13dfa0e87e49b2c027f6fe29c03d402`, and whose tree equals its first
parent. This is a real merge commit created with Git's commit-tree plumbing,
not a cherry-pick or Python-tree import.

## Closure evidence

Pending implementation, verification, and two Graham approvals.
