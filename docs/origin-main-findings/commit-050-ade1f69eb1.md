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

## Native applicability and gap

The native public protocol has the same incoming authored meaning:
`PhaseCommonSpec.seamless` belongs to the phase carrying it. The internal timing
runtime instead represents an outbound handoff on `PhaseConfig.seamless`.
`phase_seamless_to_next(phases, phase_index)` already bridges those directions
at the authoring-to-runtime boundary by reading only the next authored phase.
All native product execution families call that helper before constructing a
phase plan: unsharded scheduled execution, worker sharding, offline Dynamo
execution, and Graph-IR execution.

The timing runtime already supplies most of the upstream behavior. A non-final
outbound handoff starts a background return wait, retains the runner in the
active set, propagates detached terminal failure through
`SeamlessFailureSignal`, and waits at the final barrier. Phase sidecars,
including the server profiler, finish during phase finalization after returns
drain rather than at the issuance handoff.

The deeper profiler audit found a production gap despite the correct timing
handoff. Each native profiling phase owned an independent profiler sidecar, so
two overlapping profiling phases emitted `start, start, stop, stop`. The first
phase to drain could therefore stop server profiling while its overlapping
peer was still active. The cellular controller additionally rejected the
successor's `Ready` signal while the predecessor remained active.

This port adds one run-local, worker-local profiler ownership coordinator. The
first active profiling phase sends start; overlapping phases acquire ownership
without another control request; the last drained phase sends stop. Cellular
coordination now tracks cell readiness/completion per phase and uses that same
ownership policy, while terminal run cleanup force-stops an outstanding owner.
The implementation adds no lock, thread, channel, or public wire change.

The evidence gap is also closed. Authored lowering now covers positive,
inverse, middle-transition, and final behavior. Native runtime tests cover
return-drain sidecar ordering, local profiler ownership, and overlapping
cellular phase gates in addition to the existing real-HTTP overlap and detached
failure coverage.

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
| Server-profiler stop follows return drain | New simulated phase-sidecar integration test records successor finish at 5 ns and predecessor finish at 20 ns. |
| Overlapping profiling phases share one profiler session | New control-hook test proves exactly one start and one last-owner stop. |
| Cellular profiling accepts overlapping phase gates | New controller test releases both phases and retains profiler ownership until the predecessor drains. |

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

Complete.

- Design/finding: `36645b8b441a73e0b56f05960d4be56ef60838bb`.
- Exact target-specific merge: `82526331217f6cc85f621994dc56413d8698aede`.
  Its parents are native first parent
  `36645b8b441a73e0b56f05960d4be56ef60838bb` and exact upstream second parent
  `ade1f69eb13dfa0e87e49b2c027f6fe29c03d402`; merge tree and first-parent tree
  are both `c93040d69c790985db48bc04fa5edfe3d3a15bd7`.
- Authored-direction characterization: `009f3a749c`.
- Profiler ownership implementation and integration coverage: `3099426314`.
- Focused results: authored direction `1/1`, local profiler ownership `1/1`,
  cellular profiler overlap `1/1`, real-HTTP phase runtime `1/1`, simulated
  phase runtime `5/5`, orchestrator `6/6`, and runner `8/8`.
- Default runtime suite: `1804 passed`, `1 failed`, `7 ignored`. The only
  failure is the unchanged `aiperf_version` snapshot drift (`0.12.0` actual vs
  `0.0.0` expected).
- Engine library suite: `2364 passed`, `5 failed`, `7 ignored`. The failures are
  unchanged base-tree issues: two absent recorded-agent fixtures, one transport
  registry setup panic, one stale custom-transport decode fixture, and the same
  version snapshot drift.
- Changed-scope Clippy, Rust formatting, docs-current, and range whitespace
  checks pass. Full `--tests` Clippy reaches unrelated existing compile errors
  in `agentx_online_e2e` because two initializers omit
  `cache_bust_first_user_turn`.
- Self Graham review: APPROVED in `c3111b19fc`, with no Critical/Important
  findings. Independent parent Graham review of `482f859241..c3111b19fc`:
  APPROVED after two passes, with no Critical/Important findings.
