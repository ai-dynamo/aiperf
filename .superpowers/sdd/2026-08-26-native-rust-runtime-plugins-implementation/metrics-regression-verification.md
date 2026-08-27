# Verification record — `f53eb2fee3` fix(metrics): restore dense metric tag declaration order

Closes Graham review finding **I1** ("mandated engine gate not run; verification
record claims a green it does not have").

The original verification record asserted "the suite is fully green after the fix".
**That claim was false.** It was made on the strength of
`cargo test -p aiperf-runtime --lib` alone. `CLAUDE.md` mandates a second gate:

> `aiperf-runtime`'s `engine` module … sits behind the `engine` Cargo feature, off by
> default; `cargo test -p aiperf-runtime` alone silently runs zero tests under it, so
> run **both** invocations above.

That second gate was never run. It is red. This document records the run, corrects
the claim, and attributes every failure.

**Corrected claim:** green on `--lib`; the `--features engine` gate exits 101 with
6 failures, all attributed to base debt, missing untracked worktree fixtures, or
load flakiness. Zero failures are attributable to the fix.

## Environment

| | |
|---|---|
| Fix worktree | `/home/anthony/nvidia/projects/aiperf/ajc/rust/.worktrees/native-plugin-metrics-fix` |
| Fix branch / HEAD | `ajc/native-plugin-metrics-fix` @ `1432076c85` (parent `f53eb2fee3`) |
| Base-control worktree | `/home/anthony/nvidia/projects/aiperf/ajc/rust/.worktrees/metrics-fix-base-control` (detached) |
| Base commit | `eadd5c665f` |
| Invocation prefix | `RUSTFLAGS="--cfg tokio_unstable"`, run from each worktree's `rust/` |

`1432076c85` adds only a comment block to `tag_id.rs`, so it is behaviourally
identical to `f53eb2fee3` for every gate below.

## Gate results

| gate | exit | result |
|---|---|---|
| `cargo test -p aiperf-runtime --lib` | **0** | 1914 passed, 0 failed, 7 ignored (64.6s) |
| `cargo test -p aiperf-runtime --features engine --lib` | **101** | 2531 passed, **6 failed**, 7 ignored (489.2s) |
| `cargo clippy -p aiperf-runtime -- -D warnings` | **101** | 212 errors — byte-identical error multiset at base (see below) |

### Base control, same three gates at `eadd5c665f`

| gate | exit | result |
|---|---|---|
| `cargo test -p aiperf-runtime --features engine --lib` | **101** | 2512 passed, **25 failed**, 7 ignored (632.8s) |
| `cargo clippy -p aiperf-runtime -- -D warnings` | **101** | 212 errors |

Clippy: the `error:` line multisets from both runs were extracted, sorted, counted,
and `diff`ed — **identical**. 212 pre-existing clippy errors at base, 212 after. The
fix is a declaration reordering plus a comment and introduces none of them. Clippy
being red on this crate is pre-existing root-branch debt (review finding S4 territory),
not a property of this change.

## Attribution — the decisive result

The strongest available evidence is set containment, and it is clean:

**The 6 failures at HEAD are a strict subset of the 25 failures at base.**
`diff` of the two sorted failure lists shows 19 deletions and **zero additions**.

The fix therefore **removes 19 failures and introduces zero**. There is no test that
fails at HEAD and passed at base. This is a stronger statement than per-test
attribution and it settles I1 on its own.

The 19 removed failures are exactly the regression's blast radius — the 18 `--lib`
failures the fix targeted (`metrics_core::catalog`, `metrics_core::accumulator`,
`metrics_core::report`, `metrics::`) plus one `workers_characterization` timing flake
that happened to fire in the base run.

### Per-test attribution of the remaining 6

| failing test | attribution | evidence |
|---|---|---|
| `engine::graph_input::tests::recorded_agent_adapter_discovers_and_lowers_the_manifest_corpus` | **worktree environment** | Needs `rust/runtime/tests/fixtures/recorded_agent_replay/recordings/pinchbench-*.json`. `git ls-files` on that directory returns only `inspection.json`; the five `pinchbench-*` recordings exist **untracked** in the root repo and so are absent from every worktree. Failure text: `pinchbench-openclaw-task_meeting_council_budget-recording.json: manifest recording: No such file or directory (os error 2)`. Also fails at base. Not a defect. |
| `engine::graph_input::tests::recorded_agent_tool_execution_stages_pinch_task_pack_workspace_files` | **worktree environment** | Same missing untracked fixtures. Also fails at base. |
| `engine::artifact_stream_velo::tests::velo_stream_large_artifact_round_trips_with_bounded_memory` | **pre-existing at base** | Present in the base failure set. RSS bound violated during a 64 MiB transfer. |
| `engine::online_execution::transport_binding_differential::match_arm_binding_matches_the_registry_lookup_for_every_variant` | **pre-existing at base** | Present in the base failure set. `product_registry().transport_factory("http")` returns `None`. |
| `engine::registry::tests::workload_resources_fail_required_and_forbidden_presence_before_transport_prepare` | **pre-existing at base** | Present in the base failure set. `unknown variant \`acme_remote\`` — `AuthoredRunSpecV2`'s `transport.type` still decodes through a closed serde enum. This contradicts the plugin-ABI premise of the parent merge and warrants its own investigation on the root branch (review S4). |
| `engine::workers_characterization::tests::global_hop_paces_true_aggregate_rate` | **load-flaky** | Re-run in isolation in the fix worktree: **exit 0, passed in 1.40s**. It failed only inside the 489s full-suite run under core contention. Corroborating: the base run flaked a *different* member of the same `workers_characterization` module (`user_centric_workers_gt_1_global_data_matches_single_thread`) while `global_hop_paces_true_aggregate_rate` also failed there — the module is timing-sensitive under load, and which member trips varies run to run. The reviewer's independent run flaked a third variant (`user_centric_workers_gt_1_thread_per_core_data_matches_single_thread`). |

Net: **zero of the six is attributable to the fix.** Four are pre-existing at base
(two of those genuine root-branch defects, two more), two are missing untracked
fixtures, one is a load flake — and set containment proves the point independently.

## Residual root-branch debt (not this change's to fix)

Flagged so it is not lost, matching review finding S4:

- `acme_remote` closed-enum decode in `AuthoredRunSpecV2` — an unclosed gap in the
  plugin ABI work the parent merge claims to have delivered.
- `product_registry().transport_factory("http")` returning `None`.
- `velo_stream_large_artifact_round_trips_with_bounded_memory` RSS bound.
- 212 clippy errors on `aiperf-runtime` under `-D warnings`.
- `workers_characterization` is load-sensitive and flakes under a loaded full-suite run.
- The `pinchbench-*` recording fixtures are untracked, making two engine tests
  unrunnable in any worktree. Either track them or gate the tests on their presence.

## Process note

The failure here was not missing test coverage — three independent tests fired on the
regression. It was a red gate not run, and a green claimed that the branch did not
have. The remedy is procedural: run **both** test invocations before asserting a
verification result, and record exit codes rather than adjectives.
