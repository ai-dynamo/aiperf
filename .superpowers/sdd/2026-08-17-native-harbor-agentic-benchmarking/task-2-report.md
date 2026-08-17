# Task 2 report — early result contract and bounded matrix scheduler

## Status

The Task-2 implementation is ready for fresh independent Graham review. No
Task-2 commit has been created.

## Base

- `55ef618a50` (`feat(eval): resolve native graph packages`)
- `8dd010447f` (`docs(eval): record native graph task verification`)

## Delivered contract

- Each parsed suite owns one `Rc<ImportedTask>` snapshot per imported task.
  Trial axes, repetitions, and resolved attempts retain only cheap local
  handles to that immutable package table; no resolved trial retains a
  caller-controlled `HarborSource` or a detached package plan.
- `EpisodeAssignmentId` is stable across suite-run namespaces. `AttemptId`
  includes the caller-owned `SuiteRunId`, so a rerun appends a distinct attempt
  without changing its assignment identity.
- The local scheduler holds one worker-local `Rc<LocalSchedulerState>`.
  `RefCell<ResourcePools>` owns episode slots, CPU, memory, and per-binding
  weights across all concurrent `run` calls. RAII leases release every weight
  on success, runner error, scheduler cancellation, and future cancellation;
  a worker-local `watch` notification wakes blocked suites.
- Model-binding resource names in an authored task resolve only through that
  task's imported binding table. Every resolved trial retains its selected
  binding id, complete binding identity digest, and a capacity key derived
  from the immutable task reference plus that full binding identity. A textual
  binding name reused by different task snapshots is rejected explicitly, so
  URLs, transports, tokenizer policy, and generation defaults cannot be
  conflated by a scheduler capacity.
- Resource requests and paired-factor maps are reference-counted per authored
  axis. Admission borrows the request and allocates an `Rc` lease handle only
  after a successful pool acquisition; unsuccessful admission does not clone a
  request map.
- Output slots are allocated in manifest order before admission. Completion
  order cannot reorder results, and a runner cannot substitute another trial or
  attempt identity.
- Result axes are exact and orthogonal: integrity, execution including
  `Truncated`, score `Verified`/`Unavailable`, and comparability
  `Scored`/`Unscored`. Aggregation includes valid scored verified results only;
  valid failed zero scores remain in the denominator and retained unscored
  results do not.
- `parse_native_graph_suite_toml` is byte-capped and uses deny-unknown-field
  TOML DTOs. It bounds tasks, axis cardinality, seeds, repetitions, paired
  factors, expansion, and weights before source acquisition or output-vector
  allocation. Its injected importer resolves ordered local/pinned-Git/registry
  references once, checks the exact task id/digest, selects only declared model
  binding ids, and produces snapshot-backed trial specs. Paired factors are
  carried into the suite identity. `NativeGraphSuiteDefinition` identity also
  includes parallelism, CPU, memory, and every per-model capacity.

## RED evidence

All RED commands ran on the host with the inherited Rust wrapper unchanged.

```text
cargo test -p aiperf-runtime --test native_graph_suite -- --nocapture
error[E0432]: unresolved import `SuiteRunId`
error[E0599]: no function or associated item named `from_imported`
error[E0599]: this method takes 0 arguments but 1 argument was supplied
```

The P0 snapshot/attempt test mutated the caller's task directory after import;
it required a snapshot-backed package and distinct assignment/attempt
identities.

```text
cargo test -p aiperf-runtime --test native_graph_matrix -- --nocapture
resource_pools_are_shared_across_concurrent_suite_runs ... FAILED
  left: 2, right: 1
model_binding_weights_are_shared_across_concurrent_suite_runs ... FAILED
  left: 2, right: 1
runner_error_releases_a_global_resource_lease ... FAILED
  left: ["first-started", "second-started", "first-failed"]
 right: ["first-started", "first-failed", "second-started"]
```

Those scheduler RED tests proved the original per-run CPU/model pools were not
global to the scheduler and that the second concurrent run could start before
the first error released its lease.

```text
cargo test -p aiperf-runtime --test native_graph_matrix -- --nocapture
error[E0599]: no variant named `Scored` found for `EpisodeComparability`
error[E0599]: no variant named `Truncated` found for `EpisodeExecution`
error[E0599]: no variant named `Unavailable` found for `EpisodeScoreState`
```

```text
cargo test -p aiperf-runtime --test native_graph_suite -- --nocapture
error[E0432]: unresolved import `parse_native_graph_suite_toml`
error[E0599]: no variant named `TaskReferenceMismatch` found for `SuiteError`
error[E0599]: no variant named `TrialExpansionLimitExceeded` found for `SuiteError`
```

## GREEN evidence

Every Cargo command below ran on the host with the inherited sccache wrapper;
none disabled or replaced `RUSTC_WRAPPER`.

```text
cargo test -p aiperf-runtime --test native_graph_suite -- --nocapture
native_graph_suite: 3 passed

cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix -- --nocapture
native_graph_matrix: 10 passed
native_graph_suite: 5 passed

cargo fmt
cargo fmt --check
git diff --check

cargo clippy -p aiperf-runtime --test native_graph_suite --test native_graph_matrix
exit 0
```

The strict `-D warnings` clippy attempt found two local issues, both corrected:
the nested finite-score check in `result.rs` and a useless `into_iter` in
`suite.rs`. It also reports pre-existing whole-runtime warnings in
`graph/workload.rs`, `agentx`, `adaptive_core`, `docker_process.rs`,
`graph/driver.rs`, and `rng/configured.rs`; those are outside Task 2 and the
Docker path is expressly excluded. The non-promoted focused clippy command
above exits zero and emits no NativeGraph warnings.

## Final-review repair evidence

The final-review tests were written before the selected-binding and compact
ownership implementation. The required host RED receipt was:

```text
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix -- --nocapture
error[E0599]: no method named `selected_model_binding` found for struct `ResolvedEpisodeTrial`
error[E0599]: no variant named `CrossTaskModelBindingAlias` found for enum `SuiteError`
error[E0599]: no method named `identity_digest` found for struct `NativeGraphSuiteDefinition`
```

The final host GREEN receipt was:

```text
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix -- --nocapture
native_graph_matrix: 10 passed
native_graph_suite: 9 passed

cargo fmt --check
exit 0

cargo clippy -p aiperf-runtime --test native_graph_suite --test native_graph_matrix
exit 0; only pre-existing whole-runtime warnings, no NativeGraph warning

git diff --check
exit 0
```

The 9 suite tests include a two-task strict manifest with ordered graph/model/
policy axes and weighted CPU, memory, and model limits; exact task-local
binding rejection; same-name/different-runtime alias rejection; each
URL/transport/tokenizer/generation identity sensitivity; all scheduler-limit
identity sensitivities; and a 10,000-repetition compact expansion. The 10
matrix tests observe inverted completion before manifest-order output, prove
global episode-slot/CPU/memory/model pools across concurrent public scheduler
calls, release leases on errors and cancellation, and assert the exact runner
failure and result-identity mismatch errors.

## Acceptance-review P1 repair evidence

The final acceptance-review tests were first observed RED on the host:

```text
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix -- --nocapture
error[E0599]: no method named `resolve` found for `NativeGraphSuiteDefinition`
error[E0599]: no variant named `ForeignResourceCapacityKey` found for `SuiteError`

cargo test -p aiperf-runtime --test native_graph_matrix pending_suite_retries_when_another_suite_releases_capacity -- --nocapture
pending_suite_retries_when_another_suite_releases_capacity ... FAILED
the pending suite must retry after the other suite releases capacity
```

The latter used one blocked suite holding a global resource, then made a second
suite retain a long-running admitted episode and one pending episode. It only
released the blocker after the pending suite's long episode had started, so the
old active-only wait could not admit the pending work before the timeout.

The repaired host GREEN receipt was:

```text
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix -- --nocapture
native_graph_matrix: 11 passed
native_graph_suite: 11 passed

cargo fmt --check
exit 0

cargo clippy -p aiperf-runtime --test native_graph_suite --test native_graph_matrix
exit 0; only pre-existing whole-runtime warnings, no NativeGraph warning

git diff --check
exit 0
```

When a local suite has both active and pending work, the scheduler now selects
between its next active completion and the worker-local global availability
watch before trying admission again. Programmatic `SuiteTrialSpec` construction
checks every requested `ModelCapacityKey` against every full imported binding
key for that task. `NativeGraphSuiteDefinition::resolve` carries the
definition identity into the resolved-suite and assignment identities, while
the existing manifest-only resolver remains available for programmatic suites
without scheduler limits.

## Fresh re-review repair evidence

The first independent re-review required another TDD correction round. The
new disjoint-binding contract first failed on the host:

```text
cargo test -p aiperf-runtime --test native_graph_suite -- --nocapture
strict_suite_task_resource_binding_must_exist_in_that_task_snapshot ... FAILED
called `Result::unwrap_err()` on an `Ok` value: NativeGraphSuiteDefinition
```

Before the repair, the second task declared only `secondary` but its resource
request named `primary`; global lookup silently produced a second-task lease
containing `ModelBindingId("primary")`. The resolver now rejects that exact
cross-task substitution.

The fresh tests also record completion index `1` before manifest index `0`,
then prove manifest-order output; exercise global episode-slot, CPU, memory,
and model pools through `run_resolved_suite` for single- and multi-element
suites; use a two-task ordered strict TOML fixture with graph/model/policy
axes and task weights; reject an over-byte-cap document; and prove paired
factors and resource weights change suite identity. Runner-originated failure
is now `MatrixError::RunnerExecutionFailed`, distinct from scheduler admission.
`FuturesUnordered` stores the direct local futures, without `boxed_local`.

## Explicit exclusions

- No changes to `rust/runtime/src/eval/execution/docker_process.rs`.
- No changes to existing P0 or Compose SDD ledgers.
- No Docker execution, endpoint execution, graph execution, evaluator, CLI, or
  registry work; those belong to later NativeGraph plan tasks.
