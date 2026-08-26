<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# E10 Docker Verifier Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route ordinary single-step, NativeGraph, and explicit multi-step standard Docker verification through one transaction that always clears shared verifier files and preserves separate-verifier isolation.

**Architecture:** Add one private `DockerVerifierTransaction` beside `DockerStepSession` in `docker_process.rs`. Callers retain artifact collection and outer reverse container cleanup; the transaction owns the single verifier deadline, optional separate workspace/container setup, reserved-path preparation, selected test-tree copy, verifier execution, reward reading, and shared-only cleanup/error combination.

**Tech Stack:** Rust 2024, `aiperf-runtime`, injected `DockerRuntime`, `Deadline`, current recording-runtime integration tests, Cargo targets under `/mnt/4tb`.

**Spec:** `docs/specs/2026-08-26-e10-docker-verifier-transaction.md`

## Global Constraints

- Change only standard non-Compose Docker execution; do not alter Compose or legacy package verification.
- Keep the transaction private to `eval::execution::docker_process` and add no dependency.
- Preserve one absolute verifier `Deadline` across every verifier operation; never replace remaining time with a fresh authored timeout.
- Register a separate verifier container in the caller-owned `Vec<String>` before create so failed or uncertain creation remains covered by reverse cleanup.
- Run `clear_verifier_files` exactly once after every shared outcome, including setup, command, timeout, and reward-read failures; never run it for `Separate`.
- Combine shared primary and cleanup failures through `combine_primary_and_cleanup`; preserve a primary error when cleanup succeeds and make cleanup-only failure terminal.
- Preserve phase environment, secrets, user, workdir, network lease, artifact validation/staging, selected `verifier_test_root`, reward parsing, callback ordering, and adapter ownership.
- Demonstrate RED and GREEN with a unique `CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E10` and `CARGO_BUILD_JOBS=1`.
- Do not merge until an independent agent runs the `graham-code-review` skill and records PASS with no open blocker.

---

### Task 1: Pin the divergent transaction behavior

**Files:**

- Modify: `rust/runtime/tests/harbor_docker_runtime.rs` (`StepRecordingRuntime`, NativeGraph callback fixtures, and Docker verifier lifecycle tests)

**Interfaces:**

- Consumes: `DockerProcessSandbox::{execute_with_runtime, execute_native_graph_with_runtime, execute_multi_step_with_runtime}` and observable `DockerRuntime` requests.
- Produces: six `e10_` regression tests and a recording runtime capable of authorizing a non-rollout NativeGraph package without changing its default behavior.

- [ ] **Step 1: Extend the recording runtime only enough to exercise NativeGraph**

Add `native_graph_profile: Option<ProviderProfile>` to `StepRecordingRuntime`, initialize it through `Default`, and add this builder:

```rust
fn with_native_graph_profile(mut self, profile: ProviderProfile) -> Self {
    self.native_graph_profile = Some(profile);
    self
}
```

In `capabilities`, add `ModelEndpointIsolation` only when the field is `Some`. Implement `native_graph_provider_profile` by cloning the configured profile or returning `UnsupportedEnforcement("model endpoint isolation")`, and implement `native_graph_model_secret_environment` as an empty map only when authorized. Do not add adapter authority: `native_graph_task_root` is a non-rollout fixture and the callback does not start one.

Add a small E10 callback borrowing `&RefCell<Vec<String>>`; its `run` method asserts the lease is authorized/acquired, appends `native-graph`, and returns success. This avoids changing the existing event field to `Rc` or perturbing other fixtures.

- [ ] **Step 2: Add the ordinary-path RED tests**

Add these exact tests, using `standard_task_root`, `StepRecordingRuntime`, and event positions rather than wall-clock sleeps:

```rust
#[test]
fn e10_single_step_shared_success_cleans_reserved_paths_and_preserves_reward()

#[test]
fn e10_single_step_shared_failure_combines_cleanup_failure()

#[test]
fn e10_single_step_separate_has_no_post_verifier_shared_cleanup()
```

The success case must assert `reward.metrics["reward"] == 1.0`, exactly two reset events (prepare before test copy and cleanup after verifier/reward), and outer container removal after cleanup. The combined-failure case uses `StepRecordingRuntime::failing_shared_verifier_cleanup()` and requires `ContainerTeardown.reason` to contain both `verifier 1 failed` and `reset 2 failed`, with exactly two reset attempts. The separate case authors `environment_mode = "separate"`, asserts two distinct container creates/removes and fresh verifier workspace staging, and asserts exactly one reset event (reserved-path preparation, not a post-verifier shared cleanup).

- [ ] **Step 3: Add the NativeGraph RED tests**

Add these exact current-thread tests, using `native_graph_task_root`, an explicit `NoAdapterEgress` profile, and the E10 callback:

```rust
#[tokio::test(flavor = "current_thread")]
async fn e10_native_graph_shared_failure_cleans_after_callback()

#[tokio::test(flavor = "current_thread")]
async fn e10_native_graph_shared_failure_combines_cleanup_failure()

#[tokio::test(flavor = "current_thread")]
async fn e10_native_graph_separate_preserves_order_and_skips_shared_cleanup()
```

For the two failure cases configure `StepFailure::Verifier(1)`; configure reset call 2 to fail only in the combined-failure case. Assert deterministic order `native-graph < verifier:1 < second reset < first remove`, and require the joined diagnostic in the cleanup-failure case. For the separate case append `[verifier]\nenvironment_mode = "separate"` to the fixture manifest before import, then assert `native-graph < verifier:1 < remove`, two container creates/removes, fresh verifier workspace staging, and one preparation reset only.

- [ ] **Step 4: Run the focused tests against the pre-change implementation and record RED**

Run from the E10 worktree:

```bash
source .venv/bin/activate
CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E10 \
  cargo test -p aiperf-runtime --test harbor_docker_runtime e10_ -- --nocapture
```

Expected RED: the command exits nonzero because ordinary and NativeGraph shared success/failure paths emit only the preparation reset; the cleanup ordering and joined-error assertions fail. The separate cases may already pass and serve as preservation controls. Save the exact failing test names/output in the E10 SDD receipt; do not weaken assertions to manufacture RED.

---

### Task 2: Replace the three verifier lifecycles with one transaction

**Files:**

- Modify: `rust/runtime/src/eval/execution/docker_process.rs:1264-1438` (ordinary handoff)
- Modify: `rust/runtime/src/eval/execution/docker_process.rs:1605-1784` (NativeGraph post-callback handoff)
- Modify: `rust/runtime/src/eval/execution/docker_process.rs:2792-3047` (`DockerStepSession` and shared helper boundary)

**Interfaces:**

- Consumes: one `BenchmarkStepPlan`, caller-owned artifact snapshot/path, task/image/container identity, `DockerRuntime`, `Clock`, `SecretProvider`, and caller-owned container registry.
- Produces: `DockerVerifierTransaction::run(&mut self, step: &BenchmarkStepPlan, artifacts: &[(String, ArtifactDigest)]) -> Result<RewardDocument, EvalExecutionError>`.

- [ ] **Step 1: Define the private transaction input boundary**

Place this private shape immediately before `DockerStepSession` (field lifetimes may be split if required by the borrow checker, but do not broaden visibility):

```rust
struct DockerVerifierTransaction<'a> {
    clock: &'a Rc<dyn Clock>,
    runtime: &'a dyn DockerRuntime,
    recipe: &'a HarborSandboxRecipe,
    source_root: &'a Path,
    image: &'a str,
    agent_container: &'a str,
    verifier_container: String,
    secrets: &'a dyn SecretProvider,
    containers: &'a mut Vec<String>,
    artifact_collection: &'a Path,
}

impl DockerVerifierTransaction<'_> {
    fn run(
        &mut self,
        step: &BenchmarkStepPlan,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<RewardDocument, EvalExecutionError>;
}
```

The ordinary and NativeGraph callers pass `{agent_container}-verifier`; multi-step passes `{agent_container}-verifier-{step.name()}`. `step` is authoritative for `verifier()`, `artifacts()`, and `verifier_test_root()` so selected-tree behavior cannot diverge.

- [ ] **Step 2: Move separate verifier setup into `run` without semantic changes**

Create one optional verifier `Deadline` before staging. For `Separate`, create a `0755` temporary workspace, transfer the frozen artifact snapshot with the same deadline, validate explicit/effective verifier workdirs, and push `verifier_container.clone()` into `containers` before `create_planned_container`. Preserve create/start deadline consumption, image-workdir inspection, artifact transfer, user workdir preparation, inherited healthcheck, and network leases. Retain the temporary workspace until reward acquisition completes.

For `Shared`, select `agent_container` and do not create a workspace/container. Do not move outer `remove_containers_with_deadline` into this helper.

- [ ] **Step 3: Centralize setup, execution, reward reading, and shared cleanup**

Within one `outcome` closure:

1. call `prepare_verifier_files_with_deadline`;
2. copy `source_root.join(step.verifier_test_root())/.` to `{verifier_name}:/tests` with remaining time;
3. prepare the verifier workdir only for shared mode (separate mode already did so);
4. call `execute_planned_phase_with_deadline` with `[/bin/sh, /tests/test.sh]`;
5. create the reward workspace and call `read_reward_with_runtime` with the same `Deadline`.

After the closure, compute cleanup exactly once:

```rust
let cleanup = if verifier.mode() == VerifierMode::Shared {
    clear_verifier_files(
        self.runtime,
        &verifier_name,
        verifier_network,
        verifier_cleanup_deadline(&deadline),
    )
} else {
    Ok(())
};
combine_primary_and_cleanup(outcome, cleanup, verifier_name)
```

This cleanup must be outside the `?`-using outcome closure so command, timeout, and reward-read failures cannot bypass it. Do not run this cleanup for a separate verifier; its registered container is removed by the caller's reverse cleanup.

- [ ] **Step 4: Route all three callers through the helper and delete inline copies**

In ordinary and NativeGraph paths, bind the implicit step once with the existing `InvalidRecipe("Docker benchmark step")` diagnostic, collect artifacts as today, construct the transaction, call `run`, then build `LocalExecutionResult` with the unchanged package verifier identity.

In `DockerStepSession::run_verifier`, require the retained `artifact_collection` with the existing `InvalidRecipe("multi-step artifact collection")` diagnostic, construct the transaction, call `run`, and clear `self.artifact_collection` after the call regardless of success. Delete all three inline verifier lifecycles; `rg` should leave one standard non-Compose implementation of `prepare_verifier_files_with_deadline`/test copy/execute/reward/clear sequencing.

- [ ] **Step 5: Run the focused E10 tests and verify GREEN**

```bash
source .venv/bin/activate
CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E10 \
  cargo test -p aiperf-runtime --test harbor_docker_runtime e10_ -- --nocapture
```

Expected GREEN: all six `e10_` tests pass; ordinary and NativeGraph shared paths show two resets, joined failures retain both diagnostics, and separate paths show only their preparation reset.

- [ ] **Step 6: Run the complete deterministic Docker runtime suite**

```bash
source .venv/bin/activate
CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E10 \
  cargo test -p aiperf-runtime --test harbor_docker_runtime
```

Expected: every non-ignored `harbor_docker_runtime` test passes, including `shared_verifier_resets_tests_before_each_selected_tree_copy`, `shared_verifier_failure_reports_its_cleanup_error`, `separate_verifiers_use_fresh_staging_and_artifact_snapshots`, and `single_step_separate_verifier_uses_one_absolute_deadline_for_setup_and_reward`.

- [ ] **Step 7: Verify formatting and commit the feature**

```bash
cargo fmt --check
git diff --check
git add rust/runtime/src/eval/execution/docker_process.rs rust/runtime/tests/harbor_docker_runtime.rs
git commit -m "fix(eval): unify Docker verifier transactions"
```

If repository-wide `cargo fmt --check` reports unrelated baseline drift, run `rustfmt` only on the two E10 Rust files, rerun `git diff --check`, and record the exact unrelated paths; do not absorb them into E10.

---

### Task 3: Independent Graham gate and integration receipt

**Files:**

- Create: `.superpowers/sdd/2026-08-26-rust-e10-docker-verifier-remediation/task-e10-independent-graham-review.md`
- Modify after approval/integration: `docs/rust-code-smell-remediation-tracker.md` (E10 row and progress log)

**Interfaces:**

- Consumes: committed E10 diff, dedicated spec, RED/GREEN receipt, and complete `harbor_docker_runtime` output.
- Produces: independent Graham PASS with no unresolved blocker and an integration/tracker commit.

- [ ] **Step 1: Run an independent Graham review**

Dispatch an agent that did not author E10 and require the `graham-code-review` skill. The review must inspect deadline reuse, container registration before create, no cleanup bypass on `?`, primary+cleanup error precedence, no shared cleanup for separate mode, NativeGraph callback/adapter ordering, minimal diff surface, and absence of production `unwrap`/`expect`. Record file/line findings and a PASS/CHANGES REQUESTED verdict in the receipt path above.

- [ ] **Step 2: Resolve every review finding through another RED-to-GREEN loop**

For each blocker, use `superpowers:receiving-code-review` and `superpowers:systematic-debugging`, add or strengthen one deterministic regression, rerun the focused `e10_` command and the complete `harbor_docker_runtime` suite, commit the repair, and request a fresh independent Graham verdict. Do not integrate with an open blocker.

- [ ] **Step 3: Integrate only the approved commits and update the tracker**

After confirming the integration worktree is clean on the exact touched paths, cherry-pick the approved E10 commits. Change E10 to `Complete` and record the implementation commit(s), RED/GREEN commands, complete suite result, and Graham receipt/PASS in `docs/rust-code-smell-remediation-tracker.md`; commit that documentation separately as `docs: record E10 remediation receipt`.
