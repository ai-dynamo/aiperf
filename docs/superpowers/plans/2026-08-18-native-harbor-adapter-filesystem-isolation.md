# Native Harbor adapter filesystem isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a sealed NativeGraph environment adapter from mutating the artifacts the independent verifier will score.

**Architecture:** Replace the rollout adapter's `docker exec` child with a labelled, task-owned adapter container attached over strict stdin/stdout. The task container keeps the sole mutable worktree; the adapter container has no task-worktree mount and is reaped before artifact collection and verification.

**Tech Stack:** Rust 2024, Tokio, Docker CLI, strict JSONL adapter protocol, `aiperf-runtime`, `aiperf-e2e-tests`.

**Spec:** `docs/superpowers/specs/2026-08-18-native-harbor-adapter-filesystem-isolation-design.md`

## Global Constraints

- Apply the isolated-container path only to sealed NativeGraph rollout environment adapters; legacy starts retain their existing path.
- Preserve exact task-minted authorization, declared argv, empty environment, `no-network`, finite deadlines, full immutable ownership labels, and descriptor-only evidence.
- Never mount the mutable task worktree into the adapter container or expose Docker, model, secret, task-container, or verifier authority to it.
- Fail closed and skip verification on adapter start, protocol, identity, or cleanup failure.
- Commit forward-only and stage complete files only.

---

### Task 1: Capture the artifact-mutation product RED

**Files:**
- Modify: `rust/e2e-tests/tests/test_harbor_native_graph_rollout.rs`
- Modify only if required: `rust/e2e-tests/tests/common/mod.rs`

**Interfaces:**
- Consumes: `write_rollout_task`, `AIPerfHarness`, and the existing selected-action environment adapter fixture.
- Produces: `environment_adapter_cannot_mutate_declared_task_artifacts_after_terminal`.

- [ ] **Step 1: Write the failing Docker E2E**

Make the terminal-transition adapter immediately execute `printf north > /work/result.txt`. Configure the selected policy to produce `south`; verifier scores `south` as `0.75` and `north` as `0.25`.

```rust
assert_eq!(result.reward(), 0.75);
assert_eq!(result.execution(), EpisodeExecution::Completed);
assert_eq!(policy_calls.load(Ordering::Relaxed), 2);
```

- [ ] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
cd rust
AIPERF_E2E_BIN=$PWD/target/release/aiperf cargo test -p aiperf-e2e-tests --test test_harbor_native_graph_rollout environment_adapter_cannot_mutate_declared_task_artifacts_after_terminal -- --ignored --exact --nocapture
```

Expected: verifier sees the late `north` write and returns `0.25`.

- [ ] **Step 3: Commit the whole RED test file**

```bash
git add rust/e2e-tests/tests/test_harbor_native_graph_rollout.rs
git commit -m "test(eval): expose Harbor adapter artifact mutation"
```

### Task 2: Add an isolated, ownership-safe adapter-container spawner

**Files:**
- Modify: `rust/runtime/src/eval/execution/docker_process.rs:2823-2975`
- Modify: `rust/runtime/src/eval/execution/docker_process.rs:4412-4495`
- Test: `rust/runtime/tests/harbor_docker_runtime.rs`

**Interfaces:**
- Consumes: `DockerAdapterSpawnerRequest`, `NativeGraphAdapterAuthorization`, `AdapterSpawnRequest`, and exact ownership labels.
- Produces: `DockerCliIsolatedAdapterSpawner` and `DockerCliIsolatedAdapterLease` with a full immutable `adapter_container_id`.

- [ ] **Step 1: Write focused RED tests**

Use the fake `DockerRuntime` to inspect the new create request:

```rust
assert!(args.windows(2).any(|pair| pair == ["--network", "none"]));
assert!(!args.iter().any(|arg| arg.contains(":/work")));
assert!(args.iter().any(|arg| arg == "aiperf.adapter-role=environment"));
```

Then return missing or foreign ownership labels and assert no attach, kill, or verifier call occurs.

- [ ] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime isolated_adapter_ -- --nocapture
```

Expected: current code uses `docker exec -i` in the task container and has no isolated create request.

- [ ] **Step 3: Implement the smallest dedicated builder**

Add this private Docker boundary; do not reuse `ContainerWorkspace`:

```rust
fn create_isolated_adapter_container(
    runtime: &dyn DockerRuntime,
    name: &str,
    image: &str,
    labels: &BTreeMap<String, String>,
    resources: Option<&ResourceLimits>,
    deadline: Duration,
) -> Result<(), EvalExecutionError>
```

Build `docker create --network none` with exact labels, the role label, optional resource caps, `image`, and `sleep infinity`. Reject any request containing a workspace mount, workdir, or environment. Resolve the complete labelled ID before start/attach. Make terminate/fence target only the stored adapter ID.

- [ ] **Step 4: Verify GREEN and commit**

Run the Task 2 test command, then:

```bash
git add rust/runtime/src/eval/execution/docker_process.rs rust/runtime/tests/harbor_docker_runtime.rs
git commit -m "feat(eval): isolate Harbor environment adapters"
```

### Task 3: Change sealed rollout cleanup order

**Files:**
- Modify: `rust/runtime/src/eval/execution/docker_process.rs:350-475`
- Modify: `rust/runtime/src/eval/execution/native_graph_episode.rs:215-260`
- Test: `rust/runtime/tests/native_graph_scored_episode.rs`
- Test: `rust/runtime/tests/harbor_docker_runtime.rs`

**Interfaces:**
- Consumes: `NativeGraphLeaseRolloutStart`, Docker rollout lease, and the isolated spawner from Task 2.
- Produces: a rollout-only cleanup capability that reaps the adapter before collection while leaving the task container running.

- [ ] **Step 1: Write lifecycle RED tests**

Record trusted rollout events and assert:

```rust
assert!(adapter_kill < artifact_collection);
assert!(artifact_collection < verifier);
assert!(verifier < task_container_remove);
```

For callback/protocol failure assert one adapter kill, no collection, no verifier, and task reverse cleanup.

- [ ] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_scored_episode docker_rollout_only_episode_ -- --nocapture
```

Expected: callback lifecycle currently verifies before adapter cleanup.

- [ ] **Step 3: Implement an explicit lease capability**

Add:

```rust
fn reaps_adapter_before_artifact_lifecycle(&self) -> bool;
```

The isolated Docker rollout lease returns `true`; legacy leases return `false`. In `run_native_graph_episode_callback`, reap successful `true` leases after callback success and before `after_callback`; retain immediate callback-error cleanup. Do not condition on package names.

- [ ] **Step 4: Verify GREEN and commit**

Run the Task 3 test command and the Task 2 tests, then:

```bash
git add rust/runtime/src/eval/execution/docker_process.rs rust/runtime/src/eval/execution/native_graph_episode.rs rust/runtime/tests/native_graph_scored_episode.rs rust/runtime/tests/harbor_docker_runtime.rs
git commit -m "fix(eval): reap isolated Harbor adapters before verification"
```

### Task 4: Require image-resident adapter executables and close E2E

**Files:**
- Modify: `rust/runtime/src/eval/native_graph/package.rs`
- Modify: `rust/runtime/tests/native_graph_package.rs`
- Modify: `rust/e2e-tests/tests/test_harbor_native_graph_rollout.rs`
- Modify: `docs/specs/native-harbor-agentic-benchmarking.md`
- Modify only if applicable: `llms.txt`

**Interfaces:**
- Consumes: immutable selected environment adapter argv and isolated Docker start.
- Produces: preflight refusal for task-worktree adapters and current documentation of filesystem isolation.

- [ ] **Step 1: Write package RED**

Add a rollout fixture whose environment adapter executable is `/work/adapter.sh` and assert preflight returns `NativeGraphPackageError::InvalidAdapterExecutableLocation`. Keep `/usr/local/bin/environment-adapter` valid.

- [ ] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_package image_resident_environment_adapter -- --nocapture
```

Expected: importer currently accepts the worktree executable.

- [ ] **Step 3: Add immutable location validation**

Validate the selected environment adapter before provisioning:

```rust
if path.starts_with("/work/") || !path.starts_with('/') {
    return Err(NativeGraphPackageError::InvalidAdapterExecutableLocation(selector));
}
```

Keep raw source paths out of public errors. Do not introduce an adapter-image registry.

- [ ] **Step 4: Rebuild the CLI and verify product GREEN**

Run:

```bash
source .venv/bin/activate
cd rust
RUSTC_WRAPPER= cargo build -p aiperf-cli --release
AIPERF_E2E_BIN=$PWD/target/release/aiperf cargo test -p aiperf-e2e-tests --test test_harbor_native_graph_rollout -- --ignored --nocapture
```

Expected: mutation regression scores `0.75`; malformed policy and protocol failures still skip verification; egress isolation and selected-action tests stay green.

- [ ] **Step 5: Update current-truth docs and commit**

Describe the environment adapter as an isolated task-owned container, not a process in the mutable task worktree. Run:

```bash
/usr/bin/python3 tools/check_agent_files_sync.py
/usr/bin/python3 tools/check_docs_current.py
```

Then commit complete files:

```bash
git add rust/runtime/src/eval/native_graph/package.rs rust/runtime/tests/native_graph_package.rs rust/e2e-tests/tests/test_harbor_native_graph_rollout.rs docs/specs/native-harbor-agentic-benchmarking.md llms.txt
git commit -m "feat(eval): enforce Harbor adapter filesystem isolation"
```

### Task 5: Final verification and strict review

**Files:**
- Verify only: all Task 1–4 files.

**Interfaces:**
- Consumes: the complete isolated rollout path.
- Produces: reproducible proof that an adapter cannot alter verifier artifacts after terminal.

- [ ] **Step 1: Run focused runtime suites**

```bash
source .venv/bin/activate
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_scored_episode -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_rl_scored -- --nocapture
```

- [ ] **Step 2: Run formatting and diff checks**

```bash
source .venv/bin/activate
cd rust
cargo fmt --all --check
git diff --check
```

- [ ] **Step 3: Require a fresh strict review**

Review the full diff for task-container kills, raw filesystem/path leaks, unlabelled Docker operations, credential propagation, and lifecycle ordering. Do not claim completion until the mutation E2E, failure paths, formatting, and review are green.
