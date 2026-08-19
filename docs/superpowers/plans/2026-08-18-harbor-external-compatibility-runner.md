# Harbor External Compatibility Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an `externally_driven` Harbor task end to end while preserving Rust-owned lifecycle, verification, scoring, and a non-native fidelity classification.

**Architecture:** The existing exact external-driver selector becomes a two-stage prepared driver and a capability-limited supervised Driver session. Docker provides a distinct external authorization and one-step transaction; the existing episode runner and evaluator consume a sealed compatibility supplement rather than an untrusted raw terminal payload. The CLI enables only the single-task path and preserves refusal for unsupported suite input.

**Tech Stack:** Rust 2024, Tokio `LocalSet`, existing NativeGraph adapter protocol, Docker runtime/sandbox seams, BLAKE3 identity digests, strict serde DTOs, Harbor evaluator and mock-server integration tests.

**Spec:** `docs/superpowers/specs/2026-08-18-harbor-external-compatibility-runner-design.md`

## Global Constraints

- Keep the existing `native_graph` profile behavior and exact fidelity unchanged.
- An external driver receives only its declared argv, an empty sanitized environment, a bounded deadline, and a Driver-role protocol; it receives no model secret, model runtime, NativeGraph authorization, or direct result authority.
- Retain no raw terminal JSON, stdout, stderr, traffic, prompt, tool, path, artifact handle, or secret in public errors, metrics, verifier inputs, or frozen evidence.
- Use BLAKE3 identities for sealed package/trial/attempt/capture facts; all retained bounds are explicit.
- Refuse unknown/mismatched factory IDs, Compose, multi-step, incompatible protocol/runtime, missing terminal receipt, timeout, invalid receipt, and unsupported suite execution before claiming success.
- A failed or absent driver terminal receipt skips verification and cleanup cancels/reaps exactly once.
- Use whole-file staging and forward-only commits. Do not stage unrelated dirty files or rewrite history.

---

## File structure

| File | Responsibility |
|---|---|
| `rust/runtime/src/eval/native_graph/factories.rs` | Exact external factory selection, prepare/run driver interfaces, and sealed preparation errors. |
| `rust/runtime/src/eval/native_graph/capture.rs` | Package-bound, digest-only compatibility observation and terminal supplement. |
| `rust/runtime/src/eval/native_graph/completed_attempt.rs` | Seal compatibility completion facts and append lifecycle-only evidence. |
| `rust/runtime/src/eval/result.rs` | Carry explicit externally-driven fidelity without altering verifier score facts. |
| `rust/runtime/src/eval/native_graph/evaluator.rs` | Map a sealed compatibility completion to the selected Harbor evaluator result. |
| `rust/runtime/src/eval/execution/docker_runtime.rs` | Mint external-only adapter authorization and expose a distinct compatibility spawner. |
| `rust/runtime/src/eval/execution/docker_process.rs` | Run the external one-step transaction and enforce cleanup/terminal sequencing. |
| `rust/runtime/src/eval/execution/native_graph_episode.rs` | Define the sealed callback/backend driver-session handoff. |
| `rust/runtime/src/eval/native_graph/episode_runner.rs` | Add the external executor through the existing runner authority checks. |
| `rust/cli/src/eval/native_graph.rs` | Enable external single-task execution after exact preflight while retaining suite refusal. |
| `rust/runtime/tests/native_graph_capture.rs` | Pure capture, lifecycle, and fidelity contracts. |
| `rust/runtime/tests/native_graph_protocol.rs` | Driver terminal request/correlation/bounds tests. |
| `rust/runtime/tests/harbor_docker_runtime.rs` | Supervision, secret stripping, rejection, and cleanup tests. |
| `rust/runtime/tests/native_graph_scored_episode.rs` | Concrete executor-to-verifier-to-result integration tests. |
| `rust/cli/tests/eval_command.rs` | CLI preflight/no-model-runtime/refusal tests. |
| `rust/e2e-tests/tests/test_harbor_external_compatibility.rs` | Product-level mock-server/Docker end-to-end coverage. |

### Task 1: Seal pure compatibility contracts

**Files:**
- Modify: `rust/runtime/src/eval/native_graph/factories.rs:1007-1075`
- Modify: `rust/runtime/src/eval/native_graph/capture.rs:41-286`
- Modify: `rust/runtime/src/eval/native_graph/completed_attempt.rs`
- Modify: `rust/runtime/src/eval/result.rs`
- Modify: `rust/runtime/src/eval/native_graph/evaluator.rs`
- Modify: `rust/runtime/src/eval/native_graph/mod.rs`
- Modify: `rust/runtime/src/eval/mod.rs`
- Test: `rust/runtime/tests/native_graph_capture.rs`
- Test: `rust/runtime/tests/native_graph_rl_scored.rs`

**Interfaces:**
- Consumes: `NativeGraphPackagePlan`, `ResolvedEpisodeTrial`, `CapturePolicy`, `CompatibilityTerminalSupplement`, `NativeGraphAttemptAuthority`, and `EpisodeResult`.
- Produces: `PreparedExternalDriver`, `ExternalDriverSession`, `CompatibilityTerminalReceipt`, `EpisodeFidelity::ExternallyDriven(CompatibilityFidelity)`, and `NativeGraphCompletedAttempt::freeze_compatibility(...)`.

- [ ] **Step 1: Write the failing tests**

Add tests proving an external package selects only its exact factory, zero observations freeze as `Missing`, a compatibility supplement cannot attach to a native/no-rollout/foreign-trial completion, and compatibility lifecycle evidence never enters verifier evidence or changes reward/score.

```rust
assert_eq!(
    CapturePolicy::from_package(&external)?.begin_observation().freeze().fidelity(),
    CaptureFidelity::Missing
);
assert!(NativeGraphCompletedAttempt::freeze_compatibility(&authority, frozen, supplement).is_err());
assert!(result.fidelity().is_externally_driven());
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_capture -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_rl_scored compatibility -- --nocapture`

Expected: FAIL because no prepared external-driver/session contract, compatibility completion admission, or externally-driven result fidelity exists.

- [ ] **Step 3: Write minimal implementation**

Replace the inert driver shape with preparation and session boundaries:

```rust
pub trait NativeGraphExternalDriverFactory: Send + Sync {
    fn id(&self) -> &str;
    fn prepare(&self, package: &NativeGraphPackagePlan, trial: &ResolvedEpisodeTrial)
        -> Result<Box<dyn PreparedExternalDriver>, NativeGraphFactoryError>;
}

#[async_trait(?Send)]
pub trait PreparedExternalDriver {
    async fn run(&mut self, session: &mut dyn ExternalDriverSession)
        -> Result<CompatibilityTerminalReceipt, NativeGraphFactoryError>;
}
```

Make `CompatibilityTerminalReceipt` private-field, bounded, and digest-backed. Validate external profile, package/trial/attempt/capture identity before appending one `EvidenceKind::Compatibility` lifecycle event. Add explicit `EpisodeFidelity`; legacy construction stays non-exact.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_capture -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_rl_scored -- --nocapture`

Expected: PASS, including lifecycle-only evidence and unchanged verifier reward/score assertions.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/native_graph/{factories.rs,capture.rs,completed_attempt.rs,evaluator.rs,mod.rs} rust/runtime/src/eval/{result.rs,mod.rs} rust/runtime/tests/{native_graph_capture.rs,native_graph_rl_scored.rs}
git commit -m "feat(eval): seal Harbor external compatibility results"
```

### Task 2: Add the Driver terminal protocol boundary

**Files:**
- Modify: `rust/runtime/src/eval/native_graph/protocol.rs`
- Modify: `rust/runtime/src/eval/native_graph/supervision.rs`
- Modify: `rust/runtime/src/eval/native_graph/factories.rs`
- Test: `rust/runtime/tests/native_graph_protocol.rs`
- Test: `rust/runtime/tests/native_graph_supervision.rs`

**Interfaces:**
- Consumes: `AdapterProtocolConfig`, `AdapterRole::Driver`, `ProtocolCapability::Driver`, `ProtocolLimits`, and `PreparedExternalDriver`.
- Produces: `ExternalDriverSession::request_terminal()`, returning exactly one bounded `CompatibilityTerminalReceipt` or typed protocol error.

- [ ] **Step 1: Write the failing tests**

Add Driver-role tests for a correlated `RequestEpisodeTerminal`, exactly one candidate, wrong role/capability rejection, oversized terminal payload rejection before retained JSON, and candidate-after-settlement refusal.

```rust
assert!(session.request_terminal().await.is_err()); // no correlated candidate
assert!(session.accept_candidate(foreign_candidate).is_err());
assert!(session.accept_candidate(oversized_candidate).is_err());
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_protocol external_driver -- --nocapture`

Expected: FAIL because the current terminal message accepts raw `Value` without an external-session receipt boundary.

- [ ] **Step 3: Write minimal implementation**

Require a Driver-only protocol config, send one `HostMessage::RequestEpisodeTerminal`, validate sequence/correlation/role/capability, canonicalize bounded JSON once, and convert it directly to a private digest receipt. Keep raw `serde_json::Value` internal until conversion.

- [ ] **Step 4: Run focused protocol tests to verify they pass**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_protocol -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_supervision -- --nocapture`

Expected: PASS, including no-candidate, duplicate, wrong-role, and oversized-candidate refusal.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/native_graph/{protocol.rs,supervision.rs,factories.rs} rust/runtime/tests/{native_graph_protocol.rs,native_graph_supervision.rs}
git commit -m "feat(eval): add Harbor external driver terminal protocol"
```

### Task 3: Mint a secret-free external Docker authorization

**Files:**
- Modify: `rust/runtime/src/eval/execution/docker_runtime.rs`
- Modify: `rust/runtime/src/eval/execution/docker_process.rs`
- Modify: `rust/runtime/src/eval/native_graph/supervision.rs`
- Test: `rust/runtime/tests/harbor_docker_runtime.rs`

**Interfaces:**
- Consumes: prepared driver and exact Driver adapter, resolved external package/trial, Docker task container, and lifecycle deadlines.
- Produces: `ExternallyDrivenAdapterAuthorization` and `DockerRuntime::external_driver_spawner(...)`, which can spawn exactly one declared Driver request.

- [ ] **Step 1: Write the failing tests**

Add a strict fake runtime asserting preparation finishes before build, request argv equals manifest driver argv, request environment is empty, no secret-provider mapping occurs, and Compose/multi-step/missing-spawner paths refuse before Docker create.

```rust
assert_eq!(spawn_request.argv(), ["tools/driver.sh"]);
assert!(spawn_request.environment().is_empty());
assert_eq!(runtime.secret_provider_calls(), 0);
assert_eq!(runtime.create_calls(), 0); // incompatible plan
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime external_driver_ -- --nocapture`

Expected: FAIL because only NativeGraph adapter authorization/spawning is present.

- [ ] **Step 3: Write minimal implementation**

Add private authorization minted from immutable external package, exact driver adapter, resolved trial, task container, and bounded deadline. Its only request constructor uses declared argv and `BTreeMap::new()`. Add a separate Docker spawner that authorizes this token and never calls native model-secret or no-egress paths.

- [ ] **Step 4: Run focused Docker tests to verify they pass**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime external_driver_ -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime -- --nocapture`

Expected: PASS; Docker daemon-dependent cases remain explicitly ignored when unavailable.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/execution/{docker_runtime.rs,docker_process.rs} rust/runtime/src/eval/native_graph/supervision.rs rust/runtime/tests/harbor_docker_runtime.rs
git commit -m "feat(eval): authorize Harbor external drivers"
```

### Task 4: Execute the compatibility transaction and cleanup

**Files:**
- Modify: `rust/runtime/src/eval/execution/native_graph_episode.rs`
- Modify: `rust/runtime/src/eval/execution/docker_process.rs`
- Modify: `rust/runtime/src/eval/native_graph/episode_runner.rs`
- Test: `rust/runtime/tests/harbor_docker_runtime.rs`
- Test: `rust/runtime/tests/native_graph_scored_episode.rs`

**Interfaces:**
- Consumes: `PreparedExternalDriver`, Driver session/protocol receipt, and external Docker authorization.
- Produces: `DockerExternallyDrivenEpisodeExecutor: NativeGraphEpisodeExecutor` returning a sealed compatibility `NativeGraphCompletedAttempt`.

- [ ] **Step 1: Write the failing tests**

Add a concrete fake Docker/driver test proving `prepare → acquire → healthcheck → terminal → artifacts → verifier → cancel → reap`, and negative cases where driver error, timeout, missing candidate, or invalid candidate skip verifier but cancel/reap exactly once.

```rust
assert_eq!(events, ["prepare", "healthcheck", "terminal", "collect", "verify", "cancel", "reap"]);
assert_eq!(verifier.calls(), 0); // missing terminal candidate
assert_eq!(adapter.reap_calls(), 1);
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_scored_episode external_driver_ -- --nocapture`

Expected: FAIL because no external executor/session transaction is wired to Harbor completion.

- [ ] **Step 3: Write minimal implementation**

Add `execute_externally_driven_with_runtime` beside the native path. It performs only the approved one-step transaction, obtains a terminal receipt from the prepared driver session, invokes existing artifact collector/verifier, and freezes the Task 1 supplement. Ensure every exit uses one owner to cancel/reap once; do not alter protected uncertain-create/recovery regions.

- [ ] **Step 4: Run focused transaction tests to verify they pass**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_scored_episode -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime -- --nocapture`

Expected: PASS, with NativeGraph rollout results unchanged.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/execution/{native_graph_episode.rs,docker_process.rs} rust/runtime/src/eval/native_graph/episode_runner.rs rust/runtime/tests/{harbor_docker_runtime.rs,native_graph_scored_episode.rs}
git commit -m "feat(eval): execute Harbor external compatibility episodes"
```

### Task 5: Enable the single-task CLI path and product E2E

**Files:**
- Modify: `rust/cli/src/eval/native_graph.rs`
- Modify: `rust/cli/tests/eval_command.rs`
- Create: `rust/e2e-tests/tests/test_harbor_external_compatibility.rs`
- Modify: `docs/specs/native-harbor-agentic-benchmarking.md`
- Modify: `docs/specs/README.md`
- Modify: `llms.txt`

**Interfaces:**
- Consumes: external executor and exact preflight `select_native_graph_external_driver`.
- Produces: `aiperf profile --task <external task>` success without `--model-runtime`, externally-driven result artifact, and explicit suite refusal.

- [ ] **Step 1: Write the failing CLI and product tests**

Create an e2e task whose external driver emits one valid terminal receipt through the in-repo mock-server/Docker environment. Assert externally-driven fidelity, verifier-authored reward/score, default `Missing` capture, and no raw driver payload in exported lifecycle evidence. Add CLI tests for no-model-runtime success, unknown factory and `--agent-command` pre-Docker refusal, and suite refusal.

```rust
assert_eq!(report.fidelity(), EpisodeFidelity::ExternallyDriven(CompatibilityFidelity::Missing));
assert_eq!(report.reward(), Some(1.0));
assert!(!report_text.contains("driver-private-output"));
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ../.venv/bin/activate && RUSTC_WRAPPER= cargo test -p aiperf-cli --test eval_command external -- --nocapture && RUSTC_WRAPPER= cargo test -p aiperf-e2e-tests --test test_harbor_external_compatibility -- --nocapture`

Expected: FAIL at the current generic compatibility-runner-unavailable boundary.

- [ ] **Step 3: Wire only the supported CLI path**

After exact preflight, resolve the one-trial suite and invoke `DockerExternallyDrivenEpisodeExecutor` without model-runtime resolution. Preserve lifecycle argv and `--agent-command` validation; retain explicit suite refusal. Update canonical Harbor docs and indexes to describe the built lower-fidelity boundary.

- [ ] **Step 4: Run E2E and documentation verification**

Run:

```bash
source ../.venv/bin/activate
RUSTC_WRAPPER= cargo test -p aiperf-cli --test eval_command -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-e2e-tests --test test_harbor_external_compatibility -- --nocapture
cargo fmt --all --check
git diff --check
cd .. && /usr/bin/python3 tools/check_agent_files_sync.py && /usr/bin/python3 tools/check_docs_current.py
```

Expected: PASS; Docker-dependent tests may be ignored only when the daemon is unavailable, never when compatibility execution is unsupported.

- [ ] **Step 5: Commit**

```bash
git add rust/cli/src/eval/native_graph.rs rust/cli/tests/eval_command.rs rust/e2e-tests/tests/test_harbor_external_compatibility.rs docs/specs/native-harbor-agentic-benchmarking.md docs/specs/README.md llms.txt
git commit -m "feat(cli): run Harbor external compatibility tasks"
```

### Task 6: Completion audit

**Files:**
- Inspect: all Task 1–5 files and generated result artifacts.

**Interfaces:**
- Consumes: complete external compatibility runner and all test receipts.
- Produces: requirement-by-requirement evidence that both NativeGraph and externally driven Harbor paths execute end to end.

- [ ] **Step 1: Build and run the focused Harbor matrix**

```bash
source ../.venv/bin/activate
RUSTC_WRAPPER= cargo build -p aiperf-cli
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_capture -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_protocol -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_scored_episode -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-cli --test eval_command -- --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-e2e-tests --test test_harbor_external_compatibility -- --nocapture
```

- [ ] **Step 2: Inspect requirement evidence**

Confirm results distinguish native/external profile; unknown factory, bad argv, missing/invalid terminal, and secret exposure fail before verification; valid external execution has one compatibility lifecycle digest, verifier-authored reward/score, and no raw driver payload.

- [ ] **Step 3: Run final repository checks**

Run: `source ../.venv/bin/activate && cargo fmt --all --check && git diff --check && cd .. && /usr/bin/python3 tools/check_agent_files_sync.py && /usr/bin/python3 tools/check_docs_current.py`

Expected: PASS.

- [ ] **Step 4: Commit any audit-only documentation correction**

If the audit changes canonical documentation, commit only the complete files:

```bash
git add docs/specs/native-harbor-agentic-benchmarking.md docs/specs/README.md llms.txt
git commit -m "docs(eval): finalize Harbor compatibility evidence"
```

