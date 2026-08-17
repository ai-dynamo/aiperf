# Native Harbor Agentic Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build scored live NativeGraph Harbor evaluation and RL rollout evaluation by extending AIPerf's existing `GraphTraceProgram`, graph driver, turn coordinator, tool, endpoint, transport, observer, verifier, and cellular seams.

**Architecture:** Schema-1.1 NativeGraph source lowers into the existing `GraphTraceProgram`; a capability-selected live `TraceProgramDriver` advances bounded stages through an extended `AgentTurnCoordinator`, while every model stage executes through the existing Graph-IR scheduler and worker-local `GraphSink`/endpoint runtime. One early bounded matrix scheduler owns all single-task and suite admission, stable ordering, attempts, resource leases, and later cellular folding. Harbor evaluation owns frozen evidence and independent scoring; supervised cross-language processes implement only declared leaf operations or the separately classified `externally_driven` profile.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes and `LocalSet`, existing AIPerf Graph-IR/channel/driver/agent/tool/endpoint/observer seams, strict serde JSON/TOML DTOs, BLAKE3 identities, Docker/Compose task environments, and the existing Harbor verifier/regrade lifecycle.

## Global Constraints

- The benchmark control plane, graph scheduler, model dispatch, protocol parser, artifact authority, verifier lifecycle, suite scheduler, result aggregation, and export remain native Rust.
- `native_graph` is the exact primary profile; Rust owns every benchmark model request and the complete executable topology.
- External tools, environments, heuristics, and policies are supervised child processes. They receive no benchmark model credential or direct endpoint authority.
- `externally_driven` is a distinct compatibility profile; capture never upgrades it to exact NativeGraph.
- A complete independently verified episode is the primary result. Node metrics are explanatory telemetry.
- RL rollout/evaluation is in scope. Training and policy updates are excluded and require a separate post-release implementation plan.
- Supervised live processes use `RealClock`; source timestamps are evidence only.
- Preserve schema-1.0, ordered `[[steps]]`, legacy JSON, recorded mini-SWE replay, and existing single-task CLI JSON byte contracts.
- Schema-1.1 NativeGraph rejects authored `[[steps]]` before provisioning.
- Do not add a NativeGraph-only graph executor, scheduler, endpoint runtime, tool runtime, driver registry, or aggregate implementation universe.
- Extend `GraphTraceProgram`, `TraceProgramDriverFactory`, `AgentTurnCoordinatorFactory`, `AgentInvocationLease`, `ToolDispatcher`, graph channels/reducers, `GraphSink`, and existing application registries.
- Major implementations are selected through narrow typed capabilities. No mega-enum dispatcher, global mutable registry, or hard-coded profile constructor is permitted.
- Keep request/token hot paths worker-local. Do not add `Arc<Mutex<_>>` or unbounded NativeGraph channels.
- Registries freeze before workers start; duplicate names, unknown ids, and incomplete capabilities fail before environment provisioning.
- Each task ends with a focused RED/GREEN cycle, formatting/diff checks, and fresh independent Graham approval before its commit.
- Run Cargo commands from `rust/` with the project sccache configuration intact.
- Add the two NVIDIA SPDX lines, module docs, and public-item docs to every Rust source file.

---

## File and ownership map

- `rust/runtime/src/eval/native_graph/package.rs` — schema-1.1 package, complete model-binding DTOs, adapter manifest, and executable-source identity.
- `rust/runtime/src/eval/native_graph/model_runtime.rs` — strict runtime secret-name mapping and binding resolution into existing endpoint/transport/tokenizer factories.
- `rust/runtime/src/eval/native_graph/suite.rs` — suite manifest and deterministic trial expansion.
- `rust/runtime/src/eval/native_graph/matrix.rs` — early bounded episode scheduling, resource leases, attempts, and stable output order.
- `rust/runtime/src/eval/native_graph/result.rs` — orthogonal result axes and bounded episode summaries used by the matrix from its first delivery.
- `rust/runtime/src/eval/native_graph/evaluator.rs` — frozen verifier handoff over existing reward, score, evidence, and regrade seams.
- `rust/runtime/src/eval/native_graph/protocol.rs` — directional bounded adapter wire DTOs and role-aware state machine.
- `rust/runtime/src/eval/native_graph/artifacts.rs` — quota-bound Rust-owned artifact exchange.
- `rust/runtime/src/eval/native_graph/supervision.rs` — supervised child lifecycle, reset/reuse, and exact-profile isolation preflight.
- `rust/runtime/src/eval/native_graph/lowering.rs` — source-faithful lowering into `GraphTraceProgram` and closure validation.
- `rust/runtime/src/eval/native_graph/live_driver.rs` — live `TraceProgramDriver` and `AgentTurnCoordinator` implementations over existing graph stages.
- `rust/runtime/src/eval/native_graph/rl.rs` — reset/transition facts and Rust-derived returns.
- `rust/runtime/src/eval/native_graph/capture.rs` — compatibility-only observation and fidelity classification.
- `rust/runtime/src/graph/driver.rs` — minimal staged-driver extension and registered driver-family composition.
- `rust/runtime/src/graph/agent/turn.rs` — minimal live-turn extension with defaults preserving recorded replay.
- `rust/runtime/src/engine/graph_execution.rs` — execute driver-produced `GraphTracePlan` stages through the existing graph executor and endpoint sink.
- `rust/runtime/src/engine/execution_factories.rs` and `rust/runtime/src/engine/application.rs` — freeze the existing driver/model-binding extension points.
- `rust/runtime/src/engine/native_graph_cellular.rs` — matrix assignment supplements and associative controller fold.
- `rust/cli/src/eval/native_graph.rs` — CLI composition only after single-task and suite run functions exist.

No file named `native_graph/executor.rs`, `native_graph/registry.rs`, or `engine/native_graph_model.rs` is created. Their responsibilities remain with the existing graph executor, application registry, and endpoint runtime.

### Authoritative seam map

| Slice | Existing authority | Minimal addition |
|---|---|---|
| Executable unit | `GraphTraceProgram` / `GraphTracePlan` | NativeGraph source lowerer emits them |
| Graph scheduling/state | existing graph executor, channels, reducers | staged plan loop driven by `TraceProgramDriver` |
| Live progression | `TraceProgramDriverFactory`, `AgentTurnCoordinatorFactory` | capability-bearing live implementations and default-compatible staged methods |
| Model calls | graph `GraphSink`, endpoint runtime, transport dispatcher, `RequestObserver` | `NativeGraphModelBindingResolver` prepares existing runtime inputs |
| Tools/workspaces | `ToolDispatcher`, `AgentInvocationLease`, `SegmentStore` | supervised adapter-backed tool factory and immutable branch merge facts |
| Evaluation | Harbor verifier, `RewardDocument`, score, evidence, regrade | `EpisodeEvaluatorFactory` adapter |
| Parallel execution | matrix `EpisodeRunner` and resource leases | local first, cellular placement supplement afterward |

---

### Task 1: Schema 1.1 package, complete model bindings, and source identity

**Files:**
- Create: `rust/runtime/src/eval/native_graph/mod.rs`
- Create: `rust/runtime/src/eval/native_graph/package.rs`
- Modify: `rust/runtime/src/eval/mod.rs`
- Modify: `rust/runtime/src/eval/import/{mod.rs,normalize.rs,source_snapshot.rs}`
- Modify: `rust/runtime/src/eval/execution/plan.rs`
- Test: `rust/runtime/tests/native_graph_package.rs`
- Test: `rust/runtime/tests/harbor_import.rs`

**Interfaces:**
- Produces `NativeGraphProfile`, `AdapterRole`, `AdapterId`, `ModelBindingId`, `ModelSecretId`, `TokenizerBindingSpec`, `GenerationDefaults`, `ModelCapturePolicy`, `HeaderSecretRef`, `ModelBindingSpec`, `AdapterSpec`, and `NativeGraphPackagePlan`.
- Produces `HarborTaskPackage::native_graph() -> Option<&NativeGraphPackagePlan>`.
- Consumes the retained `AcquiredSource`, `ExecutableSourceView`, `CanonicalPackagePlan`, and `ArtifactDigest`; normalization never rereads the caller origin.

- [ ] **Step 1: Write failing schema, identity, and legacy-golden tests**

```rust
#[test]
fn model_binding_retains_every_runtime_selection_without_secret_values() {
    let imported = import_native_task(native_task_fixture()).unwrap();
    let binding = &imported.package.native_graph().unwrap().model_bindings()[0];
    assert_eq!(binding.endpoint_factory_id().as_str(), "chat");
    assert_eq!(binding.transport_factory_id(), "http");
    assert_eq!(binding.authentication()[0].secret.as_str(), "provider-key");
    assert!(!imported.package.source_bytes().windows(12).any(|w| w == b"actual-secret"));
}

#[test]
fn executable_adapter_mutation_changes_package_identity() {
    let first = import_native_task(native_task_with_adapter(b"print('a')")).unwrap();
    let second = import_native_task(native_task_with_adapter(b"print('b')")).unwrap();
    assert_ne!(first.task.digest, second.task.digest);
}

#[test]
fn schema_1_0_digest_golden_is_unchanged() {
    assert_eq!(import_legacy_golden().task.digest.as_str(), LEGACY_DIGEST);
}
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_package --test harbor_import -- native_graph --nocapture
```

Expected: compile failure because `NativeGraphPackagePlan` and model-binding types do not exist.

- [ ] **Step 3: Add strict authored and resolved DTOs**

```rust
pub struct ModelBindingSpec {
    pub id: ModelBindingId,
    pub endpoint_profile_id: String,
    pub endpoint_factory_id: EndpointId,
    pub transport_factory_id: String,
    pub model: String,
    pub urls: Vec<String>,
    pub streaming: bool,
    pub tokenizer: TokenizerBindingSpec,
    pub authentication: Vec<HeaderSecretRef>,
    pub generation: GenerationDefaults,
    pub max_connect_retries: u32,
    pub request_timeout_ms: NonZeroU64,
    pub capture: ModelCapturePolicy,
}

pub struct NativeGraphPackagePlan {
    profile: NativeGraphProfile,
    program: Option<ArtifactDigest>,
    model_bindings: Vec<ModelBindingSpec>,
    adapters: Vec<AdapterSpec>,
    driver: Option<AdapterId>,
    executable_source_digest: ArtifactDigest,
}
```

Use `#[serde(deny_unknown_fields)]` on every authored DTO. Validate unique ids, nonempty URLs/argv, canonical relative paths, finite generation values, supported tokenizer shape, unique normalized headers, positive timeout, role/profile compatibility, and exact NativeGraph versus external-driver requirements.

- [ ] **Step 4: Parse schema 1.1 without changing schema 1.0 or JSON behavior**

Reject `native_graph + [[steps]]` before environment planning. Extend the executable-source projection with the exact graph, model manifest, adapter manifest, every executable/config/policy file, complete environment tree, and selected verifier trees.

- [ ] **Step 5: Run identity, import, format, and diff checks**

```bash
cargo test -p aiperf-runtime --test native_graph_package --test harbor_import --nocapture
cargo fmt --check
git diff --check
```

- [ ] **Step 6: Obtain Graham approval and commit the slice**

```bash
git add rust/runtime/src/eval rust/runtime/tests/native_graph_package.rs rust/runtime/tests/harbor_import.rs
git commit -m "feat(eval): resolve native graph packages"
```

---

### Task 2: Early result contract and bounded matrix scheduler

**Files:**
- Create: `rust/runtime/src/eval/native_graph/result.rs`
- Create: `rust/runtime/src/eval/native_graph/suite.rs`
- Create: `rust/runtime/src/eval/native_graph/matrix.rs`
- Test: `rust/runtime/tests/native_graph_matrix.rs`
- Test: `rust/runtime/tests/native_graph_suite.rs`

**Interfaces:**
- Consumes `NativeGraphPackagePlan`, `TrialSpec`, `HarborSource`, and `ArtifactDigest` from Task 1.
- Produces `EpisodeIntegrity`, `EpisodeExecution`, `EpisodeScoreState`, `EpisodeComparability`, `EpisodeResult`, `NativeGraphSuiteManifest`, `ResolvedNativeGraphSuite`, `ResolvedEpisodeTrial`, and `ResourceLeaseRequest`.
- Produces `EpisodeAssignment`, `ResourceLimits`, `NativeGraphSuiteScheduler`, `EpisodeRunner`, `SuiteSchedulerFactory`, and `run_resolved_suite(...) -> Result<Vec<EpisodeResult>, MatrixError>` with the async signatures shown below.

- [ ] **Step 1: Write RED tests for deterministic expansion, bounded concurrency, and stable result order**

```rust
#[tokio::test(flavor = "current_thread")]
async fn completion_order_never_changes_manifest_order() {
    let scheduler = test_scheduler(2);
    let results = scheduler.run(four_trials(), delayed_scored_runner([40, 10, 30, 20])).await.unwrap();
    assert_eq!(trial_ids(&results), manifest_order());
    assert!(observed_peak_concurrency() <= 2);
}

#[test]
fn valid_failed_zero_score_remains_in_the_quality_denominator() {
    let summary = aggregate([valid_failed(0.0), valid_completed(1.0), invalid_provider()]).unwrap();
    assert_eq!((summary.valid_attempts, summary.invalid_attempts), (2, 1));
    assert_eq!(summary.mean_reward, 0.5);
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix --nocapture
```

- [ ] **Step 3: Implement pure suite expansion and the narrow runner seam**

```rust
#[async_trait(?Send)]
pub trait EpisodeRunner {
    async fn run(&self, assignment: EpisodeAssignment) -> Result<EpisodeResult, MatrixError>;
}

pub trait SuiteSchedulerFactory: Send + Sync {
    fn create(&self, limits: ResourceLimits)
        -> Result<Rc<dyn NativeGraphSuiteScheduler>, MatrixError>;
}
```

Use a bounded semaphore for episode slots and weighted pools for CPU, memory, and each `ModelBindingId`. Allocate stable output slots before spawning local tasks. Attempts receive deterministic ids and never overwrite earlier results.

- [ ] **Step 4: Prove a scored fake episode runs only through the scheduler**

The test runner returns a verifier-shaped `EpisodeResult`; direct helper execution is private to the test. Single-element and multi-element suites use the same public scheduler entry point.

- [ ] **Step 5: Verify and review**

```bash
cargo test -p aiperf-runtime --test native_graph_suite --test native_graph_matrix --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph rust/runtime/tests/native_graph_suite.rs rust/runtime/tests/native_graph_matrix.rs
git commit -m "feat(eval): schedule scored episode matrices"
```

---

### Task 3: Frozen evaluator and existing verifier lifecycle

**Files:**
- Create: `rust/runtime/src/eval/native_graph/evaluator.rs`
- Modify: `rust/runtime/src/eval/execution/coordinator.rs`
- Modify: `rust/runtime/src/eval/evidence.rs`
- Modify: `rust/runtime/src/eval/verifier/regrade.rs`
- Modify: `rust/runtime/src/eval/semantic/comparison.rs`
- Test: `rust/runtime/tests/native_graph_evaluator.rs`
- Test: `rust/runtime/tests/harbor_attempt_bundle.rs`

**Interfaces:**
- Consumes Task 2 `EpisodeRunner`, `EpisodeResult`, matrix scheduler, and existing `RewardDocument`, `ScoreVersion`, verifier staging, evidence, and regrade types.
- Produces `FrozenAttemptBundle`, `EpisodeEvaluator`, and `EpisodeEvaluatorFactory`.

- [ ] **Step 1: Write RED tests for frozen handoff, zero-score inclusion, and regrade append-only behavior**

```rust
#[test]
fn frozen_existing_harbor_attempt_preserves_verifier_input_evidence() {
    let bundle = frozen_harbor_attempt(reward_json(0.75));
    assert_eq!(bundle.verifier_input_evidence(), declared_artifact_digests());
    assert_eq!(bundle.lifecycle_evidence_digest(), expected_lifecycle_digest());
}

#[test]
fn regrade_creates_a_new_score_version_without_mutating_evidence() {
    let (before, after) = regrade_fixture();
    assert_ne!(before.score.version, after.score.version);
    assert_eq!(before.evidence, after.evidence);
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_evaluator --test harbor_attempt_bundle --nocapture
```

- [ ] **Step 3: Add the evaluator seam over the existing authority**

```rust
#[async_trait(?Send)]
pub trait EpisodeEvaluator {
    async fn evaluate(&self, attempt: FrozenAttemptBundle)
        -> Result<EpisodeResult, EpisodeEvaluationError>;
}

pub trait EpisodeEvaluatorFactory: Send + Sync {
    fn create(&self, trial: &ResolvedEpisodeTrial)
        -> Result<Rc<dyn EpisodeEvaluator>, EpisodeEvaluationError>;
}
```

The built-in evaluator delegates verifier provisioning, declared-artifact staging, `RewardDocument`, score versioning, and regrade to current Harbor code. It adds no second scoring authority. `VerifierResult.evidence` retains its current meaning: the declared artifact digests supplied to the verifier. The bundle separately freezes ordered lifecycle evidence and its identity; score lineage may reference that identity without relabeling post-verifier facts as verifier inputs.

- [ ] **Step 4: Prove frozen existing Harbor authority without claiming a live NativeGraph run**

Use an existing completed Harbor verifier/regrade fixture to prove frozen evidence, zero-score inclusion, and append-only score versions. Do not invoke the local or Docker process executor for a schema-1.1 package here: current local execution intentionally rejects that profile before a native graph driver/model runtime exists. The first actual schema-1.1 NativeGraph episode is Task 7, where its Rust-owned driver and model runtime execute through Task 2's scheduler.

- [ ] **Step 5: Verify and review**

```bash
cargo test -p aiperf-runtime --test native_graph_evaluator --test harbor_attempt_bundle --test eval_regrade --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph/evaluator.rs rust/runtime/src/eval/execution/coordinator.rs rust/runtime/src/eval/evidence.rs rust/runtime/src/eval/verifier/regrade.rs rust/runtime/src/eval/semantic/comparison.rs rust/runtime/tests
git commit -m "feat(eval): score episodes through the matrix"
```

---

### Task 4: Directional bounded adapter protocol and Rust-owned artifacts

**Files:**
- Create: `rust/runtime/src/eval/native_graph/protocol.rs`
- Create: `rust/runtime/src/eval/native_graph/artifacts.rs`
- Test: `rust/runtime/tests/native_graph_protocol.rs`
- Test: `rust/runtime/tests/native_graph_artifacts.rs`

**Interfaces:**
- Produces `HostEnvelope`, `AdapterEnvelope`, role-gated message DTOs, `ProtocolMachine`, `AdapterProtocolFactory`, `EpisodeArtifactStore`, upload/download handles, quotas, and frozen artifact manifests.
- Consumes ids and digests from Task 1. Later tasks receive only validated messages and committed artifact references.

- [ ] **Step 1: Write RED tests for direction, sequence, role, correlation, frame size, quota, and mutation**

```rust
#[test]
fn leaf_adapter_cannot_claim_episode_terminal() {
    let error = ready_machine(AdapterRole::Tool)
        .accept(episode_terminal_candidate())
        .unwrap_err();
    assert_eq!(error, ProtocolError::MessageForbiddenForRole(AdapterRole::Tool));
}

#[test]
fn committed_artifact_is_immutable_after_staging_mutation() {
    let frozen = commit_fixture(b"one");
    mutate_staging_fixture(b"two");
    assert_eq!(read_frozen(&frozen.digest), b"one");
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_protocol --test native_graph_artifacts --nocapture
```

- [ ] **Step 3: Implement strict protocol and artifact interfaces**

Use versioned JSONL, a byte cap before deserialization, bounded strings/arrays/JSON/artifact counts, monotonic sequences, operation correlation, and role capabilities. Artifact uploads use store-owned no-follow/create-new staging files, exact-length streaming hash, fsync, atomic publication, and opaque child handles; children never name trusted digests or host store paths.

- [ ] **Step 4: Verify and review**

```bash
cargo test -p aiperf-runtime --test native_graph_protocol --test native_graph_artifacts --nocapture
cargo clippy -p aiperf-runtime --test native_graph_protocol --test native_graph_artifacts -- -D warnings
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph/protocol.rs rust/runtime/src/eval/native_graph/artifacts.rs rust/runtime/tests/native_graph_protocol.rs rust/runtime/tests/native_graph_artifacts.rs
git commit -m "feat(eval): bound native graph adapters and artifacts"
```

---

### Task 5: Supervised process lifecycle and exact-profile isolation

**Files:**
- Create: `rust/runtime/src/eval/native_graph/supervision.rs`
- Modify: `rust/runtime/src/eval/provider.rs`
- Modify: `rust/runtime/src/eval/execution/{plan.rs,task_environment.rs,local_process.rs,docker_process.rs,docker_runtime.rs}`
- Test: `rust/runtime/tests/native_graph_supervision.rs`
- Test: `rust/runtime/tests/harbor_docker_runtime.rs`

**Interfaces:**
- Consumes Task 4 protocol and artifact handles.
- Produces `AdapterRuntimeFactory`, `AdapterSpawner`, `SupervisedAdapter`, lifecycle/deadline DTOs, and `ProviderRecovery`.
- Adds streaming spawn to the existing `TaskEnvironmentLease` and `ProviderCapability::ModelEndpointIsolation` with typed proof details.

- [ ] **Step 1: Write RED tests for reset failure, bounded drains, cancellation/reap, secret stripping, and endpoint bypass refusal**

```rust
#[test]
fn exact_profile_refuses_a_provider_without_endpoint_isolation() {
    let error = native_plan().preflight(provider_with_public_network_only()).unwrap_err();
    assert!(error.to_string().contains("model endpoint isolation"));
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_operation_fences_and_reaps_the_child() {
    let child = spawn_blocked_adapter().await;
    drop(child.operation);
    assert_eq!(child.observed_exit().await, AdapterExit::Reaped);
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_supervision --test harbor_docker_runtime -- supervision --nocapture
```

- [ ] **Step 3: Implement cancellation-safe process traits and streaming environment spawn**

```rust
#[async_trait(?Send)]
pub trait SupervisedAdapter {
    async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError>;
    async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError>;
    async fn cancel_and_reap(&mut self, reason: CancelReason)
        -> Result<AdapterExit, AdapterSupervisionError>;
}
```

Use bounded stdout/stderr drains, process-group ownership, distinct startup/reset/heartbeat/idle/operation/cancel/reap deadlines, and a synchronous Drop fence. Docker owns the exact `docker exec -i` client process and labelled task container.

- [ ] **Step 4: Enforce credential and network authority**

Resolve model secrets only in the host model-runtime path. Remove every referenced secret from adapter environments. Exact NativeGraph permits no adapter egress or a provider proof of enforcing mediation that denies all resolved benchmark endpoint authorities. Public networking alone fails preflight. A bypass attempt makes integrity invalid and terminates the attempt.

- [ ] **Step 5: Verify reset reuse and no cross-task pooling**

Pool keys contain task, environment, adapter implementation, role, and protocol digests. Compare seeded fresh versus reused canaries; a failed reset reaps the worker and forces a fresh process.

- [ ] **Step 6: Verify and review**

```bash
cargo test -p aiperf-runtime --test native_graph_supervision --test harbor_docker_runtime --nocapture
cargo test -p aiperf-runtime --features engine --test harbor_execution_plan --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph/supervision.rs rust/runtime/src/eval/provider.rs rust/runtime/src/eval/execution rust/runtime/tests/native_graph_supervision.rs rust/runtime/tests/harbor_docker_runtime.rs
git commit -m "feat(eval): supervise native graph adapters"
```

---

### Task 6: Source-faithful lowering into GraphTraceProgram and live driver extensions

**Files:**
- Create: `rust/runtime/src/eval/native_graph/lowering.rs`
- Create: `rust/runtime/src/eval/native_graph/live_driver.rs`
- Modify: `rust/runtime/src/eval/semantic/{mod.rs,lowering.rs}`
- Modify: `rust/runtime/src/graph/driver.rs`
- Modify: `rust/runtime/src/graph/agent/turn.rs`
- Modify: `rust/runtime/src/engine/{execution_factories.rs,graph_execution.rs}`
- Test: `rust/runtime/tests/native_graph_lowering.rs`
- Test: `rust/runtime/tests/native_graph_driver.rs`
- Test: `rust/runtime/tests/recorded_agent_driver.rs`

**Interfaces:**
- Consumes Task 1 source bytes/bindings, Task 4 protocol, Task 5 supervised adapters, existing channel/reducer/segment/tool/invocation types.
- Produces the generic semantic `GraphLowererFactory`, `NativeGraphLoweringReport`, `TraceStageResult`, `LiveAgentTurnDirective`, and `lower_native_graph(...) -> Result<(GraphTraceProgram, NativeGraphLoweringReport), NativeGraphLoweringError>`.
- Extends existing driver and turn traits with default-compatible staged progression; it does not create an episode executor.

- [ ] **Step 1: Write RED tests for exact GraphTraceProgram output, closure refusal, and recorded replay parity**

```rust
#[test]
fn native_source_lowers_to_the_existing_trace_program_type() {
    let (program, report): (GraphTraceProgram, _) = lower_fixture("model-tool-loop.json").unwrap();
    assert_eq!(program.driver.kind, "native_graph_live");
    assert!(report.nodes().all(|node| node.is_exact()));
}

#[test]
fn unbounded_cycle_is_refused_before_driver_creation() {
    assert!(matches!(lower_fixture("unbounded.json"), Err(NativeGraphLoweringError::UnboundedCycle { .. })));
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_lowering --test native_graph_driver --test recorded_agent_driver --nocapture
```

- [ ] **Step 3: Add minimal default-compatible staged methods**

```rust
pub enum TraceStageDirective {
    Execute(GraphTracePlan),
    Complete(TraceTerminalSupplement),
}

pub struct TraceStageResult {
    pub plan_identity: String,
    pub terminal_status: GraphReplyStatus,
    pub channels: BTreeMap<String, Value>,
    pub output_handles: Vec<Handle>,
}

pub enum LiveAgentTurnDirective {
    DispatchModel { binding: String, prompt: Handle },
    InvokeTool { adapter: String, input: Handle },
    StepEnvironment { adapter: String, action: Handle },
    SelectBranch { edge: String },
    Complete { outputs: Vec<Handle> },
}

#[async_trait(?Send)]
pub trait TraceProgramDriver {
    // existing open/tool_dispatcher/close/run methods remain
    async fn next_stage(&mut self, context: &TraceDriverContext<'_>)
        -> Result<Option<TraceStageDirective>, TraceDriverError> { Ok(None) }
    async fn observe_stage(&mut self, result: TraceStageResult)
        -> Result<(), TraceDriverError> { Ok(()) }
}
```

Add `AgentTurnCoordinator::next_live_turn` with a default typed unsupported result. `LiveAgentTurnDirective` uses already-validated string ids and `SegmentStore` handles, so the graph module does not depend on evaluation package types. It may select one declared model binding, tool operation, environment step, bounded branch/join action, or terminal output. Static and recorded drivers retain their current single-plan execution and byte contracts through defaults.

Add a generic `GraphLowererFactory` to the existing semantic module. It advertises source schema and execution-profile capabilities and returns `GraphTraceProgram`; the NativeGraph implementation is one registered factory, not a new lowering universe. Closure validation runs after import, lowering, every driver-authored graph rewrite, and immediately before each stage executes.

- [ ] **Step 4: Execute every driver-produced stage through the existing graph executor**

Refactor `GraphWorkerBackend` into a bounded stage loop: obtain a `GraphTracePlan`, execute it with the current graph executor and `EngineGraphSink`, return its immutable channel/terminal facts to the driver, and repeat until `Complete` or a declared bound. Cancellation and tool dispatch continue through existing active-trace and invocation-lease ownership.

- [ ] **Step 5: Replace hard-coded driver-kind selection at the existing seam**

Use a `TransactionalRegistry<Arc<dyn TraceProgramDriverFactory>>` keyed by `TraceDriverSpec.kind` inside `ExecutionFactories`. Register static, recorded replay, and live NativeGraph families at application bootstrap. This replaces the current built-in `match` without adding a NativeGraph registry.

- [ ] **Step 6: Verify closure, parity, no parallel executor, and review**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_lowering --test native_graph_driver --test recorded_agent_driver --nocapture
rg -n "NativeGraphEpisodeExecutor|native_graph/executor|NativeGraphRegistries" rust/runtime/src
cargo fmt --check
git diff --check
```

Expected: tests pass and the scan prints no matches.

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph rust/runtime/src/eval/semantic rust/runtime/src/graph/driver.rs rust/runtime/src/graph/agent/turn.rs rust/runtime/src/engine/execution_factories.rs rust/runtime/src/engine/graph_execution.rs rust/runtime/tests
git commit -m "feat(graph): drive live native graph stages"
```

---

### Task 7: Complete model-runtime resolution and scored NativeGraph vertical slice

**Files:**
- Create: `rust/runtime/src/eval/native_graph/model_runtime.rs`
- Create: `rust/cli/src/eval/native_graph.rs`
- Modify: `rust/runtime/src/extensions/mod.rs`
- Modify: `rust/runtime/src/engine/application.rs`
- Modify: `rust/runtime/src/engine/graph_execution.rs`
- Modify: `rust/cli/src/eval.rs`
- Test: `rust/runtime/tests/native_graph_model_runtime.rs`
- Test: `rust/runtime/tests/native_graph_scored_episode.rs`
- Test: `rust/cli/tests/eval_command.rs`

**Interfaces:**
- Consumes Task 1 `ModelBindingSpec`, Task 2 matrix, Task 3 evaluator, Task 5 supervision, Task 6 live driver, current `AIPerfRegistry`, `ExecutionFactories`, `ValidatedEndpointProfileV2`, `NativeTransportExecution`, input-token counter, `GraphSink`, and `RequestObserver`.
- Produces `ModelRuntimeConfig`, `ResolvedModelBinding`, `ResolvedModelBindingSet`, `NativeGraphModelBindingResolver`, `run_task(...)`, and `run_suite(...)`.
- Extends the existing `AIPerfRegistry` with typed transactional fields for the genuinely new lowerer, suite-scheduler, evaluator, adapter-protocol, adapter-runtime, environment-stepper, external-driver, fidelity-observer, and provider-recovery factories. Existing graph driver, endpoint, transport, tool, clock, segment, observer, and exporter authorities are not duplicated.

- [ ] **Step 1: Write RED tests for complete binding, unknown ids, secret ownership, and scored CLI execution**

```rust
#[test]
fn resolves_binding_to_current_endpoint_transport_tokenizer_and_capture_types() {
    let resolved = resolve_fixture("openai-http-local-tokenizer.toml").unwrap();
    assert_eq!(resolved.profile.endpoint_id.as_str(), "chat");
    assert_eq!(resolved.transport_id(), "http");
    assert_eq!(resolved.tokenizer_id(), "tiktoken");
    assert_eq!(resolved.max_connect_retries(), 2);
    assert_eq!(resolved.capture(), ModelCapturePolicy::RedactedRaw);
}

#[test]
fn resolved_model_secret_never_enters_adapter_environment() {
    let execution = resolve_with_secret("provider-key", "secret-value").unwrap();
    assert!(execution.model_headers().contains_secret("secret-value"));
    assert!(!execution.adapter_environment().contains_value("secret-value"));
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_model_runtime --test native_graph_scored_episode --nocapture
cargo test -p aiperf-cli --test eval_command -- native_graph --nocapture
```

- [ ] **Step 3: Define strict runtime-secret mapping and resolver**

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRuntimeConfig {
    pub version: u32,
    pub secrets: BTreeMap<ModelSecretId, EnvName>,
}

pub trait NativeGraphModelBindingResolver: Send + Sync {
    fn resolve(
        &self,
        specs: &[ModelBindingSpec],
        runtime: &ModelRuntimeConfig,
        secrets: &dyn SecretProvider,
    ) -> Result<ResolvedModelBindingSet, ModelRuntimeError>;
}
```

For each binding resolve endpoint factory id, transport factory id, tokenizer/server-tokenizer configuration, URLs, generation fields, retry/timeout, capture/redaction, and logical secret ids. Construct the current endpoint profile and transport execution inputs. Unknown, duplicate, incompatible, or missing components fail before adapter/environment provisioning.

- [ ] **Step 4: Prepare current worker-local endpoint and observer paths**

Create the existing `PreparedRunnerGraphEndpointRuntimeFactory` inputs from the resolved set. Every driver model stage becomes an ordinary Graph-IR LLM node dispatched through `EngineGraphSink`; token, usage, terminal, raw-capture, and timing observations use the current worker observer. No eval-specific HTTP/gRPC client is added.

- [ ] **Step 5: Freeze and resolve the new factories through the existing application registry**

Register built-in lowerer, matrix scheduler, evaluator, JSONL protocol, local/Docker adapter runtimes, environment stepper, external driver, fidelity observer, and recovery implementations through `AIPerfExtension`. Test duplicate names, missing capabilities, and injected fake selection. Resolve each narrow factory before provisioning and pass it by constructor; do not create an aggregate NativeGraph component bundle or profile `match` constructor.

- [ ] **Step 6: Deliver the first scored NativeGraph episode through the matrix**

Implement `EpisodeRunner` for the resolved NativeGraph task. Even `aiperf eval --task` constructs a one-trial `ResolvedNativeGraphSuite` and calls `run_resolved_suite`; it then preserves the legacy single-task JSON shape where applicable. `--suite` calls the already-delivered same function. Add `--model-runtime` to both selections; it may map secret ids only.

- [ ] **Step 7: Verify focused, legacy CLI, and no-direct-client checks**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_model_runtime --test native_graph_scored_episode --nocapture
cargo test -p aiperf-cli --test eval_command --nocapture
rg -n "hyper::Client|tonic::transport::Channel|reqwest" rust/runtime/src/eval/native_graph
cargo fmt --check
git diff --check
```

Expected: tests pass and the client scan prints no matches.

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph/model_runtime.rs rust/runtime/src/extensions/mod.rs rust/runtime/src/engine rust/cli/src/eval.rs rust/cli/src/eval/native_graph.rs rust/runtime/tests rust/cli/tests/eval_command.rs
git commit -m "feat(eval): score live native graph episodes"
```

---

### Task 8: Model-dependent branches, loops, invocation leases, and counterfactual proof

**Files:**
- Modify: `rust/runtime/src/eval/native_graph/{lowering.rs,live_driver.rs}`
- Modify: `rust/runtime/src/graph/agent/{turn.rs,lease.rs}`
- Modify: `rust/runtime/src/graph/tools/dispatch.rs`
- Modify: `rust/runtime/src/engine/graph_execution.rs`
- Test: `rust/runtime/tests/native_graph_live_paths.rs`
- Test: `rust/runtime/tests/native_graph_workspaces.rs`

**Interfaces:**
- Consumes the Task 6 staged driver and Task 7 resolved model runtime.
- Produces bounded conditional-edge selection, explicit joins, bounded loops/retries, delegated invocation ownership, branch workspace candidates, and merge selection through existing Graph-IR and lease seams.

- [ ] **Step 1: Write the required counterfactual RED test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn model_response_selects_distinct_tools_and_observation_changes_next_request() {
    let a = run_counterfactual(ModelReplies::new(["choose-a", "finish-a"])).await.unwrap();
    let b = run_counterfactual(ModelReplies::new(["choose-b", "finish-b"])).await.unwrap();
    assert_eq!(a.executed_tools(), ["tool-a"]);
    assert_eq!(b.executed_tools(), ["tool-b"]);
    assert!(a.model_requests()[1].contains("observation-a"));
    assert!(b.model_requests()[1].contains("observation-b"));
    assert_eq!((a.verified_reward(), b.verified_reward()), (1.0, 0.0));
}
```

Also add RED tests for losing-branch cancellation, loop horizon, retry budget, isolated child workspace, and explicit merge selection.

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_live_paths --test native_graph_workspaces --nocapture
```

- [ ] **Step 3: Implement live control through existing stage/turn contracts**

The live turn coordinator validates each model result against declared conditional edges, creates the next `GraphTracePlan`, and supplies selected channel/artifact inputs. Tool observations enter typed graph channels before the next LLM plan materializes. Loop/retry counters are Rust-owned and checked before a stage is emitted.

- [ ] **Step 4: Implement branch workspace and invocation ownership**

Open root/child `AgentInvocationLease` values through the existing factory. Parallel branches receive isolated overlays, return immutable candidate digests, and never mutate the canonical task workspace. The graph-authored selector chooses one candidate; cancelled branches close child leases before the parent proceeds.

- [ ] **Step 5: Verify response causality and review**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_live_paths --test native_graph_workspaces --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph rust/runtime/src/graph/agent rust/runtime/src/graph/tools/dispatch.rs rust/runtime/src/engine/graph_execution.rs rust/runtime/tests/native_graph_live_paths.rs rust/runtime/tests/native_graph_workspaces.rs
git commit -m "feat(eval): drive model-dependent graph paths"
```

---

### Task 9: RL rollout evaluation through the same driver, evaluator, and matrix

**Files:**
- Create: `rust/runtime/src/eval/native_graph/rl.rs`
- Modify: `rust/runtime/src/eval/native_graph/{live_driver.rs,protocol.rs,result.rs,evaluator.rs}`
- Test: `rust/runtime/tests/native_graph_rl.rs`
- Test: `rust/runtime/tests/native_graph_rl_scored.rs`

**Interfaces:**
- Consumes the already-created result/evaluator types from Tasks 2-3, protocol from Task 4, live driver from Tasks 6-8, and matrix from Task 2.
- Produces `RlEvaluationSpec`, `RlTransition`, `RlTrajectory`, `RlReturn`, `RlTermination`, `EnvironmentStepperFactory`, and `derive_return(...)`.

- [ ] **Step 1: Write RED tests for reset, finite discounted return, termination/truncation, horizon, and verifier disagreement**

```rust
#[test]
fn derives_discounted_return_from_authoritative_transitions() {
    let spec = rl_spec(3, 0.5);
    let trajectory = [transition(2.0, false, false), transition(4.0, true, false)];
    assert_eq!(derive_return(&spec, &trajectory).unwrap().discounted, 4.0);
}

#[tokio::test(flavor = "current_thread")]
async fn rl_trial_is_independently_scored_through_the_matrix() {
    let results = run_rl_suite(one_rl_trial()).await.unwrap();
    assert_eq!(results[0].verified_reward(), Some(1.0));
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_rl --test native_graph_rl_scored --nocapture
```

- [ ] **Step 3: Implement authoritative environment progression**

Each reset yields one initial observation. Each `StepEnvironment` opens exactly one correlated transition. Reject nonfinite values, duplicate/out-of-order steps, both terminal flags, post-terminal transitions, and horizon overflow. Rust derives discounted and undiscounted returns.

- [ ] **Step 4: Freeze and independently verify the trajectory**

Freeze the canonical transition stream as a digest-addressed artifact and pass environment identity, spec, stream, derived return, and terminal artifacts to the existing `EpisodeEvaluator`. Reward output remains final score authority; disagreement makes evidence invalid.

- [ ] **Step 5: Verify and review**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_rl --test native_graph_rl_scored --test native_graph_live_paths --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph rust/runtime/tests/native_graph_rl.rs rust/runtime/tests/native_graph_rl_scored.rs
git commit -m "feat(eval): score native RL rollouts"
```

---

### Task 10: Cellular placement and associative matrix folding

**Files:**
- Create: `rust/runtime/src/engine/native_graph_cellular.rs`
- Modify: `rust/runtime/src/engine/{cellular_controller.rs,cellular_cell.rs,mod.rs}`
- Modify: `rust/runtime/src/eval/native_graph/matrix.rs`
- Test: `rust/runtime/tests/native_graph_cellular.rs`

**Interfaces:**
- Consumes Task 2 `EpisodeAssignment`, `EpisodeResult`, `SuiteSchedulerFactory`, resource leases, and stable order.
- Produces `EpisodeSupplement`, `EpisodeAggregator`, and `FoldedNativeGraphSuite` as a placement/fold extension to the existing matrix, not an alternate suite executor.

- [ ] **Step 1: Write RED tests for associative folds, stable order, retries, and invalid counts**

```rust
#[test]
fn fold_is_associative_and_preserves_invalid_denominators() {
    assert_eq!(fold(fold(a.clone(), b.clone()), c.clone()), fold(a, fold(b, c)));
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_cellular --nocapture
```

- [ ] **Step 3: Implement bounded assignment supplements and controller fold**

Cells receive deterministic `EpisodeAssignment` values from the matrix and return bounded summaries plus digest-addressed artifact references. The controller owns final writes, score/CI aggregation, invalid denominators, paired comparisons, and manifest-order output.

- [ ] **Step 4: Verify and review**

```bash
cargo test -p aiperf-runtime --features engine --test native_graph_cellular --test native_graph_matrix --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/engine/native_graph_cellular.rs rust/runtime/src/engine/cellular_controller.rs rust/runtime/src/engine/cellular_cell.rs rust/runtime/src/engine/mod.rs rust/runtime/src/eval/native_graph/matrix.rs rust/runtime/tests/native_graph_cellular.rs
git commit -m "feat(eval): place native graph matrices on cells"
```

---

### Task 11: Externally driven compatibility profile

**Files:**
- Create: `rust/runtime/src/eval/native_graph/capture.rs`
- Modify: `rust/runtime/src/eval/native_graph/{live_driver.rs,result.rs,supervision.rs}`
- Test: `rust/runtime/tests/native_graph_capture.rs`
- Test: `rust/cli/tests/eval_command.rs`

**Interfaces:**
- Consumes the supervised protocol, matrix runner, frozen evaluator, and artifact authority.
- Produces `ExternalEpisodeDriverFactory`, `FidelityObserverFactory`, `CapturePolicy`, `CaptureFidelity`, and `CompatibilityObservationReport`.

- [ ] **Step 1: Write RED tests that make fidelity upgrade impossible**

```rust
#[test]
fn proxy_observation_never_becomes_native_control() {
    let result = observe_external_episode([captured_https_call()]);
    assert_eq!(result.profile(), NativeGraphProfile::ExternallyDriven);
    assert_eq!(result.fidelity(), CaptureFidelity::ObservedProxy);
    assert_ne!(result.fidelity(), CaptureFidelity::NativeControlled);
}
```

- [ ] **Step 2: Confirm RED**

```bash
cargo test -p aiperf-runtime --test native_graph_capture --nocapture
```

- [ ] **Step 3: Implement supervised external-driver lifecycle and bounded observation**

Only an `EpisodeDriver` may propose terminal outputs. Rust freezes them, invokes the existing evaluator, and reports outer timing. Optional HTTP(S) observation records bounded redacted digests and target/timing facts; pinned, non-HTTP, in-process, and bypassed calls remain partial or missing.

- [ ] **Step 4: Run compatibility episodes through the matrix**

Implement the same `EpisodeRunner` interface and preserve `externally_driven` in every single-task, suite, retry, cellular, and exported result path.

- [ ] **Step 5: Verify and review**

```bash
cargo test -p aiperf-runtime --test native_graph_capture --test native_graph_matrix --nocapture
cargo test -p aiperf-cli --test eval_command -- externally_driven --nocapture
cargo fmt --check
git diff --check
```

After Graham approval:

```bash
git add rust/runtime/src/eval/native_graph rust/runtime/tests/native_graph_capture.rs rust/cli/tests/eval_command.rs
git commit -m "feat(eval): classify externally driven episodes"
```

---

### Task 12: Real end-to-end acceptance, documentation, and release gate

**Files:**
- Create: `rust/e2e-tests/tests/test_native_harbor_agentic.rs`
- Create: `rust/e2e-tests/tests/fixtures/native_harbor/`
- Modify: `rust/e2e-tests/Cargo.toml`
- Modify: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`
- Modify: `llms.txt`, `docs/specs/README.md`
- Modify: `docs/specs/native-harbor-agentic-benchmarking.md`
- Modify: `docs/specs/agentic-eval-platform.md`, `docs/specs/semantic-agent-graph.md`

**Interfaces:**
- Consumes all prior tasks through `aiperf eval --task/--suite --model-runtime`.
- Produces release evidence for live causal model paths, exact authority, scored outcomes, RL, matrix/cellular execution, compatibility fidelity, verifier isolation, and cleanup.

- [ ] **Step 1: Add deterministic mock-server and non-Rust adapter fixtures**

The mock server exposes response modes A and B. The Python conformance adapter implements ready/reset, policy decision, tool result, environment reset/transition, artifacts, cancel acknowledgement, and contamination canary. It never receives model credentials or sends a model request.

- [ ] **Step 2: Write ignored real-Docker E2E tests before accepting product wiring**

```rust
#[test]
#[ignore = "requires Docker and release aiperf binary"]
fn counterfactual_live_responses_drive_distinct_scored_paths() {
    let a = run_native_fixture("response-a");
    let b = run_native_fixture("response-b");
    assert_eq!(a.executed_tools(), ["tool-a"]);
    assert_eq!(b.executed_tools(), ["tool-b"]);
    assert!(a.model_request(1).contains("observation-a"));
    assert!(b.model_request(1).contains("observation-b"));
    assert_eq!((a.reward(), b.reward()), (1.0, 0.0));
}
```

Also add named tests for direct endpoint bypass refusal, adapter secret absence, unknown transport/endpoint/tokenizer/secret ids, malformed/oversized frames, model/tool cancellation, artifact mutation, branch worktree isolation, reset contamination, RL return disagreement, valid failed zero-score denominators, matrix concurrency/stable order, cellular fold parity, verifier isolation, pinned-source mutation, regrade, resource cleanup, recorded mini-SWE replay regression, and compatibility fidelity.

- [ ] **Step 3: Build the exact release binary**

```bash
cargo build --release -p aiperf-cli --features full
```

- [ ] **Step 4: Run serial Docker acceptance against that binary**

```bash
AIPERF_E2E_BIN=$PWD/target/release/aiperf cargo test -p aiperf-e2e-tests --test test_native_harbor_agentic -- --ignored --test-threads=1 --nocapture
```

Expected: every named test passes and before/after task-owned Docker resource snapshots are identical.

- [ ] **Step 5: Run the full Rust verification matrix**

```bash
cargo test -p aiperf-runtime
cargo test -p aiperf-runtime --features engine
cargo test -p aiperf-cli --test eval_command
cargo clippy --all-targets
cargo fmt --check
```

- [ ] **Step 6: Update built-current documentation and run guards**

Move delivered requirements into built-current truth, synchronize the four instruction bodies, and update `llms.txt` and the specs index.

```bash
/usr/bin/python3 tools/check_agent_files_sync.py
/usr/bin/python3 tools/check_docs_current.py
git diff --check
```

- [ ] **Step 7: Obtain final full-range Graham approval and commit**

Require explicit approval of existing-seam reuse, driver-stage cancellation, worker-local endpoint composition, secret/network authority, protocol backpressure, matrix admission/order, branch workspaces, invalid denominators, and recorded replay compatibility.

```bash
git add rust/e2e-tests AGENTS.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc llms.txt docs/specs
git commit -m "test(eval): prove native agent benchmarks end to end"
```

---

## Explicit post-release training boundary

This plan creates no training module, training DTO, optimizer/checkpoint update path, or training task. After Task 12 passes in full and immutable evaluation lineage is built-current truth, a separately reviewed implementation plan may consume frozen rollout manifests and define train/evaluation splits, checkpoint identities, optimizer-state lineage, and non-mutation enforcement. Evaluation code receives no training update authority.

## Plan self-review record

- Existing-seam check: all live programs are `GraphTraceProgram`; graph stages execute through the existing graph executor and `GraphSink`; live progression extends the current driver/turn/tool/lease seams.
- Model composition check: Task 1 defines every pinned binding field, and Task 7 resolves endpoint, transport, tokenizer, secret, retry, timeout, capture, token counter, and observer inputs before provisioning.
- Dependency check: Task 2 creates result and matrix types; Task 3 creates evaluator types; Tasks 7-11 only consume already-created interfaces. CLI routing is added only in Task 7 alongside existing `run_task` and `run_suite` functions.
- Matrix check: the scheduler and scored result shape ship in Task 2, the first real scored episode uses them in Task 3, and every subsequent profile implements `EpisodeRunner`.
- Causality check: Tasks 8 and 12 require counterfactual live responses, distinct edges/tools, observation-dependent second model requests, and distinct verifier scores.
- Training check: training implementation is absent and explicitly gated behind the complete evaluation release proof.
- Architecture check: no NativeGraph executor, model client, aggregate registry universe, global state, or major-slice enum dispatcher is introduced.
