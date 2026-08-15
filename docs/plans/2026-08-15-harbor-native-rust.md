# Harbor Native-Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the native-Rust Harbor replacement through immutable evaluation contracts, native import/execution/verifier/regrade paths, semantic experiments, and P1/P2 extension points.

**Architecture:** A dedicated `eval` domain owns immutable identity, evidence, score, and import-report data. Acquisition, sandbox/agent execution, verification, and semantic experiments consume that domain but do not duplicate its identity authority. Existing Graph-IR/replay/cellular facilities remain execution infrastructure; final Harbor evidence is append-only and independent of replay terminal summaries.

**Tech Stack:** Rust 2024; Serde strict DTOs; BLAKE3; Tokio current-thread/`LocalSet`; injected process/container/provider fakes; Cargo integration tests.

**Spec:** `docs/specs/harbor-native-rust-implementation.md`, `docs/specs/harbor-replacement-platform.md`, `docs/specs/agentic-eval-platform.md`, and `docs/specs/semantic-agent-graph.md`.

## Global Constraints

- All Harbor execution is pure native Rust; no Harbor Python runtime, library, wrapper, bridge, or dependency participates in execution.
- The `eval` namespace must not collide with cellular transport `DatasetManifest`.
- Public DTOs use strict decoding and documented public fields; unsupported source, lowering, transform, and provider-capability paths return typed refusals rather than fallbacks.
- The resolved trial identity pins source, task, agent/graph, model, seed, policy, resource budget, environment, verifier, and runtime identities.
- Controller-only replay artifact folds remain infrastructure, not the authority for append-only Harbor attempt evidence.
- Separate verifiers use a fresh sandbox or restored snapshot and receive only declared artifacts and permitted evidence.
- Worker-local execution uses `Rc`/`RefCell` and `spawn_local`; do not put `Arc<Mutex<_>>` or locks across awaits on hot paths.
- Commit each logical file-scoped slice before compiling or testing; never amend, reset, checkout-discard, clean, rebase, or use stash.

---

## File structure

- `rust/runtime/src/eval/{mod,identity,source,import_report,task,trial,evidence,score}.rs`: immutable evaluation vocabulary and digest/serialization rules.
- `rust/runtime/src/eval/import/{mod,acquire,harbor,normalize}.rs`: local, pinned-Git, and registry-reference acquisition and Harbor-compatible normalization.
- `rust/runtime/src/eval/execution/{mod,recipe,agent,workspace,attempt}.rs`: recipe resolution, agent contracts, overlays, capability preflight, and attempt lifecycle.
- `rust/runtime/src/eval/verifier/{mod,artifacts,reward,regrade}.rs`: verifier isolation, declared artifact transfer, reward parsing, and score versioning.
- `rust/runtime/src/eval/semantic/{mod,lowering,comparison}.rs`: semantic graph fidelity/lowering and paired comparison reports.
- `rust/runtime/tests/{eval_identity,harbor_import,eval_execution,eval_verifier,eval_regrade,eval_semantic}.rs`: deterministic contract/unit coverage.
- `rust/e2e-tests/tests/{test_harbor_p0,test_harbor_pinned_git,test_harbor_verifier_isolation}.rs`: P0 product acceptance.

### Task 1: Immutable evaluation domain

**Files:**
- Create: `rust/runtime/src/eval/{mod.rs,identity.rs,source.rs,import_report.rs,task.rs,trial.rs,evidence.rs,score.rs}`
- Modify: `rust/runtime/src/lib.rs`
- Test: `rust/runtime/tests/eval_identity.rs`

**Interfaces:**
- Produces `EvalTaskId`, `EvalDatasetId`, `EvalTaskRef`, `EvalDatasetManifest`, `TaskSpec`, `TrialSpec`, `AttemptId`, `EvidenceEvent`, `ImportReport`, `ImportDisposition`, `ScoreVersion`, and `Blake3Digest`-validated constructors.
- `TrialSpec::identity_digest() -> Blake3Digest` must serialize its complete resolved identity in canonical field order.

- [ ] **Step 1: Write failing strict identity tests**

```rust
#[test]
fn equal_resolved_trials_have_one_digest_and_changed_seed_changes_it() {
    let trial = test_trial(7);
    assert_eq!(trial.identity_digest(), trial.clone().identity_digest());
    assert_ne!(trial.identity_digest(), test_trial(8).identity_digest());
}

#[test]
fn import_report_rejects_unknown_disposition() {
    assert!(serde_json::from_str::<ImportReport>(r#"{"disposition":"bridge"}"#).is_err());
}
```

- [ ] **Step 2: Run the red test**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_identity`

Expected: FAIL because `eval` types are absent.

- [ ] **Step 3: Implement strict immutable DTOs**

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialSpec {
    pub task: EvalTaskRef,
    pub agent: AgentVariantRef,
    pub model: ModelIdentity,
    pub seed: u64,
    pub policy: PolicyIdentity,
    pub budget: TrialBudget,
    pub environment: ArtifactDigest,
    pub verifier: ArtifactDigest,
    pub runtime: RuntimeIdentity,
}
```

Implement validating constructors so empty IDs, malformed digests, and non-finite budget values fail. Model attempts and scores as append-only identity-bearing DTOs.

- [ ] **Step 4: Run green tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_identity && cargo fmt -p aiperf-runtime --check`

Expected: PASS; strict decode, canonical trial identity, and append-only score identity are covered.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval rust/runtime/src/lib.rs rust/runtime/tests/eval_identity.rs
git commit -m "feat(eval): add immutable Harbor evaluation identity"
```

### Task 2: Source acquisition and Harbor importer

**Files:**
- Create: `rust/runtime/src/eval/import/{mod.rs,acquire.rs,harbor.rs,normalize.rs}`
- Modify: `rust/runtime/src/eval/mod.rs`
- Test: `rust/runtime/tests/harbor_import.rs`
- Test fixtures: `rust/runtime/tests/fixtures/harbor/{valid,unsupported}/`

**Interfaces:**
- Consumes Task 1 `TaskSpec`, `ImportReport`, and source artifact types.
- Produces `HarborSource`, `SourceAcquirer`, `HarborImporter`, and `ImportedTask`.
- `HarborImporter::import(source) -> Result<ImportedTask, HarborImportError>` preserves source bytes before normalization.

- [ ] **Step 1: Write failing importer tests**

```rust
#[test]
fn local_import_preserves_source_digest_and_normalizes_task() { /* fixture assertions */ }

#[test]
fn unsupported_semantics_return_report_before_provisioning() { /* no sandbox call */ }
```

- [ ] **Step 2: Run the red test**

Run: `cd rust && cargo test -p aiperf-runtime --test harbor_import`

Expected: FAIL because native acquisition/import is absent.

- [ ] **Step 3: Implement local, pinned-Git, and registry-reference acquisition**

Use injected acquisition traits for Git/registry access. Copy bytes into immutable source artifacts, resolve pinned Git revisions, reject mutable refs, and normalize task instruction, environment, verifier, artifacts, policies, and agent declarations. Emit `native`, `lossless_normalized`, `lossy_normalized`, or `unsupported` reports; unsupported returns before execution construction.

- [ ] **Step 4: Run green tests**

Run: `cd rust && cargo test -p aiperf-runtime --test harbor_import`

Expected: PASS; byte preservation, pinned revision identity, and pre-provisioning refusal are proven.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/import rust/runtime/src/eval/mod.rs rust/runtime/tests/harbor_import.rs rust/runtime/tests/fixtures/harbor
git commit -m "feat(eval): import Harbor compatible task packages"
```

### Task 3: Sandbox recipes and agent contracts

**Files:**
- Create: `rust/runtime/src/eval/execution/{mod.rs,recipe.rs,agent.rs,workspace.rs,attempt.rs}`
- Modify: `rust/runtime/src/eval/mod.rs`
- Test: `rust/runtime/tests/eval_execution.rs`

**Interfaces:**
- Consumes `TaskSpec`, `TrialSpec`, and imported task artifacts.
- Produces `HarborSandboxRecipe`, `HarborAgentContract::{External,Installed,NativeGraph}`, `EvalSandboxFactory`, `WorkspaceOverlay`, and `AttemptRunner`.
- `EvalSandboxFactory::preflight(&HarborSandboxRecipe, &HarborAgentContract) -> Result<(), EvalExecutionError>` is called before provisioning.

- [ ] **Step 1: Write failing preflight/isolation tests**

```rust
#[test]
fn missing_overlay_capability_refuses_before_environment_open() { /* fake factory */ }

#[tokio::test(flavor = "current_thread")]
async fn branches_return_immutable_patch_without_mutating_canonical_workspace() { /* assert */ }
```

- [ ] **Step 2: Run the red test**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_execution`

Expected: FAIL because Harbor execution contracts are absent.

- [ ] **Step 3: Implement recipes, contracts, and overlays**

Require explicit image digest, mount/workdir/interpreter/setup, resource/network/secret/cleanup policy, and capability requirements. Implement external and installed adapters behind worker-local traits. Make branch output an immutable patch/artifact reference; only explicit selection may update canonical state.

- [ ] **Step 4: Run green tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_execution`

Expected: PASS; preflight happens before open, overlays isolate branches, and external/installed contracts use the same policy checks.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/execution rust/runtime/src/eval/mod.rs rust/runtime/tests/eval_execution.rs
git commit -m "feat(eval): execute Harbor task agent contracts"
```

### Task 4: Verifier isolation, rewards, evidence, and regrade

**Files:**
- Create: `rust/runtime/src/eval/verifier/{mod.rs,artifacts.rs,reward.rs,regrade.rs}`
- Modify: `rust/runtime/src/eval/{evidence.rs,score.rs,mod.rs}`
- Test: `rust/runtime/tests/{eval_verifier,eval_regrade}.rs`

**Interfaces:**
- Consumes Task 3 attempt/sandbox references and Task 1 evidence/score types.
- Produces `VerifierMode::{Shared,Separate}`, `DeclaredArtifactTransfer`, `RewardDocument`, `VerifierResult`, and `RegradeRequest`.
- `regrade(request) -> Result<ScoreVersion, RegradeError>` appends a new score identity.

- [ ] **Step 1: Write failing isolation/reward/regrade tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn separate_verifier_receives_declared_artifacts_not_agent_secret_or_workspace() { /* fake sandbox */ }

#[test]
fn reward_json_precedes_reward_txt_and_preserves_multiple_metrics() { /* exact fields */ }

#[test]
fn regrade_appends_score_without_changing_original_attempt() { /* identities */ }
```

- [ ] **Step 2: Run the red tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_verifier --test eval_regrade`

Expected: FAIL because no verifier/regrade path exists.

- [ ] **Step 3: Implement verifier boundary and score versioning**

Materialize only declared artifacts at declared verifier paths. Restore a distinct snapshot or provision a fresh verifier sandbox for separate mode. Parse finite `reward.json` first, then `reward.txt`; preserve metric names and record malformed input as typed verifier evidence. Append regrade score versions against pinned verifier/evidence digests.

- [ ] **Step 4: Run green tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_verifier --test eval_regrade`

Expected: PASS; secret/workspace isolation, artifact paths, precedence, and immutability hold.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/verifier rust/runtime/src/eval/{evidence.rs,score.rs,mod.rs} rust/runtime/tests/{eval_verifier.rs,eval_regrade.rs}
git commit -m "feat(eval): verify Harbor attempts and regrade evidence"
```

### Task 5: Semantic lowering and paired experiments

**Files:**
- Create: `rust/runtime/src/eval/semantic/{mod.rs,lowering.rs,comparison.rs}`
- Modify: `rust/runtime/src/eval/mod.rs`
- Test: `rust/runtime/tests/eval_semantic.rs`

**Interfaces:**
- Consumes `TrialSpec`, immutable attempt evidence, and the existing narrow `GraphTraceProgram`.
- Produces `SemanticGraph`, `FidelityReport`, `lower_semantic_graph`, `PairedComparisonSpec`, and `PairedComparisonReport`.
- `lower_semantic_graph` is fallible and never substitutes a different graph.

- [ ] **Step 1: Write failing fidelity/comparison tests**

```rust
#[test]
fn unsupported_semantic_node_returns_typed_fidelity_refusal() { /* no fallback */ }

#[test]
fn paired_report_rejects_changed_baseline_dimensions() { /* seed/policy/image */ }
```

- [ ] **Step 2: Run the red test**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_semantic`

Expected: FAIL because the Harbor semantic boundary is absent.

- [ ] **Step 3: Implement semantic envelope and paired-report guard**

Represent source semantics separately from executable nodes; emit explicit fidelity/capability outcomes. Permit paired reports only when task, model, seed, policy, image, and budget match. Report quality, cost, latency, critical-path, token, and tool deltas independently.

- [ ] **Step 4: Run green tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_semantic`

Expected: PASS; source fidelity and baseline locking are enforced.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval/semantic rust/runtime/src/eval/mod.rs rust/runtime/tests/eval_semantic.rs
git commit -m "feat(eval): compare semantic Harbor graph variants"
```

### Task 6: P0 product acceptance

**Files:**
- Create: `rust/e2e-tests/tests/{test_harbor_p0.rs,test_harbor_pinned_git.rs,test_harbor_verifier_isolation.rs}`
- Create: `rust/e2e-tests/fixtures/harbor_p0/`
- Modify: `docs/specs/{harbor-native-rust-implementation.md,harbor-replacement-platform.md,README.md}`, `llms.txt`

**Interfaces:**
- Consumes Tasks 1–5 public native contracts.
- Proves P0 acceptance for local/pinned source, agent/verifier modes, isolation, exact artifacts, reward handling, deterministic identity, regrade, and paired reports.

- [ ] **Step 1: Add failing native P0 product cases**

```rust
#[tokio::test]
async fn local_harbor_task_runs_without_harbor_runtime() { /* external or installed fake */ }

#[tokio::test]
async fn pinned_git_source_reproduces_trial_and_artifact_manifest_identity() { /* exact digests */ }

#[tokio::test]
async fn separate_verifier_cannot_observe_agent_credentials_or_undeclared_workspace() { /* denied reads */ }
```

- [ ] **Step 2: Run the red P0 suite**

Run: `cd rust && cargo test -p aiperf-e2e-tests --test test_harbor_p0 --test test_harbor_pinned_git --test test_harbor_verifier_isolation`

Expected: FAIL until all native P0 pieces compose.

- [ ] **Step 3: Wire product composition and fixtures**

Run each fixture through native importer, trial resolution, fake external/installed agent, verifier, evidence writer, score/regrade, and paired comparison. Assert that no process attempts to invoke Harbor.

- [ ] **Step 4: Run P0 verification and documentation checks**

Run:

```bash
cd rust
cargo fmt -p aiperf-runtime --check
cargo clippy -p aiperf-runtime --all-targets --features engine
cargo test -p aiperf-runtime
cargo test -p aiperf-e2e-tests --test test_harbor_p0 --test test_harbor_pinned_git --test test_harbor_verifier_isolation
cd ..
/usr/bin/python3 tools/check_agent_files_sync.py
/usr/bin/python3 tools/check_docs_current.py
```

Expected: PASS; documents describe only implemented P0 behavior.

- [ ] **Step 5: Commit**

```bash
git add rust/e2e-tests docs/specs llms.txt AGENTS.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -m "test: prove native Rust Harbor P0"
```

### Task 7: P1/P2 extension contracts

**Files:**
- Create: `rust/runtime/src/eval/{health.rs,provider.rs,registry.rs,training.rs}`
- Modify: `rust/runtime/src/eval/mod.rs`
- Test: `rust/runtime/tests/{eval_health,eval_provider,eval_registry,eval_training}.rs`

**Interfaces:**
- Consumes immutable Task/Trial/Attempt/Evidence contracts from Task 1.
- Produces task-health/quarantine records, provider capability negotiation, registry publication references, and trajectory-export manifests.

- [ ] **Step 1: Write failing extension contract tests**

```rust
#[test]
fn provider_without_required_capability_is_refused_before_trial_start() { /* typed refusal */ }

#[test]
fn local_manifest_remains_valid_when_registry_is_offline() { /* no network */ }

#[test]
fn trajectory_export_references_immutable_attempt_evidence() { /* digest only */ }
```

- [ ] **Step 2: Run the red extension tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_health --test eval_provider --test eval_registry --test eval_training`

Expected: FAIL because P1/P2 contracts are absent.

- [ ] **Step 3: Implement non-executing P1/P2 contract layer**

Add strict types and validation only: task-health/quarantine records, provider capability matching, offline-safe registry references, and trajectory export manifests. Do not add an online registry dependency or allow these types to mutate existing attempts.

- [ ] **Step 4: Run green extension tests**

Run: `cd rust && cargo test -p aiperf-runtime --test eval_health --test eval_provider --test eval_registry --test eval_training`

Expected: PASS; P1/P2 contracts compose over immutable P0 records without changing P0 behavior.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/eval rust/runtime/tests/{eval_health.rs,eval_provider.rs,eval_registry.rs,eval_training.rs}
git commit -m "feat(eval): add Harbor P1 P2 extension contracts"
```

## Plan self-review

- Spec coverage: Tasks 1–4 implement immutable identity/import/execution/verifier/regrade; Task 5 implements semantic fidelity and paired reports; Task 6 proves every P0 acceptance requirement; Task 7 provides strict P1/P2 contracts without making online infrastructure a P0 dependency.
- Placeholder scan: each task has concrete source/test paths, named types, failure tests, commands, and commits.
- Type consistency: `TaskSpec`, `TrialSpec`, `Attempt`, `ImportReport`, `ScoreVersion`, and `GraphTraceProgram` retain the same ownership boundary across tasks.
