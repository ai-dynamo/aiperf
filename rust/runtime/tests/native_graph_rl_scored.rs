// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Frozen RL rollout completion and evaluator integration contracts.

use std::{cell::RefCell, collections::BTreeMap, fs, io::Cursor, num::NonZeroUsize, rc::Rc};

use async_trait::async_trait;

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, ArtifactQuota, EnvironmentTransitionRecord,
    EpisodeArtifactStore, EpisodeComparability, EpisodeEvaluationError, EpisodeEvaluator,
    EpisodeEvaluatorFactory, EpisodeExecution, EpisodeExecutionError, EvidenceEvent, EvidenceKind,
    FrozenAttemptBundle, FrozenRolloutEvidence, HarborEpisodeEvaluator,
    HarborEpisodeEvaluatorFactory, HarborImporter, HarborSource, LocalNativeGraphSuiteScheduler,
    MatrixError, ModelIdentity, NativeGraphAttemptAuthority, NativeGraphCompletedAttempt,
    NativeGraphCompletedAttemptError, NativeGraphEpisodeExecutor, NativeGraphEpisodeRunner,
    NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, RegradeRequest,
    ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest, ResourceLimits,
    RewardDocument, RlEvaluationPolicy, RolloutEvidenceIdentity, RuntimeIdentity, ScoreVersion,
    SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec, VerifierResult, regrade,
    run_resolved_suite,
};

fn frozen_harbor_attempt(
    authority: &NativeGraphAttemptAuthority,
    reward: f64,
) -> FrozenAttemptBundle {
    let attempt = authority.attempt_id().clone();
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        vec![ArtifactDigest::from_bytes(b"declared-patch")],
        RewardDocument::parse(Some(format!(r#"{{"reward":{reward}}}"#).as_bytes()), None)
            .expect("fixture reward is valid"),
        ArtifactDigest::from_bytes(b"verifier-rationale"),
    )
    .expect("fixture verifier result is valid");
    let initial = ScoreVersion::initial(
        attempt.clone(),
        verifier.verifier.clone(),
        verifier.evidence.clone(),
        "reward",
        reward,
        ArtifactDigest::from_bytes(b"initial-rationale"),
    )
    .expect("fixture initial score is valid");
    let rescored = regrade(
        RegradeRequest::new(initial.clone(), verifier.clone(), "reward")
            .expect("fixture regrade request is valid"),
    )
    .expect("fixture regrade succeeds");
    FrozenAttemptBundle::new(
        authority.trial_digest().clone(),
        verifier,
        vec![EvidenceEvent::new(
            attempt,
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"existing-lifecycle"),
            None,
        )],
        vec![initial, rescored],
    )
    .expect("fixture Harbor facts freeze")
}

struct LegacyCompletedEvaluator;

#[async_trait(?Send)]
impl EpisodeEvaluator for LegacyCompletedEvaluator {
    async fn evaluate(
        &self,
        attempt: FrozenAttemptBundle,
    ) -> Result<aiperf_runtime::eval::EpisodeResult, aiperf_runtime::eval::EpisodeEvaluationError>
    {
        HarborEpisodeEvaluator::new().evaluate(attempt).await
    }
}

struct LegacyCompletedEvaluatorFactory;

impl EpisodeEvaluatorFactory for LegacyCompletedEvaluatorFactory {
    fn create(
        &self,
        _: &ResolvedEpisodeTrial,
    ) -> Result<Rc<dyn EpisodeEvaluator>, EpisodeEvaluationError> {
        Ok(Rc::new(LegacyCompletedEvaluator))
    }
}

struct SealedRolloutExecutor {
    completed: RefCell<Option<NativeGraphCompletedAttempt>>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeExecutor for SealedRolloutExecutor {
    async fn execute(
        &self,
        assignment: &aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError> {
        let completed = self.completed.borrow_mut().take().ok_or_else(|| {
            EpisodeExecutionError::Configuration("fixture executed twice".to_owned())
        })?;
        if completed.frozen_attempt().trial_digest() != assignment.trial_digest()
            || completed.frozen_attempt().attempt() != assignment.attempt_id()
        {
            return Err(EpisodeExecutionError::Configuration(
                "fixture completion belongs to another assignment".to_owned(),
            ));
        }
        Ok(completed)
    }
}

fn quota() -> ArtifactQuota {
    ArtifactQuota {
        max_artifacts: 4,
        max_total_bytes: 1024,
        max_artifact_bytes: 256,
        max_download_handles: 4,
    }
}

fn freeze_reference(
    store: &mut EpisodeArtifactStore,
    bytes: &[u8],
) -> aiperf_runtime::eval::FrozenArtifactReference {
    let upload = store
        .begin_upload(u64::try_from(bytes.len()).expect("fixture byte length fits"))
        .expect("fixture upload is admitted");
    store
        .write_upload(&upload, &mut Cursor::new(bytes))
        .expect("fixture upload bytes are written");
    let artifact = store
        .commit_upload(&upload)
        .expect("fixture artifact is frozen");
    store
        .issue_reference(&artifact)
        .expect("fixture reference is issued")
}

fn frozen_rollout(
    identity: RolloutEvidenceIdentity,
    terminated: bool,
    truncated: bool,
) -> FrozenRolloutEvidence {
    let root = tempfile::tempdir().expect("fixture root is created");
    let mut store = EpisodeArtifactStore::new(root.path(), quota()).expect("fixture store opens");
    let reset = freeze_reference(&mut store, b"reset");
    let action = freeze_reference(&mut store, b"action");
    let observation = freeze_reference(&mut store, b"observation");
    let info = freeze_reference(&mut store, b"info");
    let policy =
        RlEvaluationPolicy::new("environment:v1", 1, 0.5).expect("fixture policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            2.0,
            terminated,
            truncated,
            info.artifact().clone(),
        )
        .expect("fixture transition is valid")])
        .expect("fixture trajectory freezes");
    let evidence = FrozenRolloutEvidence::freeze(identity, reset, &[action], trajectory, &store)
        .expect("fixture rollout evidence freezes");
    evidence
}

fn resolved_suite(package_label: &str, run_label: &str) -> ResolvedNativeGraphSuite {
    let task = tempfile::tempdir().expect("fixture package root is created");
    fs::create_dir_all(task.path().join("environment")).expect("fixture environment exists");
    fs::create_dir_all(task.path().join("tests")).expect("fixture tests exist");
    fs::create_dir_all(task.path().join("tools")).expect("fixture tools exist");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("fixture Dockerfile writes");
    fs::write(
        task.path().join("instruction.md"),
        format!("do {package_label}\n"),
    )
    .expect("fixture instruction writes");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("fixture test writes");
    fs::write(
        task.path().join("task.toml"),
        format!(
            r#"schema_version = "1.1"

[task]
name = "example/{package_label}"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#
        ),
    )
    .expect("fixture task manifest writes");
    fs::write(task.path().join("agent_graph.json"), b"{}\n").expect("fixture graph writes");
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
    )
    .expect("fixture model bindings write");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.py"]
executable = "tools/adapter.py"
"#,
    )
    .expect("fixture adapter manifest writes");
    fs::write(task.path().join("tools/adapter.py"), b"print('adapter')\n")
        .expect("fixture adapter writes");

    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("fixture source path is accepted");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture package imports");
    let trial = TrialSpec::new(
        imported.task.clone(),
        AgentVariantRef::new("native-graph").expect("fixture agent is valid"),
        ModelIdentity::new("provider-default", "example-model").expect("fixture model is valid"),
        7,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).expect("fixture budget is valid"),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").expect("fixture runtime is valid"),
    )
    .expect("fixture trial is valid");
    NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported,
            trial,
            NonZeroUsize::new(1).expect("one repetition is nonzero"),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).expect("fixture resources are valid"),
        )
        .expect("fixture suite trial resolves"),
    ])
    .expect("fixture suite manifest resolves")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        run_label.as_bytes(),
    )))
    .expect("fixture suite expands")
}

fn resolved_authority(package_label: &str, run_label: &str) -> NativeGraphAttemptAuthority {
    let resolved = resolved_suite(package_label, run_label);
    NativeGraphAttemptAuthority::from_resolved_trial(
        resolved.trials().first().expect("fixture has one attempt"),
    )
}

fn truncated_runner_fixture(
    package_label: &str,
    run_label: &str,
) -> (ResolvedNativeGraphSuite, NativeGraphCompletedAttempt) {
    let suite = resolved_suite(package_label, run_label);
    let authority = NativeGraphAttemptAuthority::from_resolved_trial(
        suite.trials().first().expect("fixture has one attempt"),
    );
    let rollout = frozen_rollout(authority.rollout_identity(), false, true);
    let completed = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 0.0),
        Some(rollout),
    )
    .expect("fixture truncated rollout freezes");
    (suite, completed)
}

fn one_trial_scheduler() -> LocalNativeGraphSuiteScheduler {
    LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(1, 1, 64, BTreeMap::new())
            .expect("fixture scheduler resources are valid"),
    )
    .expect("fixture scheduler initializes")
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_rollout_appends_only_its_lifecycle_digest_and_retains_negative_harbor_score() {
    let authority = resolved_authority("package-a", "run-a");
    let attempt = frozen_harbor_attempt(&authority, -0.5);
    let verifier_evidence = attempt.verifier_input_evidence().to_vec();
    let reward = attempt.verifier_result().reward.clone();
    let prior_lifecycle = attempt.lifecycle_evidence()[0].identity_digest();
    let rollout = frozen_rollout(authority.rollout_identity(), true, false);
    let rollout_digest = rollout.identity_digest();

    let completed = NativeGraphCompletedAttempt::freeze(&authority, attempt, Some(rollout))
        .expect("matching rollout becomes one completed attempt");

    assert_eq!(
        completed.frozen_attempt().verifier_input_evidence(),
        verifier_evidence
    );
    assert_eq!(completed.frozen_attempt().verifier_result().reward, reward);
    assert_eq!(completed.frozen_attempt().lifecycle_evidence().len(), 2);
    let appended = &completed.frozen_attempt().lifecycle_evidence()[1];
    assert_eq!(appended.kind, EvidenceKind::Artifact);
    assert_eq!(appended.payload, rollout_digest);
    assert_eq!(appended.parent, Some(prior_lifecycle));

    let result = HarborEpisodeEvaluator::new()
        .evaluate_native_graph(completed)
        .await
        .expect("finite Harbor score remains evaluable");
    assert_eq!(result.execution(), EpisodeExecution::Completed);
    assert_eq!(result.verified_reward(), Some(-0.5));
    assert_eq!(result.comparability(), EpisodeComparability::Scored);
}

#[tokio::test(flavor = "current_thread")]
async fn truncated_rollout_maps_to_truncated_without_zeroing_its_harbor_score() {
    let authority = resolved_authority("package-a", "run-a");
    let rollout = frozen_rollout(authority.rollout_identity(), false, true);
    let completed = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 0.0),
        Some(rollout),
    )
    .expect("matching truncated rollout becomes one completed attempt");

    let result = HarborEpisodeEvaluator::new()
        .evaluate_native_graph(completed)
        .await
        .expect("zero Harbor score remains a valid score");
    assert_eq!(result.execution(), EpisodeExecution::Truncated);
    assert_eq!(result.verified_reward(), Some(0.0));
    assert_eq!(result.comparability(), EpisodeComparability::Scored);
}

#[tokio::test(flavor = "current_thread")]
async fn matrix_runner_preserves_truncated_rollout_for_harbor_evaluation() {
    let (suite, completed) = truncated_runner_fixture("package-runner-harbor", "run-runner-harbor");
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        Rc::new(SealedRolloutExecutor {
            completed: RefCell::new(Some(completed)),
        }),
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));

    let results = run_resolved_suite(&one_trial_scheduler(), suite, runner)
        .await
        .expect("the matrix runner preserves the sealed completion for Harbor evaluation");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].execution(), EpisodeExecution::Truncated);
    assert_eq!(results[0].verified_reward(), Some(0.0));
}

#[tokio::test(flavor = "current_thread")]
async fn matrix_runner_refuses_to_send_a_truncated_rollout_to_a_legacy_evaluator() {
    let (suite, completed) = truncated_runner_fixture("package-runner-legacy", "run-runner-legacy");
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        Rc::new(SealedRolloutExecutor {
            completed: RefCell::new(Some(completed)),
        }),
        Rc::new(LegacyCompletedEvaluatorFactory),
    ));

    let error = run_resolved_suite(&one_trial_scheduler(), suite, runner)
        .await
        .expect_err("a legacy evaluator cannot discard sealed truncated execution state");

    assert!(matches!(
        error,
        MatrixError::RunnerExecutionFailed(reason)
            if reason == "selected evaluator does not support sealed rollout evidence"
    ));
}

#[test]
fn imported_authority_refuses_foreign_package_rollout_before_lifecycle_append() {
    let authority = resolved_authority("package-a", "run-a");
    let foreign_authority = resolved_authority("package-b", "run-b");
    let foreign_rollout = frozen_rollout(foreign_authority.rollout_identity(), true, false);
    let attempt = frozen_harbor_attempt(&authority, 1.0);

    let error =
        NativeGraphCompletedAttempt::freeze(&authority, attempt.clone(), Some(foreign_rollout))
            .expect_err("foreign package rollout cannot append lifecycle evidence");

    assert_eq!(
        error,
        NativeGraphCompletedAttemptError::SourceIdentityMismatch
    );
    assert_eq!(attempt.lifecycle_evidence().len(), 1);
}

#[test]
fn imported_authority_cannot_bind_a_rollout_to_another_attempt() {
    let authority = resolved_authority("package-a", "run-a");
    let foreign_authority = resolved_authority("package-a", "run-b");
    let foreign_rollout = frozen_rollout(foreign_authority.rollout_identity(), true, false);
    let attempt = frozen_harbor_attempt(&authority, 1.0);

    let error = NativeGraphCompletedAttempt::freeze(
        &foreign_authority,
        attempt.clone(),
        Some(foreign_rollout),
    )
    .expect_err("an authority cannot append a rollout to a different completed attempt");

    assert_eq!(
        error,
        NativeGraphCompletedAttemptError::AttemptIdentityMismatch
    );
    assert_eq!(attempt.lifecycle_evidence().len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn legacy_evaluator_refuses_truncated_rollout_instead_of_silently_reclassifying_it() {
    let authority = resolved_authority("package-a", "run-a");
    let rollout = frozen_rollout(authority.rollout_identity(), false, true);
    let completed = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 0.0),
        Some(rollout),
    )
    .expect("matching truncated rollout freezes");

    let error = LegacyCompletedEvaluator
        .evaluate_native_graph(completed)
        .await
        .expect_err("legacy evaluators cannot discard a sealed truncated execution state");

    assert_eq!(error, EpisodeEvaluationError::RolloutAwareEvaluatorRequired);
}

#[test]
fn rollout_with_foreign_source_is_refused_before_it_can_append_lifecycle_evidence() {
    let authority = resolved_authority("package-a", "run-a");
    let identity = authority.rollout_identity();
    let foreign_identity = RolloutEvidenceIdentity::new(
        ArtifactDigest::from_bytes(b"foreign-source"),
        identity.task().clone(),
        identity.environment_implementation().clone(),
    );
    let rollout = frozen_rollout(foreign_identity, true, false);

    let error = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 1.0),
        Some(rollout),
    )
    .expect_err("foreign rollout provenance must not enter lifecycle evidence");

    assert_eq!(
        error,
        NativeGraphCompletedAttemptError::SourceIdentityMismatch
    );
}

#[test]
fn rollout_with_foreign_task_is_refused_before_it_can_append_lifecycle_evidence() {
    let authority = resolved_authority("package-a", "run-a");
    let identity = authority.rollout_identity();
    let foreign_identity = RolloutEvidenceIdentity::new(
        identity.source().clone(),
        ArtifactDigest::from_bytes(b"foreign-task"),
        identity.environment_implementation().clone(),
    );
    let rollout = frozen_rollout(foreign_identity, true, false);

    let error = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 1.0),
        Some(rollout),
    )
    .expect_err("foreign rollout task provenance must not enter lifecycle evidence");

    assert_eq!(
        error,
        NativeGraphCompletedAttemptError::TaskIdentityMismatch
    );
}

#[test]
fn rollout_with_foreign_environment_is_refused_before_it_can_append_lifecycle_evidence() {
    let authority = resolved_authority("package-a", "run-a");
    let identity = authority.rollout_identity();
    let foreign_identity = RolloutEvidenceIdentity::new(
        identity.source().clone(),
        identity.task().clone(),
        ArtifactDigest::from_bytes(b"foreign-environment"),
    );
    let rollout = frozen_rollout(foreign_identity, true, false);

    let error = NativeGraphCompletedAttempt::freeze(
        &authority,
        frozen_harbor_attempt(&authority, 1.0),
        Some(rollout),
    )
    .expect_err("foreign rollout environment provenance must not enter lifecycle evidence");

    assert_eq!(
        error,
        NativeGraphCompletedAttemptError::EnvironmentIdentityMismatch
    );
}
