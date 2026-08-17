// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};
use std::fs;

use aiperf_runtime::eval::{
    AgentCapability, AgentVariantRef, ArtifactDigest, AttemptId, DeclaredArtifactTransfer,
    DockerProcessSandbox, EvalExecutionError, EvalSandboxFactory, HarborAgentContract,
    HarborEvaluationCoordinator, HarborImportError, HarborImporter, HarborLocalEvaluationRequest,
    HarborSandboxRecipe, HarborSource, LocalProcessSandbox, ModelIdentity, NativeSourceAcquirer,
    PolicyIdentity, RuntimeIdentity, SandboxRole, SourceAcquirer, TrialBudget,
    VerifierExecutionError, VerifierMode, VerifierSandboxFactory, WorkspaceOverlay,
};

struct RecordingFactory {
    opened: Cell<bool>,
    capabilities: Vec<AgentCapability>,
}

struct StaticAcquirer {
    bytes: Vec<u8>,
}

impl SourceAcquirer for StaticAcquirer {
    fn acquire(&self, _: &HarborSource) -> Result<Vec<u8>, HarborImportError> {
        Ok(self.bytes.clone())
    }
}

#[derive(Default)]
struct RecordingVerifier {
    modes: RefCell<Vec<VerifierMode>>,
    artifacts: RefCell<Vec<Vec<(String, ArtifactDigest)>>>,
}

impl VerifierSandboxFactory for RecordingVerifier {
    fn prepare(
        &self,
        mode: VerifierMode,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError> {
        self.modes.borrow_mut().push(mode);
        self.artifacts.borrow_mut().push(artifacts.to_vec());
        Ok(())
    }
}

impl EvalSandboxFactory for RecordingFactory {
    fn capabilities(&self) -> &[AgentCapability] {
        &self.capabilities
    }

    fn open(&self, _: &HarborSandboxRecipe) -> Result<(), EvalExecutionError> {
        self.opened.set(true);
        Ok(())
    }
}

#[test]
fn missing_overlay_capability_refuses_before_environment_open() {
    let factory = RecordingFactory {
        opened: std::cell::Cell::new(false),
        capabilities: vec![AgentCapability::ReadOnlyBase],
    };
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let contract = HarborAgentContract::installed(vec![AgentCapability::OverlayWorkspace]);

    assert!(factory.preflight(&recipe, &contract).is_err());
    assert!(!factory.opened.get());
}

#[test]
fn branches_return_immutable_patch_without_mutating_canonical_workspace() {
    let canonical = WorkspaceOverlay::canonical(
        "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .unwrap();
    let branch = canonical.branch("fix").unwrap();
    let patch = branch
        .complete("blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        .unwrap();

    assert_eq!(
        canonical.base_digest().as_str(),
        "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(patch.parent_digest(), canonical.base_digest());
}

#[test]
fn native_coordinator_imports_preflights_opens_and_prepares_declared_verifier() {
    let acquirer = StaticAcquirer {
        bytes: br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","true"],"verifier_command":["sh","-c","true"],"declared_artifacts":["/results/patch.diff"]}"#.to_vec(),
    };
    let sandbox = RecordingFactory {
        opened: Cell::new(false),
        capabilities: vec![AgentCapability::ReadOnlyBase],
    };
    let verifier = RecordingVerifier::default();
    let coordinator = HarborEvaluationCoordinator::new(&acquirer, &sandbox, &verifier);
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let transfer = DeclaredArtifactTransfer::new(vec![(
        "/results/patch.diff",
        ArtifactDigest::parse(format!("blake3:{}", "a".repeat(64))).unwrap(),
    )])
    .unwrap();

    let imported = coordinator
        .prepare(
            &HarborSource::local("task.json").unwrap(),
            &recipe,
            &HarborAgentContract::External {
                required: vec![AgentCapability::ReadOnlyBase],
            },
            VerifierMode::Separate,
            &transfer,
        )
        .unwrap();

    assert_eq!(imported.task.id.as_str(), "repair-1");
    assert!(sandbox.opened.get());
    assert_eq!(*verifier.modes.borrow(), vec![VerifierMode::Separate]);
    assert_eq!(
        *verifier.artifacts.borrow(),
        vec![transfer.artifacts().to_vec()]
    );
}

#[test]
fn local_process_sandbox_materializes_package_clears_environment_and_isolates_verifier() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","true"],"verifier_command":["sh","-c","true"],"declared_artifacts":["/results/patch.diff"]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let sandbox = LocalProcessSandbox::new();
    let agent = sandbox
        .materialize(&recipe, &imported.package, SandboxRole::Agent)
        .unwrap();
    let verifier = sandbox
        .materialize(&recipe, &imported.package, SandboxRole::SeparateVerifier)
        .unwrap();

    unsafe { std::env::set_var("AIPERF_EVAL_AMBIENT_SECRET", "do-not-leak") };
    let output = agent
        .run(
            &[
                "sh".to_owned(),
                "-c".to_owned(),
                "test -z \"$AIPERF_EVAL_AMBIENT_SECRET\" && test \"$AIPERF_EVAL_MARKER\" = set"
                    .to_owned(),
            ],
            &[("AIPERF_EVAL_MARKER".to_owned(), "set".to_owned())],
        )
        .unwrap();
    unsafe { std::env::remove_var("AIPERF_EVAL_AMBIENT_SECRET") };

    fs::write(agent.root().join("agent-only.txt"), "agent").unwrap();
    assert!(output.status.success());
    assert_eq!(fs::read(agent.root().join("task.json")).unwrap(), package);
    assert_ne!(agent.root(), verifier.root());
    assert!(!verifier.root().join("agent-only.txt").exists());
}

#[test]
fn local_process_sandbox_discards_phase_output() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["true"],"declared_artifacts":[]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let sandbox = LocalProcessSandbox::new();
    let materialized = sandbox
        .materialize(&recipe, &imported.package, SandboxRole::Agent)
        .unwrap();

    let output = materialized
        .run(
            &["sh".to_owned(), "-c".to_owned(), "printf hidden".to_owned()],
            &[],
        )
        .unwrap();

    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn local_process_sandbox_runs_agent_transfers_declared_artifacts_and_parses_verifier_reward() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let result = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Shared)
        .unwrap();

    assert_eq!(result.artifacts.len(), 1);
    assert_eq!(result.reward.metrics["reward"], 1.0);
    let score = result
        .initial_score(
            AttemptId::new("native-attempt").unwrap(),
            "reward",
            ArtifactDigest::from_bytes(b"native verifier rationale"),
        )
        .unwrap();
    assert_eq!(score.version, 0);
    assert_eq!(score.value, 1.0);
}

#[test]
fn local_process_sandbox_rejects_an_oversized_declared_artifact() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","head -c 1048577 /dev/zero > results/patch.diff"],"verifier_command":["true"],"declared_artifacts":["/results/patch.diff"]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let error = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Shared)
        .expect_err("local artifact collection must reject files beyond the host safety cap");

    assert!(matches!(
        error,
        EvalExecutionError::ArtifactCollection(message) if message.contains("maximum size")
    ));
}

#[test]
fn local_process_sandbox_rejects_an_oversized_verifier_reward() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["sh","-c","head -c 1048577 /dev/zero > reward.json"],"declared_artifacts":[]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let error = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Shared)
        .expect_err("local reward collection must reject files beyond the host safety cap");

    assert!(matches!(
        error,
        EvalExecutionError::ArtifactCollection(message) if message.contains("maximum size")
    ));
}

#[test]
fn local_process_sandbox_refuses_a_separate_verifier_before_running_the_agent() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","exit 91"],"verifier_command":["sh","-c","true"],"declared_artifacts":[]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    assert_eq!(
        LocalProcessSandbox::new().execute(&recipe, &imported.package, VerifierMode::Separate),
        Err(EvalExecutionError::UnsupportedEnforcement(
            "separate verifier isolation"
        ))
    );
}

#[test]
fn local_process_sandbox_refuses_separate_verifier_isolation() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf secret > \"$AIPERF_EVAL_ROOT/agent-secret\"; printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test ! -e agent-secret && test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let error = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Separate)
        .expect_err("a local process root is not a secure separate verifier provider");

    assert_eq!(
        error,
        EvalExecutionError::UnsupportedEnforcement("separate verifier isolation")
    );
}

#[test]
fn local_process_sandbox_materializes_the_imported_directory_package_after_origin_removal() {
    let temporary = tempfile::tempdir().unwrap();
    let package_root = temporary.path().join("package");
    fs::create_dir_all(package_root.join("fixtures/empty")).unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","test \"$(cat fixtures/input.txt)\" = original && test -d fixtures/empty && test -x fixtures/helper.sh"],"verifier_command":["sh","-c","printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":[]}"#;
    fs::write(package_root.join("task.json"), package).unwrap();
    fs::write(package_root.join("fixtures/input.txt"), "original").unwrap();
    fs::write(package_root.join("fixtures/helper.sh"), "#!/bin/sh\n").unwrap();
    let mut helper_permissions = fs::metadata(package_root.join("fixtures/helper.sh"))
        .unwrap()
        .permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut helper_permissions, 0o755);
    fs::set_permissions(package_root.join("fixtures/helper.sh"), helper_permissions).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(package_root.to_string_lossy()).unwrap())
        .unwrap();
    fs::remove_dir_all(&package_root).unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let result = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Shared)
        .unwrap();

    assert_eq!(result.reward.metrics["reward"], 1.0);
}

#[test]
fn coordinator_completed_local_execution_constructs_trial_scores_and_ordered_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0,\"quality\":0.75}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#;
    let package_path = temporary.path().join("task.json");
    fs::write(&package_path, package).unwrap();
    let sandbox = RecordingFactory {
        opened: Cell::new(false),
        capabilities: vec![AgentCapability::ReadOnlyBase],
    };
    let verifier = RecordingVerifier::default();
    let coordinator = HarborEvaluationCoordinator::new(&NativeSourceAcquirer, &sandbox, &verifier);
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let request = HarborLocalEvaluationRequest {
        source: HarborSource::local(package_path.to_string_lossy()).unwrap(),
        recipe,
        contract: HarborAgentContract::installed(vec![AgentCapability::ReadOnlyBase]),
        agent_variant: AgentVariantRef::new("installed").unwrap(),
        model: ModelIdentity::new("native", "local").unwrap(),
        seed: 7,
        policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        runtime: RuntimeIdentity::new("native-local").unwrap(),
        budget: TrialBudget::new(30.0, 30.0).unwrap(),
        attempt: AttemptId::new("completed-local").unwrap(),
        verifier_mode: VerifierMode::Shared,
        agent_command: None,
        score_metric: "reward".to_owned(),
        initial_rationale: ArtifactDigest::from_bytes(b"initial"),
        regrade_metric: "quality".to_owned(),
        regrade_rationale: ArtifactDigest::from_bytes(b"regrade"),
    };

    let completed = coordinator
        .execute_local(&LocalProcessSandbox::new(), request)
        .unwrap();

    assert_eq!(completed.trial.task.id.as_str(), "repair-1");
    assert_eq!(completed.initial_score.value, 1.0);
    assert_eq!(completed.regraded_score.value, 0.75);
    assert_eq!(
        completed.regraded_score.predecessor,
        Some(completed.initial_score.identity_digest())
    );
    assert_eq!(
        completed
            .evidence
            .iter()
            .map(|event| event.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["sandbox", "agent", "artifact", "evaluator"]
    );
    assert!(sandbox.opened.get());
}

#[test]
fn docker_process_requires_a_standard_task_directory() {
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["true"],"declared_artifacts":[]}"#;
    let imported = HarborImporter::new(&StaticAcquirer {
        bytes: package.to_vec(),
    })
    .import(&HarborSource::local("task.json").unwrap())
    .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    assert!(
        DockerProcessSandbox::new()
            .execute(
                &recipe,
                &imported.package,
                &["true".to_owned()],
                VerifierMode::Shared,
            )
            .is_err()
    );
}
