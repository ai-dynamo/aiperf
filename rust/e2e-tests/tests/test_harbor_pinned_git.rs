// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::fs;
use std::path::Path;
use std::process::Command;

use aiperf_runtime::eval::{
    AgentCapability, AgentVariantRef, ArtifactDigest, AttemptId, DeclaredArtifactTransfer,
    EvalExecutionError, EvalSandboxFactory, HarborAgentContract, HarborEvaluationCoordinator,
    HarborLocalEvaluationRequest, HarborSandboxRecipe, HarborSource, LocalProcessSandbox,
    ModelIdentity, NativeSourceAcquirer, PolicyIdentity, RuntimeIdentity, TrialBudget,
    VerifierExecutionError, VerifierMode, VerifierSandboxFactory,
};

struct LocalFactory {
    opened: Cell<u32>,
}

impl EvalSandboxFactory for LocalFactory {
    fn capabilities(&self) -> &[AgentCapability] {
        const CAPABILITIES: [AgentCapability; 1] = [AgentCapability::ReadOnlyBase];
        &CAPABILITIES
    }

    fn open(&self, _: &HarborSandboxRecipe) -> Result<(), EvalExecutionError> {
        self.opened.set(self.opened.get() + 1);
        Ok(())
    }
}

struct UnusedVerifier;

impl VerifierSandboxFactory for UnusedVerifier {
    fn prepare(
        &self,
        _: VerifierMode,
        _: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError> {
        unreachable!("local process execution provisions the verifier itself")
    }
}

struct PreparingVerifier;

impl VerifierSandboxFactory for PreparingVerifier {
    fn prepare(
        &self,
        _: VerifierMode,
        _: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError> {
        Ok(())
    }
}

fn run_git<const N: usize>(repository: &Path, arguments: [&str; N]) {
    let status = Command::new("git")
        .arg("-c")
        .arg("commit.gpgsign=false")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success());
}

fn git_output<const N: usize>(repository: &Path, arguments: [&str; N]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn evaluate(
    source: HarborSource,
    attempt: &str,
) -> aiperf_runtime::eval::HarborCompletedEvaluation {
    let factory = LocalFactory {
        opened: Cell::new(0),
    };
    let verifier = UnusedVerifier;
    let coordinator = HarborEvaluationCoordinator::new(&NativeSourceAcquirer, &factory, &verifier);
    let result = coordinator
        .execute_local(
            &LocalProcessSandbox::new(),
            HarborLocalEvaluationRequest {
                source,
                recipe: HarborSandboxRecipe::new(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "/work",
                )
                .unwrap(),
                contract: HarborAgentContract::installed(vec![AgentCapability::ReadOnlyBase]),
                agent_variant: AgentVariantRef::new("pinned-installed").unwrap(),
                model: ModelIdentity::new("native", "local").unwrap(),
                seed: 99,
                policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"pinned-policy")),
                runtime: RuntimeIdentity::new("native-local").unwrap(),
                budget: TrialBudget::new(10.0, 10.0).unwrap(),
                attempt: AttemptId::new(attempt).unwrap(),
                verifier_mode: VerifierMode::Shared,
                agent_command: None,
                score_metric: "reward".to_owned(),
                initial_rationale: ArtifactDigest::from_bytes(b"initial"),
                regrade_metric: "quality".to_owned(),
                regrade_rationale: ArtifactDigest::from_bytes(b"regrade"),
            },
        )
        .unwrap();
    assert_eq!(factory.opened.get(), 1);
    result
}

#[test]
fn pinned_git_source_executes_the_recorded_revision_after_head_mutation() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/harbor_p0/pinned-git/task.json"),
        repository.join("task.json"),
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "pinned task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);
    assert_eq!(revision.len(), 40);
    assert!(
        revision
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    );
    let source =
        HarborSource::pinned_git(repository.to_string_lossy(), revision, "task.json").unwrap();
    let first = evaluate(source.clone(), "pinned-attempt");

    fs::write(
        repository.join("task.json"),
        br#"{"not":"the pinned package"}"#,
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "later task"]);
    let second = evaluate(source, "pinned-attempt");

    assert_eq!(first.imported.task, second.imported.task);
    assert_eq!(
        first.trial.identity_digest(),
        second.trial.identity_digest()
    );
    assert_eq!(
        first.verifier_result.evidence,
        second.verifier_result.evidence
    );
    assert_eq!(first.evidence, second.evidence);
    assert_eq!(
        first.initial_score.identity_digest(),
        second.initial_score.identity_digest()
    );
    assert_eq!(
        first.regraded_score.identity_digest(),
        second.regraded_score.identity_digest()
    );
}

#[test]
fn coordinator_prepares_standard_task_from_pinned_git_tree() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    let task = repository.join("standard");
    fs::create_dir_all(task.join("environment")).unwrap();
    fs::create_dir_all(task.join("tests")).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::write(
        task.join("task.toml"),
        "schema_version = \"1.0\"\n\n[task]\nname = \"example/pinned-standard\"\n",
    )
    .unwrap();
    fs::write(task.join("instruction.md"), "Fix the recorded task.\n").unwrap();
    fs::write(task.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task.join("tests/test.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    run_git(&repository, ["add", "standard"]);
    run_git(&repository, ["commit", "-m", "standard task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);

    let sandbox = LocalFactory {
        opened: Cell::new(0),
    };
    let verifier = PreparingVerifier;
    let coordinator = HarborEvaluationCoordinator::new(&NativeSourceAcquirer, &sandbox, &verifier);
    let imported = coordinator
        .prepare(
            &HarborSource::pinned_git(repository.to_string_lossy(), revision, "standard/task.toml")
                .unwrap(),
            &HarborSandboxRecipe::new(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "/work",
            )
            .unwrap(),
            &HarborAgentContract::installed(vec![]),
            VerifierMode::Shared,
            &DeclaredArtifactTransfer::new(vec![]).unwrap(),
        )
        .expect("pinned standard task tree must be retained for coordinator preparation");

    assert_eq!(sandbox.opened.get(), 1);
    assert!(imported.package.is_standard_directory());
    assert_eq!(imported.task.id.as_str(), "example/pinned-standard");
    assert_eq!(imported.package.instruction(), "Fix the recorded task.\n");
    assert_eq!(
        imported.package.verifier_command(),
        ["/bin/sh", "tests/test.sh"]
    );
}
