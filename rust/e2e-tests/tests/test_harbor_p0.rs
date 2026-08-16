// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};
use std::fs;

use aiperf_runtime::eval::{
    AgentCapability, ArtifactDigest, AttemptId, DeclaredArtifactTransfer, EvalExecutionError,
    EvalSandboxFactory, HarborAgentContract, HarborEvaluationCoordinator, HarborImporter,
    HarborSandboxRecipe, HarborSource, LocalProcessSandbox, NativeSourceAcquirer, RegradeRequest,
    RewardDocument, ScoreVersion, VerifierExecutionError, VerifierMode, VerifierResult,
    VerifierSandboxFactory, regrade,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

struct Sandbox {
    opened: Cell<u32>,
}

impl EvalSandboxFactory for Sandbox {
    fn capabilities(&self) -> &[AgentCapability] {
        const CAPABILITIES: [AgentCapability; 1] = [AgentCapability::ReadOnlyBase];
        &CAPABILITIES
    }

    fn open(&self, _: &HarborSandboxRecipe) -> Result<(), EvalExecutionError> {
        self.opened.set(self.opened.get() + 1);
        Ok(())
    }
}

#[derive(Default)]
struct Verifier {
    received: RefCell<Vec<(VerifierMode, Vec<(String, ArtifactDigest)>)>>,
}

impl VerifierSandboxFactory for Verifier {
    fn prepare(
        &self,
        mode: VerifierMode,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError> {
        self.received.borrow_mut().push((mode, artifacts.to_vec()));
        Ok(())
    }
}

#[test]
fn local_harbor_task_runs_through_native_p0_lifecycle_without_harbor_runtime() {
    let temporary = tempfile::tempdir().unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix the test","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","true"],"verifier_command":["sh","-c","true"],"declared_artifacts":["/results/patch.diff"]}"#;
    let package_path = temporary.path().join("task.json");
    fs::write(&package_path, package).unwrap();
    let acquirer = NativeSourceAcquirer;
    let sandbox = Sandbox {
        opened: Cell::new(0),
    };
    let verifier = Verifier::default();
    let coordinator = HarborEvaluationCoordinator::new(&acquirer, &sandbox, &verifier);
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let transfer =
        DeclaredArtifactTransfer::new(vec![("/results/patch.diff", digest('c'))]).unwrap();

    let imported = coordinator
        .prepare(
            &HarborSource::local(package_path.to_string_lossy()).unwrap(),
            &recipe,
            &HarborAgentContract::External {
                required: vec![AgentCapability::ReadOnlyBase],
            },
            VerifierMode::Shared,
            &transfer,
        )
        .unwrap();
    let original = ScoreVersion::initial(
        AttemptId::new("p0-attempt").unwrap(),
        digest('d'),
        vec![imported.report.source_digest.clone()],
        "reward",
        0.0,
        digest('e'),
    )
    .unwrap();
    let result = VerifierResult::new(
        original.attempt.clone(),
        digest('f'),
        transfer
            .artifacts()
            .iter()
            .map(|(_, artifact)| artifact.clone())
            .collect(),
        RewardDocument::parse(Some(br#"{"reward":1.0,"quality":0.75}"#), Some(b"9.0")).unwrap(),
        digest('1'),
    )
    .unwrap();

    let regraded =
        regrade(RegradeRequest::new(original.clone(), result, "reward").unwrap()).unwrap();

    assert_eq!(
        imported.report.source_digest,
        ArtifactDigest::from_bytes(package)
    );
    assert_eq!(sandbox.opened.get(), 1);
    assert_eq!(
        *verifier.received.borrow(),
        vec![(VerifierMode::Shared, transfer.artifacts().to_vec())]
    );
    assert_eq!(regraded.value, 1.0);
    assert_eq!(regraded.predecessor, Some(original.identity_digest()));
    assert_eq!(regraded.evidence, vec![digest('c')]);
}

#[test]
fn local_harbor_package_executes_agent_and_separate_verifier_without_harbor_runtime() {
    let temporary = tempfile::tempdir().unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix the test","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#;
    let package_path = temporary.path().join("task.json");
    fs::write(&package_path, package).unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(package_path.to_string_lossy()).unwrap())
        .unwrap();

    let execution = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Separate)
        .unwrap();
    let original = ScoreVersion::initial(
        AttemptId::new("p0-real-attempt").unwrap(),
        ArtifactDigest::parse(imported.package.verifier()).unwrap(),
        execution
            .artifacts
            .iter()
            .map(|(_, digest)| digest.clone())
            .collect(),
        "reward",
        execution.reward.metrics["reward"],
        digest('2'),
    )
    .unwrap();
    let result = VerifierResult::new(
        original.attempt.clone(),
        ArtifactDigest::parse(imported.package.verifier()).unwrap(),
        execution
            .artifacts
            .iter()
            .map(|(_, digest)| digest.clone())
            .collect(),
        execution.reward,
        digest('3'),
    )
    .unwrap();

    let regraded = regrade(RegradeRequest::new(original, result, "reward").unwrap()).unwrap();

    assert_eq!(execution.artifacts[0].0, "/results/patch.diff");
    assert_eq!(regraded.value, 1.0);
}
