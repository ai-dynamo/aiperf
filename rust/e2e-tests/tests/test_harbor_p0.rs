// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::path::{Path, PathBuf};

use aiperf_runtime::eval::{
    AgentCapability, AgentVariantRef, ArtifactDigest, AttemptId, EvalExecutionError,
    EvalSandboxFactory, HarborAgentContract, HarborCompletedEvaluation,
    HarborEvaluationCoordinator, HarborLocalEvaluationRequest, HarborSandboxRecipe, HarborSource,
    LocalProcessSandbox, ModelIdentity, NativeSourceAcquirer, PairedComparisonSpec,
    PairedMeasurements, PolicyIdentity, RuntimeIdentity, TrialBudget, VerifierExecutionError,
    VerifierMode, VerifierSandboxFactory,
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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/harbor_p0")
        .join(name)
}

fn request(
    source: &Path,
    attempt: &str,
    contract: HarborAgentContract,
    command: Option<Vec<String>>,
) -> HarborLocalEvaluationRequest {
    HarborLocalEvaluationRequest {
        source: HarborSource::local(source.to_string_lossy()).unwrap(),
        recipe: HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap(),
        contract,
        agent_variant: AgentVariantRef::new("native-agent").unwrap(),
        model: ModelIdentity::new("native", "local").unwrap(),
        seed: 7,
        policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"native-p0-policy")),
        runtime: RuntimeIdentity::new("native-local").unwrap(),
        budget: TrialBudget::new(30.0, 30.0).unwrap(),
        attempt: AttemptId::new(attempt).unwrap(),
        verifier_mode: VerifierMode::Shared,
        agent_command: command,
        score_metric: "reward".to_owned(),
        initial_rationale: ArtifactDigest::from_bytes(b"initial score rationale"),
        regrade_metric: "quality".to_owned(),
        regrade_rationale: ArtifactDigest::from_bytes(b"regrade rationale"),
    }
}

fn execute(request: HarborLocalEvaluationRequest) -> HarborCompletedEvaluation {
    let factory = LocalFactory {
        opened: Cell::new(0),
    };
    let verifier = UnusedVerifier;
    let coordinator = HarborEvaluationCoordinator::new(&NativeSourceAcquirer, &factory, &verifier);
    let completed = coordinator
        .execute_local(&LocalProcessSandbox::new(), request)
        .unwrap();
    assert_eq!(factory.opened.get(), 1);
    completed
}

#[test]
fn native_coordinator_executes_installed_and_external_agents_with_score_lineage_and_pairing() {
    let installed = execute(request(
        &fixture("local-installed"),
        "installed-attempt",
        HarborAgentContract::installed(vec![AgentCapability::ReadOnlyBase]),
        None,
    ));
    let external = execute(request(
        &fixture("local-external"),
        "external-attempt",
        HarborAgentContract::External {
            required: vec![AgentCapability::ReadOnlyBase],
        },
        Some(vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "printf external-patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\"".to_owned(),
        ]),
    ));
    let paired_candidate = execute(request(
        &fixture("local-installed"),
        "paired-attempt",
        HarborAgentContract::External {
            required: vec![AgentCapability::ReadOnlyBase],
        },
        Some(vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "printf installed-patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\"".to_owned(),
        ]),
    ));

    assert_eq!(installed.imported.task.id.as_str(), "harbor-p0-installed");
    assert_eq!(installed.trial.task, installed.imported.task);
    assert_eq!(
        installed.imported.package.declared_artifacts(),
        ["/results/patch.diff"]
    );
    assert_eq!(installed.verifier_result.evidence.len(), 1);
    assert_eq!(
        installed.verifier_result.evidence[0],
        installed.initial_score.evidence[0]
    );
    assert_eq!(
        installed.verifier_result.evidence[0],
        ArtifactDigest::from_bytes(b"installed-patch")
    );
    assert_eq!(installed.initial_score.value, 1.0);
    assert_eq!(installed.regraded_score.value, 0.75);
    assert_eq!(
        installed.regraded_score.predecessor,
        Some(installed.initial_score.identity_digest())
    );
    assert_eq!(
        installed
            .evidence
            .iter()
            .map(|event| (event.sequence, event.kind.as_str(), event.attempt.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (0, "sandbox", "installed-attempt"),
            (1, "agent", "installed-attempt"),
            (2, "artifact", "installed-attempt"),
            (3, "evaluator", "installed-attempt"),
        ]
    );
    assert_eq!(external.imported.task.id.as_str(), "harbor-p0-external");
    assert_eq!(external.initial_score.value, 1.0);

    let spec = PairedComparisonSpec::new(
        installed.trial.task.digest.as_str(),
        "native:local",
        7,
        installed.trial.policy.digest().as_str(),
        installed.trial.environment.as_str(),
        60,
    )
    .unwrap();
    let report = HarborEvaluationCoordinator::compare_completed(
        (
            &installed,
            PairedMeasurements::new(0.75, 1.0, 1.0, 1.0, 1, 0).unwrap(),
        ),
        (
            &paired_candidate,
            PairedMeasurements::new(0.75, 0.5, 0.5, 0.5, 1, 0).unwrap(),
        ),
        &spec,
    )
    .unwrap();
    assert_eq!(report.quality_delta(), 0.0);
    assert_eq!(report.cost_delta(), -0.5);
}
