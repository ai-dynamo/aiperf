// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::path::Path;

use aiperf_runtime::eval::{
    AgentCapability, AgentVariantRef, ArtifactDigest, AttemptId, EvalExecutionError,
    EvalSandboxFactory, HarborAgentContract, HarborEvaluationCoordinator,
    HarborLocalEvaluationRequest, HarborSandboxRecipe, HarborSource, LocalProcessSandbox,
    ModelIdentity, NativeSourceAcquirer, PolicyIdentity, RuntimeIdentity, TrialBudget,
    VerifierExecutionError, VerifierMode, VerifierSandboxFactory,
};

struct LocalFactory {
    opened: Cell<bool>,
}

impl EvalSandboxFactory for LocalFactory {
    fn capabilities(&self) -> &[AgentCapability] {
        const CAPABILITIES: [AgentCapability; 1] = [AgentCapability::ReadOnlyBase];
        &CAPABILITIES
    }

    fn open(&self, _: &HarborSandboxRecipe) -> Result<(), EvalExecutionError> {
        self.opened.set(true);
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

#[test]
fn local_process_provider_rejects_separate_verifier_isolation_before_opening() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/harbor_p0/isolation");
    let factory = LocalFactory {
        opened: Cell::new(false),
    };
    let verifier = UnusedVerifier;
    let coordinator = HarborEvaluationCoordinator::new(&NativeSourceAcquirer, &factory, &verifier);
    let result = coordinator.execute_local(
        &LocalProcessSandbox::new(),
        HarborLocalEvaluationRequest {
            source: HarborSource::local(fixture.to_string_lossy()).unwrap(),
            recipe: HarborSandboxRecipe::new(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "/work",
            )
            .unwrap(),
            contract: HarborAgentContract::installed(vec![AgentCapability::ReadOnlyBase]),
            agent_variant: AgentVariantRef::new("isolation-agent").unwrap(),
            model: ModelIdentity::new("native", "local").unwrap(),
            seed: 1,
            policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"isolation-policy")),
            runtime: RuntimeIdentity::new("native-local").unwrap(),
            budget: TrialBudget::new(10.0, 10.0).unwrap(),
            attempt: AttemptId::new("isolation-attempt").unwrap(),
            verifier_mode: VerifierMode::Separate,
            agent_command: None,
            score_metric: "reward".to_owned(),
            initial_rationale: ArtifactDigest::from_bytes(b"initial"),
            regrade_metric: "reward".to_owned(),
            regrade_rationale: ArtifactDigest::from_bytes(b"regrade"),
        },
    );
    let error = result.expect_err("local process execution cannot isolate a separate verifier");

    assert_eq!(
        error.to_string(),
        "local process execution cannot provide separate verifier isolation"
    );
    assert!(!factory.opened.get());
}
