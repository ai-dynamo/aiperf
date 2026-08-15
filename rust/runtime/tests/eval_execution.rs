// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};

use aiperf_runtime::eval::{
    AgentCapability, ArtifactDigest, DeclaredArtifactTransfer, EvalExecutionError,
    EvalSandboxFactory, HarborAgentContract, HarborEvaluationCoordinator, HarborImportError,
    HarborSandboxRecipe, HarborSource, SourceAcquirer, VerifierExecutionError, VerifierMode,
    VerifierSandboxFactory, WorkspaceOverlay,
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
        bytes: br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#.to_vec(),
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
