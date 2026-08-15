// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    AgentCapability, EvalExecutionError, EvalSandboxFactory, HarborAgentContract,
    HarborSandboxRecipe, WorkspaceOverlay,
};

struct RecordingFactory {
    opened: std::cell::Cell<bool>,
    capabilities: Vec<AgentCapability>,
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
