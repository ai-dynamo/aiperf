// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic NativeGraph branch-workspace lease contracts.

use std::rc::Rc;

use aiperf_runtime::eval::ArtifactDigest;
use aiperf_runtime::graph::agent::{
    AgentInvocationEnvironment, AgentInvocationIdentity, AgentInvocationLeaseFactoryFactory,
    AgentInvocationRequest, AgentInvocationWorkspace, AgentInvocationWorkspaceCandidate,
    InMemoryAgentInvocationLeaseFactoryFactory,
};
use aiperf_runtime::graph::tools::{InMemoryToolDispatcher, ToolDispatcher};

/// A selected branch receives an isolated child lease, while a cancelled loser
/// cannot publish a candidate for the parent merge. Candidate content identity
/// is minted only by the completed child lease, never supplied by the caller.
#[tokio::test(flavor = "current_thread")]
async fn cancelled_branch_workspace_cannot_publish_a_candidate_before_parent_merge() {
    let root_dispatcher: Rc<dyn ToolDispatcher> = Rc::new(InMemoryToolDispatcher::default());
    let factory = InMemoryAgentInvocationLeaseFactoryFactory
        .create("trace", root_dispatcher)
        .expect("trace creates one lifecycle owner");
    let root_request = AgentInvocationRequest {
        identity: AgentInvocationIdentity {
            run_id: "run".into(),
            trajectory_id: "trajectory".into(),
            invocation_id: "trace::root".into(),
            parent_invocation_id: None,
        },
        environment: AgentInvocationEnvironment::Isolated,
        workspace: AgentInvocationWorkspace::Root,
    };
    let mut root_opening = factory
        .begin_open(&root_request, None)
        .expect("root opening begins");
    let mut root = root_opening.open().await.expect("root lease opens");

    let selected_request = branch_request("choose-a");
    let cancelled_request = branch_request("choose-b");
    let mut selected_opening = factory
        .begin_open(&selected_request, Some(root.as_ref()))
        .expect("selected branch opening begins");
    let mut cancelled_opening = factory
        .begin_open(&cancelled_request, Some(root.as_ref()))
        .expect("losing branch opening begins");
    let mut selected = selected_opening
        .open()
        .await
        .expect("selected branch opens");
    let mut cancelled = cancelled_opening.open().await.expect("losing branch opens");

    cancelled
        .close()
        .await
        .expect("losing branch closes before merge");
    assert!(
        cancelled
            .complete_workspace()
            .await
            .expect_err("cancelled branch cannot publish a merge candidate")
            .to_string()
            .contains("closed")
    );

    let candidate = selected
        .complete_workspace()
        .await
        .expect("selected child completion succeeds")
        .expect("isolated branch produces one candidate");
    assert_eq!(candidate.id(), "choose-a");
    assert!(candidate.digest().as_str().starts_with("blake3:"));
    selected
        .close()
        .await
        .expect("selected child closes before merge");
    root.close().await.expect("parent closes after children");
}

/// A child lease may publish only a validated artifact identity. Paths and
/// secret-shaped values are rejected before the candidate can reach a merge
/// receipt, while a factory-minted digest remains opaque.
#[test]
fn workspace_candidate_rejects_untrusted_content_before_receipt_publication() {
    for invalid in ["../../agent-workspace", "api_key=top-secret"] {
        assert!(
            AgentInvocationWorkspaceCandidate::parse("choose-a".into(), invalid).is_err(),
            "candidate parser must reject untrusted content {invalid:?}"
        );
    }

    let candidate = AgentInvocationWorkspaceCandidate::new(
        "choose-a".into(),
        ArtifactDigest::from_bytes(b"frozen-branch-workspace"),
    );
    assert_eq!(candidate.id(), "choose-a");
    assert!(candidate.digest().as_str().starts_with("blake3:"));
}

fn branch_request(candidate_id: &str) -> AgentInvocationRequest {
    AgentInvocationRequest {
        identity: AgentInvocationIdentity {
            run_id: "run".into(),
            trajectory_id: "trajectory".into(),
            invocation_id: format!("trace::route::{candidate_id}"),
            parent_invocation_id: Some("trace::root".into()),
        },
        environment: AgentInvocationEnvironment::Isolated,
        workspace: AgentInvocationWorkspace::IsolatedBranch {
            branch_id: "route".into(),
            candidate_id: candidate_id.into(),
            parent_invocation_id: "trace::root".into(),
            parent_snapshot_digest: "blake3:parent-snapshot".into(),
        },
    }
}
