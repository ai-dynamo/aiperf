// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral coverage for the worker-local recorded-agent loop seams.

use bytes::Bytes;
use std::rc::Rc;

use aiperf_runtime::graph::agent::{
    AgentInvocationEnvironment, AgentInvocationIdentity, AgentInvocationLeaseFactory,
    AgentInvocationRequest, AgentResponseSource, AgentResponseStore, AgentTurn,
    AgentTurnCoordinator, DelegatedInvocationTerminal, InMemoryAgentInvocationLeaseFactory,
    InMemoryAgentResponseStore, InMemoryAgentTrajectorySink, InMemoryInvocationLeaseFactory,
    ResponseSelection, StaticAgentTurnCoordinator, deterministic_delegated_join_order,
};
use aiperf_runtime::graph::driver::{
    AgentContinuationSpec, RecordedReplayTraceProgramDriverFactory, TraceDriverSpec,
    TraceProgramDriverFactory,
};
use aiperf_runtime::graph::tools::{
    AgentToolCall, InMemoryAgentObservationFormatter, InMemoryAgentToolCallDecoder,
    InMemoryToolDispatcher, ToolDispatchResult,
};

#[tokio::test(flavor = "current_thread")]
async fn fake_live_loop_reuses_original_wire_and_correlates_tool_results() {
    // This catches a dispatcher or coordinator that drops the original response
    // bytes, loses copied-context attribution, or attaches a tool result to the
    // wrong provider call.
    let mut response_store = InMemoryAgentResponseStore::default();
    let mut trajectory = InMemoryAgentTrajectorySink::default();
    let leases = InMemoryInvocationLeaseFactory::default();
    let dispatcher = InMemoryToolDispatcher::from_results([ToolDispatchResult::completed(
        "call-1",
        0,
        Bytes::from_static(b"tool output"),
    )]);
    let original = response_store
        .intern(
            AgentResponseSource::Recorded,
            Bytes::from_static(br#"{"tool_calls":[{"id":"call-1"}]}"#),
        )
        .expect("fake store interns original response");
    let decoder = InMemoryAgentToolCallDecoder::from_call_batches([
        vec![AgentToolCall {
            call_id: "call-1".into(),
            command: "echo tool".into(),
        }],
        vec![],
    ]);
    let formatter = InMemoryAgentObservationFormatter;
    let mut coordinator = StaticAgentTurnCoordinator::new([
        AgentTurn::new(
            ResponseSelection::Inline {
                source: AgentResponseSource::Recorded,
                wire: Bytes::from_static(br#"{"tool_calls":[{"id":"call-1"}]}"#),
            },
            false,
        ),
        AgentTurn::new(ResponseSelection::Reuse(original), false),
    ]);

    let trace = coordinator
        .run(
            &mut response_store,
            &mut trajectory,
            &leases,
            &dispatcher,
            &decoder,
            &formatter,
        )
        .await
        .expect("fake recorded loop completes");

    assert_eq!(
        trace.dispatched_response_wires[0].as_ref(),
        br#"{"tool_calls":[{"id":"call-1"}]}"#
    );
    assert_eq!(
        trace.dispatched_response_wires[1].as_ref(),
        br#"{"tool_calls":[{"id":"call-1"}]}"#
    );
    assert_eq!(trace.copied_context_turns, vec![1]);
    assert_eq!(trace.tool_results.len(), 1);
    assert_eq!(trace.tool_results[0].call_id, "call-1");
    assert_eq!(trace.tool_results[0].output.as_ref(), b"tool output");
    assert_eq!(
        trace.observations,
        [Bytes::from_static(b"call-1:tool output")]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn delegated_leases_share_only_the_parent_dispatcher_and_join_in_authored_order() {
    let dispatcher = Rc::new(InMemoryToolDispatcher::default());
    let factory = InMemoryAgentInvocationLeaseFactory::new(dispatcher);
    let root_identity = AgentInvocationIdentity {
        run_id: "run-1".into(),
        trajectory_id: "trajectory-root".into(),
        invocation_id: "root".into(),
        parent_invocation_id: None,
    };
    let mut root = factory
        .open(
            &AgentInvocationRequest {
                identity: root_identity.clone(),
                environment: AgentInvocationEnvironment::Isolated,
            },
            None,
        )
        .await
        .expect("root invocation receives a lease");
    let child_identity = AgentInvocationIdentity {
        run_id: "run-1".into(),
        trajectory_id: "trajectory-child".into(),
        invocation_id: "child".into(),
        parent_invocation_id: Some("root".into()),
    };
    let mut child = factory
        .open(
            &AgentInvocationRequest {
                identity: child_identity.clone(),
                environment: AgentInvocationEnvironment::Shared,
            },
            Some(root.as_ref()),
        )
        .await
        .expect("shared child borrows its parent lease");
    assert!(Rc::ptr_eq(&root.dispatcher(), &child.dispatcher()));

    let ordered = deterministic_delegated_join_order(vec![
        DelegatedInvocationTerminal {
            identity: child_identity,
            join_ordinal: 1,
        },
        DelegatedInvocationTerminal {
            identity: root_identity,
            join_ordinal: 0,
        },
    ])
    .expect("authored ordinals produce deterministic joins");
    assert_eq!(ordered[0].identity.invocation_id, "root");
    child.close().await.expect("child closes before parent");
    root.close().await.expect("parent closes after child");
}

#[test]
fn recorded_driver_refuses_load_resume_and_delegation_before_provisioning() {
    for spec in [
        TraceDriverSpec::recorded_replay().with_continuation(AgentContinuationSpec::Load {
            trajectory: "prior-trajectory".into(),
        }),
        TraceDriverSpec::recorded_replay().with_continuation(AgentContinuationSpec::Resume {
            checkpoint: "prior-checkpoint".into(),
        }),
        TraceDriverSpec::recorded_replay().with_delegation(),
    ] {
        let error = RecordedReplayTraceProgramDriverFactory
            .capabilities(&spec)
            .expect_err("recorded replay supports fresh, non-delegated invocations only");
        assert!(error.to_string().contains("recorded_replay"));
    }
}
