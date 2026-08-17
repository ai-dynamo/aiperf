// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial contract coverage for NativeGraph adapter protocol seams.

use std::collections::BTreeSet;

use aiperf_runtime::eval::{
    AdapterEnvelope, AdapterMessage, AdapterProtocol, AdapterProtocolConfig,
    AdapterProtocolFactory, AdapterRole, ArtifactDigest, ArtifactQuota, EpisodeArtifactStore,
    HostEnvelope, HostMessage, ModelBindingId, ProtocolCapability, ProtocolError, ProtocolLimits,
    ProtocolOperationState, ProtocolSessionState, StrictAdapterProtocolFactory,
};
use serde_json::json;

fn model_binding() -> ModelBindingId {
    serde_json::from_str("\"model-main\"").expect("model binding fixture is valid")
}

fn capabilities(role: AdapterRole) -> BTreeSet<ProtocolCapability> {
    [
        ProtocolCapability::Artifacts,
        ProtocolCapability::Checkpoint,
        match role {
            AdapterRole::Tool => ProtocolCapability::Tool,
            AdapterRole::Policy => ProtocolCapability::Policy,
            AdapterRole::Environment => ProtocolCapability::Environment,
            AdapterRole::Heuristic => ProtocolCapability::Heuristic,
            AdapterRole::Driver => ProtocolCapability::Driver,
        },
    ]
    .into_iter()
    .collect()
}

fn config(role: AdapterRole) -> AdapterProtocolConfig {
    config_with_limits(role, ProtocolLimits::default())
}

fn config_with_limits(role: AdapterRole, limits: ProtocolLimits) -> AdapterProtocolConfig {
    AdapterProtocolConfig::new(
        role,
        "episode-1",
        capabilities(role),
        [model_binding()].into_iter().collect(),
        limits,
    )
    .expect("fixture protocol configuration is valid")
}

fn new_protocol(role: AdapterRole) -> Box<dyn AdapterProtocol> {
    let strict = StrictAdapterProtocolFactory;
    let factory: &dyn AdapterProtocolFactory = &strict;
    factory
        .create(config(role))
        .expect("strict factory must supply a protocol seam")
}

fn new_protocol_with_limits(role: AdapterRole, limits: ProtocolLimits) -> Box<dyn AdapterProtocol> {
    let strict = StrictAdapterProtocolFactory;
    strict
        .create(config_with_limits(role, limits))
        .expect("strict factory must supply a protocol seam")
}

fn host(sequence: u64, operation: &str, message: HostMessage) -> HostEnvelope {
    HostEnvelope::new("episode-1", "root", sequence, operation, message)
}

fn adapter(sequence: u64, operation: &str, message: AdapterMessage) -> AdapterEnvelope {
    AdapterEnvelope::new("episode-1", "root", sequence, operation, message)
}

fn adapter_on_span(
    span: &str,
    sequence: u64,
    operation: &str,
    message: AdapterMessage,
) -> AdapterEnvelope {
    AdapterEnvelope::new("episode-1", span, sequence, operation, message)
}

fn ready(protocol: &mut dyn AdapterProtocol, role: AdapterRole) {
    protocol
        .accept_host(host(
            0,
            "hello",
            HostMessage::Hello {
                supported_versions: vec![1],
                adapter_role: role,
                capabilities: capabilities(role).into_iter().collect(),
            },
        ))
        .expect("host must negotiate its exact selected role and capabilities");
    protocol
        .accept_adapter(adapter(
            0,
            "hello",
            AdapterMessage::Ready {
                protocol_version: 1,
                capabilities: capabilities(role).into_iter().collect(),
                implementation_digest: ArtifactDigest::from_bytes(b"adapter-v1"),
            },
        ))
        .expect("adapter ready must acknowledge the negotiated capability set");
}

fn artifact_handles() -> (
    tempfile::TempDir,
    aiperf_runtime::eval::ArtifactUploadHandle,
    aiperf_runtime::eval::ArtifactDownloadHandle,
) {
    let directory = tempfile::tempdir().expect("temporary artifact root");
    let mut store = EpisodeArtifactStore::new(
        directory.path(),
        ArtifactQuota {
            max_artifacts: 1,
            max_total_bytes: 3,
            max_artifact_bytes: 3,
            max_download_handles: 1,
        },
    )
    .expect("fixture artifact store is valid");
    let upload = store
        .begin_upload(3)
        .expect("store issues upload capability");
    store
        .write_upload(&upload, &mut std::io::Cursor::new(b"one".to_vec()))
        .expect("fixture bytes fit the upload grant");
    let frozen = store
        .commit_upload(&upload)
        .expect("fixture freezes the upload");
    let download = store
        .issue_download(&frozen)
        .expect("store issues download capability");
    (directory, upload, download)
}

#[test]
fn factory_exposes_only_the_object_safe_protocol_seam() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    let admitted = protocol
        .accept_host(host(
            0,
            "hello",
            HostMessage::Hello {
                supported_versions: vec![1],
                adapter_role: AdapterRole::Tool,
                capabilities: capabilities(AdapterRole::Tool).into_iter().collect(),
            },
        ))
        .expect("host admission produces the only codec input type");
    let frame = protocol
        .encode_host_frame(&admitted)
        .expect("trait-object codec emits one strict JSONL frame");
    assert!(frame.ends_with(b"\n"));
    protocol
        .accept_adapter(adapter(
            0,
            "hello",
            AdapterMessage::Ready {
                protocol_version: 1,
                capabilities: capabilities(AdapterRole::Tool).into_iter().collect(),
                implementation_digest: ArtifactDigest::from_bytes(b"adapter-v1"),
            },
        ))
        .expect("adapter accepts the negotiated host frame");
    assert_eq!(protocol.session_state(), ProtocolSessionState::Ready);
}

#[test]
fn leaf_adapter_cannot_claim_an_episode_terminal() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    let error = protocol
        .accept_adapter(adapter(
            1,
            "terminal",
            AdapterMessage::EpisodeTerminalCandidate { output: json!({}) },
        ))
        .expect_err("only an externally driven adapter can answer a terminal request");
    assert_eq!(
        error,
        ProtocolError::MessageForbiddenForRole(AdapterRole::Tool)
    );
}

#[test]
fn shutdown_closes_the_session_without_reopening_its_operation() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(host(1, "shutdown", HostMessage::Shutdown))
        .expect("an idle session can begin shutdown");
    protocol
        .accept_adapter(adapter(1, "shutdown", AdapterMessage::ShutdownAck))
        .expect("shutdown acknowledgement closes the session");
    assert_eq!(protocol.session_state(), ProtocolSessionState::Closed);
    assert_eq!(
        protocol.operation_state("shutdown"),
        Some(ProtocolOperationState::Closed)
    );
    let error = protocol
        .accept_host(host(2, "shutdown", HostMessage::Shutdown))
        .expect_err("a closed session cannot recycle its terminal operation");
    assert_eq!(
        error,
        ProtocolError::SessionState(ProtocolSessionState::Closed)
    );
}

#[test]
fn terminal_candidate_requires_a_driver_terminal_operation() {
    let mut protocol = new_protocol(AdapterRole::Driver);
    ready(protocol.as_mut(), AdapterRole::Driver);

    let error = protocol
        .accept_adapter(adapter(
            1,
            "terminal",
            AdapterMessage::EpisodeTerminalCandidate {
                output: json!({"answer": "done"}),
            },
        ))
        .expect_err("a driver cannot terminate without a host terminal request");
    assert!(matches!(error, ProtocolError::UnknownOperation(_)));

    let mut protocol = new_protocol(AdapterRole::Driver);
    ready(protocol.as_mut(), AdapterRole::Driver);
    protocol
        .accept_host(host(
            1,
            "terminal",
            HostMessage::RequestEpisodeTerminal {
                input: json!({"reason": "external-driver-complete"}),
            },
        ))
        .expect("only Rust opens the compatibility terminal operation");
    protocol
        .accept_adapter(adapter(
            1,
            "terminal",
            AdapterMessage::EpisodeTerminalCandidate {
                output: json!({"answer": "done"}),
            },
        ))
        .expect("the matching driver response closes the terminal operation");
    assert_eq!(
        protocol.operation_state("terminal"),
        Some(ProtocolOperationState::Closed)
    );
}

#[test]
fn cancellation_closes_its_explicit_target_and_never_reuses_an_operation() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(host(
            1,
            "tool-1",
            HostMessage::InvokeTool {
                input: json!({"query": "status"}),
            },
        ))
        .expect("tool operation opens");
    protocol
        .accept_host(host(
            2,
            "cancel-1",
            HostMessage::Cancel {
                target_operation: "tool-1".to_owned(),
            },
        ))
        .expect("cancel names a live target");
    assert_eq!(
        protocol.operation_state("tool-1"),
        Some(ProtocolOperationState::Cancelling)
    );
    protocol
        .accept_adapter(adapter(
            1,
            "cancel-1",
            AdapterMessage::CancelAck {
                target_operation: "tool-1".to_owned(),
            },
        ))
        .expect("matching cancel acknowledgement closes both operations");
    assert_eq!(
        protocol.operation_state("tool-1"),
        Some(ProtocolOperationState::Closed)
    );
    assert_eq!(
        protocol.operation_state("cancel-1"),
        Some(ProtocolOperationState::Closed)
    );

    let error = protocol
        .accept_host(host(
            3,
            "tool-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect_err("closed operation correlations remain permanently reserved");
    assert_eq!(
        error,
        ProtocolError::OperationAlreadyUsed("tool-1".to_owned())
    );
}

#[test]
fn cancellation_rejects_a_target_that_is_already_cancelling() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(host(
            1,
            "tool-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect("tool operation opens");
    protocol
        .accept_host(host(
            2,
            "cancel-1",
            HostMessage::Cancel {
                target_operation: "tool-1".to_owned(),
            },
        ))
        .expect("first cancel claims its target");
    let error = protocol
        .accept_host(host(
            3,
            "cancel-2",
            HostMessage::Cancel {
                target_operation: "tool-1".to_owned(),
            },
        ))
        .expect_err("a target has only one live cancellation transition");
    assert_eq!(
        error,
        ProtocolError::CancelTargetInvalid("tool-1".to_owned())
    );
}

#[test]
fn cancel_capacity_rejection_does_not_mutate_the_target_operation() {
    let mut limits = ProtocolLimits::default();
    limits.max_operation_ledger_entries = 2;
    let mut protocol = new_protocol_with_limits(AdapterRole::Tool, limits);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(host(
            1,
            "tool-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect("the bounded ledger still admits its second operation");

    let error = protocol
        .accept_host(host(
            2,
            "cancel-1",
            HostMessage::Cancel {
                target_operation: "tool-1".to_owned(),
            },
        ))
        .expect_err("cancel admission must fail before it changes its target");
    assert!(matches!(
        error,
        ProtocolError::OperationLedgerLimit { limit: 2 }
    ));
    assert_eq!(
        protocol.operation_state("tool-1"),
        Some(ProtocolOperationState::Pending),
        "a rejected cancellation leaves the live target unchanged"
    );
}

#[test]
fn response_must_preserve_the_span_of_the_operation_it_satisfies() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(HostEnvelope::new(
            "episode-1",
            "tool-span",
            1,
            "tool-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect("host opens a span-bound tool operation");
    let error = protocol
        .accept_adapter(adapter_on_span(
            "different-span",
            1,
            "tool-1",
            AdapterMessage::ToolResult { output: json!({}) },
        ))
        .expect_err("a response cannot claim another causal span");
    assert!(matches!(error, ProtocolError::SpanMismatch { .. }));
}

#[test]
fn adapter_artifact_request_cannot_invent_a_causal_span() {
    let mut protocol = new_protocol(AdapterRole::Tool);
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(HostEnvelope::new(
            "episode-1",
            "host-tool-span",
            1,
            "tool-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect("Rust first assigns the parent operation span");

    let error = protocol
        .accept_adapter(adapter_on_span(
            "adapter-invented-span",
            1,
            "put-1",
            AdapterMessage::PutArtifactRequest {
                parent_operation: "tool-1".to_owned(),
                declared_bytes: 3,
            },
        ))
        .expect_err("artifact work must bind to a Rust-assigned causal span");
    assert!(matches!(error, ProtocolError::SpanMismatch { .. }));
}

#[test]
fn model_call_and_artifact_grants_bind_the_original_operation_facts() {
    let mut policy = new_protocol(AdapterRole::Policy);
    ready(policy.as_mut(), AdapterRole::Policy);
    policy
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("arbitrary empty JSON values remain valid decision payloads");
    policy
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-1".to_owned(),
                binding: model_binding(),
                input: json!({"messages": []}),
            },
        ))
        .expect("the authorized model intent is retained as an intermediate operation state");
    let error = policy
        .accept_host(host(
            2,
            "decision-1",
            HostMessage::DeliverModelResult {
                model_call: "different-call".to_owned(),
                output: json!(""),
            },
        ))
        .expect_err("the model result cannot satisfy a different model-call correlation");
    assert!(matches!(error, ProtocolError::ModelCallMismatch { .. }));

    let mut tool = new_protocol(AdapterRole::Tool);
    let (_directory, upload, _download) = artifact_handles();
    ready(tool.as_mut(), AdapterRole::Tool);
    tool.accept_host(host(
        1,
        "artifact-parent-1",
        HostMessage::InvokeTool { input: json!({}) },
    ))
    .expect("Rust opens the artifact request parent operation");
    tool.accept_adapter(adapter(
        1,
        "put-1",
        AdapterMessage::PutArtifactRequest {
            parent_operation: "artifact-parent-1".to_owned(),
            declared_bytes: 3,
        },
    ))
    .expect("adapter starts a bounded artifact request");
    let error = tool
        .accept_host(host(
            2,
            "put-1",
            HostMessage::PutArtifactHandle {
                upload: upload.clone(),
                declared_bytes: 2,
            },
        ))
        .expect_err("the host grant must retain the adapter's exact declared length");
    assert!(matches!(
        error,
        ProtocolError::ArtifactLengthMismatch { .. }
    ));

    let mut tool = new_protocol(AdapterRole::Tool);
    let (_directory, upload, download) = artifact_handles();
    ready(tool.as_mut(), AdapterRole::Tool);
    tool.accept_host(host(
        1,
        "artifact-parent-1",
        HostMessage::InvokeTool { input: json!({}) },
    ))
    .expect("Rust opens the artifact request parent operation");
    tool.accept_adapter(adapter(
        1,
        "put-1",
        AdapterMessage::PutArtifactRequest {
            parent_operation: "artifact-parent-1".to_owned(),
            declared_bytes: 3,
        },
    ))
    .expect("adapter starts a bounded artifact request");
    tool.accept_host(host(
        2,
        "put-1",
        HostMessage::PutArtifactHandle {
            upload: upload.clone(),
            declared_bytes: 3,
        },
    ))
    .expect("exact upload grant is accepted");
    let error = tool
        .accept_host(host(
            3,
            "put-1",
            HostMessage::ArtifactCommitted {
                upload: artifact_handles().1,
                download,
                length: 3,
            },
        ))
        .expect_err("commit cannot switch the previously granted capability");
    assert!(matches!(
        error,
        ProtocolError::ArtifactUploadMismatch { .. }
    ));
}

#[test]
fn model_call_lineage_cap_exhausts_one_decision_and_recovers_after_terminal_cleanup() {
    let mut limits = ProtocolLimits::default();
    limits.max_model_call_lineage_entries = 1;

    let mut exhausted = new_protocol_with_limits(AdapterRole::Policy, limits.clone());
    ready(exhausted.as_mut(), AdapterRole::Policy);
    exhausted
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("first decision opens");
    exhausted
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-1".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("the first model call enters the pending decision lineage");
    exhausted
        .accept_host(host(
            2,
            "decision-1",
            HostMessage::DeliverModelResult {
                model_call: "model-call-1".to_owned(),
                output: json!({}),
            },
        ))
        .expect("exact delivery retains the completed model-call correlation");
    let error = exhausted
        .accept_adapter(adapter(
            2,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-2".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect_err("one pending decision cannot retain an unbounded completed model-call lineage");
    assert!(matches!(error, ProtocolError::ModelCallLimit { limit: 1 }));

    let mut recovered = new_protocol_with_limits(AdapterRole::Policy, limits);
    ready(recovered.as_mut(), AdapterRole::Policy);
    recovered
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("decision opens");
    recovered
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-1".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("first model call enters the pending decision lineage");
    recovered
        .accept_host(host(
            2,
            "decision-1",
            HostMessage::DeliverModelResult {
                model_call: "model-call-1".to_owned(),
                output: json!({}),
            },
        ))
        .expect("exact host delivery records the completed model-call correlation");
    recovered
        .accept_adapter(adapter(
            2,
            "decision-1",
            AdapterMessage::Decision { output: json!({}) },
        ))
        .expect("terminal decision cleanup releases the completed lineage");
    recovered
        .accept_host(host(
            3,
            "decision-2",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("a new decision begins with a fresh bounded lineage");
    recovered
        .accept_adapter(adapter(
            3,
            "decision-2",
            AdapterMessage::ModelIntent {
                model_call: "model-call-2".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("terminal cleanup recovers capacity for a new decision lineage");
}

#[test]
fn session_model_call_lineage_entry_cap_rejects_another_pending_decision_before_mutation() {
    let mut limits = ProtocolLimits::default();
    limits.max_model_call_lineage_entries = 1;
    limits.max_session_model_call_lineage_entries = 1;
    let mut exhausted = new_protocol_with_limits(AdapterRole::Policy, limits.clone());
    ready(exhausted.as_mut(), AdapterRole::Policy);
    exhausted
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("first pending decision opens");
    exhausted
        .accept_host(host(
            2,
            "decision-2",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("second pending decision opens before any model correlation exists");
    exhausted
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-1".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("first decision consumes the sole session model-call slot");

    let error = exhausted
        .accept_adapter(adapter(
            2,
            "decision-2",
            AdapterMessage::ModelIntent {
                model_call: "model-call-2".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect_err("another live decision cannot exceed the session-wide lineage entry cap");
    assert!(matches!(
        error,
        ProtocolError::ModelCallSessionEntryLimit { limit: 1 }
    ));
    assert_eq!(
        exhausted.operation_state("decision-2"),
        Some(ProtocolOperationState::Pending),
        "rejected model-call admission leaves the other decision ready for a terminal response"
    );
    assert_eq!(
        exhausted.session_state(),
        ProtocolSessionState::Failed,
        "a rejected adapter admission fails the whole strict protocol session closed"
    );

    let mut recovered = new_protocol_with_limits(AdapterRole::Policy, limits);
    ready(recovered.as_mut(), AdapterRole::Policy);
    recovered
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("first decision opens in a clean session");
    recovered
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: "model-call-1".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("first model call consumes the session-wide slot");
    recovered
        .accept_host(host(
            2,
            "decision-1",
            HostMessage::DeliverModelResult {
                model_call: "model-call-1".to_owned(),
                output: json!({}),
            },
        ))
        .expect("exact result delivery makes the first decision terminal-ready");
    recovered
        .accept_adapter(adapter(
            2,
            "decision-1",
            AdapterMessage::Decision { output: json!({}) },
        ))
        .expect("terminal decision releases its complete session lineage");
    recovered
        .accept_host(host(
            3,
            "decision-2",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("a new decision opens after terminal cleanup");
    recovered
        .accept_adapter(adapter(
            3,
            "decision-2",
            AdapterMessage::ModelIntent {
                model_call: "model-call-2".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("terminal cleanup recovers the session-wide model-call slot");
}

#[test]
fn cancel_and_failure_release_session_model_call_lineage_capacity() {
    let mut limits = ProtocolLimits::default();
    limits.max_model_call_lineage_entries = 1;
    limits.max_session_model_call_lineage_entries = 2;
    let mut protocol = new_protocol_with_limits(AdapterRole::Policy, limits);
    ready(protocol.as_mut(), AdapterRole::Policy);
    for (sequence, operation) in [(1, "decision-1"), (2, "decision-2")] {
        protocol
            .accept_host(host(
                sequence,
                operation,
                HostMessage::Decide { input: json!({}) },
            ))
            .expect("two pending decisions open within the aggregate cap");
    }
    for (sequence, operation, model_call) in [
        (1, "decision-1", "model-call-1"),
        (2, "decision-2", "model-call-2"),
    ] {
        protocol
            .accept_adapter(adapter(
                sequence,
                operation,
                AdapterMessage::ModelIntent {
                    model_call: model_call.to_owned(),
                    binding: model_binding(),
                    input: json!({}),
                },
            ))
            .expect("two pending decisions consume the aggregate model-call cap");
    }

    protocol
        .accept_host(host(
            3,
            "cancel-1",
            HostMessage::Cancel {
                target_operation: "decision-1".to_owned(),
            },
        ))
        .expect("cancelling the first decision releases its lineage before an acknowledgement");
    protocol
        .accept_host(host(
            4,
            "decision-3",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("a third decision opens after cancellation");
    protocol
        .accept_adapter(adapter(
            3,
            "decision-3",
            AdapterMessage::ModelIntent {
                model_call: "model-call-3".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("cancellation released one aggregate model-call slot");

    protocol
        .accept_adapter(adapter(
            4,
            "decision-2",
            AdapterMessage::OperationFailed {
                code: "model-failed".to_owned(),
                details: json!({}),
            },
        ))
        .expect("failing a pending decision releases its lineage");
    protocol
        .accept_host(host(
            5,
            "decision-4",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("a fourth decision opens after the failure");
    protocol
        .accept_adapter(adapter(
            5,
            "decision-4",
            AdapterMessage::ModelIntent {
                model_call: "model-call-4".to_owned(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("failure released the remaining aggregate model-call slot");
}

#[test]
fn session_model_call_lineage_byte_cap_counts_maximum_length_identifiers() {
    let mut limits = ProtocolLimits::default();
    limits.max_identifier_bytes = 16;
    limits.max_model_call_lineage_entries = 1;
    limits.max_session_model_call_lineage_entries = 2;
    limits.max_session_model_call_lineage_bytes = 16;
    let mut exhausted = new_protocol_with_limits(AdapterRole::Policy, limits.clone());
    ready(exhausted.as_mut(), AdapterRole::Policy);
    exhausted
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("first pending decision opens");
    exhausted
        .accept_host(host(
            2,
            "decision-2",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("second pending decision opens");
    let first_model_call = "a".repeat(16);
    let second_model_call = "b".repeat(16);
    exhausted
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: first_model_call.clone(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("a model-call identifier exactly at the typed bound is valid");

    let error = exhausted
        .accept_adapter(adapter(
            2,
            "decision-2",
            AdapterMessage::ModelIntent {
                model_call: second_model_call.clone(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect_err("a second valid maximum-length identifier exceeds the aggregate byte budget");
    assert!(matches!(
        error,
        ProtocolError::ModelCallSessionByteLimit { limit: 16 }
    ));
    assert_eq!(
        exhausted.operation_state("decision-2"),
        Some(ProtocolOperationState::Pending),
        "byte-budget rejection leaves the decision state untouched"
    );
    assert_eq!(
        exhausted.session_state(),
        ProtocolSessionState::Failed,
        "a byte-budget rejection fails the strict protocol session closed"
    );

    let mut recovered = new_protocol_with_limits(AdapterRole::Policy, limits);
    ready(recovered.as_mut(), AdapterRole::Policy);
    recovered
        .accept_host(host(
            1,
            "decision-1",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("first decision opens in a clean session");
    recovered
        .accept_adapter(adapter(
            1,
            "decision-1",
            AdapterMessage::ModelIntent {
                model_call: first_model_call.clone(),
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("the first valid maximum-length identifier consumes the byte budget");
    recovered
        .accept_host(host(
            2,
            "decision-1",
            HostMessage::DeliverModelResult {
                model_call: first_model_call,
                output: json!({}),
            },
        ))
        .expect("exact result delivery retains but does not duplicate the identifier");
    recovered
        .accept_adapter(adapter(
            2,
            "decision-1",
            AdapterMessage::Decision { output: json!({}) },
        ))
        .expect("terminal decision releases the exact identifier byte accounting");
    recovered
        .accept_host(host(
            3,
            "decision-2",
            HostMessage::Decide { input: json!({}) },
        ))
        .expect("a new decision opens after terminal cleanup");
    recovered
        .accept_adapter(adapter(
            3,
            "decision-2",
            AdapterMessage::ModelIntent {
                model_call: second_model_call,
                binding: model_binding(),
                input: json!({}),
            },
        ))
        .expect("released byte capacity admits another valid maximum-length identifier");
}

#[test]
fn consumed_download_handles_are_released_from_the_protocol_cap() {
    let mut limits = ProtocolLimits::default();
    limits.max_artifact_handles = 1;
    let mut protocol = new_protocol_with_limits(AdapterRole::Tool, limits);
    let (_first_directory, upload, first_download) = artifact_handles();
    let (_second_directory, _second_upload, second_download) = artifact_handles();
    ready(protocol.as_mut(), AdapterRole::Tool);
    protocol
        .accept_host(host(
            1,
            "artifact-parent-1",
            HostMessage::InvokeTool { input: json!({}) },
        ))
        .expect("Rust opens the bounded artifact parent operation");
    protocol
        .accept_adapter(adapter(
            1,
            "put-1",
            AdapterMessage::PutArtifactRequest {
                parent_operation: "artifact-parent-1".to_owned(),
                declared_bytes: 3,
            },
        ))
        .expect("adapter starts one parent-bound upload request");
    protocol
        .accept_host(host(
            2,
            "put-1",
            HostMessage::PutArtifactHandle {
                upload: upload.clone(),
                declared_bytes: 3,
            },
        ))
        .expect("the sole active artifact lease is the upload capability");
    protocol
        .accept_host(host(
            3,
            "put-1",
            HostMessage::ArtifactCommitted {
                upload,
                download: first_download.clone(),
                length: 3,
            },
        ))
        .expect("commit exchanges upload for the one download lease");
    protocol
        .release_download_handle(&first_download)
        .expect("store consumption revokes the protocol download lease");

    protocol
        .accept_adapter(adapter(
            2,
            "get-1",
            AdapterMessage::GetArtifactRequest {
                parent_operation: "artifact-parent-1".to_owned(),
                request: json!({}),
            },
        ))
        .expect("a released download cannot permanently exhaust the session cap");
    protocol
        .accept_host(host(
            4,
            "get-1",
            HostMessage::GetArtifactHandle {
                download: second_download,
                length: 3,
            },
        ))
        .expect("the released cap admits a replacement download capability");
}

#[test]
fn bounded_jsonl_rejects_the_frame_before_deserializing_it() {
    let strict = StrictAdapterProtocolFactory;
    let mut limits = ProtocolLimits::default();
    limits.max_frame_bytes = 8;
    let mut protocol = strict
        .create(
            AdapterProtocolConfig::new(
                AdapterRole::Tool,
                "episode-1",
                capabilities(AdapterRole::Tool),
                BTreeSet::new(),
                limits,
            )
            .expect("tiny frame test configuration is valid"),
        )
        .expect("strict factory creates protocol");
    let error = protocol
        .accept_adapter_frame(b"definitely-not-json\n")
        .expect_err("frame sizing precedes JSON parsing");
    assert!(matches!(error, ProtocolError::FrameTooLarge { .. }));
    assert_eq!(protocol.session_state(), ProtocolSessionState::Failed);
}
