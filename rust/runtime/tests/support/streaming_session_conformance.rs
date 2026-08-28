// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable conformance assertions for `StreamingSessionProgramFactory`.
//!
//! One implementation of the session-program contract is exercised end to end:
//! strict configuration validation against the paired format and workload
//! descriptors, fresh participant initialization, fragment incorporation with
//! captured action output, causal-frontier advancement, explicit source seal,
//! and idempotent post-commit notification.
//!
//! The caller constructs the reliability reporter and moves it in. No adapter
//! owns it. Every borrow of the reporter is released before the next
//! coordinator or checkpoint `await`.
//!
//! Loaded by an integration test with
//! `#[path = "support/streaming_session_conformance.rs"] mod …;`.
#![allow(dead_code)]

use std::rc::Rc;

use aiperf_runtime::engine::registry::WorkloadDescriptor;
use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
        CheckpointParticipantId, CheckpointParticipantPlan, CommittedParticipantState,
        DecodeHorizon, DiscoveryHorizon, EventTimeWatermark, OrderedActionHorizon,
        ParticipantStateDescriptor, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant, TerminalActionHorizon,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, LeasedCheckpointGenerationView,
        StreamingGenerationTransaction,
    },
    checkpoints::memory::{MemoryCheckpointBackend, MemoryCheckpointLimits},
    failure::{SessionCoordinatorError, StreamingIssueReporter},
    format::StreamingFormatDescriptor,
    identity::{
        ContentDigest, GlobalSequence, SessionCausalFrontier, StableActionId, StableSessionKey,
    },
    reliability::HandledIssueCut,
    session::{
        DatasetActionSink, StreamingSessionPrepareContext, StreamingSessionProgramFactory,
        conversation::TranscriptEntry,
    },
    source::SourceSeal,
    unit::{DatasetActionV1, EventTimeUtc, ExecutableDatasetAction, SourcePosition},
};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::value::RawValue;

/// Coordinator-scoped hook that releases exactly one parked coordination step.
///
/// Coordinators that never park supply a no-op.
pub type SessionAdvance = Rc<dyn Fn()>;

/// Everything one session program contributes to the shared harness.
pub struct SessionConformanceCases {
    /// Strictly authored configuration the factory must accept.
    pub authored: Box<RawValue>,
    /// Authored configuration the factory must refuse before any effect.
    pub rejected_authored: Box<RawValue>,
    /// Format descriptor the program is validated against.
    pub format_descriptor: &'static StreamingFormatDescriptor,
    /// Workload descriptor the program is validated against.
    pub workload_descriptor: &'static WorkloadDescriptor,
    /// Logical run bound into the prepare context.
    pub run: StreamRunIdentity,
    /// Stable checkpoint-participant identity bound into the prepare context.
    pub participant_id: CheckpointParticipantId,
    /// Semantic digest of the validated program.
    pub program_semantic_digest: ContentDigest,
    /// Semantic namespace of the selected stream.
    pub stream_semantic_digest: ContentDigest,
    /// Budget charged for live session state.
    pub session_state_budget: StreamingResourceBudget,
    /// Budget charged for prepared checkpoint payloads.
    pub checkpoint_budget: StreamingResourceBudget,
    /// Fragments driven into the coordinator, in decode order.
    pub fragments: Vec<aiperf_runtime::streaming::unit::StreamingSessionFragment>,
    /// Seal accepted by the coordinator.
    pub seal: SourceSeal,
    /// Actions the fragments produce.
    pub expected_action_count: usize,
    /// Ordinary issues the script reports through the injected handle.
    pub expected_issue_count: u64,
    /// Hook releasing one parked coordination step.
    pub advance: SessionAdvance,
}

/// One action captured from the coordinator under test.
///
/// The action's retained content leases are released as it is captured, so a
/// long capture does not itself exhaust the session state budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedAction {
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Stable session owning the action.
    pub session_key: StableSessionKey,
    /// Declared predecessor action identities.
    pub predecessors: Vec<StableActionId>,
    /// Decoded action payload.
    pub payload: CapturedActionPayload,
}

/// Decoded closed action payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturedActionPayload {
    /// One conversation-pair request over the accumulated transcript.
    Request(Vec<TranscriptEntry>),
    /// One host-owned graph-node action.
    GraphNode(String),
    /// One terminal session action.
    Terminal(String),
}

impl CapturedAction {
    /// Return the transcript contents of a request action as UTF-8 text.
    ///
    /// # Panics
    ///
    /// Panics when the captured action is not a request or its content is not
    /// valid UTF-8.
    #[must_use]
    pub fn messages(&self) -> Vec<String> {
        match &self.payload {
            CapturedActionPayload::Request(entries) => entries
                .iter()
                .map(|entry| {
                    String::from_utf8(entry.content.clone()).expect("captured content is text")
                })
                .collect(),
            other => panic!("expected a request action, observed {other:?}"),
        }
    }
}

/// Captured causally ready output of a session coordinator.
#[derive(Default)]
pub struct CollectingActionSink {
    /// Actions in emission order.
    pub actions: Vec<CapturedAction>,
    /// Causal frontiers in emission order.
    pub frontiers: Vec<SessionCausalFrontier>,
}

impl CollectingActionSink {
    /// Return the actions emitted since the sink was last cleared.
    #[must_use]
    pub fn request_actions(&self) -> Vec<&CapturedAction> {
        self.actions
            .iter()
            .filter(|action| matches!(action.payload, CapturedActionPayload::Request(_)))
            .collect()
    }

    /// Return the terminal actions emitted since the sink was last cleared.
    #[must_use]
    pub fn terminal_actions(&self) -> Vec<&CapturedAction> {
        self.actions
            .iter()
            .filter(|action| matches!(action.payload, CapturedActionPayload::Terminal(_)))
            .collect()
    }

    /// Drop every captured action and frontier.
    pub fn clear(&mut self) {
        self.actions.clear();
        self.frontiers.clear();
    }
}

#[async_trait(?Send)]
impl DatasetActionSink for CollectingActionSink {
    async fn send_action(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<(), SessionCoordinatorError> {
        let payload = match action.payload() {
            DatasetActionV1::Request(request) => CapturedActionPayload::Request(
                rmp_serde::from_slice(&request.request).expect("captured transcript decodes"),
            ),
            DatasetActionV1::GraphNode(node) => {
                CapturedActionPayload::GraphNode(node.node_key.clone())
            }
            DatasetActionV1::SessionTerminal(terminal) => {
                CapturedActionPayload::Terminal(terminal.reason.clone())
            }
        };
        self.actions.push(CapturedAction {
            action_id: action.action_id(),
            session_key: action.session_key(),
            predecessors: action.predecessors().to_vec(),
            payload,
        });
        Ok(())
    }

    async fn advance_causal_frontier(
        &mut self,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError> {
        self.frontiers.push(frontier);
        Ok(())
    }
}

/// Build a complete checkpoint cut whose every horizon names `value`.
#[must_use]
pub fn cut_at(value: u64) -> CheckpointCut {
    let event_time = EventTimeUtc::new(i64::try_from(value).unwrap_or(i64::MAX))
        .expect("non-negative test event time");
    CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
        decoded: DecodeHorizon::new(SourcePosition::new(value)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
        event_watermark: EventTimeWatermark::Hard {
            through: event_time,
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(value),
            event_time: Some(event_time),
            digest: ContentDigest::from_bytes([0x5a; 32]),
        },
        handled_issues: HandledIssueCut::empty(),
    }
}

/// Build a checkpoint barrier at `value` for `run`.
#[must_use]
pub fn barrier_at(run: StreamRunIdentity, value: u64) -> CheckpointBarrier {
    CheckpointBarrier {
        run,
        epoch: CheckpointEpoch::new(value.max(1)),
        cut: cut_at(value),
        plan_digest: ContentDigest::from_bytes([0x55; 32]),
    }
}

/// Round-trip one prepared participant state through a committed generation.
///
/// Restore authority is only reachable from a verified committed generation, so
/// the harness commits the prepared object into an in-memory backend and reads
/// it back rather than forging one.
///
/// # Panics
///
/// Panics on any backend, staging, or verification failure.
pub async fn commit_and_restore(prepared: PreparedParticipantState) -> CommittedParticipantState {
    let run = *prepared.run();
    let descriptor: ParticipantStateDescriptor = prepared.descriptor().clone();
    let cut = descriptor.represented_cut.clone();
    let backend = MemoryCheckpointBackend::new(MemoryCheckpointLimits {
        transactions: BudgetLimits {
            max_items: 64,
            max_bytes: 4_194_304,
        },
        prepared_indexes: BudgetLimits {
            max_items: 64,
            max_bytes: 4_194_304,
        },
        storage: BudgetLimits {
            max_items: 64,
            max_bytes: 4_194_304,
        },
        result_summaries: BudgetLimits {
            max_items: 64,
            max_bytes: 4_194_304,
        },
        reads: BudgetLimits {
            max_items: 64,
            max_bytes: 4_194_304,
        },
    })
    .expect("valid memory backend");
    let expectations = CheckpointGenerationExpectations {
        run,
        participant_plan: CheckpointParticipantPlan::new([descriptor.participant_id.clone()])
            .expect("valid one-participant plan"),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    };
    let mut transaction = backend
        .begin_generation(run, None, expectations.clone())
        .await
        .expect("begin generation");
    transaction
        .stage_participant(prepared)
        .await
        .expect("stage participant");
    transaction
        .stage_results(&mut Vec::new(), &mut None)
        .await
        .expect("stage canonical empty result epoch");
    transaction
        .commit(CheckpointCommitMetadata {
            previous: None,
            epoch: CheckpointEpoch::new(1),
            cut,
            execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
            result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
            is_final: false,
            terminal_reason: None,
        })
        .await
        .expect("commit generation");
    let opened = backend
        .open_latest(&run, &expectations)
        .await
        .expect("open generation")
        .expect("generation head");
    match opened.view() {
        LeasedCheckpointGenerationView::CurrentV4(reader) => reader
            .read_participant(&descriptor)
            .await
            .expect("read verified participant"),
        LeasedCheckpointGenerationView::LegacyV3ReadOnly(_) => {
            panic!("a fresh in-memory generation is current-v4")
        }
    }
}

/// Assert the complete streaming session-program contract for one factory.
///
/// # Panics
///
/// Panics with a described failure on any contract violation.
pub async fn assert_session_conformance(
    factory: &dyn StreamingSessionProgramFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    cases: SessionConformanceCases,
) {
    assert_strict_validation(factory, &cases);

    let handle = reporter.handle();
    // Borrow of the owned reporter ends here, before every await below.
    let context = StreamingSessionPrepareContext {
        program_semantic_digest: cases.program_semantic_digest,
        run: cases.run,
        participant_id: cases.participant_id.clone(),
        stream_semantic_digest: cases.stream_semantic_digest,
        session_state_budget: cases.session_state_budget.clone(),
        checkpoint_budget: cases.checkpoint_budget.clone(),
        issue_reporter: handle,
    };
    let validated = factory
        .validate(
            &cases.authored,
            cases.format_descriptor,
            cases.workload_descriptor,
        )
        .expect("authored session configuration validates");
    let mut coordinator = factory
        .prepare(validated, &context)
        .expect("session preparation succeeds");
    assert_eq!(
        coordinator.participant_id(),
        cases.participant_id,
        "a prepared coordinator adopts the plan-frozen participant identity"
    );

    coordinator
        .initialize(None)
        .await
        .expect("fresh participant initialization");
    assert!(
        coordinator.initialize(None).await.is_err(),
        "a participant initializes exactly once"
    );

    let mut sink = CollectingActionSink::default();
    for fragment in cases.fragments {
        (cases.advance)();
        coordinator
            .ingest(fragment, &mut sink)
            .await
            .expect("an accepted fragment is incorporated");
    }
    assert_eq!(
        sink.actions.len(),
        cases.expected_action_count,
        "the scripted fragments produce exactly the declared actions"
    );
    assert!(
        !sink.frontiers.is_empty(),
        "a coordinator advances the causal frontier it has proven"
    );

    let receipt = coordinator
        .seal(cases.seal, &mut sink)
        .await
        .expect("coordinator accepts an explicit source seal");
    assert_ne!(
        receipt.digest,
        ContentDigest::from_bytes([0; 32]),
        "a seal receipt binds a real digest"
    );

    let barrier = barrier_at(cases.run, 4);
    let prepared = coordinator
        .checkpoint_view(&barrier)
        .await
        .expect("coordinator prepares a non-destructive view");
    assert_eq!(prepared.descriptor().participant_id, cases.participant_id);
    let payload = Bytes::from(prepared.payload_bytes().to_vec());
    assert!(
        !payload.is_empty(),
        "a prepared participant payload is never empty"
    );

    let total = reporter
        .summary()
        .expect("reporter summary is available after conformance")
        .total;
    assert_eq!(
        total, cases.expected_issue_count,
        "scripted ordinary faults are the only reporter receipts"
    );
}

fn assert_strict_validation(
    factory: &dyn StreamingSessionProgramFactory,
    cases: &SessionConformanceCases,
) {
    let descriptor = factory.descriptor();
    assert!(
        !descriptor.id.is_empty(),
        "a registered session program declares a stable identifier"
    );
    assert!(
        descriptor
            .fragment_input_schemas
            .contains(&cases.format_descriptor.output_schema),
        "a conformance pairing agrees on the canonical fragment schema"
    );
    factory
        .validate(
            &cases.authored,
            cases.format_descriptor,
            cases.workload_descriptor,
        )
        .expect("authored session configuration validates");
    assert!(
        factory
            .validate(
                &cases.rejected_authored,
                cases.format_descriptor,
                cases.workload_descriptor,
            )
            .is_err(),
        "unknown or malformed session configuration is refused before preparation"
    );
}
