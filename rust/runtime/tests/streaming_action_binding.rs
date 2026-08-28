// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multiplexed action-host binding, ordering, and terminal-membership tests.

use std::{cell::RefCell, collections::BTreeSet, collections::VecDeque, rc::Rc};

use aiperf_runtime::streaming::{
    action::{
        ActionAdmissionReceipt, ActionCancelReceipt, ActionDrainReceipt, ActionEventIdentity,
        ActionExecutionCancelReceiver, ActionExecutionError, ActionExecutionEvent,
        ActionFailureCode, ActionFirstTokenReceipt, ActionHandleId, ActionTerminalDisposition,
        ActionTerminalReceipt, ActiveExecutionSet, DatasetActionSchema, OrderedDatasetAction,
        PreparedStreamingActionBinding, StreamingActionBindingSet, StreamingActionDriver,
        StreamingActionDriverControl, StreamingActionDriverControlOps, StreamingActionHost,
        StreamingActionSubmitter, SubmittedAction, action_execution_control,
        canonical_action_schema, session_state::SESSION_STATE_ACTION_SINK,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointBarrier, CheckpointError, CheckpointParticipantId, CommittedParticipantReceipt,
        CommittedParticipantState, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    identity::{
        ActionAttemptId, ContentDigest, GlobalSequence, ImmutableObjectIdentity,
        LogicalReplayRunId, SessionOwnershipEpoch, StableActionId, StableOrderKey,
        StableSessionKey,
    },
    unit::{
        ActionContentLeaseSet, DatasetActionKind, DatasetActionV1, ExecutableDatasetAction,
        SessionFragmentLease, SessionRequestAction, SourcePosition, UnitProvenance,
    },
};
use async_trait::async_trait;

fn budget(max_items: usize, max_bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes,
    })
    .expect("valid limits")
}

fn run() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x31; 32]))
}

async fn request_action(
    content_budget: &StreamingResourceBudget,
    action_id: StableActionId,
) -> ExecutableDatasetAction {
    let request = vec![0_u8; 8];
    let charge = request.capacity() + 64;
    let fragment = SessionFragmentLease::try_from(
        content_budget
            .acquire(1, charge)
            .await
            .expect("content charge"),
    )
    .expect("one-item fragment");
    ExecutableDatasetAction::new(
        action_id,
        StableSessionKey::from_bytes([2; 32]),
        Default::default(),
        None,
        StableOrderKey::from_bytes([3; 32]),
        SourcePosition::new(4),
        UnitProvenance {
            source_partition: ImmutableObjectIdentity::from_bytes([5; 32]),
            source_position: SourcePosition::new(4),
            format_semantic_digest: ContentDigest::from_bytes([6; 32]),
        },
        DatasetActionV1::Request(SessionRequestAction { request }),
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect("fully charged action")
}

/// Event queue shared by the fake submitter, driver, control, and test emitter.
#[derive(Debug, Default)]
struct FakeShared {
    events: RefCell<VecDeque<ActionExecutionEvent>>,
    submitted: RefCell<Vec<StableActionId>>,
    receivers: RefCell<Vec<ActionExecutionCancelReceiver>>,
    is_issuing_stopped: RefCell<bool>,
    wake: tokio::sync::Notify,
}

#[derive(Debug)]
struct FakeSubmitter {
    schema: DatasetActionSchema,
    shared: Rc<FakeShared>,
}

#[async_trait(?Send)]
impl StreamingActionSubmitter for FakeSubmitter {
    fn accepted_schema(&self) -> DatasetActionSchema {
        self.schema.clone()
    }

    async fn submit(
        &mut self,
        action: OrderedDatasetAction,
    ) -> Result<SubmittedAction, ActionExecutionError> {
        if *self.shared.is_issuing_stopped.borrow() {
            return Err(ActionExecutionError::action(ActionFailureCode::Cancelled));
        }
        let action_id = action.action().action_id();
        self.shared.submitted.borrow_mut().push(action_id);
        let (control, receiver) = action_execution_control();
        self.shared.receivers.borrow_mut().push(receiver);
        Ok(SubmittedAction {
            handle_id: ActionHandleId::new(self.shared.submitted.borrow().len() as u64),
            control,
        })
    }
}

#[derive(Debug)]
struct FakeDriver {
    shared: Rc<FakeShared>,
}

#[async_trait(?Send)]
impl StreamingActionDriver for FakeDriver {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError> {
        loop {
            if let Some(event) = self.shared.events.borrow_mut().pop_front() {
                return Ok(event);
            }
            self.shared.wake.notified().await;
        }
    }

    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError> {
        Ok(ActionDrainReceipt {
            submitted: self.shared.submitted.borrow().len() as u64,
            terminal: self.shared.submitted.borrow().len() as u64,
            digest: ContentDigest::from_bytes([0x7a; 32]),
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeDriver {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new("test.action.driver")
    }

    async fn checkpoint_view(
        &mut self,
        _barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        Err(CheckpointError::ParticipantUnavailable {
            participant: self.participant_id(),
        })
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

#[derive(Debug)]
struct FakeControl {
    shared: Rc<FakeShared>,
}

#[async_trait(?Send)]
impl StreamingActionDriverControlOps for FakeControl {
    fn stop_issuing(&self) {
        *self.shared.is_issuing_stopped.borrow_mut() = true;
        self.shared.wake.notify_one();
    }

    fn cancel_pending(&self) {
        self.shared.events.borrow_mut().clear();
        self.shared.wake.notify_one();
    }

    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError> {
        Ok(ActionCancelReceipt {
            cancelled: 0,
            digest: ContentDigest::from_bytes([0x7b; 32]),
        })
    }
}

/// Test-side handle that drives the fake binding's event stream.
struct FakeEmitter {
    shared: Rc<FakeShared>,
}

impl FakeEmitter {
    fn identity(action_id: StableActionId, event_ordinal: u64) -> ActionEventIdentity {
        ActionEventIdentity {
            action_id,
            attempt_id: ActionAttemptId::from_bytes([0x41; 32]),
            ownership_epoch: SessionOwnershipEpoch::new(0),
            event_ordinal,
        }
    }

    fn emit(&self, event: ActionExecutionEvent) {
        self.shared.events.borrow_mut().push_back(event);
        self.shared.wake.notify_one();
    }

    fn admitted(&self, action_id: StableActionId, ordinal: u64) {
        self.emit(ActionExecutionEvent::Admitted(ActionAdmissionReceipt {
            event: Self::identity(action_id, ordinal),
        }));
    }

    fn first_token(&self, action_id: StableActionId, ordinal: u64) {
        self.emit(ActionExecutionEvent::FirstToken(ActionFirstTokenReceipt {
            event: Self::identity(action_id, ordinal),
        }));
    }

    fn terminal(&self, action_id: StableActionId, ordinal: u64) {
        self.emit(ActionExecutionEvent::Terminal(ActionTerminalReceipt {
            event: Self::identity(action_id, ordinal),
            disposition: ActionTerminalDisposition::Completed,
        }));
    }
}

fn fake_binding(schema: DatasetActionSchema) -> (PreparedStreamingActionBinding, FakeEmitter) {
    let shared = Rc::new(FakeShared::default());
    let binding = PreparedStreamingActionBinding {
        submitter: Box::new(FakeSubmitter {
            schema,
            shared: Rc::clone(&shared),
        }),
        driver: Box::new(FakeDriver {
            shared: Rc::clone(&shared),
        }),
        control: StreamingActionDriverControl::new(FakeControl {
            shared: Rc::clone(&shared),
        }),
    };
    (binding, FakeEmitter { shared })
}

fn request_schema_set() -> BTreeSet<DatasetActionSchema> {
    let mut schemas = BTreeSet::new();
    schemas.insert(canonical_action_schema(DatasetActionKind::Request));
    schemas
}

fn fake_action_host(
    active: StreamingResourceBudget,
) -> (
    StreamingActionHost,
    FakeEmitter,
    StreamingActionDriverControl,
) {
    let schema = canonical_action_schema(DatasetActionKind::Request);
    let (binding, emitter) = fake_binding(schema.clone());
    let mut set = StreamingActionBindingSet::new();
    set.insert(binding).expect("first binding");
    let (host, controls) =
        StreamingActionHost::new(run(), &request_schema_set(), set, active).expect("bound host");
    let control = controls.get(&schema).expect("control for schema").clone();
    (host, emitter, control)
}

#[tokio::test(flavor = "current_thread")]
async fn accepted_actions_have_one_terminal_receipt_in_event_order() {
    let content = budget(4, 4096);
    let (mut host, emitter, _control) = fake_action_host(budget(8, 8192));
    let action_id = StableActionId::from_bytes([0x11; 32]);
    let action = request_action(&content, action_id).await;

    let sequence = host.submit(action).await.expect("submitted");
    assert_eq!(sequence, GlobalSequence::new(0));

    emitter.admitted(action_id, 0);
    emitter.first_token(action_id, 1);
    emitter.terminal(action_id, 2);

    let batch = host.drain_events().await.expect("drained");
    assert_eq!(batch.ordinals(), vec![0, 1, 2]);
    assert_eq!(host.terminal_membership(action_id), 1);
    assert!(host.active().is_empty());

    let membership = host
        .checked_terminal_membership(action_id)
        .expect("finalized membership");
    assert_eq!(membership.sequence(), GlobalSequence::new(0));
    assert_eq!(membership.action_id(), action_id);

    let mut finalized = host.take_finalized(action_id).expect("finalized execution");
    let receipt = finalized
        .take_terminal_receipt()
        .expect("terminal receipt moves with its charge");
    assert_eq!(receipt.receipt().event.event_ordinal, 2);
    assert!(finalized.take_terminal_receipt().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn binding_map_must_cover_every_emitted_schema_exactly_once() {
    let missing = StreamingActionHost::new(
        run(),
        &request_schema_set(),
        StreamingActionBindingSet::new(),
        budget(4, 4096),
    );
    assert!(matches!(
        missing.err(),
        Some(ActionExecutionError::Action(
            ActionFailureCode::MissingBinding
        ))
    ));

    let schema = canonical_action_schema(DatasetActionKind::Request);
    let mut set = StreamingActionBindingSet::new();
    set.insert(fake_binding(schema.clone()).0).expect("first");
    let duplicate = set.insert(fake_binding(schema).0);
    assert!(matches!(
        duplicate.err(),
        Some(ActionExecutionError::Action(
            ActionFailureCode::DuplicateBinding
        ))
    ));

    let mut unexpected = StreamingActionBindingSet::new();
    unexpected
        .insert(fake_binding(canonical_action_schema(DatasetActionKind::Request)).0)
        .expect("request binding");
    unexpected
        .insert(fake_binding(canonical_action_schema(DatasetActionKind::GraphNode)).0)
        .expect("graph binding");
    let error = StreamingActionHost::new(run(), &request_schema_set(), unexpected, budget(4, 4096));
    assert!(matches!(
        error.err(),
        Some(ActionExecutionError::Action(
            ActionFailureCode::UnexpectedBinding
        ))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn active_set_refuses_replayed_ordinals_and_second_terminals() {
    let mut active = ActiveExecutionSet::new(budget(8, 8192));
    let action_id = StableActionId::from_bytes([0x12; 32]);
    let (control, _receiver) = action_execution_control();
    active
        .admit(
            action_id,
            GlobalSequence::new(0),
            SubmittedAction {
                handle_id: ActionHandleId::new(0),
                control,
            },
        )
        .expect("admitted");

    active.observe_event(action_id, 0).expect("first event");
    assert!(matches!(
        active.observe_event(action_id, 0).err(),
        Some(ActionExecutionError::Action(ActionFailureCode::EventOrder))
    ));
    assert!(matches!(
        active
            .observe_event(StableActionId::from_bytes([0x99; 32]), 0)
            .err(),
        Some(ActionExecutionError::Action(
            ActionFailureCode::UnknownAction
        ))
    ));

    let terminal = ActionTerminalReceipt {
        event: FakeEmitter::identity(action_id, 1),
        disposition: ActionTerminalDisposition::Completed,
    };
    active
        .finish(action_id, terminal.clone())
        .expect("one terminal");
    assert!(matches!(
        active.finish(action_id, terminal).err(),
        Some(ActionExecutionError::Action(
            ActionFailureCode::DuplicateTerminal
        ))
    ));
}

#[test]
fn session_state_sink_executes_no_endpoint() {
    assert_eq!(SESSION_STATE_ACTION_SINK.id, "session_state");
    assert!(SESSION_STATE_ACTION_SINK.transport_ids.is_empty());
    assert!(SESSION_STATE_ACTION_SINK.endpoint_kinds.is_empty());
    assert!(SESSION_STATE_ACTION_SINK.supports_virtual_clock);
    for kind in [
        DatasetActionKind::Request,
        DatasetActionKind::GraphNode,
        DatasetActionKind::SessionTerminal,
    ] {
        assert!(
            SESSION_STATE_ACTION_SINK
                .accepted_schemas
                .contains(&canonical_action_schema(kind).as_str()),
            "descriptor must advertise every canonical schema"
        );
    }
}
