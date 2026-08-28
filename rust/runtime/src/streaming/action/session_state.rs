// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state-only streaming action sink.
//!
//! `session_state` accepts one canonical action schema and produces the
//! admitted and terminal membership the host needs without contacting an
//! endpoint. It exists so a run whose session program emits state-only actions
//! — session terminals, graph-node bookkeeping — has a real prepared binding
//! instead of a special case threaded through the host.
//!
//! Because nothing is issued, the binding declares no transport and no endpoint
//! family: the empty descriptor lists are the accurate statement that this sink
//! is compatible with every selection, not an omission.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, VecDeque},
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{
    ActionAdmissionReceipt, ActionCancelReceipt, ActionDrainReceipt, ActionEventIdentity,
    ActionExecutionError, ActionExecutionEvent, ActionFailureCode, ActionHandleId, ActionPlacement,
    ActionResultRetention, ActionTerminalDisposition, ActionTerminalReceipt, DatasetActionSchema,
    EndpointRetrySafety, OrderedDatasetAction, PreparedStreamingActionBinding,
    StreamingActionDriver, StreamingActionDriverControl, StreamingActionDriverControlOps,
    StreamingActionSinkDescriptor, StreamingActionSinkFactory, StreamingActionSinkPrepareContext,
    StreamingActionSubmitter, SubmittedAction, ValidatedStreamingActionSinkConfig,
    action_execution_control, canonical_action_schema,
};
use crate::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
        PreparedParticipantState, StreamingCheckpointParticipant,
    },
    identity::{ActionAttemptId, ContentDigest, SessionOwnershipEpoch, StableActionId},
    unit::DatasetActionKind,
};

/// Stable participant identity of the state-only sink's driver.
const PARTICIPANT_ID: &str = "aiperf.stream.action.session_state";

/// Checkpoint schema of the driver's retained admitted-action set.
const STATE_SCHEMA_ID: &str = "aiperf.stream.action.session_state.v1";

/// Registry metadata for the built-in state-only action sink.
pub static SESSION_STATE_ACTION_SINK: StreamingActionSinkDescriptor =
    StreamingActionSinkDescriptor {
        id: "session_state",
        description: "State-only action sink that admits and finalizes without endpoint execution",
        accepted_schemas: &[
            "aiperf.action.request.v1",
            "aiperf.action.graph_node.v1",
            "aiperf.action.session_terminal.v1",
        ],
        transport_ids: &[],
        endpoint_kinds: &[],
        retention: ActionResultRetention::StreamingTerminal,
        placement: ActionPlacement::WorkerLocal,
        endpoint_retry_safety: crate::streaming::action::EndpointRetrySafety::Unproven,
        supports_virtual_clock: true,
        // The sink never reaches an endpoint, so no endpoint-retry proof
        // exists to claim and the refusing default stands.
        endpoint_retry_safety: EndpointRetrySafety::Unproven,
    };

/// Authored configuration accepted by the state-only action sink.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct AuthoredSessionStateConfig {
    /// Item capacity of the sink's private checkpoint-payload budget.
    max_state_items: usize,
    /// Byte capacity of the sink's private checkpoint-payload budget.
    max_state_bytes: usize,
}

impl Default for AuthoredSessionStateConfig {
    fn default() -> Self {
        Self {
            max_state_items: 16,
            max_state_bytes: 1 << 20,
        }
    }
}

/// Validated startup-only configuration bound to one exact schema.
#[derive(Clone, Debug)]
struct ValidatedSessionStateConfig {
    schema: DatasetActionSchema,
    limits: BudgetLimits,
}

/// Checkpointed state of the state-only sink.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SessionStateCheckpoint {
    /// Actions admitted and not yet finalized, in stable identity order.
    admitted: Vec<StableActionId>,
    /// Total actions submitted through this binding.
    submitted: u64,
    /// Total terminal receipts emitted by this binding.
    terminal: u64,
}

/// Worker-local state shared by the submitter, driver, and control.
///
/// Every field is single-threaded on purpose: this sink runs inside one worker's
/// `LocalSet`, so `Cell`/`RefCell` are the correct primitives and a lock here
/// would be pure overhead.
#[derive(Debug)]
struct SessionStateShared {
    binding_digest: ContentDigest,
    events: RefCell<VecDeque<ActionExecutionEvent>>,
    admitted: RefCell<BTreeSet<StableActionId>>,
    submitted: Cell<u64>,
    terminal: Cell<u64>,
    is_issuing_stopped: Cell<bool>,
    cancelled: Cell<u64>,
    wake: tokio::sync::Notify,
}

impl SessionStateShared {
    fn new(binding_digest: ContentDigest) -> Self {
        Self {
            binding_digest,
            events: RefCell::new(VecDeque::new()),
            admitted: RefCell::new(BTreeSet::new()),
            submitted: Cell::new(0),
            terminal: Cell::new(0),
            is_issuing_stopped: Cell::new(false),
            cancelled: Cell::new(0),
            wake: tokio::sync::Notify::new(),
        }
    }

    fn attempt_id(&self, action_id: StableActionId) -> ActionAttemptId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.action.session_state.attempt.v1");
        hasher.update(self.binding_digest.as_bytes());
        hasher.update(action_id.as_bytes());
        ActionAttemptId::from_bytes(*hasher.finalize().as_bytes())
    }

    fn digest(&self, domain: &[u8]) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain);
        hasher.update(self.binding_digest.as_bytes());
        hasher.update(&self.submitted.get().to_le_bytes());
        hasher.update(&self.terminal.get().to_le_bytes());
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

/// Submission half of the state-only binding.
#[derive(Debug)]
struct SessionStateSubmitter {
    schema: DatasetActionSchema,
    shared: Rc<SessionStateShared>,
    next_handle: u64,
    /// Retained cancellation receivers so a per-action control stays live.
    receivers: Vec<super::ActionExecutionCancelReceiver>,
}

#[async_trait(?Send)]
impl StreamingActionSubmitter for SessionStateSubmitter {
    fn accepted_schema(&self) -> DatasetActionSchema {
        self.schema.clone()
    }

    async fn submit(
        &mut self,
        action: OrderedDatasetAction,
    ) -> Result<SubmittedAction, ActionExecutionError> {
        if self.shared.is_issuing_stopped.get() {
            return Err(ActionExecutionError::action(ActionFailureCode::Cancelled));
        }
        let action_id = action.action().action_id();
        if !self.shared.admitted.borrow_mut().insert(action_id) {
            return Err(ActionExecutionError::action(ActionFailureCode::Dispatch));
        }
        // The action's leases are released here: a state-only sink retains no
        // request bytes past admission.
        drop(action);

        let event = ActionEventIdentity {
            action_id,
            attempt_id: self.shared.attempt_id(action_id),
            ownership_epoch: SessionOwnershipEpoch::new(0),
            event_ordinal: 0,
        };
        let mut terminal_event = event.clone();
        terminal_event.event_ordinal = 1;
        {
            let mut events = self.shared.events.borrow_mut();
            events.push_back(ActionExecutionEvent::Admitted(ActionAdmissionReceipt {
                event,
            }));
            events.push_back(ActionExecutionEvent::Terminal(ActionTerminalReceipt {
                event: terminal_event,
                disposition: ActionTerminalDisposition::Completed,
            }));
        }
        self.shared.submitted.set(
            self.shared
                .submitted
                .get()
                .checked_add(1)
                .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?,
        );
        self.shared.wake.notify_one();

        let handle_id = ActionHandleId::new(self.next_handle);
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?;
        let (control, receiver) = action_execution_control();
        self.receivers.push(receiver);
        Ok(SubmittedAction { handle_id, control })
    }
}

/// Multiplexed event half of the state-only binding.
#[derive(Debug)]
struct SessionStateDriver {
    shared: Rc<SessionStateShared>,
    budget: StreamingResourceBudget,
    initialization: ParticipantInitialization,
}

#[async_trait(?Send)]
impl StreamingActionDriver for SessionStateDriver {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError> {
        loop {
            // Pop before awaiting so a dropped future never consumes an event.
            if let Some(event) = self.shared.events.borrow_mut().pop_front() {
                if let ActionExecutionEvent::Terminal(receipt) = &event {
                    self.shared.admitted.borrow_mut().remove(&receipt.event.action_id);
                    self.shared.terminal.set(
                        self.shared.terminal.get().checked_add(1).ok_or_else(|| {
                            ActionExecutionError::action(ActionFailureCode::Dispatch)
                        })?,
                    );
                }
                return Ok(event);
            }
            self.shared.wake.notified().await;
        }
    }

    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError> {
        while let Some(event) = {
            let next = self.shared.events.borrow_mut().pop_front();
            next
        } {
            if let ActionExecutionEvent::Terminal(receipt) = &event {
                self.shared.admitted.borrow_mut().remove(&receipt.event.action_id);
                self.shared.terminal.set(
                    self.shared
                        .terminal
                        .get()
                        .checked_add(1)
                        .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?,
                );
            }
        }
        Ok(ActionDrainReceipt {
            submitted: self.shared.submitted.get(),
            terminal: self.shared.terminal.get(),
            digest: self.shared.digest(b"aiperf.stream.action.session_state.drain.v1"),
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for SessionStateDriver {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new(PARTICIPANT_ID)
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let state = SessionStateCheckpoint {
            admitted: self.shared.admitted.borrow().iter().copied().collect(),
            submitted: self.shared.submitted.get(),
            terminal: self.shared.terminal.get(),
        };
        let encoded =
            serde_json::to_vec(&state).map_err(|_| CheckpointError::ObjectVerification)?;
        let item_count = u64::try_from(state.admitted.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = self
            .budget
            .try_acquire(1, encoded.len())
            .map_err(|error| CheckpointError::StateBudget {
                participant: CheckpointParticipantId::new(PARTICIPANT_ID),
                code: budget_state_code(error),
            })?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(encoded), lease)?;
        PreparedParticipantState::new(
            barrier.run,
            CheckpointParticipantId::new(PARTICIPANT_ID),
            STATE_SCHEMA_ID,
            1,
            barrier.cut.clone(),
            item_count,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        if state.descriptor().schema_id != STATE_SCHEMA_ID {
            return Err(CheckpointError::ObjectVerification);
        }
        let restored: SessionStateCheckpoint = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        *self.shared.admitted.borrow_mut() = restored.admitted.into_iter().collect();
        self.shared.submitted.set(restored.submitted);
        self.shared.terminal.set(restored.terminal);
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

fn budget_state_code(
    error: crate::streaming::budget::BudgetError,
) -> crate::streaming::unit::StateBudgetFailureCode {
    use crate::streaming::{budget::BudgetError, unit::StateBudgetFailureCode};
    match error {
        BudgetError::CapacityUnavailable => StateBudgetFailureCode::ItemCapacity,
        BudgetError::RequestExceedsCapacity | BudgetError::PermitCountTooLarge => {
            StateBudgetFailureCode::ByteCapacity
        }
        _ => StateBudgetFailureCode::PermanentError,
    }
}

/// Separately borrowable control for the state-only binding.
#[derive(Debug)]
struct SessionStateControl {
    shared: Rc<SessionStateShared>,
}

#[async_trait(?Send)]
impl StreamingActionDriverControlOps for SessionStateControl {
    fn stop_issuing(&self) {
        self.shared.is_issuing_stopped.set(true);
        self.shared.wake.notify_one();
    }

    fn cancel_pending(&self) {
        let drained = self.shared.events.borrow_mut().drain(..).count();
        self.shared
            .cancelled
            .set(self.shared.cancelled.get().saturating_add(drained as u64));
        self.shared.wake.notify_one();
    }

    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError> {
        let inflight = self.shared.admitted.borrow_mut().len();
        self.shared.admitted.borrow_mut().clear();
        let cancelled = self.shared.cancelled.get().saturating_add(inflight as u64);
        self.shared.cancelled.set(cancelled);
        Ok(ActionCancelReceipt {
            cancelled,
            digest: self
                .shared
                .digest(b"aiperf.stream.action.session_state.cancel.v1"),
        })
    }
}

/// Startup factory for the built-in state-only action sink.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionStateActionSinkFactory;

impl StreamingActionSinkFactory for SessionStateActionSinkFactory {
    fn descriptor(&self) -> &'static StreamingActionSinkDescriptor {
        &SESSION_STATE_ACTION_SINK
    }

    fn validate_binding(
        &self,
        authored: &RawValue,
        action: &DatasetActionSchema,
        _transport: &crate::engine::registry::TransportDescriptor,
        _endpoint: &crate::endpoints::EndpointDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingActionSinkConfig>, ActionExecutionError> {
        let authored: AuthoredSessionStateConfig = serde_json::from_str(authored.get())
            .map_err(|_| ActionExecutionError::action(ActionFailureCode::MissingBinding))?;
        let is_known = [
            DatasetActionKind::Request,
            DatasetActionKind::GraphNode,
            DatasetActionKind::SessionTerminal,
        ]
        .into_iter()
        .any(|kind| &canonical_action_schema(kind) == action);
        if !is_known {
            return Err(ActionExecutionError::action(
                ActionFailureCode::MissingBinding,
            ));
        }
        if authored.max_state_items == 0 || authored.max_state_bytes == 0 {
            return Err(ActionExecutionError::state_budget(
                crate::streaming::unit::StateBudgetFailureCode::PermanentError,
            ));
        }
        Ok(Box::new(ValidatedSessionStateConfig {
            schema: action.clone(),
            limits: BudgetLimits {
                max_items: authored.max_state_items,
                max_bytes: authored.max_state_bytes,
            },
        }))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingActionSinkConfig>,
        context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError> {
        let config = config
            .as_any()
            .downcast_ref::<ValidatedSessionStateConfig>()
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::MissingBinding))?
            .clone();
        let budget = StreamingResourceBudget::new(config.limits).map_err(|_| {
            ActionExecutionError::state_budget(
                crate::streaming::unit::StateBudgetFailureCode::PermanentError,
            )
        })?;
        let shared = Rc::new(SessionStateShared::new(context.binding_semantic_digest));
        Ok(PreparedStreamingActionBinding {
            submitter: Box::new(SessionStateSubmitter {
                schema: config.schema,
                shared: Rc::clone(&shared),
                next_handle: 0,
                receivers: Vec::new(),
            }),
            driver: Box::new(SessionStateDriver {
                shared: Rc::clone(&shared),
                budget,
                initialization: ParticipantInitialization::default(),
            }),
            control: StreamingActionDriverControl::new(SessionStateControl { shared }),
        })
    }
}
