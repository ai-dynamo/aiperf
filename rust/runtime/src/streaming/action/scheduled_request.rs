// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-executing streaming action binding over the scheduled issuer.
//!
//! One `aiperf.action.request.v1` action becomes one request submitted through
//! [`ScheduledRuntime`]. This is the binding that makes the streaming plane
//! actually reach an inference endpoint; every other registered binding either
//! resolves state locally or refuses.
//!
//! The dispatch lifecycle is translated into `ActionExecutionEvent`s at the
//! callbacks the scheduled runtime already exposes — the first-token hook and
//! the completion handler — so no per-token hook, allocation, or lock is added
//! to the hot path. TTFT is the first token observation, exactly as the
//! transport-neutral dispatch seam defines it.
//!
//! Three facts drive the shape of this module:
//!
//! - `ScheduledRuntime` has no admission bound of its own. Boundedness here is
//!   the submitter's explicit `max_active_actions` refusal, not an assumption
//!   about the runtime.
//! - `issue_turn_*` returning `false` is an ordinary *drop*, not an error. It
//!   becomes [`ActionTerminalDisposition::Dropped`] so every accepted action
//!   still has exactly one terminal receipt and an ordinary stop condition
//!   never becomes a workload failure.
//! - The event queue is a `VecDeque` whose capacity is bounded by construction:
//!   each accepted action emits at most admitted, first-token, and terminal, so
//!   the queue can never exceed `max_active_actions * MAX_EVENTS_PER_ACTION`.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, VecDeque},
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{
    ActionAdmissionReceipt, ActionCancelReceipt, ActionDrainReceipt, ActionEventIdentity,
    ActionExecutionError, ActionExecutionEvent, ActionFailureCode, ActionFirstTokenReceipt,
    ActionHandleId, ActionPlacement, ActionResultRetention, ActionTerminalDisposition,
    ActionTerminalReceipt, DatasetActionSchema, EndpointRetrySafety, OrderedDatasetAction,
    PreparedStreamingActionBinding, StreamingActionDriver, StreamingActionDriverControl,
    StreamingActionDriverControlOps, StreamingActionSinkDescriptor,
    StreamingActionSinkPrepareContext, StreamingActionSubmitter, SubmittedAction,
    action_execution_control, canonical_action_schema,
};
use crate::{
    body_plan::RequestBody,
    multiturn::{StreamingActionTurn, TurnEndpoint, streaming_action_turn},
    scheduled::ScheduledRuntime,
    streaming::{
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
            CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
            PreparedParticipantState, StreamingCheckpointParticipant,
        },
        identity::{ActionAttemptId, ContentDigest, RunIncarnationId, SessionOwnershipEpoch,
            StableActionId},
        unit::{DatasetActionKind, DatasetActionV1},
    },
};

/// Stable participant identity of this binding's driver.
const PARTICIPANT_ID: &str = "aiperf.stream.action.scheduled_request";

/// Checkpoint schema of the driver's retained in-flight set.
const STATE_SCHEMA_ID: &str = "aiperf.stream.action.scheduled_request.v1";

/// Maximum events one accepted action can emit: admitted, first token, terminal.
const MAX_EVENTS_PER_ACTION: usize = 3;

/// Registry metadata for the endpoint-executing action binding.
pub static SCHEDULED_REQUEST_ACTION_SINK: StreamingActionSinkDescriptor =
    StreamingActionSinkDescriptor {
        id: "scheduled_request",
        description: "Issue one streaming action through the scheduled request issuer",
        accepted_schemas: &["aiperf.action.request.v1"],
        transport_ids: &["http", "grpc", "dry_run"],
        endpoint_kinds: &["chat", "completions", "responses"],
        retention: ActionResultRetention::StreamingTerminal,
        placement: ActionPlacement::WorkerLocal,
        supports_virtual_clock: true,
        // Reaching a real endpoint means a retry duplicates measured load. The
        // binding proves nothing about duplicate rejection at the target, so a
        // nonzero authored endpoint retry limit is refused.
        endpoint_retry_safety: EndpointRetrySafety::Unproven,
    };

/// Authored configuration accepted by the endpoint-executing sink.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct AuthoredScheduledRequestConfig {
    /// Maximum concurrently in-flight actions this binding admits.
    max_active_actions: usize,
    /// Maximum bytes of normalized endpoint reply retained per action.
    max_update_bytes: usize,
    /// Whether the endpoint should stream its response.
    is_streaming: bool,
}

impl Default for AuthoredScheduledRequestConfig {
    fn default() -> Self {
        Self {
            max_active_actions: 64,
            max_update_bytes: 1 << 16,
            is_streaming: true,
        }
    }
}

/// Startup-validated configuration bound to one exact schema.
#[derive(Clone, Debug)]
struct ValidatedScheduledRequestConfig {
    schema: DatasetActionSchema,
    max_active_actions: usize,
    max_update_bytes: usize,
    is_streaming: bool,
}

/// Checkpointed state of the endpoint-executing sink.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScheduledRequestCheckpoint {
    /// Actions issued and not yet terminal, in stable identity order.
    inflight: Vec<StableActionId>,
    /// Total actions submitted through this binding.
    submitted: u64,
    /// Total terminal receipts emitted by this binding.
    terminal: u64,
}

/// Per-action state retained between submit and terminal.
#[derive(Debug)]
struct ActiveScheduledAction {
    attempt_id: ActionAttemptId,
    ownership_epoch: SessionOwnershipEpoch,
    next_event_ordinal: u64,
}

/// Bounded event queue and in-flight map shared by the three binding halves.
///
/// Single-threaded on purpose: the binding runs inside one worker's `LocalSet`,
/// so `Cell`/`RefCell` are the correct primitives and a lock would be overhead
/// on the request path.
#[derive(Debug)]
struct ScheduledRequestShared {
    binding_digest: ContentDigest,
    incarnation: RunIncarnationId,
    max_active_actions: usize,
    events: RefCell<VecDeque<ActionExecutionEvent>>,
    inflight: RefCell<BTreeMap<StableActionId, ActiveScheduledAction>>,
    submitted: Cell<u64>,
    terminal: Cell<u64>,
    cancelled: Cell<u64>,
    is_issuing_stopped: Cell<bool>,
    wake: tokio::sync::Notify,
}

impl ScheduledRequestShared {
    fn new(
        binding_digest: ContentDigest,
        incarnation: RunIncarnationId,
        max_active_actions: usize,
    ) -> Self {
        Self {
            binding_digest,
            incarnation,
            max_active_actions,
            events: RefCell::new(VecDeque::with_capacity(
                max_active_actions.saturating_mul(MAX_EVENTS_PER_ACTION),
            )),
            inflight: RefCell::new(BTreeMap::new()),
            submitted: Cell::new(0),
            terminal: Cell::new(0),
            cancelled: Cell::new(0),
            is_issuing_stopped: Cell::new(false),
            wake: tokio::sync::Notify::new(),
        }
    }

    /// Reserve the next event ordinal for an action still in flight.
    ///
    /// Returns the identity the event must carry, or `None` when the action is
    /// already terminal — a late hook firing after terminal is dropped rather
    /// than emitted out of order.
    fn next_identity(&self, action_id: StableActionId) -> Option<ActionEventIdentity> {
        let mut inflight = self.inflight.borrow_mut();
        let active = inflight.get_mut(&action_id)?;
        let event_ordinal = active.next_event_ordinal;
        active.next_event_ordinal = event_ordinal.checked_add(1)?;
        Some(ActionEventIdentity {
            action_id,
            attempt_id: active.attempt_id,
            ownership_epoch: active.ownership_epoch,
            event_ordinal,
        })
    }

    fn push_event(&self, event: ActionExecutionEvent) {
        self.events.borrow_mut().push_back(event);
        self.wake.notify_one();
    }

    /// Emit the unique terminal receipt for `action_id`, if it is still live.
    fn push_terminal(&self, action_id: StableActionId, disposition: ActionTerminalDisposition) {
        let Some(event) = self.next_identity(action_id) else {
            return;
        };
        self.inflight.borrow_mut().remove(&action_id);
        self.push_event(ActionExecutionEvent::Terminal(ActionTerminalReceipt {
            event,
            disposition,
        }));
    }

    fn attempt_id(&self, action_id: StableActionId) -> ActionAttemptId {
        crate::streaming::identity::attempt_id(action_id, self.incarnation, 0)
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

/// Submission half: wrap one action's request bytes and issue them.
struct ScheduledRequestSubmitter {
    schema: DatasetActionSchema,
    shared: Rc<ScheduledRequestShared>,
    runtime: Rc<ScheduledRuntime>,
    endpoint: TurnEndpoint,
    model: Option<String>,
    is_streaming: bool,
    next_handle: u64,
    /// Retained cancellation receivers so each per-action control stays live.
    receivers: Vec<super::ActionExecutionCancelReceiver>,
}

impl std::fmt::Debug for ScheduledRequestSubmitter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScheduledRequestSubmitter")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl StreamingActionSubmitter for ScheduledRequestSubmitter {
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
        // The runtime enforces no admission bound of its own, so the refusal is
        // here or nowhere.
        if self.shared.inflight.borrow().len() >= self.shared.max_active_actions {
            return Err(ActionExecutionError::action(ActionFailureCode::Dispatch));
        }

        let action_id = action.action().action_id();
        let session_id = hex_identity(action.action().session_key().as_bytes());
        let DatasetActionV1::Request(request) = action.action().payload() else {
            return Err(ActionExecutionError::action(
                ActionFailureCode::MissingBinding,
            ));
        };
        let body = RequestBody::Wire(Bytes::from(request.request.clone()));
        let input_length = request.request.len();

        let attempt_id = self.shared.attempt_id(action_id);
        if self
            .shared
            .inflight
            .borrow_mut()
            .insert(
                action_id,
                ActiveScheduledAction {
                    attempt_id,
                    ownership_epoch: SessionOwnershipEpoch::new(0),
                    next_event_ordinal: 0,
                },
            )
            .is_some()
        {
            return Err(ActionExecutionError::action(
                ActionFailureCode::DuplicateTerminal,
            ));
        }
        // The action's content leases are released once its bytes are copied
        // into the turn, which now owns the retained capacity.
        drop(action);

        self.shared.submitted.set(
            self.shared
                .submitted
                .get()
                .checked_add(1)
                .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?,
        );

        if let Some(event) = self.shared.next_identity(action_id) {
            self.shared
                .push_event(ActionExecutionEvent::Admitted(ActionAdmissionReceipt {
                    event,
                }));
        }

        let turn = streaming_action_turn(StreamingActionTurn {
            session_id: session_id.clone(),
            correlation_id: session_id,
            endpoint: self.endpoint.clone(),
            body,
            effective_model: self.model.clone(),
            input_length,
            max_output_tokens: 0,
            is_streaming: self.is_streaming,
        });

        let first_token_shared = Rc::clone(&self.shared);
        let on_first_token: crate::scheduled::FirstTokenHandler = Box::new(move |_ttft_ns| {
            if let Some(event) = first_token_shared.next_identity(action_id) {
                first_token_shared.push_event(ActionExecutionEvent::FirstToken(
                    ActionFirstTokenReceipt { event },
                ));
            }
        });

        let complete_shared = Rc::clone(&self.shared);
        let on_complete: crate::scheduled::CompletionHandler =
            Box::new(move |_credit, outcome| {
                let disposition = terminal_disposition(&outcome.terminal);
                complete_shared.push_terminal(action_id, disposition);
                Box::pin(async {})
            });

        let (control, receiver) = action_execution_control();
        let is_issued =
            self.runtime
                .issue_turn_with_hooks(turn, 0, None, on_first_token, on_complete);
        if !is_issued {
            // A stop condition rejecting issue is an ordinary drop: the action
            // still gets exactly one terminal receipt, and the run continues.
            self.shared
                .push_terminal(action_id, ActionTerminalDisposition::Dropped);
        }

        let handle_id = ActionHandleId::new(self.next_handle);
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?;
        self.receivers.push(receiver);
        Ok(SubmittedAction { handle_id, control })
    }
}

/// Map one dispatch terminal status onto a stable action disposition.
fn terminal_disposition(
    terminal: &crate::dispatch::collector::ReplayTerminalStatus,
) -> ActionTerminalDisposition {
    use crate::dispatch::collector::ReplayTerminalStatus;
    match terminal {
        ReplayTerminalStatus::Completed => ActionTerminalDisposition::Completed,
        ReplayTerminalStatus::Canceled => ActionTerminalDisposition::Cancelled,
        // An admission refusal never reached the endpoint, so it is a drop
        // rather than a failure of the action itself.
        ReplayTerminalStatus::Rejected => ActionTerminalDisposition::Dropped,
        ReplayTerminalStatus::Failed => ActionTerminalDisposition::Failed,
    }
}

/// Render a 32-byte identity as stable lowercase hex.
fn hex_identity(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(64), |mut text, byte| {
        // Writing to a String is infallible; the result is discarded knowingly.
        let _ = write!(text, "{byte:02x}");
        text
    })
}

/// Sole mutable event stream, and a stable checkpoint participant.
#[derive(Debug)]
struct ScheduledRequestDriver {
    shared: Rc<ScheduledRequestShared>,
    budget: crate::streaming::budget::StreamingResourceBudget,
    initialization: ParticipantInitialization,
}

impl ScheduledRequestDriver {
    /// Count one terminal receipt as it leaves the queue.
    fn observe(&self, event: &ActionExecutionEvent) -> Result<(), ActionExecutionError> {
        if matches!(event, ActionExecutionEvent::Terminal(_)) {
            self.shared.terminal.set(
                self.shared
                    .terminal
                    .get()
                    .checked_add(1)
                    .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?,
            );
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingActionDriver for ScheduledRequestDriver {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError> {
        loop {
            // Pop before awaiting so a dropped future never consumes an event.
            let next = self.shared.events.borrow_mut().pop_front();
            if let Some(event) = next {
                self.observe(&event)?;
                return Ok(event);
            }
            self.shared.wake.notified().await;
        }
    }

    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError> {
        // Every accepted action must reach terminal before the receipt is
        // truthful, so wait for the in-flight map to empty rather than
        // reporting whatever happens to be queued right now.
        loop {
            while let Some(event) = {
                let next = self.shared.events.borrow_mut().pop_front();
                next
            } {
                self.observe(&event)?;
            }
            if self.shared.inflight.borrow().is_empty() {
                break;
            }
            self.shared.wake.notified().await;
        }
        Ok(ActionDrainReceipt {
            submitted: self.shared.submitted.get(),
            terminal: self.shared.terminal.get(),
            digest: self
                .shared
                .digest(b"aiperf.stream.action.scheduled_request.drain.v1"),
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for ScheduledRequestDriver {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new(PARTICIPANT_ID)
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let state = ScheduledRequestCheckpoint {
            inflight: self.shared.inflight.borrow().keys().copied().collect(),
            submitted: self.shared.submitted.get(),
            terminal: self.shared.terminal.get(),
        };
        let encoded =
            serde_json::to_vec(&state).map_err(|_| CheckpointError::ObjectVerification)?;
        let item_count =
            u64::try_from(state.inflight.len()).map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = self.budget.try_acquire(1, encoded.len()).map_err(|_| {
            CheckpointError::StateBudget {
                participant: CheckpointParticipantId::new(PARTICIPANT_ID),
                code: crate::streaming::unit::StateBudgetFailureCode::ByteCapacity,
            }
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
        let restored: ScheduledRequestCheckpoint = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
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

/// Separately borrowable control for the endpoint-executing binding.
#[derive(Debug)]
struct ScheduledRequestControl {
    shared: Rc<ScheduledRequestShared>,
}

#[async_trait(?Send)]
impl StreamingActionDriverControlOps for ScheduledRequestControl {
    fn stop_issuing(&self) {
        self.shared.is_issuing_stopped.set(true);
        self.shared.wake.notify_one();
    }

    fn cancel_pending(&self) {
        self.shared.is_issuing_stopped.set(true);
        self.shared.wake.notify_one();
    }

    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError> {
        self.shared.is_issuing_stopped.set(true);
        // Cancelling still owes each accepted action its unique terminal
        // receipt, so the in-flight set is finalized rather than discarded.
        let live: Vec<StableActionId> = self.shared.inflight.borrow().keys().copied().collect();
        let cancelled = self
            .shared
            .cancelled
            .get()
            .saturating_add(live.len() as u64);
        for action_id in live {
            self.shared
                .push_terminal(action_id, ActionTerminalDisposition::Cancelled);
        }
        self.shared.cancelled.set(cancelled);
        Ok(ActionCancelReceipt {
            cancelled,
            digest: self
                .shared
                .digest(b"aiperf.stream.action.scheduled_request.cancel.v1"),
        })
    }
}

/// Startup factory for the endpoint-executing action sink.
pub struct ScheduledRequestActionSinkFactory {
    runtime: Rc<ScheduledRuntime>,
    endpoint: TurnEndpoint,
    model: Option<String>,
    incarnation: RunIncarnationId,
    state_budget: crate::streaming::budget::StreamingResourceBudget,
}

impl std::fmt::Debug for ScheduledRequestActionSinkFactory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScheduledRequestActionSinkFactory")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ScheduledRequestActionSinkFactory {
    /// Bind the factory to one prepared endpoint and scheduled issuer.
    #[must_use]
    pub const fn new(
        runtime: Rc<ScheduledRuntime>,
        endpoint: TurnEndpoint,
        model: Option<String>,
        incarnation: RunIncarnationId,
        state_budget: crate::streaming::budget::StreamingResourceBudget,
    ) -> Self {
        Self {
            runtime,
            endpoint,
            model,
            incarnation,
            state_budget,
        }
    }
}

// The factory holds worker-local `Rc` handles, so it is neither `Send` nor
// `Sync` and is constructed on the worker that will run it rather than being
// registered globally.
impl ScheduledRequestActionSinkFactory {
    /// Strictly validate one authored binding for this sink.
    ///
    /// Mirrors [`StreamingActionSinkFactory::validate_binding`] without the
    /// `Send + Sync` bound that trait requires.
    pub fn validate_binding(
        &self,
        authored: &RawValue,
        action: &DatasetActionSchema,
    ) -> Result<ValidatedScheduledRequestConfig, ActionExecutionError> {
        let authored: AuthoredScheduledRequestConfig = serde_json::from_str(authored.get())
            .map_err(|_| ActionExecutionError::action(ActionFailureCode::MissingBinding))?;
        // Exactly one accepted schema: an unknown one is refused before
        // preparation, not discovered at submit time.
        if action != &canonical_action_schema(DatasetActionKind::Request) {
            return Err(ActionExecutionError::action(
                ActionFailureCode::MissingBinding,
            ));
        }
        if authored.max_active_actions == 0 || authored.max_update_bytes == 0 {
            return Err(ActionExecutionError::state_budget(
                crate::streaming::unit::StateBudgetFailureCode::PermanentError,
            ));
        }
        Ok(ValidatedScheduledRequestConfig {
            schema: action.clone(),
            max_active_actions: authored.max_active_actions,
            max_update_bytes: authored.max_update_bytes,
            is_streaming: authored.is_streaming,
        })
    }

    /// Prepare the split submitter, driver, and independent control handle.
    pub fn prepare(
        &self,
        config: ValidatedScheduledRequestConfig,
        context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError> {
        let shared = Rc::new(ScheduledRequestShared::new(
            context.binding_semantic_digest,
            self.incarnation,
            config.max_active_actions,
        ));
        Ok(PreparedStreamingActionBinding {
            submitter: Box::new(ScheduledRequestSubmitter {
                schema: config.schema,
                shared: Rc::clone(&shared),
                runtime: Rc::clone(&self.runtime),
                endpoint: self.endpoint.clone(),
                model: self.model.clone(),
                is_streaming: config.is_streaming,
                next_handle: 0,
                receivers: Vec::new(),
            }),
            driver: Box::new(ScheduledRequestDriver {
                shared: Rc::clone(&shared),
                budget: self.state_budget.clone(),
                initialization: ParticipantInitialization::default(),
            }),
            control: StreamingActionDriverControl::new(ScheduledRequestControl { shared }),
        })
    }
}

/// The unused import guard: the descriptor trait is the registry contract this
/// binding conforms to even though its handles are worker-local.
const _: fn() = || {
    fn assert_descriptor(descriptor: &StreamingActionSinkDescriptor) -> &'static str {
        descriptor.id
    }
    let _ = assert_descriptor(&SCHEDULED_REQUEST_ACTION_SINK);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_accepts_exactly_the_request_schema() {
        assert_eq!(SCHEDULED_REQUEST_ACTION_SINK.accepted_schemas.len(), 1);
        assert_eq!(
            SCHEDULED_REQUEST_ACTION_SINK.accepted_schemas[0],
            canonical_action_schema(DatasetActionKind::Request).as_str()
        );
    }

    #[test]
    fn endpoint_retries_are_refused_without_a_duplicate_proof() {
        assert_eq!(
            SCHEDULED_REQUEST_ACTION_SINK.endpoint_retry_safety,
            EndpointRetrySafety::Unproven
        );
    }

    #[test]
    fn authored_defaults_bound_admission_and_retained_reply_bytes() {
        let authored = AuthoredScheduledRequestConfig::default();
        assert!(authored.max_active_actions > 0);
        assert!(authored.max_update_bytes > 0);
    }
}
