// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming action binding, submission, event driving, and control contracts.

use std::{any::Any, fmt, rc::Rc};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{
    budget::BudgetLease,
    checkpoint::{StreamRunIdentity, StreamingCheckpointParticipant},
    identity::{
        ActionAttemptId, ContentDigest, GlobalSequence, SessionOwnershipEpoch, StableActionId,
        StableSessionKey,
    },
    unit::ExecutableDatasetAction,
};

pub use super::failure::{ActionExecutionError, ActionFailureCode, PlacementFailureCode};

mod reliability_view_seal {
    pub trait CheckedActionFailureTerminalEvidenceView {}
    pub trait CheckedActionTerminalMembershipView {}
    pub trait FrozenActionInventoryView {}
}

/// Borrowed checked evidence that one failed action attempt reached terminal.
///
/// Implementations are sealed to action-host child modules. An adapter cannot
/// forge terminal evidence:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::{
/// #     action::CheckedActionFailureTerminalEvidenceView,
/// #     checkpoint::StreamRunIdentity,
/// #     identity::{ContentDigest, GlobalSequence, StableActionId},
/// # };
/// struct Forged;
/// impl CheckedActionFailureTerminalEvidenceView for Forged {
///     fn run(&self) -> &StreamRunIdentity { unimplemented!() }
///     fn action_id(&self) -> StableActionId { unimplemented!() }
///     fn sequence(&self) -> GlobalSequence { unimplemented!() }
///     fn terminal_evidence_digest(&self) -> ContentDigest { unimplemented!() }
/// }
/// ```
pub trait CheckedActionFailureTerminalEvidenceView:
    reliability_view_seal::CheckedActionFailureTerminalEvidenceView
{
    /// Borrow the logical run owning the action.
    fn run(&self) -> &StreamRunIdentity;

    /// Return the stable logical action identity.
    fn action_id(&self) -> StableActionId;

    /// Return the dense host-assigned action sequence.
    fn sequence(&self) -> GlobalSequence;

    /// Return the digest of the checked terminal attempt evidence.
    fn terminal_evidence_digest(&self) -> ContentDigest;
}

/// Checked terminal membership observed after the action owner has finalized it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionTerminalMembershipOutcomeView {
    /// The action completed successfully.
    Succeeded,
    /// The action failed under one reporter-retained issue identity.
    Failed {
        /// Deterministic issue identity bound to terminal membership.
        issue_id: ContentDigest,
    },
}

/// Borrowed checked membership for one finalized action.
///
/// This trait is sealed to action-host child modules.
pub trait CheckedActionTerminalMembershipView:
    reliability_view_seal::CheckedActionTerminalMembershipView
{
    /// Borrow the logical run owning the terminal membership.
    fn run(&self) -> &StreamRunIdentity;

    /// Return the stable logical action identity.
    fn action_id(&self) -> StableActionId;

    /// Return the dense host-assigned action sequence.
    fn sequence(&self) -> GlobalSequence;

    /// Return the checked success or reporter-bound failure outcome.
    fn outcome(&self) -> ActionTerminalMembershipOutcomeView;

    /// Return the digest binding the complete terminal membership.
    fn membership_digest(&self) -> ContentDigest;
}

/// Borrowed immutable inventory used to prove dense action gap closure.
///
/// This trait is sealed to action-host child modules.
pub trait FrozenActionInventoryView: reliability_view_seal::FrozenActionInventoryView {
    /// Borrow the logical run owning the frozen inventory.
    fn run(&self) -> &StreamRunIdentity;

    /// Return the greatest action sequence covered by the inventory.
    fn through(&self) -> GlobalSequence;

    /// Return the digest binding the frozen inventory membership.
    fn membership_root(&self) -> ContentDigest;

    /// Return whether the exact terminal membership is present.
    fn contains_terminal(&self, sequence: GlobalSequence, membership_digest: ContentDigest)
    -> bool;
}

/// Stable action schema selected during compatibility validation.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DatasetActionSchema(String);

impl DatasetActionSchema {
    /// Construct an action schema from validated stable text.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the stable schema text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable registry metadata for one streaming action sink implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamingActionSinkDescriptor {
    /// Stable registry identifier.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Action schemas accepted by this implementation.
    pub accepted_schemas: &'static [&'static str],
    /// Transport implementations accepted by this binding.
    pub transport_ids: &'static [&'static str],
    /// Endpoint families accepted by this binding.
    pub endpoint_kinds: &'static [&'static str],
    /// Result retention required while actions execute.
    pub retention: ActionResultRetention,
    /// Placement supported by the binding.
    pub placement: ActionPlacement,
    /// Whether the binding can run under a virtual clock.
    pub supports_virtual_clock: bool,
}

/// Action result retention behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionResultRetention {
    /// Terminal results can be emitted under bounded streaming budgets.
    StreamingTerminal,
    /// Complete result history must remain resident.
    ResidentTotal,
}

/// Action binding placement behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPlacement {
    /// Binding state is local to one worker.
    WorkerLocal,
    /// Binding supports fenced routing across cells.
    RoutedAcrossCells,
}

/// Type-erased, strictly validated action-sink configuration.
pub trait ValidatedStreamingActionSinkConfig: fmt::Debug + Send + Sync {
    /// Borrow the concrete startup-only value.
    fn as_any(&self) -> &dyn Any;

    /// Consume the concrete startup-only value.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedStreamingActionSinkConfig for T
where
    T: Any + fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Host-owned action-sink preparation context.
#[derive(Clone, Debug)]
pub struct StreamingActionSinkPrepareContext {
    /// Semantic digest of the selected action binding plan.
    pub binding_semantic_digest: ContentDigest,
}

/// Startup action binding validation and preparation contract.
pub trait StreamingActionSinkFactory: fmt::Debug + Send + Sync {
    /// Describe the exact compiled action-sink implementation.
    fn descriptor(&self) -> &'static StreamingActionSinkDescriptor;

    /// Strictly validate one action, transport, and endpoint binding.
    fn validate_binding(
        &self,
        authored: &RawValue,
        action: &DatasetActionSchema,
        transport: &crate::engine::registry::TransportDescriptor,
        endpoint: &crate::endpoints::EndpointDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingActionSinkConfig>, ActionExecutionError>;

    /// Prepare one split submitter, driver, and independent control handle.
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingActionSinkConfig>,
        context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError>;
}

/// One action with host-assigned stable global order.
#[derive(Debug)]
pub struct OrderedDatasetAction {
    sequence: GlobalSequence,
    action: ExecutableDatasetAction,
}

impl OrderedDatasetAction {
    /// Bind one causal-ready action to a stable global sequence.
    #[must_use]
    pub const fn new(sequence: GlobalSequence, action: ExecutableDatasetAction) -> Self {
        Self { sequence, action }
    }

    /// Return the stable global sequence.
    #[must_use]
    pub const fn sequence(&self) -> GlobalSequence {
        self.sequence
    }

    /// Borrow the move-only executable action.
    #[must_use]
    pub const fn action(&self) -> &ExecutableDatasetAction {
        &self.action
    }

    /// Consume the ordered wrapper and retain action lease ownership.
    #[must_use]
    pub fn into_action(self) -> ExecutableDatasetAction {
        self.action
    }
}

/// Stable slab-local handle for one submitted action.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActionHandleId(u64);

impl ActionHandleId {
    /// Construct a handle from one bounded slab coordinate.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the slab coordinate.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Cloneable per-action cancellation control.
#[derive(Clone, Debug)]
pub struct ActionExecutionControl {
    sender: tokio::sync::watch::Sender<bool>,
}

impl ActionExecutionControl {
    /// Request cancellation and wake the binding that owns this action.
    pub fn cancel(&self) {
        self.sender.send_replace(true);
    }

    /// Return whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        *self.sender.borrow()
    }
}

/// Receiver retained by an action binding for one cancellation signal.
#[derive(Debug)]
pub struct ActionExecutionCancelReceiver {
    receiver: tokio::sync::watch::Receiver<bool>,
}

impl ActionExecutionCancelReceiver {
    /// Return whether cancellation has already been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    /// Wait until cancellation is requested or the control is dropped.
    pub async fn cancelled(&mut self) {
        while !*self.receiver.borrow_and_update() {
            if self.receiver.changed().await.is_err() {
                return;
            }
        }
    }
}

/// Construct a cloneable per-action cancellation pair.
#[must_use]
pub fn action_execution_control() -> (ActionExecutionControl, ActionExecutionCancelReceiver) {
    let (sender, receiver) = tokio::sync::watch::channel(false);
    (
        ActionExecutionControl { sender },
        ActionExecutionCancelReceiver { receiver },
    )
}

/// Compact submission receipt retained by the host active-execution set.
#[derive(Clone, Debug)]
pub struct SubmittedAction {
    /// Stable slab-local handle.
    pub handle_id: ActionHandleId,
    /// Separately cloneable per-action control.
    pub control: ActionExecutionControl,
}

/// Submission half of one prepared action binding.
#[async_trait(?Send)]
pub trait StreamingActionSubmitter {
    /// Return the exact action schema accepted by this binding.
    fn accepted_schema(&self) -> DatasetActionSchema;

    /// Submit one move-only globally ordered action.
    async fn submit(
        &mut self,
        action: OrderedDatasetAction,
    ) -> Result<SubmittedAction, ActionExecutionError>;
}

/// Multiplexed event-stream half of one prepared action binding.
#[async_trait(?Send)]
pub trait StreamingActionDriver: StreamingCheckpointParticipant {
    /// Wait for the next event from any active slab entry.
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError>;

    /// Join every accepted action and return a terminal coverage receipt.
    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError>;
}

/// Implementation behind a separately cloneable driver control handle.
#[async_trait(?Send)]
pub trait StreamingActionDriverControlOps {
    /// Synchronously fence new issue.
    fn stop_issuing(&self);

    /// Synchronously cancel pending, unissued actions.
    fn cancel_pending(&self);

    /// Cancel and join in-flight action work.
    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError>;
}

/// Cheaply cloneable control for a separately borrowed mutable driver.
#[derive(Clone)]
pub struct StreamingActionDriverControl {
    inner: Rc<dyn StreamingActionDriverControlOps>,
}

impl StreamingActionDriverControl {
    /// Erase one worker-local driver control implementation.
    #[must_use]
    pub fn new<T>(control: T) -> Self
    where
        T: StreamingActionDriverControlOps + 'static,
    {
        Self {
            inner: Rc::new(control),
        }
    }

    /// Synchronously fence new issue and wake a pending driver future.
    pub fn stop_issuing(&self) {
        self.inner.stop_issuing();
    }

    /// Synchronously cancel pending work and wake a pending driver future.
    pub fn cancel_pending(&self) {
        self.inner.cancel_pending();
    }

    /// Cancel and join in-flight work without borrowing the driver.
    pub async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError> {
        self.inner.cancel_inflight().await
    }
}

impl fmt::Debug for StreamingActionDriverControl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingActionDriverControl")
            .finish_non_exhaustive()
    }
}

/// Split prepared action binding with no aliasing of its mutable driver.
pub struct PreparedStreamingActionBinding {
    /// Submission half used by the pipeline.
    pub submitter: Box<dyn StreamingActionSubmitter>,
    /// Sole mutable multiplexed event stream.
    pub driver: Box<dyn StreamingActionDriver>,
    /// Concrete cloneable control that can wake the driver.
    pub control: StreamingActionDriverControl,
}

/// Common identity and ordering fields carried by every action event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionEventIdentity {
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Incarnation-local physical attempt identity.
    pub attempt_id: ActionAttemptId,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Strictly increasing event ordinal within the attempt.
    pub event_ordinal: u64,
}

/// Receipt proving action admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionAdmissionReceipt {
    /// Event identity and order.
    pub event: ActionEventIdentity,
}

/// Receipt identifying the first observed output token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionFirstTokenReceipt {
    /// Event identity and order.
    pub event: ActionEventIdentity,
}

/// Endpoint-derived state update fed only to the session coordinator.
#[derive(Debug)]
pub struct EndpointSessionUpdate {
    /// Event identity and order.
    pub event: ActionEventIdentity,
    /// Endpoint-neutral normalized update bytes with move-only capacity ownership.
    pub payload: BudgetedActionUpdate,
}

/// Move-only normalized endpoint update and its exact retained capacity.
#[derive(Debug)]
pub struct BudgetedActionUpdate {
    bytes: Bytes,
    lease: BudgetLease,
}

impl BudgetedActionUpdate {
    /// Bind compact normalized bytes to an exact one-item byte charge.
    pub fn new(bytes: Bytes, lease: BudgetLease) -> Result<Self, ActionExecutionError> {
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes.len() {
            return Err(ActionExecutionError::action(
                ActionFailureCode::BudgetInvariant,
            ));
        }
        let bytes = Bytes::from(bytes.as_ref().to_vec().into_boxed_slice());
        Ok(Self { bytes, lease })
    }

    /// Borrow the exact normalized endpoint update bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the exact retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Stable terminal disposition for one action attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionTerminalDisposition {
    /// Action completed successfully.
    Completed,
    /// Action failed after admission.
    Failed,
    /// Action was cancelled.
    Cancelled,
    /// Action was dropped before endpoint issue.
    Dropped,
}

/// Unique terminal receipt for one action attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionTerminalReceipt {
    /// Event identity and order.
    pub event: ActionEventIdentity,
    /// Terminal action disposition.
    pub disposition: ActionTerminalDisposition,
}

/// Multiplexed event emitted by a prepared action driver.
#[derive(Debug)]
pub enum ActionExecutionEvent {
    /// Action admission became authoritative.
    Admitted(ActionAdmissionReceipt),
    /// First output token was observed.
    FirstToken(ActionFirstTokenReceipt),
    /// Endpoint-derived session update was observed.
    SessionUpdate(EndpointSessionUpdate),
    /// Unique final event for the attempt.
    Terminal(ActionTerminalReceipt),
}

/// Receipt proving all accepted work has reached a terminal state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionDrainReceipt {
    /// Number of submitted actions joined by the driver.
    pub submitted: u64,
    /// Number of terminal receipts emitted by the driver.
    pub terminal: u64,
    /// Digest binding the exact joined action set.
    pub digest: ContentDigest,
}

/// Receipt proving explicit in-flight cancellation completed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionCancelReceipt {
    /// Number of actions cancelled while in flight.
    pub cancelled: u64,
    /// Digest binding the exact cancelled action set.
    pub digest: ContentDigest,
}

/// Session routing identity attached to a prepared binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActionBindingRoute {
    /// Stable session owning the route.
    pub session_key: StableSessionKey,
    /// Current fenced route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
}
