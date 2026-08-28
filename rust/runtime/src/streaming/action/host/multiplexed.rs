// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run-scoped multiplexing of prepared streaming action bindings.
//!
//! This module is a descendant of the private action host, so it inherits the
//! authority to mint [`CheckedActionTerminalMembership`]. Nothing outside the
//! host subtree can construct that proof, which is why the host — and not the
//! pipeline, the reliability owner, or an adapter — is the single place where
//! an action becomes terminal.
//!
//! Two invariants drive the shape of the types here:
//!
//! - An accepted action has exactly one terminal receipt. Duplicate terminals,
//!   out-of-order event ordinals, and terminals for unknown actions are typed
//!   refusals, not tolerated noise.
//! - Every retained item is budget-charged. The admission lease and the
//!   terminal-receipt lease are inseparable from the state they cover, so a
//!   consumer cannot take receipt bytes and leave the charge behind.

use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
};

use super::CheckedActionTerminalMembership;
use crate::streaming::{
    action::{
        ActionCancelReceipt, ActionDrainReceipt, ActionExecutionEvent, ActionExecutionError,
        ActionFailureCode, ActionTerminalDisposition, ActionTerminalMembershipOutcomeView,
        ActionTerminalReceipt, CheckedActionTerminalMembershipView, DatasetActionSchema,
        OrderedDatasetAction, PreparedStreamingActionBinding, StreamingActionDriver,
        StreamingActionDriverControl, StreamingActionSubmitter, SubmittedAction,
    },
    budget::{BudgetError, BudgetLease, StreamingResourceBudget, ordered_map_entry_bytes},
    checkpoint::StreamRunIdentity,
    identity::{ContentDigest, GlobalSequence, StableActionId},
    unit::{DatasetActionKind, DatasetActionV1, ExecutableDatasetAction, StateBudgetFailureCode},
};

/// Stable action schema emitted for one generation-one action kind.
///
/// The session program emits closed [`DatasetActionV1`] payloads, so the set of
/// schemas a run can emit is derivable rather than authored. Binding validation
/// compares prepared bindings against exactly this derived set.
#[must_use]
pub fn canonical_action_schema(kind: DatasetActionKind) -> DatasetActionSchema {
    DatasetActionSchema::new(match kind {
        DatasetActionKind::Request => "aiperf.action.request.v1",
        DatasetActionKind::GraphNode => "aiperf.action.graph_node.v1",
        DatasetActionKind::SessionTerminal => "aiperf.action.session_terminal.v1",
    })
}

/// Return the closed action kind carried by one executable payload.
#[must_use]
pub const fn action_kind(payload: &DatasetActionV1) -> DatasetActionKind {
    match payload {
        DatasetActionV1::Request(_) => DatasetActionKind::Request,
        DatasetActionV1::GraphNode(_) => DatasetActionKind::GraphNode,
        DatasetActionV1::SessionTerminal(_) => DatasetActionKind::SessionTerminal,
    }
}

/// Translate a budget refusal into the action plane's typed failure.
fn budget_failure(error: BudgetError) -> ActionExecutionError {
    match error {
        BudgetError::CapacityUnavailable => {
            ActionExecutionError::state_budget(StateBudgetFailureCode::ItemCapacity)
        }
        BudgetError::RequestExceedsCapacity | BudgetError::PermitCountTooLarge => {
            ActionExecutionError::state_budget(StateBudgetFailureCode::ByteCapacity)
        }
        _ => ActionExecutionError::state_budget(StateBudgetFailureCode::PermanentError),
    }
}

fn digest_of(domain: &[u8], fields: &[&[u8]]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

const fn disposition_tag(disposition: ActionTerminalDisposition) -> u8 {
    match disposition {
        ActionTerminalDisposition::Completed => 0,
        ActionTerminalDisposition::Failed => 1,
        ActionTerminalDisposition::Cancelled => 2,
        ActionTerminalDisposition::Dropped => 3,
    }
}

/// Move-only terminal receipt inseparable from its retained budget charge.
///
/// The results plane receives the whole wrapper. There is deliberately no
/// accessor that yields the receipt while dropping the lease: releasing the
/// charge is what `into_parts` makes explicit and auditable.
#[derive(Debug)]
pub struct BudgetOwnedActionTerminalReceipt {
    receipt: ActionTerminalReceipt,
    lease: BudgetLease,
}

impl BudgetOwnedActionTerminalReceipt {
    /// Bind one terminal receipt to the exact charge that retains it.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::BudgetInvariant`] unless the lease charges
    /// exactly one item and at least the receipt's retained size.
    pub fn new(
        receipt: ActionTerminalReceipt,
        lease: BudgetLease,
    ) -> Result<Self, ActionExecutionError> {
        if lease.charged_items() != 1
            || lease.charged_bytes() < std::mem::size_of::<ActionTerminalReceipt>()
        {
            return Err(ActionExecutionError::action(
                ActionFailureCode::BudgetInvariant,
            ));
        }
        Ok(Self { receipt, lease })
    }

    /// Borrow the terminal receipt without releasing its charge.
    #[must_use]
    pub const fn receipt(&self) -> &ActionTerminalReceipt {
        &self.receipt
    }

    /// Return the exact retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    /// Move the receipt and its inseparable charge together.
    #[must_use]
    pub fn into_parts(self) -> (ActionTerminalReceipt, BudgetLease) {
        (self.receipt, self.lease)
    }
}

/// One in-flight action and every charge retained on its behalf.
#[derive(Debug)]
pub struct ActiveExecution {
    submitted: SubmittedAction,
    sequence: GlobalSequence,
    last_event_ordinal: u64,
    has_observed_event: bool,
    terminal_receipt: Option<BudgetOwnedActionTerminalReceipt>,
    lease: BudgetLease,
}

impl ActiveExecution {
    /// Borrow the compact submission receipt retained for this action.
    #[must_use]
    pub const fn submitted(&self) -> &SubmittedAction {
        &self.submitted
    }

    /// Return the dense host-assigned global sequence.
    #[must_use]
    pub const fn sequence(&self) -> GlobalSequence {
        self.sequence
    }

    /// Return the greatest observed event ordinal, when any event was observed.
    #[must_use]
    pub const fn last_event_ordinal(&self) -> Option<u64> {
        if self.has_observed_event {
            Some(self.last_event_ordinal)
        } else {
            None
        }
    }

    /// Return whether a terminal receipt has been recorded.
    #[must_use]
    pub const fn has_terminal_receipt(&self) -> bool {
        self.terminal_receipt.is_some()
    }

    /// Return the exact admission byte charge retained for this action.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    /// Transfer the complete budget-owned terminal wrapper to the results plane.
    ///
    /// The receipt and its charge move together; there is no accessor that can
    /// separate them.
    #[must_use]
    pub fn take_terminal_receipt(&mut self) -> Option<BudgetOwnedActionTerminalReceipt> {
        self.terminal_receipt.take()
    }
}

/// Bounded set of actions the host has accepted and not yet finalized.
#[derive(Debug)]
pub struct ActiveExecutionSet {
    entries: BTreeMap<StableActionId, ActiveExecution>,
    budget: StreamingResourceBudget,
}

impl ActiveExecutionSet {
    /// Construct an empty set bounded by one shared resource budget.
    #[must_use]
    pub const fn new(budget: StreamingResourceBudget) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget,
        }
    }

    /// Return the number of retained in-flight actions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether no action is in flight.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow one retained in-flight action.
    #[must_use]
    pub fn get(&self, action_id: StableActionId) -> Option<&ActiveExecution> {
        self.entries.get(&action_id)
    }

    /// Mutably borrow one retained in-flight action.
    pub fn get_mut(&mut self, action_id: StableActionId) -> Option<&mut ActiveExecution> {
        self.entries.get_mut(&action_id)
    }

    /// Charge and retain one accepted action.
    ///
    /// # Errors
    ///
    /// Returns a state-budget failure when no capacity remains, and
    /// [`ActionFailureCode::Dispatch`] when the same identity is admitted twice.
    pub fn admit(
        &mut self,
        action_id: StableActionId,
        sequence: GlobalSequence,
        submitted: SubmittedAction,
    ) -> Result<(), ActionExecutionError> {
        if self.entries.contains_key(&action_id) {
            return Err(ActionExecutionError::action(ActionFailureCode::Dispatch));
        }
        let entry_bytes = ordered_map_entry_bytes::<StableActionId, ActiveExecution>()
            .map_err(budget_failure)?;
        let lease = self
            .budget
            .try_acquire(1, entry_bytes)
            .map_err(budget_failure)?;
        self.entries.insert(
            action_id,
            ActiveExecution {
                submitted,
                sequence,
                last_event_ordinal: 0,
                has_observed_event: false,
                terminal_receipt: None,
                lease,
            },
        );
        Ok(())
    }

    /// Record one strictly increasing non-terminal event ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::UnknownAction`] for an action that was never
    /// admitted, and [`ActionFailureCode::EventOrder`] for a repeated or
    /// out-of-order ordinal, or for any event after the terminal receipt.
    pub fn observe_event(
        &mut self,
        action_id: StableActionId,
        event_ordinal: u64,
    ) -> Result<(), ActionExecutionError> {
        let entry = self
            .entries
            .get_mut(&action_id)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::UnknownAction))?;
        if entry.terminal_receipt.is_some() {
            return Err(ActionExecutionError::action(ActionFailureCode::EventOrder));
        }
        if entry.has_observed_event && event_ordinal <= entry.last_event_ordinal {
            return Err(ActionExecutionError::action(ActionFailureCode::EventOrder));
        }
        entry.has_observed_event = true;
        entry.last_event_ordinal = event_ordinal;
        Ok(())
    }

    /// Record the unique terminal receipt for one admitted action.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::UnknownAction`] when the action was never
    /// admitted, [`ActionFailureCode::DuplicateTerminal`] on a second terminal,
    /// and [`ActionFailureCode::EventOrder`] when the terminal ordinal does not
    /// strictly follow every observed event.
    pub fn finish(
        &mut self,
        action_id: StableActionId,
        receipt: ActionTerminalReceipt,
    ) -> Result<(), ActionExecutionError> {
        let entry_bytes = std::mem::size_of::<ActionTerminalReceipt>();
        let lease = self
            .budget
            .try_acquire(1, entry_bytes)
            .map_err(budget_failure)?;
        let entry = self
            .entries
            .get_mut(&action_id)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::UnknownAction))?;
        if entry.terminal_receipt.is_some() {
            return Err(ActionExecutionError::action(
                ActionFailureCode::DuplicateTerminal,
            ));
        }
        if entry.has_observed_event && receipt.event.event_ordinal <= entry.last_event_ordinal {
            return Err(ActionExecutionError::action(ActionFailureCode::EventOrder));
        }
        entry.has_observed_event = true;
        entry.last_event_ordinal = receipt.event.event_ordinal;
        entry.terminal_receipt = Some(BudgetOwnedActionTerminalReceipt::new(receipt, lease)?);
        Ok(())
    }

    /// Release one finalized action and return its retained state.
    #[must_use]
    pub fn remove(&mut self, action_id: StableActionId) -> Option<ActiveExecution> {
        self.entries.remove(&action_id)
    }

    /// Iterate retained in-flight actions in stable identity order.
    pub fn iter(&self) -> impl Iterator<Item = (&StableActionId, &ActiveExecution)> {
        self.entries.iter()
    }
}

/// One prepared binding retained by the host under its exact schema.
struct HostBinding {
    schema: DatasetActionSchema,
    submitter: Box<dyn StreamingActionSubmitter>,
    driver: Box<dyn StreamingActionDriver>,
    control: StreamingActionDriverControl,
}

/// Exact schema-to-binding map validated before the host is constructed.
#[derive(Default)]
pub struct StreamingActionBindingSet {
    bindings: BTreeMap<DatasetActionSchema, PreparedStreamingActionBinding>,
}

impl StreamingActionBindingSet {
    /// Construct an empty binding set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: BTreeMap::new(),
        }
    }

    /// Insert one prepared binding under the schema its submitter accepts.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::DuplicateBinding`] when a binding for the
    /// same schema is already present.
    pub fn insert(
        &mut self,
        binding: PreparedStreamingActionBinding,
    ) -> Result<(), ActionExecutionError> {
        let schema = binding.submitter.accepted_schema();
        if self.bindings.contains_key(&schema) {
            return Err(ActionExecutionError::action(
                ActionFailureCode::DuplicateBinding,
            ));
        }
        self.bindings.insert(schema, binding);
        Ok(())
    }

    /// Return the schemas covered by this set.
    #[must_use]
    pub fn schemas(&self) -> BTreeSet<DatasetActionSchema> {
        self.bindings.keys().cloned().collect()
    }

    /// Return whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl std::fmt::Debug for StreamingActionBindingSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamingActionBindingSet")
            .field("schemas", &self.schemas())
            .finish()
    }
}

/// Ordered batch of events observed by one `drain_events` pass.
#[derive(Debug, Default)]
pub struct ActionEventBatch {
    events: Vec<ActionExecutionEvent>,
}

impl ActionEventBatch {
    /// Borrow the observed events in emission order.
    #[must_use]
    pub fn events(&self) -> &[ActionExecutionEvent] {
        &self.events
    }

    /// Return the observed event ordinals in emission order.
    #[must_use]
    pub fn ordinals(&self) -> Vec<u64> {
        self.events
            .iter()
            .map(|event| event_identity(event).event_ordinal)
            .collect()
    }

    /// Consume the batch and return its events.
    #[must_use]
    pub fn into_events(self) -> Vec<ActionExecutionEvent> {
        self.events
    }
}

fn event_identity(event: &ActionExecutionEvent) -> &crate::streaming::action::ActionEventIdentity {
    match event {
        ActionExecutionEvent::Admitted(receipt) => &receipt.event,
        ActionExecutionEvent::FirstToken(receipt) => &receipt.event,
        ActionExecutionEvent::SessionUpdate(update) => &update.event,
        ActionExecutionEvent::Terminal(receipt) => &receipt.event,
    }
}

/// Run-scoped multiplexing host owning every prepared action binding.
///
/// The host owns each binding's submitter and driver. Phase control retains the
/// separately borrowable [`StreamingActionDriverControl`], so cancellation never
/// needs the driver's `&mut` borrow and can therefore run while `next_event` is
/// pending.
pub struct StreamingActionHost {
    run: StreamRunIdentity,
    bindings: Vec<HostBinding>,
    schema_index: BTreeMap<DatasetActionSchema, usize>,
    active: ActiveExecutionSet,
    finalized: BTreeMap<StableActionId, ActiveExecution>,
    next_sequence: u64,
    terminal_membership: BTreeMap<StableActionId, CheckedActionTerminalMembership>,
}

impl std::fmt::Debug for StreamingActionHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamingActionHost")
            .field("run", &self.run)
            .field("schemas", &self.schema_index.keys().collect::<Vec<_>>())
            .field("active", &self.active.len())
            .field("next_sequence", &self.next_sequence)
            .finish()
    }
}

impl StreamingActionHost {
    /// Bind exactly one prepared binding per emitted schema.
    ///
    /// The returned control map is what phase control retains: the host keeps
    /// its own clone so it can join the drivers, but neither borrow blocks the
    /// other.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::MissingBinding`] when an emitted schema has
    /// no prepared binding, and [`ActionFailureCode::UnexpectedBinding`] when a
    /// prepared binding covers a schema the run cannot emit.
    pub fn new(
        run: StreamRunIdentity,
        emitted_schemas: &BTreeSet<DatasetActionSchema>,
        set: StreamingActionBindingSet,
        budget: StreamingResourceBudget,
    ) -> Result<
        (
            Self,
            BTreeMap<DatasetActionSchema, StreamingActionDriverControl>,
        ),
        ActionExecutionError,
    > {
        let prepared = set.schemas();
        if emitted_schemas.difference(&prepared).next().is_some() {
            return Err(ActionExecutionError::action(
                ActionFailureCode::MissingBinding,
            ));
        }
        if prepared.difference(emitted_schemas).next().is_some() {
            return Err(ActionExecutionError::action(
                ActionFailureCode::UnexpectedBinding,
            ));
        }

        let mut bindings = Vec::with_capacity(set.bindings.len());
        let mut schema_index = BTreeMap::new();
        let mut controls = BTreeMap::new();
        for (schema, binding) in set.bindings {
            if binding.submitter.accepted_schema() != schema {
                return Err(ActionExecutionError::action(
                    ActionFailureCode::MissingBinding,
                ));
            }
            schema_index.insert(schema.clone(), bindings.len());
            controls.insert(schema.clone(), binding.control.clone());
            bindings.push(HostBinding {
                schema,
                submitter: binding.submitter,
                driver: binding.driver,
                control: binding.control,
            });
        }

        Ok((
            Self {
                run,
                bindings,
                schema_index,
                active: ActiveExecutionSet::new(budget),
                finalized: BTreeMap::new(),
                next_sequence: 0,
                terminal_membership: BTreeMap::new(),
            },
            controls,
        ))
    }

    /// Borrow the logical run owning every binding.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the bounded in-flight action set.
    #[must_use]
    pub const fn active(&self) -> &ActiveExecutionSet {
        &self.active
    }

    /// Mutably borrow the bounded in-flight action set.
    pub fn active_mut(&mut self) -> &mut ActiveExecutionSet {
        &mut self.active
    }

    /// Take one finalized execution and its inseparable retained charges.
    ///
    /// The results plane calls this to move the budget-owned terminal receipt
    /// out of the host; until it does, the charge stays held.
    #[must_use]
    pub fn take_finalized(&mut self, action_id: StableActionId) -> Option<ActiveExecution> {
        self.finalized.remove(&action_id)
    }

    /// Return the number of finalized executions the host still retains.
    #[must_use]
    pub fn finalized_len(&self) -> usize {
        self.finalized.len()
    }

    /// Submit one causally ready action and assign its dense global sequence.
    ///
    /// The sequence is assigned only after the action has passed schema routing
    /// and budget admission, so a refused action never consumes a sequence and
    /// the assigned order stays dense.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::MissingBinding`] when the action's schema is
    /// unbound, a state-budget failure when the active set is full, and the
    /// binding's own error when submission fails.
    pub async fn submit(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<GlobalSequence, ActionExecutionError> {
        let schema = canonical_action_schema(action_kind(action.payload()));
        let index = *self
            .schema_index
            .get(&schema)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::MissingBinding))?;
        let action_id = action.action_id();
        let sequence = GlobalSequence::new(self.next_sequence);

        let submitted = self.bindings[index]
            .submitter
            .submit(OrderedDatasetAction::new(sequence, action))
            .await?;
        self.active.admit(action_id, sequence, submitted)?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::Dispatch))?;
        Ok(sequence)
    }

    /// Wait for the next event from any binding and apply it to host state.
    ///
    /// # Errors
    ///
    /// Returns [`ActionFailureCode::MissingBinding`] when no binding exists, the
    /// driver's own error, or an ordering refusal from the active set.
    pub async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError> {
        if self.bindings.is_empty() {
            return Err(ActionExecutionError::action(
                ActionFailureCode::MissingBinding,
            ));
        }
        let event = {
            let mut pending: Vec<
                Pin<
                    Box<
                        dyn Future<Output = Result<ActionExecutionEvent, ActionExecutionError>> + '_,
                    >,
                >,
            > = Vec::with_capacity(self.bindings.len());
            for binding in &mut self.bindings {
                pending.push(Box::pin(binding.driver.next_event()));
            }
            // Every driver's `next_event` must be cancel-safe: `select_all`
            // drops the losing futures, and a driver that consumed an event
            // before yielding would lose it here.
            let (result, _index, _rest) = futures::future::select_all(pending).await;
            result?
        };
        self.apply_event(&event)?;
        Ok(event)
    }

    /// Pump every binding until no action remains in flight.
    ///
    /// # Errors
    ///
    /// Returns the first driver or ordering failure observed.
    pub async fn drain_events(&mut self) -> Result<ActionEventBatch, ActionExecutionError> {
        let mut batch = ActionEventBatch::default();
        while !self.active.is_empty() {
            let event = self.next_event().await?;
            batch.events.push(event);
        }
        Ok(batch)
    }

    /// Return the number of finalized terminal memberships for one action.
    ///
    /// This is `1` for exactly one accepted terminal and `0` otherwise; the
    /// active set refuses a second terminal before it can be counted.
    #[must_use]
    pub fn terminal_membership(&self, action_id: StableActionId) -> usize {
        usize::from(self.terminal_membership.contains_key(&action_id))
    }

    /// Borrow the sealed checked terminal membership for one finalized action.
    #[must_use]
    pub fn checked_terminal_membership(
        &self,
        action_id: StableActionId,
    ) -> Option<&dyn CheckedActionTerminalMembershipView> {
        self.terminal_membership
            .get(&action_id)
            .map(|membership| membership as &dyn CheckedActionTerminalMembershipView)
    }

    /// Fence issue, cancel in-flight work, and join every driver.
    ///
    /// # Errors
    ///
    /// Returns the first control or driver failure observed.
    pub async fn cancel_and_join(
        &mut self,
    ) -> Result<(Vec<ActionCancelReceipt>, Vec<ActionDrainReceipt>), ActionExecutionError> {
        let mut cancelled = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            binding.control.stop_issuing();
            binding.control.cancel_pending();
            cancelled.push(binding.control.cancel_inflight().await?);
        }
        let mut drained = Vec::with_capacity(self.bindings.len());
        for binding in &mut self.bindings {
            drained.push(binding.driver.drain().await?);
        }
        Ok((cancelled, drained))
    }

    /// Borrow the schema bound to one binding index, for diagnostics.
    #[must_use]
    pub fn bound_schemas(&self) -> Vec<&DatasetActionSchema> {
        self.bindings.iter().map(|binding| &binding.schema).collect()
    }

    fn apply_event(&mut self, event: &ActionExecutionEvent) -> Result<(), ActionExecutionError> {
        match event {
            ActionExecutionEvent::Terminal(receipt) => {
                let action_id = receipt.event.action_id;
                self.active.finish(action_id, receipt.clone())?;
                let membership = self.mint_membership(action_id, receipt)?;
                self.terminal_membership.insert(action_id, membership);
                if let Some(entry) = self.active.remove(action_id) {
                    self.finalized.insert(action_id, entry);
                }
                Ok(())
            }
            other => {
                let identity = event_identity(other);
                self.active
                    .observe_event(identity.action_id, identity.event_ordinal)
            }
        }
    }

    /// Mint the sealed terminal membership for one finalized action.
    ///
    /// Only this host subtree can construct the proof, which is why the failure
    /// issue identity is derived here from the run, action, sequence, and
    /// disposition rather than accepted from a caller.
    fn mint_membership(
        &self,
        action_id: StableActionId,
        receipt: &ActionTerminalReceipt,
    ) -> Result<CheckedActionTerminalMembership, ActionExecutionError> {
        let sequence = self
            .active
            .get(action_id)
            .map(ActiveExecution::sequence)
            .ok_or_else(|| ActionExecutionError::action(ActionFailureCode::UnknownAction))?;
        let disposition = [disposition_tag(receipt.disposition)];
        let outcome = match receipt.disposition {
            ActionTerminalDisposition::Completed => ActionTerminalMembershipOutcomeView::Succeeded,
            _ => ActionTerminalMembershipOutcomeView::Failed {
                issue_id: digest_of(
                    b"aiperf.stream.action.issue.v1",
                    &[
                        self.run.logical_replay_run().as_bytes(),
                        action_id.as_bytes(),
                        &sequence.get().to_le_bytes(),
                        &disposition,
                    ],
                ),
            },
        };
        let membership_digest = digest_of(
            b"aiperf.stream.action.membership.v1",
            &[
                self.run.logical_replay_run().as_bytes(),
                action_id.as_bytes(),
                &sequence.get().to_le_bytes(),
                &disposition,
                &receipt.event.event_ordinal.to_le_bytes(),
            ],
        );
        Ok(CheckedActionTerminalMembership::new(
            self.run,
            action_id,
            sequence,
            outcome,
            membership_digest,
        ))
    }
}
