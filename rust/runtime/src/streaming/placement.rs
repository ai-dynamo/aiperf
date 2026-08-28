// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming placement contracts and the bounded worker-local implementation.
//!
//! Placement is split into five separately owned halves so a capacity wait never
//! holds a borrow of the route map: `admission` waits, `policy` decides,
//! `submitter` prepares, `driver` reports, and `control` fences. The local
//! implementation satisfies the same split as a cellular one would, without a
//! transport hop; it declares no persistent route map and therefore returns no
//! route charge.
//!
//! Routing decisions are taken against the causally ready
//! [`ExecutableDatasetAction`] rather than a globally ordered one. The dense
//! [`GlobalSequence`] belongs to the action host, which assigns it only as part
//! of submitting the action, so a prepared placement binds its sequence after
//! the fact through [`StreamingPlacementSubmitter::bind_sequence`]. Placement
//! therefore never needs to mint an order of its own.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, VecDeque},
    fmt,
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::{
    action::ActionExecutionEvent,
    budget::{BudgetLease, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, PreparedParticipantState,
        StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::PlacementFailureCode,
    identity::{
        ContentDigest, GlobalSequence, SessionCausalFrontier, SessionOwnershipEpoch,
        StableActionId, StableSessionKey,
    },
    unit::{ExecutableDatasetAction, StateBudgetFailureCode},
};

/// Placement authority or capacity failure.
///
/// [`PlacementFailureCode`] is the stable routing vocabulary owned by the
/// streaming failure module. This wrapper mirrors the shape of
/// `SessionCoordinatorError` so a capacity refusal is never reported as a
/// routing refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementError {
    /// Placement authority refused the action.
    Placement(PlacementFailureCode),
    /// Placement state exceeded an explicit capacity.
    StateBudget(StateBudgetFailureCode),
}

impl PlacementError {
    /// Construct a routing failure.
    #[must_use]
    pub const fn placement(code: PlacementFailureCode) -> Self {
        Self::Placement(code)
    }

    /// Construct a placement state-budget failure.
    #[must_use]
    pub const fn state_budget(code: StateBudgetFailureCode) -> Self {
        Self::StateBudget(code)
    }
}

impl fmt::Display for PlacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Placement(code) => write!(formatter, "streaming placement refused: {code:?}"),
            Self::StateBudget(code) => write!(formatter, "streaming placement capacity: {code:?}"),
        }
    }
}

impl std::error::Error for PlacementError {}

/// Stable slab-local handle for one prepared placement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlacementHandleId(u64);

impl PlacementHandleId {
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

/// Route capacity one action would newly occupy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementRouteCharge {
    /// Stable session the route would serve.
    pub session: StableSessionKey,
    /// Route-map entries the charge covers.
    pub items: usize,
    /// Retained route-map bytes the charge covers.
    pub bytes: usize,
}

/// Move-only proof that one route's exact capacity is held.
///
/// Dropping the reservation without
/// [`StreamingPlacementPolicy::install_route_reservation`] returns the whole
/// charge, which is how a cancelled admission installs no route.
#[derive(Debug)]
pub struct PlacementRouteReservation {
    session: StableSessionKey,
    lease: BudgetLease,
}

impl PlacementRouteReservation {
    /// Bind one proven capacity lease to the session it was proven for.
    #[must_use]
    pub const fn new(session: StableSessionKey, lease: BudgetLease) -> Self {
        Self { session, lease }
    }

    /// Return the stable session the reservation was proven for.
    #[must_use]
    pub const fn session(&self) -> StableSessionKey {
        self.session
    }

    /// Return the exact retained item charge.
    #[must_use]
    pub fn charged_items(&self) -> usize {
        self.lease.charged_items()
    }

    /// Return the exact retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    /// Move the reservation's exact capacity out for installation.
    #[must_use]
    pub fn into_lease(self) -> BudgetLease {
        self.lease
    }
}

/// Deterministic routing decision for one causally ready action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementDecision {
    /// Stable route identity within the policy.
    pub route_id: u32,
    /// Destination cell, absent for a worker-local placement.
    pub destination_cell: Option<u32>,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
}

/// Prepared placement of one causally ready action.
///
/// `global_sequence` is absent until the action host assigns one and the
/// pipeline calls [`StreamingPlacementSubmitter::bind_sequence`]. Placement
/// never numbers actions itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementHandle {
    /// Stable slab-local handle.
    pub id: PlacementHandleId,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Stable session owning the placed action.
    pub session: StableSessionKey,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Dense host-assigned global order, once the host has assigned it.
    pub global_sequence: Option<GlobalSequence>,
}

/// Receipt proving one placement was prepared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementPreparedReceipt {
    /// Prepared handle.
    pub handle: PlacementHandleId,
    /// Digest binding the exact prepared content.
    pub content_digest: ContentDigest,
}

/// Receipt proving one placement released its slab entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementReleasedReceipt {
    /// Released handle.
    pub handle: PlacementHandleId,
}

/// Receipt describing one placement failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementFailureReceipt {
    /// Handle whose placement failed, when one had been prepared.
    pub handle: Option<PlacementHandleId>,
    /// Stable failure classification.
    pub code: PlacementFailureCode,
}

/// Multiplexed event emitted by a placement driver.
// The action variant carries a move-only budget-owning payload; boxing it would
// add a per-event allocation on the settle path.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum PlacementEvent {
    /// One placement became prepared.
    Prepared(PlacementPreparedReceipt),
    /// One placement released its slab entry.
    Released(PlacementReleasedReceipt),
    /// The only route back into session state.
    Action(ActionExecutionEvent),
    /// One placement failed.
    Failed(PlacementFailureReceipt),
}

/// Synchronous routing authority and checkpoint participant.
pub trait StreamingPlacementPolicy: StreamingCheckpointParticipant {
    /// Return the exact new route capacity this action needs, when any.
    fn route_admission(
        &self,
        action: &ExecutableDatasetAction,
    ) -> Result<Option<PlacementRouteCharge>, PlacementError>;

    /// Install one proven route reservation without waiting.
    fn install_route_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError>;

    /// Decide one action's route synchronously.
    fn place(
        &mut self,
        action: &ExecutableDatasetAction,
    ) -> Result<PlacementDecision, PlacementError>;

    /// Release route capacity at a session's causal terminal.
    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError>;

    /// Return the number of installed routes, for bounded-state assertions.
    fn installed_route_count(&self) -> usize;
}

/// Separately borrowable async capacity owner for deterministic placement.
///
/// This half exists so a pending capacity wait never holds a borrow of the
/// policy: a terminal event arriving while `reserve_route` is pending must
/// still be able to call [`StreamingPlacementPolicy::observe_session_terminal`]
/// and release the capacity the reservation is waiting for.
#[async_trait(?Send)]
pub trait StreamingPlacementAdmission {
    /// Wait for exact route capacity without borrowing the policy.
    async fn reserve_route(
        &mut self,
        charge: PlacementRouteCharge,
    ) -> Result<PlacementRouteReservation, PlacementError>;
}

/// Preparation half of one prepared placement binding.
#[async_trait(?Send)]
pub trait StreamingPlacementSubmitter {
    /// Prepare one placed action for execution.
    async fn prepare(
        &mut self,
        decision: PlacementDecision,
        action: &ExecutableDatasetAction,
    ) -> Result<PlacementHandle, PlacementError>;

    /// Bind the host-assigned dense global order to one prepared placement.
    fn bind_sequence(
        &mut self,
        handle: PlacementHandleId,
        sequence: GlobalSequence,
    ) -> Result<(), PlacementError>;

    /// Release one prepared placement's slab entry.
    async fn release(&mut self, handle: PlacementHandleId) -> Result<(), PlacementError>;

    /// Return the number of retained prepared placements.
    fn prepared_count(&self) -> usize;
}

/// Sole mutable placement event stream and checkpoint participant.
#[async_trait(?Send)]
pub trait StreamingPlacementDriver: StreamingCheckpointParticipant {
    /// Wait for the next placement event.
    async fn next_event(&mut self) -> Result<PlacementEvent, PlacementError>;

    /// Join every prepared placement.
    async fn drain(&mut self) -> Result<(), PlacementError>;
}

/// Cheaply cloneable placement control that can wake a borrowed driver.
#[async_trait(?Send)]
pub trait StreamingPlacementControl {
    /// Synchronously fence new preparation.
    fn stop_preparing(&self);

    /// Synchronously drop pending, unprepared placements.
    fn cancel_pending(&self);

    /// Cancel and join in-flight placements.
    async fn cancel_inflight(&self) -> Result<(), PlacementError>;
}

/// Split prepared placement binding with no aliasing of its mutable driver.
pub struct PreparedStreamingPlacementBinding {
    /// Separately borrowable capacity owner.
    pub admission: Box<dyn StreamingPlacementAdmission>,
    /// Synchronous routing authority.
    pub policy: Box<dyn StreamingPlacementPolicy>,
    /// Preparation half.
    pub submitter: Box<dyn StreamingPlacementSubmitter>,
    /// Sole mutable event stream.
    pub driver: Box<dyn StreamingPlacementDriver>,
    /// Cloneable control surface.
    pub control: Rc<dyn StreamingPlacementControl>,
}

/// State shared by the four halves of one worker-local placement binding.
#[derive(Debug, Default)]
struct LocalPlacementShared {
    events: RefCell<VecDeque<PlacementEvent>>,
    prepared: RefCell<BTreeMap<PlacementHandleId, PlacementHandle>>,
    is_preparing_stopped: Cell<bool>,
    wake: tokio::sync::Notify,
}

/// Bounded worker-local placement policy with no persistent route map.
///
/// Because every action executes on the pipeline's own worker, there is nothing
/// to route to and no per-session route state to retain. `route_admission`
/// therefore returns `Ok(None)` for every action, which expresses the
/// "the local implementation returns no route charge" requirement as behaviour
/// rather than as a comment. Session ownership epochs are still tracked, so a
/// stale epoch is refusable and `observe_session_terminal` fences a route the
/// same way a cellular policy would.
pub struct LocalStreamingPlacement {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    epochs: BTreeMap<StableSessionKey, SessionOwnershipEpoch>,
    installed: BTreeMap<StableSessionKey, BudgetLease>,
    state_budget: StreamingResourceBudget,
}

impl fmt::Debug for LocalStreamingPlacement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalStreamingPlacement")
            .field("participant_id", &self.participant_id)
            .field("sessions", &self.epochs.len())
            .field("installed", &self.installed.len())
            .finish()
    }
}

impl LocalStreamingPlacement {
    /// Construct a worker-local policy under one stable participant identity.
    #[must_use]
    pub fn new(
        participant_id: CheckpointParticipantId,
        run: StreamRunIdentity,
        state_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            participant_id,
            run,
            epochs: BTreeMap::new(),
            installed: BTreeMap::new(),
            state_budget,
        }
    }

    /// Return the current fenced ownership epoch for one session.
    #[must_use]
    pub fn ownership_epoch(&self, session: StableSessionKey) -> SessionOwnershipEpoch {
        self.epochs
            .get(&session)
            .copied()
            .unwrap_or_else(|| SessionOwnershipEpoch::new(0))
    }
}

impl StreamingPlacementPolicy for LocalStreamingPlacement {
    fn route_admission(
        &self,
        _action: &ExecutableDatasetAction,
    ) -> Result<Option<PlacementRouteCharge>, PlacementError> {
        Ok(None)
    }

    fn install_route_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError> {
        let session = reservation.session();
        // A worker-local policy never asks for a charge, so an installed
        // reservation can only come from a caller that fabricated one. Retain it
        // rather than dropping it silently, so the accounting stays exact.
        self.installed.insert(session, reservation.into_lease());
        Ok(())
    }

    fn place(
        &mut self,
        action: &ExecutableDatasetAction,
    ) -> Result<PlacementDecision, PlacementError> {
        let session = action.session_key();
        let ownership_epoch = self.ownership_epoch(session);
        self.epochs.insert(session, ownership_epoch);
        Ok(PlacementDecision {
            route_id: 0,
            destination_cell: None,
            ownership_epoch,
        })
    }

    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        _causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError> {
        let current = self.ownership_epoch(session);
        if ownership_epoch.get() < current.get() {
            return Err(PlacementError::placement(
                PlacementFailureCode::StaleOwnershipEpoch,
            ));
        }
        let next = ownership_epoch.get().checked_add(1).ok_or_else(|| {
            PlacementError::placement(PlacementFailureCode::TargetOverflow)
        })?;
        self.epochs
            .insert(session, SessionOwnershipEpoch::new(next));
        // Dropping the lease here is what returns the exact route capacity and
        // wakes any admission future parked on it.
        self.installed.remove(&session);
        Ok(())
    }

    fn installed_route_count(&self) -> usize {
        self.installed.len()
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for LocalStreamingPlacement {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        local_participant_view(
            &self.run,
            self.participant_id.clone(),
            barrier,
            &self.state_budget,
        )
        .await
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

/// Worker-local admission half.
///
/// A worker-local policy returns no charge, so this half is only exercised when
/// a caller supplies one explicitly; it then proves the exact capacity against
/// an authored budget before returning.
pub struct LocalPlacementAdmission {
    budget: StreamingResourceBudget,
}

impl LocalPlacementAdmission {
    /// Bind one authored route-capacity budget.
    #[must_use]
    pub const fn new(budget: StreamingResourceBudget) -> Self {
        Self { budget }
    }
}

#[async_trait(?Send)]
impl StreamingPlacementAdmission for LocalPlacementAdmission {
    async fn reserve_route(
        &mut self,
        charge: PlacementRouteCharge,
    ) -> Result<PlacementRouteReservation, PlacementError> {
        let lease = self
            .budget
            .acquire(charge.items, charge.bytes)
            .await
            .map_err(|_| PlacementError::state_budget(StateBudgetFailureCode::ItemCapacity))?;
        Ok(PlacementRouteReservation::new(charge.session, lease))
    }
}

/// Worker-local preparation half over a bounded slab.
pub struct LocalPlacementSubmitter {
    shared: Rc<LocalPlacementShared>,
    max_prepared: usize,
    next_handle: u64,
}

impl LocalPlacementSubmitter {
    fn new(shared: Rc<LocalPlacementShared>, max_prepared: usize) -> Self {
        Self {
            shared,
            max_prepared,
            next_handle: 0,
        }
    }
}

#[async_trait(?Send)]
impl StreamingPlacementSubmitter for LocalPlacementSubmitter {
    async fn prepare(
        &mut self,
        decision: PlacementDecision,
        action: &ExecutableDatasetAction,
    ) -> Result<PlacementHandle, PlacementError> {
        if self.shared.is_preparing_stopped.get() {
            return Err(PlacementError::placement(PlacementFailureCode::Cancelled));
        }
        let mut prepared = self.shared.prepared.borrow_mut();
        // The slab bound is a second, independent limit on in-flight work under
        // the active-execution lease: a placement cannot outlive its slab entry.
        if prepared.len() >= self.max_prepared {
            return Err(PlacementError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        let id = PlacementHandleId::new(self.next_handle);
        self.next_handle = self.next_handle.checked_add(1).ok_or_else(|| {
            PlacementError::placement(PlacementFailureCode::TargetOverflow)
        })?;
        let handle = PlacementHandle {
            id,
            action_id: action.action_id(),
            session: action.session_key(),
            ownership_epoch: decision.ownership_epoch,
            global_sequence: None,
        };
        prepared.insert(id, handle);
        drop(prepared);
        self.shared
            .events
            .borrow_mut()
            .push_back(PlacementEvent::Prepared(PlacementPreparedReceipt {
                handle: id,
                content_digest: ContentDigest::from_bytes(*blake3::hash(
                    action.action_id().as_bytes(),
                )
                .as_bytes()),
            }));
        self.shared.wake.notify_one();
        Ok(handle)
    }

    fn bind_sequence(
        &mut self,
        handle: PlacementHandleId,
        sequence: GlobalSequence,
    ) -> Result<(), PlacementError> {
        let mut prepared = self.shared.prepared.borrow_mut();
        let entry = prepared
            .get_mut(&handle)
            .ok_or_else(|| PlacementError::placement(PlacementFailureCode::RouteUnavailable))?;
        if entry.global_sequence.is_some() {
            return Err(PlacementError::placement(
                PlacementFailureCode::DigestMismatch,
            ));
        }
        entry.global_sequence = Some(sequence);
        Ok(())
    }

    async fn release(&mut self, handle: PlacementHandleId) -> Result<(), PlacementError> {
        let removed = self.shared.prepared.borrow_mut().remove(&handle);
        if removed.is_none() {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        self.shared.wake.notify_one();
        Ok(())
    }

    fn prepared_count(&self) -> usize {
        self.shared.prepared.borrow().len()
    }
}

/// Worker-local event stream half.
pub struct LocalPlacementDriver {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    shared: Rc<LocalPlacementShared>,
    state_budget: StreamingResourceBudget,
}

#[async_trait(?Send)]
impl StreamingPlacementDriver for LocalPlacementDriver {
    async fn next_event(&mut self) -> Result<PlacementEvent, PlacementError> {
        loop {
            if let Some(event) = self.shared.events.borrow_mut().pop_front() {
                return Ok(event);
            }
            // `Notify` is created before the borrow is released above, so a
            // producer that pushes between the pop and the wait still wakes this
            // future: `notify_one` stores a permit.
            self.shared.wake.notified().await;
        }
    }

    async fn drain(&mut self) -> Result<(), PlacementError> {
        self.shared.prepared.borrow_mut().clear();
        self.shared.events.borrow_mut().clear();
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for LocalPlacementDriver {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        local_participant_view(
            &self.run,
            self.participant_id.clone(),
            barrier,
            &self.state_budget,
        )
        .await
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

/// Worker-local control half.
pub struct LocalPlacementControlHandle {
    shared: Rc<LocalPlacementShared>,
}

#[async_trait(?Send)]
impl StreamingPlacementControl for LocalPlacementControlHandle {
    fn stop_preparing(&self) {
        self.shared.is_preparing_stopped.set(true);
        self.shared.wake.notify_one();
    }

    fn cancel_pending(&self) {
        self.shared.wake.notify_one();
    }

    async fn cancel_inflight(&self) -> Result<(), PlacementError> {
        let handles: Vec<_> = self
            .shared
            .prepared
            .borrow()
            .keys()
            .copied()
            .collect();
        let mut events = self.shared.events.borrow_mut();
        for handle in handles {
            events.push_back(PlacementEvent::Failed(PlacementFailureReceipt {
                handle: Some(handle),
                code: PlacementFailureCode::Cancelled,
            }));
        }
        drop(events);
        self.shared.wake.notify_one();
        Ok(())
    }
}

/// Assemble one bounded worker-local placement binding.
///
/// `max_prepared` bounds the placement slab and `route_budget` bounds any route
/// charge a caller supplies explicitly; the worker-local policy itself declares
/// no charge.
#[must_use]
pub fn local_placement_binding(
    run: StreamRunIdentity,
    policy_participant: CheckpointParticipantId,
    driver_participant: CheckpointParticipantId,
    max_prepared: usize,
    route_budget: StreamingResourceBudget,
    state_budget: StreamingResourceBudget,
) -> PreparedStreamingPlacementBinding {
    let shared = Rc::new(LocalPlacementShared::default());
    PreparedStreamingPlacementBinding {
        admission: Box::new(LocalPlacementAdmission::new(route_budget)),
        policy: Box::new(LocalStreamingPlacement::new(
            policy_participant,
            run.clone(),
            state_budget.clone(),
        )),
        submitter: Box::new(LocalPlacementSubmitter::new(
            Rc::clone(&shared),
            max_prepared,
        )),
        driver: Box::new(LocalPlacementDriver {
            participant_id: driver_participant,
            run,
            shared: Rc::clone(&shared),
            state_budget,
        }),
        control: Rc::new(LocalPlacementControlHandle { shared }),
    }
}

/// Build the empty barrier-bound participant view shared by both local halves.
///
/// A worker-local placement retains no route map, so its checkpoint view is the
/// empty payload bound to the barrier's cut. The payload still carries an exact
/// one-item, zero-byte charge because the checkpoint contract accepts no
/// uncharged bytes, not even none of them.
async fn local_participant_view(
    run: &StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    barrier: &CheckpointBarrier,
    budget: &StreamingResourceBudget,
) -> Result<PreparedParticipantState, CheckpointError> {
    let lease = budget.try_acquire(1, 0).map_err(|_| CheckpointError::StateBudget {
        participant: participant_id.clone(),
        code: StateBudgetFailureCode::ItemCapacity,
    })?;
    let payload = BudgetedCheckpointBytes::new(Bytes::new(), lease)?;
    PreparedParticipantState::new(
        run.clone(),
        participant_id,
        "aiperf.streaming.placement.local",
        1,
        barrier.cut.clone(),
        0,
        payload,
    )
}
