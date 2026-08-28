// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Crash-safe route migration driven through the global checkpoint generation.
//!
//! One property makes this module correct: **the epoch increment and the new
//! owner become visible only through
//! [`StreamingCheckpointCoordinator::commit_barrier`]'s CAS.** A crash at any
//! point before that CAS restores the old owner, because
//! [`SessionRouteState`](super::streaming_placement::SessionRouteState) collapses
//! to its committed `old` on restore; a crash at any point after restores the
//! new one, because the committed generation already named it. Nothing the
//! destination cell has staged is authoritative in between.
//!
//! The transaction is exposed as seven separately callable steps rather than one
//! opaque call, so the observable crash points are the method boundaries and a
//! test can stop at any of them without a fault-injection hook inside the
//! driver. [`CellularStreamingController::migrate`] is their composition.
//!
//! | Step | Method | Crash here restores |
//! | --- | --- | --- |
//! | 1-2 | [`CellularStreamingController::freeze_session`] | old |
//! | 3 | [`CellularStreamingController::drain_old_cell`] | old |
//! | 4 | [`CellularStreamingController::commit_fence`] | old |
//! | 5 | [`CellularStreamingController::stage_new_cell`] | old |
//! | 6 | [`CellularStreamingController::commit_route_generation`] | **new** |
//! | 7 | [`CellularStreamingController::promote_owner`] | new |
//!
//! Every byte routes through the controller. The destination cell receives only
//! prepare frames built from controller-owned content and the origin cell
//! receives only cancel and terminal commands, so there is no cell-to-cell
//! channel and no canonical session state on either cell.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::streaming::budget::{BudgetLease, StreamingResourceBudget};
use crate::streaming::checkpoint::{
    BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
    CommittedParticipantReceipt, CommittedParticipantState, PreparedParticipantState,
    StreamRunIdentity, StreamingCheckpointParticipant,
};
use crate::streaming::checkpoint_coordinator::{
    PreparedCheckpointResultInput, StreamingCheckpointCoordinator,
};
use crate::streaming::failure::PlacementFailureCode;
use crate::streaming::identity::{
    GlobalSequence, SessionOwnershipEpoch, StableActionId, StableSessionKey,
};
use crate::streaming::unit::StateBudgetFailureCode;

use super::streaming_placement::{
    ActiveExecutionSet, CellularRouteAdmission, CheckpointedSessionRoute, PlacementError,
    PlacementRouteCharge, ROUTE_ENTRY_BYTES, ReleaseFence, SessionRoute, SessionRouteState,
    StickySessionPlacement,
};

/// Schema identity of the checkpointed route set.
const ROUTE_SET_SCHEMA_ID: &str = "aiperf.streaming.placement.routes";
/// Schema version of the checkpointed route set.
const ROUTE_SET_SCHEMA_VERSION: u32 = 1;

/// A migration failed at placement or at the checkpoint boundary.
///
/// The two are kept apart because their recovery differs: a placement refusal is
/// an abort that leaves the committed owner intact, while a checkpoint failure
/// may or may not have advanced the head and is classified by the coordinator's
/// own reliability routing.
#[derive(Debug)]
pub enum MigrationError {
    /// Placement authority or capacity refused a step.
    Placement(PlacementError),
    /// The checkpoint boundary refused or failed a barrier.
    Checkpoint(CheckpointError),
}

impl From<PlacementError> for MigrationError {
    fn from(error: PlacementError) -> Self {
        Self::Placement(error)
    }
}

impl From<CheckpointError> for MigrationError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

impl MigrationError {
    /// The placement failure classification, when the failure was a placement.
    #[must_use]
    pub const fn placement_code(&self) -> Option<PlacementFailureCode> {
        match self {
            Self::Placement(error) => Some(error.failure_code()),
            Self::Checkpoint(_) => None,
        }
    }
}

impl fmt::Display for MigrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Placement(error) => write!(formatter, "route migration refused: {error}"),
            Self::Checkpoint(error) => write!(formatter, "route migration checkpoint: {error}"),
        }
    }
}

impl std::error::Error for MigrationError {}

/// One post-fence fragment held at the controller under an owned charge.
///
/// Held fragments are neither prepared to the old cell nor released on the new
/// one. They are ordinary budgeted state, so a saturated hold stops upstream
/// source pulls through the same backpressure every other queue uses; there is
/// no unbounded migration queue, no spill file, and no per-migration task.
#[derive(Debug)]
pub struct HeldFragment {
    /// The action this fragment would prepare.
    pub action_id: StableActionId,
    /// The frozen-out global order of the fragment.
    pub sequence: GlobalSequence,
    /// The capacity the hold occupies for as long as it is held.
    pub lease: BudgetLease,
}

/// Checkpointed payload of the whole route set.
///
/// A named struct rather than a bare `Vec` so the schema can gain a field
/// without a version-ambiguous decode; `deny_unknown_fields` makes a foreign or
/// newer payload a verification failure rather than a silent partial restore.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointedRouteSet {
    /// Every installed route, in stable session-key order.
    pub routes: Vec<CheckpointedSessionRoute>,
}

/// The route set's checkpoint participant over a shared placement policy.
///
/// The controller and the coordinator both need the placement: the controller
/// mutates the state machine and the coordinator reads it at a barrier cut. They
/// share one `Rc<RefCell<_>>` rather than duplicating the map, and the borrow is
/// taken and released inside each method so no borrow is ever held across an
/// `.await`.
pub struct SessionRoutePlacementParticipant {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    placement: Rc<RefCell<StickySessionPlacement>>,
    state_budget: StreamingResourceBudget,
}

impl fmt::Debug for SessionRoutePlacementParticipant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionRoutePlacementParticipant")
            .field("participant_id", &self.participant_id)
            .finish_non_exhaustive()
    }
}

impl SessionRoutePlacementParticipant {
    /// Bind one participant identity to the shared route map.
    #[must_use]
    pub const fn new(
        run: StreamRunIdentity,
        participant_id: CheckpointParticipantId,
        placement: Rc<RefCell<StickySessionPlacement>>,
        state_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            run,
            participant_id,
            placement,
            state_budget,
        }
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for SessionRoutePlacementParticipant {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let set = CheckpointedRouteSet {
            routes: self.placement.borrow().checkpoint_route_entries(),
        };
        let encoded = serde_json::to_vec(&set).map_err(|_| CheckpointError::ObjectVerification)?;
        let item_count = set.routes.len() as u64;
        let bytes = Bytes::from(encoded);
        let lease = self.state_budget.try_acquire(1, bytes.len()).map_err(|_| {
            CheckpointError::StateBudget {
                participant: self.participant_id.clone(),
                code: StateBudgetFailureCode::ByteCapacity,
            }
        })?;
        let payload = BudgetedCheckpointBytes::new(bytes, lease)?;
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            ROUTE_SET_SCHEMA_ID,
            ROUTE_SET_SCHEMA_VERSION,
            barrier.cut.clone(),
            item_count,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        let Some(state) = state else {
            return Ok(());
        };
        if state.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let set: CheckpointedRouteSet = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        // `restore_route_states` collapses every in-flight migration to its
        // committed owner and reacquires exactly one charge per session, so a
        // crashed migration can leak neither authority nor budget.
        self.placement
            .borrow_mut()
            .restore_route_states(set.routes)
            .await
            .map_err(|_| CheckpointError::StateBudget {
                participant: self.participant_id.clone(),
                code: StateBudgetFailureCode::ItemCapacity,
            })
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }
}

/// Controller-side driver of the fenced route-migration transaction.
///
/// It owns both halves the transaction needs — the placement state machine and
/// the checkpoint coordinator — which is what lets step 6 be a single
/// coordinator CAS rather than a participant committing a barrier that covers
/// the whole participant set.
pub struct CellularStreamingController {
    placement: Rc<RefCell<StickySessionPlacement>>,
    coordinator: StreamingCheckpointCoordinator,
    admission: CellularRouteAdmission,
    active: ActiveExecutionSet,
    held: BTreeMap<(StableSessionKey, StableActionId), HeldFragment>,
    hold_budget: StreamingResourceBudget,
    plan_digest: [u8; 32],
    stale_event_refusals: u64,
}

impl fmt::Debug for CellularStreamingController {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CellularStreamingController")
            .field("held_fragments", &self.held.len())
            .field("stale_event_refusals", &self.stale_event_refusals)
            .finish_non_exhaustive()
    }
}

impl CellularStreamingController {
    /// Bind one controller to its shared placement and its own coordinator.
    #[must_use]
    pub fn new(
        placement: Rc<RefCell<StickySessionPlacement>>,
        coordinator: StreamingCheckpointCoordinator,
        hold_budget: StreamingResourceBudget,
        plan_digest: [u8; 32],
    ) -> Self {
        let admission = placement.borrow().admission();
        Self {
            placement,
            coordinator,
            admission,
            active: ActiveExecutionSet::new(),
            held: BTreeMap::new(),
            hold_budget,
            plan_digest,
            stale_event_refusals: 0,
        }
    }

    /// Borrow the shared placement policy.
    #[must_use]
    pub fn placement(&self) -> &Rc<RefCell<StickySessionPlacement>> {
        &self.placement
    }

    /// Borrow the controller-side active execution set.
    #[must_use]
    pub const fn active(&self) -> &ActiveExecutionSet {
        &self.active
    }

    /// Mutably borrow the controller-side active execution set.
    pub const fn active_mut(&mut self) -> &mut ActiveExecutionSet {
        &mut self.active
    }

    /// Borrow the checkpoint coordinator.
    #[must_use]
    pub const fn coordinator(&self) -> &StreamingCheckpointCoordinator {
        &self.coordinator
    }

    /// A separately owned admission handle over the route budget.
    #[must_use]
    pub fn admission(&self) -> CellularRouteAdmission {
        self.admission.clone()
    }

    /// Post-fence fragments currently held at the controller.
    #[must_use]
    pub fn held_fragment_count(&self) -> usize {
        self.held.len()
    }

    /// Placement events refused for naming a fenced ownership epoch.
    #[must_use]
    pub const fn stale_event_refusals(&self) -> u64 {
        self.stale_event_refusals
    }

    /// Admit or refuse one inbound placement event by ownership epoch.
    ///
    /// A refused event is counted and dropped: it never reaches session state,
    /// the active execution set, or a result segment.
    pub fn admit_placement_event(
        &mut self,
        session: StableSessionKey,
        epoch: SessionOwnershipEpoch,
    ) -> Result<(), PlacementError> {
        let outcome = self.placement.borrow().admit_event_epoch(session, epoch);
        if outcome.is_err() {
            self.stale_event_refusals = self.stale_event_refusals.saturating_add(1);
        }
        outcome
    }

    /// Admit one terminal receipt from the fenced origin cell.
    ///
    /// Keyed on the freeze point, never on the epoch: a receipt at or below the
    /// fence was committed at step 4 and must still be accepted after the epoch
    /// increased, or the run hangs on an active set that never closes.
    pub fn admit_fenced_terminal_receipt(
        &self,
        session: StableSessionKey,
        sequence: GlobalSequence,
    ) -> Result<(), PlacementError> {
        if self
            .placement
            .borrow()
            .admits_fenced_terminal_receipt(session, sequence)
        {
            return Ok(());
        }
        Err(PlacementError::placement(
            PlacementFailureCode::StaleOwnershipEpoch,
        ))
    }

    /// Hold one post-fence fragment under an ordinary content lease.
    pub async fn hold_fragment(
        &mut self,
        session: StableSessionKey,
        action_id: StableActionId,
        sequence: GlobalSequence,
        bytes: usize,
    ) -> Result<(), PlacementError> {
        let lease = self
            .hold_budget
            .acquire(1, bytes)
            .await
            .map_err(|_| PlacementError::placement(PlacementFailureCode::RouteUnavailable))?;
        self.held.insert(
            (session, action_id),
            HeldFragment {
                action_id,
                sequence,
                lease,
            },
        );
        Ok(())
    }

    /// Step 1 and 2: freeze the session and stop old-epoch prepares.
    pub fn freeze_session(
        &mut self,
        session: StableSessionKey,
        through: GlobalSequence,
    ) -> Result<(), MigrationError> {
        self.placement.borrow_mut().begin_fence(session, through)?;
        Ok(())
    }

    /// Step 3: drain or explicitly cancel every origin-cell action at or below
    /// the freeze point.
    ///
    /// A staged, never-issued action is cancelled outright. An action that was
    /// already granted issue authority is left alone: its load may be on the
    /// wire, so it must reach terminal, and its receipt is committed at step 4.
    /// Returns the number of cancellations.
    pub fn drain_old_cell(
        &mut self,
        session: StableSessionKey,
        staged: &[(u32, StableActionId)],
    ) -> Result<usize, MigrationError> {
        let Some(state) = self.placement.borrow().route_state_for(session) else {
            return Err(MigrationError::Placement(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            )));
        };
        if state.fence_through().is_none() {
            return Err(MigrationError::Placement(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            )));
        }
        let mut cancelled = 0;
        for (route_id, action_id) in staged {
            if self.active.cancel_staged(*route_id, *action_id)? {
                cancelled += 1;
            }
        }
        Ok(cancelled)
    }

    /// Step 4: commit controller session state, terminal receipts, and the old
    /// fence. The owner is unchanged by this generation.
    pub async fn commit_fence(&mut self, barrier: CheckpointBarrier) -> Result<(), MigrationError> {
        let mut results = PreparedCheckpointResultInput::empty();
        self.coordinator
            .commit_barrier(barrier, &mut results)
            .await?;
        Ok(())
    }

    /// Step 5: bind and prepare immutable content on the destination without
    /// releasing it.
    ///
    /// The transient second route charge is acquired here and held as a separate
    /// lease, so a `reserve_route` waiter observes the real cost of a session in
    /// `Prepared` rather than a hidden one.
    pub async fn stage_new_cell(
        &mut self,
        session: StableSessionKey,
        destination: u32,
        staged: &[(StableActionId, GlobalSequence)],
    ) -> Result<SessionRoute, MigrationError> {
        let reservation = self
            .admission
            .reserve_route(PlacementRouteCharge {
                session,
                items: 1,
                bytes: ROUTE_ENTRY_BYTES,
            })
            .await?;
        let new = self.placement.borrow_mut().stage_destination(
            session,
            destination,
            reservation.lease,
        )?;
        for (action_id, sequence) in staged {
            self.active.accept_prepare(ReleaseFence {
                plan_digest: self.plan_digest,
                route_id: new.route_id,
                action_id: *action_id,
                global_sequence: *sequence,
                ownership_epoch: new.ownership_epoch,
            })?;
        }
        Ok(new)
    }

    /// Step 6: the CAS. Commit the incremented epoch and the new owner.
    ///
    /// The staged owner is adopted in memory first, because the view this
    /// barrier takes is exactly what a restart reads back. The coordinator does
    /// not roll back a committed head, so this either commits or does not; there
    /// is no half-committed epoch, and a refused CAS reverts to the owner it
    /// replaced. A post-commit notification failure is not an abort — the head
    /// advanced and the migration succeeded.
    pub async fn commit_route_generation(
        &mut self,
        session: StableSessionKey,
        barrier: CheckpointBarrier,
    ) -> Result<(), MigrationError> {
        let (old, _new) = self.placement.borrow_mut().adopt_staged_owner(session)?;
        let mut results = PreparedCheckpointResultInput::empty();
        match self.coordinator.commit_barrier(barrier, &mut results).await {
            Ok(_) => Ok(()),
            Err(error) => {
                self.placement.borrow_mut().revert_adoption(session, old)?;
                Err(MigrationError::Checkpoint(error))
            }
        }
    }

    /// Step 7: promote the staged owner and release everything past the fence.
    ///
    /// Returns the committed epoch and the fragments the hold is handing back.
    pub fn promote_owner(
        &mut self,
        session: StableSessionKey,
    ) -> Result<(SessionOwnershipEpoch, Vec<HeldFragment>), MigrationError> {
        let epoch = self.placement.borrow_mut().commit_owner(session)?;
        let released_keys: Vec<_> = self
            .held
            .keys()
            .filter(|(held_session, _)| *held_session == session)
            .copied()
            .collect();
        let mut released = Vec::with_capacity(released_keys.len());
        for key in released_keys {
            if let Some(fragment) = self.held.remove(&key) {
                released.push(fragment);
            }
        }
        Ok((epoch, released))
    }

    /// Abort an in-flight migration, restoring the committed owner.
    pub fn abort_migration(&mut self, session: StableSessionKey) -> Result<(), MigrationError> {
        self.placement.borrow_mut().abort_migration(session)?;
        Ok(())
    }

    /// Run the complete seven-step transaction.
    ///
    /// A session already owned by `destination` returns its committed epoch
    /// without a second CAS and without re-preparing content, so an identical
    /// retry is idempotent. Any failure before step 6 aborts back to the
    /// committed owner and drops the migration lease.
    pub async fn migrate(
        &mut self,
        session: StableSessionKey,
        destination: u32,
        through: GlobalSequence,
        staged: &[(StableActionId, GlobalSequence)],
        fence_barrier: CheckpointBarrier,
        route_barrier: CheckpointBarrier,
    ) -> Result<SessionOwnershipEpoch, MigrationError> {
        if let Some(SessionRouteState::Owned(route)) =
            self.placement.borrow().route_state_for(session)
            && route.destination_cell == destination
        {
            return Ok(route.ownership_epoch);
        }
        self.freeze_session(session, through)?;
        let old_staged: Vec<_> = staged
            .iter()
            .filter(|(_, sequence)| *sequence <= through)
            .map(|(action_id, _)| (destination, *action_id))
            .collect();
        if let Err(error) = self.drain_old_cell(session, &old_staged) {
            self.abort_migration(session)?;
            return Err(error);
        }
        if let Err(error) = self.commit_fence(fence_barrier).await {
            self.abort_migration(session)?;
            return Err(error);
        }
        let post_fence: Vec<_> = staged
            .iter()
            .filter(|(_, sequence)| *sequence > through)
            .copied()
            .collect();
        if let Err(error) = self.stage_new_cell(session, destination, &post_fence).await {
            self.abort_migration(session)?;
            return Err(error);
        }
        if let Err(error) = self.commit_route_generation(session, route_barrier).await {
            self.abort_migration(session)?;
            return Err(error);
        }
        let (epoch, _released) = self.promote_owner(session)?;
        Ok(epoch)
    }
}
