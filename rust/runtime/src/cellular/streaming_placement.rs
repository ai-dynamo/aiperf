// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic sticky session placement and the no-early-issue fence.
//!
//! This module sits above the bounded transfer plane of
//! [`crate::cellular::streaming_transport`] and decides *where* a session's
//! actions execute and *when* a staged action is allowed to issue. Three
//! properties are load-bearing:
//!
//! - **Sticky, reproducible assignment.** [`assign_cell`] hashes
//!   `(plan_digest, StableSessionKey)` with a domain-separated,
//!   length-prefixed BLAKE3 and takes the remainder over the cell count. The
//!   same plan digest and session always select the same destination, so a
//!   controller restart reproduces its placement without replaying history and
//!   a session never splits across cells mid-run.
//! - **Route capacity is owned, not counted.** An installed route holds the
//!   [`BudgetLease`] it was admitted under
//!   ([`BudgetOwnedSessionRoute`]). Retiring a route drops the lease, which is
//!   what wakes a pending reservation. Because
//!   [`CellularRouteAdmission`] holds only a cheap clone of the accounting
//!   handle and *no* reference to the route map, a terminal event can retire a
//!   route while a reservation is pending — the two are separately borrowable.
//! - **Prepare cannot issue.** [`ActiveExecutionSet::accept_prepare`] records
//!   staged state and has no argument, field, or captured handle through which
//!   an endpoint submitter is reachable. Only
//!   [`ActiveExecutionSet::issue_if_fenced`], reached solely from
//!   [`ActiveExecutionSet::accept_release`], moves an entry to issued, and only
//!   on an exact `(plan_digest, route_id, action_id, global_sequence,
//!   ownership_epoch)` match. The moment that release is sent is decided by the
//!   controller's [`Clock`] alone ([`release_at_controller_target`]); a skewed
//!   cell clock cannot move issue earlier because the cell never reads a
//!   timestamp at all.
//!
//! The module is cellular-shaped but transport-free: it imports no Velo type
//! and performs no I/O, so its determinism is testable against the boundary.

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::streaming::action::OrderedDatasetAction;
use crate::streaming::budget::{BudgetError, BudgetLease, StreamingResourceBudget};
use crate::streaming::failure::{
    PlacementFailureCode, StableStreamingFailure, StreamingFailureStage,
};
use crate::streaming::identity::{
    GlobalSequence, SessionCausalFrontier, SessionOwnershipEpoch, StableActionId, StableSessionKey,
};

/// Domain separator for sticky placement assignment.
///
/// Changing these bytes reassigns every session, so the value is versioned and
/// a bump is a deliberate, reviewable wire change.
const STICKY_PLACEMENT_DOMAIN: &[u8] = b"aiperf-streaming-sticky-placement-v1\0";

/// Exact byte charge retained by one installed route entry.
///
/// A single named constant rather than an ad-hoc `size_of` at each call site:
/// restore-time reacquisition must charge exactly what admission charged, or a
/// restored set that fit the authored budget when it was written could fail to
/// fit when it is read back.
pub const ROUTE_ENTRY_BYTES: usize =
    size_of::<StableSessionKey>() + size_of::<SessionRoute>() + size_of::<GlobalSequence>();

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Select the destination cell for one session under one plan.
///
/// Keyed by the plan digest so two runs with different plans do not share a
/// placement, and length-prefixed so no pair of distinct inputs can collide by
/// concatenation. `cell_count` is validated non-zero by
/// [`StickySessionPlacement::new`], so the remainder is total.
#[must_use]
pub fn assign_cell(plan_digest: &[u8; 32], session: StableSessionKey, cell_count: u32) -> u32 {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, STICKY_PLACEMENT_DOMAIN);
    update_field(&mut hasher, plan_digest);
    update_field(&mut hasher, session.as_bytes());
    let digest = *hasher.finalize().as_bytes();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    let raw = u64::from_le_bytes(head);
    // The remainder of a `u64` by a nonzero `u32` always fits a `u32`.
    (raw % u64::from(cell_count)) as u32
}

/// A placement failure with a stable stage and machine-readable code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementError {
    code: PlacementFailureCode,
}

impl PlacementError {
    /// Construct a placement-stage failure.
    #[must_use]
    pub const fn placement(code: PlacementFailureCode) -> Self {
        Self { code }
    }

    /// Return the stable failure classification.
    #[must_use]
    pub const fn failure_code(self) -> PlacementFailureCode {
        self.code
    }
}

impl fmt::Display for PlacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "placement failed: {}", self.code.code())
    }
}

impl std::error::Error for PlacementError {}

impl StableStreamingFailure for PlacementError {
    fn stage(&self) -> StreamingFailureStage {
        StreamingFailureStage::Placement
    }

    fn code(&self) -> &'static str {
        self.code.code()
    }
}

/// Map a budget outcome onto the placement failure vocabulary.
///
/// A closed budget is a cancellation, not a capacity problem: the run is
/// shutting down and retrying would never succeed.
const fn map_budget_error(error: BudgetError) -> PlacementError {
    match error {
        BudgetError::Closed => PlacementError::placement(PlacementFailureCode::Cancelled),
        _ => PlacementError::placement(PlacementFailureCode::RouteUnavailable),
    }
}

/// A session's pinned destination and its current ownership epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRoute {
    /// Stable route identifier stamped into every wire frame.
    pub route_id: u32,
    /// Destination cell ordinal.
    pub destination_cell: u32,
    /// Monotonic fencing epoch for this owner.
    pub ownership_epoch: SessionOwnershipEpoch,
}

/// One installed route holding the capacity it was admitted under.
///
/// The lease is the only owner of the route's charge, so removing the entry is
/// what returns capacity — there is no separate release path to forget.
#[derive(Debug)]
pub struct BudgetOwnedSessionRoute {
    /// The destination this session is pinned to.
    pub route: SessionRoute,
    /// Non-cloneable lease released when the route retires.
    pub lease: BudgetLease,
    /// Greatest global sequence this policy has placed on the route.
    ///
    /// Retirement waits until the session's causal frontier covers this, so a
    /// route is never reused while an earlier action is still in flight.
    pub highest_sequence: GlobalSequence,
}

/// Exact item-and-byte charge required to install one route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementRouteCharge {
    /// Session the route will be installed for.
    pub session: StableSessionKey,
    /// Objects retained by the route entry.
    pub items: usize,
    /// Bytes retained by the route entry.
    pub bytes: usize,
}

/// Admitted route capacity, not yet installed.
#[derive(Debug)]
pub struct PlacementRouteReservation {
    /// Session the capacity was admitted for.
    pub session: StableSessionKey,
    /// The admitted, non-cloneable capacity.
    pub lease: BudgetLease,
}

/// The route one action was placed on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionPlacementDecision {
    /// Stable route identifier.
    pub route_id: u32,
    /// Destination cell ordinal.
    pub destination_cell: u32,
    /// Fencing epoch of the owning session route.
    pub ownership_epoch: SessionOwnershipEpoch,
}

/// Route selection for streaming sessions.
///
/// Every method is synchronous on purpose. Awaiting for capacity lives in
/// [`CellularRouteAdmission`], which owns no route state, so a terminal event
/// can retire a route through this trait while a reservation is pending
/// elsewhere in the same `select!`.
pub trait StreamingPlacementPolicy {
    /// Return the capacity this action requires, or `None` when its session
    /// already owns a route.
    fn route_admission(
        &self,
        action: &OrderedDatasetAction,
    ) -> Result<Option<PlacementRouteCharge>, PlacementError>;

    /// Hand admitted capacity to the policy for the immediately following
    /// [`StreamingPlacementPolicy::place`].
    fn install_route_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError>;

    /// Resolve the destination for one causally ready action.
    fn place(
        &mut self,
        action: &OrderedDatasetAction,
    ) -> Result<SessionPlacementDecision, PlacementError>;

    /// Retire a session's route once its causal frontier proves completion.
    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError>;
}

/// Separately borrowable async route-capacity owner.
///
/// Holds a cheap clone of the same accounting handle as the policy and **no**
/// reference to the route map. That split is what lets a terminal event retire
/// a route — dropping its lease — and thereby wake a `reserve_route` that is
/// already pending, without either side borrowing the other.
#[derive(Clone, Debug)]
pub struct CellularRouteAdmission {
    route_budget: StreamingResourceBudget,
}

impl CellularRouteAdmission {
    /// Construct an admission owner over the shared route budget.
    #[must_use]
    pub fn new(route_budget: StreamingResourceBudget) -> Self {
        Self { route_budget }
    }

    /// Await the exact charge for one route entry.
    pub async fn reserve_route(
        &mut self,
        charge: PlacementRouteCharge,
    ) -> Result<PlacementRouteReservation, PlacementError> {
        let lease = self
            .route_budget
            .acquire(charge.items, charge.bytes)
            .await
            .map_err(map_budget_error)?;
        Ok(PlacementRouteReservation {
            session: charge.session,
            lease,
        })
    }
}

/// Deterministic per-session cellular placement with owned route capacity.
///
/// `BTreeMap` rather than `HashMap` because a checkpoint view of the route set
/// must serialize in a stable order for a reproducible participant digest.
#[derive(Debug)]
pub struct StickySessionPlacement {
    plan_digest: [u8; 32],
    cell_count: u32,
    routes: BTreeMap<StableSessionKey, BudgetOwnedSessionRoute>,
    pending_reservation: Option<PlacementRouteReservation>,
    route_budget: StreamingResourceBudget,
}

impl StickySessionPlacement {
    /// Construct a placement policy for one plan over `cell_count` cells.
    ///
    /// A zero cell count is refused here so the assignment remainder is total
    /// at every later call site.
    pub fn new(
        plan_digest: [u8; 32],
        cell_count: u32,
        route_budget: StreamingResourceBudget,
    ) -> Result<Self, PlacementError> {
        if cell_count == 0 {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        Ok(Self {
            plan_digest,
            cell_count,
            routes: BTreeMap::new(),
            pending_reservation: None,
            route_budget,
        })
    }

    /// A separately owned admission handle over the same route budget.
    #[must_use]
    pub fn admission(&self) -> CellularRouteAdmission {
        CellularRouteAdmission::new(self.route_budget.clone())
    }

    /// Number of currently installed routes.
    #[must_use]
    pub fn installed_route_count(&self) -> usize {
        self.routes.len()
    }

    /// Borrow the route a session is pinned to, when one is installed.
    #[must_use]
    pub fn route_for(&self, session: StableSessionKey) -> Option<SessionRoute> {
        self.routes.get(&session).map(|owned| owned.route)
    }

    /// The charge a session needs, or `None` when it already owns a route.
    pub fn required_route_charge(
        &self,
        session: StableSessionKey,
    ) -> Result<Option<PlacementRouteCharge>, PlacementError> {
        if self.routes.contains_key(&session) {
            return Ok(None);
        }
        Ok(Some(PlacementRouteCharge {
            session,
            items: 1,
            bytes: ROUTE_ENTRY_BYTES,
        }))
    }

    /// Stage admitted capacity for the immediately following placement.
    ///
    /// Only one reservation may be pending: the fused pipeline performs install
    /// and place as adjacent serial operations with no intervening `.await`, so
    /// a second pending reservation is an invariant violation rather than a
    /// recoverable condition.
    pub fn install_pending_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError> {
        if self.pending_reservation.is_some() {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        self.pending_reservation = Some(reservation);
        Ok(())
    }

    /// Resolve a destination for one session at one global sequence.
    ///
    /// An already-routed session returns its pinned destination and never
    /// consumes capacity. A new session consumes the pending reservation, which
    /// must name that exact session.
    pub fn place_session(
        &mut self,
        session: StableSessionKey,
        sequence: GlobalSequence,
    ) -> Result<SessionPlacementDecision, PlacementError> {
        if let Some(owned) = self.routes.get_mut(&session) {
            if self.pending_reservation.is_some() {
                return Err(PlacementError::placement(
                    PlacementFailureCode::RouteUnavailable,
                ));
            }
            if sequence > owned.highest_sequence {
                owned.highest_sequence = sequence;
            }
            return Ok(Self::decision(owned.route));
        }
        let Some(reservation) = self.pending_reservation.take() else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        if reservation.session != session {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        let destination = assign_cell(&self.plan_digest, session, self.cell_count);
        let route = SessionRoute {
            route_id: destination,
            destination_cell: destination,
            ownership_epoch: SessionOwnershipEpoch::new(0),
        };
        self.routes.insert(
            session,
            BudgetOwnedSessionRoute {
                route,
                lease: reservation.lease,
                highest_sequence: sequence,
            },
        );
        Ok(Self::decision(route))
    }

    /// Retire a route once its frontier covers everything placed on it.
    ///
    /// Dropping the entry drops its lease, which returns capacity to the shared
    /// budget and wakes a pending `reserve_route` on the separately owned
    /// admission object. An unknown session is a no-op: terminal events are
    /// delivered at least once.
    pub fn retire_route_if_fenced(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError> {
        let Some(owned) = self.routes.get(&session) else {
            return Ok(());
        };
        if owned.route.ownership_epoch != ownership_epoch {
            return Err(PlacementError::placement(
                PlacementFailureCode::StaleOwnershipEpoch,
            ));
        }
        if causal_frontier.through_sequence < owned.highest_sequence {
            return Ok(());
        }
        drop(self.routes.remove(&session));
        Ok(())
    }

    /// Reinstall a restored route set, charging every entry before publishing.
    ///
    /// The complete set is rebuilt into a local map first, so an exhausted
    /// budget leaves the live route map untouched rather than half-restored.
    pub async fn restore_routes(
        &mut self,
        restored: impl IntoIterator<Item = (StableSessionKey, SessionRoute, GlobalSequence)>,
    ) -> Result<(), PlacementError> {
        let mut rebuilt = BTreeMap::new();
        for (session, route, highest_sequence) in restored {
            let lease = self
                .route_budget
                .acquire(1, ROUTE_ENTRY_BYTES)
                .await
                .map_err(map_budget_error)?;
            rebuilt.insert(
                session,
                BudgetOwnedSessionRoute {
                    route,
                    lease,
                    highest_sequence,
                },
            );
        }
        self.routes = rebuilt;
        Ok(())
    }

    const fn decision(route: SessionRoute) -> SessionPlacementDecision {
        SessionPlacementDecision {
            route_id: route.route_id,
            destination_cell: route.destination_cell,
            ownership_epoch: route.ownership_epoch,
        }
    }
}

impl StreamingPlacementPolicy for StickySessionPlacement {
    fn route_admission(
        &self,
        action: &OrderedDatasetAction,
    ) -> Result<Option<PlacementRouteCharge>, PlacementError> {
        self.required_route_charge(action.action().session_key())
    }

    fn install_route_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError> {
        self.install_pending_reservation(reservation)
    }

    fn place(
        &mut self,
        action: &OrderedDatasetAction,
    ) -> Result<SessionPlacementDecision, PlacementError> {
        self.place_session(action.action().session_key(), action.sequence())
    }

    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError> {
        self.retire_route_if_fenced(session, ownership_epoch, causal_frontier)
    }
}

/// The exact tuple a release must carry to grant issue authority.
///
/// All five fields are compared. A release that matches four of them is not a
/// near miss to be tolerated: it names a different action, a different plan, or
/// a fenced owner, and issuing it would duplicate or misattribute load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseFence {
    /// Plan the release was authored under.
    pub plan_digest: [u8; 32],
    /// Route the action was staged on.
    pub route_id: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Stable global order of the staged action.
    pub global_sequence: GlobalSequence,
    /// Fencing epoch of the owning session route.
    pub ownership_epoch: SessionOwnershipEpoch,
}

/// Lifecycle of one entry in the active execution set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StagedState {
    /// Prepared on the cell; not yet granted issue authority.
    Staged,
    /// A matching release was accepted; the cell may issue.
    Issued,
    /// A terminal event was accepted; the entry is being reclaimed.
    Terminal,
}

/// Outcome of a fenced release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IssueGrant {
    /// The action moved from staged to issued on this call.
    Issued,
    /// A byte-identical release already granted authority; nothing changed.
    AlreadyIssued,
}

/// One staged action and the fence it must be released against.
///
/// Deliberately holds no submitter, no endpoint handle, and no [`Clock`]: the
/// type is the fence. Adding any of those fields would make the prepare path
/// able to issue, which is exactly the invariant this module exists to hold.
#[derive(Clone, Copy, Debug)]
struct ActiveExecutionEntry {
    plan_digest: [u8; 32],
    global_sequence: GlobalSequence,
    ownership_epoch: SessionOwnershipEpoch,
    state: StagedState,
}

/// Controller-side set of actions staged on cells and awaiting release.
///
/// [`ActiveExecutionSet::accept_prepare`] can only ever stage. Issue authority
/// is granted exclusively by [`ActiveExecutionSet::issue_if_fenced`], which is
/// private and reachable only through
/// [`ActiveExecutionSet::accept_release`]. There is no field on this type, nor
/// any argument to the prepare path, through which an endpoint action submitter
/// is reachable.
#[derive(Debug, Default)]
pub struct ActiveExecutionSet {
    entries: BTreeMap<(u32, StableActionId), ActiveExecutionEntry>,
    issued: u64,
}

impl ActiveExecutionSet {
    /// Construct an empty active set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one action as staged. This path never issues.
    ///
    /// A byte-identical re-prepare is idempotent, because prepare frames are
    /// retransmitted on reconnect. A conflicting re-prepare for the same
    /// `(route_id, action_id)` is refused and leaves the existing entry intact.
    pub fn accept_prepare(
        &mut self,
        fence: ReleaseFence,
    ) -> Result<(), PlacementError> {
        let key = (fence.route_id, fence.action_id);
        if let Some(existing) = self.entries.get(&key) {
            if existing.plan_digest == fence.plan_digest
                && existing.global_sequence == fence.global_sequence
                && existing.ownership_epoch == fence.ownership_epoch
            {
                return Ok(());
            }
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        self.entries.insert(
            key,
            ActiveExecutionEntry {
                plan_digest: fence.plan_digest,
                global_sequence: fence.global_sequence,
                ownership_epoch: fence.ownership_epoch,
                state: StagedState::Staged,
            },
        );
        Ok(())
    }

    /// Validate the exact release tuple, then grant issue authority once.
    pub fn accept_release(&mut self, fence: &ReleaseFence) -> Result<IssueGrant, PlacementError> {
        self.issue_if_fenced(fence)
    }

    /// The single path from staged to issued.
    ///
    /// Every rejection leaves the entry staged — fenced, but neither issued nor
    /// dropped — so a late or corrected release can still arrive.
    fn issue_if_fenced(&mut self, fence: &ReleaseFence) -> Result<IssueGrant, PlacementError> {
        let Some(entry) = self.entries.get_mut(&(fence.route_id, fence.action_id)) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        if entry.plan_digest != fence.plan_digest {
            return Err(PlacementError::placement(
                PlacementFailureCode::DigestMismatch,
            ));
        }
        if fence.ownership_epoch < entry.ownership_epoch {
            return Err(PlacementError::placement(
                PlacementFailureCode::StaleOwnershipEpoch,
            ));
        }
        if entry.global_sequence != fence.global_sequence
            || entry.ownership_epoch != fence.ownership_epoch
        {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        match entry.state {
            StagedState::Staged => {
                entry.state = StagedState::Issued;
                self.issued = self.issued.saturating_add(1);
                Ok(IssueGrant::Issued)
            }
            StagedState::Issued => Ok(IssueGrant::AlreadyIssued),
            StagedState::Terminal => Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            )),
        }
    }

    /// Mark one issued action terminal and reclaim its entry.
    pub fn observe_terminal(
        &mut self,
        route_id: u32,
        action_id: StableActionId,
    ) -> Result<(), PlacementError> {
        let Some(entry) = self.entries.get_mut(&(route_id, action_id)) else {
            return Ok(());
        };
        entry.state = StagedState::Terminal;
        self.entries.remove(&(route_id, action_id));
        Ok(())
    }

    /// Lifecycle state of one tracked action.
    #[must_use]
    pub fn state_of(&self, route_id: u32, action_id: StableActionId) -> Option<StagedState> {
        self.entries
            .get(&(route_id, action_id))
            .map(|entry| entry.state)
    }

    /// Number of actions currently staged and not yet issued.
    #[must_use]
    pub fn staged_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.state == StagedState::Staged)
            .count()
    }

    /// Total actions this set has ever granted issue authority to.
    #[must_use]
    pub const fn issued_count(&self) -> u64 {
        self.issued
    }
}

/// Sink for a controller-authored release.
///
/// Kept abstract so [`release_at_controller_target`] can be exercised under a
/// virtual clock without a transport, and so the transfer plane remains the
/// only thing that knows how a release reaches the wire.
pub trait ReleaseSubmitter {
    /// Send one authenticated release for the fenced action.
    fn submit_release(&mut self, fence: &ReleaseFence) -> Result<(), PlacementError>;
}

/// Sleep on the controller's clock until the authored release coordinate, then
/// send the release.
///
/// The cell never reads, compares, or interprets this timestamp; it only
/// observes that a valid release arrived. That is what makes a skewed cell
/// clock unable to move issue earlier. [`Clock::sleep`] treats a non-positive
/// duration as ready-now, so a target already in the past releases immediately;
/// only an `i64` subtraction overflow is a target that cannot be represented.
pub async fn release_at_controller_target<S: ReleaseSubmitter>(
    clock: Rc<dyn Clock>,
    target_ns: i64,
    submitter: &mut S,
    fence: ReleaseFence,
) -> Result<(), PlacementError> {
    let delay = target_ns
        .checked_sub(clock.now_ns())
        .ok_or_else(|| PlacementError::placement(PlacementFailureCode::TargetOverflow))?;
    clock.sleep(delay).await;
    submitter.submit_release(&fence)
}
