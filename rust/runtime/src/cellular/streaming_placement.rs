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

/// Ownership state of one session's route during and outside migration.
///
/// The variant carries the *committed* owner in `old` for both migration
/// variants. That is the whole crash-safety argument in one type: a decoded
/// route set can only ever name one authoritative owner, and it is the owner
/// the last committed generation named.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionRouteState {
    /// Steady state: one committed owner.
    Owned(SessionRoute),
    /// Sequence frozen at `through`; the old owner accepts no new prepares and
    /// is draining or cancelling everything at or below the freeze point.
    Fencing {
        /// The committed owner being fenced.
        old: SessionRoute,
        /// Freeze point. Actions at or below it belong to `old`.
        through: GlobalSequence,
    },
    /// Content is staged on `new` but unreleased; `old` is still the committed
    /// authority until the route generation commits.
    Prepared {
        /// The still-authoritative committed owner.
        old: SessionRoute,
        /// The staged, not-yet-authoritative destination.
        new: SessionRoute,
        /// Freeze point.
        through: GlobalSequence,
    },
}

impl SessionRouteState {
    /// The owner that may currently issue. Always `old` until commit.
    #[must_use]
    pub const fn authoritative(&self) -> &SessionRoute {
        match self {
            Self::Owned(route) => route,
            Self::Fencing { old, .. } | Self::Prepared { old, .. } => old,
        }
    }

    /// Whether a prepare for `sequence` may still be submitted.
    ///
    /// Post-freeze actions are staged on `new` (`Prepared`) or held at the
    /// controller (`Fencing`); neither admits a prepare to `old`.
    #[must_use]
    pub const fn admits_prepare(&self, sequence: GlobalSequence) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Fencing { through, .. } | Self::Prepared { through, .. } => {
                sequence.get() <= through.get()
            }
        }
    }

    /// The freeze point, when a migration is in flight.
    #[must_use]
    pub const fn fence_through(&self) -> Option<GlobalSequence> {
        match self {
            Self::Owned(_) => None,
            Self::Fencing { through, .. } | Self::Prepared { through, .. } => Some(*through),
        }
    }

    /// Whether this session is mid-migration and must not be retired.
    #[must_use]
    pub const fn is_migrating(&self) -> bool {
        !matches!(self, Self::Owned(_))
    }
}

/// One installed route holding the capacity it was admitted under.
///
/// The lease is the only owner of the route's charge, so removing the entry is
/// what returns capacity — there is no separate release path to forget.
#[derive(Debug)]
pub struct BudgetOwnedSessionRoute {
    /// Committed and in-flight ownership.
    pub state: SessionRouteState,
    /// Non-cloneable lease released when the route retires.
    pub lease: BudgetLease,
    /// Capacity for the staged destination while `Prepared`.
    ///
    /// Kept as a *separate* lease so the transient double route charge is
    /// visible to [`CellularRouteAdmission`]'s waiters rather than hidden. It is
    /// promoted into `lease` on commit and dropped on abort, and it is never
    /// restored from a checkpoint.
    pub migration_lease: Option<BudgetLease>,
    /// Greatest global sequence this policy has placed on the route.
    ///
    /// Retirement waits until the session's causal frontier covers this, so a
    /// route is never reused while an earlier action is still in flight.
    pub highest_sequence: GlobalSequence,
}

impl BudgetOwnedSessionRoute {
    /// The committed owner of this route.
    #[must_use]
    pub const fn route(&self) -> &SessionRoute {
        self.state.authoritative()
    }
}

/// One session's checkpointed route entry.
///
/// Leases are never serialized: capacity is proven again at restore against the
/// live budget, so a checkpoint can never resurrect a charge that the restarted
/// process cannot actually afford.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointedSessionRoute {
    /// Session the entry routes.
    pub session: StableSessionKey,
    /// Committed and in-flight ownership at the barrier cut.
    pub state: SessionRouteState,
    /// Greatest global sequence placed on the route at the barrier cut.
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

    /// Borrow the committed route a session is pinned to, when one is installed.
    ///
    /// Mid-migration this is still the *old* owner: staged content on the
    /// destination is not authority until the route generation commits.
    #[must_use]
    pub fn route_for(&self, session: StableSessionKey) -> Option<SessionRoute> {
        self.routes.get(&session).map(|owned| *owned.route())
    }

    /// Borrow the full ownership state, including any in-flight migration.
    #[must_use]
    pub fn route_state_for(&self, session: StableSessionKey) -> Option<SessionRouteState> {
        self.routes.get(&session).map(|owned| owned.state)
    }

    /// Number of sessions with exactly one authoritative owner.
    ///
    /// Every installed entry has exactly one, by construction of
    /// [`SessionRouteState::authoritative`]; the count exists so a restored set
    /// can assert it rather than assume it.
    #[must_use]
    pub fn active_owner_count(&self) -> usize {
        self.routes.len()
    }

    /// The number of transient migration charges currently held.
    #[must_use]
    pub fn migration_lease_count(&self) -> usize {
        self.routes
            .values()
            .filter(|owned| owned.migration_lease.is_some())
            .count()
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
            // A post-freeze action belongs to neither owner yet: the controller
            // holds it under an ordinary content lease until step 7 releases it.
            if !owned.state.admits_prepare(sequence) {
                return Err(PlacementError::placement(
                    PlacementFailureCode::RouteUnavailable,
                ));
            }
            if sequence > owned.highest_sequence {
                owned.highest_sequence = sequence;
            }
            return Ok(Self::decision(*owned.route()));
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
                state: SessionRouteState::Owned(route),
                lease: reservation.lease,
                migration_lease: None,
                highest_sequence: sequence,
            },
        );
        Ok(Self::decision(route))
    }

    /// Freeze the session at `through` and stop admitting old-epoch prepares.
    ///
    /// Step 1 of the migration transaction. Nothing durable changes, so a crash
    /// here restores the committed owner. A session that is already `Fencing` or
    /// `Prepared` is refused: two concurrent migrations of one session would
    /// each believe they own the fence, so the second fails closed.
    pub fn begin_fence(
        &mut self,
        session: StableSessionKey,
        through: GlobalSequence,
    ) -> Result<(), PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let SessionRouteState::Owned(old) = owned.state else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        owned.state = SessionRouteState::Fencing { old, through };
        Ok(())
    }

    /// Stage a destination without granting it authority.
    ///
    /// Step 5. The returned route carries the *candidate* epoch; it becomes the
    /// committed epoch only when [`StickySessionPlacement::commit_owner`] runs
    /// after the route generation's CAS.
    pub fn stage_destination(
        &mut self,
        session: StableSessionKey,
        destination_cell: u32,
        migration_lease: BudgetLease,
    ) -> Result<SessionRoute, PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let SessionRouteState::Fencing { old, through } = owned.state else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let next_epoch = old
            .ownership_epoch
            .get()
            .checked_add(1)
            .ok_or_else(|| PlacementError::placement(PlacementFailureCode::TargetOverflow))?;
        let new = SessionRoute {
            route_id: destination_cell,
            destination_cell,
            ownership_epoch: SessionOwnershipEpoch::new(next_epoch),
        };
        owned.state = SessionRouteState::Prepared { old, new, through };
        owned.migration_lease = Some(migration_lease);
        Ok(new)
    }

    /// Adopt the staged owner in memory, immediately before the route
    /// generation's CAS.
    ///
    /// Step 6a. The checkpoint view taken *by* that CAS must already name the
    /// new owner, because the generation it publishes is exactly what a restart
    /// will read back. Both leases are still held: the transient double charge
    /// is not returned until the generation has committed and step 7 settles it.
    ///
    /// Returns `(old, new)` so a refused CAS can revert to precisely the owner
    /// it replaced rather than to a re-derived one.
    pub fn adopt_staged_owner(
        &mut self,
        session: StableSessionKey,
    ) -> Result<(SessionRoute, SessionRoute), PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let SessionRouteState::Prepared { old, new, .. } = owned.state else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        owned.state = SessionRouteState::Owned(new);
        Ok((old, new))
    }

    /// Undo an adoption whose route generation did not commit.
    ///
    /// Nothing durable named `new`, so the committed owner is still `old`. The
    /// migration lease is dropped here, which returns the transient charge.
    pub fn revert_adoption(
        &mut self,
        session: StableSessionKey,
        old: SessionRoute,
    ) -> Result<(), PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        owned.state = SessionRouteState::Owned(old);
        drop(owned.migration_lease.take());
        Ok(())
    }

    /// Settle the transient double charge after the route generation committed.
    ///
    /// Step 7. The old lease is dropped and the migration lease promoted in its
    /// place, so exactly one route charge is returned in the same operation that
    /// completes the migration.
    pub fn commit_owner(
        &mut self,
        session: StableSessionKey,
    ) -> Result<SessionOwnershipEpoch, PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let SessionRouteState::Owned(new) = owned.state else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        let Some(promoted) = owned.migration_lease.take() else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        // Assignment drops the old lease, returning exactly one route charge.
        owned.lease = promoted;
        Ok(new.ownership_epoch)
    }

    /// Revert an in-flight migration to its committed owner.
    ///
    /// Idempotent on an `Owned` session so an abort path can run unconditionally
    /// without first re-reading the state it is about to discard.
    pub fn abort_migration(&mut self, session: StableSessionKey) -> Result<(), PlacementError> {
        let Some(owned) = self.routes.get_mut(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        owned.state = SessionRouteState::Owned(*owned.state.authoritative());
        // Dropping the migration lease is what returns the transient charge and
        // wakes a `reserve_route` parked on it.
        drop(owned.migration_lease.take());
        Ok(())
    }

    /// Refuse an event stamped with a fenced ownership epoch.
    ///
    /// Strictly-less is stale. Equal is current. Greater is impossible: only the
    /// controller mints epochs, and only through a committed generation.
    pub fn admit_event_epoch(
        &self,
        session: StableSessionKey,
        epoch: SessionOwnershipEpoch,
    ) -> Result<(), PlacementError> {
        let Some(owned) = self.routes.get(&session) else {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        };
        if epoch < owned.route().ownership_epoch {
            return Err(PlacementError::placement(
                PlacementFailureCode::StaleOwnershipEpoch,
            ));
        }
        Ok(())
    }

    /// Whether a terminal receipt for `sequence` from the old cell is still
    /// admissible after the epoch increased.
    ///
    /// Keyed on the fence, never on the epoch. A receipt at or below the freeze
    /// point was committed at step 4 and is replayed idempotently; fencing it
    /// out on epoch alone would silently drop committed terminal work and hang
    /// the run on an unclosed active set.
    #[must_use]
    pub fn admits_fenced_terminal_receipt(
        &self,
        session: StableSessionKey,
        sequence: GlobalSequence,
    ) -> bool {
        self.routes
            .get(&session)
            .is_some_and(|owned| sequence <= owned.highest_sequence)
    }

    /// The complete route set at one barrier cut, in stable key order.
    #[must_use]
    pub fn checkpoint_route_entries(&self) -> Vec<CheckpointedSessionRoute> {
        self.routes
            .iter()
            .map(|(session, owned)| CheckpointedSessionRoute {
                session: *session,
                state: owned.state,
                highest_sequence: owned.highest_sequence,
            })
            .collect()
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
        // Retiring mid-migration would drop the committed lease and let a
        // waiting `reserve_route` install a route for a session that is about to
        // gain a new owner. The terminal is accepted; the retirement is not.
        if owned.state.is_migrating() {
            return Ok(());
        }
        if owned.route().ownership_epoch != ownership_epoch {
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
        self.restore_route_states(restored.into_iter().map(|(session, route, highest)| {
            CheckpointedSessionRoute {
                session,
                state: SessionRouteState::Owned(route),
                highest_sequence: highest,
            }
        }))
        .await
    }

    /// Reinstall a restored route set, collapsing every in-flight migration.
    ///
    /// A restored `Fencing` or `Prepared` means the process died before the
    /// route generation's CAS, so the committed owner is `old` and the state
    /// collapses to `Owned(old)`. The migration is simply not resumed; it may be
    /// requested again. No migration lease is ever restored, so a crashed
    /// migration cannot leak budget across a restart.
    pub async fn restore_route_states(
        &mut self,
        restored: impl IntoIterator<Item = CheckpointedSessionRoute>,
    ) -> Result<(), PlacementError> {
        let mut rebuilt = BTreeMap::new();
        for entry in restored {
            let lease = self
                .route_budget
                .acquire(1, ROUTE_ENTRY_BYTES)
                .await
                .map_err(map_budget_error)?;
            rebuilt.insert(
                entry.session,
                BudgetOwnedSessionRoute {
                    state: SessionRouteState::Owned(*entry.state.authoritative()),
                    lease,
                    migration_lease: None,
                    highest_sequence: entry.highest_sequence,
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
        if fence.ownership_epoch != entry.ownership_epoch {
            // A cell holds no epoch counter of its own and never infers one, so
            // both directions of inequality are the same refusal: the release
            // names an owner this staged action was not prepared under. The
            // entry stays staged so a corrected release can still arrive.
            return Err(PlacementError::placement(
                PlacementFailureCode::StaleOwnershipEpoch,
            ));
        }
        if entry.global_sequence != fence.global_sequence {
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

    /// Discard one staged, never-issued action during a migration drain.
    ///
    /// The explicit cancel path for step 3: an action that has been granted
    /// issue authority is *not* cancellable here, because the load it names may
    /// already be on the wire. Such an entry must be drained to terminal
    /// instead, which is why the refusal is distinguishable from "not staged".
    pub fn cancel_staged(
        &mut self,
        route_id: u32,
        action_id: StableActionId,
    ) -> Result<bool, PlacementError> {
        let key = (route_id, action_id);
        let Some(entry) = self.entries.get(&key) else {
            return Ok(false);
        };
        if entry.state != StagedState::Staged {
            return Err(PlacementError::placement(
                PlacementFailureCode::RouteUnavailable,
            ));
        }
        self.entries.remove(&key);
        Ok(true)
    }

    /// Lifecycle state of one tracked action.    #[must_use]
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
