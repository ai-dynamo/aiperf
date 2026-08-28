// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded multiplexed placement transfer between the controller and its cells.
//!
//! One [`PreparedCellularPlacementBinding`] is the single run-scoped,
//! bidirectional wire that carries the strict authenticated placement
//! vocabulary of [`crate::cellular::streaming_protocol`]: controller-to-cell
//! `BindContentSynthesisProfile` / `PrepareAction` / `ReleaseAction` pushes and
//! cell-to-controller ordered [`CellPlacementEvent`] returns.
//!
//! Three properties are load-bearing and are what this module exists to prove:
//!
//! - **Fixed owner count.** Per binding there is exactly one event driver and a
//!   constant number of owned tasks, whatever the route or in-flight action
//!   count. Growth lives in an indexed slab, never in tasks, channels, or
//!   drivers. No task is detached: [`BindingOwners`] holds every join handle and
//!   [`CellularPlacementDriver::drain`] joins rather than aborts.
//! - **Two windows, both item-and-byte.** A per-route unacknowledged-prepare
//!   window and a per-binding returned-event window, each accounted through the
//!   shared [`StreamingResourceBudget`]. Every wire-bearing value is non-`Clone`
//!   and carries its [`BudgetLease`], which releases on `Drop`.
//! - **Route-scoped failure.** A sequence gap or a conflicting duplicate fails
//!   the one route that produced it. Other destinations keep delivering, because
//!   a controller bug aimed at one cell must not stall the rest.
//!
//! A `ReleaseAction` rides inside the prepare window's own per-handle
//! reservation: the charge taken at prepare time covers the prepare frame plus
//! exactly one release frame. Without that co-reservation the plan's "a cell may
//! issue only on a valid `ReleaseAction`" invariant is unschedulable under
//! saturation, because the release that would free capacity would itself be
//! queued behind a full window.
//!
//! This module owns no serialization and no cryptography. It calls the sealing,
//! authentication, and bounded-decode entry points on
//! [`CellSecurityContext`](crate::engine::cellular_registration) and the
//! existing worker-signed admission ledger for the return direction, and it
//! introduces no admission purpose, sequence array, or replay window of its own.

// The transfer plane is complete and unit-tested here; the placement policy
// that selects routes and the result plane that consumes issued actions are the
// following cellular streaming tasks, so several entry points have no in-tree
// caller yet.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use bytes::Bytes;
use tokio::sync::{mpsc, watch};

use crate::cellular::streaming_protocol::{
    BindContentSynthesisProfileV1, CellPlacementEvent, ContentSynthesisProfileBoundReceipt,
    ControllerStreamingPurpose, ControllerStreamingSessionId, FrameBudgetReservation,
    PrepareAction, PreparedActionContent, ReleaseAction, STREAMING_CELLULAR_PROTOCOL_VERSION,
    StreamingCellularLimits,
};
use crate::engine::cellular_bootstrap::CellularRole;
use crate::engine::cellular_registration::{
    AdmissionPurpose, AdmissionRejection, CellRegistrationAuthority, CellRegistrationCredential,
    CellSecurityContext,
};
use crate::streaming::budget::{BudgetError, BudgetLease, StreamingResourceBudget};
use crate::streaming::failure::{ActionExecutionError, PlacementFailureCode};
use crate::streaming::identity::{
    ActionAttemptId, GlobalSequence, SessionOwnershipEpoch, StableActionId,
};
use crate::streaming::session::conversation::SessionStateVersion;

/// Authored per-binding bounds for the placement transfer window.
///
/// The prepare window and the event window are independent on purpose: a cell
/// must not be able to push unbounded events back while the prepare window sits
/// empty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CellularTransferLimits {
    /// Maximum unacknowledged prepare commands in flight per route.
    pub max_items: usize,
    /// Maximum unacknowledged prepare bytes in flight per route.
    pub max_bytes: usize,
    /// Maximum buffered inbound events across all routes.
    pub max_event_items: usize,
    /// Maximum buffered inbound event bytes across all routes.
    pub max_event_bytes: usize,
}

impl Default for CellularTransferLimits {
    fn default() -> Self {
        Self {
            max_items: 64,
            max_bytes: 8 * 1024 * 1024,
            max_event_items: 256,
            max_event_bytes: 8 * 1024 * 1024,
        }
    }
}

/// One controller-selected destination cell, resolved before binding.
#[derive(Clone, Debug)]
pub(crate) struct PreparedCellRoute {
    /// Stable route identifier stamped into every frame for this destination.
    pub route_id: u32,
    /// Destination cell ordinal; the [`CellularRole::Cell`] discriminant.
    pub destination_cell: u32,
    /// Registered Velo peer for the destination, from the registration ledger.
    pub peer: velo::PeerInfo,
}

impl PreparedCellRoute {
    /// Destination role this route addresses.
    fn destination(&self) -> CellularRole {
        CellularRole::Cell(self.destination_cell)
    }
}

/// Failure vocabulary for the cellular streaming transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CellularStreamingError {
    /// Frame authentication or bounded decode refused the frame.
    Admission(AdmissionRejection),
    /// The route observed a sequence gap or a conflicting duplicate.
    RouteFailed {
        /// Route whose dense sequence was violated.
        route_id: u32,
        /// Sequence at which the violation was observed.
        sequence: GlobalSequence,
    },
    /// A route never acknowledged the bound synthesis-profile digest, or
    /// acknowledged a different one.
    ProfileBindRefused {
        /// Route that refused or mismatched.
        route_id: u32,
    },
    /// No route carries the requested identifier.
    UnknownRoute {
        /// Requested route identifier.
        route_id: u32,
    },
    /// No slab entry carries the requested handle, or it is not preparable.
    UnknownHandle {
        /// Requested handle.
        handle: PlacementHandleId,
    },
    /// The transfer budget was closed or the request exceeds capacity.
    Budget(BudgetError),
    /// Transfer was cancelled by [`CellularPlacementControl`].
    Cancelled,
    /// Velo transport I/O.
    Transport(String),
}

impl fmt::Display for CellularStreamingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(rejection) => {
                write!(formatter, "streaming transfer refused: {rejection}")
            }
            Self::RouteFailed { route_id, sequence } => write!(
                formatter,
                "streaming route {route_id} failed at global sequence {}",
                sequence.get()
            ),
            Self::ProfileBindRefused { route_id } => write!(
                formatter,
                "streaming route {route_id} refused the content synthesis profile binding"
            ),
            Self::UnknownRoute { route_id } => {
                write!(formatter, "streaming route {route_id} is not bound")
            }
            Self::UnknownHandle { handle } => {
                write!(
                    formatter,
                    "streaming placement handle {} is not staged",
                    handle.get()
                )
            }
            Self::Budget(error) => write!(formatter, "streaming transfer budget: {error}"),
            Self::Cancelled => formatter.write_str("streaming transfer was cancelled"),
            Self::Transport(detail) => write!(formatter, "streaming transfer transport: {detail}"),
        }
    }
}

impl std::error::Error for CellularStreamingError {}

impl From<BudgetError> for CellularStreamingError {
    fn from(error: BudgetError) -> Self {
        Self::Budget(error)
    }
}

impl From<AdmissionRejection> for CellularStreamingError {
    fn from(rejection: AdmissionRejection) -> Self {
        Self::Admission(rejection)
    }
}

impl From<CellularStreamingError> for ActionExecutionError {
    fn from(error: CellularStreamingError) -> Self {
        let code = match error {
            CellularStreamingError::Admission(_)
            | CellularStreamingError::ProfileBindRefused { .. } => {
                PlacementFailureCode::DigestMismatch
            }
            CellularStreamingError::RouteFailed { .. }
            | CellularStreamingError::UnknownRoute { .. }
            | CellularStreamingError::UnknownHandle { .. }
            | CellularStreamingError::Transport(_) => PlacementFailureCode::RouteUnavailable,
            CellularStreamingError::Budget(_) | CellularStreamingError::Cancelled => {
                PlacementFailureCode::Cancelled
            }
        };
        Self::placement(code)
    }
}

/// Stable binding-local handle for one submitted placement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct PlacementHandleId(u64);

impl PlacementHandleId {
    /// Construct a handle from one bounded slab coordinate.
    #[must_use]
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying coordinate.
    #[must_use]
    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Controller-side acknowledgement that one action was accepted for transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlacementHandle {
    /// Binding-local handle identity.
    pub handle: PlacementHandleId,
    /// Route the action was placed on.
    pub route_id: u32,
    /// Stable global order of the placed action.
    pub global_sequence: GlobalSequence,
}

/// One controller-authored placement, already routed and content-synthesized.
///
/// Selecting the route and synthesizing the content are the placement policy's
/// job; this transfer plane only carries the decision it is handed.
#[derive(Clone, Debug)]
pub(crate) struct PlacementDecision {
    /// Destination route.
    pub route_id: u32,
    /// Stable action identity.
    pub action_id: StableActionId,
    /// Attempt identity within the action.
    pub attempt_id: ActionAttemptId,
    /// Stable global order.
    pub global_sequence: GlobalSequence,
    /// Fencing epoch of the owning session route.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Session state version the cell must already have incorporated.
    pub prior_session_state_version: SessionStateVersion,
    /// Immutable prepared content.
    pub content: PreparedActionContent,
}

/// One decoded, route-attributed event handed to the fused pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PlacementEvent {
    /// Route that produced the event.
    pub route_id: u32,
    /// Handle the event resolved to, when it names a staged action.
    pub handle: Option<PlacementHandleId>,
    /// The authenticated wire event.
    pub event: CellPlacementEvent,
}

/// Lifecycle of one slab entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlabState {
    /// The prepare frame was sent; no receipt has been accepted.
    Sent,
    /// The cell acknowledged the prepare; the prepare charge is released.
    Prepared,
    /// A release frame was sent; the cell may issue.
    Released,
    /// A terminal receipt was accepted; the entry is being reclaimed.
    Terminal,
}

/// One in-flight action's retained state and its budget charges.
///
/// The entry is the only owner of both leases, so reclaiming it is what returns
/// window capacity — there is no separate release path to forget.
struct SlabEntry {
    handle: PlacementHandleId,
    route_id: u32,
    action_id: StableActionId,
    attempt_id: ActionAttemptId,
    global_sequence: GlobalSequence,
    ownership_epoch: SessionOwnershipEpoch,
    /// BLAKE3 over the exact encoded payload. Identical retransmission is
    /// recognised by comparing this, never by comparing decoded structs.
    payload_digest: [u8; 32],
    state: SlabState,
    /// Prepare-frame charge, dropped when the prepare receipt is accepted.
    prepare_lease: Option<BudgetLease>,
    /// Co-reserved release-frame charge, consumed by `release`.
    release_reserve: Option<BudgetLease>,
}

/// Bounded indexed slab of in-flight actions keyed by [`PlacementHandleId`].
///
/// One allocation per binding: no per-action task, channel, or driver. Free
/// slots are recycled through an explicit free list so a long run's slab is
/// bounded by peak concurrency rather than by total action count.
#[derive(Default)]
struct PlacementActionSlab {
    entries: Vec<Option<SlabEntry>>,
    free: Vec<u32>,
    index: HashMap<(u32, StableActionId), PlacementHandleId>,
}

impl PlacementActionSlab {
    fn insert(&mut self, mut entry: SlabEntry) -> PlacementHandleId {
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                let slot = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
                self.entries.push(None);
                slot
            }
        };
        let handle = PlacementHandleId::new(u64::from(slot));
        entry.handle = handle;
        self.index.insert((entry.route_id, entry.action_id), handle);
        self.entries[slot as usize] = Some(entry);
        handle
    }

    fn get_mut(&mut self, handle: PlacementHandleId) -> Option<&mut SlabEntry> {
        let slot = usize::try_from(handle.get()).ok()?;
        self.entries.get_mut(slot)?.as_mut()
    }

    fn lookup(&self, route_id: u32, action_id: StableActionId) -> Option<PlacementHandleId> {
        self.index.get(&(route_id, action_id)).copied()
    }

    /// Reclaim one entry, dropping both leases and returning window capacity.
    fn remove(&mut self, handle: PlacementHandleId) -> Option<SlabEntry> {
        let slot = usize::try_from(handle.get()).ok()?;
        let entry = self.entries.get_mut(slot)?.take()?;
        self.index.remove(&(entry.route_id, entry.action_id));
        self.free.push(slot as u32);
        Some(entry)
    }

    /// Reclaim every entry on one route; used when the route fails.
    fn drain_route(&mut self, route_id: u32) -> Vec<SlabEntry> {
        let handles: Vec<PlacementHandleId> = self
            .entries
            .iter()
            .filter_map(|slot| slot.as_ref())
            .filter(|entry| entry.route_id == route_id)
            .map(|entry| entry.handle)
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| self.remove(handle))
            .collect()
    }

    fn live_count(&self) -> usize {
        self.entries.iter().filter(|slot| slot.is_some()).count()
    }
}

/// Outcome of admitting one sequenced frame on a route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SequenceAdmission {
    /// First observation of this sequence; process it.
    Fresh,
    /// Byte-identical retransmission; drop it and re-emit the prior receipt.
    IdempotentDuplicate,
}

/// Per-route dense sequence check.
///
/// The dense requirement is what makes the fixed replay window on the
/// authentication boundary defensible: a gap wider than the window would
/// otherwise be silently patched over. A gap or a conflicting duplicate fails
/// the route; a byte-identical duplicate is accepted and dropped.
///
/// Idempotence is keyed on the *payload* digest, not the frame bytes. A genuine
/// retransmission re-signs the same payload under a fresh frame sequence,
/// because the authentication boundary's replay window would reject reused
/// frame bytes before this check could ever run.
#[derive(Default)]
struct RouteSequencer {
    highest_accepted: Option<GlobalSequence>,
    last_digest: Option<[u8; 32]>,
    failed: bool,
}

impl RouteSequencer {
    fn admit(
        &mut self,
        route_id: u32,
        sequence: GlobalSequence,
        digest: [u8; 32],
    ) -> Result<SequenceAdmission, CellularStreamingError> {
        if self.failed {
            return Err(CellularStreamingError::RouteFailed { route_id, sequence });
        }
        match self.highest_accepted {
            None => {
                self.highest_accepted = Some(sequence);
                self.last_digest = Some(digest);
                Ok(SequenceAdmission::Fresh)
            }
            Some(highest) if sequence == highest => {
                if self.last_digest == Some(digest) {
                    Ok(SequenceAdmission::IdempotentDuplicate)
                } else {
                    self.failed = true;
                    Err(CellularStreamingError::RouteFailed { route_id, sequence })
                }
            }
            Some(highest) if sequence.get() == highest.get().saturating_add(1) => {
                self.highest_accepted = Some(sequence);
                self.last_digest = Some(digest);
                Ok(SequenceAdmission::Fresh)
            }
            Some(_) => {
                self.failed = true;
                Err(CellularStreamingError::RouteFailed { route_id, sequence })
            }
        }
    }

    const fn is_failed(&self) -> bool {
        self.failed
    }
}

/// Test-visible counters proving the fixed owner-count invariant.
#[derive(Clone, Debug, Default)]
pub(crate) struct BindingDiagnostics {
    driver_count: Rc<Cell<usize>>,
    owner_task_count: Rc<Cell<usize>>,
}

impl BindingDiagnostics {
    /// Event drivers alive for this binding; exactly one while it is open.
    #[must_use]
    pub(crate) fn driver_count(&self) -> usize {
        self.driver_count.get()
    }

    /// Owned, joinable tasks alive for this binding.
    #[must_use]
    pub(crate) fn owner_task_count(&self) -> usize {
        self.owner_task_count.get()
    }

    fn add_driver(&self) {
        self.driver_count.set(self.driver_count.get() + 1);
    }

    fn drop_driver(&self) {
        self.driver_count
            .set(self.driver_count.get().saturating_sub(1));
    }

    fn add_owner(&self) {
        self.owner_task_count.set(self.owner_task_count.get() + 1);
    }

    fn drop_owner(&self) {
        self.owner_task_count
            .set(self.owner_task_count.get().saturating_sub(1));
    }
}

/// Controller-to-cell command egress.
///
/// Abstracted so the transfer logic is exercised without a socket, and so the
/// Velo binding is the only place that knows about peers and handler names.
#[async_trait::async_trait(?Send)]
pub(crate) trait PlacementCommandSink {
    /// Deliver one sealed controller frame to an exact destination peer.
    async fn send_command(
        &self,
        peer: &velo::PeerInfo,
        frame: Bytes,
    ) -> Result<(), CellularStreamingError>;
}

/// Cell-to-controller event egress.
#[async_trait::async_trait(?Send)]
pub(crate) trait PlacementEventSink {
    /// Deliver one worker-signed event frame to the controller.
    async fn send_event(&self, frame: Bytes) -> Result<(), CellularStreamingError>;
}

/// One inbound controller frame owning its receive permit.
///
/// Non-`Clone`: the permit moves with the payload and releases only on `Drop`.
pub(crate) struct BudgetOwnedControllerFrame {
    bytes: Bytes,
    lease: BudgetLease,
}

impl BudgetOwnedControllerFrame {
    /// Bind inbound bytes to the receive permit charged for them.
    pub(crate) fn new(bytes: Bytes, lease: BudgetLease) -> Self {
        Self { bytes, lease }
    }

    /// Split the frame from its permit.
    pub(crate) fn into_parts(self) -> (Bytes, BudgetLease) {
        (self.bytes, self.lease)
    }
}

/// One decoded cell event owning its event-window permit.
struct BudgetOwnedCellEvent {
    route_id: u32,
    event: CellPlacementEvent,
    lease: BudgetLease,
}

/// Owned, joinable task handles and their diagnostics counter.
///
/// Every task this binding starts is registered here, which is what makes
/// [`CellularPlacementDriver::drain`] a join rather than an abort.
#[derive(Default)]
pub(crate) struct BindingOwners {
    handles: Vec<tokio::task::JoinHandle<()>>,
    diagnostics: BindingDiagnostics,
}

impl BindingOwners {
    fn adopt(&mut self, handle: tokio::task::JoinHandle<()>) {
        self.diagnostics.add_owner();
        self.handles.push(handle);
    }

    async fn join(&mut self) {
        for handle in self.handles.drain(..) {
            // A panicking owner is reported by the join result; the binding
            // still finishes draining so the pipeline's active set closes.
            if let Err(error) = handle.await {
                tracing::debug!(error = %error, component = "cellular_streaming", "binding owner task ended abnormally");
            }
            self.diagnostics.drop_owner();
        }
    }
}

/// One route's egress state on the controller side.
struct RouteSender {
    route: PreparedCellRoute,
    sink: Rc<dyn PlacementCommandSink>,
    sequencer: RouteSequencer,
    bound_profile_digest: Option<[u8; 32]>,
}

/// Controller-side command submitter.
///
/// Holds the route table and the shared action slab; holds no borrow of the
/// placement policy, so the fused pipeline can keep a `&mut` submitter while it
/// polls the driver in the same `select!`.
pub(crate) struct CellularPlacementSubmitter {
    routes: Box<[RouteSender]>,
    slab: Rc<RefCell<PlacementActionSlab>>,
    security: Rc<CellSecurityContext>,
    session: ControllerStreamingSessionId,
    plan_digest: [u8; 32],
    synthesis_profile_digest: Option<[u8; 32]>,
    budget: StreamingResourceBudget,
    cellular_limits: StreamingCellularLimits,
    stop_preparing: Rc<Cell<bool>>,
}

impl CellularPlacementSubmitter {
    /// Transfer one prepared action to its selected route, blocking on the
    /// route's prepare window when it is full.
    ///
    /// The returned future is `Pending` — never an error and never a timer wake
    /// — while the window is saturated. It is woken by a receipt releasing a
    /// lease or by [`CellularPlacementControl::cancel_pending`] closing the
    /// budget.
    pub(crate) async fn prepare(
        &mut self,
        decision: PlacementDecision,
    ) -> Result<PlacementHandle, CellularStreamingError> {
        if self.stop_preparing.get() {
            return Err(CellularStreamingError::Cancelled);
        }
        let route_index = self.route_index(decision.route_id)?;
        if self.routes[route_index].sequencer.is_failed() {
            return Err(CellularStreamingError::RouteFailed {
                route_id: decision.route_id,
                sequence: decision.global_sequence,
            });
        }

        let action = PrepareAction {
            version: STREAMING_CELLULAR_PROTOCOL_VERSION,
            plan_digest: self.plan_digest,
            synthesis_profile_digest: self.synthesis_profile_digest,
            route_id: decision.route_id,
            destination_cell: self.routes[route_index].route.destination_cell,
            action_id: decision.action_id,
            attempt_id: decision.attempt_id,
            global_sequence: decision.global_sequence,
            ownership_epoch: decision.ownership_epoch,
            prior_session_state_version: decision.prior_session_state_version,
            content: decision.content,
        };
        let payload_digest = *blake3::hash(&action.content.digest).as_bytes();

        // One acquisition covers the prepare frame and the release frame that
        // will eventually free this handle. The `.await` holds no slab borrow.
        let ceiling = self.cellular_limits.max_frame_bytes;
        let mut charge = self.budget.acquire(2, ceiling.saturating_mul(2)).await?;
        let prepare_charge = charge.split_off(1, ceiling)?;
        let reservation = FrameBudgetReservation::new(prepare_charge, ceiling)?;

        let route = &self.routes[route_index];
        let frame = self
            .security
            .seal_streaming_to_cell(
                ControllerStreamingPurpose::PrepareAction,
                route.route.destination(),
                self.session,
                &route.route.peer,
                &action,
                reservation,
            )
            .map_err(|error| CellularStreamingError::Transport(error.to_string()))?;
        let (bytes, prepare_lease) = frame.into_parts();
        route.sink.send_command(&route.route.peer, bytes).await?;

        let handle = self.slab.borrow_mut().insert(SlabEntry {
            handle: PlacementHandleId::new(0),
            route_id: decision.route_id,
            action_id: decision.action_id,
            attempt_id: decision.attempt_id,
            global_sequence: decision.global_sequence,
            ownership_epoch: decision.ownership_epoch,
            payload_digest,
            state: SlabState::Sent,
            prepare_lease: Some(prepare_lease),
            release_reserve: Some(charge),
        });
        Ok(PlacementHandle {
            handle,
            route_id: decision.route_id,
            global_sequence: decision.global_sequence,
        })
    }

    /// Grant issue authority for one already-transferred action.
    ///
    /// This never waits on the prepare window: the release frame's charge was
    /// co-reserved at prepare time, so a saturated window cannot deadlock the
    /// only operation that can drain it.
    pub(crate) async fn release(
        &mut self,
        handle: PlacementHandleId,
    ) -> Result<(), CellularStreamingError> {
        let (route_id, action_id, global_sequence, ownership_epoch, reserve) = {
            let mut slab = self.slab.borrow_mut();
            let entry = slab
                .get_mut(handle)
                .ok_or(CellularStreamingError::UnknownHandle { handle })?;
            let reserve = entry
                .release_reserve
                .take()
                .ok_or(CellularStreamingError::UnknownHandle { handle })?;
            entry.state = SlabState::Released;
            (
                entry.route_id,
                entry.action_id,
                entry.global_sequence,
                entry.ownership_epoch,
                reserve,
            )
        };

        let route_index = self.route_index(route_id)?;
        let release = ReleaseAction {
            version: STREAMING_CELLULAR_PROTOCOL_VERSION,
            plan_digest: self.plan_digest,
            route_id,
            action_id,
            global_sequence,
            ownership_epoch,
        };
        let reservation =
            FrameBudgetReservation::new(reserve, self.cellular_limits.max_frame_bytes)?;
        let route = &self.routes[route_index];
        let frame = self
            .security
            .seal_streaming_to_cell(
                ControllerStreamingPurpose::ReleaseAction,
                route.route.destination(),
                self.session,
                &route.route.peer,
                &release,
                reservation,
            )
            .map_err(|error| CellularStreamingError::Transport(error.to_string()))?;
        let (bytes, lease) = frame.into_parts();
        route.sink.send_command(&route.route.peer, bytes).await?;
        // The release frame's charge is held until the wire hand-off completes,
        // then returned; the handle's terminal receipt reclaims the entry.
        drop(lease);
        Ok(())
    }

    fn route_index(&self, route_id: u32) -> Result<usize, CellularStreamingError> {
        self.routes
            .iter()
            .position(|route| route.route.route_id == route_id)
            .ok_or(CellularStreamingError::UnknownRoute { route_id })
    }

    /// Number of actions currently retained in the binding's slab.
    #[must_use]
    pub(crate) fn in_flight(&self) -> usize {
        self.slab.borrow().live_count()
    }
}

/// Controller-side ordered event driver.
///
/// Exactly one exists per binding and it owns the binding's joinable tasks.
/// The controller spawns nothing for the receive path itself: `next_event` is
/// polled directly by the fused pipeline's `select!`, which is what lets
/// `drain` join owners instead of aborting them.
pub(crate) struct CellularPlacementDriver {
    events: mpsc::Receiver<BudgetOwnedCellEvent>,
    owners: BindingOwners,
    slab: Rc<RefCell<PlacementActionSlab>>,
    cancel: watch::Receiver<bool>,
    diagnostics: BindingDiagnostics,
}

impl CellularPlacementDriver {
    /// Await the next authenticated placement event.
    pub(crate) async fn next_event(&mut self) -> Result<PlacementEvent, CellularStreamingError> {
        loop {
            if *self.cancel.borrow() {
                return Err(CellularStreamingError::Cancelled);
            }
            tokio::select! {
                biased;
                changed = self.cancel.changed() => {
                    if changed.is_err() || *self.cancel.borrow() {
                        return Err(CellularStreamingError::Cancelled);
                    }
                }
                received = self.events.recv() => {
                    let Some(event) = received else {
                        return Err(CellularStreamingError::Cancelled);
                    };
                    return self.apply(event);
                }
            }
        }
    }

    /// Apply one event's effect on retained state and release its charges.
    ///
    /// The event-window permit is released here, as the decoded event is handed
    /// to the pipeline; the prepare-window permit is released by the receipt
    /// that terminates its handle.
    fn apply(
        &mut self,
        owned: BudgetOwnedCellEvent,
    ) -> Result<PlacementEvent, CellularStreamingError> {
        let BudgetOwnedCellEvent {
            route_id,
            event,
            lease,
        } = owned;
        let handle = {
            let mut slab = self.slab.borrow_mut();
            match &event {
                CellPlacementEvent::Prepared { receipt } => {
                    let handle = slab.lookup(route_id, receipt.action_id);
                    if let Some(handle) = handle
                        && let Some(entry) = slab.get_mut(handle)
                    {
                        if entry.state == SlabState::Sent {
                            entry.state = SlabState::Prepared;
                        }
                        // Releasing the prepare charge is what wakes a
                        // `prepare` parked on a saturated window.
                        drop(entry.prepare_lease.take());
                    }
                    handle
                }
                CellPlacementEvent::Released { receipt } => {
                    let handle = slab.lookup(route_id, receipt.action_id);
                    if let Some(handle) = handle {
                        slab.remove(handle);
                    }
                    handle
                }
                CellPlacementEvent::Failed { receipt } => {
                    let handle = slab.lookup(route_id, receipt.action_id);
                    if let Some(handle) = handle {
                        slab.remove(handle);
                    }
                    handle
                }
                CellPlacementEvent::Action { .. }
                | CellPlacementEvent::ContentSynthesisProfileBound { .. } => None,
            }
        };
        drop(lease);
        Ok(PlacementEvent {
            route_id,
            handle,
            event,
        })
    }

    /// Fail one route, reclaiming every handle it still owns.
    ///
    /// Failure is route-scoped: a controller bug aimed at one destination must
    /// not stall the others.
    pub(crate) fn fail_route(&mut self, route_id: u32) -> Vec<PlacementHandleId> {
        let drained = self.slab.borrow_mut().drain_route(route_id);
        drained.into_iter().map(|entry| entry.handle).collect()
    }

    /// Join every owned task after preparation has stopped.
    pub(crate) async fn drain(&mut self) -> Result<(), CellularStreamingError> {
        self.owners.join().await;
        Ok(())
    }

    /// Counters proving the fixed owner-count invariant.
    #[must_use]
    pub(crate) fn diagnostics(&self) -> BindingDiagnostics {
        self.diagnostics.clone()
    }
}

impl Drop for CellularPlacementDriver {
    fn drop(&mut self) {
        self.diagnostics.drop_driver();
    }
}

/// Independently borrowable stop and cancel control.
#[derive(Clone)]
pub(crate) struct CellularPlacementControl {
    stop_preparing: Rc<Cell<bool>>,
    cancel: watch::Sender<bool>,
    budget: StreamingResourceBudget,
}

impl CellularPlacementControl {
    /// Refuse further preparation while letting outstanding handles terminate.
    pub(crate) fn stop_preparing(&self) {
        self.stop_preparing.set(true);
    }

    /// Wake a parked driver and every pending budget acquisition.
    ///
    /// Closing the budget is what unblocks the acquisitions; the watch send is
    /// what unblocks a `next_event` parked on an empty channel.
    pub(crate) fn cancel_pending(&self) {
        self.stop_preparing.set(true);
        self.cancel.send_replace(true);
        self.budget.close();
    }
}

/// Cell-side staging and issue endpoint.
///
/// This is the transfer half only: it authenticates, bounds, sequences, and
/// stages. Fencing an accepted prepare against session ownership, and the issue
/// path itself, belong to the placement policy that sits above it.
pub(crate) struct CellularExecutionEndpoint {
    inbound: mpsc::Receiver<BudgetOwnedControllerFrame>,
    events: Rc<dyn PlacementEventSink>,
    credential: Rc<CellRegistrationCredential>,
    security: Rc<CellSecurityContext>,
    peer: velo::PeerInfo,
    controller_peer: velo::PeerInfo,
    destination: CellularRole,
    sequencer: RouteSequencer,
    staged: HashMap<(u32, StableActionId), StagedAction>,
    limits: StreamingCellularLimits,
    bound_profile_digest: Option<[u8; 32]>,
}

/// One action the cell has staged but not yet been granted authority to issue.
struct StagedAction {
    attempt_id: ActionAttemptId,
    global_sequence: GlobalSequence,
    content_digest: [u8; 32],
    issued: bool,
    lease: BudgetLease,
}

impl CellularExecutionEndpoint {
    /// Authenticate, bound-decode, sequence, and stage one controller frame.
    ///
    /// Returns the event the cell owes the controller, or `None` when the frame
    /// was a byte-identical retransmission whose receipt was already sent.
    pub(crate) fn accept_prepare(
        &mut self,
        frame: BudgetOwnedControllerFrame,
    ) -> Result<Option<CellPlacementEvent>, CellularStreamingError> {
        let (bytes, lease) = frame.into_parts();
        let payload = self.security.authenticate_streaming_from_controller(
            ControllerStreamingPurpose::PrepareAction,
            self.destination,
            &self.controller_peer,
            crate::cellular::streaming_protocol::BudgetOwnedFrame::new(bytes, lease),
            self.limits,
        )?;
        let owned = self.security.decode_prepare_action(payload, self.limits)?;
        let (action, lease) = owned.into_parts();

        if action.synthesis_profile_digest != self.bound_profile_digest {
            return Err(CellularStreamingError::ProfileBindRefused {
                route_id: action.route_id,
            });
        }
        let payload_digest = *blake3::hash(&action.content.digest).as_bytes();
        match self
            .sequencer
            .admit(action.route_id, action.global_sequence, payload_digest)?
        {
            SequenceAdmission::IdempotentDuplicate => {
                // Never re-stage, re-charge, or re-issue; the controller
                // already holds the receipt this frame would produce.
                drop(lease);
                Ok(None)
            }
            SequenceAdmission::Fresh => {
                let receipt = crate::cellular::streaming_protocol::PlacementPreparedReceipt {
                    route_id: action.route_id,
                    action_id: action.action_id,
                    global_sequence: action.global_sequence,
                    content_digest: action.content.digest,
                };
                self.staged.insert(
                    (action.route_id, action.action_id),
                    StagedAction {
                        attempt_id: action.attempt_id,
                        global_sequence: action.global_sequence,
                        content_digest: action.content.digest,
                        issued: false,
                        lease,
                    },
                );
                Ok(Some(CellPlacementEvent::Prepared { receipt }))
            }
        }
    }

    /// Authenticate one release frame and grant issue authority.
    pub(crate) fn accept_release(
        &mut self,
        frame: BudgetOwnedControllerFrame,
    ) -> Result<CellPlacementEvent, CellularStreamingError> {
        let (bytes, lease) = frame.into_parts();
        let payload = self.security.authenticate_streaming_from_controller(
            ControllerStreamingPurpose::ReleaseAction,
            self.destination,
            &self.controller_peer,
            crate::cellular::streaming_protocol::BudgetOwnedFrame::new(bytes, lease),
            self.limits,
        )?;
        let release: ReleaseAction = rmp_serde::from_slice(payload.as_slice())
            .map_err(|_| CellularStreamingError::Admission(AdmissionRejection::Malformed))?;
        drop(payload);

        let staged = self
            .staged
            .get_mut(&(release.route_id, release.action_id))
            .ok_or(CellularStreamingError::RouteFailed {
                route_id: release.route_id,
                sequence: release.global_sequence,
            })?;
        if staged.global_sequence != release.global_sequence || staged.issued {
            return Err(CellularStreamingError::RouteFailed {
                route_id: release.route_id,
                sequence: release.global_sequence,
            });
        }
        staged.issued = true;
        Ok(CellPlacementEvent::Released {
            receipt: crate::cellular::streaming_protocol::PlacementReleasedReceipt {
                route_id: release.route_id,
                action_id: release.action_id,
                global_sequence: release.global_sequence,
            },
        })
    }

    /// Seal and send one placement event back to the controller.
    pub(crate) async fn emit(
        &self,
        event: &CellPlacementEvent,
    ) -> Result<(), CellularStreamingError> {
        let sealed = self
            .credential
            .seal_payload(AdmissionPurpose::StreamingPlacementEvent, &self.peer, event)
            .map_err(|error| CellularStreamingError::Transport(error.to_string()))?;
        self.events.send_event(Bytes::from(sealed)).await
    }

    /// Receive the next authenticated controller frame, or `None` on close.
    pub(crate) async fn next_frame(&mut self) -> Option<BudgetOwnedControllerFrame> {
        self.inbound.recv().await
    }

    /// Number of actions staged and not yet reclaimed.
    #[must_use]
    pub(crate) fn staged_count(&self) -> usize {
        self.staged.len()
    }
}

/// Controller-side inbound event admission.
///
/// Authenticates before decoding, charges the event window, and hands the
/// result to the binding's single bounded driver channel. The whole inbound
/// path is this one function plus a `try_send`: the Velo handler that calls it
/// owns no per-action state.
pub(crate) async fn admit_placement_event(
    authority: &CellRegistrationAuthority,
    budget: &StreamingResourceBudget,
    limits: StreamingCellularLimits,
    route_id: u32,
    destination_cell: u32,
    frame: Bytes,
    sender: &mpsc::Sender<BudgetOwnedCellEvent>,
) -> Result<(), CellularStreamingError> {
    if frame.len() > limits.max_frame_bytes {
        return Err(CellularStreamingError::Admission(
            AdmissionRejection::Oversized,
        ));
    }
    let lease = budget.acquire(1, frame.len()).await?;
    let opened = authority
        .open_payload::<CellPlacementEvent>(AdmissionPurpose::StreamingPlacementEvent, &frame)?;
    if opened.role() != CellularRole::Cell(destination_cell) {
        return Err(CellularStreamingError::Admission(AdmissionRejection::Role));
    }
    let event = opened.into_payload();
    sender
        .try_send(BudgetOwnedCellEvent {
            route_id,
            event,
            lease,
        })
        .map_err(|_| CellularStreamingError::Transport("event window is full".to_owned()))
}

/// One run-scoped multiplexed placement transfer with separately borrowable
/// handles.
pub(crate) struct PreparedCellularPlacementBinding {
    /// Controller-side command submitter.
    pub submitter: CellularPlacementSubmitter,
    /// Controller-side ordered event driver — exactly one per binding.
    pub driver: CellularPlacementDriver,
    /// Cancellation and stop control, independently borrowable.
    pub control: CellularPlacementControl,
    /// Bound content-synthesis profile digest agreed by every route.
    pub bound_synthesis_profile_digest: Option<[u8; 32]>,
}

/// Everything the controller must already have proven before a binding exists.
pub(crate) struct CellularBindingContext {
    /// Controller process security context that seals every command.
    pub security: Rc<CellSecurityContext>,
    /// Pinned controller streaming session, derived from the proven binding.
    pub session: ControllerStreamingSessionId,
    /// Capability plan digest both peers independently recomputed.
    pub plan_digest: [u8; 32],
    /// Shared transfer accounting.
    pub budget: StreamingResourceBudget,
    /// Per-binding window bounds.
    pub limits: CellularTransferLimits,
    /// Per-frame and per-payload capacity bounds.
    pub cellular_limits: StreamingCellularLimits,
}

/// Prepare one multiplexed binding across every selected route.
///
/// When a synthesis profile is authored this sends the exact bound digest to
/// every route before returning, and stamps that digest into every subsequent
/// `PrepareAction`, so a cell that resolved a different profile refuses the
/// first action it is handed rather than synthesizing against the wrong one.
/// The acknowledgements themselves arrive on the returned event sender, which
/// the caller has not yet connected to a transport at this point, so matching
/// them against the route table belongs to the caller that owns the wire.
pub(crate) async fn prepare_cellular_placement_binding(
    routes: Box<[PreparedCellRoute]>,
    sinks: Box<[Rc<dyn PlacementCommandSink>]>,
    context: CellularBindingContext,
    synthesis_profile: Option<BindContentSynthesisProfileV1>,
) -> Result<
    (
        PreparedCellularPlacementBinding,
        mpsc::Sender<BudgetOwnedCellEvent>,
    ),
    CellularStreamingError,
> {
    if routes.len() != sinks.len() {
        return Err(CellularStreamingError::Transport(
            "route and sink tables disagree".to_owned(),
        ));
    }
    let bound_synthesis_profile_digest = synthesis_profile
        .as_ref()
        .map(|profile| profile.bound_profile_digest);

    let mut senders = Vec::with_capacity(routes.len());
    for (route, sink) in routes.into_vec().into_iter().zip(sinks.into_vec()) {
        if let Some(profile) = synthesis_profile.as_ref() {
            let ceiling = context.cellular_limits.max_frame_bytes;
            let lease = context.budget.acquire(1, ceiling).await?;
            let reservation = FrameBudgetReservation::new(lease, ceiling)?;
            let frame = context
                .security
                .seal_streaming_to_cell(
                    ControllerStreamingPurpose::BindContentSynthesisProfile,
                    route.destination(),
                    context.session,
                    &route.peer,
                    profile,
                    reservation,
                )
                .map_err(|error| CellularStreamingError::Transport(error.to_string()))?;
            let (bytes, lease) = frame.into_parts();
            sink.send_command(&route.peer, bytes).await?;
            drop(lease);
        }
        senders.push(RouteSender {
            route,
            sink,
            sequencer: RouteSequencer::default(),
            bound_profile_digest: bound_synthesis_profile_digest,
        });
    }

    let slab = Rc::new(RefCell::new(PlacementActionSlab::default()));
    let stop_preparing = Rc::new(Cell::new(false));
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let (event_tx, event_rx) = mpsc::channel(context.limits.max_event_items.max(1));
    let diagnostics = BindingDiagnostics::default();
    diagnostics.add_driver();

    let submitter = CellularPlacementSubmitter {
        routes: senders.into_boxed_slice(),
        slab: Rc::clone(&slab),
        security: Rc::clone(&context.security),
        session: context.session,
        plan_digest: context.plan_digest,
        synthesis_profile_digest: bound_synthesis_profile_digest,
        budget: context.budget.clone(),
        cellular_limits: context.cellular_limits,
        stop_preparing: Rc::clone(&stop_preparing),
    };
    let driver = CellularPlacementDriver {
        events: event_rx,
        owners: BindingOwners {
            handles: Vec::new(),
            diagnostics: diagnostics.clone(),
        },
        slab,
        cancel: cancel_rx,
        diagnostics: diagnostics.clone(),
    };
    let control = CellularPlacementControl {
        stop_preparing,
        cancel: cancel_tx,
        budget: context.budget,
    };

    Ok((
        PreparedCellularPlacementBinding {
            submitter,
            driver,
            control,
            bound_synthesis_profile_digest,
        },
        event_tx,
    ))
}

/// Acknowledge one bound synthesis profile from the cell side.
#[must_use]
pub(crate) fn profile_bound_event(
    plan_digest: [u8; 32],
    profile: &BindContentSynthesisProfileV1,
) -> CellPlacementEvent {
    CellPlacementEvent::ContentSynthesisProfileBound {
        receipt: ContentSynthesisProfileBoundReceipt {
            version: STREAMING_CELLULAR_PROTOCOL_VERSION,
            plan_digest,
            authored_profile_digest: profile.authored_profile_digest,
            bound_profile_digest: profile.bound_profile_digest,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::budget::BudgetLimits;

    fn sequence(value: u64) -> GlobalSequence {
        GlobalSequence::new(value)
    }

    #[test]
    fn dense_sequences_admit_and_identical_retransmit_is_idempotent() {
        let mut sequencer = RouteSequencer::default();
        assert_eq!(
            sequencer.admit(7, sequence(0), [1; 32]).unwrap(),
            SequenceAdmission::Fresh
        );
        assert_eq!(
            sequencer.admit(7, sequence(1), [2; 32]).unwrap(),
            SequenceAdmission::Fresh
        );
        // A retransmission re-signs the same payload under a fresh frame
        // sequence, so idempotence is keyed on the payload digest.
        assert_eq!(
            sequencer.admit(7, sequence(1), [2; 32]).unwrap(),
            SequenceAdmission::IdempotentDuplicate
        );
        assert!(!sequencer.is_failed());
    }

    #[test]
    fn conflicting_duplicate_and_gap_both_fail_the_route() {
        let mut conflicting = RouteSequencer::default();
        conflicting.admit(3, sequence(0), [1; 32]).unwrap();
        assert_eq!(
            conflicting.admit(3, sequence(0), [9; 32]),
            Err(CellularStreamingError::RouteFailed {
                route_id: 3,
                sequence: sequence(0)
            })
        );
        assert!(conflicting.is_failed());

        let mut gapped = RouteSequencer::default();
        gapped.admit(3, sequence(0), [1; 32]).unwrap();
        assert!(gapped.admit(3, sequence(4), [2; 32]).is_err());
        assert!(gapped.is_failed());
    }

    #[test]
    fn slab_recycles_slots_and_reclaims_route_scoped_entries() {
        let mut slab = PlacementActionSlab::default();
        let budget = StreamingResourceBudget::new(crate::streaming::budget::BudgetLimits {
            max_items: 16,
            max_bytes: 4096,
        })
        .unwrap();
        let mut insert = |slab: &mut PlacementActionSlab, route_id: u32, action: u8| {
            slab.insert(SlabEntry {
                handle: PlacementHandleId::new(0),
                route_id,
                action_id: StableActionId::from_bytes([action; 32]),
                attempt_id: ActionAttemptId::from_bytes([action; 32]),
                global_sequence: sequence(u64::from(action)),
                ownership_epoch: SessionOwnershipEpoch::new(0),
                payload_digest: [action; 32],
                state: SlabState::Sent,
                prepare_lease: budget.try_acquire(1, 8).ok(),
                release_reserve: budget.try_acquire(1, 8).ok(),
            })
        };
        let first = insert(&mut slab, 1, 1);
        let _second = insert(&mut slab, 1, 2);
        let other = insert(&mut slab, 2, 3);
        assert_eq!(slab.live_count(), 3);

        slab.remove(first).unwrap();
        assert_eq!(slab.free.len(), 1);
        assert_eq!(slab.drain_route(1).len(), 1);
        // Route-scoped failure leaves other destinations untouched.
        assert_eq!(slab.live_count(), 1);
        assert!(slab.get_mut(other).is_some());
        assert_eq!(budget.snapshot().used_items, 2);
    }

    const FRAME_CEILING: usize = 4096;

    const CELLULAR_LIMITS: StreamingCellularLimits = StreamingCellularLimits {
        max_frame_bytes: FRAME_CEILING,
        max_payload_bytes: 2048,
        max_content_items: 4,
        max_content_bytes: 1024,
    };

    /// Controller-to-cell egress that hands frames to an in-process channel.
    ///
    /// The transfer logic is what is under test, so the wire is a channel; the
    /// security boundary on both ends is the real one, because "authenticate
    /// before decode" is part of what these tests assert.
    struct LoopbackCommandSink {
        inbound: mpsc::Sender<BudgetOwnedControllerFrame>,
        budget: StreamingResourceBudget,
    }

    #[async_trait::async_trait(?Send)]
    impl PlacementCommandSink for LoopbackCommandSink {
        async fn send_command(
            &self,
            _peer: &velo::PeerInfo,
            frame: Bytes,
        ) -> Result<(), CellularStreamingError> {
            let lease = self.budget.acquire(1, frame.len().max(1)).await?;
            self.inbound
                .send(BudgetOwnedControllerFrame::new(frame, lease))
                .await
                .map_err(|_| CellularStreamingError::Transport("cell inbound closed".to_owned()))
        }
    }

    /// Cell-to-controller egress that retains sealed frames for the test to
    /// feed back through the real controller admission path.
    #[derive(Default)]
    struct CollectingEventSink {
        frames: RefCell<Vec<Bytes>>,
    }

    #[async_trait::async_trait(?Send)]
    impl PlacementEventSink for CollectingEventSink {
        async fn send_event(&self, frame: Bytes) -> Result<(), CellularStreamingError> {
            self.frames.borrow_mut().push(frame);
            Ok(())
        }
    }

    struct Fixture {
        binding: PreparedCellularPlacementBinding,
        events_tx: mpsc::Sender<BudgetOwnedCellEvent>,
        endpoints: Vec<CellularExecutionEndpoint>,
        authority: CellRegistrationAuthority,
        emitted: Rc<CollectingEventSink>,
        transfer_budget: StreamingResourceBudget,
        event_budget: StreamingResourceBudget,
    }

    fn content(byte_length: usize) -> PreparedActionContent {
        let mut content = PreparedActionContent {
            schema: crate::streaming::action::DatasetActionSchema::new("aiperf.stream.action.v1"),
            canonical_request: vec![0x7B; byte_length],
            content_leases: Vec::new(),
            item_count: 0,
            byte_length: byte_length as u64,
            digest: [0; 32],
        };
        content.digest = content.compute_digest();
        content
    }

    fn decision(route_id: u32, global_sequence: u64, action: u8) -> PlacementDecision {
        PlacementDecision {
            route_id,
            action_id: StableActionId::from_bytes([action; 32]),
            attempt_id: ActionAttemptId::from_bytes([action; 32]),
            global_sequence: sequence(global_sequence),
            ownership_epoch: SessionOwnershipEpoch::new(1),
            prior_session_state_version: SessionStateVersion::INITIAL,
            content: content(64),
        }
    }

    /// One binding with `route_count` routes, each addressing its own cell.
    async fn fixture(route_count: u32, transfer: BudgetLimits) -> Fixture {
        let (authority, controller_sealer, credentials, cell_inbound) =
            CellRegistrationAuthority::mint_streaming_security(route_count).expect("security");
        let session = ControllerStreamingSessionId::from_bytes([0x5A; 32]);
        let transfer_budget = StreamingResourceBudget::new(transfer).expect("transfer budget");
        let inbound_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 256,
            max_bytes: 256 * FRAME_CEILING,
        })
        .expect("inbound budget");
        let event_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 64,
            max_bytes: 64 * FRAME_CEILING,
        })
        .expect("event budget");
        let emitted = Rc::new(CollectingEventSink::default());

        let mut routes = Vec::new();
        let mut sinks: Vec<Rc<dyn PlacementCommandSink>> = Vec::new();
        let mut endpoints = Vec::new();
        for (index, (credential, inbound_security)) in
            credentials.into_iter().zip(cell_inbound).enumerate()
        {
            let cell_id = index as u32;
            // Seal and authenticate must name the same peer: the frame binds the
            // peer it travels over, not the signer's own address.
            let peer = velo::PeerInfo::new(
                velo::InstanceId::new_v4(),
                velo::WorkerAddress::from_encoded(vec![0xC0, cell_id as u8]),
            );
            inbound_security
                .install_controller_streaming_session(session)
                .expect("pin the controller streaming session");
            let (frame_tx, frame_rx) = mpsc::channel(64);
            routes.push(PreparedCellRoute {
                route_id: cell_id,
                destination_cell: cell_id,
                peer: peer.clone(),
            });
            sinks.push(Rc::new(LoopbackCommandSink {
                inbound: frame_tx,
                budget: inbound_budget.clone(),
            }));
            endpoints.push(CellularExecutionEndpoint {
                inbound: frame_rx,
                events: Rc::clone(&emitted) as Rc<dyn PlacementEventSink>,
                credential: Rc::new(credential),
                security: Rc::new(inbound_security),
                peer: peer.clone(),
                controller_peer: peer,
                destination: CellularRole::Cell(cell_id),
                sequencer: RouteSequencer::default(),
                staged: HashMap::new(),
                limits: CELLULAR_LIMITS,
                bound_profile_digest: None,
            });
        }

        let (binding, events_tx) = prepare_cellular_placement_binding(
            routes.into_boxed_slice(),
            sinks.into_boxed_slice(),
            CellularBindingContext {
                security: Rc::new(controller_sealer),
                session,
                plan_digest: [7; 32],
                budget: transfer_budget.clone(),
                limits: CellularTransferLimits {
                    max_items: 8,
                    max_bytes: 8 * FRAME_CEILING,
                    max_event_items: 32,
                    max_event_bytes: 32 * FRAME_CEILING,
                },
                cellular_limits: CELLULAR_LIMITS,
            },
            None,
        )
        .await
        .expect("prepare binding");

        Fixture {
            binding,
            events_tx,
            endpoints,
            authority,
            emitted,
            transfer_budget,
            event_budget,
        }
    }

    /// Drive one queued controller frame through the cell and return its sealed
    /// event to the controller's real admission path.
    async fn round_trip_one_frame(
        endpoint: &mut CellularExecutionEndpoint,
        emitted: &CollectingEventSink,
        authority: &CellRegistrationAuthority,
        event_budget: &StreamingResourceBudget,
        events_tx: &mpsc::Sender<BudgetOwnedCellEvent>,
        route_id: u32,
        is_release: bool,
    ) {
        let frame = endpoint
            .next_frame()
            .await
            .expect("queued controller frame");
        let event = if is_release {
            endpoint.accept_release(frame).expect("accept release")
        } else {
            endpoint
                .accept_prepare(frame)
                .expect("accept prepare")
                .expect("a fresh prepare owes a receipt")
        };
        endpoint.emit(&event).await.expect("emit receipt");
        let sealed = emitted.frames.borrow_mut().remove(0);
        admit_placement_event(
            authority,
            event_budget,
            CELLULAR_LIMITS,
            route_id,
            route_id,
            sealed,
            events_tx,
        )
        .await
        .expect("admit the authenticated event");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_route_window_backpressures_without_spawning_per_action_drivers() {
        let Fixture {
            mut binding,
            events_tx,
            mut endpoints,
            authority,
            emitted,
            event_budget,
            ..
        } = fixture(
            1,
            BudgetLimits {
                max_items: 5,
                max_bytes: 5 * FRAME_CEILING,
            },
        )
        .await;
        let diagnostics = binding.driver.diagnostics();

        binding
            .submitter
            .prepare(decision(0, 0, 1))
            .await
            .expect("first placement");
        binding
            .submitter
            .prepare(decision(0, 1, 2))
            .await
            .expect("second placement");

        // The window is full: the third placement parks rather than failing,
        // and no timer is involved in waking it.
        let mut third = std::pin::pin!(binding.submitter.prepare(decision(0, 2, 3)));
        let is_parked = tokio::select! {
            biased;
            _ = third.as_mut() => false,
            () = std::future::ready(()) => true,
        };
        assert!(
            is_parked,
            "a saturated prepare window must park a placement"
        );
        assert_eq!(
            diagnostics.driver_count(),
            1,
            "actions grow in the slab, never in drivers"
        );

        round_trip_one_frame(
            &mut endpoints[0],
            &emitted,
            &authority,
            &event_budget,
            &events_tx,
            0,
            false,
        )
        .await;
        let event = binding.driver.next_event().await.expect("prepared receipt");
        assert!(matches!(event.event, CellPlacementEvent::Prepared { .. }));

        third
            .await
            .expect("the released prepare charge admits the parked placement");
        assert_eq!(diagnostics.driver_count(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn release_is_not_blocked_by_a_saturated_prepare_window() {
        let Fixture {
            mut binding,
            mut endpoints,
            ..
        } = fixture(
            1,
            BudgetLimits {
                max_items: 4,
                max_bytes: 4 * FRAME_CEILING,
            },
        )
        .await;
        let first = binding
            .submitter
            .prepare(decision(0, 0, 1))
            .await
            .expect("first placement");
        binding
            .submitter
            .prepare(decision(0, 1, 2))
            .await
            .expect("second placement fills the window");

        // The co-reserved release charge is why this completes at all: without
        // it the only operation that can drain the window would queue behind it.
        binding
            .submitter
            .release(first.handle)
            .await
            .expect("release must not queue behind a full prepare window");

        for _ in 0..2 {
            let frame = endpoints[0].next_frame().await.expect("prepare frame");
            endpoints[0]
                .accept_prepare(frame)
                .expect("accept prepare")
                .expect("receipt");
        }
        let frame = endpoints[0].next_frame().await.expect("release frame");
        assert!(matches!(
            endpoints[0].accept_release(frame).expect("accept release"),
            CellPlacementEvent::Released { .. }
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancel_pending_wakes_a_parked_prepare_and_a_parked_driver() {
        // The endpoints are retained so the cell inbound channels stay open;
        // a closed receiver would fail the send rather than park the window.
        let Fixture {
            mut binding,
            endpoints: _endpoints,
            ..
        } = fixture(
            1,
            BudgetLimits {
                max_items: 2,
                max_bytes: 2 * FRAME_CEILING,
            },
        )
        .await;
        let control = binding.control.clone();
        binding
            .submitter
            .prepare(decision(0, 0, 1))
            .await
            .expect("first placement fills the window");

        let mut parked = std::pin::pin!(binding.submitter.prepare(decision(0, 1, 2)));
        let is_parked = tokio::select! {
            biased;
            _ = parked.as_mut() => false,
            () = std::future::ready(()) => true,
        };
        assert!(is_parked);

        control.cancel_pending();
        assert_eq!(
            parked.await.err(),
            Some(CellularStreamingError::Budget(BudgetError::Closed)),
            "closing the budget is what wakes a parked acquisition"
        );
        assert_eq!(
            binding.driver.next_event().await.err(),
            Some(CellularStreamingError::Cancelled),
            "the watch send is what wakes a driver parked on an empty channel"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn binding_task_count_is_independent_of_action_count() {
        let Fixture {
            mut binding,
            events_tx,
            mut endpoints,
            authority,
            emitted,
            event_budget,
            ..
        } = fixture(
            1,
            BudgetLimits {
                max_items: 128,
                max_bytes: 128 * FRAME_CEILING,
            },
        )
        .await;
        let diagnostics = binding.driver.diagnostics();

        for index in 0..32_u64 {
            binding
                .submitter
                .prepare(decision(0, index, index as u8 + 1))
                .await
                .expect("placement");
            round_trip_one_frame(
                &mut endpoints[0],
                &emitted,
                &authority,
                &event_budget,
                &events_tx,
                0,
                false,
            )
            .await;
            binding.driver.next_event().await.expect("prepared receipt");
            assert_eq!(diagnostics.driver_count(), 1);
            assert_eq!(diagnostics.owner_task_count(), 0);
        }
        assert_eq!(
            binding.submitter.in_flight(),
            32,
            "growth lands in the slab, not in owners"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drain_joins_owned_tasks_instead_of_aborting_them() {
        let Fixture {
            mut binding,
            endpoints: _endpoints,
            ..
        } = fixture(
            1,
            BudgetLimits {
                max_items: 8,
                max_bytes: 8 * FRAME_CEILING,
            },
        )
        .await;
        let diagnostics = binding.driver.diagnostics();
        let (finished, ran_to_completion) = tokio::sync::oneshot::channel::<()>();
        binding.driver.owners.adopt(tokio::spawn(async move {
            let _ = finished.send(());
        }));
        assert_eq!(diagnostics.owner_task_count(), 1);

        binding.driver.drain().await.expect("drain");
        assert_eq!(diagnostics.owner_task_count(), 0);
        assert!(
            ran_to_completion.await.is_ok(),
            "drain joins its owners; it never aborts them"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_placement_round_trips_through_authentication_staging_and_release() {
        let Fixture {
            mut binding,
            events_tx,
            mut endpoints,
            authority,
            emitted,
            transfer_budget,
            event_budget,
        } = fixture(
            1,
            BudgetLimits {
                max_items: 8,
                max_bytes: 8 * FRAME_CEILING,
            },
        )
        .await;

        let placed = binding
            .submitter
            .prepare(decision(0, 0, 1))
            .await
            .expect("placement");
        round_trip_one_frame(
            &mut endpoints[0],
            &emitted,
            &authority,
            &event_budget,
            &events_tx,
            0,
            false,
        )
        .await;
        let prepared = binding.driver.next_event().await.expect("prepared receipt");
        assert_eq!(prepared.handle, Some(placed.handle));
        assert_eq!(endpoints[0].staged_count(), 1);

        binding
            .submitter
            .release(placed.handle)
            .await
            .expect("release");
        round_trip_one_frame(
            &mut endpoints[0],
            &emitted,
            &authority,
            &event_budget,
            &events_tx,
            0,
            true,
        )
        .await;
        let released = binding.driver.next_event().await.expect("released receipt");
        assert_eq!(released.handle, Some(placed.handle));
        assert!(matches!(
            released.event,
            CellPlacementEvent::Released { .. }
        ));

        // The terminal receipt reclaims the entry, and with it both charges.
        assert_eq!(binding.submitter.in_flight(), 0);
        let snapshot = transfer_budget.snapshot();
        assert_eq!((snapshot.used_items, snapshot.used_bytes), (0, 0));
    }
}
