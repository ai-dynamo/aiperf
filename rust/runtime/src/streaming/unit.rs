// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical host-owned units exchanged by streaming dataset stages.

use std::{fmt, rc::Rc};

use serde::{Deserialize, Deserializer, Serialize};
use smallvec::SmallVec;

use super::budget::BudgetLease;
use super::identity::{
    ContentDigest, ImmutableObjectIdentity, StableActionId, StableOrderKey, StableRecordId,
    StableSessionKey,
};

/// Nanoseconds since the Unix epoch for a streaming event.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EventTimeUtc(i64);

impl EventTimeUtc {
    /// Construct a UTC event time, rejecting values before the Unix epoch.
    pub const fn new(nanoseconds: i64) -> Result<Self, EventTimeError> {
        if nanoseconds < 0 {
            return Err(EventTimeError::BeforeUnixEpoch(nanoseconds));
        }
        Ok(Self(nanoseconds))
    }

    /// Return nanoseconds since the Unix epoch.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for EventTimeUtc {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanoseconds = i64::deserialize(deserializer)?;
        Self::new(nanoseconds).map_err(serde::de::Error::custom)
    }
}

/// Invalid UTC event-time value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventTimeError {
    /// The value names an instant before the supported Unix epoch.
    BeforeUnixEpoch(i64),
}

impl fmt::Display for EventTimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforeUnixEpoch(value) => {
                write!(formatter, "event time {value}ns is before the Unix epoch")
            }
        }
    }
}

impl std::error::Error for EventTimeError {}

/// Stable unsigned coordinate within the resolved source stream.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourcePosition(u64);

impl SourcePosition {
    /// Construct a source position.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying coordinate.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advance by `delta`, failing instead of wrapping the coordinate.
    pub const fn checked_add(self, delta: u64) -> Result<Self, SourcePositionError> {
        match self.0.checked_add(delta) {
            Some(value) => Ok(Self(value)),
            None => Err(SourcePositionError::CoordinateOverflow {
                position: self.0,
                delta,
            }),
        }
    }
}

/// Invalid arithmetic on a source coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourcePositionError {
    /// Advancing the coordinate would exceed `u64::MAX`.
    CoordinateOverflow {
        /// Coordinate before the attempted advance.
        position: u64,
        /// Requested advance.
        delta: u64,
    },
}

impl fmt::Display for SourcePositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoordinateOverflow { position, delta } => {
                write!(
                    formatter,
                    "source position {position} + {delta} overflows u64"
                )
            }
        }
    }
}

impl std::error::Error for SourcePositionError {}

/// Closed generation-one family of actions produced by a session program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetActionKind {
    /// Materialize and issue one endpoint request.
    Request,
    /// Execute one host-owned graph node.
    GraphNode,
    /// Publish a terminal session update.
    SessionTerminal,
}

impl DatasetActionKind {
    pub(crate) const fn canonical_tag(self) -> u8 {
        match self {
            Self::Request => 0,
            Self::GraphNode => 1,
            Self::SessionTerminal => 2,
        }
    }
}

/// Neutral state-budget failure classification shared by runtime layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateBudgetFailureCode {
    /// No item slots remain.
    ItemCapacity,
    /// No in-memory byte capacity remains.
    ByteCapacity,
    /// No spill capacity remains.
    SpillCapacity,
    /// No provisional-state capacity remains.
    ProvisionalCapacity,
}

/// Source and format receipt carried by one canonical unit.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitProvenance {
    /// Immutable object generation that supplied the unit.
    pub source_partition: ImmutableObjectIdentity,
    /// Coordinate assigned by the source/decoder contract.
    pub source_position: SourcePosition,
    /// Semantic digest of the selected decoder format.
    pub format_semantic_digest: ContentDigest,
}

/// Closed generation-one mutation vocabulary emitted by streaming decoders.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionMutationV1 {
    /// Append an endpoint-neutral authored conversation turn.
    ConversationTurn(ConversationTurnFragment),
    /// Append a recorded agent event without executing its source tool.
    AgentEvent(AgentEventFragment),
    /// Declare a graph node and its endpoint-neutral authored request bytes.
    GraphNode(GraphNodeFragment),
    /// Declare one stable graph dependency edge.
    GraphEdge(GraphEdgeFragment),
    /// Explicitly close a session.
    SessionClose(SessionCloseFragment),
}

/// Endpoint-neutral authored conversation turn.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConversationTurnFragment {
    /// Authored conversation role.
    pub role: String,
    /// Authored content bytes.
    pub content: Vec<u8>,
    /// Stable ordinal within the producer session.
    pub turn_ordinal: u64,
}

/// Recorded agent event that remains non-executable at the decoder boundary.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentEventFragment {
    /// Host-interpreted event family.
    pub event_kind: String,
    /// Opaque authored event payload.
    pub payload: Vec<u8>,
    /// Stable ordinal within the producer session.
    pub event_ordinal: u64,
}

/// Authored graph node declaration.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNodeFragment {
    /// Stable producer node key.
    pub node_key: String,
    /// Endpoint-neutral authored request bytes.
    pub request: Vec<u8>,
}

/// Authored graph edge declaration.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEdgeFragment {
    /// Stable producer key of the predecessor node.
    pub from: String,
    /// Stable producer key of the successor node.
    pub to: String,
}

/// Explicit session-close mutation.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionCloseFragment {
    /// Authored close reason.
    pub reason: String,
}

/// Move-only ownership token retaining capacity until a fragment is incorporated.
#[derive(Debug)]
pub struct SessionFragmentLease(BudgetLease);

impl SessionFragmentLease {
    /// Return the fragment's retained item charge.
    #[must_use]
    pub fn charged_items(&self) -> usize {
        self.0.charged_items()
    }

    /// Return the fragment's retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.0.charged_bytes()
    }

    /// Return unused fragment bytes while retaining its one item permit.
    pub fn shrink_bytes_to(&mut self, bytes: usize) -> Result<(), super::budget::BudgetError> {
        self.0.shrink_to(1, bytes)
    }

    /// Transfer the fragment's charge into shared retained-content ownership.
    #[must_use]
    pub fn into_retained(self) -> RetainedContentLease {
        RetainedContentLease(Rc::new(RetainedContentLeaseInner { lease: self.0 }))
    }
}

impl TryFrom<BudgetLease> for SessionFragmentLease {
    type Error = super::budget::BudgetError;

    fn try_from(lease: BudgetLease) -> Result<Self, Self::Error> {
        if lease.charged_items() != 1 {
            return Err(Self::Error::InvalidFragmentItemCharge {
                charged_items: lease.charged_items(),
            });
        }
        Ok(Self(lease))
    }
}

/// Shared ownership of one previously acquired content charge.
#[derive(Debug)]
pub struct RetainedContentLease(Rc<RetainedContentLeaseInner>);

#[derive(Debug)]
struct RetainedContentLeaseInner {
    lease: BudgetLease,
}

impl Clone for RetainedContentLease {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl RetainedContentLease {
    /// Return the retained item charge.
    #[must_use]
    pub fn charged_items(&self) -> usize {
        self.0.lease.charged_items()
    }

    /// Return the retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.0.lease.charged_bytes()
    }
}

/// Content charges retained by an executable action and its continuations.
#[derive(Debug)]
pub struct ActionContentLeaseSet {
    leases: SmallVec<[RetainedContentLease; 2]>,
}

impl ActionContentLeaseSet {
    /// Create an action lease set from one incorporated fragment.
    #[must_use]
    pub fn from_retained(lease: RetainedContentLease) -> Self {
        Self {
            leases: smallvec::smallvec![lease],
        }
    }

    /// Add one distinct retained-content lease.
    ///
    /// An `Rc`-identical clone is consumed but not inserted, so it cannot
    /// overstate the capacity backing this action.
    pub fn insert(&mut self, lease: RetainedContentLease) -> bool {
        if self
            .leases
            .iter()
            .any(|existing| Rc::ptr_eq(&existing.0, &lease.0))
        {
            return false;
        }
        self.leases.push(lease);
        true
    }

    /// Move all distinct handles from another non-empty lease set into this one.
    pub fn merge(&mut self, other: Self) {
        for lease in other.leases {
            self.insert(lease);
        }
    }

    /// Retain the same charges for a continuation without acquiring capacity.
    #[must_use]
    pub fn retain_for_continuation(&self) -> Self {
        Self {
            leases: self.leases.iter().cloned().collect(),
        }
    }

    /// Return the number of distinct retained-content handles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.leases.len()
    }

    /// Return whether the set has no retained-content handles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }

    /// Sum the item charges of the distinct retained-content handles.
    pub fn charged_items(&self) -> Result<usize, super::budget::BudgetError> {
        self.leases.iter().try_fold(0usize, |total, lease| {
            total
                .checked_add(lease.charged_items())
                .ok_or(super::budget::BudgetError::AccountingOverflow)
        })
    }

    /// Sum the byte charges of the distinct retained-content handles.
    pub fn charged_bytes(&self) -> Result<usize, super::budget::BudgetError> {
        self.leases.iter().try_fold(0usize, |total, lease| {
            total
                .checked_add(lease.charged_bytes())
                .ok_or(super::budget::BudgetError::AccountingOverflow)
        })
    }

    fn retained_allocation_bytes(&self) -> Result<usize, super::budget::BudgetError> {
        if !self.leases.spilled() {
            return Ok(0);
        }
        self.leases
            .capacity()
            .checked_mul(std::mem::size_of::<RetainedContentLease>())
            .ok_or(super::budget::BudgetError::AccountingOverflow)
    }
}

/// Canonical session-addressed output of a streaming decoder.
#[derive(Debug)]
pub struct StreamingSessionFragment {
    /// Stable logical or physical record identity.
    pub record_id: StableRecordId,
    /// Stable key joining fragments across source partitions.
    pub session_key: StableSessionKey,
    /// Stable coordinate within the source stream.
    pub source_position: SourcePosition,
    /// Immutable object generation containing this fragment.
    pub source_partition: ImmutableObjectIdentity,
    /// Authored event time, when supplied by the format.
    pub event_time: Option<EventTimeUtc>,
    /// Stable tie-break for equal-time fragments.
    pub stable_tie_break: StableOrderKey,
    /// Stable records that must be incorporated before this mutation.
    pub predecessors: SmallVec<[StableRecordId; 2]>,
    /// Host-owned mutation payload.
    pub mutation: SessionMutationV1,
    /// Source and format provenance receipt.
    pub provenance: UnitProvenance,
    /// Ownership of bytes retained by this fragment.
    pub lease: SessionFragmentLease,
}

/// Closed generation-one executable action vocabulary.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatasetActionV1 {
    /// Endpoint-neutral request material owned by the host action binding.
    Request(SessionRequestAction),
    /// Host-owned graph-node action.
    GraphNode(SessionGraphAction),
    /// Terminal session action.
    SessionTerminal(SessionTerminalAction),
}

impl DatasetActionV1 {
    /// Return retained heap capacity that requires content-lease coverage.
    pub fn retained_allocation_bytes(&self) -> Result<usize, super::budget::BudgetError> {
        match self {
            Self::Request(action) => Ok(action.request.capacity()),
            Self::GraphNode(action) => action
                .node_key
                .capacity()
                .checked_add(action.request.capacity())
                .ok_or(super::budget::BudgetError::AccountingOverflow),
            Self::SessionTerminal(action) => Ok(action.reason.capacity()),
        }
    }
}

/// Endpoint-neutral request action payload.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRequestAction {
    /// Canonical authored request bytes interpreted by the selected binding.
    pub request: Vec<u8>,
}

/// Graph-node action payload.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionGraphAction {
    /// Stable producer node key.
    pub node_key: String,
    /// Canonical authored request bytes interpreted by the selected binding.
    pub request: Vec<u8>,
}

/// Terminal session action payload.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionTerminalAction {
    /// Authored terminal reason.
    pub reason: String,
}

/// Causally ready host-owned output of a streaming session program.
#[derive(Debug)]
pub struct ExecutableDatasetAction {
    /// Stable logical action identity.
    action_id: StableActionId,
    /// Stable session owning the action.
    session_key: StableSessionKey,
    /// Stable actions that must complete before this action is ready.
    predecessors: SmallVec<[StableActionId; 2]>,
    /// Authored event time, when supplied.
    event_time: Option<EventTimeUtc>,
    /// Stable tie-break within an equal-time bucket.
    stable_order: StableOrderKey,
    /// Stable source coordinate that caused this action.
    source_position: SourcePosition,
    /// Source and format provenance receipt.
    provenance: UnitProvenance,
    /// Closed host-owned action payload.
    payload: DatasetActionV1,
    /// Content charges retained until this action and every continuation finish.
    content_leases: ActionContentLeaseSet,
}

impl ExecutableDatasetAction {
    /// Construct an action whose payload capacity and spilled metadata are lease-covered.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        action_id: StableActionId,
        session_key: StableSessionKey,
        predecessors: SmallVec<[StableActionId; 2]>,
        event_time: Option<EventTimeUtc>,
        stable_order: StableOrderKey,
        source_position: SourcePosition,
        provenance: UnitProvenance,
        payload: DatasetActionV1,
        content_leases: ActionContentLeaseSet,
    ) -> Result<Self, super::budget::BudgetError> {
        let payload_bytes = payload.retained_allocation_bytes()?;
        let predecessor_bytes = if predecessors.spilled() {
            predecessors
                .capacity()
                .checked_mul(std::mem::size_of::<StableActionId>())
                .ok_or(super::budget::BudgetError::AccountingOverflow)?
        } else {
            0
        };
        let lease_set_bytes = content_leases.retained_allocation_bytes()?;
        let required_bytes = payload_bytes
            .checked_add(predecessor_bytes)
            .and_then(|bytes| bytes.checked_add(lease_set_bytes))
            .ok_or(super::budget::BudgetError::AccountingOverflow)?;
        let retained_bytes = content_leases.charged_bytes()?;
        if required_bytes > retained_bytes {
            return Err(super::budget::BudgetError::ActionPayloadUndercharged {
                required_bytes,
                retained_bytes,
            });
        }
        Ok(Self {
            action_id,
            session_key,
            predecessors,
            event_time,
            stable_order,
            source_position,
            provenance,
            payload,
            content_leases,
        })
    }

    /// Return the stable logical action identity.
    #[must_use]
    pub fn action_id(&self) -> StableActionId {
        self.action_id
    }

    /// Return the stable session owning the action.
    #[must_use]
    pub fn session_key(&self) -> StableSessionKey {
        self.session_key
    }

    /// Borrow stable predecessor action identities.
    #[must_use]
    pub fn predecessors(&self) -> &[StableActionId] {
        &self.predecessors
    }

    /// Return the authored event time, when supplied.
    #[must_use]
    pub fn event_time(&self) -> Option<EventTimeUtc> {
        self.event_time
    }

    /// Return the stable equal-time tie-break key.
    #[must_use]
    pub fn stable_order(&self) -> StableOrderKey {
        self.stable_order
    }

    /// Return the stable source coordinate that caused this action.
    #[must_use]
    pub fn source_position(&self) -> SourcePosition {
        self.source_position
    }

    /// Borrow source and format provenance.
    #[must_use]
    pub fn provenance(&self) -> &UnitProvenance {
        &self.provenance
    }

    /// Borrow the closed host-owned action payload.
    #[must_use]
    pub fn payload(&self) -> &DatasetActionV1 {
        &self.payload
    }

    /// Borrow the action's retained content charges.
    #[must_use]
    pub fn content_leases(&self) -> &ActionContentLeaseSet {
        &self.content_leases
    }
}
