// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical host-owned units exchanged by streaming dataset stages.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use smallvec::SmallVec;

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

/// Move-only ownership token retaining bytes until a fragment is incorporated.
///
/// The zero-charged bootstrap constructor is intentionally crate-private:
///
/// ```compile_fail
/// use aiperf_runtime::streaming::unit::SessionFragmentLease;
///
/// let _lease = SessionFragmentLease::zero_charged();
/// ```
#[derive(Debug)]
pub struct SessionFragmentLease {
    _private: (),
}

impl SessionFragmentLease {
    /// Construct the zero-charged token used before budget integration.
    #[must_use]
    #[allow(dead_code, reason = "Task 1B replaces this bootstrap constructor")]
    pub(crate) const fn zero_charged() -> Self {
        Self { _private: () }
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
    pub action_id: StableActionId,
    /// Stable session owning the action.
    pub session_key: StableSessionKey,
    /// Stable actions that must complete before this action is ready.
    pub predecessors: SmallVec<[StableActionId; 2]>,
    /// Authored event time, when supplied.
    pub event_time: Option<EventTimeUtc>,
    /// Stable tie-break within an equal-time bucket.
    pub stable_order: StableOrderKey,
    /// Stable source coordinate that caused this action.
    pub source_position: SourcePosition,
    /// Source and format provenance receipt.
    pub provenance: UnitProvenance,
    /// Closed host-owned action payload.
    pub payload: DatasetActionV1,
}
