// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict, authenticated cellular streaming command and event vocabulary.
//!
//! Controller-to-cell commands ([`PrepareAction`], [`ReleaseAction`],
//! [`BindContentSynthesisProfileV1`]) are sealed by the controller into a
//! [`ControllerAuthenticatedFrame`] and opened by the destination cell only
//! after destination, purpose, peer, signature, controller session, and
//! sequence have all been proven. Cell-to-controller [`CellPlacementEvent`]s
//! reuse the existing worker-signed authenticated-frame path under the two
//! appended admission purposes `StreamingPlacementEvent` and
//! `StreamingResultPartition`.
//!
//! Every payload is canonical **named** MessagePack. Positional encoding would
//! make `deny_unknown_fields` inert, because a positional struct is encoded as
//! an array and has no field names for a decoder to reject. This is a
//! deliberate divergence from the surrounding registration code, which stays
//! positional; the two encodings never meet, because a streaming payload is
//! only ever read by a streaming seed.
//!
//! Variable-length payload decoding never trusts a declared length. Each
//! variable field is read through a bounded [`serde::de::DeserializeSeed`] (or,
//! where a runtime limit cannot be threaded through a derived decoder, a
//! newtype with a fixed ceiling), so a hostile length cannot cause an
//! allocation even after the signature has verified.

// The frame boundary is complete and unit-tested here; its transport handlers,
// queues, and drivers are the next cellular streaming task. Until they land the
// sealer, verifier, and wire vocabulary have no in-tree caller.
#![allow(dead_code)]

use std::fmt;
use std::marker::PhantomData;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::engine::cellular_bootstrap::CellularRole;
use crate::engine::cellular_registration::AdmissionRejection;
use crate::streaming::action::DatasetActionSchema;
use crate::streaming::budget::BudgetLease;
use crate::streaming::failure::PlacementFailureCode;
use crate::streaming::identity::{
    ActionAttemptId, GlobalSequence, SessionOwnershipEpoch, StableActionId,
};
use crate::streaming::session::conversation::SessionStateVersion;

/// Fixed protocol version for every cellular streaming frame and payload.
pub const STREAMING_CELLULAR_PROTOCOL_VERSION: u16 = 1;

/// Domain separator for the controller-signed streaming frame transcript.
pub(crate) const CONTROLLER_FRAME_TRANSCRIPT_DOMAIN: &[u8] =
    b"aiperf-cellular-controller-frame-v1\0";

/// Domain separator for the derived controller streaming session identity.
pub(crate) const CONTROLLER_SESSION_DOMAIN: &[u8] = b"aiperf-cellular-controller-session-v1\0";

/// Number of distinct controller-to-cell streaming purposes.
pub(crate) const CONTROLLER_STREAMING_PURPOSE_COUNT: usize = 3;

/// Stable message text a bounded seed emits and the frame boundary maps onto
/// [`AdmissionRejection::ContentLimitExceeded`].
const CONTENT_LIMIT_MESSAGE: &str = "streaming content limit exceeded";

/// Fixed ceiling on one normalized session-update body.
///
/// A runtime limit cannot be threaded through a derived decoder, so the wire
/// twin carries its bound in the type. The runtime `max_payload_bytes` is
/// applied in addition, never instead.
pub const MAX_SESSION_UPDATE_BYTES: usize = 4 * 1024 * 1024;

/// Fixed per-frame and per-payload capacity limits.
///
/// The transport reserves exactly `max_frame_bytes` before sealing and refuses
/// an inbound frame larger than it before any outer decode, so neither side can
/// be induced to allocate by a declared length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingCellularLimits {
    /// Largest encoded [`ControllerAuthenticatedFrame`] accepted or produced.
    pub max_frame_bytes: usize,
    /// Largest inner payload accepted after frame authentication.
    pub max_payload_bytes: usize,
    /// Largest number of content leases in one prepared action.
    pub max_content_items: usize,
    /// Largest total content byte length in one prepared action.
    pub max_content_bytes: usize,
}

/// One controller-to-cell streaming operation class.
///
/// Each variant owns an independent outbound sequence per destination cell and
/// an independent inbound replay window on that cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ControllerStreamingPurpose {
    /// Freezes the run-scoped content synthesis profile binding.
    BindContentSynthesisProfile = 1,
    /// Transfers one fenced prepared action without issue authority.
    PrepareAction = 2,
    /// Grants issue authority for one exactly-matching prepared action.
    ReleaseAction = 3,
}

impl ControllerStreamingPurpose {
    /// Index into the fixed per-purpose sequence and replay arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize - 1
    }

    /// Whether this purpose may target the given destination role.
    ///
    /// Generation one places streaming actions on benchmark cells only;
    /// aggregators fold results and never execute.
    #[must_use]
    pub const fn supports(self, destination: CellularRole) -> bool {
        matches!(destination, CellularRole::Cell(_))
    }
}

/// Controller process-and-connection identity bound into every streaming frame.
///
/// Derived, not transported: both sides compute it from the controller peer
/// binding each has already independently proven during registration. A
/// restarted controller has a fresh velo instance identity and therefore a
/// different session, so its old frames fail closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControllerStreamingSessionId([u8; 32]);

impl ControllerStreamingSessionId {
    /// Construct from an already-derived digest.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the canonical digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// One immutable content lease referenced by a prepared action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentLeaseDescriptor {
    /// Content-addressed identity of the retained bytes.
    pub content_id: [u8; 32],
    /// Exact retained byte length.
    pub byte_length: u64,
    /// Digest of the retained bytes.
    pub digest: [u8; 32],
}

/// Fenced request material for one prepared action.
///
/// Decoded only through [`PreparedActionContentSeed`]; the derived
/// `Deserialize` is deliberately absent so no caller can bypass the limits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedActionContent {
    /// Action schema the destination binding must accept.
    pub schema: DatasetActionSchema,
    /// Canonical endpoint request bytes.
    pub canonical_request: Vec<u8>,
    /// Immutable content leases this action references.
    pub content_leases: Vec<ContentLeaseDescriptor>,
    /// Declared lease count, checked against `content_leases.len()`.
    pub item_count: u64,
    /// Declared total byte length, checked against the decoded bytes.
    pub byte_length: u64,
    /// Digest binding schema, request bytes, and leases.
    pub digest: [u8; 32],
}

impl PreparedActionContent {
    /// Recompute the canonical content digest from the decoded fields.
    #[must_use]
    pub fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, b"aiperf-streaming-prepared-content-v1\0");
        update_field(&mut hasher, self.schema.as_str().as_bytes());
        update_field(&mut hasher, &self.canonical_request);
        for lease in &self.content_leases {
            update_field(&mut hasher, &lease.content_id);
            update_field(&mut hasher, &lease.byte_length.to_le_bytes());
            update_field(&mut hasher, &lease.digest);
        }
        *hasher.finalize().as_bytes()
    }

    /// Exact retained byte cost of this content.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.canonical_request.len()
    }
}

/// Run-scoped binding of the authored content synthesis profile to its
/// controller-reconstructed form.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindContentSynthesisProfileV1 {
    /// Fixed payload version.
    pub version: u16,
    /// Digest of the accepted streaming plan.
    pub plan_digest: [u8; 32],
    /// Digest of the authored profile as configured.
    pub authored_profile_digest: [u8; 32],
    /// Digest of the profile as bound by the controller.
    pub bound_profile_digest: [u8; 32],
}

/// One fenced prepared action, without issue authority.
///
/// Decoded only through [`PrepareActionSeed`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareAction {
    /// Fixed payload version.
    pub version: u16,
    /// Digest of the accepted streaming plan.
    pub plan_digest: [u8; 32],
    /// Frozen synthesis profile this action depends on, when one is bound.
    pub synthesis_profile_digest: Option<[u8; 32]>,
    /// Controller-owned route identity.
    pub route_id: u32,
    /// Destination cell identity, checked against the frame destination.
    pub destination_cell: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Incarnation-local attempt identity.
    pub attempt_id: ActionAttemptId,
    /// Dense controller-assigned global order position.
    pub global_sequence: GlobalSequence,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Session state version this action was linearized against.
    pub prior_session_state_version: SessionStateVersion,
    /// Fenced request material.
    pub content: PreparedActionContent,
}

/// Issue authority for one exactly-matching prepared action.
///
/// A cell may never issue on a [`PrepareAction`]; only a release naming the
/// same action, global sequence, route, and ownership epoch grants issuance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseAction {
    /// Fixed payload version.
    pub version: u16,
    /// Digest of the accepted streaming plan.
    pub plan_digest: [u8; 32],
    /// Controller-owned route identity.
    pub route_id: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Dense controller-assigned global order position.
    pub global_sequence: GlobalSequence,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
}

/// Identity and ordering carried by every wire action event.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireActionEventIdentity {
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Incarnation-local attempt identity.
    pub attempt_id: ActionAttemptId,
    /// Fenced session route epoch.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Strictly increasing ordinal within the attempt.
    pub event_ordinal: u64,
}

/// Terminal disposition of one attempt on the wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum WireActionTerminalDisposition {
    /// Action completed successfully.
    Completed,
    /// Action failed after admission.
    Failed,
    /// Action was cancelled.
    Cancelled,
    /// Action was dropped before endpoint issue.
    Dropped,
}

/// Normalized session-update bytes carrying their own fixed ceiling.
///
/// The derived decoder cannot receive a runtime limit, so the bound lives in
/// the type and is enforced before the buffer is materialized.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SessionUpdateBytes(Vec<u8>);

impl SessionUpdateBytes {
    /// Construct from bytes that fit the fixed ceiling.
    pub fn new(bytes: Vec<u8>) -> Result<Self, AdmissionRejection> {
        if bytes.len() > MAX_SESSION_UPDATE_BYTES {
            return Err(AdmissionRejection::ContentLimitExceeded);
        }
        Ok(Self(bytes))
    }

    /// Borrow the normalized bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SessionUpdateBytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        BoundedBytesSeed {
            max_bytes: MAX_SESSION_UPDATE_BYTES,
        }
        .deserialize(deserializer)
        .map(Self)
    }
}

/// One per-turn action execution observation reported by a cell.
///
/// Owned wire twin of the runtime action-execution event, which transitively
/// owns a [`BudgetLease`] and is therefore neither `Clone` nor `Serialize`. The
/// twin carries normalized bytes and the receiving side charges its own budget.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireActionExecutionEvent {
    /// Action admission became authoritative.
    Admitted {
        /// Event identity and order.
        event: WireActionEventIdentity,
    },
    /// First output token was observed.
    FirstToken {
        /// Event identity and order.
        event: WireActionEventIdentity,
    },
    /// Endpoint-derived normalized session update.
    SessionUpdate {
        /// Event identity and order.
        event: WireActionEventIdentity,
        /// Normalized update bytes.
        payload: SessionUpdateBytes,
    },
    /// Unique final event for the attempt.
    Terminal {
        /// Event identity and order.
        event: WireActionEventIdentity,
        /// Terminal disposition.
        disposition: WireActionTerminalDisposition,
    },
}

/// Cell acknowledgement of a frozen synthesis profile binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentSynthesisProfileBoundReceipt {
    /// Fixed payload version.
    pub version: u16,
    /// Digest of the accepted streaming plan.
    pub plan_digest: [u8; 32],
    /// Digest of the authored profile as the cell resolved it.
    pub authored_profile_digest: [u8; 32],
    /// Digest of the bound profile as the cell resolved it.
    pub bound_profile_digest: [u8; 32],
}

/// Cell receipt for one accepted prepared action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementPreparedReceipt {
    /// Controller-owned route identity.
    pub route_id: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Dense controller-assigned global order position.
    pub global_sequence: GlobalSequence,
    /// Content digest the cell computed over the received content.
    pub content_digest: [u8; 32],
}

/// Cell receipt for one released action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementReleasedReceipt {
    /// Controller-owned route identity.
    pub route_id: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Dense controller-assigned global order position.
    pub global_sequence: GlobalSequence,
}

/// Cell report of a terminal placement failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementFailureReceipt {
    /// Controller-owned route identity.
    pub route_id: u32,
    /// Stable logical action identity.
    pub action_id: StableActionId,
    /// Stable failure classification.
    pub code: PlacementFailureCode,
}

/// One ordered cell-to-controller placement event.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CellPlacementEvent {
    /// The cell froze the controller's synthesis profile binding.
    ContentSynthesisProfileBound {
        /// Acknowledged binding.
        receipt: ContentSynthesisProfileBoundReceipt,
    },
    /// The cell accepted a prepared action.
    Prepared {
        /// Prepared-action receipt.
        receipt: PlacementPreparedReceipt,
    },
    /// The cell issued a released action.
    Released {
        /// Released-action receipt.
        receipt: PlacementReleasedReceipt,
    },
    /// One per-turn execution observation.
    Action {
        /// Execution observation.
        event: WireActionExecutionEvent,
    },
    /// The cell failed the action terminally.
    Failed {
        /// Terminal failure receipt.
        receipt: PlacementFailureReceipt,
    },
}

/// Bounded reader for [`CellPlacementEvent`].
///
/// The frame boundary has already length-checked the payload; this seed applies
/// the runtime payload limit a second time and reads the only variable-length
/// field through its fixed-ceiling newtype.
pub(crate) struct CellPlacementEventSeed {
    limits: StreamingCellularLimits,
}

impl CellPlacementEventSeed {
    pub(crate) const fn new(limits: StreamingCellularLimits) -> Self {
        Self { limits }
    }

    /// Decode one authenticated placement event under the configured limits.
    pub(crate) fn decode(self, payload: &[u8]) -> Result<CellPlacementEvent, AdmissionRejection> {
        if payload.len() > self.limits.max_payload_bytes {
            return Err(AdmissionRejection::Oversized);
        }
        let event: CellPlacementEvent = rmp_serde::from_slice(payload).map_err(map_seed_error)?;
        if let CellPlacementEvent::Action {
            event: WireActionExecutionEvent::SessionUpdate { payload, .. },
        } = &event
            && payload.as_slice().len() > self.limits.max_payload_bytes
        {
            return Err(AdmissionRejection::ContentLimitExceeded);
        }
        Ok(event)
    }
}

/// One controller-signed frame addressed to an exact destination role.
///
/// `destination` names the **recipient**, not the signer: the signer is always
/// the controller. The worker-signed authenticated frame cannot serve this
/// direction because its `role` is the signer and [`CellularRole`]
/// intentionally has no controller variant.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ControllerAuthenticatedFrame {
    pub(crate) version: u16,
    pub(crate) destination: CellularRole,
    pub(crate) controller_session: [u8; 32],
    pub(crate) sequence: u64,
    pub(crate) peer_info: Vec<u8>,
    pub(crate) payload: Vec<u8>,
    pub(crate) signature: Vec<u8>,
}

/// Encoded frame bytes and the exact budget charge that owns them.
///
/// Not `Clone`: the permit moves with the payload and is released only by
/// `Drop`.
#[derive(Debug)]
pub struct BudgetOwnedFrame {
    bytes: bytes::Bytes,
    lease: BudgetLease,
}

impl BudgetOwnedFrame {
    /// Bind encoded frame bytes to the lease that paid for them.
    #[must_use]
    pub fn new(bytes: bytes::Bytes, lease: BudgetLease) -> Self {
        Self { bytes, lease }
    }

    /// Borrow the encoded frame.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Split the frame into its bytes and its still-live charge.
    #[must_use]
    pub fn into_parts(self) -> (bytes::Bytes, BudgetLease) {
        (self.bytes, self.lease)
    }
}

/// A signature-verified, session-checked, replay-checked payload that has not
/// yet been typed.
#[derive(Debug)]
pub struct AuthenticatedStreamingPayload {
    bytes: Vec<u8>,
    lease: BudgetLease,
}

impl AuthenticatedStreamingPayload {
    /// Bind an authenticated payload to the charge that owns it.
    #[must_use]
    pub fn new(bytes: Vec<u8>, lease: BudgetLease) -> Self {
        Self { bytes, lease }
    }

    /// Borrow the authenticated payload bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Take the charge, dropping the raw bytes.
    #[must_use]
    pub fn into_lease(self) -> BudgetLease {
        self.lease
    }
}

/// A decoded prepared action still owning its exact capacity charge.
#[derive(Debug)]
pub struct BudgetOwnedPrepareAction {
    action: PrepareAction,
    lease: BudgetLease,
}

impl BudgetOwnedPrepareAction {
    /// Bind a decoded action to the charge that owns it.
    #[must_use]
    pub fn new(action: PrepareAction, lease: BudgetLease) -> Self {
        Self { action, lease }
    }

    /// Borrow the decoded action.
    #[must_use]
    pub fn action(&self) -> &PrepareAction {
        &self.action
    }

    /// Split into the decoded action and its still-live charge.
    #[must_use]
    pub fn into_parts(self) -> (PrepareAction, BudgetLease) {
        (self.action, self.lease)
    }
}

/// A decoded synthesis profile binding still owning its capacity charge.
#[derive(Debug)]
pub struct BudgetOwnedSynthesisProfileBinding {
    binding: BindContentSynthesisProfileV1,
    lease: BudgetLease,
}

impl BudgetOwnedSynthesisProfileBinding {
    /// Bind a decoded profile binding to the charge that owns it.
    #[must_use]
    pub fn new(binding: BindContentSynthesisProfileV1, lease: BudgetLease) -> Self {
        Self { binding, lease }
    }

    /// Borrow the decoded profile binding.
    #[must_use]
    pub fn binding(&self) -> &BindContentSynthesisProfileV1 {
        &self.binding
    }

    /// Split into the decoded binding and its still-live charge.
    #[must_use]
    pub fn into_parts(self) -> (BindContentSynthesisProfileV1, BudgetLease) {
        (self.binding, self.lease)
    }
}

/// An acquired outbound frame reservation of exactly `max_frame_bytes`.
///
/// The synchronous sealer cannot allocate a frame without one, and shrinks the
/// lease to the encoded length before returning.
#[derive(Debug)]
pub struct FrameBudgetReservation {
    lease: BudgetLease,
    max_frame_bytes: usize,
}

impl FrameBudgetReservation {
    /// Bind an acquired lease to the frame ceiling it was acquired against.
    pub(crate) fn new(
        lease: BudgetLease,
        max_frame_bytes: usize,
    ) -> Result<Self, AdmissionRejection> {
        if lease.charged_items() != 1 || lease.charged_bytes() != max_frame_bytes {
            return Err(AdmissionRejection::ContentLimitExceeded);
        }
        Ok(Self {
            lease,
            max_frame_bytes,
        })
    }

    /// Return the frame ceiling this reservation was acquired against.
    #[must_use]
    pub fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    /// Shrink the reservation to the exact encoded length and take the lease.
    pub(crate) fn into_lease_for(
        mut self,
        encoded_len: usize,
    ) -> Result<BudgetLease, AdmissionRejection> {
        if encoded_len > self.max_frame_bytes {
            return Err(AdmissionRejection::Oversized);
        }
        self.lease
            .shrink_to(1, encoded_len)
            .map_err(|_| AdmissionRejection::ContentLimitExceeded)?;
        Ok(self.lease)
    }
}

/// Map a seed decode failure onto the fixed rejection classes.
///
/// Limit refusals are reported separately from ordinary malformation so a
/// reviewer can tell "hostile length" from "corrupt bytes"; both `Display` as
/// the single opaque `AdmissionRejected`.
pub(crate) fn map_seed_error(error: rmp_serde::decode::Error) -> AdmissionRejection {
    if error.to_string().contains(CONTENT_LIMIT_MESSAGE) {
        AdmissionRejection::ContentLimitExceeded
    } else {
        AdmissionRejection::Malformed
    }
}

/// Read a byte string whose declared length is checked before allocation.
struct BoundedBytesSeed {
    max_bytes: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedBytesSeed {
    type Value = Vec<u8>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_bytes(self)
    }
}

impl<'de> Visitor<'de> for BoundedBytesSeed {
    type Value = Vec<u8>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {} bytes", self.max_bytes)
    }

    fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Self::Value, E> {
        if value.len() > self.max_bytes {
            return Err(E::custom(CONTENT_LIMIT_MESSAGE));
        }
        Ok(value.to_vec())
    }

    fn visit_byte_buf<E: de::Error>(self, value: Vec<u8>) -> Result<Self::Value, E> {
        if value.len() > self.max_bytes {
            return Err(E::custom(CONTENT_LIMIT_MESSAGE));
        }
        Ok(value)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        // Refuse the declared length before reserving; fall back to a bounded
        // push loop when the decoder declines to hint.
        if sequence
            .size_hint()
            .is_some_and(|hint| hint > self.max_bytes)
        {
            return Err(de::Error::custom(CONTENT_LIMIT_MESSAGE));
        }
        let mut out = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.max_bytes));
        while let Some(byte) = sequence.next_element::<u8>()? {
            if out.len() == self.max_bytes {
                return Err(de::Error::custom(CONTENT_LIMIT_MESSAGE));
            }
            out.push(byte);
        }
        Ok(out)
    }
}

/// Read a bounded-length string without trusting a declared length.
struct BoundedStringSeed {
    max_bytes: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedStringSeed {
    type Value = String;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for BoundedStringSeed {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "a string of at most {} bytes", self.max_bytes)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        if value.len() > self.max_bytes {
            return Err(E::custom(CONTENT_LIMIT_MESSAGE));
        }
        Ok(value.to_owned())
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        self.visit_str(&value)
    }
}

/// Read a homogeneous sequence whose element count is checked before reserving.
struct BoundedVecSeed<T> {
    max_items: usize,
    element: PhantomData<T>,
}

impl<T> BoundedVecSeed<T> {
    const fn new(max_items: usize) -> Self {
        Self {
            max_items,
            element: PhantomData,
        }
    }
}

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for BoundedVecSeed<T> {
    type Value = Vec<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for BoundedVecSeed<T> {
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {} elements", self.max_items)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        if sequence
            .size_hint()
            .is_some_and(|hint| hint > self.max_items)
        {
            return Err(de::Error::custom(CONTENT_LIMIT_MESSAGE));
        }
        let mut out = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.max_items));
        while let Some(element) = sequence.next_element::<T>()? {
            if out.len() == self.max_items {
                return Err(de::Error::custom(CONTENT_LIMIT_MESSAGE));
            }
            out.push(element);
        }
        Ok(out)
    }
}

fn assign<T, E: de::Error>(slot: &mut Option<T>, field: &'static str, value: T) -> Result<(), E> {
    if slot.is_some() {
        return Err(E::duplicate_field(field));
    }
    *slot = Some(value);
    Ok(())
}

const PREPARED_CONTENT_FIELDS: &[&str] = &[
    "schema",
    "canonical_request",
    "content_leases",
    "item_count",
    "byte_length",
    "digest",
];

/// Bounded reader for [`PreparedActionContent`].
///
/// `DatasetActionSchema` derives an unbounded transparent `Deserialize`; this
/// seed reads the schema through [`BoundedStringSeed`] instead, so a hostile
/// schema string cannot be materialized.
pub(crate) struct PreparedActionContentSeed {
    limits: StreamingCellularLimits,
}

impl PreparedActionContentSeed {
    pub(crate) const fn new(limits: StreamingCellularLimits) -> Self {
        Self { limits }
    }
}

impl<'de> DeserializeSeed<'de> for PreparedActionContentSeed {
    type Value = PreparedActionContent;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("PreparedActionContent", PREPARED_CONTENT_FIELDS, self)
    }
}

impl<'de> Visitor<'de> for PreparedActionContentSeed {
    type Value = PreparedActionContent;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a named PreparedActionContent map")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut schema: Option<String> = None;
        let mut canonical_request: Option<Vec<u8>> = None;
        let mut content_leases: Option<Vec<ContentLeaseDescriptor>> = None;
        let mut item_count: Option<u64> = None;
        let mut byte_length: Option<u64> = None;
        let mut digest: Option<[u8; 32]> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "schema" => {
                    let value = map.next_value_seed(BoundedStringSeed {
                        max_bytes: self.limits.max_content_bytes,
                    })?;
                    assign(&mut schema, "schema", value)?;
                }
                "canonical_request" => {
                    let value = map.next_value_seed(BoundedBytesSeed {
                        max_bytes: self.limits.max_content_bytes,
                    })?;
                    assign(&mut canonical_request, "canonical_request", value)?;
                }
                "content_leases" => {
                    let value =
                        map.next_value_seed(BoundedVecSeed::<ContentLeaseDescriptor>::new(
                            self.limits.max_content_items,
                        ))?;
                    assign(&mut content_leases, "content_leases", value)?;
                }
                "item_count" => assign(&mut item_count, "item_count", map.next_value()?)?,
                "byte_length" => assign(&mut byte_length, "byte_length", map.next_value()?)?,
                "digest" => assign(&mut digest, "digest", map.next_value()?)?,
                other => return Err(de::Error::unknown_field(other, PREPARED_CONTENT_FIELDS)),
            }
        }

        let content = PreparedActionContent {
            schema: DatasetActionSchema::new(
                schema.ok_or_else(|| de::Error::missing_field("schema"))?,
            ),
            canonical_request: canonical_request
                .ok_or_else(|| de::Error::missing_field("canonical_request"))?,
            content_leases: content_leases
                .ok_or_else(|| de::Error::missing_field("content_leases"))?,
            item_count: item_count.ok_or_else(|| de::Error::missing_field("item_count"))?,
            byte_length: byte_length.ok_or_else(|| de::Error::missing_field("byte_length"))?,
            digest: digest.ok_or_else(|| de::Error::missing_field("digest"))?,
        };

        // Declared counts are checked with exact arithmetic against what was
        // actually decoded, then the digest re-binds every decoded field.
        let declared_items = usize::try_from(content.item_count)
            .map_err(|_| de::Error::custom(CONTENT_LIMIT_MESSAGE))?;
        let declared_bytes = usize::try_from(content.byte_length)
            .map_err(|_| de::Error::custom(CONTENT_LIMIT_MESSAGE))?;
        if declared_items != content.content_leases.len()
            || declared_bytes != content.canonical_request.len()
            || declared_items > self.limits.max_content_items
            || declared_bytes > self.limits.max_content_bytes
        {
            return Err(de::Error::custom(CONTENT_LIMIT_MESSAGE));
        }
        if content.compute_digest() != content.digest {
            return Err(de::Error::custom("prepared content digest mismatch"));
        }
        Ok(content)
    }
}

const PREPARE_ACTION_FIELDS: &[&str] = &[
    "version",
    "plan_digest",
    "synthesis_profile_digest",
    "route_id",
    "destination_cell",
    "action_id",
    "attempt_id",
    "global_sequence",
    "ownership_epoch",
    "prior_session_state_version",
    "content",
];

/// Bounded reader for [`PrepareAction`].
pub(crate) struct PrepareActionSeed {
    limits: StreamingCellularLimits,
}

impl PrepareActionSeed {
    pub(crate) const fn new(limits: StreamingCellularLimits) -> Self {
        Self { limits }
    }

    /// Decode one authenticated prepare payload under the configured limits.
    pub(crate) fn decode(self, payload: &[u8]) -> Result<PrepareAction, AdmissionRejection> {
        if payload.len() > self.limits.max_payload_bytes {
            return Err(AdmissionRejection::Oversized);
        }
        let mut deserializer = rmp_serde::Deserializer::from_read_ref(payload);
        self.deserialize(&mut deserializer).map_err(map_seed_error)
    }
}

impl<'de> DeserializeSeed<'de> for PrepareActionSeed {
    type Value = PrepareAction;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("PrepareAction", PREPARE_ACTION_FIELDS, self)
    }
}

impl<'de> Visitor<'de> for PrepareActionSeed {
    type Value = PrepareAction;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a named PrepareAction map")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut version: Option<u16> = None;
        let mut plan_digest: Option<[u8; 32]> = None;
        let mut synthesis_profile_digest: Option<Option<[u8; 32]>> = None;
        let mut route_id: Option<u32> = None;
        let mut destination_cell: Option<u32> = None;
        let mut action_id: Option<StableActionId> = None;
        let mut attempt_id: Option<ActionAttemptId> = None;
        let mut global_sequence: Option<GlobalSequence> = None;
        let mut ownership_epoch: Option<SessionOwnershipEpoch> = None;
        let mut prior_session_state_version: Option<SessionStateVersion> = None;
        let mut content: Option<PreparedActionContent> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "version" => assign(&mut version, "version", map.next_value()?)?,
                "plan_digest" => assign(&mut plan_digest, "plan_digest", map.next_value()?)?,
                "synthesis_profile_digest" => assign(
                    &mut synthesis_profile_digest,
                    "synthesis_profile_digest",
                    map.next_value()?,
                )?,
                "route_id" => assign(&mut route_id, "route_id", map.next_value()?)?,
                "destination_cell" => {
                    assign(&mut destination_cell, "destination_cell", map.next_value()?)?;
                }
                "action_id" => assign(&mut action_id, "action_id", map.next_value()?)?,
                "attempt_id" => assign(&mut attempt_id, "attempt_id", map.next_value()?)?,
                "global_sequence" => {
                    assign(&mut global_sequence, "global_sequence", map.next_value()?)?;
                }
                "ownership_epoch" => {
                    assign(&mut ownership_epoch, "ownership_epoch", map.next_value()?)?;
                }
                "prior_session_state_version" => assign(
                    &mut prior_session_state_version,
                    "prior_session_state_version",
                    map.next_value()?,
                )?,
                "content" => {
                    let value = map.next_value_seed(PreparedActionContentSeed::new(self.limits))?;
                    assign(&mut content, "content", value)?;
                }
                other => return Err(de::Error::unknown_field(other, PREPARE_ACTION_FIELDS)),
            }
        }

        let action = PrepareAction {
            version: version.ok_or_else(|| de::Error::missing_field("version"))?,
            plan_digest: plan_digest.ok_or_else(|| de::Error::missing_field("plan_digest"))?,
            synthesis_profile_digest: synthesis_profile_digest
                .ok_or_else(|| de::Error::missing_field("synthesis_profile_digest"))?,
            route_id: route_id.ok_or_else(|| de::Error::missing_field("route_id"))?,
            destination_cell: destination_cell
                .ok_or_else(|| de::Error::missing_field("destination_cell"))?,
            action_id: action_id.ok_or_else(|| de::Error::missing_field("action_id"))?,
            attempt_id: attempt_id.ok_or_else(|| de::Error::missing_field("attempt_id"))?,
            global_sequence: global_sequence
                .ok_or_else(|| de::Error::missing_field("global_sequence"))?,
            ownership_epoch: ownership_epoch
                .ok_or_else(|| de::Error::missing_field("ownership_epoch"))?,
            prior_session_state_version: prior_session_state_version
                .ok_or_else(|| de::Error::missing_field("prior_session_state_version"))?,
            content: content.ok_or_else(|| de::Error::missing_field("content"))?,
        };
        if action.version != STREAMING_CELLULAR_PROTOCOL_VERSION {
            return Err(de::Error::custom("unsupported streaming payload version"));
        }
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> StreamingCellularLimits {
        StreamingCellularLimits {
            max_frame_bytes: 64 * 1024,
            max_payload_bytes: 32 * 1024,
            max_content_items: 4,
            max_content_bytes: 1024,
        }
    }

    fn content(request: &[u8], leases: Vec<ContentLeaseDescriptor>) -> PreparedActionContent {
        let mut content = PreparedActionContent {
            schema: DatasetActionSchema::new("aiperf.stream.action.v1"),
            canonical_request: request.to_vec(),
            item_count: leases.len() as u64,
            byte_length: request.len() as u64,
            content_leases: leases,
            digest: [0; 32],
        };
        content.digest = content.compute_digest();
        content
    }

    fn prepare_action() -> PrepareAction {
        PrepareAction {
            version: STREAMING_CELLULAR_PROTOCOL_VERSION,
            plan_digest: [7; 32],
            synthesis_profile_digest: Some([9; 32]),
            route_id: 3,
            destination_cell: 1,
            action_id: StableActionId::from_bytes([1; 32]),
            attempt_id: ActionAttemptId::from_bytes([2; 32]),
            global_sequence: GlobalSequence::new(42),
            ownership_epoch: SessionOwnershipEpoch::new(5),
            prior_session_state_version: SessionStateVersion::INITIAL,
            content: content(b"{\"model\":\"m\"}", Vec::new()),
        }
    }

    #[test]
    fn prepare_action_round_trips_named_messagepack_and_rejects_unknown_fields() {
        let action = prepare_action();
        let encoded = rmp_serde::to_vec_named(&action).expect("named encoding");
        let decoded = PrepareActionSeed::new(limits())
            .decode(&encoded)
            .expect("named decode");
        assert_eq!(decoded, action);

        // A positional encoding of the same value has no field names, so the
        // named seed cannot read it and `deny_unknown_fields` stays meaningful.
        let positional = rmp_serde::to_vec(&action).expect("positional encoding");
        assert_eq!(
            PrepareActionSeed::new(limits()).decode(&positional),
            Err(AdmissionRejection::Malformed)
        );

        #[derive(Serialize)]
        struct PrepareActionWithExtraField<'a> {
            version: u16,
            plan_digest: [u8; 32],
            synthesis_profile_digest: Option<[u8; 32]>,
            route_id: u32,
            destination_cell: u32,
            action_id: StableActionId,
            attempt_id: ActionAttemptId,
            global_sequence: GlobalSequence,
            ownership_epoch: SessionOwnershipEpoch,
            prior_session_state_version: SessionStateVersion,
            content: &'a PreparedActionContent,
            credential: &'static str,
        }

        let hostile = rmp_serde::to_vec_named(&PrepareActionWithExtraField {
            version: action.version,
            plan_digest: action.plan_digest,
            synthesis_profile_digest: action.synthesis_profile_digest,
            route_id: action.route_id,
            destination_cell: action.destination_cell,
            action_id: action.action_id,
            attempt_id: action.attempt_id,
            global_sequence: action.global_sequence,
            ownership_epoch: action.ownership_epoch,
            prior_session_state_version: action.prior_session_state_version,
            content: &action.content,
            credential: "forged",
        })
        .expect("named encoding");
        let mut deserializer = rmp_serde::Deserializer::from_read_ref(&hostile);
        let error = PrepareActionSeed::new(limits())
            .deserialize(&mut deserializer)
            .expect_err("unknown field must be refused");
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn bounded_seed_refuses_declared_content_beyond_limits() {
        let over_limit = content(&vec![0_u8; 2048], Vec::new());
        let mut action = prepare_action();
        action.content = over_limit;
        let encoded = rmp_serde::to_vec_named(&action).expect("named encoding");
        assert_eq!(
            PrepareActionSeed::new(limits()).decode(&encoded),
            Err(AdmissionRejection::ContentLimitExceeded)
        );

        // A declared count that disagrees with the decoded length is refused
        // even when both sit inside the limits.
        let mut action = prepare_action();
        action.content.item_count = 2;
        let encoded = rmp_serde::to_vec_named(&action).expect("named encoding");
        assert_eq!(
            PrepareActionSeed::new(limits()).decode(&encoded),
            Err(AdmissionRejection::ContentLimitExceeded)
        );
    }

    #[test]
    fn prepared_content_digest_mismatch_is_refused() {
        let mut action = prepare_action();
        action.content.canonical_request[0] ^= 0xFF;
        let encoded = rmp_serde::to_vec_named(&action).expect("named encoding");
        assert_eq!(
            PrepareActionSeed::new(limits()).decode(&encoded),
            Err(AdmissionRejection::Malformed)
        );
    }

    #[test]
    fn placement_events_round_trip_named_messagepack() {
        let identity = WireActionEventIdentity {
            action_id: StableActionId::from_bytes([1; 32]),
            attempt_id: ActionAttemptId::from_bytes([2; 32]),
            ownership_epoch: SessionOwnershipEpoch::new(5),
            event_ordinal: 3,
        };
        let events = [
            CellPlacementEvent::Prepared {
                receipt: PlacementPreparedReceipt {
                    route_id: 3,
                    action_id: identity.action_id,
                    global_sequence: GlobalSequence::new(42),
                    content_digest: [4; 32],
                },
            },
            CellPlacementEvent::Action {
                event: WireActionExecutionEvent::SessionUpdate {
                    event: identity,
                    payload: SessionUpdateBytes::new(b"update".to_vec()).expect("bounded"),
                },
            },
            CellPlacementEvent::Action {
                event: WireActionExecutionEvent::Terminal {
                    event: identity,
                    disposition: WireActionTerminalDisposition::Completed,
                },
            },
            CellPlacementEvent::Failed {
                receipt: PlacementFailureReceipt {
                    route_id: 3,
                    action_id: identity.action_id,
                    code: PlacementFailureCode::StaleOwnershipEpoch,
                },
            },
        ];
        for event in events {
            let encoded = rmp_serde::to_vec_named(&event).expect("named encoding");
            let decoded = CellPlacementEventSeed::new(limits())
                .decode(&encoded)
                .expect("named decode");
            assert_eq!(decoded, event);
        }
    }
}
