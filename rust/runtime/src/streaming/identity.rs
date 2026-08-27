// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable identity vocabulary for native streaming datasets.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::unit::{DatasetActionKind, EventTimeUtc, UnitProvenance};

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

fn domain_hash(domain: &'static [u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, domain);
    for field in fields {
        update_field(&mut hasher, field);
    }
    *hasher.finalize().as_bytes()
}

macro_rules! digest_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Clone,
            Copy,
            Debug,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            serde::Deserialize,
            serde::Serialize,
        )]
        #[serde(transparent)]
        pub struct $name([u8; 32]);

        impl $name {
            /// Construct from canonical digest bytes.
            #[must_use]
            pub const fn from_bytes(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            /// Borrow canonical digest bytes.
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }
        }
    };
}

digest_id!(
    /// Identity of one immutable source-object generation.
    ImmutableObjectIdentity
);
digest_id!(
    /// Stable identity of a physical or producer-keyed source record.
    StableRecordId
);
digest_id!(
    /// Stable identity shared by all fragments in one logical session.
    StableSessionKey
);
digest_id!(
    /// Stable identity of one logical executable dataset action.
    StableActionId
);
digest_id!(
    /// Identity of one incarnation-local attempt of a logical action.
    ActionAttemptId
);
digest_id!(
    /// Stable identity of an independently requested logical replay run.
    LogicalReplayRunId
);
digest_id!(
    /// Process-incarnation identity allocated while acquiring writer authority.
    RunIncarnationId
);
digest_id!(
    /// Canonical content digest used for duplicate classification and receipts.
    ContentDigest
);
digest_id!(
    /// Stable topology-independent tie-break key for equal-time units.
    StableOrderKey
);

/// Monotonic controller-assigned position in global replay order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GlobalSequence(u64);

impl GlobalSequence {
    /// Construct a global sequence value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying sequence value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Proof of the causally complete prefix emitted by the session coordinator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionCausalFrontier {
    /// Greatest global sequence covered by this proof.
    pub through_sequence: GlobalSequence,
    /// Greatest event time proven causally complete, when event time is used.
    pub event_time: Option<EventTimeUtc>,
    /// Digest binding the source and session frontier evidence.
    pub digest: ContentDigest,
}

/// Monotonic fencing epoch for one session route owner.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionOwnershipEpoch(u64);

impl SessionOwnershipEpoch {
    /// Construct an ownership epoch.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying epoch value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Derive the stable identity of a decoder record without a producer key.
#[must_use]
pub fn physical_record_id(
    stream_identity: &[u8],
    partition_generation: &ImmutableObjectIdentity,
    decoder_coordinate: &[u8],
    format_semantic_digest: &[u8; 32],
) -> StableRecordId {
    StableRecordId::from_bytes(domain_hash(
        b"aiperf.stream.physical.v1",
        &[
            stream_identity,
            partition_generation.as_bytes(),
            decoder_coordinate,
            format_semantic_digest,
        ],
    ))
}

/// Derive a logical record identity from a format-validated producer key.
#[must_use]
pub fn stable_record_id_from_key(namespace: &[u8], producer_key: &[u8]) -> StableRecordId {
    StableRecordId::from_bytes(domain_hash(
        b"aiperf.stream.logical-record.v1",
        &[namespace, producer_key],
    ))
}

/// Derive a stable cross-partition session key from a validated producer key.
#[must_use]
pub fn stable_session_key(namespace: &[u8], producer_key: &[u8]) -> StableSessionKey {
    StableSessionKey::from_bytes(domain_hash(
        b"aiperf.stream.session.v1",
        &[namespace, producer_key],
    ))
}

/// Derive the non-joining fallback session key for a one-turn record.
#[must_use]
pub fn one_turn_session_key(record_id: StableRecordId) -> StableSessionKey {
    StableSessionKey::from_bytes(domain_hash(
        b"aiperf.stream.one-turn-session.v1",
        &[record_id.as_bytes()],
    ))
}

/// Derive a logical action identity from semantic and causal inputs only.
#[must_use]
pub fn stable_action_id(
    program_digest: &[u8; 32],
    session: StableSessionKey,
    causes: &[StableRecordId],
    kind: DatasetActionKind,
    causal_ordinal: u64,
) -> StableActionId {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, b"aiperf.stream.action.v1");
    update_field(&mut hasher, program_digest);
    update_field(&mut hasher, session.as_bytes());
    update_field(&mut hasher, &(causes.len() as u64).to_le_bytes());
    for cause in causes {
        update_field(&mut hasher, cause.as_bytes());
    }
    update_field(&mut hasher, &[kind.canonical_tag()]);
    update_field(&mut hasher, &causal_ordinal.to_le_bytes());
    StableActionId::from_bytes(*hasher.finalize().as_bytes())
}

/// Derive the physical identity of one incarnation-local action attempt.
#[must_use]
pub fn attempt_id(
    action: StableActionId,
    incarnation: RunIncarnationId,
    attempt_ordinal: u64,
) -> ActionAttemptId {
    ActionAttemptId::from_bytes(domain_hash(
        b"aiperf.stream.attempt.v1",
        &[
            action.as_bytes(),
            incarnation.as_bytes(),
            &attempt_ordinal.to_le_bytes(),
        ],
    ))
}

/// Minimal receipt required to classify a producer-keyed logical record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalRecordReceipt {
    /// Logical record identity.
    pub record_id: StableRecordId,
    /// Digest of the complete canonical logical content and timing identity.
    pub content_digest: ContentDigest,
    /// Source and format provenance for the observed logical record.
    pub provenance: UnitProvenance,
}

/// Classification of a candidate record relative to an existing receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DuplicateDisposition {
    /// The same logical record and canonical content was observed again.
    Identical,
    /// The candidate has a distinct logical record identity.
    New,
}

/// Stable logical-identity validation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdentityError {
    /// One logical record identity was associated with conflicting content.
    LogicalIdentityConflict {
        /// Previously retained receipt.
        existing: Box<LogicalRecordReceipt>,
        /// Conflicting candidate receipt.
        candidate: Box<LogicalRecordReceipt>,
    },
}

impl IdentityError {
    /// Return the stable machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::LogicalIdentityConflict { .. } => "logical_identity_conflict",
        }
    }
}

impl fmt::Display for IdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogicalIdentityConflict {
                existing,
                candidate,
            } => write!(
                formatter,
                "{}: record {:?} has existing digest {:?} and candidate digest {:?}",
                self.code(),
                existing.record_id,
                existing.content_digest,
                candidate.content_digest
            ),
        }
    }
}

impl std::error::Error for IdentityError {}

/// Classify a logical record candidate using stable identity and canonical content.
pub fn classify_logical_duplicate(
    existing: &LogicalRecordReceipt,
    candidate: &LogicalRecordReceipt,
) -> Result<DuplicateDisposition, IdentityError> {
    if existing.record_id != candidate.record_id {
        return Ok(DuplicateDisposition::New);
    }
    if existing.content_digest == candidate.content_digest {
        return Ok(DuplicateDisposition::Identical);
    }
    Err(IdentityError::LogicalIdentityConflict {
        existing: Box::new(existing.clone()),
        candidate: Box::new(candidate.clone()),
    })
}
