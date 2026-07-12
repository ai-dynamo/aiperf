// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-self-referential durability receipt epochs, targets, events, and journal.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::path::Path;

use crate::spool::{ImmutableClass, PointerClass};
use crate::{
    ArchiveId, ArchiveState, CanonicalJsonError, CanonicalJsonValue, Digest,
    DurabilityFaultInjector, IndexEntry, IndexError, IndexKey, IndexMutationSetV1, IndexPageSource,
    IndexRootV1, IndexSnapshot, MutationMode, QualifiedSpool, SessionId, SpoolError, TimeDomain,
    domain_digest,
};

const LOCAL_RECEIPTS: &str = "LOCAL-RECEIPTS";
const MAX_BATCH_RECORDS: usize = 1_024;
const MAX_BATCH_BYTES: usize = 1 << 20;
const BATCH_MAGIC: &str = "aiperf.archive.receipt-batch.v1";
const HEAD_MAGIC: &str = "aiperf.archive.receipt-head.v1";
const POINTER_MAGIC: &str = "aiperf.archive.receipt-pointer.v1";

macro_rules! digest_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(Digest);

        impl $name {
            #[doc = "Constructs the typed ID from its digest."]
            #[must_use]
            pub const fn new(digest: Digest) -> Self {
                Self(digest)
            }

            #[doc = "Returns the underlying digest."]
            #[must_use]
            pub const fn digest(self) -> Digest {
                self.0
            }
        }
    };
}

digest_id!(
    ReceiptObserverEpochId,
    "Identity of one execution-specific receipt observation Clock epoch."
);
digest_id!(
    ReceiptTargetId,
    "Identity of one immutable WAL-range or remote-publication target."
);
digest_id!(
    ReceiptEventId,
    "Identity of one response-observed or recovery-verified event."
);
digest_id!(
    ReceiptBatchId,
    "Identity of one bounded immutable receipt batch."
);

/// Non-zero runner execution UUID represented as exact bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionId([u8; 16]);

impl ExecutionId {
    /// Constructs a non-zero execution ID.
    pub fn new(bytes: [u8; 16]) -> Result<Self, ReceiptError> {
        if bytes == [0; 16] {
            return Err(ReceiptError::InvalidField("execution_id"));
        }
        Ok(Self(bytes))
    }

    /// Returns the exact UUID bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// One independently persisted observation epoch registered before any event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptObserverEpochV1 {
    /// Domain-separated immutable epoch ID.
    pub observer_epoch_id: ReceiptObserverEpochId,
    /// Runner execution ID.
    pub execution_id: ExecutionId,
    /// Collection session, absent for source-free sync.
    pub telemetry_session_id: Option<SessionId>,
    /// Real or virtual Clock domain.
    pub time_domain: TimeDomain,
    /// Anchor Clock value.
    pub anchor_clock_ns: i64,
    /// Signed Unix epoch nanoseconds, absent for virtual time.
    pub anchor_unix_epoch_ns: Option<i128>,
    /// Anchor acquisition uncertainty.
    pub capture_uncertainty_ns: u64,
    /// Exact runner distribution provenance.
    pub runner_distribution_id: Digest,
}

impl ReceiptObserverEpochV1 {
    /// Constructs and identifies one observer epoch.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        execution_id: ExecutionId,
        telemetry_session_id: Option<SessionId>,
        time_domain: TimeDomain,
        anchor_clock_ns: i64,
        anchor_unix_epoch_ns: Option<i128>,
        capture_uncertainty_ns: u64,
        runner_distribution_id: Digest,
    ) -> Result<Self, ReceiptError> {
        match (time_domain, anchor_unix_epoch_ns) {
            (TimeDomain::Real, Some(_)) | (TimeDomain::Virtual, None) => {}
            _ => return Err(ReceiptError::InvalidField("anchor_unix_epoch_ns")),
        }
        let session = optional_id16(telemetry_session_id.as_ref().map(SessionId::as_bytes));
        let unix = optional_i128(anchor_unix_epoch_ns);
        let domain = [match time_domain {
            TimeDomain::Real => 1,
            TimeDomain::Virtual => 2,
        }];
        let observer_epoch_id = ReceiptObserverEpochId::new(domain_digest(
            "aiperf.archive.receipt-observer-epoch.v1",
            &[
                execution_id.as_bytes(),
                &session,
                &domain,
                &anchor_clock_ns.to_be_bytes(),
                &unix,
                &capture_uncertainty_ns.to_be_bytes(),
                runner_distribution_id.as_bytes(),
            ],
        ));
        Ok(Self {
            observer_epoch_id,
            execution_id,
            telemetry_session_id,
            time_domain,
            anchor_clock_ns,
            anchor_unix_epoch_ns,
            capture_uncertainty_ns,
            runner_distribution_id,
        })
    }

    /// Derives Unix placement only through this epoch's real anchor.
    pub fn unix_ns_at(&self, clock_ns: i64) -> Result<Option<i128>, ReceiptError> {
        self.anchor_unix_epoch_ns
            .map(|unix| {
                unix.checked_add(i128::from(clock_ns) - i128::from(self.anchor_clock_ns))
                    .ok_or(ReceiptError::ArithmeticOverflow)
            })
            .transpose()
    }

    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            ("anchor_clock_ns", integer(i128::from(self.anchor_clock_ns))),
            (
                "anchor_unix_epoch_ns",
                self.anchor_unix_epoch_ns
                    .map_or(CanonicalJsonValue::Null, |value| string(value.to_string())),
            ),
            (
                "capture_uncertainty_ns",
                integer(i128::from(self.capture_uncertainty_ns)),
            ),
            ("execution_id", string(uuid(self.execution_id.as_bytes()))),
            (
                "observer_epoch_id",
                string(self.observer_epoch_id.digest().to_hex()),
            ),
            (
                "runner_distribution_id",
                string(self.runner_distribution_id.to_hex()),
            ),
            (
                "telemetry_session_id",
                self.telemetry_session_id
                    .map_or(CanonicalJsonValue::Null, |id| string(uuid(id.as_bytes()))),
            ),
            (
                "time_domain",
                string(match self.time_domain {
                    TimeDomain::Real => "real",
                    TimeDomain::Virtual => "virtual",
                }),
            ),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "observer_epoch")?;
        let execution_id = ExecutionId::new(parse_uuid(text(fields, "execution_id")?)?)?;
        let session = parse_optional_session(fields.get("telemetry_session_id"))?;
        let time_domain = match text(fields, "time_domain")? {
            "real" => TimeDomain::Real,
            "virtual" => TimeDomain::Virtual,
            _ => return Err(ReceiptError::InvalidField("time_domain")),
        };
        let unix = match fields.get("anchor_unix_epoch_ns") {
            Some(CanonicalJsonValue::Null) => None,
            Some(CanonicalJsonValue::String(value)) => Some(
                value
                    .parse()
                    .map_err(|_| ReceiptError::InvalidField("anchor_unix_epoch_ns"))?,
            ),
            _ => return Err(ReceiptError::InvalidField("anchor_unix_epoch_ns")),
        };
        let epoch = Self::new(
            execution_id,
            session,
            time_domain,
            signed_i64(fields, "anchor_clock_ns")?,
            unix,
            unsigned(fields, "capture_uncertainty_ns")?,
            digest(fields, "runner_distribution_id")?,
        )?;
        if epoch.observer_epoch_id.digest() != digest(fields, "observer_epoch_id")? {
            return Err(ReceiptError::IdentityMismatch("observer_epoch_id"));
        }
        Ok(epoch)
    }
}

/// Stable object-version byte interpretation owned by one store adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObjectVersionKind {
    /// Provider generation/version bytes.
    Generation = 1,
    /// Provider ETag bytes under adapter-defined semantics.
    Etag = 2,
    /// Adapter-owned opaque stable bytes.
    Opaque = 3,
}

impl ObjectVersionKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Generation => "generation",
            Self::Etag => "etag",
            Self::Opaque => "opaque",
        }
    }

    fn parse(value: &str) -> Result<Self, ReceiptError> {
        match value {
            "generation" => Ok(Self::Generation),
            "etag" => Ok(Self::Etag),
            "opaque" => Ok(Self::Opaque),
            _ => Err(ReceiptError::InvalidField("object_version_kind")),
        }
    }
}

/// Provider-neutral stable object version used in publication targets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StableObjectVersion {
    /// Adapter ID defining version-byte equality.
    pub adapter_id: String,
    /// Stable version kind.
    pub kind: ObjectVersionKind,
    /// Exact stable bytes.
    pub value: Vec<u8>,
}

impl StableObjectVersion {
    /// Constructs a nonempty stable version.
    pub fn new(
        adapter_id: impl Into<String>,
        kind: ObjectVersionKind,
        value: Vec<u8>,
    ) -> Result<Self, ReceiptError> {
        let adapter_id = adapter_id.into();
        if adapter_id.is_empty() || value.is_empty() {
            return Err(ReceiptError::InvalidField("stable_object_version"));
        }
        Ok(Self {
            adapter_id,
            kind,
            value,
        })
    }

    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            ("adapter_id", string(self.adapter_id.clone())),
            ("kind", string(self.kind.name())),
            ("value", string(base64url(&self.value))),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "stable_object_version")?;
        Self::new(
            text(fields, "adapter_id")?,
            ObjectVersionKind::parse(text(fields, "kind")?)?,
            decode_base64url(text(fields, "value")?)?,
        )
    }
}

/// Resulting writer-claim state bound into publication identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum WriterClaimState {
    /// Exact writer claim remains active.
    Active = 1,
    /// No writer claim remains.
    Absent = 2,
}

impl WriterClaimState {
    const fn name(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Absent => "absent",
        }
    }

    fn parse(value: &str) -> Result<Self, ReceiptError> {
        match value {
            "active" => Ok(Self::Active),
            "absent" => Ok(Self::Absent),
            _ => Err(ReceiptError::InvalidField("writer_claim_state")),
        }
    }
}

/// Immutable WAL-range receipt target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalRangeTargetV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// One exact WAL segment ID.
    pub wal_segment_id: Digest,
    /// Durable ordered-prefix hash.
    pub durable_prefix_hash: Digest,
    /// Inclusive first global sequence.
    pub first_record_seq: u64,
    /// Inclusive last global sequence.
    pub last_record_seq: u64,
    /// Ascending projection-coverage digest.
    pub projection_coverage_digest: Digest,
}

/// Immutable remote-publication receipt target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemotePublicationTargetV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Sealed generation hash.
    pub generation_hash: Digest,
    /// Complete index-root hash.
    pub index_root_hash: Digest,
    /// Installed head hash.
    pub installed_head_hash: Digest,
    /// Verified provider CAS version.
    pub object_version: StableObjectVersion,
    /// Resulting archive state.
    pub archive_state: ArchiveState,
    /// Resulting writer-claim state.
    pub writer_claim_state: WriterClaimState,
}

/// Closed receipt-target discriminant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ReceiptTargetKind {
    /// One contiguous range in exactly one WAL segment.
    WalRange = 1,
    /// One verified remote head publication.
    RemotePublication = 2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TargetBody {
    WalRange(WalRangeTargetV1),
    RemotePublication(RemotePublicationTargetV1),
}

/// One immutable target whose ID contains no observation time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptTargetV1 {
    /// Domain-separated target identity.
    pub receipt_target_id: ReceiptTargetId,
    body: TargetBody,
}

impl ReceiptTargetV1 {
    /// Constructs one single-segment contiguous WAL target.
    pub fn wal_range(target: WalRangeTargetV1) -> Result<Self, ReceiptError> {
        if target.first_record_seq > target.last_record_seq {
            return Err(ReceiptError::InvalidField("wal_range"));
        }
        Self::from_body(TargetBody::WalRange(target))
    }

    /// Constructs one remote-publication target.
    pub fn remote_publication(target: RemotePublicationTargetV1) -> Result<Self, ReceiptError> {
        if target.archive_state == ArchiveState::RemotelyFinalized
            && target.writer_claim_state != WriterClaimState::Absent
        {
            return Err(ReceiptError::InvalidField("terminal_writer_claim"));
        }
        Self::from_body(TargetBody::RemotePublication(target))
    }

    /// Returns the target discriminant.
    #[must_use]
    pub const fn kind(&self) -> ReceiptTargetKind {
        match self.body {
            TargetBody::WalRange(_) => ReceiptTargetKind::WalRange,
            TargetBody::RemotePublication(_) => ReceiptTargetKind::RemotePublication,
        }
    }

    /// Returns the archive identity.
    #[must_use]
    pub const fn archive_id(&self) -> ArchiveId {
        match &self.body {
            TargetBody::WalRange(target) => target.archive_id,
            TargetBody::RemotePublication(target) => target.archive_id,
        }
    }

    /// Returns WAL details when applicable.
    #[must_use]
    pub fn as_wal_range(&self) -> Option<&WalRangeTargetV1> {
        match &self.body {
            TargetBody::WalRange(target) => Some(target),
            TargetBody::RemotePublication(_) => None,
        }
    }

    /// Returns publication details when applicable.
    #[must_use]
    pub fn as_remote_publication(&self) -> Option<&RemotePublicationTargetV1> {
        match &self.body {
            TargetBody::RemotePublication(target) => Some(target),
            TargetBody::WalRange(_) => None,
        }
    }

    fn from_body(body: TargetBody) -> Result<Self, ReceiptError> {
        let bytes = target_value(&body).to_bytes();
        Ok(Self {
            receipt_target_id: ReceiptTargetId::new(domain_digest(
                "aiperf.archive.receipt-target.v1",
                &[&bytes],
            )),
            body,
        })
    }

    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            (
                "receipt_target_id",
                string(self.receipt_target_id.digest().to_hex()),
            ),
            ("target", target_value(&self.body)),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "receipt_target")?;
        let target = fields
            .get("target")
            .ok_or(ReceiptError::InvalidField("target"))?;
        let body = as_object(target, "target")?;
        let decoded = match text(body, "kind")? {
            "wal_range" => Self::wal_range(WalRangeTargetV1 {
                archive_id: parse_archive_id(text(body, "archive_id")?)?,
                session_id: parse_session_id(text(body, "session_id")?)?,
                wal_segment_id: digest(body, "wal_segment_id")?,
                durable_prefix_hash: digest(body, "durable_prefix_hash")?,
                first_record_seq: unsigned(body, "first_record_seq")?,
                last_record_seq: unsigned(body, "last_record_seq")?,
                projection_coverage_digest: digest(body, "projection_coverage_digest")?,
            })?,
            "remote_publication" => Self::remote_publication(RemotePublicationTargetV1 {
                archive_id: parse_archive_id(text(body, "archive_id")?)?,
                generation_hash: digest(body, "generation_hash")?,
                index_root_hash: digest(body, "index_root_hash")?,
                installed_head_hash: digest(body, "installed_head_hash")?,
                object_version: StableObjectVersion::from_value(
                    body.get("object_version")
                        .ok_or(ReceiptError::InvalidField("object_version"))?,
                )?,
                archive_state: parse_archive_state(text(body, "archive_state")?)?,
                writer_claim_state: WriterClaimState::parse(text(body, "writer_claim_state")?)?,
            })?,
            _ => return Err(ReceiptError::InvalidField("target_kind")),
        };
        if decoded.receipt_target_id.digest() != digest(fields, "receipt_target_id")? {
            return Err(ReceiptError::IdentityMismatch("receipt_target_id"));
        }
        Ok(decoded)
    }
}

/// Observation kind; recovery verification never backfills response time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObservationKind {
    /// LocalSet observed the immutable completion response.
    ResponseObserved = 1,
    /// A later execution independently verified the target.
    RecoveryVerified = 2,
}

impl ObservationKind {
    const fn name(self) -> &'static str {
        match self {
            Self::ResponseObserved => "response_observed",
            Self::RecoveryVerified => "recovery_verified",
        }
    }
}

/// One immutable observation event for an earlier target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptEventV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Journal-global event sequence.
    pub receipt_seq: u64,
    /// Immutable target identity.
    pub receipt_target_id: ReceiptTargetId,
    /// Observation epoch identity.
    pub observer_epoch_id: ReceiptObserverEpochId,
    /// Response or recovery observation.
    pub observation_kind: ObservationKind,
    /// Exact Clock value under the epoch.
    pub observation_clock_ns: i64,
    /// Domain-separated event identity.
    pub event_id: ReceiptEventId,
}

impl ReceiptEventV1 {
    /// Constructs one event with exactly one typed observation Clock value.
    #[must_use]
    pub fn new(
        archive_id: ArchiveId,
        receipt_seq: u64,
        receipt_target_id: ReceiptTargetId,
        observer_epoch_id: ReceiptObserverEpochId,
        observation_kind: ObservationKind,
        observation_clock_ns: i64,
    ) -> Self {
        let kind = [observation_kind as u8];
        let event_id = ReceiptEventId::new(domain_digest(
            "aiperf.archive.receipt-event.v1",
            &[
                receipt_target_id.digest().as_bytes(),
                observer_epoch_id.digest().as_bytes(),
                &kind,
                &observation_clock_ns.to_be_bytes(),
            ],
        ));
        Self {
            archive_id,
            receipt_seq,
            receipt_target_id,
            observer_epoch_id,
            observation_kind,
            observation_clock_ns,
            event_id,
        }
    }

    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            ("event_id", string(self.event_id.digest().to_hex())),
            (
                "observation_clock_ns",
                integer(i128::from(self.observation_clock_ns)),
            ),
            ("observation_kind", string(self.observation_kind.name())),
            (
                "observer_epoch_id",
                string(self.observer_epoch_id.digest().to_hex()),
            ),
            ("receipt_seq", integer(i128::from(self.receipt_seq))),
            (
                "receipt_target_id",
                string(self.receipt_target_id.digest().to_hex()),
            ),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "receipt_event")?;
        let kind = match text(fields, "observation_kind")? {
            "response_observed" => ObservationKind::ResponseObserved,
            "recovery_verified" => ObservationKind::RecoveryVerified,
            _ => return Err(ReceiptError::InvalidField("observation_kind")),
        };
        let event = Self::new(
            parse_archive_id(text(fields, "archive_id")?)?,
            unsigned(fields, "receipt_seq")?,
            ReceiptTargetId::new(digest(fields, "receipt_target_id")?),
            ReceiptObserverEpochId::new(digest(fields, "observer_epoch_id")?),
            kind,
            signed_i64(fields, "observation_clock_ns")?,
        );
        if event.event_id.digest() != digest(fields, "event_id")? {
            return Err(ReceiptError::IdentityMismatch("event_id"));
        }
        Ok(event)
    }
}

/// One immutable bounded batch: epochs/targets sorted, then events by sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptBatchV1 {
    /// Batch content identity.
    pub batch_id: ReceiptBatchId,
    /// Sorted observer epochs.
    pub observer_epochs: Vec<ReceiptObserverEpochV1>,
    /// Sorted targets.
    pub targets: Vec<ReceiptTargetV1>,
    /// Strictly increasing events.
    pub events: Vec<ReceiptEventV1>,
    bytes: Vec<u8>,
}

impl ReceiptBatchV1 {
    /// Validates ordering and size bounds and constructs one batch.
    pub fn new(
        mut epochs: Vec<ReceiptObserverEpochV1>,
        mut targets: Vec<ReceiptTargetV1>,
        mut events: Vec<ReceiptEventV1>,
    ) -> Result<Self, ReceiptError> {
        epochs.sort_unstable_by_key(|epoch| epoch.observer_epoch_id);
        targets.sort_unstable_by_key(|target| target.receipt_target_id);
        events.sort_unstable_by_key(|event| event.receipt_seq);
        unique(
            epochs.iter().map(|epoch| epoch.observer_epoch_id.digest()),
            "observer_epoch",
        )?;
        unique(
            targets
                .iter()
                .map(|target| target.receipt_target_id.digest()),
            "target",
        )?;
        unique(events.iter().map(|event| event.event_id.digest()), "event")?;
        if events
            .windows(2)
            .any(|pair| pair[0].receipt_seq >= pair[1].receipt_seq)
        {
            return Err(ReceiptError::DuplicateRecord("receipt_seq"));
        }
        let count = epochs.len() + targets.len() + events.len();
        if count == 0 || count > MAX_BATCH_RECORDS {
            return Err(ReceiptError::BatchRecordLimit(count));
        }
        let bytes = batch_value(&epochs, &targets, &events).to_bytes();
        if bytes.len() > MAX_BATCH_BYTES {
            return Err(ReceiptError::BatchByteLimit(bytes.len()));
        }
        let batch_id =
            ReceiptBatchId::new(domain_digest("aiperf.archive.receipt-batch.v1", &[&bytes]));
        Ok(Self {
            batch_id,
            observer_epochs: epochs,
            targets,
            events,
            bytes,
        })
    }

    /// Returns exact canonical bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decodes and verifies exact canonical batch bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, ReceiptError> {
        let value = CanonicalJsonValue::parse_canonical(bytes).map_err(ReceiptError::Canonical)?;
        let fields = as_object(&value, "receipt_batch")?;
        if text(fields, "magic")? != BATCH_MAGIC || unsigned(fields, "version")? != 1 {
            return Err(ReceiptError::InvalidField("batch_magic"));
        }
        let epochs = as_array(fields.get("observer_epochs"), "observer_epochs")?
            .iter()
            .map(ReceiptObserverEpochV1::from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let targets = as_array(fields.get("targets"), "targets")?
            .iter()
            .map(ReceiptTargetV1::from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let events = as_array(fields.get("events"), "events")?
            .iter()
            .map(ReceiptEventV1::from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let batch = Self::new(epochs, targets, events)?;
        if batch.bytes != bytes {
            return Err(ReceiptError::Canonical(CanonicalJsonError::NonCanonical));
        }
        Ok(batch)
    }
}

/// Tagged receipt-index key authority.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReceiptIndexKeyV1;

impl ReceiptIndexKeyV1 {
    /// Builds `0x01 || observer_epoch_id`.
    #[must_use]
    pub fn observer_epoch(id: ReceiptObserverEpochId) -> IndexKey {
        let mut bytes = Vec::with_capacity(33);
        bytes.push(0x01);
        bytes.extend_from_slice(id.digest().as_bytes());
        IndexKey::new(bytes).expect("tagged epoch key is nonempty")
    }

    /// Builds the exact target-kind/session/range/generation tagged key.
    #[must_use]
    pub fn target(target: &ReceiptTargetV1) -> IndexKey {
        let mut bytes = Vec::with_capacity(90);
        bytes.push(0x02);
        bytes.push(target.kind() as u8);
        match &target.body {
            TargetBody::WalRange(target) => {
                bytes.extend_from_slice(target.session_id.as_bytes());
                bytes.extend_from_slice(&target.first_record_seq.to_be_bytes());
                bytes.extend_from_slice(&[0; 32]);
            }
            TargetBody::RemotePublication(target) => {
                bytes.extend_from_slice(&[0; 16]);
                bytes.extend_from_slice(&[0; 8]);
                bytes.extend_from_slice(target.generation_hash.as_bytes());
            }
        }
        bytes.extend_from_slice(target.receipt_target_id.digest().as_bytes());
        IndexKey::new(bytes).expect("tagged target key is nonempty")
    }

    /// Builds `0x03 || target_id || receipt_seq || event_id`.
    #[must_use]
    pub fn event(event: &ReceiptEventV1) -> IndexKey {
        let mut bytes = Vec::with_capacity(73);
        bytes.push(0x03);
        bytes.extend_from_slice(event.receipt_target_id.digest().as_bytes());
        bytes.extend_from_slice(&event.receipt_seq.to_be_bytes());
        bytes.extend_from_slice(event.event_id.digest().as_bytes());
        IndexKey::new(bytes).expect("tagged event key is nonempty")
    }
}

/// Computes aggregate WAL coverage in ascending contiguous record order.
pub fn receipt_range_coverage(mut frames: Vec<(u64, Digest)>) -> Result<Digest, ReceiptError> {
    frames.sort_unstable_by_key(|(sequence, _)| *sequence);
    if frames
        .windows(2)
        .any(|pair| pair[0].0.checked_add(1) != Some(pair[1].0))
    {
        return Err(ReceiptError::NonContiguousWalRange);
    }
    let encoded: Vec<Vec<u8>> = frames
        .iter()
        .map(|(sequence, digest)| {
            let mut value = Vec::with_capacity(40);
            value.extend_from_slice(&sequence.to_be_bytes());
            value.extend_from_slice(digest.as_bytes());
            value
        })
        .collect();
    let fields: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
    Ok(domain_digest(
        "aiperf.archive.receipt-range-coverage.v1",
        &fields,
    ))
}

// The persistent journal implementation follows the record definitions so all
// transaction code consumes these frozen identities rather than re-encoding them.

/// Persistent receipt journal under an already-held qualified spool lock.
pub struct ReceiptJournal<'a> {
    spool: &'a QualifiedSpool,
    archive_id: ArchiveId,
    head: ReceiptHead,
    head_hash: Digest,
    index: IndexSnapshot,
    rolled_back_current: bool,
}

impl Debug for ReceiptJournal<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReceiptJournal")
            .field("archive_id", &self.archive_id)
            .field("head", &self.head)
            .field("rolled_back_current", &self.rolled_back_current)
            .finish_non_exhaustive()
    }
}

impl<'a> ReceiptJournal<'a> {
    /// Creates the mandatory epoch-only bootstrap transaction.
    pub fn bootstrap(
        spool: &'a QualifiedSpool,
        archive_id: ArchiveId,
        epoch: ReceiptObserverEpochV1,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<Self, ReceiptError> {
        if spool.path().join(LOCAL_RECEIPTS).exists() {
            return Err(ReceiptError::AlreadyExists);
        }
        let batch = ReceiptBatchV1::new(vec![epoch.clone()], vec![], vec![])?;
        let empty = IndexSnapshot::empty().map_err(ReceiptError::Index)?;
        let entry = record_entry(
            ReceiptIndexKeyV1::observer_epoch(epoch.observer_epoch_id),
            epoch.value(),
        )?;
        let index = empty
            .apply(
                &IndexMutationSetV1::new(vec![], vec![entry]).map_err(ReceiptError::Index)?,
                MutationMode::Normal,
            )
            .map_err(ReceiptError::Index)?;
        write_batch(spool, &batch, faults)?;
        write_receipt_index(spool, &index, faults)?;
        let head = ReceiptHead {
            archive_id,
            commit_seq: 0,
            parent_head_hash: None,
            batch_id: batch.batch_id,
            index_root: index.root().clone(),
            observer_epoch_count: 1,
            target_count: 0,
            event_count: 0,
            last_receipt_seq: None,
        };
        let head_object = ReceiptHeadObject::new(head.clone())?;
        write_head(spool, &head_object, faults)?;
        let pointer = ReceiptPointer {
            archive_id,
            current: head_object.reference(),
            preceding: None,
        };
        write_pointer(spool, &pointer, faults)?;
        Ok(Self {
            spool,
            archive_id,
            head,
            head_hash: head_object.hash,
            index,
            rolled_back_current: false,
        })
    }

    /// Recovers from the checksummed receipt pointer and indexed object graph.
    pub fn recover(
        spool: &'a QualifiedSpool,
        archive_id: ArchiveId,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<Self, ReceiptError> {
        let bytes = spool
            .read_relative(Path::new(LOCAL_RECEIPTS))
            .map_err(ReceiptError::Spool)?;
        let pointer = ReceiptPointer::decode(&bytes)?;
        if pointer.archive_id != archive_id {
            return Err(ReceiptError::IdentityMismatch("receipt_archive_id"));
        }
        let current = verify_head(spool, &pointer.current, archive_id);
        let (head, hash, index, rolled_back_current) = match current {
            Ok((head, hash, index)) => (head, hash, index, false),
            Err(current_error) => {
                let Some(preceding) = pointer.preceding.as_ref() else {
                    return Err(current_error);
                };
                match verify_head(spool, preceding, archive_id) {
                    Ok((head, hash, index)) => {
                        write_pointer(
                            spool,
                            &ReceiptPointer {
                                archive_id,
                                current: preceding.clone(),
                                preceding: None,
                            },
                            faults,
                        )?;
                        (head, hash, index, true)
                    }
                    Err(preceding_error) => {
                        return Err(ReceiptError::NoValidHead {
                            current: current_error.to_string(),
                            preceding: preceding_error.to_string(),
                        });
                    }
                }
            }
        };
        Ok(Self {
            spool,
            archive_id,
            head,
            head_hash: hash,
            index,
            rolled_back_current,
        })
    }

    /// Returns the durable observer-epoch count.
    #[must_use]
    pub const fn observer_epoch_count(&self) -> u64 {
        self.head.observer_epoch_count
    }

    /// Returns the durable immutable-target count.
    #[must_use]
    pub const fn target_count(&self) -> u64 {
        self.head.target_count
    }

    /// Returns the durable event count.
    #[must_use]
    pub const fn event_count(&self) -> u64 {
        self.head.event_count
    }

    /// Returns the last event sequence, absent before any event.
    #[must_use]
    pub const fn last_receipt_seq(&self) -> Option<u64> {
        self.head.last_receipt_seq
    }

    /// Reports a recovery rollback from current to preceding receipt head.
    #[must_use]
    pub const fn rolled_back_current(&self) -> bool {
        self.rolled_back_current
    }

    /// Persists another execution epoch before that execution observes completion.
    pub fn append_observer_epoch(
        &mut self,
        epoch: ReceiptObserverEpochV1,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<(), ReceiptError> {
        let entry = record_entry(
            ReceiptIndexKeyV1::observer_epoch(epoch.observer_epoch_id),
            epoch.value(),
        )?;
        let batch = ReceiptBatchV1::new(vec![epoch], vec![], vec![])?;
        self.commit(batch, vec![entry], 1, 0, 0, None, faults)
    }

    /// Persists an immutable target and one later observation event.
    pub fn record_event(
        &mut self,
        target: ReceiptTargetV1,
        event: ReceiptEventV1,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<(), ReceiptError> {
        if target.archive_id() != self.archive_id
            || event.archive_id != self.archive_id
            || event.receipt_target_id != target.receipt_target_id
        {
            return Err(ReceiptError::IdentityMismatch("target_event_archive"));
        }
        let expected = self.head.last_receipt_seq.map_or(0, |value| value + 1);
        if event.receipt_seq != expected {
            return Err(ReceiptError::ReceiptSequence {
                expected,
                actual: event.receipt_seq,
            });
        }
        if self
            .index
            .get(&ReceiptIndexKeyV1::observer_epoch(event.observer_epoch_id))
            .is_none()
        {
            return Err(ReceiptError::MissingObserverEpoch(event.observer_epoch_id));
        }
        let target_key = ReceiptIndexKeyV1::target(&target);
        let target_bytes = target.value().to_bytes();
        let target_is_new = match self.index.get(&target_key) {
            Some(existing) if existing.descriptor_bytes() == target_bytes => false,
            Some(_) => return Err(ReceiptError::IdentityMismatch("target_descriptor")),
            None => true,
        };
        let mut additions = Vec::new();
        let batch_targets = if target_is_new {
            additions.push(IndexEntry::new(target_key, target_bytes).map_err(ReceiptError::Index)?);
            vec![target]
        } else {
            vec![]
        };
        additions.push(record_entry(
            ReceiptIndexKeyV1::event(&event),
            event.value(),
        )?);
        let batch = ReceiptBatchV1::new(vec![], batch_targets, vec![event.clone()])?;
        self.commit(
            batch,
            additions,
            0,
            u64::from(target_is_new),
            1,
            Some(event.receipt_seq),
            faults,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        batch: ReceiptBatchV1,
        additions: Vec<IndexEntry>,
        epoch_delta: u64,
        target_delta: u64,
        event_delta: u64,
        last_receipt_seq: Option<u64>,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<(), ReceiptError> {
        let mutation = IndexMutationSetV1::new(vec![], additions).map_err(ReceiptError::Index)?;
        let next_index = self
            .index
            .apply(&mutation, MutationMode::Normal)
            .map_err(ReceiptError::Index)?;
        write_batch(self.spool, &batch, faults)?;
        write_receipt_index(self.spool, &next_index, faults)?;
        let next_head = ReceiptHead {
            archive_id: self.archive_id,
            commit_seq: self
                .head
                .commit_seq
                .checked_add(1)
                .ok_or(ReceiptError::ArithmeticOverflow)?,
            parent_head_hash: Some(self.head_hash),
            batch_id: batch.batch_id,
            index_root: next_index.root().clone(),
            observer_epoch_count: self
                .head
                .observer_epoch_count
                .checked_add(epoch_delta)
                .ok_or(ReceiptError::ArithmeticOverflow)?,
            target_count: self
                .head
                .target_count
                .checked_add(target_delta)
                .ok_or(ReceiptError::ArithmeticOverflow)?,
            event_count: self
                .head
                .event_count
                .checked_add(event_delta)
                .ok_or(ReceiptError::ArithmeticOverflow)?,
            last_receipt_seq: last_receipt_seq.or(self.head.last_receipt_seq),
        };
        let next_object = ReceiptHeadObject::new(next_head.clone())?;
        write_head(self.spool, &next_object, faults)?;
        write_pointer(
            self.spool,
            &ReceiptPointer {
                archive_id: self.archive_id,
                current: next_object.reference(),
                preceding: Some(ReceiptHeadReference {
                    commit_seq: self.head.commit_seq,
                    hash: self.head_hash,
                    key: receipt_head_key(self.head.commit_seq, self.head_hash),
                }),
            },
            faults,
        )?;
        self.head = next_head;
        self.head_hash = next_object.hash;
        self.index = next_index;
        self.rolled_back_current = false;
        Ok(())
    }
}

/// Receipt schema, identity, transaction, or recovery failure.
#[derive(Debug)]
pub enum ReceiptError {
    /// Required field is absent or invalid.
    InvalidField(&'static str),
    /// Recomputed immutable identity differs from stored identity.
    IdentityMismatch(&'static str),
    /// Duplicate record exists in one batch.
    DuplicateRecord(&'static str),
    /// Batch record count is outside `1..=1024`.
    BatchRecordLimit(usize),
    /// Batch bytes exceed 1 MiB.
    BatchByteLimit(usize),
    /// WAL range is not contiguous.
    NonContiguousWalRange,
    /// Event sequence is not exactly next.
    ReceiptSequence {
        /// Expected next sequence.
        expected: u64,
        /// Supplied sequence.
        actual: u64,
    },
    /// Event references an epoch not yet durable.
    MissingObserverEpoch(ReceiptObserverEpochId),
    /// Receipt pointer already exists during bootstrap.
    AlreadyExists,
    /// Neither pointer head verifies.
    NoValidHead {
        /// Current head failure.
        current: String,
        /// Preceding head failure.
        preceding: String,
    },
    /// Checked arithmetic overflowed.
    ArithmeticOverflow,
    /// Canonical JSON failed.
    Canonical(CanonicalJsonError),
    /// Persistent index failed.
    Index(IndexError),
    /// Qualified spool transaction failed.
    Spool(SpoolError),
}

impl Display for ReceiptError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid receipt field {field}"),
            Self::IdentityMismatch(field) => {
                write!(formatter, "receipt identity mismatch: {field}")
            }
            Self::DuplicateRecord(kind) => write!(formatter, "duplicate receipt {kind}"),
            Self::BatchRecordLimit(count) => write!(
                formatter,
                "receipt batch record count {count} is outside 1..=1024"
            ),
            Self::BatchByteLimit(bytes) => {
                write!(formatter, "receipt batch has {bytes} bytes above 1 MiB")
            }
            Self::NonContiguousWalRange => {
                formatter.write_str("receipt WAL range is not contiguous")
            }
            Self::ReceiptSequence { expected, actual } => write!(
                formatter,
                "receipt sequence mismatch: expected {expected}, found {actual}"
            ),
            Self::MissingObserverEpoch(id) => write!(
                formatter,
                "receipt observer epoch {} is not durable",
                id.digest()
            ),
            Self::AlreadyExists => formatter.write_str("LOCAL-RECEIPTS already exists"),
            Self::NoValidHead { current, preceding } => write!(
                formatter,
                "current receipt head invalid ({current}); preceding invalid ({preceding})"
            ),
            Self::ArithmeticOverflow => formatter.write_str("receipt arithmetic overflow"),
            Self::Canonical(error) => write!(formatter, "canonical receipt JSON failed: {error}"),
            Self::Index(error) => write!(formatter, "receipt index failed: {error}"),
            Self::Spool(error) => write!(formatter, "receipt spool failed: {error}"),
        }
    }
}

impl std::error::Error for ReceiptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Canonical(error) => Some(error),
            Self::Index(error) => Some(error),
            Self::Spool(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReceiptHead {
    archive_id: ArchiveId,
    commit_seq: u64,
    parent_head_hash: Option<Digest>,
    batch_id: ReceiptBatchId,
    index_root: IndexRootV1,
    observer_epoch_count: u64,
    target_count: u64,
    event_count: u64,
    last_receipt_seq: Option<u64>,
}

impl ReceiptHead {
    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            ("batch_id", string(self.batch_id.digest().to_hex())),
            ("commit_seq", integer(i128::from(self.commit_seq))),
            ("event_count", integer(i128::from(self.event_count))),
            ("index_root", index_root_value(&self.index_root)),
            (
                "last_receipt_seq",
                self.last_receipt_seq
                    .map_or(CanonicalJsonValue::Null, |value| integer(i128::from(value))),
            ),
            (
                "observer_epoch_count",
                integer(i128::from(self.observer_epoch_count)),
            ),
            (
                "parent_head_hash",
                self.parent_head_hash
                    .map_or(CanonicalJsonValue::Null, |hash| string(hash.to_hex())),
            ),
            ("target_count", integer(i128::from(self.target_count))),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "receipt_head")?;
        Ok(Self {
            archive_id: parse_archive_id(text(fields, "archive_id")?)?,
            commit_seq: unsigned(fields, "commit_seq")?,
            parent_head_hash: parse_optional_digest(fields.get("parent_head_hash"))?,
            batch_id: ReceiptBatchId::new(digest(fields, "batch_id")?),
            index_root: parse_index_root(
                fields
                    .get("index_root")
                    .ok_or(ReceiptError::InvalidField("index_root"))?,
            )?,
            observer_epoch_count: unsigned(fields, "observer_epoch_count")?,
            target_count: unsigned(fields, "target_count")?,
            event_count: unsigned(fields, "event_count")?,
            last_receipt_seq: parse_optional_u64(fields.get("last_receipt_seq"))?,
        })
    }
}

struct ReceiptHeadObject {
    head: ReceiptHead,
    hash: Digest,
    key: String,
    bytes: Vec<u8>,
}

impl ReceiptHeadObject {
    fn new(head: ReceiptHead) -> Result<Self, ReceiptError> {
        if (head.commit_seq == 0) != head.parent_head_hash.is_none() {
            return Err(ReceiptError::InvalidField("parent_head_hash"));
        }
        let bytes = envelope(HEAD_MAGIC, head.value());
        let hash = domain_digest("aiperf.archive.receipt-event.v1", &[&bytes]);
        let key = receipt_head_key(head.commit_seq, hash);
        Ok(Self {
            head,
            hash,
            key,
            bytes,
        })
    }

    fn decode(bytes: &[u8]) -> Result<Self, ReceiptError> {
        let head = ReceiptHead::from_value(&decode_envelope(HEAD_MAGIC, bytes)?)?;
        let object = Self::new(head)?;
        if object.bytes != bytes {
            return Err(ReceiptError::Canonical(CanonicalJsonError::NonCanonical));
        }
        Ok(object)
    }

    fn reference(&self) -> ReceiptHeadReference {
        ReceiptHeadReference {
            commit_seq: self.head.commit_seq,
            hash: self.hash,
            key: self.key.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReceiptHeadReference {
    commit_seq: u64,
    hash: Digest,
    key: String,
}

impl ReceiptHeadReference {
    fn value(&self) -> CanonicalJsonValue {
        object(vec![
            ("commit_seq", integer(i128::from(self.commit_seq))),
            ("hash", string(self.hash.to_hex())),
            ("key", string(self.key.clone())),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ReceiptError> {
        let fields = as_object(value, "head_reference")?;
        let reference = Self {
            commit_seq: unsigned(fields, "commit_seq")?,
            hash: digest(fields, "hash")?,
            key: text(fields, "key")?.to_owned(),
        };
        if reference.key != receipt_head_key(reference.commit_seq, reference.hash) {
            return Err(ReceiptError::InvalidField("head_key"));
        }
        Ok(reference)
    }
}

struct ReceiptPointer {
    archive_id: ArchiveId,
    current: ReceiptHeadReference,
    preceding: Option<ReceiptHeadReference>,
}

impl ReceiptPointer {
    fn bytes(&self) -> Vec<u8> {
        envelope(
            POINTER_MAGIC,
            object(vec![
                ("archive_id", string(uuid(self.archive_id.as_bytes()))),
                ("current", self.current.value()),
                (
                    "preceding",
                    self.preceding
                        .as_ref()
                        .map_or(CanonicalJsonValue::Null, ReceiptHeadReference::value),
                ),
            ]),
        )
    }

    fn decode(bytes: &[u8]) -> Result<Self, ReceiptError> {
        let payload = decode_envelope(POINTER_MAGIC, bytes)?;
        let fields = as_object(&payload, "receipt_pointer")?;
        let current = ReceiptHeadReference::from_value(
            fields
                .get("current")
                .ok_or(ReceiptError::InvalidField("current"))?,
        )?;
        let preceding = match fields.get("preceding") {
            Some(CanonicalJsonValue::Null) => None,
            Some(value) => Some(ReceiptHeadReference::from_value(value)?),
            None => return Err(ReceiptError::InvalidField("preceding")),
        };
        if preceding
            .as_ref()
            .is_some_and(|head| current.commit_seq != head.commit_seq + 1)
        {
            return Err(ReceiptError::InvalidField("head_link"));
        }
        Ok(Self {
            archive_id: parse_archive_id(text(fields, "archive_id")?)?,
            current,
            preceding,
        })
    }
}

fn verify_head(
    spool: &QualifiedSpool,
    reference: &ReceiptHeadReference,
    archive_id: ArchiveId,
) -> Result<(ReceiptHead, Digest, IndexSnapshot), ReceiptError> {
    let bytes = spool
        .read_relative(Path::new(&reference.key))
        .map_err(ReceiptError::Spool)?;
    let object = ReceiptHeadObject::decode(&bytes)?;
    if object.hash != reference.hash
        || object.key != reference.key
        || object.head.commit_seq != reference.commit_seq
        || object.head.archive_id != archive_id
    {
        return Err(ReceiptError::IdentityMismatch("receipt_head"));
    }
    verify_ancestry(spool, &object)?;
    let batch = spool
        .read_relative(Path::new(&receipt_batch_key(object.head.batch_id)))
        .map_err(ReceiptError::Spool)?;
    if ReceiptBatchV1::decode(&batch)?.batch_id != object.head.batch_id {
        return Err(ReceiptError::IdentityMismatch("receipt_batch"));
    }
    let index = IndexSnapshot::load(object.head.index_root.clone(), &ReceiptPageSource { spool })
        .map_err(ReceiptError::Index)?;
    validate_counts(&index, &object.head)?;
    Ok((object.head, object.hash, index))
}

fn verify_ancestry(
    spool: &QualifiedSpool,
    current: &ReceiptHeadObject,
) -> Result<(), ReceiptError> {
    let mut head = current.head.clone();
    while head.commit_seq != 0 {
        let parent_hash = head
            .parent_head_hash
            .ok_or(ReceiptError::InvalidField("parent_head_hash"))?;
        let key = receipt_head_key(head.commit_seq - 1, parent_hash);
        let bytes = spool
            .read_relative(Path::new(&key))
            .map_err(ReceiptError::Spool)?;
        let parent = ReceiptHeadObject::decode(&bytes)?;
        if parent.hash != parent_hash || parent.head.archive_id != head.archive_id {
            return Err(ReceiptError::IdentityMismatch("receipt_ancestry"));
        }
        let batch = spool
            .read_relative(Path::new(&receipt_batch_key(parent.head.batch_id)))
            .map_err(ReceiptError::Spool)?;
        if ReceiptBatchV1::decode(&batch)?.batch_id != parent.head.batch_id {
            return Err(ReceiptError::IdentityMismatch("receipt_ancestry_batch"));
        }
        head = parent.head;
    }
    if head.parent_head_hash.is_some() {
        return Err(ReceiptError::InvalidField("receipt_genesis_parent"));
    }
    Ok(())
}

fn validate_counts(index: &IndexSnapshot, head: &ReceiptHead) -> Result<(), ReceiptError> {
    let mut epochs = 0_u64;
    let mut targets = 0_u64;
    let mut sequences = BTreeSet::new();
    for entry in index.entries() {
        match entry.key().as_bytes().first() {
            Some(0x01) => epochs += 1,
            Some(0x02) => targets += 1,
            Some(0x03) => {
                let bytes = entry.key().as_bytes();
                if bytes.len() != 73 {
                    return Err(ReceiptError::InvalidField("event_key"));
                }
                sequences.insert(u64::from_be_bytes(
                    bytes[33..41].try_into().expect("event key length checked"),
                ));
            }
            _ => return Err(ReceiptError::InvalidField("receipt_index_tag")),
        }
    }
    let event_count =
        u64::try_from(sequences.len()).map_err(|_| ReceiptError::ArithmeticOverflow)?;
    let last = sequences.last().copied();
    if sequences.iter().copied().ne(0..event_count)
        || (epochs, targets, event_count, last)
            != (
                head.observer_epoch_count,
                head.target_count,
                head.event_count,
                head.last_receipt_seq,
            )
    {
        return Err(ReceiptError::IdentityMismatch("receipt_head_counts"));
    }
    Ok(())
}

fn write_batch(
    spool: &QualifiedSpool,
    batch: &ReceiptBatchV1,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), ReceiptError> {
    spool
        .write_immutable(
            Path::new(&receipt_batch_key(batch.batch_id)),
            batch.canonical_bytes(),
            ImmutableClass::ReceiptBatch,
            faults,
        )
        .map_err(ReceiptError::Spool)
}

fn write_receipt_index(
    spool: &QualifiedSpool,
    index: &IndexSnapshot,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), ReceiptError> {
    for (hash, bytes) in index.page_objects() {
        spool
            .write_immutable(
                Path::new(&receipt_index_key(hash)),
                bytes,
                ImmutableClass::ReceiptIndex,
                faults,
            )
            .map_err(ReceiptError::Spool)?;
    }
    Ok(())
}

fn write_head(
    spool: &QualifiedSpool,
    head: &ReceiptHeadObject,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), ReceiptError> {
    spool
        .write_immutable(
            Path::new(&head.key),
            &head.bytes,
            ImmutableClass::ReceiptHead,
            faults,
        )
        .map_err(ReceiptError::Spool)
}

fn write_pointer(
    spool: &QualifiedSpool,
    pointer: &ReceiptPointer,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), ReceiptError> {
    spool
        .replace_pointer(
            LOCAL_RECEIPTS,
            &pointer.bytes(),
            PointerClass::Receipt,
            faults,
        )
        .map_err(ReceiptError::Spool)
}

#[derive(Debug)]
struct ReceiptPageSource<'a> {
    spool: &'a QualifiedSpool,
}

impl IndexPageSource for ReceiptPageSource<'_> {
    fn get(&self, hash: Digest) -> Result<Vec<u8>, IndexError> {
        self.spool
            .read_relative(Path::new(&receipt_index_key(hash)))
            .map_err(|error| IndexError::PageSource(error.to_string()))
    }
}

fn record_entry(key: IndexKey, value: CanonicalJsonValue) -> Result<IndexEntry, ReceiptError> {
    IndexEntry::new(key, value.to_bytes()).map_err(ReceiptError::Index)
}

fn target_value(body: &TargetBody) -> CanonicalJsonValue {
    match body {
        TargetBody::WalRange(target) => object(vec![
            ("archive_id", string(uuid(target.archive_id.as_bytes()))),
            (
                "durable_prefix_hash",
                string(target.durable_prefix_hash.to_hex()),
            ),
            (
                "first_record_seq",
                integer(i128::from(target.first_record_seq)),
            ),
            ("kind", string("wal_range")),
            (
                "last_record_seq",
                integer(i128::from(target.last_record_seq)),
            ),
            (
                "projection_coverage_digest",
                string(target.projection_coverage_digest.to_hex()),
            ),
            ("session_id", string(uuid(target.session_id.as_bytes()))),
            ("wal_segment_id", string(target.wal_segment_id.to_hex())),
        ]),
        TargetBody::RemotePublication(target) => object(vec![
            ("archive_id", string(uuid(target.archive_id.as_bytes()))),
            (
                "archive_state",
                string(archive_state_name(target.archive_state)),
            ),
            ("generation_hash", string(target.generation_hash.to_hex())),
            ("index_root_hash", string(target.index_root_hash.to_hex())),
            (
                "installed_head_hash",
                string(target.installed_head_hash.to_hex()),
            ),
            ("kind", string("remote_publication")),
            ("object_version", target.object_version.value()),
            (
                "writer_claim_state",
                string(target.writer_claim_state.name()),
            ),
        ]),
    }
}

fn batch_value(
    epochs: &[ReceiptObserverEpochV1],
    targets: &[ReceiptTargetV1],
    events: &[ReceiptEventV1],
) -> CanonicalJsonValue {
    object(vec![
        (
            "events",
            CanonicalJsonValue::Array(events.iter().map(ReceiptEventV1::value).collect()),
        ),
        ("magic", string(BATCH_MAGIC)),
        (
            "observer_epochs",
            CanonicalJsonValue::Array(epochs.iter().map(ReceiptObserverEpochV1::value).collect()),
        ),
        (
            "targets",
            CanonicalJsonValue::Array(targets.iter().map(ReceiptTargetV1::value).collect()),
        ),
        ("version", integer(1)),
    ])
}

fn unique(
    values: impl IntoIterator<Item = Digest>,
    kind: &'static str,
) -> Result<(), ReceiptError> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(ReceiptError::DuplicateRecord(kind));
        }
    }
    Ok(())
}

fn envelope(kind: &str, payload: CanonicalJsonValue) -> Vec<u8> {
    let payload_bytes = payload.to_bytes();
    let checksum = domain_digest(
        "aiperf.archive.receipt-batch.v1",
        &[kind.as_bytes(), &1_u64.to_be_bytes(), &payload_bytes],
    );
    object(vec![
        ("checksum", string(checksum.to_hex())),
        ("magic", string(kind)),
        ("payload", payload),
        (
            "payload_byte_length",
            integer(i128::try_from(payload_bytes.len()).expect("usize fits i128")),
        ),
        ("version", integer(1)),
    ])
    .to_bytes()
}

fn decode_envelope(kind: &str, bytes: &[u8]) -> Result<CanonicalJsonValue, ReceiptError> {
    let value = CanonicalJsonValue::parse_canonical(bytes).map_err(ReceiptError::Canonical)?;
    let fields = as_object(&value, "receipt_envelope")?;
    if text(fields, "magic")? != kind || unsigned(fields, "version")? != 1 {
        return Err(ReceiptError::InvalidField("envelope_magic"));
    }
    let payload = fields
        .get("payload")
        .cloned()
        .ok_or(ReceiptError::InvalidField("payload"))?;
    let payload_bytes = payload.to_bytes();
    if unsigned(fields, "payload_byte_length")?
        != u64::try_from(payload_bytes.len()).map_err(|_| ReceiptError::ArithmeticOverflow)?
    {
        return Err(ReceiptError::InvalidField("payload_byte_length"));
    }
    let expected = domain_digest(
        "aiperf.archive.receipt-batch.v1",
        &[kind.as_bytes(), &1_u64.to_be_bytes(), &payload_bytes],
    );
    if digest(fields, "checksum")? != expected {
        return Err(ReceiptError::IdentityMismatch("envelope_checksum"));
    }
    Ok(payload)
}

fn receipt_batch_key(id: ReceiptBatchId) -> String {
    format!("receipts/batches/{}.json", id.digest().to_hex())
}

fn receipt_index_key(hash: Digest) -> String {
    format!("receipts/index/{}.json", hash.to_hex())
}

fn receipt_head_key(commit_seq: u64, hash: Digest) -> String {
    format!("receipts/heads/head-{commit_seq}-{}.json", hash.to_hex())
}

fn index_root_value(root: &IndexRootV1) -> CanonicalJsonValue {
    object(vec![
        ("height", integer(i128::from(root.height))),
        (
            "logical_entry_count",
            integer(i128::from(root.logical_entry_count)),
        ),
        (
            "maximum_key",
            root.maximum_key
                .as_ref()
                .map_or(CanonicalJsonValue::Null, |key| string(hex(key.as_bytes()))),
        ),
        (
            "minimum_key",
            root.minimum_key
                .as_ref()
                .map_or(CanonicalJsonValue::Null, |key| string(hex(key.as_bytes()))),
        ),
        (
            "root_byte_length",
            integer(i128::from(root.root_byte_length)),
        ),
        ("root_hash", string(root.root_hash.to_hex())),
    ])
}

fn parse_index_root(value: &CanonicalJsonValue) -> Result<IndexRootV1, ReceiptError> {
    let fields = as_object(value, "index_root")?;
    Ok(IndexRootV1 {
        root_hash: digest(fields, "root_hash")?,
        root_byte_length: unsigned(fields, "root_byte_length")?,
        height: u16::try_from(unsigned(fields, "height")?)
            .map_err(|_| ReceiptError::InvalidField("height"))?,
        logical_entry_count: unsigned(fields, "logical_entry_count")?,
        minimum_key: parse_optional_key(fields.get("minimum_key"))?,
        maximum_key: parse_optional_key(fields.get("maximum_key"))?,
    })
}

fn archive_state_name(state: ArchiveState) -> &'static str {
    match state {
        ArchiveState::Open => "open",
        ArchiveState::StopRequested => "stop_requested",
        ArchiveState::LocallyFinalized => "locally_finalized",
        ArchiveState::RemotelyFinalized => "remotely_finalized",
        ArchiveState::Failed => "failed",
    }
}

fn parse_archive_state(value: &str) -> Result<ArchiveState, ReceiptError> {
    match value {
        "open" => Ok(ArchiveState::Open),
        "stop_requested" => Ok(ArchiveState::StopRequested),
        "locally_finalized" => Ok(ArchiveState::LocallyFinalized),
        "remotely_finalized" => Ok(ArchiveState::RemotelyFinalized),
        "failed" => Ok(ArchiveState::Failed),
        _ => Err(ReceiptError::InvalidField("archive_state")),
    }
}

fn object(entries: Vec<(&str, CanonicalJsonValue)>) -> CanonicalJsonValue {
    CanonicalJsonValue::object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
    .expect("static receipt keys are unique")
}

fn string(value: impl Into<String>) -> CanonicalJsonValue {
    CanonicalJsonValue::String(value.into())
}

const fn integer(value: i128) -> CanonicalJsonValue {
    CanonicalJsonValue::Integer(value)
}

fn as_object<'a>(
    value: &'a CanonicalJsonValue,
    field: &'static str,
) -> Result<&'a BTreeMap<String, CanonicalJsonValue>, ReceiptError> {
    value.as_object().ok_or(ReceiptError::InvalidField(field))
}

fn as_array<'a>(
    value: Option<&'a CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a [CanonicalJsonValue], ReceiptError> {
    match value {
        Some(CanonicalJsonValue::Array(values)) => Ok(values),
        _ => Err(ReceiptError::InvalidField(field)),
    }
}

fn text<'a>(
    fields: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, ReceiptError> {
    fields
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(ReceiptError::InvalidField(field))
}

fn unsigned(
    fields: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, ReceiptError> {
    fields
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(ReceiptError::InvalidField(field))
}

fn signed_i64(
    fields: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<i64, ReceiptError> {
    fields
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(ReceiptError::InvalidField(field))
}

fn digest(
    fields: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<Digest, ReceiptError> {
    Digest::parse(text(fields, field)?).map_err(|_| ReceiptError::InvalidField(field))
}

fn parse_optional_digest(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<Digest>, ReceiptError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => Digest::parse(value)
            .map(Some)
            .map_err(|_| ReceiptError::InvalidField("optional_digest")),
        _ => Err(ReceiptError::InvalidField("optional_digest")),
    }
}

fn parse_optional_u64(value: Option<&CanonicalJsonValue>) -> Result<Option<u64>, ReceiptError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::Integer(value)) => u64::try_from(*value)
            .map(Some)
            .map_err(|_| ReceiptError::InvalidField("optional_u64")),
        _ => Err(ReceiptError::InvalidField("optional_u64")),
    }
}

fn parse_optional_key(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<IndexKey>, ReceiptError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => IndexKey::new(decode_hex(value)?)
            .map(Some)
            .map_err(ReceiptError::Index),
        _ => Err(ReceiptError::InvalidField("optional_index_key")),
    }
}

fn optional_id16(value: Option<&[u8; 16]>) -> Vec<u8> {
    value.map_or_else(
        || vec![0],
        |value| {
            let mut bytes = Vec::with_capacity(17);
            bytes.push(1);
            bytes.extend_from_slice(value);
            bytes
        },
    )
}

fn optional_i128(value: Option<i128>) -> Vec<u8> {
    value.map_or_else(
        || vec![0],
        |value| {
            let mut bytes = Vec::with_capacity(17);
            bytes.push(1);
            bytes.extend_from_slice(&value.to_be_bytes());
            bytes
        },
    )
}

fn parse_optional_session(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<SessionId>, ReceiptError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => parse_session_id(value).map(Some),
        _ => Err(ReceiptError::InvalidField("telemetry_session_id")),
    }
}

fn uuid(bytes: &[u8; 16]) -> String {
    let value = hex(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &value[..8],
        &value[8..12],
        &value[12..16],
        &value[16..20],
        &value[20..]
    )
}

fn parse_archive_id(value: &str) -> Result<ArchiveId, ReceiptError> {
    ArchiveId::new(parse_uuid(value)?).map_err(|_| ReceiptError::InvalidField("archive_id"))
}

fn parse_session_id(value: &str) -> Result<SessionId, ReceiptError> {
    SessionId::new(parse_uuid(value)?).map_err(|_| ReceiptError::InvalidField("session_id"))
}

fn parse_uuid(value: &str) -> Result<[u8; 16], ReceiptError> {
    if value.len() != 36
        || value.as_bytes()[8] != b'-'
        || value.as_bytes()[13] != b'-'
        || value.as_bytes()[18] != b'-'
        || value.as_bytes()[23] != b'-'
    {
        return Err(ReceiptError::InvalidField("uuid"));
    }
    let compact: String = value
        .chars()
        .filter(|character| *character != '-')
        .collect();
    decode_hex(&compact)?
        .try_into()
        .map_err(|_| ReceiptError::InvalidField("uuid"))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn decode_hex(value: &str) -> Result<Vec<u8>, ReceiptError> {
    if !value.len().is_multiple_of(2) {
        return Err(ReceiptError::InvalidField("hex"));
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0]).ok_or(ReceiptError::InvalidField("hex"))?;
        let low = nibble(pair[1]).ok_or(ReceiptError::InvalidField("hex"))?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        output.push(char::from(ALPHABET[((value >> 18) & 0x3f) as usize]));
        output.push(char::from(ALPHABET[((value >> 12) & 0x3f) as usize]));
        if chunk.len() > 1 {
            output.push(char::from(ALPHABET[((value >> 6) & 0x3f) as usize]));
        }
        if chunk.len() > 2 {
            output.push(char::from(ALPHABET[(value & 0x3f) as usize]));
        }
    }
    output
}

fn decode_base64url(value: &str) -> Result<Vec<u8>, ReceiptError> {
    if value.contains('=') || value.len() % 4 == 1 {
        return Err(ReceiptError::InvalidField("base64url"));
    }
    let mut output = Vec::with_capacity(value.len() * 3 / 4);
    for chunk in value.as_bytes().chunks(4) {
        let mut sextets = [0_u8; 4];
        for (index, byte) in chunk.iter().enumerate() {
            sextets[index] = base64_sextet(*byte).ok_or(ReceiptError::InvalidField("base64url"))?;
        }
        let combined = (u32::from(sextets[0]) << 18)
            | (u32::from(sextets[1]) << 12)
            | (u32::from(sextets[2]) << 6)
            | u32::from(sextets[3]);
        output.push((combined >> 16) as u8);
        if chunk.len() > 2 {
            output.push((combined >> 8) as u8);
        }
        if chunk.len() > 3 {
            output.push(combined as u8);
        }
    }
    if base64url(&output) != value {
        return Err(ReceiptError::InvalidField("base64url"));
    }
    Ok(output)
}

fn base64_sextet(value: u8) -> Option<u8> {
    match value {
        b'A'..=b'Z' => Some(value - b'A'),
        b'a'..=b'z' => Some(value - b'a' + 26),
        b'0'..=b'9' => Some(value - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::{DurabilityEdge, FailAtDurabilityEdge, NoDurabilityFaults};

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn epoch(seed: u8, telemetry_session_id: Option<SessionId>) -> ReceiptObserverEpochV1 {
        ReceiptObserverEpochV1::new(
            ExecutionId::new([seed; 16]).unwrap(),
            telemetry_session_id,
            TimeDomain::Real,
            10,
            Some(1_700_000_000_000_000_000),
            2,
            Digest::from_bytes([seed.wrapping_add(1); 32]),
        )
        .unwrap()
    }

    fn target() -> ReceiptTargetV1 {
        ReceiptTargetV1::wal_range(WalRangeTargetV1 {
            archive_id: archive(),
            session_id: session(),
            wal_segment_id: Digest::from_bytes([3; 32]),
            durable_prefix_hash: Digest::from_bytes([4; 32]),
            first_record_seq: 7,
            last_record_seq: 9,
            projection_coverage_digest: Digest::from_bytes([5; 32]),
        })
        .unwrap()
    }

    fn spool(temp: &TempDir) -> QualifiedSpool {
        QualifiedSpool::open(temp.path().join("spool")).unwrap()
    }

    fn receipt_edges() -> [DurabilityEdge; 16] {
        [
            DurabilityEdge::ReceiptBatchTempWritten,
            DurabilityEdge::ReceiptBatchFileSynced,
            DurabilityEdge::ReceiptBatchRenamed,
            DurabilityEdge::ReceiptBatchDirectorySynced,
            DurabilityEdge::ReceiptIndexTempWritten,
            DurabilityEdge::ReceiptIndexFileSynced,
            DurabilityEdge::ReceiptIndexRenamed,
            DurabilityEdge::ReceiptIndexDirectorySynced,
            DurabilityEdge::ReceiptHeadTempWritten,
            DurabilityEdge::ReceiptHeadFileSynced,
            DurabilityEdge::ReceiptHeadRenamed,
            DurabilityEdge::ReceiptHeadDirectorySynced,
            DurabilityEdge::ReceiptPointerTempWritten,
            DurabilityEdge::ReceiptPointerFileSynced,
            DurabilityEdge::ReceiptPointerRenamed,
            DurabilityEdge::ReceiptPointerDirectorySynced,
        ]
    }

    #[test]
    fn epoch_target_event_batch_round_trips_without_self_attestation() {
        let epoch = epoch(7, None);
        let target = target();
        let event = ReceiptEventV1::new(
            archive(),
            0,
            target.receipt_target_id,
            epoch.observer_epoch_id,
            ObservationKind::RecoveryVerified,
            99,
        );
        let batch = ReceiptBatchV1::new(vec![epoch], vec![target], vec![event]).unwrap();
        assert_eq!(
            ReceiptBatchV1::decode(batch.canonical_bytes()).unwrap(),
            batch
        );
    }

    #[test]
    fn target_id_excludes_epoch_observation_kind_and_time() {
        let epoch = epoch(7, Some(session()));
        let target = target();
        let response = ReceiptEventV1::new(
            archive(),
            0,
            target.receipt_target_id,
            epoch.observer_epoch_id,
            ObservationKind::ResponseObserved,
            100,
        );
        let recovery = ReceiptEventV1::new(
            archive(),
            1,
            target.receipt_target_id,
            epoch.observer_epoch_id,
            ObservationKind::RecoveryVerified,
            101,
        );
        assert_eq!(response.receipt_target_id, recovery.receipt_target_id);
        assert_ne!(response.event_id, recovery.event_id);
    }

    #[test]
    fn remote_terminal_target_requires_absent_claim_and_stable_version_is_exact() {
        let publication = RemotePublicationTargetV1 {
            archive_id: archive(),
            generation_hash: Digest::from_bytes([1; 32]),
            index_root_hash: Digest::from_bytes([2; 32]),
            installed_head_hash: Digest::from_bytes([3; 32]),
            object_version: StableObjectVersion::new(
                "memory-v1",
                ObjectVersionKind::Opaque,
                vec![0, 1, 2, 253, 254, 255],
            )
            .unwrap(),
            archive_state: ArchiveState::RemotelyFinalized,
            writer_claim_state: WriterClaimState::Absent,
        };
        let target = ReceiptTargetV1::remote_publication(publication.clone()).unwrap();
        assert_eq!(
            ReceiptTargetV1::from_value(&target.value()).unwrap(),
            target
        );
        assert!(
            ReceiptTargetV1::remote_publication(RemotePublicationTargetV1 {
                writer_claim_state: WriterClaimState::Active,
                ..publication
            })
            .is_err()
        );
    }

    #[test]
    fn tagged_keys_pin_discriminants_and_unused_zero_fields() {
        let epoch = epoch(7, None);
        let target = target();
        let event = ReceiptEventV1::new(
            archive(),
            42,
            target.receipt_target_id,
            epoch.observer_epoch_id,
            ObservationKind::ResponseObserved,
            100,
        );
        assert_eq!(
            ReceiptIndexKeyV1::observer_epoch(epoch.observer_epoch_id).as_bytes()[0],
            0x01
        );
        let target_key = ReceiptIndexKeyV1::target(&target);
        assert_eq!(target_key.as_bytes()[0], 0x02);
        assert_eq!(target_key.as_bytes()[1], ReceiptTargetKind::WalRange as u8);
        assert_eq!(&target_key.as_bytes()[26..58], &[0; 32]);
        let event_key = ReceiptIndexKeyV1::event(&event);
        assert_eq!(event_key.as_bytes()[0], 0x03);
        assert_eq!(&event_key.as_bytes()[33..41], &42_u64.to_be_bytes());
    }

    #[test]
    fn epoch_only_bootstrap_is_reachable_after_every_transaction_edge() {
        for edge in receipt_edges() {
            let temp = TempDir::new().unwrap();
            let spool = spool(&temp);
            let result = ReceiptJournal::bootstrap(
                &spool,
                archive(),
                epoch(7, None),
                &FailAtDurabilityEdge::first(edge),
            );
            assert!(matches!(
                result,
                Err(ReceiptError::Spool(SpoolError::FaultInjected(actual))) if actual == edge
            ));
            if spool.path().join(LOCAL_RECEIPTS).exists() {
                let recovered =
                    ReceiptJournal::recover(&spool, archive(), &NoDurabilityFaults).unwrap();
                assert_eq!(recovered.observer_epoch_count(), 1, "edge={edge:?}");
                assert_eq!(recovered.target_count(), 0);
                assert_eq!(recovered.event_count(), 0);
            } else {
                let journal = ReceiptJournal::bootstrap(
                    &spool,
                    archive(),
                    epoch(7, None),
                    &NoDurabilityFaults,
                )
                .unwrap();
                assert_eq!(journal.observer_epoch_count(), 1, "edge={edge:?}");
            }
        }
    }

    #[test]
    fn event_transaction_is_old_or_new_after_every_crash_edge() {
        for edge in receipt_edges() {
            let temp = TempDir::new().unwrap();
            let spool = spool(&temp);
            let observer = epoch(7, Some(session()));
            let mut journal =
                ReceiptJournal::bootstrap(&spool, archive(), observer.clone(), &NoDurabilityFaults)
                    .unwrap();
            let target = target();
            let event = ReceiptEventV1::new(
                archive(),
                0,
                target.receipt_target_id,
                observer.observer_epoch_id,
                ObservationKind::ResponseObserved,
                100,
            );
            let result = journal.record_event(target, event, &FailAtDurabilityEdge::first(edge));
            assert!(matches!(
                result,
                Err(ReceiptError::Spool(SpoolError::FaultInjected(actual))) if actual == edge
            ));
            drop(journal);
            let recovered =
                ReceiptJournal::recover(&spool, archive(), &NoDurabilityFaults).unwrap();
            assert!(recovered.event_count() <= 1, "edge={edge:?}");
            assert_eq!(recovered.target_count(), recovered.event_count());
            assert_eq!(
                recovered.last_receipt_seq(),
                (recovered.event_count() == 1).then_some(0)
            );
        }
    }

    #[test]
    fn target_reuse_and_distinct_recovery_event_do_not_recount_target() {
        let temp = TempDir::new().unwrap();
        let spool = spool(&temp);
        let first_epoch = epoch(7, Some(session()));
        let mut journal =
            ReceiptJournal::bootstrap(&spool, archive(), first_epoch.clone(), &NoDurabilityFaults)
                .unwrap();
        let target = target();
        journal
            .record_event(
                target.clone(),
                ReceiptEventV1::new(
                    archive(),
                    0,
                    target.receipt_target_id,
                    first_epoch.observer_epoch_id,
                    ObservationKind::ResponseObserved,
                    100,
                ),
                &NoDurabilityFaults,
            )
            .unwrap();
        let recovery_epoch = epoch(9, None);
        journal
            .append_observer_epoch(recovery_epoch.clone(), &NoDurabilityFaults)
            .unwrap();
        journal
            .record_event(
                target.clone(),
                ReceiptEventV1::new(
                    archive(),
                    1,
                    target.receipt_target_id,
                    recovery_epoch.observer_epoch_id,
                    ObservationKind::RecoveryVerified,
                    200,
                ),
                &NoDurabilityFaults,
            )
            .unwrap();
        assert_eq!(journal.observer_epoch_count(), 2);
        assert_eq!(journal.target_count(), 1);
        assert_eq!(journal.event_count(), 2);
        assert_eq!(journal.last_receipt_seq(), Some(1));
    }

    #[test]
    fn event_requires_durable_epoch_and_exact_next_sequence() {
        let temp = TempDir::new().unwrap();
        let spool = spool(&temp);
        let durable = epoch(7, None);
        let mut journal =
            ReceiptJournal::bootstrap(&spool, archive(), durable, &NoDurabilityFaults).unwrap();
        let target = target();
        let unknown = epoch(9, None);
        let event = ReceiptEventV1::new(
            archive(),
            0,
            target.receipt_target_id,
            unknown.observer_epoch_id,
            ObservationKind::RecoveryVerified,
            100,
        );
        assert!(matches!(
            journal.record_event(target.clone(), event, &NoDurabilityFaults),
            Err(ReceiptError::MissingObserverEpoch(_))
        ));
        journal
            .append_observer_epoch(unknown.clone(), &NoDurabilityFaults)
            .unwrap();
        let wrong = ReceiptEventV1::new(
            archive(),
            1,
            target.receipt_target_id,
            unknown.observer_epoch_id,
            ObservationKind::RecoveryVerified,
            100,
        );
        assert!(matches!(
            journal.record_event(target, wrong, &NoDurabilityFaults),
            Err(ReceiptError::ReceiptSequence {
                expected: 0,
                actual: 1
            })
        ));
    }

    #[test]
    fn corrupt_current_receipt_head_rolls_back_only_to_preceding() {
        let temp = TempDir::new().unwrap();
        let spool = spool(&temp);
        let mut journal =
            ReceiptJournal::bootstrap(&spool, archive(), epoch(7, None), &NoDurabilityFaults)
                .unwrap();
        journal
            .append_observer_epoch(epoch(9, None), &NoDurabilityFaults)
            .unwrap();
        drop(journal);
        let pointer_bytes = fs::read(spool.path().join(LOCAL_RECEIPTS)).unwrap();
        let pointer = ReceiptPointer::decode(&pointer_bytes).unwrap();
        let path = spool.path().join(pointer.current.key);
        let mut corrupt = fs::read(&path).unwrap();
        corrupt[0] ^= 1;
        fs::write(path, corrupt).unwrap();
        let recovered = ReceiptJournal::recover(&spool, archive(), &NoDurabilityFaults).unwrap();
        assert!(recovered.rolled_back_current());
        assert_eq!(recovered.observer_epoch_count(), 1);
    }

    #[test]
    fn coverage_digest_is_permutation_invariant_but_requires_contiguity() {
        let first = Digest::from_bytes([1; 32]);
        let second = Digest::from_bytes([2; 32]);
        assert_eq!(
            receipt_range_coverage(vec![(7, first), (8, second)]).unwrap(),
            receipt_range_coverage(vec![(8, second), (7, first)]).unwrap()
        );
        assert!(matches!(
            receipt_range_coverage(vec![(7, first), (9, second)]),
            Err(ReceiptError::NonContiguousWalRange)
        ));
    }
}
