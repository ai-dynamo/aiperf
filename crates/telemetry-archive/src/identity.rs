// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Outcome-neutral reservation and terminal frame identities.

use std::fmt::{self, Display, Formatter};

use crate::descriptor::FRAME_IDENTITY_V1;
use crate::{Digest, domain_digest};

macro_rules! id16_type {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name([u8; 16]);

        impl $name {
            #[doc = "Constructs the identifier, rejecting the reserved all-zero sentinel."]
            pub fn new(bytes: [u8; 16]) -> Result<Self, FrameIdentityError> {
                if bytes == [0; 16] {
                    return Err(FrameIdentityError::ZeroIdentifier(stringify!($name)));
                }
                Ok(Self(bytes))
            }

            #[doc = "Returns the exact 16 identifier bytes."]
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; 16] {
                &self.0
            }
        }
    };
}

macro_rules! digest_type {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(Digest);

        impl $name {
            #[doc = "Constructs the typed identity from its digest."]
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

id16_type!(
    ArchiveId,
    "A non-zero archive UUID represented as exact bytes."
);
id16_type!(
    SessionId,
    "A non-zero collection-session UUID represented as exact bytes."
);
digest_type!(
    BatchId,
    "The stable logical candidate/control batch identity."
);
digest_type!(
    ProjectionReservationId,
    "The outcome-neutral reservation identity assigned before projection."
);
digest_type!(FrameId, "The terminal success-or-loss frame identity.");

/// The terminal outcome used by source-scrape batch identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SourceOutcome {
    /// Successful non-empty exposition.
    Success = 1,
    /// Successful empty exposition.
    Empty = 2,
    /// Non-successful HTTP status.
    Http = 3,
    /// Transport failure.
    Transport = 4,
    /// Absolute-deadline failure.
    Timeout = 5,
    /// Strict parser failure.
    Parse = 6,
    /// Unsupported wire format.
    UnsupportedFormat = 7,
    /// Unsupported advertised feature.
    UnsupportedFeature = 8,
    /// Source disabled before network IO.
    Disabled = 9,
    /// Source shutdown observation.
    Shutdown = 10,
}

/// The outcome-neutral reservation class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ReservationKind {
    /// A physical source scrape.
    SourceScrape = 1,
    /// A lifecycle-only marker.
    LifecycleMarker = 2,
    /// One exact loss range.
    ExactLoss = 3,
    /// One cumulative saturation snapshot.
    LossSaturation = 4,
}

/// The closed terminal payload class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TerminalKind {
    /// Successful terminal source-scrape projection.
    SourceScrape = 1,
    /// Lifecycle-only marker.
    LifecycleMarker = 2,
    /// Exact loss frame.
    LossExact = 3,
    /// Cumulative loss-saturation snapshot.
    LossSaturation = 4,
    /// Source projection or owner-terminalization loss.
    SourceProjectionFailed = 5,
}

/// Inputs whose descriptor-encoded detail bytes define one lifecycle batch.
#[derive(Clone, Copy, Debug)]
pub struct LifecycleBatchInput<'a> {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Owner-assigned global record sequence.
    pub record_seq: u64,
    /// Frozen marker-kind discriminant.
    pub marker_kind: u8,
    /// Descriptor-encoded run/phase/state/completion/boundary fields.
    pub detail_bytes: &'a [u8],
}

/// Inputs whose descriptor-encoded fields define one exact-loss batch.
#[derive(Clone, Copy, Debug)]
pub struct ExactLossBatchInput<'a> {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Session-global loss sequence.
    pub loss_seq: u64,
    /// Source identity, or `None` for the explicit global sentinel.
    pub source_id: Option<&'a str>,
    /// Frozen loss-kind discriminant.
    pub loss_kind: u8,
    /// Frozen loss-reason discriminant.
    pub reason: u8,
    /// Descriptor-encoded inclusive ranges and boundary evidence.
    pub detail_bytes: &'a [u8],
}

/// Inputs whose cumulative state defines one saturation-snapshot batch.
#[derive(Clone, Copy, Debug)]
pub struct SaturationBatchInput {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Stable saturation slot identity.
    pub slot_id: Digest,
    /// Slot-local snapshot sequence.
    pub snapshot_seq: u64,
    /// Cumulative omitted range count.
    pub omitted_range_count: u64,
    /// Cumulative omitted entry count.
    pub omitted_entry_count: u64,
    /// Order-sensitive omitted-entry accumulator.
    pub omitted_rolling_digest: Digest,
}

/// The single v1 authority for batch, reservation, and terminal frame IDs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameIdentityV1;

impl FrameIdentityV1 {
    /// Returns the checked-in identity descriptor fingerprint.
    #[must_use]
    pub fn fingerprint() -> Digest {
        FRAME_IDENTITY_V1.fingerprint()
    }

    /// Derives a source-scrape batch from immutable terminal source facts.
    pub fn source_scrape_batch(
        archive_id: ArchiveId,
        session_id: SessionId,
        source_id: &str,
        source_record_seq: u64,
        outcome: SourceOutcome,
        decoded_unchanged_digest: Option<Digest>,
    ) -> Result<BatchId, FrameIdentityError> {
        validate_source_id(source_id)?;
        let sequence = source_record_seq.to_be_bytes();
        let kind = [ReservationKind::SourceScrape as u8];
        let outcome = [outcome as u8];
        let unchanged = optional_digest(decoded_unchanged_digest);
        Ok(BatchId::new(domain_digest(
            "aiperf.archive.batch.v1",
            &[
                archive_id.as_bytes(),
                session_id.as_bytes(),
                &kind,
                source_id.as_bytes(),
                &sequence,
                &outcome,
                &unchanged,
            ],
        )))
    }

    /// Derives a lifecycle-only batch after the owner assigns its sequence.
    #[must_use]
    pub fn lifecycle_batch(input: LifecycleBatchInput<'_>) -> BatchId {
        let kind = [ReservationKind::LifecycleMarker as u8];
        let marker_kind = [input.marker_kind];
        BatchId::new(domain_digest(
            "aiperf.archive.batch.v1",
            &[
                input.archive_id.as_bytes(),
                input.session_id.as_bytes(),
                &kind,
                &input.record_seq.to_be_bytes(),
                &marker_kind,
                input.detail_bytes,
            ],
        ))
    }

    /// Derives an exact source/global loss batch.
    pub fn exact_loss_batch(input: ExactLossBatchInput<'_>) -> Result<BatchId, FrameIdentityError> {
        if let Some(source_id) = input.source_id {
            validate_source_id(source_id)?;
        }
        let kind = [ReservationKind::ExactLoss as u8];
        let source = optional_source(input.source_id);
        let loss_kind = [input.loss_kind];
        let reason = [input.reason];
        Ok(BatchId::new(domain_digest(
            "aiperf.archive.batch.v1",
            &[
                input.archive_id.as_bytes(),
                input.session_id.as_bytes(),
                &kind,
                &input.loss_seq.to_be_bytes(),
                &source,
                &loss_kind,
                &reason,
                input.detail_bytes,
            ],
        )))
    }

    /// Derives a cumulative saturation-snapshot batch.
    #[must_use]
    pub fn saturation_batch(input: SaturationBatchInput) -> BatchId {
        let kind = [ReservationKind::LossSaturation as u8];
        BatchId::new(domain_digest(
            "aiperf.archive.batch.v1",
            &[
                input.archive_id.as_bytes(),
                input.session_id.as_bytes(),
                &kind,
                input.slot_id.as_bytes(),
                &input.snapshot_seq.to_be_bytes(),
                &input.omitted_range_count.to_be_bytes(),
                &input.omitted_entry_count.to_be_bytes(),
                input.omitted_rolling_digest.as_bytes(),
            ],
        ))
    }

    /// Derives the outcome-neutral reservation assigned before projection.
    pub fn projection_reservation(
        archive_id: ArchiveId,
        session_id: SessionId,
        kind: ReservationKind,
        source_id: Option<&str>,
        batch_id: BatchId,
        record_seq: u64,
    ) -> Result<ProjectionReservationId, FrameIdentityError> {
        if let Some(source_id) = source_id {
            validate_source_id(source_id)?;
        }
        if matches!(kind, ReservationKind::SourceScrape) && source_id.is_none() {
            return Err(FrameIdentityError::SourceRequired);
        }
        let kind = [kind as u8];
        let source = optional_source(source_id);
        Ok(ProjectionReservationId::new(domain_digest(
            "aiperf.archive.projection-reservation.v1",
            &[
                Self::fingerprint().as_bytes(),
                archive_id.as_bytes(),
                session_id.as_bytes(),
                &kind,
                &source,
                batch_id.digest().as_bytes(),
                &record_seq.to_be_bytes(),
            ],
        )))
    }

    /// Derives the terminal frame ID only after success or loss kind is known.
    #[must_use]
    pub fn terminal_frame(
        terminal_kind: TerminalKind,
        reservation_id: ProjectionReservationId,
        record_seq: u64,
    ) -> FrameId {
        let terminal_kind = [terminal_kind as u8];
        FrameId::new(domain_digest(
            "aiperf.archive.frame.v1",
            &[
                Self::fingerprint().as_bytes(),
                &terminal_kind,
                reservation_id.digest().as_bytes(),
                &record_seq.to_be_bytes(),
            ],
        ))
    }
}

/// Invalid input to the closed frame-identity matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameIdentityError {
    /// An identifier attempted to use the reserved all-zero sentinel.
    ZeroIdentifier(&'static str),
    /// A stored source ID is empty.
    EmptySourceId,
    /// A source scrape omitted its mandatory source ID.
    SourceRequired,
}

impl Display for FrameIdentityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroIdentifier(name) => {
                write!(
                    formatter,
                    "{name} cannot use the reserved all-zero identifier"
                )
            }
            Self::EmptySourceId => formatter.write_str("archive source ID cannot be empty"),
            Self::SourceRequired => {
                formatter.write_str("source-scrape reservation requires a source ID")
            }
        }
    }
}

impl std::error::Error for FrameIdentityError {}

fn validate_source_id(source_id: &str) -> Result<(), FrameIdentityError> {
    if source_id.is_empty() {
        return Err(FrameIdentityError::EmptySourceId);
    }
    Ok(())
}

fn optional_digest(digest: Option<Digest>) -> Vec<u8> {
    match digest {
        Some(digest) => {
            let mut encoded = Vec::with_capacity(33);
            encoded.push(1);
            encoded.extend_from_slice(digest.as_bytes());
            encoded
        }
        None => vec![0],
    }
}

fn optional_source(source: Option<&str>) -> Vec<u8> {
    match source {
        Some(source) => {
            let mut encoded = Vec::with_capacity(source.len() + 1);
            encoded.push(1);
            encoded.extend_from_slice(source.as_bytes());
            encoded
        }
        None => vec![0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_id() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session_id() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    #[test]
    fn reservation_is_outcome_neutral_but_terminal_kind_is_not() {
        let batch = FrameIdentityV1::source_scrape_batch(
            archive_id(),
            session_id(),
            "source-a",
            7,
            SourceOutcome::Success,
            None,
        )
        .unwrap();
        let reservation = FrameIdentityV1::projection_reservation(
            archive_id(),
            session_id(),
            ReservationKind::SourceScrape,
            Some("source-a"),
            batch,
            9,
        )
        .unwrap();
        assert_ne!(
            FrameIdentityV1::terminal_frame(TerminalKind::SourceScrape, reservation, 9),
            FrameIdentityV1::terminal_frame(TerminalKind::SourceProjectionFailed, reservation, 9)
        );
    }

    #[test]
    fn null_and_present_optional_fields_do_not_collide() {
        let digest = Digest::from_bytes([0; 32]);
        let without = FrameIdentityV1::source_scrape_batch(
            archive_id(),
            session_id(),
            "source-a",
            1,
            SourceOutcome::Empty,
            None,
        )
        .unwrap();
        let with = FrameIdentityV1::source_scrape_batch(
            archive_id(),
            session_id(),
            "source-a",
            1,
            SourceOutcome::Empty,
            Some(digest),
        )
        .unwrap();
        assert_ne!(without, with);
    }

    #[test]
    fn source_scrape_requires_nonempty_source() {
        assert_eq!(
            FrameIdentityV1::source_scrape_batch(
                archive_id(),
                session_id(),
                "",
                1,
                SourceOutcome::Success,
                None,
            ),
            Err(FrameIdentityError::EmptySourceId)
        );
    }
}
