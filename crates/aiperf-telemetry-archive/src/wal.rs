// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical WAL frames, prefix chains, sealed segments, and strict recovery.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

use crate::descriptor::WAL_V1;
use crate::{
    ArchiveId, BatchId, Digest, FrameId, FrameIdentityV1, ProjectionEvidence,
    ProjectionReservationId, RequiredProjection, SessionId, TableId, TerminalKind, domain_digest,
};

const FRAME_MAGIC: &[u8; 8] = b"AIPFWF01";
const SEGMENT_MAGIC: &[u8; 8] = b"AIPFWS01";
const FOOTER_MAGIC: &[u8; 8] = b"AIPFWEND";
const WIRE_VERSION: u16 = 1;
const FRAME_TRAILER_BYTES: usize = 32 + 4;
const FOOTER_WITHOUT_DIGEST_BYTES: usize = 8 + 2 + 8 + 8 + 8 + 32;
const FOOTER_BYTES: usize = FOOTER_WITHOUT_DIGEST_BYTES + 32;

/// The default hard upper bound for one encoded WAL frame.
pub const DEFAULT_MAX_WAL_FRAME_BYTES: u64 = 1 << 30;

/// One canonical final WAL frame header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalFrameHeaderV1 {
    /// Terminal success/loss identity.
    pub frame_id: FrameId,
    /// Stable candidate/control batch identity.
    pub batch_id: BatchId,
    /// Outcome-neutral owner reservation identity.
    pub projection_reservation_id: ProjectionReservationId,
    /// Inclusive global archive sequence.
    pub record_seq: u64,
    /// Closed authoritative frame Clock value.
    pub authoritative_frame_clock_ns: i64,
    /// Terminal payload class.
    pub terminal_kind: TerminalKind,
    /// Required table evidence, sorted by table ID.
    pub required_projections: Vec<RequiredProjection>,
    /// Shared raw objects referenced by this frame, sorted by digest.
    pub raw_reference_ids: Vec<Digest>,
    /// Shared raw objects materially introduced by this frame, sorted by digest.
    pub raw_material_ids: Vec<Digest>,
    /// Exact payload length.
    pub payload_len: u64,
}

impl WalFrameHeaderV1 {
    /// Constructs and validates a final header, deriving the terminal frame ID.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_id: BatchId,
        projection_reservation_id: ProjectionReservationId,
        record_seq: u64,
        authoritative_frame_clock_ns: i64,
        terminal_kind: TerminalKind,
        mut required_projections: Vec<RequiredProjection>,
        mut raw_reference_ids: Vec<Digest>,
        mut raw_material_ids: Vec<Digest>,
        payload_len: u64,
    ) -> Result<Self, WalError> {
        required_projections.sort_unstable_by_key(|projection| projection.table);
        ensure_unique_tables(&required_projections)?;
        raw_reference_ids.sort_unstable();
        ensure_unique_digests(&raw_reference_ids, "raw reference")?;
        raw_material_ids.sort_unstable();
        ensure_unique_digests(&raw_material_ids, "raw material")?;
        let frame_id =
            FrameIdentityV1::terminal_frame(terminal_kind, projection_reservation_id, record_seq);
        let header = Self {
            frame_id,
            batch_id,
            projection_reservation_id,
            record_seq,
            authoritative_frame_clock_ns,
            terminal_kind,
            required_projections,
            raw_reference_ids,
            raw_material_ids,
            payload_len,
        };
        header.validate()?;
        Ok(header)
    }

    /// Encodes exact canonical final-header bytes.
    pub fn encode(&self) -> Result<Vec<u8>, WalError> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FRAME_MAGIC);
        bytes.extend_from_slice(&WIRE_VERSION.to_be_bytes());
        bytes.extend_from_slice(WAL_V1.fingerprint().as_bytes());
        bytes.push(self.terminal_kind as u8);
        bytes.extend_from_slice(self.frame_id.digest().as_bytes());
        bytes.extend_from_slice(self.batch_id.digest().as_bytes());
        bytes.extend_from_slice(self.projection_reservation_id.digest().as_bytes());
        bytes.extend_from_slice(&self.record_seq.to_be_bytes());
        bytes.extend_from_slice(&self.authoritative_frame_clock_ns.to_be_bytes());
        bytes.extend_from_slice(
            &u16::try_from(self.required_projections.len())
                .map_err(|_| WalError::CountOverflow("required projections"))?
                .to_be_bytes(),
        );
        for projection in &self.required_projections {
            bytes.push(projection.table as u8);
            bytes.extend_from_slice(&projection.evidence.row_count.to_be_bytes());
            bytes.extend_from_slice(projection.evidence.logical_multiset_digest.as_bytes());
        }
        encode_digest_list(&self.raw_reference_ids, &mut bytes, "raw references")?;
        encode_digest_list(&self.raw_material_ids, &mut bytes, "raw materials")?;
        bytes.extend_from_slice(&self.payload_len.to_be_bytes());
        Ok(bytes)
    }

    fn validate(&self) -> Result<(), WalError> {
        if self.required_projections.is_empty() {
            return Err(WalError::MissingProjection);
        }
        ensure_sorted_unique_tables(&self.required_projections)?;
        ensure_sorted_unique_digests(&self.raw_reference_ids, "raw reference")?;
        ensure_sorted_unique_digests(&self.raw_material_ids, "raw material")?;
        let expected = FrameIdentityV1::terminal_frame(
            self.terminal_kind,
            self.projection_reservation_id,
            self.record_seq,
        );
        if self.frame_id != expected {
            return Err(WalError::FrameIdentityMismatch);
        }
        Ok(())
    }
}

/// One fully finalized WAL frame with canonical header, payload, digest, and CRC.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalFrame {
    header: WalFrameHeaderV1,
    payload: Vec<u8>,
    frame_digest: Digest,
    crc32c: u32,
}

impl WalFrame {
    /// Finalizes a frame from its canonical header and exact payload bytes.
    pub fn new(header: WalFrameHeaderV1, payload: Vec<u8>) -> Result<Self, WalError> {
        let payload_len = u64::try_from(payload.len()).map_err(|_| WalError::LengthOverflow)?;
        if payload_len != header.payload_len {
            return Err(WalError::PayloadLengthMismatch {
                declared: header.payload_len,
                actual: payload_len,
            });
        }
        let header_bytes = header.encode()?;
        let frame_digest = domain_digest("aiperf.archive.wal-frame.v1", &[&header_bytes, &payload]);
        let frame_length = frame_body_length(header_bytes.len(), payload.len())?;
        let mut crc_preimage =
            Vec::with_capacity(8 + header_bytes.len() + payload.len() + Digest::BYTE_LEN);
        crc_preimage.extend_from_slice(&frame_length.to_be_bytes());
        crc_preimage.extend_from_slice(&header_bytes);
        crc_preimage.extend_from_slice(&payload);
        crc_preimage.extend_from_slice(frame_digest.as_bytes());
        let crc32c = Crc32c::checksum(&crc_preimage);
        Ok(Self {
            header,
            payload,
            frame_digest,
            crc32c,
        })
    }

    /// Returns the final header.
    #[must_use]
    pub const fn header(&self) -> &WalFrameHeaderV1 {
        &self.header
    }

    /// Returns exact payload bytes.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Returns the authoritative frame digest.
    #[must_use]
    pub const fn frame_digest(&self) -> Digest {
        self.frame_digest
    }

    /// Returns the stored big-endian CRC-32C value.
    #[must_use]
    pub const fn crc32c(&self) -> u32 {
        self.crc32c
    }

    /// Encodes the complete length-prefixed frame.
    pub fn encode(&self) -> Result<Vec<u8>, WalError> {
        let header_bytes = self.header.encode()?;
        let frame_length = frame_body_length(header_bytes.len(), self.payload.len())?;
        if frame_length > DEFAULT_MAX_WAL_FRAME_BYTES {
            return Err(WalError::FrameTooLarge {
                declared: frame_length,
                maximum: DEFAULT_MAX_WAL_FRAME_BYTES,
            });
        }
        let mut bytes = Vec::with_capacity(
            usize::try_from(frame_length)
                .map_err(|_| WalError::LengthOverflow)?
                .saturating_add(8),
        );
        bytes.extend_from_slice(&frame_length.to_be_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&self.payload);
        bytes.extend_from_slice(self.frame_digest.as_bytes());
        bytes.extend_from_slice(&self.crc32c.to_be_bytes());
        Ok(bytes)
    }

    /// Strictly decodes one complete frame and rejects any trailing bytes.
    pub fn decode(bytes: &[u8], maximum: u64) -> Result<Self, WalError> {
        match decode_frame_prefix(bytes, maximum)? {
            FramePrefixDecode::Complete { frame, consumed } if consumed == bytes.len() => Ok(frame),
            FramePrefixDecode::Complete { .. } => Err(WalError::TrailingFrameBytes),
            FramePrefixDecode::Incomplete => Err(WalError::IncompleteFrame),
        }
    }
}

/// Canonical immutable segment header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalSegmentHeaderV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Content-derived segment identity.
    pub segment_id: Digest,
    /// Verified head hash at segment creation.
    pub previous_head_hash: Digest,
    /// Verified genesis hash.
    pub genesis_hash: Digest,
    /// Frozen archive-writer compatibility identity.
    pub writer_compatibility_id: Digest,
    /// First permitted global record sequence.
    pub first_record_seq: u64,
    /// Table schema fingerprints in ascending table order.
    pub table_schema_fingerprints: Vec<(TableId, Digest)>,
}

impl WalSegmentHeaderV1 {
    /// Constructs a canonical segment header and derives its segment ID.
    pub fn new(
        archive_id: ArchiveId,
        session_id: SessionId,
        previous_head_hash: Digest,
        genesis_hash: Digest,
        writer_compatibility_id: Digest,
        first_record_seq: u64,
        mut table_schema_fingerprints: Vec<(TableId, Digest)>,
    ) -> Result<Self, WalError> {
        table_schema_fingerprints.sort_unstable_by_key(|(table, _)| *table);
        for pair in table_schema_fingerprints.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(WalError::DuplicateTable(pair[0].0));
            }
        }
        let mut schema_bytes = Vec::new();
        for (table, fingerprint) in &table_schema_fingerprints {
            schema_bytes.push(*table as u8);
            schema_bytes.extend_from_slice(fingerprint.as_bytes());
        }
        let segment_id = domain_digest(
            "aiperf.archive.wal-segment.v1",
            &[
                archive_id.as_bytes(),
                session_id.as_bytes(),
                previous_head_hash.as_bytes(),
                genesis_hash.as_bytes(),
                writer_compatibility_id.as_bytes(),
                &first_record_seq.to_be_bytes(),
                &schema_bytes,
            ],
        );
        Ok(Self {
            archive_id,
            session_id,
            segment_id,
            previous_head_hash,
            genesis_hash,
            writer_compatibility_id,
            first_record_seq,
            table_schema_fingerprints,
        })
    }

    /// Encodes the complete segment preamble used as the initial prefix preimage.
    pub fn encode(&self) -> Result<Vec<u8>, WalError> {
        let mut body = Vec::new();
        body.extend_from_slice(WAL_V1.fingerprint().as_bytes());
        body.extend_from_slice(self.archive_id.as_bytes());
        body.extend_from_slice(self.session_id.as_bytes());
        body.extend_from_slice(self.segment_id.as_bytes());
        body.extend_from_slice(self.previous_head_hash.as_bytes());
        body.extend_from_slice(self.genesis_hash.as_bytes());
        body.extend_from_slice(self.writer_compatibility_id.as_bytes());
        body.extend_from_slice(&self.first_record_seq.to_be_bytes());
        body.extend_from_slice(
            &u16::try_from(self.table_schema_fingerprints.len())
                .map_err(|_| WalError::CountOverflow("table schema fingerprints"))?
                .to_be_bytes(),
        );
        let mut preceding = None;
        for (table, fingerprint) in &self.table_schema_fingerprints {
            if preceding >= Some(*table) {
                return Err(WalError::UnsortedTable(*table));
            }
            preceding = Some(*table);
            body.push(*table as u8);
            body.extend_from_slice(fingerprint.as_bytes());
        }
        let mut bytes = Vec::with_capacity(8 + 4 + body.len());
        bytes.extend_from_slice(SEGMENT_MAGIC);
        bytes.extend_from_slice(
            &u32::try_from(body.len())
                .map_err(|_| WalError::LengthOverflow)?
                .to_be_bytes(),
        );
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }
}

/// In-memory canonical segment assembly used by file and memory sinks.
#[derive(Clone, Debug)]
pub struct WalSegmentBuilder {
    header: WalSegmentHeaderV1,
    header_bytes: Vec<u8>,
    frame_bytes: Vec<u8>,
    frame_count: u64,
    last_record_seq: Option<u64>,
    prefix: Digest,
}

impl WalSegmentBuilder {
    /// Starts an empty open segment.
    pub fn new(header: WalSegmentHeaderV1) -> Result<Self, WalError> {
        let header_bytes = header.encode()?;
        let prefix = domain_digest("aiperf.archive.wal-prefix.v1", &[&header_bytes]);
        Ok(Self {
            header,
            header_bytes,
            frame_bytes: Vec::new(),
            frame_count: 0,
            last_record_seq: None,
            prefix,
        })
    }

    /// Appends one sequence-contiguous complete frame and advances the prefix chain.
    pub fn append(&mut self, frame: &WalFrame) -> Result<(), WalError> {
        let expected = self
            .last_record_seq
            .map_or(self.header.first_record_seq, |sequence| sequence + 1);
        if frame.header.record_seq != expected {
            return Err(WalError::RecordSequence {
                expected,
                actual: frame.header.record_seq,
            });
        }
        let encoded = frame.encode()?;
        self.frame_bytes.extend_from_slice(&encoded);
        self.prefix = next_prefix(self.prefix, frame.header.record_seq, frame.frame_digest);
        self.last_record_seq = Some(frame.header.record_seq);
        self.frame_count += 1;
        Ok(())
    }

    /// Returns complete open-segment bytes without a footer.
    #[must_use]
    pub fn open_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.header_bytes.len() + self.frame_bytes.len());
        bytes.extend_from_slice(&self.header_bytes);
        bytes.extend_from_slice(&self.frame_bytes);
        bytes
    }

    /// Returns the current cryptographic prefix.
    #[must_use]
    pub const fn prefix(&self) -> Digest {
        self.prefix
    }

    /// Seals a non-empty segment with its canonical footer and segment digest.
    pub fn seal(self) -> Result<SealedWalSegment, WalError> {
        let Some(last_record_seq) = self.last_record_seq else {
            return Err(WalError::CannotSealEmptySegment);
        };
        let mut footer_without_digest = Vec::with_capacity(FOOTER_WITHOUT_DIGEST_BYTES);
        footer_without_digest.extend_from_slice(FOOTER_MAGIC);
        footer_without_digest.extend_from_slice(&WIRE_VERSION.to_be_bytes());
        footer_without_digest.extend_from_slice(&self.frame_count.to_be_bytes());
        footer_without_digest.extend_from_slice(&self.header.first_record_seq.to_be_bytes());
        footer_without_digest.extend_from_slice(&last_record_seq.to_be_bytes());
        footer_without_digest.extend_from_slice(self.prefix.as_bytes());
        let segment_digest = domain_digest(
            "aiperf.archive.wal-segment.v1",
            &[
                &self.header_bytes,
                self.prefix.as_bytes(),
                &footer_without_digest,
            ],
        );
        let mut bytes =
            Vec::with_capacity(self.header_bytes.len() + self.frame_bytes.len() + FOOTER_BYTES);
        bytes.extend_from_slice(&self.header_bytes);
        bytes.extend_from_slice(&self.frame_bytes);
        bytes.extend_from_slice(&footer_without_digest);
        bytes.extend_from_slice(segment_digest.as_bytes());
        Ok(SealedWalSegment {
            bytes,
            segment_id: self.header.segment_id,
            final_prefix: self.prefix,
            segment_digest,
            first_record_seq: self.header.first_record_seq,
            last_record_seq,
            frame_count: self.frame_count,
        })
    }
}

/// Complete immutable sealed WAL bytes and their verified authority fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedWalSegment {
    bytes: Vec<u8>,
    segment_id: Digest,
    final_prefix: Digest,
    segment_digest: Digest,
    first_record_seq: u64,
    last_record_seq: u64,
    frame_count: u64,
}

impl SealedWalSegment {
    /// Returns exact sealed bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the segment ID from its header.
    #[must_use]
    pub const fn segment_id(&self) -> Digest {
        self.segment_id
    }

    /// Returns the final ordered prefix.
    #[must_use]
    pub const fn final_prefix(&self) -> Digest {
        self.final_prefix
    }

    /// Returns the immutable segment digest.
    #[must_use]
    pub const fn segment_digest(&self) -> Digest {
        self.segment_digest
    }

    /// Returns the inclusive first sequence.
    #[must_use]
    pub const fn first_record_seq(&self) -> u64 {
        self.first_record_seq
    }

    /// Returns the inclusive last sequence.
    #[must_use]
    pub const fn last_record_seq(&self) -> u64 {
        self.last_record_seq
    }

    /// Returns the frame count.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Strictly recovered segment state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredWal {
    /// Verified segment header.
    pub header: WalSegmentHeaderV1,
    /// Every complete verified frame in sequence order.
    pub frames: Vec<WalFrame>,
    /// Recomputed prefix after the last verified frame.
    pub final_prefix: Digest,
    /// Byte offset immediately after the final complete frame.
    pub valid_len: usize,
    /// Physically incomplete open-tail bytes safe to discard.
    pub discarded_tail_bytes: usize,
    /// Verified sealed segment digest, absent for open recovery.
    pub segment_digest: Option<Digest>,
}

impl RecoveredWal {
    /// Recovers an open segment, discarding only an incomplete physical tail.
    pub fn open(bytes: &[u8], maximum_frame_bytes: u64) -> Result<Self, WalError> {
        let (header, header_len) = decode_segment_header(bytes)?;
        recover_frames(
            bytes,
            header,
            header_len,
            bytes.len(),
            maximum_frame_bytes,
            true,
            None,
        )
    }

    /// Recovers and verifies a sealed segment including its footer/digest.
    pub fn sealed(bytes: &[u8], maximum_frame_bytes: u64) -> Result<Self, WalError> {
        if bytes.len() < FOOTER_BYTES {
            return Err(WalError::IncompleteFooter);
        }
        let footer_start = bytes.len() - FOOTER_BYTES;
        let footer = decode_footer(&bytes[footer_start..])?;
        let (header, header_len) = decode_segment_header(bytes)?;
        let recovered = recover_frames(
            bytes,
            header,
            header_len,
            footer_start,
            maximum_frame_bytes,
            false,
            Some(&footer),
        )?;
        let header_bytes = &bytes[..header_len];
        let footer_without_digest =
            &bytes[footer_start..footer_start + FOOTER_WITHOUT_DIGEST_BYTES];
        let expected_segment_digest = domain_digest(
            "aiperf.archive.wal-segment.v1",
            &[
                header_bytes,
                recovered.final_prefix.as_bytes(),
                footer_without_digest,
            ],
        );
        if expected_segment_digest != footer.segment_digest {
            return Err(WalError::SegmentDigestMismatch);
        }
        Ok(Self {
            segment_digest: Some(footer.segment_digest),
            ..recovered
        })
    }
}

/// Standard reflected CRC-32C/Castagnoli used only as the WAL torn-write check.
#[derive(Clone, Copy, Debug, Default)]
pub struct Crc32c;

impl Crc32c {
    /// Computes CRC-32C with polynomial `0x82f63b78`, init/final XOR `0xffffffff`.
    #[must_use]
    pub fn checksum(bytes: &[u8]) -> u32 {
        let mut crc = 0xffff_ffff_u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = 0_u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
            }
        }
        crc ^ 0xffff_ffff
    }
}

/// A malformed frame/segment or prohibited recovery ambiguity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WalError {
    /// At least one table projection must be declared.
    MissingProjection,
    /// The same table was declared twice.
    DuplicateTable(TableId),
    /// A table list is not in canonical ascending order.
    UnsortedTable(TableId),
    /// A raw declaration repeats an ID.
    DuplicateDigest(&'static str),
    /// A raw declaration list is not in ascending order.
    UnsortedDigest(&'static str),
    /// A list count exceeds its frozen integer width.
    CountOverflow(&'static str),
    /// A byte length cannot be represented safely.
    LengthOverflow,
    /// Header payload length differs from supplied bytes.
    PayloadLengthMismatch {
        /// Header declaration.
        declared: u64,
        /// Actual payload bytes.
        actual: u64,
    },
    /// A frame exceeds the configured hard maximum.
    FrameTooLarge {
        /// Declared frame body bytes.
        declared: u64,
        /// Configured maximum.
        maximum: u64,
    },
    /// A complete-frame decoder observed trailing bytes.
    TrailingFrameBytes,
    /// The physical frame ends before its declared length.
    IncompleteFrame,
    /// A sealed footer is physically incomplete.
    IncompleteFooter,
    /// Frame/segment magic is wrong.
    InvalidMagic(&'static str),
    /// Wire version is unsupported.
    UnsupportedVersion(u16),
    /// Descriptor fingerprint does not match this implementation.
    DescriptorFingerprintMismatch,
    /// Terminal kind discriminant is unknown.
    UnknownTerminalKind(u8),
    /// Header frame ID does not match terminal kind/reservation/sequence.
    FrameIdentityMismatch,
    /// Frame length disagrees with parsed header/payload/trailer sizes.
    FrameLengthMismatch,
    /// CRC-32C failed for a physically complete frame.
    CrcMismatch,
    /// BLAKE3 failed for a physically complete frame.
    FrameDigestMismatch,
    /// Segment frames are not sequence-contiguous.
    RecordSequence {
        /// Required next sequence.
        expected: u64,
        /// Actual frame sequence.
        actual: u64,
    },
    /// An empty segment cannot be sealed.
    CannotSealEmptySegment,
    /// Sealed footer frame facts disagree with recovered frames.
    FooterMismatch,
    /// Sealed segment digest failed.
    SegmentDigestMismatch,
    /// Input ended while decoding a complete authority field.
    UnexpectedEof(&'static str),
}

impl Display for WalError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProjection => {
                formatter.write_str("WAL frame declares no table projection")
            }
            Self::DuplicateTable(table) => write!(formatter, "duplicate WAL table {table:?}"),
            Self::UnsortedTable(table) => write!(formatter, "unsorted WAL table {table:?}"),
            Self::DuplicateDigest(kind) => write!(formatter, "duplicate WAL {kind} ID"),
            Self::UnsortedDigest(kind) => write!(formatter, "unsorted WAL {kind} IDs"),
            Self::CountOverflow(kind) => write!(formatter, "WAL {kind} count overflow"),
            Self::LengthOverflow => formatter.write_str("WAL byte length overflow"),
            Self::PayloadLengthMismatch { declared, actual } => write!(
                formatter,
                "WAL payload length mismatch: declared {declared}, found {actual}"
            ),
            Self::FrameTooLarge { declared, maximum } => write!(
                formatter,
                "WAL frame declares {declared} bytes above maximum {maximum}"
            ),
            Self::TrailingFrameBytes => {
                formatter.write_str("trailing bytes after complete WAL frame")
            }
            Self::IncompleteFrame => formatter.write_str("physically incomplete WAL frame"),
            Self::IncompleteFooter => formatter.write_str("physically incomplete WAL footer"),
            Self::InvalidMagic(kind) => write!(formatter, "invalid WAL {kind} magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported WAL version {version}")
            }
            Self::DescriptorFingerprintMismatch => {
                formatter.write_str("WAL descriptor fingerprint mismatch")
            }
            Self::UnknownTerminalKind(kind) => {
                write!(formatter, "unknown WAL terminal kind {kind}")
            }
            Self::FrameIdentityMismatch => {
                formatter.write_str("WAL terminal frame identity mismatch")
            }
            Self::FrameLengthMismatch => formatter.write_str("WAL frame length mismatch"),
            Self::CrcMismatch => formatter.write_str("WAL frame CRC-32C mismatch"),
            Self::FrameDigestMismatch => formatter.write_str("WAL frame BLAKE3 mismatch"),
            Self::RecordSequence { expected, actual } => write!(
                formatter,
                "WAL record sequence mismatch: expected {expected}, found {actual}"
            ),
            Self::CannotSealEmptySegment => formatter.write_str("cannot seal an empty WAL segment"),
            Self::FooterMismatch => {
                formatter.write_str("WAL footer does not match recovered frames")
            }
            Self::SegmentDigestMismatch => formatter.write_str("WAL segment digest mismatch"),
            Self::UnexpectedEof(field) => {
                write!(formatter, "unexpected EOF while decoding {field}")
            }
        }
    }
}

impl std::error::Error for WalError {}

enum FramePrefixDecode {
    Complete { frame: WalFrame, consumed: usize },
    Incomplete,
}

struct Footer {
    frame_count: u64,
    first_record_seq: u64,
    last_record_seq: u64,
    final_prefix: Digest,
    segment_digest: Digest,
}

fn frame_body_length(header_len: usize, payload_len: usize) -> Result<u64, WalError> {
    let length = header_len
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(FRAME_TRAILER_BYTES))
        .ok_or(WalError::LengthOverflow)?;
    u64::try_from(length).map_err(|_| WalError::LengthOverflow)
}

fn decode_frame_prefix(bytes: &[u8], maximum: u64) -> Result<FramePrefixDecode, WalError> {
    if bytes.len() < 8 {
        return Ok(FramePrefixDecode::Incomplete);
    }
    let frame_length = u64::from_be_bytes(bytes[..8].try_into().expect("checked length"));
    if frame_length > maximum {
        return Err(WalError::FrameTooLarge {
            declared: frame_length,
            maximum,
        });
    }
    let consumed = usize::try_from(frame_length)
        .map_err(|_| WalError::LengthOverflow)?
        .checked_add(8)
        .ok_or(WalError::LengthOverflow)?;
    if bytes.len() < consumed {
        return Ok(FramePrefixDecode::Incomplete);
    }
    let complete = &bytes[..consumed];
    let body = &complete[8..];
    if body.len() < FRAME_TRAILER_BYTES {
        return Err(WalError::FrameLengthMismatch);
    }
    let stored_crc =
        u32::from_be_bytes(body[body.len() - 4..].try_into().expect("checked trailer"));
    let calculated_crc = Crc32c::checksum(&complete[..complete.len() - 4]);
    if stored_crc != calculated_crc {
        return Err(WalError::CrcMismatch);
    }
    let digest_offset = body.len() - FRAME_TRAILER_BYTES;
    let stored_digest = Digest::from_bytes(
        body[digest_offset..digest_offset + 32]
            .try_into()
            .expect("checked trailer"),
    );
    let mut cursor = Cursor::new(&body[..digest_offset]);
    let header = decode_frame_header(&mut cursor)?;
    let payload_len = usize::try_from(header.payload_len).map_err(|_| WalError::LengthOverflow)?;
    if cursor.remaining() != payload_len {
        return Err(WalError::FrameLengthMismatch);
    }
    let header_end = cursor.position();
    let payload = cursor.take(payload_len, "frame payload")?.to_vec();
    let calculated_digest = domain_digest(
        "aiperf.archive.wal-frame.v1",
        &[&body[..header_end], &payload],
    );
    if stored_digest != calculated_digest {
        return Err(WalError::FrameDigestMismatch);
    }
    let frame = WalFrame {
        header,
        payload,
        frame_digest: stored_digest,
        crc32c: stored_crc,
    };
    Ok(FramePrefixDecode::Complete { frame, consumed })
}

fn decode_frame_header(cursor: &mut Cursor<'_>) -> Result<WalFrameHeaderV1, WalError> {
    if cursor.take(8, "frame magic")? != FRAME_MAGIC {
        return Err(WalError::InvalidMagic("frame"));
    }
    let version = cursor.u16("frame version")?;
    if version != WIRE_VERSION {
        return Err(WalError::UnsupportedVersion(version));
    }
    if cursor.digest("WAL descriptor fingerprint")? != WAL_V1.fingerprint() {
        return Err(WalError::DescriptorFingerprintMismatch);
    }
    let terminal_kind = decode_terminal_kind(cursor.u8("terminal kind")?)?;
    let frame_id = FrameId::new(cursor.digest("frame ID")?);
    let batch_id = BatchId::new(cursor.digest("batch ID")?);
    let projection_reservation_id =
        ProjectionReservationId::new(cursor.digest("projection reservation ID")?);
    let record_seq = cursor.u64("record sequence")?;
    let authoritative_frame_clock_ns = cursor.i64("authoritative frame Clock")?;
    let projection_count = usize::from(cursor.u16("projection count")?);
    let mut required_projections = Vec::with_capacity(projection_count);
    for _ in 0..projection_count {
        let table = TableId::from_u8(cursor.u8("projection table")?)
            .map_err(|_| WalError::InvalidMagic("projection table"))?;
        let row_count = cursor.u64("projection row count")?;
        let logical_multiset_digest = cursor.digest("projection multiset digest")?;
        required_projections.push(RequiredProjection {
            table,
            evidence: ProjectionEvidence {
                row_count,
                logical_multiset_digest,
            },
        });
    }
    let raw_reference_ids = decode_digest_list(cursor, "raw references")?;
    let raw_material_ids = decode_digest_list(cursor, "raw materials")?;
    let payload_len = cursor.u64("payload length")?;
    let header = WalFrameHeaderV1 {
        frame_id,
        batch_id,
        projection_reservation_id,
        record_seq,
        authoritative_frame_clock_ns,
        terminal_kind,
        required_projections,
        raw_reference_ids,
        raw_material_ids,
        payload_len,
    };
    header.validate()?;
    Ok(header)
}

fn decode_segment_header(bytes: &[u8]) -> Result<(WalSegmentHeaderV1, usize), WalError> {
    if bytes.len() < 12 {
        return Err(WalError::UnexpectedEof("segment preamble"));
    }
    if &bytes[..8] != SEGMENT_MAGIC {
        return Err(WalError::InvalidMagic("segment"));
    }
    let body_len = usize::try_from(u32::from_be_bytes(
        bytes[8..12].try_into().expect("checked preamble"),
    ))
    .map_err(|_| WalError::LengthOverflow)?;
    let header_len = 12_usize
        .checked_add(body_len)
        .ok_or(WalError::LengthOverflow)?;
    if bytes.len() < header_len {
        return Err(WalError::UnexpectedEof("segment header"));
    }
    let mut cursor = Cursor::new(&bytes[12..header_len]);
    if cursor.digest("WAL descriptor fingerprint")? != WAL_V1.fingerprint() {
        return Err(WalError::DescriptorFingerprintMismatch);
    }
    let archive_id = ArchiveId::new(cursor.array16("archive ID")?)
        .map_err(|_| WalError::InvalidMagic("archive ID"))?;
    let session_id = SessionId::new(cursor.array16("session ID")?)
        .map_err(|_| WalError::InvalidMagic("session ID"))?;
    let segment_id = cursor.digest("segment ID")?;
    let previous_head_hash = cursor.digest("previous head hash")?;
    let genesis_hash = cursor.digest("genesis hash")?;
    let writer_compatibility_id = cursor.digest("writer compatibility ID")?;
    let first_record_seq = cursor.u64("first record sequence")?;
    let count = usize::from(cursor.u16("schema fingerprint count")?);
    let mut table_schema_fingerprints = Vec::with_capacity(count);
    for _ in 0..count {
        let table = TableId::from_u8(cursor.u8("schema table")?)
            .map_err(|_| WalError::InvalidMagic("schema table"))?;
        let fingerprint = cursor.digest("table schema fingerprint")?;
        table_schema_fingerprints.push((table, fingerprint));
    }
    if cursor.remaining() != 0 {
        return Err(WalError::FrameLengthMismatch);
    }
    let canonical = WalSegmentHeaderV1::new(
        archive_id,
        session_id,
        previous_head_hash,
        genesis_hash,
        writer_compatibility_id,
        first_record_seq,
        table_schema_fingerprints,
    )?;
    if canonical.segment_id != segment_id {
        return Err(WalError::SegmentDigestMismatch);
    }
    Ok((canonical, header_len))
}

#[allow(clippy::too_many_arguments)]
fn recover_frames(
    bytes: &[u8],
    header: WalSegmentHeaderV1,
    header_len: usize,
    frames_end: usize,
    maximum_frame_bytes: u64,
    allow_incomplete_tail: bool,
    footer: Option<&Footer>,
) -> Result<RecoveredWal, WalError> {
    if frames_end < header_len || frames_end > bytes.len() {
        return Err(WalError::FrameLengthMismatch);
    }
    let mut cursor = header_len;
    let mut frames = Vec::new();
    let mut expected_seq = header.first_record_seq;
    let mut prefix = domain_digest("aiperf.archive.wal-prefix.v1", &[&bytes[..header_len]]);
    while cursor < frames_end {
        match decode_frame_prefix(&bytes[cursor..frames_end], maximum_frame_bytes)? {
            FramePrefixDecode::Complete { frame, consumed } => {
                if frame.header.record_seq != expected_seq {
                    return Err(WalError::RecordSequence {
                        expected: expected_seq,
                        actual: frame.header.record_seq,
                    });
                }
                prefix = next_prefix(prefix, expected_seq, frame.frame_digest);
                expected_seq = expected_seq
                    .checked_add(1)
                    .ok_or(WalError::LengthOverflow)?;
                cursor = cursor
                    .checked_add(consumed)
                    .ok_or(WalError::LengthOverflow)?;
                frames.push(frame);
            }
            FramePrefixDecode::Incomplete if allow_incomplete_tail => break,
            FramePrefixDecode::Incomplete => return Err(WalError::IncompleteFrame),
        }
    }
    let discarded_tail_bytes = frames_end - cursor;
    if !allow_incomplete_tail && discarded_tail_bytes != 0 {
        return Err(WalError::IncompleteFrame);
    }
    if let Some(footer) = footer {
        let actual_count = u64::try_from(frames.len()).map_err(|_| WalError::LengthOverflow)?;
        let actual_last = frames.last().map(|frame| frame.header.record_seq);
        if footer.frame_count != actual_count
            || footer.first_record_seq != header.first_record_seq
            || actual_last != Some(footer.last_record_seq)
            || footer.final_prefix != prefix
        {
            return Err(WalError::FooterMismatch);
        }
    }
    Ok(RecoveredWal {
        header,
        frames,
        final_prefix: prefix,
        valid_len: cursor,
        discarded_tail_bytes,
        segment_digest: None,
    })
}

fn decode_footer(bytes: &[u8]) -> Result<Footer, WalError> {
    if bytes.len() != FOOTER_BYTES {
        return Err(WalError::IncompleteFooter);
    }
    let mut cursor = Cursor::new(bytes);
    if cursor.take(8, "footer magic")? != FOOTER_MAGIC {
        return Err(WalError::InvalidMagic("footer"));
    }
    let version = cursor.u16("footer version")?;
    if version != WIRE_VERSION {
        return Err(WalError::UnsupportedVersion(version));
    }
    Ok(Footer {
        frame_count: cursor.u64("footer frame count")?,
        first_record_seq: cursor.u64("footer first sequence")?,
        last_record_seq: cursor.u64("footer last sequence")?,
        final_prefix: cursor.digest("footer final prefix")?,
        segment_digest: cursor.digest("footer segment digest")?,
    })
}

fn next_prefix(previous: Digest, record_seq: u64, frame_digest: Digest) -> Digest {
    domain_digest(
        "aiperf.archive.wal-prefix.v1",
        &[
            previous.as_bytes(),
            &record_seq.to_be_bytes(),
            frame_digest.as_bytes(),
        ],
    )
}

fn ensure_unique_tables(projections: &[RequiredProjection]) -> Result<(), WalError> {
    let mut seen = BTreeSet::new();
    for projection in projections {
        if !seen.insert(projection.table) {
            return Err(WalError::DuplicateTable(projection.table));
        }
    }
    Ok(())
}

fn ensure_sorted_unique_tables(projections: &[RequiredProjection]) -> Result<(), WalError> {
    if projections.is_empty() {
        return Err(WalError::MissingProjection);
    }
    for pair in projections.windows(2) {
        if pair[0].table == pair[1].table {
            return Err(WalError::DuplicateTable(pair[0].table));
        }
        if pair[0].table > pair[1].table {
            return Err(WalError::UnsortedTable(pair[1].table));
        }
    }
    Ok(())
}

fn ensure_unique_digests(digests: &[Digest], kind: &'static str) -> Result<(), WalError> {
    let mut seen = BTreeSet::new();
    for digest in digests {
        if !seen.insert(*digest) {
            return Err(WalError::DuplicateDigest(kind));
        }
    }
    Ok(())
}

fn ensure_sorted_unique_digests(digests: &[Digest], kind: &'static str) -> Result<(), WalError> {
    for pair in digests.windows(2) {
        if pair[0] == pair[1] {
            return Err(WalError::DuplicateDigest(kind));
        }
        if pair[0] > pair[1] {
            return Err(WalError::UnsortedDigest(kind));
        }
    }
    Ok(())
}

fn encode_digest_list(
    digests: &[Digest],
    output: &mut Vec<u8>,
    kind: &'static str,
) -> Result<(), WalError> {
    output.extend_from_slice(
        &u16::try_from(digests.len())
            .map_err(|_| WalError::CountOverflow(kind))?
            .to_be_bytes(),
    );
    for digest in digests {
        output.extend_from_slice(digest.as_bytes());
    }
    Ok(())
}

fn decode_digest_list(
    cursor: &mut Cursor<'_>,
    kind: &'static str,
) -> Result<Vec<Digest>, WalError> {
    let count = usize::from(cursor.u16(kind)?);
    let mut digests = Vec::with_capacity(count);
    for _ in 0..count {
        digests.push(cursor.digest(kind)?);
    }
    ensure_sorted_unique_digests(&digests, kind)?;
    Ok(digests)
}

fn decode_terminal_kind(value: u8) -> Result<TerminalKind, WalError> {
    match value {
        1 => Ok(TerminalKind::SourceScrape),
        2 => Ok(TerminalKind::LifecycleMarker),
        3 => Ok(TerminalKind::LossExact),
        4 => Ok(TerminalKind::LossSaturation),
        5 => Ok(TerminalKind::SourceProjectionFailed),
        _ => Err(WalError::UnknownTerminalKind(value)),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    const fn position(&self) -> usize {
        self.position
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize, field: &'static str) -> Result<&'a [u8], WalError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(WalError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(WalError::UnexpectedEof(field));
        }
        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, WalError> {
        Ok(self.take(1, field)?[0])
    }

    fn u16(&mut self, field: &'static str) -> Result<u16, WalError> {
        Ok(u16::from_be_bytes(
            self.take(2, field)?.try_into().expect("checked length"),
        ))
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, WalError> {
        Ok(u64::from_be_bytes(
            self.take(8, field)?.try_into().expect("checked length"),
        ))
    }

    fn i64(&mut self, field: &'static str) -> Result<i64, WalError> {
        Ok(i64::from_be_bytes(
            self.take(8, field)?.try_into().expect("checked length"),
        ))
    }

    fn array16(&mut self, field: &'static str) -> Result<[u8; 16], WalError> {
        Ok(self.take(16, field)?.try_into().expect("checked length"))
    }

    fn digest(&mut self, field: &'static str) -> Result<Digest, WalError> {
        Ok(Digest::from_bytes(
            self.take(32, field)?.try_into().expect("checked length"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReservationKind, SourceOutcome};

    fn archive_id() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session_id() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn frame(record_seq: u64, payload: &[u8]) -> WalFrame {
        let batch = FrameIdentityV1::source_scrape_batch(
            archive_id(),
            session_id(),
            "source-a",
            record_seq,
            SourceOutcome::Success,
            Some(Digest::from_bytes([0x33; 32])),
        )
        .unwrap();
        let reservation = FrameIdentityV1::projection_reservation(
            archive_id(),
            session_id(),
            ReservationKind::SourceScrape,
            Some("source-a"),
            batch,
            record_seq,
        )
        .unwrap();
        let header = WalFrameHeaderV1::new(
            batch,
            reservation,
            record_seq,
            i64::try_from(record_seq).unwrap() * 10,
            TerminalKind::SourceScrape,
            vec![
                RequiredProjection {
                    table: TableId::Samples,
                    evidence: ProjectionEvidence::empty(),
                },
                RequiredProjection {
                    table: TableId::Attempts,
                    evidence: ProjectionEvidence {
                        row_count: 1,
                        logical_multiset_digest: Digest::from_bytes([0x44; 32]),
                    },
                },
            ],
            vec![],
            vec![],
            u64::try_from(payload.len()).unwrap(),
        )
        .unwrap();
        WalFrame::new(header, payload.to_vec()).unwrap()
    }

    fn segment_header(first_record_seq: u64) -> WalSegmentHeaderV1 {
        WalSegmentHeaderV1::new(
            archive_id(),
            session_id(),
            Digest::from_bytes([0x55; 32]),
            Digest::from_bytes([0x66; 32]),
            Digest::from_bytes([0x77; 32]),
            first_record_seq,
            vec![
                (TableId::Samples, Digest::from_bytes([0x88; 32])),
                (TableId::Attempts, Digest::from_bytes([0x99; 32])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn crc32c_profile_matches_the_pinned_check_value() {
        assert_eq!(Crc32c::checksum(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn frame_round_trip_pins_crc_preimage_and_terminal_identity() {
        let frame = frame(7, b"payload");
        let bytes = frame.encode().unwrap();
        assert_eq!(
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            concat!(
                "0000000000000126414950465746303100015b3e7ae627de3cc8402712069338b9f05cd62ff81bc071253898cb50e86a610c01",
                "1465f517a8f96ef02683447ff42842b6433bd7e4d801cd229f60fb565440e2b5670c295b397b5238b29e9e9a317d8779c50436",
                "bde5f271c2a49080cac76caf72c99572bb2c8b54c232116d5a3d5cf9ee3e825724eb6c89a13d893394b8fcb279000000000000",
                "000700000000000000460002010000000000000001444444444444444444444444444444444444444444444444444444444444",
                "44440300000000000000003347e9cebf4db3b818d29d0745f48900e2c563168efb8e7a18acd1c5939c44930000000000000000",
                "000000077061796c6f61648a3065654864f390aaae80191be279c38b1c5dccf9db3be98041dda036bf24932414c062"
            )
        );
        assert_eq!(
            WalFrame::decode(&bytes, DEFAULT_MAX_WAL_FRAME_BYTES).unwrap(),
            frame
        );
        assert_eq!(frame.crc32c(), Crc32c::checksum(&bytes[..bytes.len() - 4]));
        assert_eq!(
            frame.header.required_projections[0].table,
            TableId::Attempts
        );
        assert_eq!(
            frame.header.required_projections[1].evidence,
            ProjectionEvidence::empty()
        );
    }

    #[test]
    fn complete_corruption_is_never_reclassified_as_an_incomplete_tail() {
        let first = frame(1, b"first");
        let second = frame(2, b"second");
        let mut builder = WalSegmentBuilder::new(segment_header(1)).unwrap();
        builder.append(&first).unwrap();
        builder.append(&second).unwrap();
        let mut bytes = builder.open_bytes();
        let second_offset =
            segment_header(1).encode().unwrap().len() + first.encode().unwrap().len();
        bytes[second_offset + 20] ^= 1;
        assert_eq!(
            RecoveredWal::open(&bytes, DEFAULT_MAX_WAL_FRAME_BYTES),
            Err(WalError::CrcMismatch)
        );
    }

    #[test]
    fn open_recovery_discards_only_every_possible_incomplete_final_tail() {
        let first = frame(1, b"first");
        let second = frame(2, b"second");
        let mut builder = WalSegmentBuilder::new(segment_header(1)).unwrap();
        builder.append(&first).unwrap();
        let prefix_after_first = builder.prefix();
        let first_end = builder.open_bytes().len();
        builder.append(&second).unwrap();
        let complete = builder.open_bytes();
        for cut in first_end..complete.len() {
            let recovered =
                RecoveredWal::open(&complete[..cut], DEFAULT_MAX_WAL_FRAME_BYTES).unwrap();
            assert_eq!(recovered.frames.len(), 1, "cut={cut}");
            assert_eq!(recovered.final_prefix, prefix_after_first, "cut={cut}");
            assert_eq!(recovered.valid_len, first_end, "cut={cut}");
            assert_eq!(recovered.discarded_tail_bytes, cut - first_end, "cut={cut}");
        }
        let recovered = RecoveredWal::open(&complete, DEFAULT_MAX_WAL_FRAME_BYTES).unwrap();
        assert_eq!(recovered.frames.len(), 2);
        assert_eq!(recovered.discarded_tail_bytes, 0);
    }

    #[test]
    fn sealed_footer_and_segment_digest_are_authority() {
        let mut builder = WalSegmentBuilder::new(segment_header(9)).unwrap();
        builder.append(&frame(9, b"one")).unwrap();
        builder.append(&frame(10, b"two")).unwrap();
        let sealed = builder.seal().unwrap();
        let recovered = RecoveredWal::sealed(sealed.bytes(), DEFAULT_MAX_WAL_FRAME_BYTES).unwrap();
        assert_eq!(recovered.frames.len(), 2);
        assert_eq!(recovered.final_prefix, sealed.final_prefix());
        assert_eq!(recovered.segment_digest, Some(sealed.segment_digest()));

        let mut corrupt = sealed.bytes().to_vec();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert_eq!(
            RecoveredWal::sealed(&corrupt, DEFAULT_MAX_WAL_FRAME_BYTES),
            Err(WalError::SegmentDigestMismatch)
        );
    }

    #[test]
    fn prefix_binds_frame_order_and_sequence() {
        let mut ordered = WalSegmentBuilder::new(segment_header(1)).unwrap();
        ordered.append(&frame(1, b"a")).unwrap();
        ordered.append(&frame(2, b"b")).unwrap();

        let mut changed = WalSegmentBuilder::new(segment_header(1)).unwrap();
        changed.append(&frame(1, b"b")).unwrap();
        changed.append(&frame(2, b"a")).unwrap();
        assert_ne!(ordered.prefix(), changed.prefix());
        assert!(matches!(
            changed.append(&frame(4, b"c")),
            Err(WalError::RecordSequence {
                expected: 3,
                actual: 4
            })
        ));
    }
}
