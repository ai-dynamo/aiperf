// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic cellular capture transfer.
//!
//! After a cell finishes its run it hands the controller two things, in this
//! order:
//!
//! 1. zero or more [`ExactRecordsChunkV1`] frames carrying the cell's exact-record
//!    artifact bytes, and
//! 2. exactly one [`CellCaptureBundleV1`] carrying the cell's presence-tagged
//!    folded projections.
//!
//! The chunk stream is *conditional*: a cell emits chunks only when the run
//! plan's [`ExportCapturePlan`](crate::export::capture::ExportCapturePlan)
//! declares that some exporter needs exact records
//! (`requires_exact_records`). The bundle is *unconditional* — it is sent even
//! when every projection is [`FoldedProjection::Absent`] — because its absence is
//! the controller's only positive signal that a cell never reported.
//!
//! [`CaptureAssembler`] is the controller side. It admits a frame only from a
//! registered cell, validates chunk identity (declared length, index against the
//! declared total, the terminal flag, and the chunk's BLAKE3 digest) before the
//! payload is retained, refuses a replayed index, and reassembles by index so
//! arrival order is irrelevant. [`CaptureAssembler::finish`] refuses to produce a
//! result while any expected cell is still missing its bundle or any chunk.
//!
//! This module is transport-neutral: it owns the frames and the validation, not
//! the wire.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

/// Payload bytes per chunk the cell uses when the caller does not choose one.
///
/// Bounded so a cell's capture never materializes as one unbounded frame on the
/// controller's channel.
pub const DEFAULT_CAPTURE_CHUNK_BYTES: usize = 1 << 20;

/// Hard upper bound on one chunk's declared and actual payload length.
///
/// The controller enforces this before retaining a payload, so a hostile or
/// buggy cell cannot drive controller memory with one frame.
pub const MAX_CAPTURE_CHUNK_BYTES: usize = 8 << 20;

/// One bounded slice of a cell's exact-record artifact.
///
/// `digest` is the BLAKE3 hash of `bytes` alone (no framing), so the controller
/// can validate a chunk without reference to its neighbours.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRecordsChunkV1 {
    /// Zero-based identifier of the emitting cell.
    pub cell_id: u32,
    /// Zero-based position of this chunk within the cell's chunk sequence.
    pub chunk_index: u32,
    /// Total chunk count the cell will emit; identical on every chunk.
    pub total_chunks: u32,
    /// Byte offset of this chunk's payload within the cell's artifact.
    pub byte_offset: u64,
    /// Declared payload length; must equal `bytes.len()`.
    pub byte_length: u64,
    /// BLAKE3 digest of `bytes`.
    pub digest: [u8; 32],
    /// Whether this is the cell's last chunk (`chunk_index + 1 == total_chunks`).
    pub is_terminal: bool,
    /// The chunk payload.
    pub bytes: Vec<u8>,
}

/// A folded projection that is always transmitted, tagged with its presence.
///
/// A cell that has nothing for a projection sends [`Self::Absent`] rather than
/// omitting the field, so "the cell had no data" and "the cell never reported"
/// stay distinguishable at the controller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "presence", content = "value", rename_all = "snake_case")]
pub enum FoldedProjection<T> {
    /// The cell produced this projection.
    Present(T),
    /// The cell produced nothing for this projection.
    Absent,
}

impl<T> FoldedProjection<T> {
    /// Whether this projection carries a value.
    pub fn is_present(&self) -> bool {
        matches!(self, Self::Present(_))
    }

    /// The carried value, if any.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Present(value) => Some(value),
            Self::Absent => None,
        }
    }
}

/// The mandatory frame that closes one cell's capture.
///
/// It declares what the cell shipped (so the controller can prove nothing was
/// lost) and carries the cell's folded projections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellCaptureBundleV1 {
    /// Zero-based identifier of the emitting cell.
    pub cell_id: u32,
    /// How many [`ExactRecordsChunkV1`] frames this cell emitted; `0` when the
    /// run plan requested no exact records.
    pub exact_chunk_count: u32,
    /// Total exact-record bytes this cell emitted across all its chunks.
    pub exact_byte_length: u64,
    /// Folded metric projection.
    pub folded_metrics: FoldedProjection<serde_json::Value>,
    /// Folded summary projection.
    pub folded_summary: FoldedProjection<serde_json::Value>,
}

/// A refusal from the controller-side capture admission and assembly path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureTransferError {
    /// The requested chunk size is zero or above [`MAX_CAPTURE_CHUNK_BYTES`].
    InvalidChunkSize {
        /// The refused size.
        chunk_bytes: usize,
    },
    /// A frame arrived from a cell the controller never registered.
    UnexpectedCell {
        /// The refused cell identity.
        cell_id: u32,
    },
    /// A chunk arrived after that cell already closed with its bundle.
    ChunkAfterBundle {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
    },
    /// The same chunk index arrived twice for one cell.
    DuplicateChunk {
        /// The offending cell.
        cell_id: u32,
        /// The replayed chunk index.
        chunk_index: u32,
    },
    /// A second bundle arrived for a cell that already closed.
    DuplicateBundle {
        /// The offending cell.
        cell_id: u32,
    },
    /// The chunk's BLAKE3 digest does not cover its payload.
    DigestMismatch {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
    },
    /// The chunk's declared `byte_length` disagrees with its payload.
    LengthMismatch {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
        /// The declared length.
        declared: u64,
        /// The actual payload length.
        actual: u64,
    },
    /// The chunk payload exceeds [`MAX_CAPTURE_CHUNK_BYTES`].
    ChunkTooLarge {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
        /// The refused payload length.
        byte_length: u64,
    },
    /// `chunk_index` is not below the chunk's declared `total_chunks`.
    ChunkIndexOutOfRange {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
        /// The declared total.
        total_chunks: u32,
    },
    /// `is_terminal` disagrees with `chunk_index + 1 == total_chunks`.
    TerminalFlagMismatch {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
    },
    /// One cell declared two different chunk totals.
    InconsistentChunkCount {
        /// The offending cell.
        cell_id: u32,
        /// The total established by earlier frames.
        expected: u32,
        /// The total this frame declared.
        received: u32,
    },
    /// A chunk's `byte_offset` does not continue the preceding chunks.
    OffsetMismatch {
        /// The offending cell.
        cell_id: u32,
        /// The offending chunk index.
        chunk_index: u32,
        /// The declared offset.
        declared: u64,
        /// The offset the preceding chunks imply.
        expected: u64,
    },
    /// A cell's chunk sequence has a hole.
    MissingChunk {
        /// The incomplete cell.
        cell_id: u32,
        /// The absent chunk index.
        chunk_index: u32,
    },
    /// An expected cell never sent its mandatory bundle.
    MissingBundle {
        /// The silent cell.
        cell_id: u32,
    },
    /// The assembled byte count disagrees with the cell's declared total.
    ByteLengthMismatch {
        /// The offending cell.
        cell_id: u32,
        /// The length the bundle declared.
        declared: u64,
        /// The length actually assembled.
        assembled: u64,
    },
}

impl fmt::Display for CaptureTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChunkSize { chunk_bytes } => write!(
                formatter,
                "capture chunk size {chunk_bytes} must be in 1..={MAX_CAPTURE_CHUNK_BYTES}"
            ),
            Self::UnexpectedCell { cell_id } => {
                write!(formatter, "capture frame from unregistered cell {cell_id}")
            }
            Self::ChunkAfterBundle {
                cell_id,
                chunk_index,
            } => write!(
                formatter,
                "cell {cell_id} sent chunk {chunk_index} after closing with its bundle"
            ),
            Self::DuplicateChunk {
                cell_id,
                chunk_index,
            } => write!(
                formatter,
                "cell {cell_id} sent chunk {chunk_index} more than once"
            ),
            Self::DuplicateBundle { cell_id } => {
                write!(
                    formatter,
                    "cell {cell_id} sent more than one capture bundle"
                )
            }
            Self::DigestMismatch {
                cell_id,
                chunk_index,
            } => write!(
                formatter,
                "cell {cell_id} chunk {chunk_index} digest does not cover its payload"
            ),
            Self::LengthMismatch {
                cell_id,
                chunk_index,
                declared,
                actual,
            } => write!(
                formatter,
                "cell {cell_id} chunk {chunk_index} declared {declared} bytes but carried {actual}"
            ),
            Self::ChunkTooLarge {
                cell_id,
                chunk_index,
                byte_length,
            } => write!(
                formatter,
                "cell {cell_id} chunk {chunk_index} of {byte_length} bytes exceeds the {MAX_CAPTURE_CHUNK_BYTES} byte limit"
            ),
            Self::ChunkIndexOutOfRange {
                cell_id,
                chunk_index,
                total_chunks,
            } => write!(
                formatter,
                "cell {cell_id} chunk index {chunk_index} is not below its declared total {total_chunks}"
            ),
            Self::TerminalFlagMismatch {
                cell_id,
                chunk_index,
            } => write!(
                formatter,
                "cell {cell_id} chunk {chunk_index} terminal flag disagrees with its position"
            ),
            Self::InconsistentChunkCount {
                cell_id,
                expected,
                received,
            } => write!(
                formatter,
                "cell {cell_id} declared {received} chunks after establishing {expected}"
            ),
            Self::OffsetMismatch {
                cell_id,
                chunk_index,
                declared,
                expected,
            } => write!(
                formatter,
                "cell {cell_id} chunk {chunk_index} declared offset {declared}, expected {expected}"
            ),
            Self::MissingChunk {
                cell_id,
                chunk_index,
            } => write!(formatter, "cell {cell_id} never sent chunk {chunk_index}"),
            Self::MissingBundle { cell_id } => {
                write!(formatter, "cell {cell_id} never sent its capture bundle")
            }
            Self::ByteLengthMismatch {
                cell_id,
                declared,
                assembled,
            } => write!(
                formatter,
                "cell {cell_id} declared {declared} capture bytes but {assembled} were assembled"
            ),
        }
    }
}

impl std::error::Error for CaptureTransferError {}

/// Split one cell's exact-record artifact into bounded, digest-bearing chunks.
///
/// An empty artifact yields no chunks — which is also the shape a cell emits when
/// the run plan requested no exact records at all.
pub fn chunk_exact_records(
    cell_id: u32,
    bytes: &[u8],
    chunk_bytes: usize,
) -> Result<Vec<ExactRecordsChunkV1>, CaptureTransferError> {
    if chunk_bytes == 0 || chunk_bytes > MAX_CAPTURE_CHUNK_BYTES {
        return Err(CaptureTransferError::InvalidChunkSize { chunk_bytes });
    }
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let total_chunks = bytes.len().div_ceil(chunk_bytes);
    let total_chunks = u32::try_from(total_chunks).map_err(|_| {
        // More chunks than a u32 index can name means the artifact is far past
        // any supportable size; refuse rather than silently truncating.
        CaptureTransferError::InvalidChunkSize { chunk_bytes }
    })?;
    let chunks = bytes
        .chunks(chunk_bytes)
        .enumerate()
        .map(|(position, payload)| {
            let chunk_index = position as u32;
            ExactRecordsChunkV1 {
                cell_id,
                chunk_index,
                total_chunks,
                byte_offset: (position * chunk_bytes) as u64,
                byte_length: payload.len() as u64,
                digest: *blake3::hash(payload).as_bytes(),
                is_terminal: chunk_index + 1 == total_chunks,
                bytes: payload.to_vec(),
            }
        })
        .collect();
    Ok(chunks)
}

/// One cell's in-progress capture on the controller.
#[derive(Debug, Default)]
struct CellCaptureState {
    /// Retained payloads keyed by chunk index, so arrival order is irrelevant.
    chunks: BTreeMap<u32, ExactRecordsChunkV1>,
    /// Chunk total established by the cell's first chunk.
    total_chunks: Option<u32>,
    /// The cell's closing bundle, once it has arrived.
    bundle: Option<CellCaptureBundleV1>,
}

/// Controller-side admission and reassembly for the cellular capture transfer.
///
/// Constructed over the controller's registered cell set; a frame from any other
/// cell is refused before its payload is retained.
#[derive(Debug)]
pub struct CaptureAssembler {
    cells: BTreeMap<u32, CellCaptureState>,
}

impl CaptureAssembler {
    /// Admit capture frames from exactly `expected_cells`.
    pub fn new(expected_cells: impl IntoIterator<Item = u32>) -> Self {
        let expected: BTreeSet<u32> = expected_cells.into_iter().collect();
        Self {
            cells: expected
                .into_iter()
                .map(|cell_id| (cell_id, CellCaptureState::default()))
                .collect(),
        }
    }

    /// Whether any cell has delivered a capture frame yet.
    ///
    /// The controller uses this to distinguish "capture was never requested" from
    /// "capture was requested and is incomplete".
    pub fn has_activity(&self) -> bool {
        self.cells
            .values()
            .any(|state| state.bundle.is_some() || !state.chunks.is_empty())
    }

    /// Validate and retain one exact-records chunk.
    pub fn accept_chunk(&mut self, chunk: ExactRecordsChunkV1) -> Result<(), CaptureTransferError> {
        let cell_id = chunk.cell_id;
        let chunk_index = chunk.chunk_index;
        let state = self
            .cells
            .get_mut(&cell_id)
            .ok_or(CaptureTransferError::UnexpectedCell { cell_id })?;
        if state.bundle.is_some() {
            return Err(CaptureTransferError::ChunkAfterBundle {
                cell_id,
                chunk_index,
            });
        }
        let actual = chunk.bytes.len() as u64;
        if actual > MAX_CAPTURE_CHUNK_BYTES as u64 {
            return Err(CaptureTransferError::ChunkTooLarge {
                cell_id,
                chunk_index,
                byte_length: actual,
            });
        }
        if chunk.byte_length != actual {
            return Err(CaptureTransferError::LengthMismatch {
                cell_id,
                chunk_index,
                declared: chunk.byte_length,
                actual,
            });
        }
        if chunk_index >= chunk.total_chunks {
            return Err(CaptureTransferError::ChunkIndexOutOfRange {
                cell_id,
                chunk_index,
                total_chunks: chunk.total_chunks,
            });
        }
        if chunk.is_terminal != (chunk_index + 1 == chunk.total_chunks) {
            return Err(CaptureTransferError::TerminalFlagMismatch {
                cell_id,
                chunk_index,
            });
        }
        match state.total_chunks {
            Some(expected) if expected != chunk.total_chunks => {
                return Err(CaptureTransferError::InconsistentChunkCount {
                    cell_id,
                    expected,
                    received: chunk.total_chunks,
                });
            }
            Some(_) => {}
            None => state.total_chunks = Some(chunk.total_chunks),
        }
        if state.chunks.contains_key(&chunk_index) {
            return Err(CaptureTransferError::DuplicateChunk {
                cell_id,
                chunk_index,
            });
        }
        if *blake3::hash(&chunk.bytes).as_bytes() != chunk.digest {
            return Err(CaptureTransferError::DigestMismatch {
                cell_id,
                chunk_index,
            });
        }
        state.chunks.insert(chunk_index, chunk);
        Ok(())
    }

    /// Validate and retain one cell's closing bundle.
    pub fn accept_bundle(
        &mut self,
        bundle: CellCaptureBundleV1,
    ) -> Result<(), CaptureTransferError> {
        let cell_id = bundle.cell_id;
        let state = self
            .cells
            .get_mut(&cell_id)
            .ok_or(CaptureTransferError::UnexpectedCell { cell_id })?;
        if state.bundle.is_some() {
            return Err(CaptureTransferError::DuplicateBundle { cell_id });
        }
        state.bundle = Some(bundle);
        Ok(())
    }

    /// Expected cells that have not yet closed with their mandatory bundle.
    pub fn missing_cells(&self) -> Vec<u32> {
        self.cells
            .iter()
            .filter(|(_, state)| state.bundle.is_none())
            .map(|(cell_id, _)| *cell_id)
            .collect()
    }

    /// Check every expected cell and reassemble its artifact in chunk order.
    pub fn finish(self) -> Result<AssembledCapture, CaptureTransferError> {
        let mut exact_records = BTreeMap::new();
        let mut bundles = BTreeMap::new();
        for (cell_id, state) in self.cells {
            let bundle = state
                .bundle
                .ok_or(CaptureTransferError::MissingBundle { cell_id })?;
            let total = state.total_chunks.unwrap_or(0);
            if total != bundle.exact_chunk_count {
                return Err(CaptureTransferError::InconsistentChunkCount {
                    cell_id,
                    expected: total,
                    received: bundle.exact_chunk_count,
                });
            }
            let mut assembled = Vec::with_capacity(bundle.exact_byte_length as usize);
            for chunk_index in 0..total {
                let chunk =
                    state
                        .chunks
                        .get(&chunk_index)
                        .ok_or(CaptureTransferError::MissingChunk {
                            cell_id,
                            chunk_index,
                        })?;
                let expected_offset = assembled.len() as u64;
                if chunk.byte_offset != expected_offset {
                    return Err(CaptureTransferError::OffsetMismatch {
                        cell_id,
                        chunk_index,
                        declared: chunk.byte_offset,
                        expected: expected_offset,
                    });
                }
                assembled.extend_from_slice(&chunk.bytes);
            }
            if assembled.len() as u64 != bundle.exact_byte_length {
                return Err(CaptureTransferError::ByteLengthMismatch {
                    cell_id,
                    declared: bundle.exact_byte_length,
                    assembled: assembled.len() as u64,
                });
            }
            exact_records.insert(cell_id, assembled);
            bundles.insert(cell_id, bundle);
        }
        Ok(AssembledCapture {
            exact_records,
            bundles,
        })
    }
}

/// The controller's complete, checked capture across every expected cell.
#[derive(Debug)]
pub struct AssembledCapture {
    exact_records: BTreeMap<u32, Vec<u8>>,
    bundles: BTreeMap<u32, CellCaptureBundleV1>,
}

impl AssembledCapture {
    /// One cell's reassembled exact-record artifact; empty when that cell
    /// shipped no exact records.
    pub fn exact_records_for(&self, cell_id: u32) -> Option<&[u8]> {
        self.exact_records
            .get(&cell_id)
            .map(|bytes| bytes.as_slice())
    }

    /// One cell's folded projections.
    pub fn bundle_for(&self, cell_id: u32) -> Option<&CellCaptureBundleV1> {
        self.bundles.get(&cell_id)
    }

    /// Cell identities in ascending order.
    pub fn cell_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.exact_records.keys().copied()
    }

    /// Total exact-record bytes assembled across every cell.
    pub fn total_exact_bytes(&self) -> u64 {
        self.exact_records
            .values()
            .map(|bytes| bytes.len() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_round_trips_through_the_assembler() {
        let bytes: Vec<u8> = (0..4_097u32).map(|index| index as u8).collect();
        let chunks = chunk_exact_records(2, &bytes, 512).expect("chunked");
        let mut assembler = CaptureAssembler::new([2]);
        for chunk in chunks.iter().rev() {
            assembler.accept_chunk(chunk.clone()).expect("accepted");
        }
        assembler
            .accept_bundle(CellCaptureBundleV1 {
                cell_id: 2,
                exact_chunk_count: chunks.len() as u32,
                exact_byte_length: bytes.len() as u64,
                folded_metrics: FoldedProjection::Absent,
                folded_summary: FoldedProjection::Absent,
            })
            .expect("bundle accepted");
        let assembled = assembler.finish().expect("complete");
        assert_eq!(assembled.exact_records_for(2), Some(bytes.as_slice()));
    }

    #[test]
    fn declared_length_must_cover_the_payload() {
        let mut chunk = chunk_exact_records(0, b"abcd", 16).expect("chunked")[0].clone();
        chunk.byte_length = 3;
        let mut assembler = CaptureAssembler::new([0]);
        assert_eq!(
            assembler.accept_chunk(chunk),
            Err(CaptureTransferError::LengthMismatch {
                cell_id: 0,
                chunk_index: 0,
                declared: 3,
                actual: 4,
            })
        );
    }

    #[test]
    fn bundle_chunk_count_must_match_what_arrived() {
        let chunks = chunk_exact_records(0, b"abcd", 2).expect("chunked");
        let mut assembler = CaptureAssembler::new([0]);
        assembler.accept_chunk(chunks[0].clone()).expect("accepted");
        assembler.accept_chunk(chunks[1].clone()).expect("accepted");
        assembler
            .accept_bundle(CellCaptureBundleV1 {
                cell_id: 0,
                exact_chunk_count: 1,
                exact_byte_length: 4,
                folded_metrics: FoldedProjection::Absent,
                folded_summary: FoldedProjection::Absent,
            })
            .expect("bundle accepted");
        assert_eq!(
            assembler.finish().map(|_| ()),
            Err(CaptureTransferError::InconsistentChunkCount {
                cell_id: 0,
                expected: 2,
                received: 1,
            })
        );
    }
}
