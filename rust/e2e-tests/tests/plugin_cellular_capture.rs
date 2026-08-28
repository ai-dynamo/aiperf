// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the generic cellular capture transfer.
//!
//! A cell ships its exact-record artifact to the controller as a bounded
//! sequence of `ExactRecordsChunkV1` frames and then always closes with one
//! `CellCaptureBundleV1` carrying its presence-tagged folded projections. The
//! controller authenticates every frame against its registered cell set,
//! validates chunk identity (index, count, length, BLAKE3 digest), reassembles
//! in order regardless of arrival order, and refuses to finish while any
//! expected cell is still missing its bundle.

use aiperf_runtime::cellular::capture::{
    CaptureAssembler, CaptureTransferError, CellCaptureBundleV1, FoldedProjection,
    chunk_exact_records,
};
use aiperf_runtime::export::capture::{
    ExactRecordField, ExportCapturePlan, RetentionReason,
};

/// Deterministic artifact payload large enough to need several chunks.
fn artifact_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8).collect()
}

/// The bundle a cell sends when it has shipped `total_chunks` exact chunks.
fn bundle_for(cell_id: u32, chunks: &[aiperf_runtime::cellular::capture::ExactRecordsChunkV1]) -> CellCaptureBundleV1 {
    CellCaptureBundleV1 {
        cell_id,
        exact_chunk_count: chunks.len() as u32,
        exact_byte_length: chunks.iter().map(|chunk| chunk.bytes.len() as u64).sum(),
        folded_metrics: FoldedProjection::Present(serde_json::json!({"requests": 4})),
        folded_summary: FoldedProjection::Absent,
    }
}

#[test]
fn capture_chunks_assembled_in_order() {
    let bytes = artifact_bytes(2_500);
    let chunks = chunk_exact_records(0, &bytes, 1_000).expect("chunking succeeds");
    assert_eq!(chunks.len(), 3, "2500 bytes at 1000 bytes/chunk");
    assert!(chunks[2].is_terminal);
    assert!(!chunks[0].is_terminal);

    let mut assembler = CaptureAssembler::new([0]);
    // Deliver out of order: 2, 0, 1.
    for index in [2usize, 0, 1] {
        assembler
            .accept_chunk(chunks[index].clone())
            .expect("chunk accepted");
    }
    assembler
        .accept_bundle(bundle_for(0, &chunks))
        .expect("bundle accepted");

    let assembled = assembler.finish().expect("capture complete");
    assert_eq!(
        assembled.exact_records_for(0).expect("cell 0 payload"),
        bytes.as_slice(),
        "chunks reassemble to the original artifact regardless of arrival order"
    );
}

#[test]
fn capture_chunk_digest_mismatch_rejected() {
    let bytes = artifact_bytes(64);
    let chunks = chunk_exact_records(0, &bytes, 1_000).expect("chunking succeeds");
    let mut corrupted = chunks[0].clone();
    corrupted.bytes[0] ^= 0xFF;

    let mut assembler = CaptureAssembler::new([0]);
    let error = assembler
        .accept_chunk(corrupted)
        .expect_err("a corrupted chunk must be refused");
    assert!(
        matches!(error, CaptureTransferError::DigestMismatch { cell_id: 0, chunk_index: 0 }),
        "unexpected error: {error}"
    );
}

#[test]
fn capture_missing_cell_detected() {
    let bytes = artifact_bytes(10);
    let chunks = chunk_exact_records(0, &bytes, 1_000).expect("chunking succeeds");

    let mut assembler = CaptureAssembler::new([0, 1]);
    assembler
        .accept_chunk(chunks[0].clone())
        .expect("chunk accepted");
    assembler
        .accept_bundle(bundle_for(0, &chunks))
        .expect("bundle accepted");

    assert_eq!(
        assembler.missing_cells(),
        vec![1],
        "cell 1 never sent its mandatory bundle"
    );
    let error = assembler
        .finish()
        .expect_err("finish must refuse while a cell is missing");
    assert!(
        matches!(error, CaptureTransferError::MissingBundle { cell_id: 1 }),
        "unexpected error: {error}"
    );
}

#[test]
fn capture_empty_when_not_requested() {
    // A plan with no exact-record requirement does not request capture, so the
    // cell ships zero bytes but still closes with its folded bundle.
    let plan = ExportCapturePlan::default();
    assert!(!plan.requires_exact_records);

    let requesting = ExportCapturePlan::with_requirement(
        ExactRecordField::RequestIndex,
        RetentionReason::RequiredByExporter("otel".to_string()),
    );
    assert!(requesting.requires_exact_records);

    let mut assembler = CaptureAssembler::new([0]);
    assembler
        .accept_bundle(CellCaptureBundleV1 {
            cell_id: 0,
            exact_chunk_count: 0,
            exact_byte_length: 0,
            folded_metrics: FoldedProjection::Absent,
            folded_summary: FoldedProjection::Absent,
        })
        .expect("empty bundle accepted");

    assert!(assembler.missing_cells().is_empty());
    let assembled = assembler.finish().expect("capture complete");
    assert_eq!(
        assembled.total_exact_bytes(),
        0,
        "no capture requested means no bytes transferred"
    );
    assert!(assembled.exact_records_for(0).expect("cell present").is_empty());
}

#[test]
fn capture_duplicate_chunk_rejected() {
    let bytes = artifact_bytes(2_000);
    let chunks = chunk_exact_records(3, &bytes, 1_000).expect("chunking succeeds");

    let mut assembler = CaptureAssembler::new([3]);
    assembler
        .accept_chunk(chunks[0].clone())
        .expect("first delivery accepted");
    let error = assembler
        .accept_chunk(chunks[0].clone())
        .expect_err("a replayed chunk index must be refused");
    assert!(
        matches!(error, CaptureTransferError::DuplicateChunk { cell_id: 3, chunk_index: 0 }),
        "unexpected error: {error}"
    );
}

#[test]
fn capture_unexpected_cell_rejected() {
    let bytes = artifact_bytes(10);
    let chunks = chunk_exact_records(7, &bytes, 1_000).expect("chunking succeeds");

    // Only cell 0 is registered; cell 7 is not.
    let mut assembler = CaptureAssembler::new([0]);
    let error = assembler
        .accept_chunk(chunks[0].clone())
        .expect_err("a chunk from an unregistered cell must be refused");
    assert!(
        matches!(error, CaptureTransferError::UnexpectedCell { cell_id: 7 }),
        "unexpected error: {error}"
    );

    let error = assembler
        .accept_bundle(bundle_for(7, &chunks))
        .expect_err("a bundle from an unregistered cell must be refused");
    assert!(
        matches!(error, CaptureTransferError::UnexpectedCell { cell_id: 7 }),
        "unexpected error: {error}"
    );
}
