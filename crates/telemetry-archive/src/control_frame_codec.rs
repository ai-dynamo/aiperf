// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable lifecycle and loss control-frame projection.
//!
//! Control DTO validation, `FrameIdentityV1`, canonical logical evidence,
//! Arrow IPC, and the WAL final header are one closed operation. Recovery
//! decodes the same payload envelope as source frames and revalidates exact
//! table/cardinality/evidence declarations before returning physical rows.

use std::fmt::{self, Display, Formatter};
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::builder::{
    ArrayBuilder, Decimal128Builder, FixedSizeBinaryBuilder, Int64Builder, ListBuilder, MapBuilder,
    StringBuilder, StringDictionaryBuilder, StructBuilder, UInt64Builder, make_builder,
};
use arrow_array::types::Int8Type;
use arrow_array::{Array, ArrayRef, RecordBatch, make_array};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, SchemaRef};

use crate::{
    ArchiveId, ArchiveSchemasV1, ArchiveWalFrame, BatchId, BoundaryReference, BoundaryRole,
    CanonicalLogicalRow, Digest, ExactLossBatchInput, ExactLossRangeV1, FrameIdentityError,
    FrameIdentityV1, FrameTableProjectionV1, LifecycleBatchInput, LifecycleMarkerError,
    LifecycleMarkerV1, LogicalRowError, LogicalValue, LossKindV1, LossSaturationSnapshotV1,
    LossValidationError, ParquetProjectionError, ProjectionReservationId, RequiredProjection,
    ReservationKind, SaturationBatchInput, SchemaError, SessionId, TableId, TerminalKind, WalError,
    WalFrame, WalFrameHeaderV1,
};

const PAYLOAD_MAGIC: &[u8; 8] = b"AIPFWP01";
const PAYLOAD_VERSION: u16 = 1;

/// Closed v1 converter between lifecycle/loss DTOs and durable WAL frames.
#[derive(Clone, Debug)]
pub struct ControlFrameCodecV1 {
    schemas: ArchiveSchemasV1,
}

impl ControlFrameCodecV1 {
    /// Loads and retains the checked-in six-table schema authority.
    pub fn new() -> Result<Self, ControlFrameCodecError> {
        Ok(Self {
            schemas: ArchiveSchemasV1::load().map_err(ControlFrameCodecError::Schema)?,
        })
    }

    /// Uses an already loaded schema authority.
    #[must_use]
    pub fn with_schemas(schemas: ArchiveSchemasV1) -> Self {
        Self { schemas }
    }

    /// Encodes one standalone lifecycle marker after owner sequencing.
    pub fn encode_lifecycle_frame(
        &self,
        marker: LifecycleMarkerV1,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        marker
            .validate()
            .map_err(ControlFrameCodecError::Lifecycle)?;
        let detail = lifecycle_detail_bytes(&marker)?;
        let batch_id = FrameIdentityV1::lifecycle_batch(LifecycleBatchInput {
            archive_id: marker.archive_id,
            session_id: marker.session_id,
            record_seq: marker.record_seq,
            marker_kind: marker.kind as u8,
            detail_bytes: &detail,
        });
        let reservation = FrameIdentityV1::projection_reservation(
            marker.archive_id,
            marker.session_id,
            ReservationKind::LifecycleMarker,
            marker.source_id.as_deref(),
            batch_id,
            marker.record_seq,
        )
        .map_err(ControlFrameCodecError::Identity)?;
        let frame_id = FrameIdentityV1::terminal_frame(
            TerminalKind::LifecycleMarker,
            reservation,
            marker.record_seq,
        );
        let projection = self.projection(
            marker.archive_id,
            marker.session_id,
            marker.source_id.clone(),
            frame_id,
            marker.clock_ns,
            TableId::Markers,
            vec![lifecycle_values(&marker, frame_id)],
        )?;
        self.finish(
            batch_id,
            reservation,
            marker.record_seq,
            marker.clock_ns,
            TerminalKind::LifecycleMarker,
            projection,
        )
    }

    /// Encodes one exact source/global loss with its own control reservation.
    pub fn encode_exact_loss_frame(
        &self,
        loss: ExactLossRangeV1,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        loss.validate().map_err(ControlFrameCodecError::Loss)?;
        let detail = exact_loss_detail_bytes(&loss)?;
        let batch_id = FrameIdentityV1::exact_loss_batch(ExactLossBatchInput {
            archive_id: loss.archive_id,
            session_id: loss.session_id,
            loss_seq: loss.loss_seq,
            source_id: loss.source_id.as_deref(),
            loss_kind: loss.loss_kind as u8,
            reason: loss.reason as u8,
            detail_bytes: &detail,
        })
        .map_err(ControlFrameCodecError::Identity)?;
        let reservation = FrameIdentityV1::projection_reservation(
            loss.archive_id,
            loss.session_id,
            ReservationKind::ExactLoss,
            loss.source_id.as_deref(),
            batch_id,
            loss.record_seq,
        )
        .map_err(ControlFrameCodecError::Identity)?;
        self.encode_loss_projection(&loss, batch_id, reservation, TerminalKind::LossExact)
    }

    /// Encodes a source projection failure under its original source reservation.
    ///
    /// The new `loss_seq` remains row/payload/frame-digest evidence; it does not
    /// replace the source-scrape batch or reservation that already consumed the
    /// owner-assigned record sequence.
    pub fn encode_source_projection_failed(
        &self,
        loss: ExactLossRangeV1,
        source_batch_id: BatchId,
        source_reservation_id: ProjectionReservationId,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        loss.validate().map_err(ControlFrameCodecError::Loss)?;
        if loss.loss_kind != LossKindV1::ProjectionFailed {
            return Err(ControlFrameCodecError::SourceProjectionLossKind);
        }
        self.encode_loss_projection(
            &loss,
            source_batch_id,
            source_reservation_id,
            TerminalKind::SourceProjectionFailed,
        )
    }

    /// Encodes one latest cumulative saturation-slot snapshot.
    pub fn encode_loss_saturation_frame(
        &self,
        snapshot: LossSaturationSnapshotV1,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        snapshot.validate().map_err(ControlFrameCodecError::Loss)?;
        let batch_id = FrameIdentityV1::saturation_batch(SaturationBatchInput {
            archive_id: snapshot.archive_id,
            session_id: snapshot.session_id,
            slot_id: snapshot.saturation_slot_id,
            snapshot_seq: snapshot.saturation_snapshot_seq,
            omitted_range_count: snapshot.cumulative_omitted_range_count,
            omitted_entry_count: snapshot.cumulative_omitted_entry_count,
            omitted_rolling_digest: snapshot.omitted_rolling_digest,
        });
        let reservation = FrameIdentityV1::projection_reservation(
            snapshot.archive_id,
            snapshot.session_id,
            ReservationKind::LossSaturation,
            snapshot.source_id.as_deref(),
            batch_id,
            snapshot.record_seq,
        )
        .map_err(ControlFrameCodecError::Identity)?;
        let frame_id = FrameIdentityV1::terminal_frame(
            TerminalKind::LossSaturation,
            reservation,
            snapshot.record_seq,
        );
        let projection = self.projection(
            snapshot.archive_id,
            snapshot.session_id,
            snapshot.source_id.clone(),
            frame_id,
            snapshot.loss_observed_ns,
            TableId::Losses,
            vec![saturation_values(&snapshot, frame_id)],
        )?;
        self.finish(
            batch_id,
            reservation,
            snapshot.record_seq,
            snapshot.loss_observed_ns,
            TerminalKind::LossSaturation,
            projection,
        )
    }

    /// Rebuilds one verified lifecycle/exact-loss/saturation projection.
    pub fn decode_control_frame(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        wal_frame: WalFrame,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        let expected_table = match wal_frame.header().terminal_kind {
            TerminalKind::LifecycleMarker => TableId::Markers,
            TerminalKind::LossExact
            | TerminalKind::LossSaturation
            | TerminalKind::SourceProjectionFailed => TableId::Losses,
            TerminalKind::SourceScrape => {
                return Err(ControlFrameCodecError::UnsupportedTerminalKind);
            }
        };
        let projections = decode_payload(
            &self.schemas,
            archive_id,
            session_id,
            wal_frame.header().frame_id,
            wal_frame.header().authoritative_frame_clock_ns,
            wal_frame.payload(),
        )?;
        if projections.len() != 1
            || projections[0].table != expected_table
            || projections[0].batch.num_rows() != 1
            || wal_frame.header().required_projections.len() != 1
        {
            return Err(ControlFrameCodecError::ControlProjectionShape);
        }
        let projection = &projections[0];
        let required = &wal_frame.header().required_projections[0];
        let evidence = projection
            .validate(&self.schemas)
            .map_err(ControlFrameCodecError::Projection)?;
        if required.table != expected_table || required.evidence != evidence {
            return Err(ControlFrameCodecError::ProjectionEvidenceMismatch);
        }
        validate_recovered_control_semantics(&wal_frame, projection)?;
        Ok(ArchiveWalFrame {
            wal_frame,
            table_projections: projections,
        })
    }

    fn encode_loss_projection(
        &self,
        loss: &ExactLossRangeV1,
        batch_id: BatchId,
        reservation: ProjectionReservationId,
        terminal_kind: TerminalKind,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        let frame_id = FrameIdentityV1::terminal_frame(terminal_kind, reservation, loss.record_seq);
        let projection = self.projection(
            loss.archive_id,
            loss.session_id,
            loss.source_id.clone(),
            frame_id,
            loss.loss_observed_ns,
            TableId::Losses,
            vec![exact_loss_values(loss, frame_id)],
        )?;
        self.finish(
            batch_id,
            reservation,
            loss.record_seq,
            loss.loss_observed_ns,
            terminal_kind,
            projection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn projection(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        source_id: Option<String>,
        frame_id: crate::FrameId,
        authoritative_frame_clock_ns: i64,
        table: TableId,
        rows: Vec<Vec<LogicalValue>>,
    ) -> Result<FrameTableProjectionV1, ControlFrameCodecError> {
        let schema = self
            .schemas
            .table(table)
            .map_err(ControlFrameCodecError::Schema)?;
        let batch = logical_record_batch(schema.schema().clone(), &rows)?;
        let logical_rows = rows
            .iter()
            .map(|row| {
                CanonicalLogicalRow::encode(schema.logical_schema(), row)
                    .map_err(ControlFrameCodecError::LogicalRow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FrameTableProjectionV1 {
            archive_id,
            session_id,
            source_id,
            frame_id,
            authoritative_frame_clock_ns,
            table,
            batch,
            logical_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        batch_id: BatchId,
        reservation: ProjectionReservationId,
        record_seq: u64,
        authoritative_frame_clock_ns: i64,
        terminal_kind: TerminalKind,
        projection: FrameTableProjectionV1,
    ) -> Result<ArchiveWalFrame, ControlFrameCodecError> {
        let evidence = projection
            .validate(&self.schemas)
            .map_err(ControlFrameCodecError::Projection)?;
        if evidence.row_count != 1 {
            return Err(ControlFrameCodecError::ControlProjectionShape);
        }
        let payload = encode_payload(std::slice::from_ref(&projection))?;
        let payload_len =
            u64::try_from(payload.len()).map_err(|_| ControlFrameCodecError::LengthOverflow)?;
        let header = WalFrameHeaderV1::new(
            batch_id,
            reservation,
            record_seq,
            authoritative_frame_clock_ns,
            terminal_kind,
            vec![RequiredProjection {
                table: projection.table,
                evidence,
            }],
            Vec::new(),
            Vec::new(),
            payload_len,
        )
        .map_err(ControlFrameCodecError::Wal)?;
        if header.frame_id != projection.frame_id {
            return Err(ControlFrameCodecError::TerminalIdentityMismatch);
        }
        let wal_frame = WalFrame::new(header, payload).map_err(ControlFrameCodecError::Wal)?;
        Ok(ArchiveWalFrame {
            wal_frame,
            table_projections: vec![projection],
        })
    }
}

impl Default for ControlFrameCodecV1 {
    fn default() -> Self {
        Self::new().expect("checked-in archive schemas are valid")
    }
}

fn lifecycle_values(marker: &LifecycleMarkerV1, frame_id: crate::FrameId) -> Vec<LogicalValue> {
    let boundary = marker.boundary.as_ref();
    vec![
        uuid(marker.archive_id),
        uuid_session(marker.session_id),
        digest(frame_id.digest()),
        unsigned(marker.record_seq),
        unsigned(marker.marker_seq),
        enum_value(marker.kind.as_str()),
        signed(marker.clock_ns),
        optional_decimal(marker.unix_epoch_ns),
        optional_text(marker.run_id.as_deref()),
        optional_text(marker.phase_id.as_deref()),
        optional_text(marker.source_id.as_deref()),
        marker
            .phase_state
            .map_or(LogicalValue::Null, |value| enum_value(value.as_str())),
        marker
            .completion_reason
            .map_or(LogicalValue::Null, |value| enum_value(value.as_str())),
        optional_text(boundary.map(|value| value.transition_id.as_str())),
        optional_text(boundary.map(|value| value.boundary_id.as_str())),
        optional_text(boundary.and_then(|value| value.coalescing_group_id.as_deref())),
        boundary.map_or(LogicalValue::Null, |value| {
            enum_value(boundary_role(value.role))
        }),
        optional_signed(marker.phase_start_ns),
        optional_signed(marker.sent_end_ns),
        optional_signed(marker.requests_end_ns),
        optional_digest(marker.attribute_epoch_id),
        string_map(&marker.attributes),
    ]
}

fn exact_loss_values(loss: &ExactLossRangeV1, frame_id: crate::FrameId) -> Vec<LogicalValue> {
    loss_values(
        loss.archive_id,
        loss.session_id,
        loss.source_id.as_deref(),
        frame_id,
        loss.record_seq,
        loss.loss_seq,
        loss.count,
        loss.loss_kind,
        loss.reason.as_str(),
        loss.first_source_record_seq,
        loss.last_source_record_seq,
        loss.first_request_attempt_seq,
        loss.last_request_attempt_seq,
        loss.first_tick,
        loss.last_tick,
        loss.first_deadline_ns,
        loss.last_deadline_ns,
        loss.loss_observed_ns,
        &loss.boundary_refs,
        loss.boundary_overflow_count,
        loss.boundary_overflow_digest,
        "exact",
        None,
        None,
        0,
        0,
        None,
    )
}

fn saturation_values(
    snapshot: &LossSaturationSnapshotV1,
    frame_id: crate::FrameId,
) -> Vec<LogicalValue> {
    loss_values(
        snapshot.archive_id,
        snapshot.session_id,
        snapshot.source_id.as_deref(),
        frame_id,
        snapshot.record_seq,
        snapshot.loss_seq,
        snapshot.count(),
        snapshot.loss_kind,
        snapshot.reason.as_str(),
        snapshot.first_source_record_seq,
        snapshot.last_source_record_seq,
        snapshot.first_request_attempt_seq,
        snapshot.last_request_attempt_seq,
        snapshot.first_tick,
        snapshot.last_tick,
        snapshot.first_deadline_ns,
        snapshot.last_deadline_ns,
        snapshot.loss_observed_ns,
        &[],
        0,
        None,
        "overflow_summary",
        Some(snapshot.saturation_slot_id),
        Some(snapshot.saturation_snapshot_seq),
        snapshot.cumulative_omitted_range_count,
        snapshot.cumulative_omitted_entry_count,
        Some(snapshot.omitted_rolling_digest),
    )
}

#[allow(clippy::too_many_arguments)]
fn loss_values(
    archive_id: ArchiveId,
    session_id: SessionId,
    source_id: Option<&str>,
    frame_id: crate::FrameId,
    record_seq: u64,
    loss_seq: u64,
    count: u64,
    loss_kind: LossKindV1,
    reason: &str,
    first_source_record_seq: Option<u64>,
    last_source_record_seq: Option<u64>,
    first_request_attempt_seq: Option<u64>,
    last_request_attempt_seq: Option<u64>,
    first_tick: Option<u64>,
    last_tick: Option<u64>,
    first_deadline_ns: Option<i64>,
    last_deadline_ns: Option<i64>,
    loss_observed_ns: i64,
    boundary_refs: &[BoundaryReference],
    boundary_overflow_count: u64,
    boundary_overflow_digest: Option<Digest>,
    range_completeness: &str,
    saturation_slot_id: Option<Digest>,
    saturation_snapshot_seq: Option<u64>,
    cumulative_omitted_range_count: u64,
    cumulative_omitted_entry_count: u64,
    omitted_rolling_digest: Option<Digest>,
) -> Vec<LogicalValue> {
    vec![
        uuid(archive_id),
        uuid_session(session_id),
        optional_text(source_id),
        digest(frame_id.digest()),
        unsigned(record_seq),
        unsigned(loss_seq),
        unsigned(count),
        enum_value(loss_kind.as_str()),
        enum_value(reason),
        optional_unsigned(first_source_record_seq),
        optional_unsigned(last_source_record_seq),
        optional_unsigned(first_request_attempt_seq),
        optional_unsigned(last_request_attempt_seq),
        optional_unsigned(first_tick),
        optional_unsigned(last_tick),
        optional_signed(first_deadline_ns),
        optional_signed(last_deadline_ns),
        signed(loss_observed_ns),
        LogicalValue::List(boundary_refs.iter().map(boundary_value).collect()),
        unsigned(boundary_overflow_count),
        optional_digest(boundary_overflow_digest),
        enum_value(range_completeness),
        optional_digest(saturation_slot_id),
        optional_unsigned(saturation_snapshot_seq),
        unsigned(cumulative_omitted_range_count),
        unsigned(cumulative_omitted_entry_count),
        optional_digest(omitted_rolling_digest),
    ]
}

fn boundary_value(reference: &BoundaryReference) -> LogicalValue {
    LogicalValue::Struct(vec![
        text(&reference.transition_id),
        text(&reference.boundary_id),
        text(&reference.phase_id),
        text(&reference.source_id),
        enum_value(boundary_role(reference.role)),
        optional_text(reference.coalescing_group_id.as_deref()),
    ])
}

fn boundary_role(role: BoundaryRole) -> &'static str {
    match role {
        BoundaryRole::PhaseStart => "phase_start",
        BoundaryRole::PhaseEnd => "phase_end",
    }
}

fn lifecycle_detail_bytes(marker: &LifecycleMarkerV1) -> Result<Vec<u8>, ControlFrameCodecError> {
    let mut detail = DetailEncoder::new();
    detail.required(&marker.marker_seq.to_be_bytes())?;
    detail.required(&marker.clock_ns.to_be_bytes())?;
    let unix_epoch_ns = marker.unix_epoch_ns.map(i128::to_be_bytes);
    detail.optional(unix_epoch_ns.as_ref().map(<[u8; 16]>::as_slice))?;
    detail.optional(marker.run_id.as_deref().map(str::as_bytes))?;
    detail.optional(marker.phase_id.as_deref().map(str::as_bytes))?;
    detail.optional(marker.source_id.as_deref().map(str::as_bytes))?;
    let phase_state = marker.phase_state.map(|value| [value as u8]);
    detail.optional(phase_state.as_ref().map(<[u8; 1]>::as_slice))?;
    let completion_reason = marker.completion_reason.map(|value| [value as u8]);
    detail.optional(completion_reason.as_ref().map(<[u8; 1]>::as_slice))?;
    match &marker.boundary {
        None => detail.optional(None)?,
        Some(boundary) => detail.optional(Some(&encode_boundary_detail(boundary)?))?,
    }
    let phase_start_ns = marker.phase_start_ns.map(i64::to_be_bytes);
    detail.optional(phase_start_ns.as_ref().map(<[u8; 8]>::as_slice))?;
    let sent_end_ns = marker.sent_end_ns.map(i64::to_be_bytes);
    detail.optional(sent_end_ns.as_ref().map(<[u8; 8]>::as_slice))?;
    let requests_end_ns = marker.requests_end_ns.map(i64::to_be_bytes);
    detail.optional(requests_end_ns.as_ref().map(<[u8; 8]>::as_slice))?;
    let attribute_epoch_id = marker.attribute_epoch_id.map(|value| *value.as_bytes());
    detail.optional(attribute_epoch_id.as_ref().map(<[u8; 32]>::as_slice))?;
    detail.required(&encode_string_map_detail(&marker.attributes)?)?;
    Ok(detail.finish())
}

fn exact_loss_detail_bytes(loss: &ExactLossRangeV1) -> Result<Vec<u8>, ControlFrameCodecError> {
    let mut detail = DetailEncoder::new();
    detail.required(&loss.count.to_be_bytes())?;
    for value in [
        loss.first_source_record_seq,
        loss.last_source_record_seq,
        loss.first_request_attempt_seq,
        loss.last_request_attempt_seq,
        loss.first_tick,
        loss.last_tick,
    ] {
        let value = value.map(u64::to_be_bytes);
        detail.optional(value.as_ref().map(<[u8; 8]>::as_slice))?;
    }
    let first_deadline_ns = loss.first_deadline_ns.map(i64::to_be_bytes);
    detail.optional(first_deadline_ns.as_ref().map(<[u8; 8]>::as_slice))?;
    let last_deadline_ns = loss.last_deadline_ns.map(i64::to_be_bytes);
    detail.optional(last_deadline_ns.as_ref().map(<[u8; 8]>::as_slice))?;
    detail.required(&loss.loss_observed_ns.to_be_bytes())?;
    let mut boundaries = Vec::new();
    boundaries.extend_from_slice(
        &u64::try_from(loss.boundary_refs.len())
            .map_err(|_| ControlFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    for boundary in &loss.boundary_refs {
        encode_bytes(&mut boundaries, &encode_boundary_detail(boundary)?)?;
    }
    detail.required(&boundaries)?;
    detail.required(&loss.boundary_overflow_count.to_be_bytes())?;
    let boundary_overflow_digest = loss.boundary_overflow_digest.map(|value| *value.as_bytes());
    detail.optional(boundary_overflow_digest.as_ref().map(<[u8; 32]>::as_slice))?;
    Ok(detail.finish())
}

fn encode_boundary_detail(boundary: &BoundaryReference) -> Result<Vec<u8>, ControlFrameCodecError> {
    let mut encoded = DetailEncoder::new();
    for value in [
        boundary.transition_id.as_bytes(),
        boundary.boundary_id.as_bytes(),
        boundary.phase_id.as_bytes(),
        boundary.source_id.as_bytes(),
        &[boundary.role as u8],
    ] {
        encoded.required(value)?;
    }
    encoded.optional(boundary.coalescing_group_id.as_deref().map(str::as_bytes))?;
    Ok(encoded.finish())
}

fn encode_string_map_detail(
    values: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<u8>, ControlFrameCodecError> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(
        &u64::try_from(values.len())
            .map_err(|_| ControlFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    for (key, value) in values {
        encode_bytes(&mut encoded, key.as_bytes())?;
        encode_bytes(&mut encoded, value.as_bytes())?;
    }
    Ok(encoded)
}

struct DetailEncoder {
    bytes: Vec<u8>,
}

impl DetailEncoder {
    const fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn required(&mut self, value: &[u8]) -> Result<(), ControlFrameCodecError> {
        encode_bytes(&mut self.bytes, value)
    }

    fn optional(&mut self, value: Option<&[u8]>) -> Result<(), ControlFrameCodecError> {
        match value {
            None => self.required(&[0]),
            Some(value) => {
                let mut present = Vec::with_capacity(value.len() + 1);
                present.push(1);
                present.extend_from_slice(value);
                self.required(&present)
            }
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

fn validate_recovered_control_semantics(
    wal_frame: &WalFrame,
    projection: &FrameTableProjectionV1,
) -> Result<(), ControlFrameCodecError> {
    let record_seq = required_u64(&projection.batch, "record_seq")?;
    if record_seq != wal_frame.header().record_seq {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    match wal_frame.header().terminal_kind {
        TerminalKind::LifecycleMarker => {
            if required_u64(&projection.batch, "marker_seq")? != record_seq {
                return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
            }
        }
        TerminalKind::LossExact => validate_exact_loss_row(&projection.batch, false)?,
        TerminalKind::SourceProjectionFailed => validate_exact_loss_row(&projection.batch, true)?,
        TerminalKind::LossSaturation => validate_saturation_row(&projection.batch)?,
        TerminalKind::SourceScrape => return Err(ControlFrameCodecError::UnsupportedTerminalKind),
    }
    Ok(())
}

fn validate_exact_loss_row(
    batch: &RecordBatch,
    projection_failed: bool,
) -> Result<(), ControlFrameCodecError> {
    if required_enum(batch, "range_completeness")? != "exact"
        || optional_fixed(batch, "saturation_slot_id")?.is_some()
        || optional_u64_column(batch, "saturation_snapshot_seq")?.is_some()
        || required_u64(batch, "cumulative_omitted_range_count")? != 0
        || required_u64(batch, "cumulative_omitted_entry_count")? != 0
        || optional_fixed(batch, "omitted_rolling_digest")?.is_some()
    {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    if projection_failed && required_enum(batch, "loss_kind")? != "projection_failed" {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    Ok(())
}

fn validate_saturation_row(batch: &RecordBatch) -> Result<(), ControlFrameCodecError> {
    if required_enum(batch, "range_completeness")? != "overflow_summary"
        || optional_fixed(batch, "saturation_slot_id")?.is_none()
        || optional_u64_column(batch, "saturation_snapshot_seq")?.is_none()
        || required_u64(batch, "cumulative_omitted_range_count")? == 0
        || required_u64(batch, "cumulative_omitted_entry_count")? == 0
        || optional_fixed(batch, "omitted_rolling_digest")?.is_none()
        || required_u64(batch, "count")? != required_u64(batch, "cumulative_omitted_entry_count")?
    {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    Ok(())
}

fn required_u64(batch: &RecordBatch, name: &str) -> Result<u64, ControlFrameCodecError> {
    let array = batch
        .column_by_name(name)
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?;
    if array.is_null(0) {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    Ok(array.value(0))
}

fn optional_u64_column(
    batch: &RecordBatch,
    name: &str,
) -> Result<Option<u64>, ControlFrameCodecError> {
    let array = batch
        .column_by_name(name)
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?;
    Ok((!array.is_null(0)).then(|| array.value(0)))
}

fn optional_fixed<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a [u8]>, ControlFrameCodecError> {
    let array = batch
        .column_by_name(name)
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?;
    Ok((!array.is_null(0)).then(|| array.value(0)))
}

fn required_enum<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a str, ControlFrameCodecError> {
    let array = batch
        .column_by_name(name)
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?
        .as_any()
        .downcast_ref::<arrow_array::DictionaryArray<Int8Type>>()
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?;
    if array.is_null(0) {
        return Err(ControlFrameCodecError::RecoveredSemanticMismatch);
    }
    let key = usize::try_from(array.keys().value(0))
        .map_err(|_| ControlFrameCodecError::RecoveredSemanticMismatch)?;
    let values = array
        .values()
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .ok_or(ControlFrameCodecError::RecoveredSemanticMismatch)?;
    Ok(values.value(key))
}

fn text(value: &str) -> LogicalValue {
    LogicalValue::String(value.to_owned())
}

fn enum_value(value: &str) -> LogicalValue {
    text(value)
}

fn unsigned(value: u64) -> LogicalValue {
    LogicalValue::Unsigned(u128::from(value))
}

fn signed(value: i64) -> LogicalValue {
    LogicalValue::Signed(i128::from(value))
}

fn optional_unsigned(value: Option<u64>) -> LogicalValue {
    value.map_or(LogicalValue::Null, unsigned)
}

fn optional_signed(value: Option<i64>) -> LogicalValue {
    value.map_or(LogicalValue::Null, signed)
}

fn optional_decimal(value: Option<i128>) -> LogicalValue {
    value.map_or(LogicalValue::Null, LogicalValue::Decimal128)
}

fn optional_text(value: Option<&str>) -> LogicalValue {
    value.map_or(LogicalValue::Null, text)
}

fn string_map(value: &std::collections::BTreeMap<String, String>) -> LogicalValue {
    LogicalValue::StringMap(
        value
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn uuid(value: ArchiveId) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn uuid_session(value: SessionId) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn digest(value: Digest) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn optional_digest(value: Option<Digest>) -> LogicalValue {
    value.map_or(LogicalValue::Null, digest)
}

fn logical_record_batch(
    schema: SchemaRef,
    rows: &[Vec<LogicalValue>],
) -> Result<RecordBatch, ControlFrameCodecError> {
    if rows.iter().any(|row| row.len() != schema.fields().len()) {
        return Err(ControlFrameCodecError::LogicalFieldCount);
    }
    let builder_types = schema
        .fields()
        .iter()
        .map(|field| builder_data_type(field.data_type()))
        .collect::<Vec<_>>();
    let mut builders = builder_types
        .iter()
        .map(|data_type| make_builder(data_type, rows.len()))
        .collect::<Vec<_>>();
    for row in rows {
        for ((builder, data_type), value) in builders.iter_mut().zip(&builder_types).zip(row) {
            append_value(builder.as_mut(), data_type, value)?;
        }
    }
    let arrays = builders
        .iter_mut()
        .zip(schema.fields())
        .map(|(builder, field)| retype_array(builder.finish(), field.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema, arrays).map_err(ControlFrameCodecError::Arrow)
}

fn builder_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| {
                    Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(builder_data_type(field.data_type())),
                    )
                })
                .collect(),
        ),
        DataType::List(field) => DataType::List(Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(builder_data_type(field.data_type())),
        )),
        DataType::Map(field, _) => DataType::Map(
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_data_type(builder_data_type(field.data_type())),
            ),
            false,
        ),
        _ => data_type.clone(),
    }
}

fn retype_array(
    array: ArrayRef,
    desired_type: &DataType,
) -> Result<ArrayRef, ControlFrameCodecError> {
    let data = array.to_data();
    let desired_children = match desired_type {
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| field.data_type())
            .collect::<Vec<_>>(),
        DataType::List(field) | DataType::Map(field, _) => vec![field.data_type()],
        DataType::Dictionary(_, value) => vec![value.as_ref()],
        _ => Vec::new(),
    };
    if data.child_data().len() != desired_children.len() {
        return Err(ControlFrameCodecError::LogicalFieldCount);
    }
    let child_data = data
        .child_data()
        .iter()
        .zip(desired_children)
        .map(|(child, desired)| {
            retype_array(make_array(child.clone()), desired).map(|value| value.to_data())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rebuilt = data
        .into_builder()
        .data_type(desired_type.clone())
        .child_data(child_data)
        .build()
        .map_err(ControlFrameCodecError::Arrow)?;
    Ok(make_array(rebuilt))
}

fn append_value(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    value: &LogicalValue,
) -> Result<(), ControlFrameCodecError> {
    if matches!(value, LogicalValue::Null) {
        return append_null(builder, data_type);
    }
    match data_type {
        DataType::Int64 => {
            let LogicalValue::Signed(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            downcast_builder::<Int64Builder>(builder, data_type)?.append_value(
                i64::try_from(*value).map_err(|_| ControlFrameCodecError::IntegerOutOfRange)?,
            );
            Ok(())
        }
        DataType::UInt64 => {
            let LogicalValue::Unsigned(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            downcast_builder::<UInt64Builder>(builder, data_type)?.append_value(
                u64::try_from(*value).map_err(|_| ControlFrameCodecError::IntegerOutOfRange)?,
            );
            Ok(())
        }
        DataType::Decimal128(38, 0) => {
            let LogicalValue::Decimal128(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            downcast_builder::<Decimal128Builder>(builder, data_type)?.append_value(*value);
            Ok(())
        }
        DataType::Utf8 => {
            let LogicalValue::String(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            downcast_builder::<StringBuilder>(builder, data_type)?.append_value(value);
            Ok(())
        }
        DataType::FixedSizeBinary(width) => {
            let LogicalValue::Binary(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            if value.len() != usize::try_from(*width).unwrap_or(usize::MAX) {
                return Err(ControlFrameCodecError::FixedBinaryLength);
            }
            downcast_builder::<FixedSizeBinaryBuilder>(builder, data_type)?
                .append_value(value)
                .map_err(ControlFrameCodecError::Arrow)
        }
        DataType::Dictionary(index, values)
            if index.as_ref() == &DataType::Int8 && values.as_ref() == &DataType::Utf8 =>
        {
            let LogicalValue::String(value) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            downcast_builder::<StringDictionaryBuilder<Int8Type>>(builder, data_type)?
                .append(value)
                .map(|_| ())
                .map_err(ControlFrameCodecError::Arrow)
        }
        DataType::Struct(fields) => {
            let LogicalValue::Struct(values) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            if values.len() != fields.len() {
                return Err(ControlFrameCodecError::LogicalFieldCount);
            }
            let builder = downcast_builder::<StructBuilder>(builder, data_type)?;
            for ((child, field), value) in builder
                .field_builders_mut()
                .iter_mut()
                .zip(fields)
                .zip(values)
            {
                append_value(child.as_mut(), field.data_type(), value)?;
            }
            builder.append(true);
            Ok(())
        }
        DataType::List(field) => {
            let builder =
                downcast_builder::<ListBuilder<Box<dyn ArrayBuilder>>>(builder, data_type)?;
            let LogicalValue::List(values) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            for value in values {
                append_value(builder.values().as_mut(), field.data_type(), value)?;
            }
            builder.append(true);
            Ok(())
        }
        DataType::Map(field, _) => {
            let DataType::Struct(fields) = field.data_type() else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            let LogicalValue::StringMap(entries) = value else {
                return Err(ControlFrameCodecError::LogicalTypeMismatch);
            };
            let builder = downcast_builder::<
                MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>,
            >(builder, data_type)?;
            let (keys, values) = builder.entries();
            for (key, value) in entries {
                append_value(keys.as_mut(), fields[0].data_type(), &text(key))?;
                append_value(values.as_mut(), fields[1].data_type(), &text(value))?;
            }
            builder.append(true).map_err(ControlFrameCodecError::Arrow)
        }
        _ => Err(ControlFrameCodecError::UnsupportedArrowType(
            data_type.clone(),
        )),
    }
}

fn append_null(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
) -> Result<(), ControlFrameCodecError> {
    match data_type {
        DataType::Int64 => downcast_builder::<Int64Builder>(builder, data_type)?.append_null(),
        DataType::UInt64 => downcast_builder::<UInt64Builder>(builder, data_type)?.append_null(),
        DataType::Decimal128(38, 0) => {
            downcast_builder::<Decimal128Builder>(builder, data_type)?.append_null()
        }
        DataType::Utf8 => downcast_builder::<StringBuilder>(builder, data_type)?.append_null(),
        DataType::FixedSizeBinary(_) => {
            downcast_builder::<FixedSizeBinaryBuilder>(builder, data_type)?.append_null()
        }
        DataType::Dictionary(index, values)
            if index.as_ref() == &DataType::Int8 && values.as_ref() == &DataType::Utf8 =>
        {
            downcast_builder::<StringDictionaryBuilder<Int8Type>>(builder, data_type)?.append_null()
        }
        DataType::Struct(fields) => {
            let builder = downcast_builder::<StructBuilder>(builder, data_type)?;
            for (child, field) in builder.field_builders_mut().iter_mut().zip(fields) {
                append_null(child.as_mut(), field.data_type())?;
            }
            builder.append(false);
        }
        DataType::List(_) => {
            downcast_builder::<ListBuilder<Box<dyn ArrayBuilder>>>(builder, data_type)?
                .append(false);
        }
        DataType::Map(_, _) => {
            downcast_builder::<MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>>(
                builder, data_type,
            )?
            .append(false)
            .map_err(ControlFrameCodecError::Arrow)?;
        }
        _ => {
            return Err(ControlFrameCodecError::UnsupportedArrowType(
                data_type.clone(),
            ));
        }
    }
    Ok(())
}

fn downcast_builder<'a, T: 'static>(
    builder: &'a mut dyn ArrayBuilder,
    data_type: &DataType,
) -> Result<&'a mut T, ControlFrameCodecError> {
    builder
        .as_any_mut()
        .downcast_mut::<T>()
        .ok_or_else(|| ControlFrameCodecError::BuilderType(data_type.clone()))
}

fn encode_payload(
    projections: &[FrameTableProjectionV1],
) -> Result<Vec<u8>, ControlFrameCodecError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(PAYLOAD_MAGIC);
    payload.extend_from_slice(&PAYLOAD_VERSION.to_be_bytes());
    payload.extend_from_slice(
        &u16::try_from(projections.len())
            .map_err(|_| ControlFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    for projection in projections {
        payload.push(projection.table as u8);
        match projection.source_id.as_deref() {
            None => payload.push(0),
            Some(source_id) => {
                payload.push(1);
                encode_bytes(&mut payload, source_id.as_bytes())?;
            }
        }
        let mut ipc = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc, projection.batch.schema().as_ref())
                .map_err(ControlFrameCodecError::Arrow)?;
            writer
                .write(&projection.batch)
                .map_err(ControlFrameCodecError::Arrow)?;
            writer.finish().map_err(ControlFrameCodecError::Arrow)?;
        }
        encode_bytes(&mut payload, &ipc)?;
    }
    Ok(payload)
}

fn decode_payload(
    schemas: &ArchiveSchemasV1,
    archive_id: ArchiveId,
    session_id: SessionId,
    frame_id: crate::FrameId,
    authoritative_frame_clock_ns: i64,
    payload: &[u8],
) -> Result<Vec<FrameTableProjectionV1>, ControlFrameCodecError> {
    let mut cursor = PayloadCursor::new(payload);
    if cursor.take(PAYLOAD_MAGIC.len())? != PAYLOAD_MAGIC {
        return Err(ControlFrameCodecError::PayloadMagic);
    }
    let version = cursor.u16()?;
    if version != PAYLOAD_VERSION {
        return Err(ControlFrameCodecError::PayloadVersion(version));
    }
    let count = usize::from(cursor.u16()?);
    let mut projections = Vec::with_capacity(count);
    let mut previous = None;
    for _ in 0..count {
        let table = TableId::from_u8(cursor.u8()?).map_err(ControlFrameCodecError::LogicalRow)?;
        if previous >= Some(table) {
            return Err(ControlFrameCodecError::ControlProjectionShape);
        }
        previous = Some(table);
        let source_id = match cursor.u8()? {
            0 => None,
            1 => Some(
                std::str::from_utf8(cursor.bytes()?)
                    .map_err(|_| ControlFrameCodecError::PayloadUtf8)?
                    .to_owned(),
            ),
            _ => return Err(ControlFrameCodecError::PayloadTag),
        };
        let ipc = cursor.bytes()?;
        let mut reader =
            StreamReader::try_new(Cursor::new(ipc), None).map_err(ControlFrameCodecError::Arrow)?;
        let batch = reader
            .next()
            .ok_or(ControlFrameCodecError::MissingIpcBatch)?
            .map_err(ControlFrameCodecError::Arrow)?;
        if reader.next().is_some() {
            return Err(ControlFrameCodecError::ExtraIpcBatch);
        }
        let schema = schemas
            .table(table)
            .map_err(ControlFrameCodecError::Schema)?;
        if batch.schema().as_ref() != schema.schema().as_ref() {
            return Err(ControlFrameCodecError::IpcSchemaMismatch(table));
        }
        let logical_rows = schema
            .canonical_rows(&batch)
            .map_err(ControlFrameCodecError::Schema)?;
        projections.push(FrameTableProjectionV1 {
            archive_id,
            session_id,
            source_id,
            frame_id,
            authoritative_frame_clock_ns,
            table,
            batch,
            logical_rows,
        });
    }
    if !cursor.is_empty() {
        return Err(ControlFrameCodecError::PayloadTrailingBytes);
    }
    Ok(projections)
}

fn encode_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<(), ControlFrameCodecError> {
    output.extend_from_slice(
        &u64::try_from(value.len())
            .map_err(|_| ControlFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    output.extend_from_slice(value);
    Ok(())
}

struct PayloadCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ControlFrameCodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ControlFrameCodecError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ControlFrameCodecError::PayloadTruncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ControlFrameCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ControlFrameCodecError> {
        Ok(u16::from_be_bytes(
            self.take(2)?.try_into().expect("checked two bytes"),
        ))
    }

    fn bytes(&mut self) -> Result<&'a [u8], ControlFrameCodecError> {
        let length = u64::from_be_bytes(self.take(8)?.try_into().expect("checked eight bytes"));
        let length = usize::try_from(length).map_err(|_| ControlFrameCodecError::LengthOverflow)?;
        self.take(length)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Control DTO, Arrow evidence, identity, or recoverable WAL failure.
#[derive(Debug)]
pub enum ControlFrameCodecError {
    /// Lifecycle DTO violated its kind-specific matrix.
    Lifecycle(LifecycleMarkerError),
    /// Loss DTO violated its range/saturation matrix.
    Loss(LossValidationError),
    /// Frame identity input was invalid.
    Identity(FrameIdentityError),
    /// Checked-in schema load or value validation failed.
    Schema(SchemaError),
    /// Arrow array, record-batch, or IPC operation failed.
    Arrow(ArrowError),
    /// Canonical logical-row encoding failed.
    LogicalRow(LogicalRowError),
    /// Whole-frame projection invariants failed.
    Projection(ParquetProjectionError),
    /// WAL header/frame construction failed.
    Wal(WalError),
    /// Source-projection terminal used a non-projection loss DTO.
    SourceProjectionLossKind,
    /// This codec received an ordinary source-scrape frame.
    UnsupportedTerminalKind,
    /// Control payload did not contain exactly one expected one-row table.
    ControlProjectionShape,
    /// Header declaration disagreed with canonical payload evidence.
    ProjectionEvidenceMismatch,
    /// Derived terminal identity and projected row disagreed.
    TerminalIdentityMismatch,
    /// Recovered control row violated terminal-kind semantic invariants.
    RecoveredSemanticMismatch,
    /// Logical row did not have the schema's field count.
    LogicalFieldCount,
    /// Logical value variant disagreed with the Arrow field.
    LogicalTypeMismatch,
    /// Integer exceeded its frozen physical width.
    IntegerOutOfRange,
    /// Fixed-size binary bytes had the wrong length.
    FixedBinaryLength,
    /// Dynamic Arrow builder disagreed with its field.
    BuilderType(DataType),
    /// Minimal control-row builder encountered another Arrow type.
    UnsupportedArrowType(DataType),
    /// Count or byte offset overflowed its frozen width.
    LengthOverflow,
    /// WAL payload magic was not the shared v1 frame-payload magic.
    PayloadMagic,
    /// WAL payload version is unsupported.
    PayloadVersion(u16),
    /// WAL payload ended before a declared field.
    PayloadTruncated,
    /// WAL payload carried an unknown option tag.
    PayloadTag,
    /// WAL payload source identity was not UTF-8.
    PayloadUtf8,
    /// WAL payload carried trailing bytes.
    PayloadTrailingBytes,
    /// IPC stream contained no batch.
    MissingIpcBatch,
    /// IPC stream contained more than one batch.
    ExtraIpcBatch,
    /// IPC stream did not use the checked-in schema.
    IpcSchemaMismatch(TableId),
}

impl Display for ControlFrameCodecError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lifecycle(error) => write!(formatter, "lifecycle marker: {error}"),
            Self::Loss(error) => write!(formatter, "telemetry loss: {error}"),
            Self::Identity(error) => write!(formatter, "frame identity: {error}"),
            Self::Schema(error) => write!(formatter, "archive schema: {error}"),
            Self::Arrow(error) => write!(formatter, "Arrow/IPC: {error}"),
            Self::LogicalRow(error) => write!(formatter, "logical row: {error}"),
            Self::Projection(error) => write!(formatter, "frame projection: {error}"),
            Self::Wal(error) => write!(formatter, "WAL frame: {error}"),
            Self::SourceProjectionLossKind => formatter
                .write_str("source projection failure requires projection_failed loss kind"),
            Self::UnsupportedTerminalKind => {
                formatter.write_str("control codec received source-scrape terminal kind")
            }
            Self::ControlProjectionShape => formatter
                .write_str("control payload must contain exactly one expected one-row projection"),
            Self::ProjectionEvidenceMismatch => {
                formatter.write_str("control payload evidence disagrees with its WAL declaration")
            }
            Self::TerminalIdentityMismatch => {
                formatter.write_str("control row and WAL terminal identities disagree")
            }
            Self::RecoveredSemanticMismatch => {
                formatter.write_str("recovered control row violates terminal semantics")
            }
            Self::LogicalFieldCount => formatter.write_str("control logical field count mismatch"),
            Self::LogicalTypeMismatch => formatter.write_str("control logical value type mismatch"),
            Self::IntegerOutOfRange => formatter.write_str("control integer is out of range"),
            Self::FixedBinaryLength => formatter.write_str("control fixed binary length mismatch"),
            Self::BuilderType(data_type) => {
                write!(formatter, "control Arrow builder mismatch for {data_type}")
            }
            Self::UnsupportedArrowType(data_type) => {
                write!(formatter, "unsupported control Arrow type {data_type}")
            }
            Self::LengthOverflow => formatter.write_str("control payload length overflow"),
            Self::PayloadMagic => formatter.write_str("invalid control WAL payload magic"),
            Self::PayloadVersion(version) => {
                write!(
                    formatter,
                    "unsupported control WAL payload version {version}"
                )
            }
            Self::PayloadTruncated => formatter.write_str("truncated control WAL payload"),
            Self::PayloadTag => formatter.write_str("invalid control WAL payload tag"),
            Self::PayloadUtf8 => formatter.write_str("control payload source is not UTF-8"),
            Self::PayloadTrailingBytes => formatter.write_str("trailing control WAL payload bytes"),
            Self::MissingIpcBatch => formatter.write_str("control IPC stream has no batch"),
            Self::ExtraIpcBatch => formatter.write_str("control IPC stream has multiple batches"),
            Self::IpcSchemaMismatch(table) => {
                write!(formatter, "control IPC schema mismatch for {table:?}")
            }
        }
    }
}

impl std::error::Error for ControlFrameCodecError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        LifecycleCompletionReasonV1, LifecycleMarkerKindV1, LifecyclePhaseStateV1,
        ProjectionEvidence, SourceOutcome, loss_saturation_slot_id_v1,
    };

    fn id(seed: u8) -> [u8; 16] {
        let mut value = [seed; 16];
        value[15] = seed.wrapping_add(1);
        value
    }

    fn archive_id() -> ArchiveId {
        ArchiveId::new(id(1)).unwrap()
    }

    fn session_id() -> SessionId {
        SessionId::new(id(2)).unwrap()
    }

    fn marker(kind: LifecycleMarkerKindV1, record_seq: u64) -> LifecycleMarkerV1 {
        let mut marker = LifecycleMarkerV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            record_seq,
            marker_seq: record_seq,
            kind,
            clock_ns: i64::try_from(record_seq).unwrap() * 100,
            unix_epoch_ns: Some(1_700_000_000_000_000_000 + i128::from(record_seq)),
            run_id: None,
            phase_id: None,
            source_id: None,
            phase_state: None,
            completion_reason: None,
            boundary: None,
            phase_start_ns: None,
            sent_end_ns: None,
            requests_end_ns: None,
            attribute_epoch_id: None,
            attributes: BTreeMap::new(),
        };
        match kind {
            LifecycleMarkerKindV1::SessionStarted => {}
            LifecycleMarkerKindV1::SessionStopped => {
                marker.completion_reason = Some(LifecycleCompletionReasonV1::Shutdown);
            }
            LifecycleMarkerKindV1::RunStarted => marker.run_id = Some("run-a".to_owned()),
            LifecycleMarkerKindV1::RunStopped => {
                marker.run_id = Some("run-a".to_owned());
                marker.completion_reason = Some(LifecycleCompletionReasonV1::Completed);
            }
            LifecycleMarkerKindV1::PhaseStarted => {
                marker.run_id = Some("run-a".to_owned());
                marker.phase_id = Some("profiling".to_owned());
                marker.phase_state = Some(LifecyclePhaseStateV1::Started);
                marker.phase_start_ns = Some(marker.clock_ns);
            }
            LifecycleMarkerKindV1::PhaseSendingComplete => {
                marker.run_id = Some("run-a".to_owned());
                marker.phase_id = Some("profiling".to_owned());
                marker.phase_state = Some(LifecyclePhaseStateV1::SendingComplete);
                marker.phase_start_ns = Some(marker.clock_ns - 20);
                marker.sent_end_ns = Some(marker.clock_ns);
            }
            LifecycleMarkerKindV1::PhaseComplete => {
                marker.run_id = Some("run-a".to_owned());
                marker.phase_id = Some("profiling".to_owned());
                marker.phase_state = Some(LifecyclePhaseStateV1::Complete);
                marker.completion_reason = Some(LifecycleCompletionReasonV1::Duration);
                marker.phase_start_ns = Some(marker.clock_ns - 30);
                marker.sent_end_ns = Some(marker.clock_ns - 10);
                marker.requests_end_ns = Some(marker.clock_ns);
            }
            LifecycleMarkerKindV1::SourceState => {
                marker.source_id = Some("server-a".to_owned());
                marker
                    .attributes
                    .insert("state".to_owned(), "disabled".to_owned());
            }
            LifecycleMarkerKindV1::TopologyChange => {
                marker.source_id = Some("server-a".to_owned());
                marker.attribute_epoch_id = Some(Digest::from_bytes([7; 32]));
                marker
                    .attributes
                    .insert("cluster".to_owned(), "lab-a".to_owned());
            }
            LifecycleMarkerKindV1::ArchiveDegraded => {
                marker
                    .attributes
                    .insert("reason".to_owned(), "writer_lag".to_owned());
            }
            LifecycleMarkerKindV1::ArchiveRecovered => {
                marker
                    .attributes
                    .insert("state".to_owned(), "healthy".to_owned());
            }
        }
        marker
    }

    fn exact_loss(kind: LossKindV1, record_seq: u64) -> ExactLossRangeV1 {
        let missed = kind == LossKindV1::MissedCadence;
        ExactLossRangeV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            source_id: Some("server-a".to_owned()),
            record_seq,
            loss_seq: record_seq + 10,
            count: 2,
            loss_kind: kind,
            reason: kind.reason(),
            first_source_record_seq: (!missed).then_some(20),
            last_source_record_seq: (!missed).then_some(21),
            first_request_attempt_seq: (!missed).then_some(30),
            last_request_attempt_seq: (!missed).then_some(31),
            first_tick: missed.then_some(40),
            last_tick: missed.then_some(41),
            first_deadline_ns: missed.then_some(4_000),
            last_deadline_ns: missed.then_some(5_000),
            loss_observed_ns: 5_100 + i64::try_from(record_seq).unwrap(),
            boundary_refs: Vec::new(),
            boundary_overflow_count: 0,
            boundary_overflow_digest: None,
        }
    }

    fn saturation(kind: LossKindV1, record_seq: u64) -> LossSaturationSnapshotV1 {
        let missed = kind == LossKindV1::MissedCadence;
        let source_id = Some("server-a".to_owned());
        let reason = kind.reason();
        LossSaturationSnapshotV1 {
            archive_id: archive_id(),
            session_id: session_id(),
            saturation_slot_id: loss_saturation_slot_id_v1(
                archive_id(),
                session_id(),
                source_id.as_deref(),
                kind,
                reason,
            ),
            source_id,
            record_seq,
            loss_seq: record_seq + 20,
            loss_kind: kind,
            reason,
            first_source_record_seq: (!missed).then_some(50),
            last_source_record_seq: (!missed).then_some(60),
            first_request_attempt_seq: (!missed).then_some(70),
            last_request_attempt_seq: (!missed).then_some(80),
            first_tick: missed.then_some(90),
            last_tick: missed.then_some(100),
            first_deadline_ns: missed.then_some(9_000),
            last_deadline_ns: missed.then_some(10_000),
            loss_observed_ns: 10_100 + i64::try_from(record_seq).unwrap(),
            saturation_snapshot_seq: 3,
            cumulative_omitted_range_count: 4,
            cumulative_omitted_entry_count: 11,
            omitted_rolling_digest: Digest::from_bytes([8; 32]),
        }
    }

    fn wal_roundtrip(codec: &ControlFrameCodecV1, encoded: ArchiveWalFrame) -> ArchiveWalFrame {
        let bytes = encoded.wal_frame.encode().unwrap();
        let wal = WalFrame::decode(&bytes, crate::wal::DEFAULT_MAX_WAL_FRAME_BYTES).unwrap();
        codec
            .decode_control_frame(archive_id(), session_id(), wal)
            .unwrap()
    }

    #[test]
    fn missed_cadence_exact_loss_round_trips_wal_and_nonzero_evidence() {
        let codec = ControlFrameCodecV1::new().unwrap();
        let encoded = codec
            .encode_exact_loss_frame(exact_loss(LossKindV1::MissedCadence, 7))
            .unwrap();

        assert_eq!(
            encoded.wal_frame.header().terminal_kind,
            TerminalKind::LossExact
        );
        assert_eq!(
            encoded.wal_frame.header().required_projections,
            vec![RequiredProjection {
                table: TableId::Losses,
                evidence: encoded.table_projections[0]
                    .validate(&codec.schemas)
                    .unwrap(),
            }]
        );
        assert_eq!(
            encoded.wal_frame.header().required_projections[0]
                .evidence
                .row_count,
            1
        );
        assert_ne!(
            encoded.wal_frame.header().required_projections[0].evidence,
            ProjectionEvidence::empty()
        );

        let recovered = wal_roundtrip(&codec, encoded);
        assert_eq!(recovered.table_projections[0].batch.num_rows(), 1);
        assert_eq!(recovered.table_projections[0].table, TableId::Losses);
    }

    #[test]
    fn every_lifecycle_and_loss_variant_round_trips_its_terminal_kind() {
        let codec = ControlFrameCodecV1::new().unwrap();
        let marker_kinds = [
            LifecycleMarkerKindV1::SessionStarted,
            LifecycleMarkerKindV1::SessionStopped,
            LifecycleMarkerKindV1::RunStarted,
            LifecycleMarkerKindV1::RunStopped,
            LifecycleMarkerKindV1::PhaseStarted,
            LifecycleMarkerKindV1::PhaseSendingComplete,
            LifecycleMarkerKindV1::PhaseComplete,
            LifecycleMarkerKindV1::SourceState,
            LifecycleMarkerKindV1::TopologyChange,
            LifecycleMarkerKindV1::ArchiveDegraded,
            LifecycleMarkerKindV1::ArchiveRecovered,
        ];
        for (index, kind) in marker_kinds.into_iter().enumerate() {
            let encoded = codec
                .encode_lifecycle_frame(marker(kind, u64::try_from(index).unwrap()))
                .unwrap_or_else(|error| panic!("{kind:?}: {error}"));
            assert_eq!(
                wal_roundtrip(&codec, encoded)
                    .wal_frame
                    .header()
                    .terminal_kind,
                TerminalKind::LifecycleMarker
            );
        }

        let loss_kinds = [
            LossKindV1::MissedCadence,
            LossKindV1::ArchiveRejected,
            LossKindV1::ProjectionFailed,
            LossKindV1::WriterFailed,
            LossKindV1::ShutdownAbandoned,
        ];
        for (index, kind) in loss_kinds.into_iter().enumerate() {
            let record_seq = 20 + u64::try_from(index).unwrap();
            let exact = codec
                .encode_exact_loss_frame(exact_loss(kind, record_seq))
                .unwrap_or_else(|error| panic!("exact {kind:?}: {error}"));
            assert_eq!(
                wal_roundtrip(&codec, exact)
                    .wal_frame
                    .header()
                    .terminal_kind,
                TerminalKind::LossExact
            );
            let saturated = codec
                .encode_loss_saturation_frame(saturation(kind, record_seq + 10))
                .unwrap_or_else(|error| panic!("saturation {kind:?}: {error}"));
            assert_eq!(
                wal_roundtrip(&codec, saturated)
                    .wal_frame
                    .header()
                    .terminal_kind,
                TerminalKind::LossSaturation
            );
        }
    }

    #[test]
    fn source_projection_failure_reuses_source_reservation_and_records_loss_seq() {
        let codec = ControlFrameCodecV1::new().unwrap();
        let mut loss = exact_loss(LossKindV1::ProjectionFailed, 90);
        loss.count = 1;
        loss.last_source_record_seq = loss.first_source_record_seq;
        loss.last_request_attempt_seq = loss.first_request_attempt_seq;
        let source_batch = FrameIdentityV1::source_scrape_batch(
            archive_id(),
            session_id(),
            "server-a",
            20,
            SourceOutcome::Success,
            None,
        )
        .unwrap();
        let source_reservation = FrameIdentityV1::projection_reservation(
            archive_id(),
            session_id(),
            ReservationKind::SourceScrape,
            Some("server-a"),
            source_batch,
            loss.record_seq,
        )
        .unwrap();
        let encoded = codec
            .encode_source_projection_failed(loss, source_batch, source_reservation)
            .unwrap();
        assert_eq!(encoded.wal_frame.header().batch_id, source_batch);
        assert_eq!(
            encoded.wal_frame.header().projection_reservation_id,
            source_reservation
        );
        assert_eq!(
            wal_roundtrip(&codec, encoded)
                .wal_frame
                .header()
                .terminal_kind,
            TerminalKind::SourceProjectionFailed
        );
    }

    #[test]
    fn boundary_evidence_binds_exact_loss_identity_and_global_sentinel_round_trips() {
        let codec = ControlFrameCodecV1::new().unwrap();
        let mut boundary_loss = exact_loss(LossKindV1::ArchiveRejected, 95);
        boundary_loss.count = 1;
        boundary_loss.last_source_record_seq = boundary_loss.first_source_record_seq;
        boundary_loss.last_request_attempt_seq = boundary_loss.first_request_attempt_seq;
        boundary_loss.boundary_refs = vec![BoundaryReference {
            transition_id: "warmup-to-profiling".to_owned(),
            boundary_id: "server-a-warmup-end".to_owned(),
            phase_id: "warmup".to_owned(),
            source_id: "server-a".to_owned(),
            role: BoundaryRole::PhaseEnd,
            coalescing_group_id: None,
        }];
        boundary_loss.boundary_overflow_count = 2;
        boundary_loss.boundary_overflow_digest = Some(Digest::from_bytes([6; 32]));
        let without_boundary = {
            let mut value = boundary_loss.clone();
            value.boundary_refs.clear();
            value.boundary_overflow_count = 0;
            value.boundary_overflow_digest = None;
            codec.encode_exact_loss_frame(value).unwrap()
        };
        let with_boundary = codec.encode_exact_loss_frame(boundary_loss).unwrap();
        assert_ne!(
            without_boundary.wal_frame.header().batch_id,
            with_boundary.wal_frame.header().batch_id
        );
        wal_roundtrip(&codec, with_boundary);

        let mut global = exact_loss(LossKindV1::WriterFailed, 96);
        global.source_id = None;
        global.first_source_record_seq = None;
        global.last_source_record_seq = None;
        global.first_request_attempt_seq = None;
        global.last_request_attempt_seq = None;
        global.count = 3;
        let recovered = wal_roundtrip(&codec, codec.encode_exact_loss_frame(global).unwrap());
        assert_eq!(recovered.table_projections[0].source_id, None);
    }

    #[test]
    fn recovery_rejects_zero_nonzero_evidence_substitution_and_empty_control_rows() {
        let codec = ControlFrameCodecV1::new().unwrap();
        let encoded = codec
            .encode_lifecycle_frame(marker(LifecycleMarkerKindV1::SessionStarted, 100))
            .unwrap();
        let original = encoded.wal_frame.header();
        let false_zero_header = WalFrameHeaderV1::new(
            original.batch_id,
            original.projection_reservation_id,
            original.record_seq,
            original.authoritative_frame_clock_ns,
            original.terminal_kind,
            vec![RequiredProjection {
                table: TableId::Markers,
                evidence: ProjectionEvidence::empty(),
            }],
            Vec::new(),
            Vec::new(),
            original.payload_len,
        )
        .unwrap();
        let false_zero =
            WalFrame::new(false_zero_header, encoded.wal_frame.payload().to_vec()).unwrap();
        assert!(matches!(
            codec.decode_control_frame(archive_id(), session_id(), false_zero),
            Err(ControlFrameCodecError::ProjectionEvidenceMismatch)
        ));

        let empty_projection = codec
            .projection(
                archive_id(),
                session_id(),
                None,
                original.frame_id,
                original.authoritative_frame_clock_ns,
                TableId::Markers,
                Vec::new(),
            )
            .unwrap();
        let empty_evidence = empty_projection.validate(&codec.schemas).unwrap();
        assert_eq!(empty_evidence, ProjectionEvidence::empty());
        let empty_payload = encode_payload(std::slice::from_ref(&empty_projection)).unwrap();
        let empty_header = WalFrameHeaderV1::new(
            original.batch_id,
            original.projection_reservation_id,
            original.record_seq,
            original.authoritative_frame_clock_ns,
            original.terminal_kind,
            vec![RequiredProjection {
                table: TableId::Markers,
                evidence: empty_evidence,
            }],
            Vec::new(),
            Vec::new(),
            u64::try_from(empty_payload.len()).unwrap(),
        )
        .unwrap();
        let empty = WalFrame::new(empty_header, empty_payload).unwrap();
        assert!(matches!(
            codec.decode_control_frame(archive_id(), session_id(), empty),
            Err(ControlFrameCodecError::ControlProjectionShape)
        ));
    }
}
