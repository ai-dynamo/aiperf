// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded streaming decoder for Baseten literal-prompt Parquet traces.
//!
//! One immutable Parquet object is one partition. Its footer is read once
//! through two bounded ranged reads, its column chunks are read one row group
//! at a time, and only the staged row group's rows are ever resident. Nothing
//! proportional to the object, the partition inventory, or the session
//! population is retained.
//!
//! Column, KV-hint, output-cap, and recorded-outcome semantics mirror the
//! finite loader in [`crate::dataset::loader::baseten`]: the same required and
//! optional column names, the same `max(1)` output-length floor, the same
//! `omit_kv_hints`/`force_min_tokens` gates, and the same session-column
//! resolution order.
//!
//! Replay *scheduling* is deliberately absent. Minimum-timestamp
//! normalization, replay speedup, idle-gap reflow, and closed-loop
//! back-pressure each need the complete trace resident, which a bounded
//! decoder does not have. This decoder emits the absolute recorded facts —
//! the recorded request start as event time, and the recorded durations in the
//! replay-parameter payload — and the downstream session program and action
//! host own global ordering and pacing.
//!
//! One kept row becomes one causally ordered fragment pair: an endpoint-neutral
//! [`ConversationTurnFragment`] carrying the verbatim recorded prompt, and an
//! [`AgentEventFragment`] carrying the canonical replay-parameter document.
//! The generation-one mutation vocabulary has no slot for per-turn request
//! parameters, and splicing a JSON document into turn content would re-send it
//! as prior-turn text, so the non-executable agent-event side channel is the
//! honest encoding inside the frozen vocabulary.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Buf, Bytes};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use smallvec::SmallVec;

use crate::streaming::{
    budget::StreamingResourceBudget,
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{DecodeFailureCode, OrdinaryStreamingFailure, StreamFormatError},
    format::{
        DecodeBatchBudget, DecodeReceipt, DecodeStep, DecodedFragmentBatch, DecoderCheckpoint,
        DecoderResumeState, FormatEvent, FormatEventSink, FormatProjection, FormatSealReceipt,
        FormatStateRetention, SessionWatermark, StreamingDatasetFormat,
        StreamingDatasetFormatFactory, StreamingFormatDescriptor, StreamingFormatPrepareContext,
        StreamingPartitionDecoder, ValidatedStreamingFormatConfig,
    },
    identity::{
        ContentDigest, ImmutableObjectIdentity, StableOrderKey, StableRecordId, StableSessionKey,
        one_turn_session_key, physical_record_id, stable_session_key,
    },
    reliability::{
        OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
        StreamingIssueReporterHandle,
    },
    source::{
        AcquiredPartitionAccess, AcquiredSeekableLocalPartition, AcquisitionBudget,
        PartitionAccessKind, SourceFrontier, SourceSeal, StreamingSourceDescriptor,
    },
    unit::{
        AgentEventFragment, ConversationTurnFragment, EventTimeUtc, SessionFragmentLease,
        SessionMutationV1, SourcePosition, StreamingSessionFragment, UnitProvenance,
    },
};

/// Stable registry identifier for the Baseten Parquet streaming format.
pub const BASETEN_FORMAT_ID: &str = "baseten_trace";
/// Canonical event family of the paired replay-parameter fragment.
pub const BASETEN_REPLAY_TURN_EVENT_KIND: &str = "aiperf.baseten.replay-turn.v1";
/// Stable schema identity of this decoder's checkpoint payload.
pub const BASETEN_FORMAT_SCHEMA_ID: &str = "aiperf.streaming.format.baseten";
/// Current decoder checkpoint schema version.
pub const BASETEN_FORMAT_SCHEMA_VERSION: u32 = 1;

const COL_TIME: &str = "timestamp_start_unix_ms";
const COL_PROMPT: &str = "prompt";
const COL_INPUT_TOKENS: &str = "input_tokens";
const COL_OUTPUT_TOKENS: &str = "output_tokens";
const COL_SESSION: &str = "provided_session_id";
const COL_POOR_MAN_SESSION: &str = "poor_man_session_id";
const COL_DURATION_E2E: &str = "duration_e2e_ms";
const COL_DURATION_TTFT: &str = "duration_ttft_ms";
const COL_CACHED_TOKENS: &str = "cached_tokens_reference";
const COL_TOTAL_HASHES: &str = "total_hashes";
const COL_BLOCK_SIZE: &str = "block_size";

/// Required Baseten trace columns; absence is a validation failure before decode.
const REQUIRED_COLUMNS: [&str; 4] = [COL_TIME, COL_PROMPT, COL_INPUT_TOKENS, COL_OUTPUT_TOKENS];

/// Exact encoded width of the decoder's opaque resume cursor.
const CURSOR_BYTES: usize = 52;
/// Cursor encoding version occupying the cursor's first byte.
const CURSOR_VERSION: u8 = 1;
/// Rows decoded into one Arrow batch while staging a row group.
const ARROW_BATCH_ROWS: usize = 128;
/// Fixed Parquet footer trailer width: metadata length plus magic.
const PARQUET_FOOTER_BYTES: usize = 8;
/// Recorded milliseconds to event-time nanoseconds.
const NANOS_PER_MILLI: i64 = 1_000_000;

/// Compile-time semantic identity of this decoder implementation.
///
/// Bumped whenever emitted fragment content changes. It is not the per-run
/// schema digest, which additionally binds the authored configuration and the
/// first partition's projected Arrow schema.
const BASETEN_SEMANTIC_DIGEST: ContentDigest = ContentDigest::from_bytes([
    0xba, 0x5e, 0x7e, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
]);

/// Immutable registry metadata for the Baseten Parquet streaming format.
pub static BASETEN_FORMAT_DESCRIPTOR: StreamingFormatDescriptor = StreamingFormatDescriptor {
    id: BASETEN_FORMAT_ID,
    description: "Baseten literal-prompt Parquet trace replay",
    semantic_digest: BASETEN_SEMANTIC_DIGEST,
    media_types: &["application/vnd.apache.parquet"],
    input_schemas: &["baseten.trace.parquet.v1"],
    required_access: PartitionAccessKind::SeekableLocal,
    projection: FormatProjection::BoundedFields,
    output_schema: "aiperf.stream.session-fragment.v1",
    has_event_time: true,
    has_stable_record_ids: true,
    retention: FormatStateRetention::BoundedMemory,
    supports_virtual_clock: true,
};

/// Recorded session column selected for cross-partition session grouping.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BasetenSessionColumn {
    /// Prefer the producer-authored `provided_session_id` column.
    #[default]
    Provided,
    /// Prefer the heuristic `poor_man_session_id` column.
    PoorMan,
}

impl BasetenSessionColumn {
    const fn column(self) -> &'static str {
        match self {
            Self::Provided => COL_SESSION,
            Self::PoorMan => COL_POOR_MAN_SESSION,
        }
    }

    const fn fallback(self) -> &'static str {
        match self {
            Self::Provided => COL_POOR_MAN_SESSION,
            Self::PoorMan => COL_SESSION,
        }
    }

    const fn canonical_tag(self) -> u8 {
        match self {
            Self::Provided => 0,
            Self::PoorMan => 1,
        }
    }
}

/// Strictly authored Baseten decoding policy, frozen before any partition.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasetenFormatConfig {
    /// Recorded session column preferred for session grouping.
    #[serde(default)]
    pub session_column: BasetenSessionColumn,
    /// Suppress the `hash_ids`/`block_size` routing hints.
    #[serde(default)]
    pub omit_kv_hints: bool,
    /// Emit `min_tokens` alongside the capped `max_tokens`.
    #[serde(default)]
    pub force_min_tokens: bool,
    /// Upper bound applied to every recorded output length.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Stable-hash whole-session admission fraction in `0.0..=1.0`.
    #[serde(default)]
    pub session_sample_ratio: Option<f64>,
    /// Refuse a row group whose column-chunk extent exceeds this bound.
    pub max_row_group_bytes: usize,
    /// Refuse a row whose decoded prompt exceeds this bound.
    pub max_prompt_bytes: usize,
}

/// Validated configuration plus everything resolved once at startup.
#[derive(Debug)]
struct ValidatedBasetenConfig {
    config: BasetenFormatConfig,
    /// Canonical bytes folded into the per-run schema digest.
    canonical_config_bytes: Vec<u8>,
    /// Stable-hash admission threshold, absent when every session is admitted.
    sample_threshold: Option<u64>,
}

fn schema_error() -> StreamFormatError {
    StreamFormatError::decode(DecodeFailureCode::Schema)
}

fn canonical_config_bytes(config: &BasetenFormatConfig) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.push(config.session_column.canonical_tag());
    bytes.push(u8::from(config.omit_kv_hints));
    bytes.push(u8::from(config.force_min_tokens));
    match config.max_output_tokens {
        Some(cap) => {
            bytes.push(1);
            bytes.extend_from_slice(&cap.to_le_bytes());
        }
        None => bytes.push(0),
    }
    match config.session_sample_ratio {
        Some(ratio) => {
            bytes.push(1);
            bytes.extend_from_slice(&ratio.to_bits().to_le_bytes());
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(config.max_row_group_bytes as u64).to_le_bytes());
    bytes.extend_from_slice(&(config.max_prompt_bytes as u64).to_le_bytes());
    bytes
}

/// Translate an authored admission fraction into a stable-hash threshold.
///
/// A fraction of exactly one admits every session and needs no hashing at all,
/// which is why the threshold is absent rather than `u64::MAX`.
fn sample_threshold(ratio: Option<f64>) -> Option<u64> {
    let ratio = ratio?;
    if ratio >= 1.0 {
        return None;
    }
    // 2^64 as f64 is exact; the product truncates toward zero, so a ratio below
    // one can never produce a threshold that admits every key.
    Some((ratio * 18_446_744_073_709_551_616.0) as u64)
}

/// Registry entry for the Baseten Parquet streaming format.
#[derive(Debug, Default)]
pub struct BasetenFormatFactory;

impl StreamingDatasetFormatFactory for BasetenFormatFactory {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor {
        &BASETEN_FORMAT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError> {
        if !source
            .access
            .contains(&BASETEN_FORMAT_DESCRIPTOR.required_access)
        {
            return Err(schema_error());
        }
        let config: BasetenFormatConfig =
            serde_json::from_str(authored.get()).map_err(|_| schema_error())?;
        if config.max_row_group_bytes == 0 || config.max_prompt_bytes == 0 {
            return Err(schema_error());
        }
        if config.max_output_tokens == Some(0) {
            return Err(schema_error());
        }
        if let Some(ratio) = config.session_sample_ratio
            && (!ratio.is_finite() || !(0.0..=1.0).contains(&ratio))
        {
            return Err(schema_error());
        }
        let canonical_config_bytes = canonical_config_bytes(&config);
        let sample_threshold = sample_threshold(config.session_sample_ratio);
        Ok(Box::new(ValidatedBasetenConfig {
            config,
            canonical_config_bytes,
            sample_threshold,
        }))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingFormatConfig>,
        context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError> {
        let config: Box<ValidatedBasetenConfig> =
            config.into_any().downcast().map_err(|_| schema_error())?;
        Ok(Box::new(BasetenFormat {
            run: context.run,
            stream_identity: context.stream_semantic_digest,
            reporter: context.issue_reporter.clone(),
            fragment_budget: context.fragment_budget.clone(),
            acquisition_budget: context.acquisition_budget.clone(),
            config: Rc::new(*config),
            frozen_schema_digest: Rc::new(RefCell::new(None)),
            partitions_sealed: Rc::new(Cell::new(0)),
            latest_event_time: Rc::new(Cell::new(None)),
            participant_id: CheckpointParticipantId::new("streaming-format-baseten"),
            initialization: ParticipantInitialization::default(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Absolute-offset window over one immutable object
// ---------------------------------------------------------------------------

/// Absolute-offset view over one retained window of an immutable object.
///
/// Parquet addresses column chunks by absolute file offset, so `Length::len`
/// reports the full object length while only the staged window is resident.
/// Every read outside that window is an error rather than a silent short read.
#[derive(Debug)]
struct RetainedWindow {
    object_len: u64,
    window_start: u64,
    bytes: Bytes,
}

impl RetainedWindow {
    fn window(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let offset = start.checked_sub(self.window_start).ok_or_else(|| {
            parquet::errors::ParquetError::General("read precedes the retained window".into())
        })?;
        let offset = usize::try_from(offset).map_err(|_| {
            parquet::errors::ParquetError::General("retained window offset overflow".into())
        })?;
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| {
                parquet::errors::ParquetError::General("read exceeds the retained window".into())
            })?;
        Ok(self.bytes.slice(offset..end))
    }
}

impl parquet::file::reader::Length for RetainedWindow {
    fn len(&self) -> u64 {
        self.object_len
    }
}

impl parquet::file::reader::ChunkReader for RetainedWindow {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        let offset = start.checked_sub(self.window_start).ok_or_else(|| {
            parquet::errors::ParquetError::General("read precedes the retained window".into())
        })?;
        let offset = usize::try_from(offset).map_err(|_| {
            parquet::errors::ParquetError::General("retained window offset overflow".into())
        })?;
        if offset > self.bytes.len() {
            return Err(parquet::errors::ParquetError::General(
                "read exceeds the retained window".into(),
            ));
        }
        Ok(self.bytes.slice(offset..).reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        self.window(start, length)
    }
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

/// Exact decoder-private position inside one immutable Parquet object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BasetenCursor {
    /// Next row group to stage.
    row_group: u32,
    /// Next row within the staged row group.
    row_in_group: u32,
    /// Rows already emitted from this partition.
    rows_emitted: u64,
    /// Frozen per-run schema digest bound into the cursor.
    schema_digest: ContentDigest,
}

impl BasetenCursor {
    /// 1 version + 3 reserved + 4 row group + 4 row + 8 emitted + 32 digest.
    fn encode(self) -> [u8; CURSOR_BYTES] {
        let mut buffer = [0_u8; CURSOR_BYTES];
        buffer[0] = CURSOR_VERSION;
        buffer[4..8].copy_from_slice(&self.row_group.to_le_bytes());
        buffer[8..12].copy_from_slice(&self.row_in_group.to_le_bytes());
        buffer[12..20].copy_from_slice(&self.rows_emitted.to_le_bytes());
        buffer[20..52].copy_from_slice(self.schema_digest.as_bytes());
        buffer
    }

    fn decode(bytes: &[u8]) -> Result<Self, StreamFormatError> {
        let invalid = || StreamFormatError::decode(DecodeFailureCode::InvalidCursor);
        // The reserved bytes are asserted zero so an unset field can never be
        // smuggled through a checkpoint.
        if bytes.len() != CURSOR_BYTES || bytes[0] != CURSOR_VERSION || bytes[1..4] != [0, 0, 0] {
            return Err(invalid());
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&bytes[20..52]);
        let row_group = bytes[4..8].try_into().map_err(|_| invalid())?;
        let row_in_group = bytes[8..12].try_into().map_err(|_| invalid())?;
        let rows_emitted = bytes[12..20].try_into().map_err(|_| invalid())?;
        Ok(Self {
            row_group: u32::from_le_bytes(row_group),
            row_in_group: u32::from_le_bytes(row_in_group),
            rows_emitted: u64::from_le_bytes(rows_emitted),
            schema_digest: ContentDigest::from_bytes(digest),
        })
    }
}

// ---------------------------------------------------------------------------
// Read plan
// ---------------------------------------------------------------------------

/// Bounded byte extent and row span of one row group.
#[derive(Clone, Copy, Debug)]
struct RowGroupPlan {
    index: usize,
    start: u64,
    length: usize,
    first_row: u64,
    num_rows: u64,
}

/// Everything resolved once per partition, before any row is decoded.
struct BasetenReadPlan {
    metadata: Arc<parquet::file::metadata::ParquetMetaData>,
    projection: parquet::arrow::ProjectionMask,
    session_column: Option<&'static str>,
    row_groups: Vec<RowGroupPlan>,
    schema_digest: ContentDigest,
    object_len: u64,
}

/// Resolve the session column exactly as the finite loader does: the requested
/// column when present, otherwise the other one, otherwise no grouping column.
fn resolve_session_column(
    requested: BasetenSessionColumn,
    has_provided: bool,
    has_poor_man: bool,
) -> Option<&'static str> {
    let has_requested = match requested {
        BasetenSessionColumn::Provided => has_provided,
        BasetenSessionColumn::PoorMan => has_poor_man,
    };
    if has_requested {
        return Some(requested.column());
    }
    let has_fallback = match requested {
        BasetenSessionColumn::Provided => has_poor_man,
        BasetenSessionColumn::PoorMan => has_provided,
    };
    has_fallback.then_some(requested.fallback())
}

fn update_digest_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

// ---------------------------------------------------------------------------
// Row projection
// ---------------------------------------------------------------------------

/// One projected Baseten row with no replay-scheduling slots.
#[derive(Clone, Debug)]
struct BasetenRowProjection {
    row_in_group: u32,
    absolute_row: u64,
    timestamp_start_unix_ms: u64,
    prompt: String,
    input_tokens: u64,
    output_tokens: u64,
    duration_e2e_ms: Option<f64>,
    duration_ttft_ms: Option<f64>,
    cached_tokens_reference: Option<u64>,
    total_hashes: Vec<i64>,
    block_size: Option<u64>,
    session_id: Option<String>,
}

/// One row's decode outcome, retained so a refusal still names its position.
struct RowOutcome {
    row_in_group: u32,
    result: Result<BasetenRowProjection, DecodeFailureCode>,
}

fn downcast<'a, T: 'static>(
    array: &'a dyn arrow::array::Array,
) -> Result<&'a T, DecodeFailureCode> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or(DecodeFailureCode::Schema)
}

fn unsigned_value(
    array: &dyn arrow::array::Array,
    row: usize,
) -> Result<Option<u64>, DecodeFailureCode> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Int32 => i64::from(downcast::<Int32Array>(array)?.value(row)),
        DataType::Int64 => downcast::<Int64Array>(array)?.value(row),
        DataType::UInt32 => return Ok(Some(u64::from(downcast::<UInt32Array>(array)?.value(row)))),
        DataType::UInt64 => return Ok(Some(downcast::<UInt64Array>(array)?.value(row))),
        _ => return Err(DecodeFailureCode::Schema),
    };
    u64::try_from(value)
        .map(Some)
        .map_err(|_| DecodeFailureCode::Syntax)
}

fn float_value(
    array: &dyn arrow::array::Array,
    row: usize,
) -> Result<Option<f64>, DecodeFailureCode> {
    use arrow::array::{Float32Array, Float64Array};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Float32 => f64::from(downcast::<Float32Array>(array)?.value(row)),
        DataType::Float64 => downcast::<Float64Array>(array)?.value(row),
        // Integer-typed duration columns are accepted and widened, matching the
        // finite loader's tolerance for producer-side type drift.
        _ => return Ok(unsigned_value(array, row)?.map(|value| value as f64)),
    };
    if !value.is_finite() {
        return Err(DecodeFailureCode::Syntax);
    }
    Ok(Some(value))
}

fn string_value(
    array: &dyn arrow::array::Array,
    row: usize,
) -> Result<Option<String>, DecodeFailureCode> {
    use arrow::array::{LargeStringArray, StringArray, StringViewArray};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => Ok(Some(downcast::<StringArray>(array)?.value(row).to_owned())),
        DataType::LargeUtf8 => Ok(Some(
            downcast::<LargeStringArray>(array)?.value(row).to_owned(),
        )),
        DataType::Utf8View => Ok(Some(
            downcast::<StringViewArray>(array)?.value(row).to_owned(),
        )),
        DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64 => {
            Ok(unsigned_value(array, row)?.map(|value| value.to_string()))
        }
        _ => Err(DecodeFailureCode::Schema),
    }
}

fn hash_values(array: &dyn arrow::array::Array, row: usize) -> Result<Vec<i64>, DecodeFailureCode> {
    use arrow::array::{Int32Array, Int64Array, LargeListArray, ListArray};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(Vec::new());
    }
    let values = match array.data_type() {
        DataType::List(_) => downcast::<ListArray>(array)?.value(row),
        DataType::LargeList(_) => downcast::<LargeListArray>(array)?.value(row),
        _ => return Err(DecodeFailureCode::Schema),
    };
    match values.data_type() {
        DataType::Int32 => Ok(downcast::<Int32Array>(values.as_ref())?
            .iter()
            .flatten()
            .map(i64::from)
            .collect()),
        DataType::Int64 => Ok(downcast::<Int64Array>(values.as_ref())?
            .iter()
            .flatten()
            .collect()),
        _ => Err(DecodeFailureCode::Schema),
    }
}

fn optional_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Option<&'a dyn arrow::array::Array> {
    batch
        .schema()
        .index_of(name)
        .ok()
        .map(|index| batch.column(index).as_ref())
}

fn required_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<&'a dyn arrow::array::Array, DecodeFailureCode> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| DecodeFailureCode::Schema)?;
    Ok(batch.column(index).as_ref())
}

/// Project one Arrow batch row into an owned Baseten row.
fn project_row(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
    absolute_row: u64,
    row_in_group: u32,
    session_column: Option<&str>,
    max_prompt_bytes: usize,
) -> Result<BasetenRowProjection, DecodeFailureCode> {
    let timestamp_start_unix_ms =
        unsigned_value(required_column(batch, COL_TIME)?, row)?.ok_or(DecodeFailureCode::Schema)?;
    let prompt =
        string_value(required_column(batch, COL_PROMPT)?, row)?.ok_or(DecodeFailureCode::Schema)?;
    if prompt.len() > max_prompt_bytes {
        return Err(DecodeFailureCode::OversizedRecord);
    }
    let input_tokens = unsigned_value(required_column(batch, COL_INPUT_TOKENS)?, row)?
        .ok_or(DecodeFailureCode::Schema)?;
    let output_tokens = unsigned_value(required_column(batch, COL_OUTPUT_TOKENS)?, row)?
        .ok_or(DecodeFailureCode::Schema)?;

    let duration_e2e_ms = match optional_column(batch, COL_DURATION_E2E) {
        Some(array) => float_value(array, row)?,
        None => None,
    };
    let duration_ttft_ms = match optional_column(batch, COL_DURATION_TTFT) {
        Some(array) => float_value(array, row)?,
        None => None,
    };
    let cached_tokens_reference = match optional_column(batch, COL_CACHED_TOKENS) {
        Some(array) => unsigned_value(array, row)?,
        None => None,
    };
    let total_hashes = match optional_column(batch, COL_TOTAL_HASHES) {
        Some(array) => hash_values(array, row)?,
        None => Vec::new(),
    };
    let block_size = match optional_column(batch, COL_BLOCK_SIZE) {
        Some(array) => unsigned_value(array, row)?,
        None => None,
    };
    let session_id = match session_column.and_then(|name| optional_column(batch, name)) {
        Some(array) => string_value(array, row)?,
        None => None,
    };

    Ok(BasetenRowProjection {
        row_in_group,
        absolute_row,
        timestamp_start_unix_ms,
        prompt,
        input_tokens,
        output_tokens,
        duration_e2e_ms,
        duration_ttft_ms,
        cached_tokens_reference,
        total_hashes,
        block_size,
        session_id,
    })
}

/// Apply the recorded output length floor and the authored cap.
fn resolved_output_length(output_tokens: u64, cap: Option<u32>) -> u32 {
    // A canceled recorded request stores `output_tokens = 0`, but a replayed
    // request needs `max_tokens >= 1`.
    let floored = u32::try_from(output_tokens).unwrap_or(u32::MAX).max(1);
    match cap {
        Some(cap) => floored.min(cap),
        None => floored,
    }
}

/// Canonical replay-parameter document for one row.
///
/// Keys are emitted in sorted order and an absent recorded field stays absent,
/// so the payload is byte-stable across runs and hosts.
fn canonical_replay_parameters(
    row: &BasetenRowProjection,
    config: &BasetenFormatConfig,
) -> Vec<u8> {
    use std::fmt::Write as _;

    let output_length = resolved_output_length(row.output_tokens, config.max_output_tokens);
    let mut json = String::with_capacity(128 + row.total_hashes.len() * 12);
    json.push('{');
    if !config.omit_kv_hints
        && let Some(block_size) = row.block_size
    {
        let _ = write!(json, "\"block_size\":{block_size},");
    }
    if !config.omit_kv_hints && !row.total_hashes.is_empty() {
        json.push_str("\"hash_ids\":[");
        for (index, hash) in row.total_hashes.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            let _ = write!(json, "{hash}");
        }
        json.push_str("],");
    }
    let _ = write!(json, "\"input_tokens\":{},", row.input_tokens);
    let _ = write!(json, "\"max_tokens\":{output_length},");
    if config.force_min_tokens {
        let _ = write!(json, "\"min_tokens\":{output_length},");
    }
    let has_recorded = row.cached_tokens_reference.is_some()
        || row.duration_e2e_ms.is_some()
        || row.duration_ttft_ms.is_some();
    if has_recorded {
        json.push_str("\"recorded\":{");
        let mut is_first = true;
        if let Some(cached) = row.cached_tokens_reference {
            let _ = write!(json, "\"cached_tokens_reference\":{cached}");
            is_first = false;
        }
        if let Some(e2e) = row.duration_e2e_ms {
            if !is_first {
                json.push(',');
            }
            let _ = write!(json, "\"duration_e2e_ms\":{e2e}");
            is_first = false;
        }
        if let Some(ttft) = row.duration_ttft_ms {
            if !is_first {
                json.push(',');
            }
            let _ = write!(json, "\"duration_ttft_ms\":{ttft}");
        }
        json.push_str("},");
    }
    let _ = write!(
        json,
        "\"recorded_start_unix_ms\":{},",
        row.timestamp_start_unix_ms
    );
    let _ = write!(json, "\"schema\":\"{BASETEN_REPLAY_TURN_EVENT_KIND}\"");
    json.push('}');
    json.into_bytes()
}

// ---------------------------------------------------------------------------
// Format
// ---------------------------------------------------------------------------

/// Run-scoped Baseten format owner.
struct BasetenFormat {
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    reporter: StreamingIssueReporterHandle,
    fragment_budget: StreamingResourceBudget,
    acquisition_budget: AcquisitionBudget,
    config: Rc<ValidatedBasetenConfig>,
    /// Frozen at the first prepared partition; drift after that is terminal.
    frozen_schema_digest: Rc<RefCell<Option<ContentDigest>>>,
    /// Shared with every decoder so exhaustion is observable at seal time.
    partitions_sealed: Rc<Cell<u64>>,
    latest_event_time: Rc<Cell<Option<EventTimeUtc>>>,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
}

impl BasetenFormat {
    /// Read the footer once and resolve the frozen per-partition read plan.
    async fn prepare_read_plan(
        &self,
        snapshot: &AcquiredSeekableLocalPartition,
        object_len: u64,
    ) -> Result<BasetenReadPlan, StreamFormatError> {
        let footer_offset = object_len
            .checked_sub(PARQUET_FOOTER_BYTES as u64)
            .ok_or_else(schema_error)?;
        let tail = self
            .read_exact(snapshot, footer_offset, PARQUET_FOOTER_BYTES)
            .await?;
        let tail: [u8; PARQUET_FOOTER_BYTES] =
            tail.as_slice().try_into().map_err(|_| schema_error())?;
        let footer = parquet::file::metadata::FooterTail::try_new(&tail)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;
        let metadata_length = footer.metadata_length();
        // The metadata extent is known before a byte of it is allocated.
        if metadata_length > self.config.config.max_row_group_bytes {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::OversizedRecord,
            ));
        }
        let metadata_offset = footer_offset
            .checked_sub(metadata_length as u64)
            .ok_or_else(schema_error)?;
        let metadata_bytes = self
            .read_exact(snapshot, metadata_offset, metadata_length)
            .await?;
        let metadata =
            parquet::file::metadata::ParquetMetaDataReader::decode_metadata(&metadata_bytes)
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;
        let metadata = Arc::new(metadata);

        let arrow_metadata = parquet::arrow::arrow_reader::ArrowReaderMetadata::try_new(
            Arc::clone(&metadata),
            parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
        )
        .map_err(|_| schema_error())?;
        let schema = arrow_metadata.schema().clone();

        let mut indices = Vec::with_capacity(REQUIRED_COLUMNS.len() + 6);
        for column in REQUIRED_COLUMNS {
            indices.push(schema.index_of(column).map_err(|_| schema_error())?);
        }
        let mut push_optional = |name: &str| {
            if let Ok(index) = schema.index_of(name) {
                indices.push(index);
            }
        };
        push_optional(COL_DURATION_E2E);
        push_optional(COL_DURATION_TTFT);
        push_optional(COL_CACHED_TOKENS);
        if !self.config.config.omit_kv_hints {
            push_optional(COL_TOTAL_HASHES);
            push_optional(COL_BLOCK_SIZE);
        }
        let session_column = resolve_session_column(
            self.config.config.session_column,
            schema.index_of(COL_SESSION).is_ok(),
            schema.index_of(COL_POOR_MAN_SESSION).is_ok(),
        );
        if let Some(column) = session_column {
            indices.push(schema.index_of(column).map_err(|_| schema_error())?);
        }
        indices.sort_unstable();
        indices.dedup();

        let projection = parquet::arrow::ProjectionMask::roots(
            metadata.file_metadata().schema_descr(),
            indices.iter().copied(),
        );

        let mut hasher = blake3::Hasher::new();
        update_digest_field(&mut hasher, b"aiperf.baseten.parquet.schema.v1");
        for index in &indices {
            let field = schema.field(*index);
            update_digest_field(&mut hasher, field.name().as_bytes());
            update_digest_field(&mut hasher, field.data_type().to_string().as_bytes());
            update_digest_field(&mut hasher, &[u8::from(field.is_nullable())]);
        }
        update_digest_field(&mut hasher, session_column.unwrap_or_default().as_bytes());
        update_digest_field(&mut hasher, &self.config.canonical_config_bytes);
        let schema_digest = ContentDigest::from_bytes(*hasher.finalize().as_bytes());

        let mut row_groups = Vec::with_capacity(metadata.num_row_groups());
        let mut first_row = 0_u64;
        for index in 0..metadata.num_row_groups() {
            let group = metadata.row_group(index);
            let columns = group.columns();
            let Some(first) = columns.first() else {
                return Err(schema_error());
            };
            let mut start = first.byte_range().0;
            let mut end = start;
            for column in columns {
                let (chunk_start, chunk_length) = column.byte_range();
                start = start.min(chunk_start);
                end = end.max(chunk_start.saturating_add(chunk_length));
            }
            let length = usize::try_from(end.saturating_sub(start))
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::OversizedRecord))?;
            let num_rows = u64::try_from(group.num_rows()).map_err(|_| schema_error())?;
            row_groups.push(RowGroupPlan {
                index,
                start,
                length,
                first_row,
                num_rows,
            });
            first_row = first_row.checked_add(num_rows).ok_or_else(schema_error)?;
        }

        Ok(BasetenReadPlan {
            metadata,
            projection,
            session_column,
            row_groups,
            schema_digest,
            object_len,
        })
    }

    /// Read exactly `length` bytes, looping over bounded snapshot reads.
    async fn read_exact(
        &self,
        snapshot: &AcquiredSeekableLocalPartition,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, StreamFormatError> {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(length)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::OversizedRecord))?;
        let mut cursor = offset;
        while buffer.len() < length {
            let remaining = NonZeroUsize::new(length - buffer.len()).ok_or_else(schema_error)?;
            let chunk = snapshot
                .read_at(cursor, remaining, &self.acquisition_budget)
                .await
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;
            let bytes = chunk.as_bytes();
            if bytes.is_empty() {
                return Err(StreamFormatError::decode(DecodeFailureCode::Syntax));
            }
            cursor = cursor
                .checked_add(bytes.len() as u64)
                .ok_or_else(schema_error)?;
            buffer.extend_from_slice(bytes);
        }
        Ok(buffer)
    }

    /// Freeze the schema digest on the first partition and refuse later drift.
    ///
    /// The refusal is returned rather than reported: the reliability contract
    /// pairs partition scope with a *source*-owned failure
    /// (`failure_matches_scope`), so a format-owned schema drift has no valid
    /// partition-scoped receipt. The host classifier sees the typed error from
    /// `begin_partition`, before any fragment of the drifted partition exists.
    fn freeze_or_verify_schema_digest(
        &self,
        digest: ContentDigest,
    ) -> Result<(), StreamFormatError> {
        let frozen = *self.frozen_schema_digest.borrow();
        match frozen {
            None => {
                *self.frozen_schema_digest.borrow_mut() = Some(digest);
                Ok(())
            }
            Some(frozen) if frozen == digest => Ok(()),
            Some(_) => Err(schema_error()),
        }
    }

    fn completeness_digest(&self, through: EventTimeUtc) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_digest_field(&mut hasher, b"aiperf.baseten.completeness.v1");
        update_digest_field(&mut hasher, BASETEN_SEMANTIC_DIGEST.as_bytes());
        update_digest_field(&mut hasher, &through.get().to_le_bytes());
        update_digest_field(&mut hasher, &self.partitions_sealed.get().to_le_bytes());
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    fn seal_digest(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_digest_field(&mut hasher, b"aiperf.baseten.seal.v1");
        update_digest_field(&mut hasher, BASETEN_SEMANTIC_DIGEST.as_bytes());
        let frozen = self
            .frozen_schema_digest
            .borrow()
            .unwrap_or(BASETEN_SEMANTIC_DIGEST);
        update_digest_field(&mut hasher, frozen.as_bytes());
        update_digest_field(&mut hasher, &self.partitions_sealed.get().to_le_bytes());
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Canonical checkpoint payload: frozen digest, sealed count, latest time.
    fn encode_state(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(49);
        match *self.frozen_schema_digest.borrow() {
            Some(digest) => {
                bytes.push(1);
                bytes.extend_from_slice(digest.as_bytes());
            }
            None => {
                bytes.push(0);
                bytes.extend_from_slice(&[0_u8; 32]);
            }
        }
        bytes.extend_from_slice(&self.partitions_sealed.get().to_le_bytes());
        let event_time = self.latest_event_time.get().map_or(-1, EventTimeUtc::get);
        bytes.extend_from_slice(&event_time.to_le_bytes());
        bytes
    }

    fn restore_state(&mut self, bytes: &[u8]) -> Result<(), CheckpointError> {
        if bytes.len() != 49 || bytes[0] > 1 {
            return Err(CheckpointError::ObjectVerification);
        }
        if bytes[0] == 1 {
            let mut digest = [0_u8; 32];
            digest.copy_from_slice(&bytes[1..33]);
            *self.frozen_schema_digest.borrow_mut() = Some(ContentDigest::from_bytes(digest));
        }
        let sealed: [u8; 8] = bytes[33..41]
            .try_into()
            .map_err(|_| CheckpointError::ObjectVerification)?;
        self.partitions_sealed.set(u64::from_le_bytes(sealed));
        let event_time: [u8; 8] = bytes[41..49]
            .try_into()
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let event_time = i64::from_le_bytes(event_time);
        self.latest_event_time
            .set(EventTimeUtc::new(event_time).ok());
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingDatasetFormat for BasetenFormat {
    async fn begin_partition(
        &mut self,
        partition: crate::streaming::source::AcquiredPartition,
        resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError> {
        let identity = *partition.identity();
        let position = partition.position();
        let object_len = partition.size_bytes().ok_or_else(schema_error)?;
        let AcquiredPartitionAccess::SeekableLocal(snapshot) = partition.into_access() else {
            return Err(schema_error());
        };

        let plan = self.prepare_read_plan(&snapshot, object_len).await?;
        self.freeze_or_verify_schema_digest(plan.schema_digest)?;

        let cursor = match resume {
            Some(checkpoint) => {
                if checkpoint.partition != identity
                    || checkpoint.format_semantic_digest != BASETEN_SEMANTIC_DIGEST
                {
                    return Err(StreamFormatError::decode(DecodeFailureCode::InvalidCursor));
                }
                let cursor = BasetenCursor::decode(checkpoint.state.as_bytes())?;
                if cursor.schema_digest != plan.schema_digest
                    || cursor.row_group as usize > plan.row_groups.len()
                {
                    return Err(StreamFormatError::decode(DecodeFailureCode::InvalidCursor));
                }
                cursor
            }
            None => BasetenCursor {
                row_group: 0,
                row_in_group: 0,
                rows_emitted: 0,
                schema_digest: plan.schema_digest,
            },
        };

        Ok(Box::new(BasetenPartitionDecoder {
            identity,
            position,
            snapshot,
            plan: Rc::new(plan),
            cursor,
            staged: VecDeque::new(),
            staged_group: 0,
            deferred: None,
            fragment_budget: self.fragment_budget.clone(),
            acquisition_budget: self.acquisition_budget.clone(),
            config: Rc::clone(&self.config),
            input_domain: StreamingInputDomainIdentity::new(self.stream_identity, identity),
            run: self.run,
            reporter: self.reporter.clone(),
            partitions_sealed: Rc::clone(&self.partitions_sealed),
            latest_event_time: Rc::clone(&self.latest_event_time),
            fragments_emitted: 0,
            is_exhausted: false,
        }))
    }

    async fn advance_source_frontier(
        &mut self,
        _frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError> {
        // A source frontier proves partition discovery, not event-time
        // completeness, so this asserts only the greatest event time this
        // decoder has actually emitted.
        let Some(through) = self.latest_event_time.get() else {
            return Ok(());
        };
        output
            .send(FormatEvent::SessionFrontier(SessionWatermark {
                through,
                digest: self.completeness_digest(through),
            }))
            .await
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError> {
        if let Some(through) = self.latest_event_time.get() {
            output
                .send(FormatEvent::SessionFrontier(SessionWatermark {
                    through,
                    digest: self.completeness_digest(through),
                }))
                .await?;
        }
        Ok(FormatSealReceipt {
            digest: self.seal_digest(),
            partition_count: self.partitions_sealed.get(),
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for BasetenFormat {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let payload = self.encode_state();
        let lease = self
            .fragment_budget
            .try_acquire(1, payload.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(payload), lease)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            BASETEN_FORMAT_SCHEMA_ID,
            BASETEN_FORMAT_SCHEMA_VERSION,
            barrier.cut.clone(),
            self.partitions_sealed.get(),
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else { return Ok(()) };
        let bytes = state.payload_bytes().to_vec();
        self.restore_state(&bytes)
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Partition decoder
// ---------------------------------------------------------------------------

/// Bounded decoder for one immutable Parquet object.
struct BasetenPartitionDecoder {
    identity: ImmutableObjectIdentity,
    position: SourcePosition,
    snapshot: AcquiredSeekableLocalPartition,
    plan: Rc<BasetenReadPlan>,
    cursor: BasetenCursor,
    /// Rows decoded from the staged row group and not yet emitted.
    staged: VecDeque<BasetenRowProjection>,
    /// Row group the staged rows came from.
    staged_group: u32,
    /// Second fragment of a pair split across two caller-bounded batches.
    deferred: Option<StreamingSessionFragment>,
    fragment_budget: StreamingResourceBudget,
    acquisition_budget: AcquisitionBudget,
    config: Rc<ValidatedBasetenConfig>,
    input_domain: StreamingInputDomainIdentity,
    run: StreamRunIdentity,
    reporter: StreamingIssueReporterHandle,
    partitions_sealed: Rc<Cell<u64>>,
    latest_event_time: Rc<Cell<Option<EventTimeUtc>>>,
    fragments_emitted: u64,
    is_exhausted: bool,
}

impl BasetenPartitionDecoder {
    fn record_id_for(&self, row_group: u32, row_in_group: u32, tag: u8) -> StableRecordId {
        let mut coordinate = [0_u8; 17];
        coordinate[0..8].copy_from_slice(&self.position.get().to_le_bytes());
        coordinate[8..12].copy_from_slice(&row_group.to_le_bytes());
        coordinate[12..16].copy_from_slice(&row_in_group.to_le_bytes());
        coordinate[16] = tag;
        physical_record_id(
            self.input_domain.stream_identity().as_bytes(),
            &self.identity,
            &coordinate,
            BASETEN_SEMANTIC_DIGEST.as_bytes(),
        )
    }

    fn session_key_for(
        &self,
        session_id: Option<&str>,
        record_id: StableRecordId,
    ) -> StableSessionKey {
        match session_id {
            Some(id) => stable_session_key(
                self.input_domain.stream_identity().as_bytes(),
                id.as_bytes(),
            ),
            // Rows without a recorded session id never join across partitions.
            None => one_turn_session_key(record_id),
        }
    }

    /// Whole-session stable-hash admission, identical in every partition.
    fn is_session_admitted(&self, session_key: StableSessionKey) -> bool {
        let Some(threshold) = self.config.sample_threshold else {
            return true;
        };
        let mut hasher = blake3::Hasher::new();
        update_digest_field(&mut hasher, b"aiperf.baseten.session-sample.v1");
        update_digest_field(&mut hasher, session_key.as_bytes());
        let digest = hasher.finalize();
        let mut head = [0_u8; 8];
        head.copy_from_slice(&digest.as_bytes()[0..8]);
        u64::from_le_bytes(head) < threshold
    }

    /// Exclude one invalid row and continue with the next.
    async fn quarantine_row(&self, row_in_group: u32, code: DecodeFailureCode) {
        let record_id = self.record_id_for(self.cursor.row_group, row_in_group, 0);
        let issue = OrdinaryStreamingIssue::record(
            self.run,
            self.input_domain.clone(),
            record_id,
            StreamingIssueClass::Permanent,
            self.plan.schema_digest,
            self.position,
            0,
            self.plan.schema_digest,
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(code)),
        );
        if let Ok(issue) = issue {
            let _ = self.reporter.report(issue).await;
        }
    }

    fn next_row_group(&self) -> Option<RowGroupPlan> {
        self.plan
            .row_groups
            .get(self.cursor.row_group as usize)
            .copied()
    }

    /// Stage one row group, or report that the partition is exhausted.
    ///
    /// The cursor is not advanced past the staged group until every one of its
    /// staged rows has been emitted, so a checkpoint taken mid-group resumes at
    /// the first unemitted row rather than skipping the remainder.
    async fn stage_next_row_group(&mut self) -> Result<bool, StreamFormatError> {
        let Some(group) = self.next_row_group() else {
            return Ok(false);
        };
        // Refusal precedes allocation: the column-chunk extent is metadata.
        if group.length > self.config.config.max_row_group_bytes {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::OversizedRecord,
            ));
        }

        let skip = self.cursor.row_in_group;
        let outcomes = if u64::from(skip) >= group.num_rows {
            Vec::new()
        } else {
            let window = self.read_window(group).await?;
            decode_row_group(
                window,
                Arc::clone(&self.plan.metadata),
                self.plan.projection.clone(),
                group,
                skip,
                self.plan.session_column,
                self.config.config.max_prompt_bytes,
            )?
        };

        self.staged_group = self.cursor.row_group;
        for outcome in outcomes {
            match outcome.result {
                Ok(row) => {
                    if self.is_session_admitted(self.staged_session_key(&row)) {
                        self.staged.push_back(row);
                    }
                }
                Err(code) => self.quarantine_row(outcome.row_in_group, code).await,
            }
        }
        if self.staged.is_empty() {
            self.advance_past_staged_group();
        }
        Ok(true)
    }

    /// Move the cursor to the first row of the next row group.
    fn advance_past_staged_group(&mut self) {
        self.cursor.row_group = self.cursor.row_group.saturating_add(1);
        self.cursor.row_in_group = 0;
    }

    /// Session key a projected row will carry, used for whole-session sampling.
    fn staged_session_key(&self, row: &BasetenRowProjection) -> StableSessionKey {
        let turn_id = self.record_id_for(self.staged_group, row.row_in_group, 0);
        self.session_key_for(row.session_id.as_deref(), turn_id)
    }

    /// Read one row group's contiguous column-chunk extent into an owned window.
    ///
    /// The acquisition lease is released on this thread; the retained window
    /// owns compact `Bytes` sized exactly to the refused-or-admitted extent.
    async fn read_window(&self, group: RowGroupPlan) -> Result<RetainedWindow, StreamFormatError> {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(group.length)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::OversizedRecord))?;
        let mut cursor = group.start;
        while buffer.len() < group.length {
            let remaining =
                NonZeroUsize::new(group.length - buffer.len()).ok_or_else(schema_error)?;
            let chunk = self
                .snapshot
                .read_at(cursor, remaining, &self.acquisition_budget)
                .await
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;
            let bytes = chunk.as_bytes();
            if bytes.is_empty() {
                return Err(StreamFormatError::decode(DecodeFailureCode::Syntax));
            }
            cursor = cursor
                .checked_add(bytes.len() as u64)
                .ok_or_else(schema_error)?;
            buffer.extend_from_slice(bytes);
        }
        Ok(RetainedWindow {
            object_len: self.plan.object_len,
            window_start: group.start,
            bytes: Bytes::from(buffer),
        })
    }

    /// Mint the causal fragment pair for one projected row.
    async fn fragments_for_row(
        &self,
        row: &BasetenRowProjection,
        payload: Vec<u8>,
    ) -> Result<(StreamingSessionFragment, StreamingSessionFragment), StreamFormatError> {
        let turn_id = self.record_id_for(self.staged_group, row.row_in_group, 0);
        let parameters_id = self.record_id_for(self.staged_group, row.row_in_group, 1);
        let session_key = self.session_key_for(row.session_id.as_deref(), turn_id);
        let event_time = i64::try_from(row.timestamp_start_unix_ms)
            .ok()
            .and_then(|ms| ms.checked_mul(NANOS_PER_MILLI))
            .and_then(|ns| EventTimeUtc::new(ns).ok())
            .ok_or_else(schema_error)?;
        let provenance = || UnitProvenance {
            source_partition: self.identity,
            source_position: self.position,
            format_semantic_digest: BASETEN_SEMANTIC_DIGEST,
        };

        let content = row.prompt.clone().into_bytes();
        let turn_lease = self.fragment_lease(content.len()).await?;
        let turn = StreamingSessionFragment {
            record_id: turn_id,
            session_key,
            source_position: self.position,
            source_partition: self.identity,
            event_time: Some(event_time),
            stable_tie_break: StableOrderKey::from_bytes(*turn_id.as_bytes()),
            predecessors: SmallVec::new(),
            mutation: SessionMutationV1::ConversationTurn(ConversationTurnFragment {
                role: "user".to_owned(),
                content,
                turn_ordinal: row.absolute_row,
            }),
            provenance: provenance(),
            lease: turn_lease,
        };

        let parameters_lease = self.fragment_lease(payload.len()).await?;
        let parameters = StreamingSessionFragment {
            record_id: parameters_id,
            session_key,
            source_position: self.position,
            source_partition: self.identity,
            event_time: Some(event_time),
            stable_tie_break: StableOrderKey::from_bytes(*parameters_id.as_bytes()),
            predecessors: smallvec::smallvec![turn_id],
            mutation: SessionMutationV1::AgentEvent(AgentEventFragment {
                event_kind: BASETEN_REPLAY_TURN_EVENT_KIND.to_owned(),
                payload,
                event_ordinal: row.absolute_row,
            }),
            provenance: provenance(),
            lease: parameters_lease,
        };

        let latest = self
            .latest_event_time
            .get()
            .map_or(event_time, |current| current.max(event_time));
        self.latest_event_time.set(Some(latest));
        Ok((turn, parameters))
    }

    async fn fragment_lease(
        &self,
        bytes: usize,
    ) -> Result<SessionFragmentLease, StreamFormatError> {
        let lease = self
            .fragment_budget
            .acquire(1, bytes)
            .await
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        SessionFragmentLease::try_from(lease)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))
    }

    async fn resume_cursor(&self) -> Result<DecoderResumeState, StreamFormatError> {
        let lease = self
            .fragment_budget
            .acquire(1, CURSOR_BYTES)
            .await
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        DecoderResumeState::new(Bytes::copy_from_slice(&self.cursor.encode()), lease)
    }
}

#[async_trait(?Send)]
impl StreamingPartitionDecoder for BasetenPartitionDecoder {
    async fn next_batch(
        &mut self,
        budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError> {
        if budget.max_fragments == 0 {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::BudgetInvariant,
            ));
        }

        let mut fragments: Vec<StreamingSessionFragment> = Vec::new();
        let mut retained_bytes = 0_usize;
        if let Some(deferred) = self.deferred.take() {
            retained_bytes = deferred.lease.charged_bytes();
            fragments.push(deferred);
        }

        loop {
            if fragments.len() >= budget.max_fragments {
                break;
            }
            let Some(row) = self.staged.front() else {
                if self.stage_next_row_group().await? {
                    continue;
                }
                break;
            };
            let payload = canonical_replay_parameters(row, &self.config.config);
            let pair_bytes = row.prompt.len().saturating_add(payload.len());
            if !fragments.is_empty()
                && (fragments.len().saturating_add(2) > budget.max_fragments
                    || retained_bytes.saturating_add(pair_bytes) > budget.max_bytes)
            {
                break;
            }
            let Some(row) = self.staged.pop_front() else {
                break;
            };
            let (turn, parameters) = self.fragments_for_row(&row, payload).await?;
            retained_bytes = retained_bytes.saturating_add(pair_bytes);
            fragments.push(turn);
            if fragments.len() < budget.max_fragments {
                fragments.push(parameters);
            } else {
                // The pair is causally ordered, so leading the next batch with
                // the parameter fragment is safe.
                self.deferred = Some(parameters);
            }
            self.cursor.row_in_group = row.row_in_group.saturating_add(1);
            self.cursor.rows_emitted = self.cursor.rows_emitted.saturating_add(1);
            if self.staged.is_empty() {
                self.advance_past_staged_group();
            }
        }

        if fragments.is_empty() {
            // Acquiring the cursor charge is where the decoder parks while a
            // previously issued batch still holds the output budget.
            let final_state = self.resume_cursor().await?;
            if !self.is_exhausted {
                self.is_exhausted = true;
                self.partitions_sealed
                    .set(self.partitions_sealed.get().saturating_add(1));
            }
            return Ok(DecodeStep::End(DecodeReceipt {
                partition: self.identity,
                fragment_count: self.fragments_emitted,
                final_state,
            }));
        }
        self.fragments_emitted = self
            .fragments_emitted
            .saturating_add(fragments.len() as u64);
        let resume_after = self.resume_cursor().await?;
        Ok(DecodeStep::Batch(DecodedFragmentBatch {
            fragments,
            resume_after,
        }))
    }

    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError> {
        let encoded = self.cursor.encode();
        let lease = self
            .fragment_budget
            .try_acquire(1, encoded.len())
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        DecoderResumeState::new(Bytes::copy_from_slice(&encoded), lease)
    }
}

/// Decode one row group's projected columns into owned rows.
///
/// The window, metadata, and projection are all owned, so the decode holds no
/// borrow of the acquired snapshot.
#[allow(clippy::too_many_arguments)]
fn decode_row_group(
    window: RetainedWindow,
    metadata: Arc<parquet::file::metadata::ParquetMetaData>,
    projection: parquet::arrow::ProjectionMask,
    group: RowGroupPlan,
    skip: u32,
    session_column: Option<&'static str>,
    max_prompt_bytes: usize,
) -> Result<Vec<RowOutcome>, StreamFormatError> {
    use parquet::arrow::arrow_reader::{
        ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
    };

    let reader_metadata = ArrowReaderMetadata::try_new(metadata, ArrowReaderOptions::new())
        .map_err(|_| schema_error())?;
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(window, reader_metadata)
        .with_row_groups(vec![group.index])
        .with_projection(projection)
        .with_offset(skip as usize)
        .with_batch_size(ARROW_BATCH_ROWS)
        .build()
        .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;

    let mut outcomes = Vec::new();
    let mut row_in_group = u64::from(skip);
    for batch in reader {
        let batch = batch.map_err(|_| StreamFormatError::decode(DecodeFailureCode::Syntax))?;
        for row in 0..batch.num_rows() {
            let ordinal = u32::try_from(row_in_group).map_err(|_| schema_error())?;
            let absolute_row = group.first_row.saturating_add(row_in_group);
            outcomes.push(RowOutcome {
                row_in_group: ordinal,
                result: project_row(
                    &batch,
                    row,
                    absolute_row,
                    ordinal,
                    session_column,
                    max_prompt_bytes,
                ),
            });
            row_in_group = row_in_group.saturating_add(1);
        }
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests;
