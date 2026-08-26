// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten Parquet trace replay loader.
//!
//! Ports `src/aiperf/dataset/loader/baseten_trace.py` +
//! `_baseten_replay_timemodel.py`: literal-prompt Parquet traces with
//! session grouping by the strongest repeated-session-id column, open-loop
//! (absolute timestamp) or closed-loop (back-pressure delay) replay, an
//! idle-gap reflow so a sparse trace does not idle, and per-turn
//! `min_tokens`/`hash_ids`/`block_size` KV-cache routing hints injected into
//! the outgoing request body.
//!
//! Scope cut from the Python loader: Synthesizer integration is not ported. Trace
//! synthesis is already rejected for this loader before it ever reaches
//! composition -- `engine/execute.rs`'s `build_file_dataset` only allows
//! synthesis for `mooncake_trace`/`bailian_trace`/`burst_gpt`, so any
//! `--synthesis-*` config on `baseten_trace` errors at the engine layer
//! (matching Python's conservative rejection: prompt reshaping would desync
//! the forwarded `hash_ids` KV hints from the verbatim-replayed prompt).
//! `--trace-session-sample-ratio` whole-session subsampling is ported.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{Map, Value, json};
use smallvec::smallvec;

use crate::dataset::compose::{ComposeConfig, Composer};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin,
};
use crate::dataset::model::{
    ContentGroup, Conversation, ConversationContextMode, MediaKind, RecordedOutcome, SessionId,
    Turn,
};
use crate::dataset::segment::SegmentPool;
use crate::dataset::tokenizer::TextTokenizer;

fn cap_output(value: Option<u32>, cap: Option<u32>) -> Option<u32> {
    match (value, cap) {
        (Some(value), Some(cap)) => Some(value.min(cap)),
        (value, _) => value,
    }
}

/// Baseten Parquet trace loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasetenTraceDatasetLoader;
/// Baseten trace composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasetenTraceComposer;

const COL_TIME: &str = "timestamp_start_unix_ms";
const COL_SESSION: &str = "provided_session_id";
const COL_POOR_MAN_SESSION: &str = "poor_man_session_id";
const PARQUET_BATCH_SIZE: usize = 128;

fn required_columns() -> [&'static str; 4] {
    [COL_TIME, "prompt", "input_tokens", "output_tokens"]
}

/// One parsed Baseten trace row, mutated in place as replay timing is derived.
#[derive(Debug, Clone)]
struct BasetenRow {
    prompt: String,
    input_tokens: u64,
    output_tokens: u64,
    total_hashes: Vec<i64>,
    provided_session_id: Option<String>,
    poor_man_session_id: Option<String>,
    duration_e2e_ms: Option<f64>,
    duration_ttft_ms: Option<f64>,
    cached_tokens_reference: Option<u64>,
    block_size: Option<usize>,
    /// Normalized (min-subtracted, speedup-scaled) timestamp in ms. Cleared
    /// to `None` for closed-loop continuation turns once back-pressure
    /// converts the gap into `delay`.
    timestamp: Option<f64>,
    delay: Option<f64>,
    /// Recorded output length, floored to at least 1 (canceled requests
    /// record `output_tokens=0`, but a request needs `max_tokens >= 1`).
    output_length: u32,
}

fn stringify_session_value(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        other => Some(other.to_string()),
    }
}

fn parse_row(value: &Value, origin: &impl std::fmt::Display) -> Result<BasetenRow> {
    let object = value.as_object().ok_or_else(|| {
        DatasetError::Validation(format!("{origin}: Baseten trace row must be an object"))
    })?;
    let field = |name: &str| object.get(name);
    let timestamp_start_unix_ms = field(COL_TIME).and_then(Value::as_u64).ok_or_else(|| {
        DatasetError::Validation(format!("{origin}: missing or invalid {COL_TIME}"))
    })?;
    let prompt = field("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| DatasetError::Validation(format!("{origin}: missing or invalid prompt")))?
        .to_string();
    let input_tokens = field("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            DatasetError::Validation(format!("{origin}: missing or invalid input_tokens"))
        })?;
    let output_tokens = field("output_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            DatasetError::Validation(format!("{origin}: missing or invalid output_tokens"))
        })?;
    let total_hashes = field("total_hashes")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default();
    let duration_e2e_ms = field("duration_e2e_ms").and_then(Value::as_f64);
    let duration_ttft_ms = field("duration_ttft_ms").and_then(Value::as_f64);
    let cached_tokens_reference = field("cached_tokens_reference").and_then(Value::as_u64);
    let block_size = field("block_size")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());

    Ok(BasetenRow {
        prompt,
        input_tokens,
        output_tokens,
        total_hashes,
        provided_session_id: stringify_session_value(field(COL_SESSION)),
        poor_man_session_id: stringify_session_value(field(COL_POOR_MAN_SESSION)),
        duration_e2e_ms,
        duration_ttft_ms,
        cached_tokens_reference,
        block_size,
        timestamp: Some(timestamp_start_unix_ms as f64),
        delay: None,
        output_length: (output_tokens as u32).max(1),
    })
}

/// Which session-id column groups these rows into sessions most strongly:
/// the sum and count of repeated (session_id -> count>1) groups, matching
/// Python's `_score_session_groups`.
fn score_session_groups(ids: &[Option<String>]) -> (usize, usize) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for id in ids.iter().flatten() {
        *counts.entry(id.as_str()).or_insert(0) += 1;
    }
    let repeated: Vec<usize> = counts
        .values()
        .copied()
        .filter(|&count| count > 1)
        .collect();
    (repeated.iter().sum(), repeated.len())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionKey {
    Provided,
    PoorMan,
}

/// Port of Python's `choose_baseten_session_key`.
fn choose_session_key(rows: &[BasetenRow]) -> Option<SessionKey> {
    let provided: Vec<Option<String>> = rows
        .iter()
        .map(|row| row.provided_session_id.clone())
        .collect();
    let poor_man: Vec<Option<String>> = rows
        .iter()
        .map(|row| row.poor_man_session_id.clone())
        .collect();
    let provided_score = score_session_groups(&provided);
    let poor_score = score_session_groups(&poor_man);
    if provided_score > poor_score && provided_score.0 > 0 {
        return Some(SessionKey::Provided);
    }
    if poor_score > provided_score && poor_score.0 > 0 {
        return Some(SessionKey::PoorMan);
    }
    if provided_score == poor_score && provided_score.0 > 0 {
        return Some(SessionKey::Provided);
    }
    None
}

/// Port of Python baseten `_sample_sessions`: keep whole sessions (and null-session
/// rows) at `ratio`, always retaining at least one session when any existed.
fn sample_trace_sessions(
    rows: &mut Vec<BasetenRow>,
    session_key: Option<SessionKey>,
    ratio: f64,
    rng_root: crate::rng::RngRoot,
) -> Result<()> {
    use crate::rng::compat::python_random::PythonRandomGenerator;
    use crate::rng::derive::DerivedRandomGenerator;
    use crate::rng::random_generator::RandomGenerator;

    let Some(session_key) = session_key else {
        tracing::warn!(
            "trace_session_sample_ratio requested, but neither provided_session_id \
             nor poor_man_session_id forms multi-row sessions; skipping sampling"
        );
        return Ok(());
    };

    let mut session_first_ts: HashMap<String, f64> = HashMap::new();
    let mut null_row_count = 0_usize;
    for row in rows.iter() {
        let session_id = match session_key {
            SessionKey::Provided => row.provided_session_id.as_deref(),
            SessionKey::PoorMan => row.poor_man_session_id.as_deref(),
        };
        let Some(session_id) = session_id else {
            null_row_count += 1;
            continue;
        };
        let ts = row.timestamp.unwrap_or(0.0);
        session_first_ts
            .entry(session_id.to_string())
            .and_modify(|existing| *existing = existing.min(ts))
            .or_insert(ts);
    }

    let mut session_entries: Vec<(f64, String)> = session_first_ts
        .into_iter()
        .map(|(sid, ts)| (ts, sid))
        .collect();
    session_entries.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let original_count = session_entries.len();
    let mut rng = PythonRandomGenerator::from_rng_root(
        rng_root,
        crate::rng::namespace::DATASET_LOADER_BASETEN_TRACE_SESSION_SAMPLING,
    );
    let mut sampled_entries: Vec<(f64, String)> = session_entries
        .iter()
        .filter(|_| rng.uniform(0.0, 1.0) < ratio)
        .cloned()
        .collect();
    if sampled_entries.is_empty() && original_count > 0 {
        let chosen = rng
            .choice(&session_entries)
            .map_err(|error| DatasetError::Validation(error.to_string()))?
            .clone();
        sampled_entries.push(chosen);
    }
    let kept_sessions: std::collections::HashSet<String> =
        sampled_entries.into_iter().map(|(_, sid)| sid).collect();
    let sampled_null_rows: std::collections::HashSet<usize> = (0..null_row_count)
        .filter(|_| rng.uniform(0.0, 1.0) < ratio)
        .collect();

    tracing::info!(
        kept_sessions = kept_sessions.len(),
        original_sessions = original_count,
        kept_null_rows = sampled_null_rows.len(),
        null_row_count,
        ratio,
        ?session_key,
        "sampled baseten_trace sessions"
    );

    let mut null_ordinal = 0_usize;
    rows.retain(|row| {
        let session_id = match session_key {
            SessionKey::Provided => row.provided_session_id.as_deref(),
            SessionKey::PoorMan => row.poor_man_session_id.as_deref(),
        };
        match session_id {
            Some(session_id) => kept_sessions.contains(session_id),
            None => {
                let keep = sampled_null_rows.contains(&null_ordinal);
                null_ordinal += 1;
                keep
            }
        }
    });
    Ok(())
}

/// Port of `_baseten_replay_timemodel.reflow_idle_gaps`: collapse global idle
/// gaps larger than `cap_ms` so fixed-schedule replay of a sparse trace does
/// not idle through dead-air stretches. Ordering and relative spacing up to
/// the cap are preserved; the earliest event keeps its original value.
fn reflow_idle_gaps(timestamps_ms: &[f64], cap_ms: Option<f64>) -> Vec<f64> {
    let n = timestamps_ms.len();
    let values: Vec<i64> = timestamps_ms.iter().map(|&t| t as i64).collect();
    let Some(cap_ms) = cap_ms else {
        return values.iter().map(|&v| v as f64).collect();
    };
    if n <= 1 {
        return values.iter().map(|&v| v as f64).collect();
    }
    let cap = cap_ms.ceil() as i64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (values[i], i));
    let mut out = vec![0_i64; n];
    let first = order[0];
    out[first] = values[first];
    let mut prev_old = values[first];
    let mut prev_new = values[first];
    for &i in &order[1..] {
        let gap = values[i] - prev_old;
        prev_new += gap.min(cap);
        out[i] = prev_new;
        prev_old = values[i];
    }
    out.iter().map(|&v| v as f64).collect()
}

/// Replay-timing knobs threaded from the dataset `options` bag
/// (`--replay-speedup`, `--max-idle-gap-cap-seconds`, `--open-loop-replay`,
/// `--open-loop-strict`, `--omit-kv-hints`, `--force-min-tokens`,
/// `--inter-turn-delay-cap-seconds`, `--trace-session-sample-ratio`), matching
/// `--uuid-and-strip`'s wiring.
struct ReplayOptions {
    speedup: f64,
    max_idle_gap_cap_ms: Option<f64>,
    open_loop: bool,
    open_loop_strict: bool,
    omit_kv_hints: bool,
    force_min_tokens: bool,
    inter_turn_delay_cap_ms: Option<f64>,
    /// Whole-session keep fraction in `(0, 1]`; `None` keeps every session.
    session_sample_ratio: Option<f64>,
}

impl ReplayOptions {
    fn from_options(options: &Map<String, Value>) -> Result<Self> {
        let speedup = options
            .get("replay_speedup")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if !speedup.is_finite() || speedup <= 0.0 {
            return Err(DatasetError::Validation(
                "replay_speedup must be finite and positive".into(),
            ));
        }
        let max_idle_gap_cap_ms = options
            .get("max_idle_gap_cap_seconds")
            .and_then(Value::as_f64)
            .map(|seconds| seconds * 1000.0);
        if max_idle_gap_cap_ms.is_some_and(|cap| cap <= 0.0) {
            return Err(DatasetError::Validation(
                "max_idle_gap_cap_seconds must be positive".into(),
            ));
        }
        let open_loop = options
            .get("open_loop_replay")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let open_loop_strict = options
            .get("open_loop_strict")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if open_loop_strict && !open_loop {
            return Err(DatasetError::Validation(
                "open_loop_strict requires open_loop_replay".into(),
            ));
        }
        Ok(Self {
            speedup,
            max_idle_gap_cap_ms,
            open_loop,
            open_loop_strict,
            omit_kv_hints: options
                .get("omit_kv_hints")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            force_min_tokens: options
                .get("force_min_tokens")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            inter_turn_delay_cap_ms: options
                .get("inter_turn_delay_cap_seconds")
                .and_then(Value::as_f64)
                .map(|seconds| seconds * 1000.0),
            session_sample_ratio: {
                let ratio = options
                    .get("trace_session_sample_ratio")
                    .and_then(Value::as_f64);
                if let Some(ratio) = ratio {
                    if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
                        return Err(DatasetError::Validation(
                            "trace_session_sample_ratio must be in (0.0, 1.0]".into(),
                        ));
                    }
                    if ratio >= 1.0 { None } else { Some(ratio) }
                } else {
                    None
                }
            },
        })
    }

    fn clamp_delay(&self, delay_ms: f64) -> f64 {
        match self.inter_turn_delay_cap_ms {
            Some(cap) if delay_ms > cap => cap,
            _ => delay_ms,
        }
    }
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Copy)]
enum ColumnarKind {
    Parquet,
    ArrowIpc,
}

#[cfg(feature = "parquet")]
#[derive(Debug)]
struct ColumnarSource {
    path: PathBuf,
    file: std::fs::File,
    kind: ColumnarKind,
    schema: arrow::datatypes::SchemaRef,
}

#[cfg(feature = "parquet")]
impl ColumnarSource {
    fn open(path: &Path) -> Result<Self> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let kind = match path.extension().and_then(|suffix| suffix.to_str()) {
            Some("parquet") => ColumnarKind::Parquet,
            Some("arrow" | "ipc") => ColumnarKind::ArrowIpc,
            _ => {
                return Err(DatasetError::Validation(format!(
                    "unsupported Baseten columnar file {}",
                    path.display()
                )));
            }
        };
        let file = std::fs::File::open(path).map_err(|error| {
            DatasetError::Validation(format!("failed to open {}: {error}", path.display()))
        })?;
        let metadata_file = file.try_clone().map_err(|error| {
            DatasetError::Validation(format!("failed to inspect {}: {error}", path.display()))
        })?;
        let schema = match kind {
            ColumnarKind::Parquet => ParquetRecordBatchReaderBuilder::try_new(metadata_file)
                .map_err(|error| {
                    DatasetError::Validation(format!(
                        "failed to open {} as Parquet: {error}",
                        path.display()
                    ))
                })?
                .schema()
                .clone(),
            ColumnarKind::ArrowIpc => {
                arrow::ipc::reader::FileReader::try_new_buffered(metadata_file, None)
                    .map_err(|error| {
                        DatasetError::Validation(format!(
                            "failed to open {} as Arrow IPC: {error}",
                            path.display()
                        ))
                    })?
                    .schema()
            }
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
            kind,
            schema,
        })
    }

    fn has_columns(&self, columns: &[&str]) -> bool {
        columns
            .iter()
            .all(|column| self.schema.index_of(column).is_ok())
    }

    fn for_each_batch(
        &self,
        columns: &[&str],
        mut visit: impl FnMut(arrow::record_batch::RecordBatch) -> Result<()>,
    ) -> Result<()> {
        use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};

        let indices = columns
            .iter()
            .map(|column| {
                self.schema.index_of(column).map_err(|_| {
                    DatasetError::Validation(format!(
                        "{} is missing required column {column}",
                        self.path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let file = self.file.try_clone().map_err(|error| {
            DatasetError::Validation(format!("failed to read {}: {error}", self.path.display()))
        })?;
        match self.kind {
            ColumnarKind::Parquet => {
                let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
                    DatasetError::Validation(format!(
                        "failed to read Parquet metadata from {}: {error}",
                        self.path.display()
                    ))
                })?;
                let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
                let reader = builder
                    .with_batch_size(PARQUET_BATCH_SIZE)
                    .with_projection(projection)
                    .build()
                    .map_err(|error| {
                        DatasetError::Validation(format!(
                            "failed to build Parquet reader for {}: {error}",
                            self.path.display()
                        ))
                    })?;
                for batch in reader {
                    let batch = batch.map_err(|error| {
                        DatasetError::Validation(format!(
                            "failed to decode Parquet batch from {}: {error}",
                            self.path.display()
                        ))
                    })?;
                    visit_bounded_batches(batch, &mut visit)?;
                }
            }
            ColumnarKind::ArrowIpc => {
                let reader = arrow::ipc::reader::FileReader::try_new_buffered(file, Some(indices))
                    .map_err(|error| {
                        DatasetError::Validation(format!(
                            "failed to build Arrow IPC reader for {}: {error}",
                            self.path.display()
                        ))
                    })?;
                for batch in reader {
                    let batch = batch.map_err(|error| {
                        DatasetError::Validation(format!(
                            "failed to decode Arrow IPC batch from {}: {error}",
                            self.path.display()
                        ))
                    })?;
                    visit_bounded_batches(batch, &mut visit)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "parquet")]
fn visit_bounded_batches(
    batch: arrow::record_batch::RecordBatch,
    visit: &mut impl FnMut(arrow::record_batch::RecordBatch) -> Result<()>,
) -> Result<()> {
    for offset in (0..batch.num_rows()).step_by(PARQUET_BATCH_SIZE) {
        let batch = batch.slice(offset, (batch.num_rows() - offset).min(PARQUET_BATCH_SIZE));
        #[cfg(test)]
        record_columnar_scan(&batch);
        visit(batch)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "parquet"))]
thread_local! {
    static COLUMNAR_SCANS: std::cell::RefCell<Vec<(Vec<String>, usize)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

#[cfg(all(test, feature = "parquet"))]
fn record_columnar_scan(batch: &arrow::record_batch::RecordBatch) {
    COLUMNAR_SCANS.with(|scans| {
        scans.borrow_mut().push((
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect(),
            batch.num_rows(),
        ));
    });
}

#[cfg(all(test, feature = "parquet"))]
fn take_columnar_scans() -> Vec<(Vec<String>, usize)> {
    COLUMNAR_SCANS.with(|scans| std::mem::take(&mut *scans.borrow_mut()))
}

#[cfg(feature = "parquet")]
fn columnar_schema_has_columns(path: &Path, columns: &[&str]) -> bool {
    ColumnarSource::open(path).is_ok_and(|source| source.has_columns(columns))
}

#[cfg(not(feature = "parquet"))]
fn columnar_schema_has_columns(_path: &Path, _columns: &[&str]) -> bool {
    false
}

#[cfg(feature = "parquet")]
fn column_value_error(path: &Path, column: &str, ordinal: usize, detail: &str) -> DatasetError {
    DatasetError::Validation(format!(
        "{}: row {ordinal}: column {column}: {detail}",
        path.display()
    ))
}

#[cfg(feature = "parquet")]
fn downcast_array<'a, T: 'static>(
    array: &'a dyn arrow::array::Array,
    path: &Path,
    column: &str,
    ordinal: usize,
) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| column_value_error(path, column, ordinal, "internal Arrow type mismatch"))
}

#[cfg(feature = "parquet")]
fn unsigned_value(
    array: &dyn arrow::array::Array,
    row: usize,
    path: &Path,
    column: &str,
    ordinal: usize,
) -> Result<Option<u64>> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Int32 => {
            i64::from(downcast_array::<Int32Array>(array, path, column, ordinal)?.value(row))
        }
        DataType::Int64 => downcast_array::<Int64Array>(array, path, column, ordinal)?.value(row),
        DataType::UInt32 => {
            return Ok(Some(u64::from(
                downcast_array::<UInt32Array>(array, path, column, ordinal)?.value(row),
            )));
        }
        DataType::UInt64 => {
            return Ok(Some(
                downcast_array::<UInt64Array>(array, path, column, ordinal)?.value(row),
            ));
        }
        data_type => {
            return Err(column_value_error(
                path,
                column,
                ordinal,
                &format!("expected an integer, got {data_type}"),
            ));
        }
    };
    u64::try_from(value)
        .map(Some)
        .map_err(|_| column_value_error(path, column, ordinal, "must be non-negative"))
}

#[cfg(feature = "parquet")]
fn float_value(
    array: &dyn arrow::array::Array,
    row: usize,
    path: &Path,
    column: &str,
    ordinal: usize,
) -> Result<Option<f64>> {
    use arrow::array::{Float32Array, Float64Array};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Float32 => {
            f64::from(downcast_array::<Float32Array>(array, path, column, ordinal)?.value(row))
        }
        DataType::Float64 => {
            downcast_array::<Float64Array>(array, path, column, ordinal)?.value(row)
        }
        _ => {
            return unsigned_value(array, row, path, column, ordinal)
                .map(|value| value.map(|v| v as f64));
        }
    };
    if !value.is_finite() {
        return Err(column_value_error(path, column, ordinal, "must be finite"));
    }
    Ok(Some(value))
}

#[cfg(feature = "parquet")]
fn string_value(
    array: &dyn arrow::array::Array,
    row: usize,
    path: &Path,
    column: &str,
    ordinal: usize,
) -> Result<Option<String>> {
    use arrow::array::{LargeStringArray, StringArray, StringViewArray};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => Ok(Some(
            downcast_array::<StringArray>(array, path, column, ordinal)?
                .value(row)
                .to_owned(),
        )),
        DataType::LargeUtf8 => Ok(Some(
            downcast_array::<LargeStringArray>(array, path, column, ordinal)?
                .value(row)
                .to_owned(),
        )),
        DataType::Utf8View => Ok(Some(
            downcast_array::<StringViewArray>(array, path, column, ordinal)?
                .value(row)
                .to_owned(),
        )),
        DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64 => {
            Ok(unsigned_value(array, row, path, column, ordinal)?.map(|value| value.to_string()))
        }
        data_type => Err(column_value_error(
            path,
            column,
            ordinal,
            &format!("expected a string or integer, got {data_type}"),
        )),
    }
}

#[cfg(feature = "parquet")]
fn hash_values(
    array: &dyn arrow::array::Array,
    row: usize,
    path: &Path,
    ordinal: usize,
) -> Result<Vec<i64>> {
    use arrow::array::{Int32Array, Int64Array, LargeListArray, ListArray};
    use arrow::datatypes::DataType;

    if array.is_null(row) {
        return Ok(Vec::new());
    }
    let values = match array.data_type() {
        DataType::List(_) => {
            downcast_array::<ListArray>(array, path, "total_hashes", ordinal)?.value(row)
        }
        DataType::LargeList(_) => {
            downcast_array::<LargeListArray>(array, path, "total_hashes", ordinal)?.value(row)
        }
        data_type => {
            return Err(column_value_error(
                path,
                "total_hashes",
                ordinal,
                &format!("expected an integer list, got {data_type}"),
            ));
        }
    };
    match values.data_type() {
        DataType::Int32 => {
            Ok(
                downcast_array::<Int32Array>(values.as_ref(), path, "total_hashes", ordinal)?
                    .iter()
                    .flatten()
                    .map(i64::from)
                    .collect(),
            )
        }
        DataType::Int64 => {
            Ok(
                downcast_array::<Int64Array>(values.as_ref(), path, "total_hashes", ordinal)?
                    .iter()
                    .flatten()
                    .collect(),
            )
        }
        data_type => Err(column_value_error(
            path,
            "total_hashes",
            ordinal,
            &format!("expected integer list elements, got {data_type}"),
        )),
    }
}

#[cfg(feature = "parquet")]
fn resolve_session_column(
    has_provided: bool,
    has_poor_man: bool,
    configured: Option<&str>,
) -> Result<Option<&'static str>> {
    let requested = match configured.unwrap_or(COL_SESSION) {
        COL_SESSION => COL_SESSION,
        COL_POOR_MAN_SESSION => COL_POOR_MAN_SESSION,
        other => {
            return Err(DatasetError::Validation(format!(
                "AIPERF_DATASET_BASETEN_SESSION_COLUMN must be {COL_SESSION} or \
                 {COL_POOR_MAN_SESSION}, got {other}"
            )));
        }
    };
    let has_requested = if requested == COL_SESSION {
        has_provided
    } else {
        has_poor_man
    };
    if has_requested {
        return Ok(Some(requested));
    }
    let (fallback, has_fallback) = if requested == COL_SESSION {
        (COL_POOR_MAN_SESSION, has_poor_man)
    } else {
        (COL_SESSION, has_provided)
    };
    Ok(has_fallback.then_some(fallback))
}

#[cfg(feature = "parquet")]
fn selected_session_column(source: &ColumnarSource) -> Result<Option<&'static str>> {
    let configured = match std::env::var("AIPERF_DATASET_BASETEN_SESSION_COLUMN") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => {
            return Err(DatasetError::Validation(format!(
                "failed to read AIPERF_DATASET_BASETEN_SESSION_COLUMN: {error}"
            )));
        }
    };
    resolve_session_column(
        source.has_columns(&[COL_SESSION]),
        source.has_columns(&[COL_POOR_MAN_SESSION]),
        configured.as_deref(),
    )
}

#[cfg(feature = "parquet")]
struct ColumnarRows {
    rows: Vec<(BasetenRow, Option<String>, usize)>,
    min_timestamp: f64,
}

#[cfg(feature = "parquet")]
#[derive(Debug)]
struct TraceMetadata {
    timestamp: f64,
    session_id: Option<String>,
}

#[cfg(feature = "parquet")]
struct BasetenBatchIndices {
    timestamp: usize,
    prompt: usize,
    input_tokens: usize,
    output_tokens: usize,
    duration_e2e_ms: Option<usize>,
    duration_ttft_ms: Option<usize>,
    cached_tokens_reference: Option<usize>,
    total_hashes: Option<usize>,
    block_size: Option<usize>,
}

#[cfg(feature = "parquet")]
impl BasetenBatchIndices {
    fn new(batch: &arrow::record_batch::RecordBatch, path: &Path) -> Result<Self> {
        let schema = batch.schema();
        let required = |column: &str| {
            schema.index_of(column).map_err(|_| {
                DatasetError::Validation(format!(
                    "{}: projected batch is missing column {column}",
                    path.display()
                ))
            })
        };
        Ok(Self {
            timestamp: required(COL_TIME)?,
            prompt: required("prompt")?,
            input_tokens: required("input_tokens")?,
            output_tokens: required("output_tokens")?,
            duration_e2e_ms: schema.index_of("duration_e2e_ms").ok(),
            duration_ttft_ms: schema.index_of("duration_ttft_ms").ok(),
            cached_tokens_reference: schema.index_of("cached_tokens_reference").ok(),
            total_hashes: schema.index_of("total_hashes").ok(),
            block_size: schema.index_of("block_size").ok(),
        })
    }

    fn decode(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        row: usize,
        path: &Path,
        ordinal: usize,
    ) -> Result<BasetenRow> {
        let required_unsigned = |index: usize, column: &str| {
            unsigned_value(batch.column(index).as_ref(), row, path, column, ordinal)?
                .ok_or_else(|| column_value_error(path, column, ordinal, "must not be null"))
        };
        let timestamp = required_unsigned(self.timestamp, COL_TIME)?;
        let prompt = string_value(
            batch.column(self.prompt).as_ref(),
            row,
            path,
            "prompt",
            ordinal,
        )?
        .ok_or_else(|| column_value_error(path, "prompt", ordinal, "must not be null"))?;
        let input_tokens = required_unsigned(self.input_tokens, "input_tokens")?;
        let output_tokens = required_unsigned(self.output_tokens, "output_tokens")?;
        let output_length = u32::try_from(output_tokens)
            .map_err(|_| column_value_error(path, "output_tokens", ordinal, "does not fit in u32"))?
            .max(1);
        let duration_e2e_ms = self
            .duration_e2e_ms
            .map(|index| {
                float_value(
                    batch.column(index).as_ref(),
                    row,
                    path,
                    "duration_e2e_ms",
                    ordinal,
                )
            })
            .transpose()?
            .flatten();
        let duration_ttft_ms = self
            .duration_ttft_ms
            .map(|index| {
                float_value(
                    batch.column(index).as_ref(),
                    row,
                    path,
                    "duration_ttft_ms",
                    ordinal,
                )
            })
            .transpose()?
            .flatten();
        let cached_tokens_reference = self
            .cached_tokens_reference
            .map(|index| {
                unsigned_value(
                    batch.column(index).as_ref(),
                    row,
                    path,
                    "cached_tokens_reference",
                    ordinal,
                )
            })
            .transpose()?
            .flatten();
        let total_hashes = self
            .total_hashes
            .map(|index| hash_values(batch.column(index).as_ref(), row, path, ordinal))
            .transpose()?
            .unwrap_or_default();
        let block_size = self
            .block_size
            .map(|index| {
                unsigned_value(
                    batch.column(index).as_ref(),
                    row,
                    path,
                    "block_size",
                    ordinal,
                )?
                .map(|value| {
                    usize::try_from(value).map_err(|_| {
                        column_value_error(path, "block_size", ordinal, "does not fit in usize")
                    })
                })
                .transpose()
            })
            .transpose()?
            .flatten();
        Ok(BasetenRow {
            prompt,
            input_tokens,
            output_tokens,
            total_hashes,
            provided_session_id: None,
            poor_man_session_id: None,
            duration_e2e_ms,
            duration_ttft_ms,
            cached_tokens_reference,
            block_size,
            timestamp: Some(timestamp as f64),
            delay: None,
            output_length,
        })
    }
}

#[cfg(feature = "parquet")]
fn sampled_metadata_mask(
    metadata: &[TraceMetadata],
    ratio: Option<f64>,
    has_session_column: bool,
    rng_root: crate::rng::RngRoot,
) -> Result<Vec<bool>> {
    use crate::rng::compat::python_random::PythonRandomGenerator;
    use crate::rng::derive::DerivedRandomGenerator;
    use crate::rng::random_generator::RandomGenerator;

    let Some(ratio) = ratio.filter(|_| has_session_column) else {
        return Ok(vec![true; metadata.len()]);
    };
    let mut first_timestamps = HashMap::<&str, f64>::new();
    for row in metadata {
        if let Some(session_id) = row.session_id.as_deref() {
            first_timestamps
                .entry(session_id)
                .and_modify(|timestamp| *timestamp = timestamp.min(row.timestamp))
                .or_insert(row.timestamp);
        }
    }
    let mut sessions = first_timestamps
        .into_iter()
        .map(|(session_id, timestamp)| (timestamp, session_id))
        .collect::<Vec<_>>();
    sessions.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(right.1))
    });
    let mut rng = PythonRandomGenerator::from_rng_root(
        rng_root,
        crate::rng::namespace::DATASET_LOADER_BASETEN_TRACE_SESSION_SAMPLING,
    );
    let mut kept_sessions = sessions
        .iter()
        .filter(|_| rng.uniform(0.0, 1.0) < ratio)
        .map(|(_, session_id)| *session_id)
        .collect::<std::collections::HashSet<_>>();
    if kept_sessions.is_empty() && !sessions.is_empty() {
        let (_, session_id) = rng
            .choice(&sessions)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        kept_sessions.insert(*session_id);
    }
    Ok(metadata
        .iter()
        .map(|row| match row.session_id.as_deref() {
            Some(session_id) => kept_sessions.contains(session_id),
            None => rng.uniform(0.0, 1.0) < ratio,
        })
        .collect())
}

#[cfg(feature = "parquet")]
fn read_columnar_rows(
    path: &Path,
    replay: &ReplayOptions,
    rng_root: crate::rng::RngRoot,
    max_rows: Option<usize>,
) -> Result<ColumnarRows> {
    let source = ColumnarSource::open(path)?;
    let session_column = selected_session_column(&source)?;
    let mut metadata_columns = vec![COL_TIME];
    if let Some(session_column) = session_column {
        metadata_columns.push(session_column);
    }
    let mut metadata = Vec::new();
    source.for_each_batch(&metadata_columns, |batch| {
        if max_rows.is_some_and(|limit| metadata.len() >= limit) {
            return Ok(());
        }
        let timestamp_index = batch.schema().index_of(COL_TIME).map_err(|_| {
            DatasetError::Validation(format!("{} is missing {COL_TIME}", path.display()))
        })?;
        let session_index = session_column
            .map(|column| batch.schema().index_of(column))
            .transpose()
            .map_err(|_| {
                DatasetError::Validation(format!("{} is missing session column", path.display()))
            })?;
        for row in 0..batch.num_rows() {
            if max_rows.is_some_and(|limit| metadata.len() >= limit) {
                break;
            }
            let ordinal = metadata.len() + 1;
            let timestamp = unsigned_value(
                batch.column(timestamp_index).as_ref(),
                row,
                path,
                COL_TIME,
                ordinal,
            )?
            .ok_or_else(|| column_value_error(path, COL_TIME, ordinal, "must not be null"))?
                as f64;
            let session_id = session_index
                .map(|index| {
                    string_value(
                        batch.column(index).as_ref(),
                        row,
                        path,
                        session_column.unwrap_or(COL_SESSION),
                        ordinal,
                    )
                })
                .transpose()?
                .flatten();
            metadata.push(TraceMetadata {
                timestamp,
                session_id,
            });
        }
        Ok(())
    })?;
    let min_timestamp = metadata
        .iter()
        .map(|row| row.timestamp)
        .fold(f64::INFINITY, f64::min);
    let keep = sampled_metadata_mask(
        &metadata,
        replay.session_sample_ratio,
        session_column.is_some(),
        rng_root,
    )?;

    let mut columns = required_columns().to_vec();
    for column in [
        "duration_e2e_ms",
        "duration_ttft_ms",
        "cached_tokens_reference",
    ] {
        if source.has_columns(&[column]) {
            columns.push(column);
        }
    }
    if !replay.omit_kv_hints {
        for column in ["total_hashes", "block_size"] {
            if source.has_columns(&[column]) {
                columns.push(column);
            }
        }
    }
    let mut rows = Vec::with_capacity(keep.iter().filter(|is_kept| **is_kept).count());
    let mut ordinal = 0_usize;
    source.for_each_batch(&columns, |batch| {
        let indices = BasetenBatchIndices::new(&batch, path)?;
        for row in 0..batch.num_rows() {
            if ordinal >= metadata.len() {
                break;
            }
            let metadata_row = metadata.get(ordinal).ok_or_else(|| {
                DatasetError::Validation(format!(
                    "{} changed row count between columnar scans",
                    path.display()
                ))
            })?;
            let is_kept = keep.get(ordinal).copied().unwrap_or(false);
            ordinal += 1;
            if !is_kept {
                continue;
            }
            rows.push((
                indices.decode(&batch, row, path, ordinal)?,
                metadata_row.session_id.clone(),
                ordinal,
            ));
        }
        Ok(())
    })?;
    if ordinal != metadata.len() {
        return Err(DatasetError::Validation(format!(
            "{} changed row count between columnar scans",
            path.display()
        )));
    }
    Ok(ColumnarRows {
        rows,
        min_timestamp,
    })
}

#[cfg(not(feature = "parquet"))]
fn read_columnar_rows(
    path: &Path,
    _replay: &ReplayOptions,
    _rng_root: crate::rng::RngRoot,
    _max_rows: Option<usize>,
) -> Result<()> {
    Err(DatasetError::Validation(format!(
        "baseten_trace dataset {} requires an aiperf runner built with the `parquet` feature",
        path.display()
    )))
}

#[async_trait]
impl DatasetLoader for BasetenTraceDatasetLoader {
    fn name(&self) -> &str {
        "baseten_trace"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        let Some(path) = &probe.path else {
            return false;
        };
        if !matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("parquet" | "arrow" | "ipc")
        ) {
            return false;
        }
        columnar_schema_has_columns(path, &required_columns())
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let replay = ReplayOptions::from_options(&config.options)?;
        let (mut raw_rows, columnar_min_timestamp) = match &config.source {
            DatasetSource::Path(path) => {
                #[cfg(feature = "parquet")]
                {
                    let columnar =
                        read_columnar_rows(path, &replay, config.rng_root, config.max_rows)?;
                    let rows = columnar
                        .rows
                        .into_iter()
                        .map(|(row, group_key, ordinal)| {
                            (
                                row,
                                group_key,
                                RowOrigin::FileLine {
                                    path: path.clone(),
                                    line: ordinal,
                                },
                            )
                        })
                        .collect();
                    (rows, Some(columnar.min_timestamp))
                }
                #[cfg(not(feature = "parquet"))]
                {
                    read_columnar_rows(path, &replay, config.rng_root, config.max_rows)?;
                    unreachable!()
                }
            }
            DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => {
                // Public / `--hf-dataset --hf-format baseten_trace` path: reuse the
                // shared remote parquet/jsonl acquisition, then apply baseten
                // replay normalization below.
                let rows = crate::dataset::loader::public::load_raw_rows(config).await?;
                let rows = rows
                    .into_iter()
                    .map(|raw| {
                        parse_row(&raw.value, &raw.origin).map(|row| (row, None, raw.origin))
                    })
                    .collect::<Result<Vec<_>>>()?;
                (rows, None)
            }
            _ => {
                return Err(DatasetError::Validation(
                    "baseten_trace requires a Parquet file path, URL, or Hugging Face source"
                        .into(),
                ));
            }
        };
        let min_timestamp = columnar_min_timestamp.unwrap_or_else(|| {
            raw_rows
                .iter()
                .filter_map(|(row, _, _)| row.timestamp)
                .fold(f64::INFINITY, f64::min)
        });
        let mut out = Vec::with_capacity(raw_rows.len());
        let mut next_ordinal = 0_u64;
        for (row, group_key, origin) in raw_rows.drain(..) {
            let mut row = row;
            if let Some(timestamp) = row.timestamp {
                let mut normalized = timestamp - min_timestamp;
                if !in_window_ms(normalized, config) {
                    continue;
                }
                if config
                    .max_input_tokens
                    .is_some_and(|cap| row.input_tokens > cap)
                {
                    continue;
                }
                if replay.speedup != 1.0 {
                    normalized /= replay.speedup;
                }
                row.timestamp = Some(normalized);
            }
            let value = row_to_value(&row);
            let wire = serde_json::to_vec(&value).map_err(DatasetError::from)?;
            let group_key = if columnar_min_timestamp.is_some() {
                Some(group_key.unwrap_or_else(|| {
                    let group_key = format!("baseten_{next_ordinal:06}");
                    next_ordinal += 1;
                    group_key
                }))
            } else {
                None
            };
            out.push(RawRow {
                value,
                wire: Some(Bytes::from(wire)),
                session_id: None,
                group_key,
                origin,
            });
        }
        Ok(out)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

fn in_window_ms(timestamp_ms: f64, config: &LoadConfig) -> bool {
    config
        .start_offset_ms
        .is_none_or(|start| timestamp_ms >= start)
        && config.end_offset_ms.is_none_or(|end| timestamp_ms <= end)
}

/// Round-trips a [`BasetenRow`] through JSON so it can ride `RawRow::value`/
/// `wire` between `load` and `compose` (the loader/composer split means the
/// two run as separate registry steps, not a single pass over open file
/// handles).
fn row_to_value(row: &BasetenRow) -> Value {
    json!({
        "prompt": row.prompt,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "total_hashes": row.total_hashes,
        "provided_session_id": row.provided_session_id,
        "poor_man_session_id": row.poor_man_session_id,
        "duration_e2e_ms": row.duration_e2e_ms,
        "duration_ttft_ms": row.duration_ttft_ms,
        "cached_tokens_reference": row.cached_tokens_reference,
        "block_size": row.block_size,
        "timestamp": row.timestamp,
        "output_length": row.output_length,
    })
}

fn row_from_value(value: &Value) -> Result<BasetenRow> {
    let object = value
        .as_object()
        .ok_or_else(|| DatasetError::Validation("invalid baseten_trace intermediate row".into()))?;
    Ok(BasetenRow {
        prompt: object
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        input_tokens: object
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: object
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        total_hashes: object
            .get("total_hashes")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default(),
        provided_session_id: object
            .get("provided_session_id")
            .and_then(|value| value.as_str().map(str::to_string)),
        poor_man_session_id: object
            .get("poor_man_session_id")
            .and_then(|value| value.as_str().map(str::to_string)),
        duration_e2e_ms: object.get("duration_e2e_ms").and_then(Value::as_f64),
        duration_ttft_ms: object.get("duration_ttft_ms").and_then(Value::as_f64),
        cached_tokens_reference: object
            .get("cached_tokens_reference")
            .and_then(Value::as_u64),
        block_size: object
            .get("block_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
        timestamp: object.get("timestamp").and_then(Value::as_f64),
        delay: None,
        output_length: object
            .get("output_length")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(1),
    })
}

impl Composer for BasetenTraceComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let replay = ReplayOptions::from_options(&config.format_options)?;
        let has_resolved_groups = rows.iter().all(|row| row.group_key.is_some());
        let mut group_keys = Vec::with_capacity(rows.len());
        let mut parsed = Vec::with_capacity(rows.len());
        for row in rows {
            group_keys.push(row.group_key);
            parsed.push(row_from_value(&row.value)?);
        }

        // Open-loop idle-gap reflow runs on every timed row, before grouping,
        // so a sparse (sampled) trace does not idle through dead air.
        // Closed-loop defers the reflow until after back-pressure clears
        // continuation timestamps, so it only touches session-start turns.
        if replay.open_loop && replay.max_idle_gap_cap_ms.is_some() {
            apply_idle_gap_cap(&mut parsed, replay.max_idle_gap_cap_ms);
        }

        let session_key = (!has_resolved_groups)
            .then(|| choose_session_key(&parsed))
            .flatten();
        if !has_resolved_groups && let Some(ratio) = replay.session_sample_ratio {
            sample_trace_sessions(&mut parsed, session_key, ratio, config.rng_root)?;
        }

        let mut groups: HashMap<String, Vec<BasetenRow>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        let mut next_ordinal: u64 = 0;
        for (index, row) in parsed.drain(..).enumerate() {
            let session_id = group_keys
                .get_mut(index)
                .and_then(Option::take)
                .or_else(|| match session_key {
                    Some(SessionKey::Provided) => row.provided_session_id.clone(),
                    Some(SessionKey::PoorMan) => row.poor_man_session_id.clone(),
                    None => None,
                })
                .unwrap_or_else(|| {
                    let id = format!("baseten_{next_ordinal:06}");
                    next_ordinal += 1;
                    id
                });
            if !groups.contains_key(&session_id) {
                order.push(session_id.clone());
            }
            groups.entry(session_id).or_default().push(row);
        }
        for rows in groups.values_mut() {
            rows.sort_by(|a, b| {
                a.timestamp
                    .unwrap_or(0.0)
                    .partial_cmp(&b.timestamp.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        // Order sessions by first-event timestamp (then session id), matching
        // Python's `_order_groups`.
        order.sort_by(|a, b| {
            let ta = groups[a]
                .first()
                .and_then(|row| row.timestamp)
                .unwrap_or(0.0);
            let tb = groups[b]
                .first()
                .and_then(|row| row.timestamp)
                .unwrap_or(0.0);
            ta.partial_cmp(&tb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        let mut sessions: Vec<(String, Vec<BasetenRow>)> = order
            .into_iter()
            .map(|id| {
                let rows = groups.remove(&id).unwrap_or_default();
                (id, rows)
            })
            .collect();

        if replay.open_loop && replay.open_loop_strict {
            let mut exploded = Vec::new();
            for (session_id, rows) in sessions {
                for (index, row) in rows.into_iter().enumerate() {
                    exploded.push((format!("{session_id}#{index}"), vec![row]));
                }
            }
            sessions = exploded;
        } else if !replay.open_loop {
            for (_, rows) in &mut sessions {
                apply_back_pressure(rows, &replay);
            }
            if replay.max_idle_gap_cap_ms.is_some() {
                let mut flat: Vec<&mut BasetenRow> = sessions
                    .iter_mut()
                    .flat_map(|(_, rows)| rows.iter_mut())
                    .collect();
                let timed_ms: Vec<f64> = flat.iter().filter_map(|row| row.timestamp).collect();
                let timed_indices: Vec<usize> = flat
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| row.timestamp.is_some())
                    .map(|(index, _)| index)
                    .collect();
                let reflowed = reflow_idle_gaps(&timed_ms, replay.max_idle_gap_cap_ms);
                for (position, &index) in timed_indices.iter().enumerate() {
                    flat[index].timestamp = Some(reflowed[position]);
                }
            }
        }

        let mut conversations = Vec::with_capacity(sessions.len());
        for (session_id, rows) in sessions {
            let mut conversation = Conversation::new(SessionId::new(session_id));
            if rows.len() > 1 {
                conversation.context_mode =
                    Some(ConversationContextMode::MessageArrayWithResponses);
            }
            let mut parent = None;
            for row in &rows {
                // Content-addressing identity uses the real tokenizer output
                // (so distinct prompts never collide into one segment just
                // because they share a recorded token count); the REPORTED
                // input_tokens below stays the recorded value regardless,
                // since the wire sends this literal prompt text verbatim and
                // the original request was tokenized by a possibly-different
                // tokenizer -- matching Python's faithful-replay design
                // (`trace.input_length = int(trace.input_tokens)`, fully
                // decoupled from any local re-tokenization).
                let encode_tokens = tokenizer.encode(&row.prompt)?;
                let handle = segments.intern_text(
                    parent,
                    "user",
                    Bytes::from(row.prompt.clone()),
                    encode_tokens.into_boxed_slice(),
                )?;
                parent = Some(handle);
                // Cap once and reuse for both max_tokens and the injected
                // min_tokens hint -- Python applies max_osl capping
                // (_cap_grouped_traces_max_osl) before building the request
                // body, so min_tokens reflects the CAPPED length. Using the
                // raw recorded length for min_tokens here would let it
                // exceed max_tokens whenever --max-output-tokens caps below
                // a row's recorded output_tokens, an invalid request.
                let capped_output_length =
                    cap_output(Some(row.output_length), config.max_output_tokens)
                        .unwrap_or(row.output_length);
                let extra_body = build_request_body(row, capped_output_length, &replay);
                let extra_body_handle = if extra_body.as_object().is_some_and(|obj| !obj.is_empty())
                {
                    let bytes = serde_json::to_vec(&extra_body).map_err(DatasetError::from)?;
                    let handle = segments.intern_raw(parent, Bytes::from(bytes))?;
                    parent = Some(handle);
                    Some(handle)
                } else {
                    None
                };
                let mut turn = Turn {
                    timestamp_ms: row.timestamp,
                    delay_ms: row.delay,
                    recorded_outcome: recorded_outcome(row),
                    input_tokens: Some(row.input_tokens),
                    max_tokens: Some(capped_output_length),
                    extra_body: extra_body_handle,
                    content: smallvec![ContentGroup {
                        kind: MediaKind::Text,
                        name: "text".into(),
                        handles: smallvec![handle],
                        uuids: smallvec![],
                    }],
                    ..Turn::default()
                };
                config.finalizer()?.finalize_turn(&mut turn)?;
                conversation.turns.push(turn);
            }
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

fn recorded_outcome(row: &BasetenRow) -> Option<RecordedOutcome> {
    if row.duration_e2e_ms.is_none()
        && row.duration_ttft_ms.is_none()
        && row.cached_tokens_reference.is_none()
    {
        return None;
    }
    Some(RecordedOutcome {
        duration_e2e_ms: row.duration_e2e_ms,
        duration_ttft_ms: row.duration_ttft_ms,
        cached_tokens_reference: row.cached_tokens_reference,
    })
}

fn apply_idle_gap_cap(rows: &mut [BasetenRow], cap_ms: Option<f64>) {
    let timed_ms: Vec<f64> = rows.iter().filter_map(|row| row.timestamp).collect();
    if timed_ms.is_empty() {
        return;
    }
    let reflowed = reflow_idle_gaps(&timed_ms, cap_ms);
    let mut position = 0;
    for row in rows.iter_mut() {
        if row.timestamp.is_some() {
            row.timestamp = Some(reflowed[position]);
            position += 1;
        }
    }
}

/// Convert continuation turns from absolute timestamps to inter-turn delays.
/// The recorded start-to-start gap already includes the prior turn's service
/// time, and fixed-schedule replay applies `delay` AFTER the prior turn
/// completes, so subtract the prior turn's recorded end-to-end duration to
/// avoid double-counting server time. `duration_e2e_ms` is not
/// speedup-scaled, so divide it to match the already-scaled timestamps.
fn apply_back_pressure(rows: &mut [BasetenRow], replay: &ReplayOptions) {
    let mut prev_ts: Option<f64> = None;
    let mut prev_e2e_ms = 0.0_f64;
    for (index, row) in rows.iter_mut().enumerate() {
        let ts = row.timestamp.unwrap_or(0.0);
        if index == 0 {
            prev_ts = Some(ts);
            prev_e2e_ms = row.duration_e2e_ms.unwrap_or(0.0) / replay.speedup;
            continue;
        }
        let gap = (ts - prev_ts.unwrap_or(0.0)).max(0.0);
        let delay = (gap - prev_e2e_ms).max(0.0);
        row.delay = Some(replay.clamp_delay(delay));
        row.timestamp = None;
        prev_ts = Some(ts);
        prev_e2e_ms = row.duration_e2e_ms.unwrap_or(0.0) / replay.speedup;
    }
}

/// KV-cache-aware routing hints injected per-turn: `min_tokens` from the
/// max-osl-capped output length (matching `max_tokens`, both derived from
/// the same value so the pair is never contradictory), plus
/// `hash_ids`/`block_size` unless `--omit-kv-hints` is set. Inert when
/// there is no routing choice; some strict frontends reject unknown body
/// params, hence the opt-out.
fn build_request_body(
    row: &BasetenRow,
    capped_output_length: u32,
    replay: &ReplayOptions,
) -> Value {
    let mut body = Map::new();
    if replay.force_min_tokens {
        body.insert("min_tokens".into(), json!(capped_output_length));
    }
    if !replay.omit_kv_hints {
        if !row.total_hashes.is_empty() {
            body.insert("hash_ids".into(), json!(row.total_hashes));
        }
        if let Some(block_size) = row.block_size {
            body.insert("block_size".into(), json!(block_size));
        }
    }
    Value::Object(body)
}

#[cfg(all(test, feature = "parquet"))]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use parquet::data_type::{ByteArrayType, Int64Type};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    use super::*;
    use crate::dataset::Payload;
    use crate::dataset::loader::{DatasetFormatRegistration, DatasetProbe, LoaderRegistry};
    use crate::dataset::tokenizer::TiktokenTokenizer;
    use crate::rng::RngRoot;

    struct FixtureRow {
        timestamp_start_unix_ms: i64,
        prompt: &'static str,
        input_tokens: i64,
        output_tokens: i64,
        provided_session_id: &'static str,
        duration_e2e_ms: i64,
        duration_ttft_ms: Option<i64>,
        cached_tokens_reference: Option<i64>,
    }

    /// Write a minimal Baseten-shaped Parquet fixture (the required columns
    /// plus `provided_session_id`/`duration_e2e_ms`) to a temp file, return
    /// its path.
    fn write_fixture(directory: &Path, rows: &[FixtureRow]) -> std::path::PathBuf {
        let schema = Arc::new(
            parse_message_type(
                "message schema {
                    REQUIRED INT64 timestamp_start_unix_ms;
                    REQUIRED BYTE_ARRAY prompt (UTF8);
                    REQUIRED INT64 input_tokens;
                    REQUIRED INT64 output_tokens;
                    REQUIRED BYTE_ARRAY provided_session_id (UTF8);
                    REQUIRED INT64 duration_e2e_ms;
                    OPTIONAL INT64 duration_ttft_ms;
                    OPTIONAL INT64 cached_tokens_reference;
                }",
            )
            .unwrap(),
        );
        let path = directory.join("baseten.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, Default::default()).unwrap();
        let mut row_group = writer.next_row_group().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(
                &rows
                    .iter()
                    .map(|r| r.timestamp_start_unix_ms)
                    .collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<ByteArrayType>()
            .write_batch(
                &rows
                    .iter()
                    .map(|r| parquet::data_type::ByteArray::from(r.prompt.as_bytes().to_vec()))
                    .collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(
                &rows.iter().map(|r| r.input_tokens).collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(
                &rows.iter().map(|r| r.output_tokens).collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<ByteArrayType>()
            .write_batch(
                &rows
                    .iter()
                    .map(|r| {
                        parquet::data_type::ByteArray::from(
                            r.provided_session_id.as_bytes().to_vec(),
                        )
                    })
                    .collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(
                &rows.iter().map(|r| r.duration_e2e_ms).collect::<Vec<_>>(),
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        let values = rows
            .iter()
            .filter_map(|row| row.duration_ttft_ms)
            .collect::<Vec<_>>();
        let definition_levels = rows
            .iter()
            .map(|row| i16::from(row.duration_ttft_ms.is_some()))
            .collect::<Vec<_>>();
        column
            .typed::<Int64Type>()
            .write_batch(&values, Some(&definition_levels), None)
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        let values = rows
            .iter()
            .filter_map(|row| row.cached_tokens_reference)
            .collect::<Vec<_>>();
        let definition_levels = rows
            .iter()
            .map(|row| i16::from(row.cached_tokens_reference.is_some()))
            .collect::<Vec<_>>();
        column
            .typed::<Int64Type>()
            .write_batch(&values, Some(&definition_levels), None)
            .unwrap();
        column.close().unwrap();

        row_group.close().unwrap();
        writer.close().unwrap();
        path
    }

    fn write_arrow_fixture(
        directory: &Path,
        suffix: &str,
        rows: &[FixtureRow],
    ) -> std::path::PathBuf {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_TIME, DataType::Int64, false),
            Field::new("prompt", DataType::Utf8, false),
            Field::new("input_tokens", DataType::Int64, false),
            Field::new("output_tokens", DataType::Int64, false),
            Field::new(COL_SESSION, DataType::Utf8, false),
            Field::new("duration_e2e_ms", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|row| row.timestamp_start_unix_ms),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|row| row.prompt),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|row| row.input_tokens),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|row| row.output_tokens),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|row| row.provided_session_id),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|row| row.duration_e2e_ms),
                )),
            ],
        )
        .unwrap();
        let path = directory.join(format!("baseten.{suffix}"));
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        path
    }

    fn write_session_policy_fixture(directory: &Path) -> std::path::PathBuf {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_TIME, DataType::Int64, false),
            Field::new("prompt", DataType::Utf8, false),
            Field::new("input_tokens", DataType::Int64, false),
            Field::new("output_tokens", DataType::Int64, false),
            Field::new(COL_SESSION, DataType::Utf8, false),
            Field::new(COL_POOR_MAN_SESSION, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0_i64..6)),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
                Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(1, 6))),
                Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(1, 6))),
                Arc::new(StringArray::from(vec!["p1", "p1", "p2", "p2", "p3", "p4"])),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n("one", 6))),
            ],
        )
        .unwrap();
        let path = directory.join("session-policy.arrow");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        path
    }

    fn write_wide_late_invalid_fixture(directory: &Path) -> std::path::PathBuf {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_TIME, DataType::Int64, false),
            Field::new("prompt", DataType::Utf8, false),
            Field::new("input_tokens", DataType::Int64, false),
            Field::new("output_tokens", DataType::Int64, false),
            Field::new(COL_SESSION, DataType::Utf8, false),
            Field::new("unused_blob", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0_i64..131)),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "prompt", 131,
                ))),
                Arc::new(Int64Array::from_iter_values(
                    (0..131).map(|index| if index == 130 { -1 } else { 1 }),
                )),
                Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(1, 131))),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "session", 131,
                ))),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "unused-wide-value",
                    131,
                ))),
            ],
        )
        .unwrap();
        let path = directory.join("wide-late-invalid.arrow");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        path
    }

    async fn build(
        path: std::path::PathBuf,
        options: Map<String, Value>,
    ) -> crate::dataset::Dataset {
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(BasetenTraceDatasetLoader),
                Arc::new(BasetenTraceComposer),
            ))
            .unwrap();
        let mut load = LoadConfig::new(DatasetSource::Path(path));
        load.options = options.clone();
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(9)));
        compose.format_options = options;
        registry
            .build_dataset(
                Some("baseten_trace"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    #[test]
    fn can_load_requires_parquet_extension_and_required_columns() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[FixtureRow {
                timestamp_start_unix_ms: 0,
                prompt: "hi",
                input_tokens: 1,
                output_tokens: 1,
                provided_session_id: "s",
                duration_e2e_ms: 0,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            }],
        );
        assert!(BasetenTraceDatasetLoader.can_load(&DatasetProbe {
            value: None,
            path: Some(path.clone()),
        }));

        let not_parquet = directory.path().join("baseten.jsonl");
        std::fs::write(&not_parquet, "{}\n").unwrap();
        assert!(!BasetenTraceDatasetLoader.can_load(&DatasetProbe {
            value: None,
            path: Some(not_parquet),
        }));
    }

    #[tokio::test]
    async fn arrow_ipc_detection_and_composition_match_parquet() {
        let directory = tempfile::tempdir().unwrap();
        let rows = [
            FixtureRow {
                timestamp_start_unix_ms: 1_000,
                prompt: "first",
                input_tokens: 4,
                output_tokens: 3,
                provided_session_id: "shared",
                duration_e2e_ms: 10,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            },
            FixtureRow {
                timestamp_start_unix_ms: 1_250,
                prompt: "second",
                input_tokens: 5,
                output_tokens: 2,
                provided_session_id: "shared",
                duration_e2e_ms: 20,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            },
        ];
        let parquet = write_fixture(directory.path(), &rows);
        let arrow = write_arrow_fixture(directory.path(), "arrow", &rows);
        let ipc = write_arrow_fixture(directory.path(), "ipc", &rows);
        for path in [&arrow, &ipc] {
            assert!(BasetenTraceDatasetLoader.can_load(&DatasetProbe {
                value: None,
                path: Some(path.clone()),
            }));
        }

        let parquet_dataset = build(parquet, Map::new()).await;
        let direct_rows = BasetenTraceDatasetLoader
            .load(&LoadConfig::new(DatasetSource::Path(arrow.clone())))
            .await
            .unwrap();
        assert_eq!(
            direct_rows
                .iter()
                .map(|row| &row.origin)
                .collect::<Vec<_>>(),
            vec![
                &RowOrigin::FileLine {
                    path: arrow.clone(),
                    line: 1,
                },
                &RowOrigin::FileLine {
                    path: arrow.clone(),
                    line: 2,
                },
            ]
        );
        let arrow_dataset = build(arrow, Map::new()).await;
        let ipc_dataset = build(ipc, Map::new()).await;
        assert_eq!(parquet_dataset.conversations().len(), 1);
        assert_eq!(arrow_dataset.conversations().len(), 1);
        assert_eq!(ipc_dataset.conversations().len(), 1);
        let parquet_turns = &parquet_dataset.conversations()[0].turns;
        let arrow_turns = &arrow_dataset.conversations()[0].turns;
        let ipc_turns = &ipc_dataset.conversations()[0].turns;
        assert_eq!(arrow_turns.len(), parquet_turns.len());
        assert_eq!(ipc_turns.len(), parquet_turns.len());
        for ((arrow_turn, ipc_turn), parquet_turn) in
            arrow_turns.iter().zip(ipc_turns).zip(parquet_turns)
        {
            for columnar_turn in [arrow_turn, ipc_turn] {
                assert_eq!(columnar_turn.timestamp_ms, parquet_turn.timestamp_ms);
                assert_eq!(columnar_turn.input_tokens, parquet_turn.input_tokens);
                assert_eq!(columnar_turn.max_tokens, parquet_turn.max_tokens);
            }
        }
    }

    #[tokio::test]
    async fn configured_default_session_column_controls_grouping() {
        let directory = tempfile::tempdir().unwrap();
        let dataset = build(write_session_policy_fixture(directory.path()), Map::new()).await;
        let mut turn_counts = dataset
            .conversations()
            .iter()
            .map(|conversation| conversation.turns.len())
            .collect::<Vec<_>>();
        turn_counts.sort_unstable();
        assert_eq!(turn_counts, vec![1, 1, 2, 2]);
    }

    #[tokio::test]
    async fn projected_batches_skip_unused_columns_and_validate_late_rows() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_wide_late_invalid_fixture(directory.path());
        let _ = take_columnar_scans();
        let error = BasetenTraceDatasetLoader
            .load(&LoadConfig::new(DatasetSource::Path(path.clone())))
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&path.display().to_string()));
        assert!(message.contains("row 131"));
        assert!(message.contains("column input_tokens"));
        let scans = take_columnar_scans();
        assert!(!scans.is_empty());
        assert!(scans.iter().all(|(columns, rows)| {
            *rows <= PARQUET_BATCH_SIZE && !columns.iter().any(|column| column == "unused_blob")
        }));
    }

    #[tokio::test]
    async fn open_loop_replay_keeps_absolute_timestamps_and_injects_kv_hints() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[
                FixtureRow {
                    timestamp_start_unix_ms: 1_000,
                    prompt: "first",
                    input_tokens: 10,
                    output_tokens: 5,
                    provided_session_id: "s1",
                    duration_e2e_ms: 200,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
                FixtureRow {
                    timestamp_start_unix_ms: 3_000,
                    prompt: "second",
                    input_tokens: 8,
                    output_tokens: 0, // canceled request: output_tokens=0 floors to 1.
                    provided_session_id: "s2",
                    duration_e2e_ms: 100,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
            ],
        );
        let dataset = build(path, Map::new()).await;
        assert_eq!(dataset.conversations().len(), 2);
        // Ordered by first-event timestamp: s1 (1000) before s2 (3000).
        let first = &dataset.conversations()[0];
        assert_eq!(first.turns[0].timestamp_ms, Some(0.0)); // normalized: 1000 - min(1000).
        assert_eq!(first.turns[0].max_tokens, Some(5));
        let second = &dataset.conversations()[1];
        assert_eq!(second.turns[0].timestamp_ms, Some(2000.0));
        assert_eq!(second.turns[0].max_tokens, Some(1)); // floored from 0.

        let Payload::Raw { wire } = dataset
            .segments()
            .get(first.turns[0].extra_body.unwrap())
            .unwrap()
        else {
            panic!("expected raw extra_body payload");
        };
        let body: Value = serde_json::from_slice(wire).unwrap();
        assert_eq!(body["min_tokens"], json!(5));
    }

    #[tokio::test]
    async fn max_output_tokens_cap_applies_to_both_max_tokens_and_min_tokens() {
        // min_tokens must never exceed max_tokens: Python caps output_length
        // (max_osl) BEFORE building the request body, so the injected
        // min_tokens reflects the capped value, not the raw recorded one.
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[FixtureRow {
                timestamp_start_unix_ms: 0,
                prompt: "hi",
                input_tokens: 3,
                output_tokens: 100,
                provided_session_id: "s1",
                duration_e2e_ms: 0,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            }],
        );
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(BasetenTraceDatasetLoader),
                Arc::new(BasetenTraceComposer),
            ))
            .unwrap();
        let load = LoadConfig::new(DatasetSource::Path(path));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(9)));
        compose.max_output_tokens = Some(10);
        let dataset = registry
            .build_dataset(
                Some("baseten_trace"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.max_tokens, Some(10));
        let Payload::Raw { wire } = dataset.segments().get(turn.extra_body.unwrap()).unwrap()
        else {
            panic!("expected raw extra_body payload");
        };
        let body: Value = serde_json::from_slice(wire).unwrap();
        assert_eq!(body["min_tokens"], json!(10));
    }

    #[tokio::test]
    async fn omit_kv_hints_and_no_force_min_tokens_disable_injection() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[FixtureRow {
                timestamp_start_unix_ms: 0,
                prompt: "hi",
                input_tokens: 3,
                output_tokens: 4,
                provided_session_id: "s1",
                duration_e2e_ms: 0,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            }],
        );
        let mut options = Map::new();
        options.insert("omit_kv_hints".into(), json!(true));
        options.insert("force_min_tokens".into(), json!(false));
        let dataset = build(path, options).await;
        assert!(dataset.conversations()[0].turns[0].extra_body.is_none());
    }

    #[tokio::test]
    async fn recorded_outcomes_survive_when_request_hints_are_disabled() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[FixtureRow {
                timestamp_start_unix_ms: 0,
                prompt: "hi",
                input_tokens: 128,
                output_tokens: 4,
                provided_session_id: "s1",
                duration_e2e_ms: 800,
                duration_ttft_ms: Some(120),
                cached_tokens_reference: Some(64),
            }],
        );
        let mut options = Map::new();
        options.insert("omit_kv_hints".into(), json!(true));
        options.insert("force_min_tokens".into(), json!(false));

        let dataset = build(path, options).await;
        let turn = &dataset.conversations()[0].turns[0];
        let outcome = turn.recorded_outcome.as_ref().unwrap();
        assert_eq!(outcome.duration_e2e_ms, Some(800.0));
        assert_eq!(outcome.duration_ttft_ms, Some(120.0));
        assert_eq!(outcome.cached_tokens_reference, Some(64));
        assert!(turn.extra_body.is_none());
    }

    #[tokio::test]
    async fn recorded_outcomes_survive_closed_loop_replay() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[
                FixtureRow {
                    timestamp_start_unix_ms: 0,
                    prompt: "turn one",
                    input_tokens: 128,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 200,
                    duration_ttft_ms: Some(40),
                    cached_tokens_reference: Some(0),
                },
                FixtureRow {
                    timestamp_start_unix_ms: 1_000,
                    prompt: "turn two",
                    input_tokens: 192,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 100,
                    duration_ttft_ms: Some(30),
                    cached_tokens_reference: Some(128),
                },
            ],
        );
        let mut options = Map::new();
        options.insert("open_loop_replay".into(), json!(false));

        let dataset = build(path, options).await;
        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns[1].delay_ms, Some(800.0));
        let first = turns[0].recorded_outcome.as_ref().unwrap();
        assert_eq!(first.duration_ttft_ms, Some(40.0));
        assert_eq!(first.cached_tokens_reference, Some(0));
        let second = turns[1].recorded_outcome.as_ref().unwrap();
        assert_eq!(second.duration_ttft_ms, Some(30.0));
        assert_eq!(second.cached_tokens_reference, Some(128));
    }

    #[tokio::test]
    async fn missing_ttft_and_cached_outcomes_remain_absent() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[FixtureRow {
                timestamp_start_unix_ms: 0,
                prompt: "hi",
                input_tokens: 1,
                output_tokens: 1,
                provided_session_id: "s",
                duration_e2e_ms: 0,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
            }],
        );

        let dataset = build(path, Map::new()).await;
        let outcome = dataset.conversations()[0].turns[0]
            .recorded_outcome
            .as_ref()
            .unwrap();
        assert_eq!(outcome.duration_e2e_ms, Some(0.0));
        assert_eq!(outcome.duration_ttft_ms, None);
        assert_eq!(outcome.cached_tokens_reference, None);
    }

    #[tokio::test]
    async fn same_session_rows_group_into_one_multiturn_conversation() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[
                FixtureRow {
                    timestamp_start_unix_ms: 0,
                    prompt: "turn one",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 50,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
                FixtureRow {
                    timestamp_start_unix_ms: 500,
                    prompt: "turn two",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 50,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
            ],
        );
        let dataset = build(path, Map::new()).await;
        assert_eq!(dataset.conversations().len(), 1);
        assert_eq!(dataset.conversations()[0].turns.len(), 2);
        assert_eq!(
            dataset.conversations()[0].context_mode,
            Some(ConversationContextMode::MessageArrayWithResponses)
        );
    }

    #[tokio::test]
    async fn closed_loop_replay_converts_gap_into_delay_minus_prior_service_time() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[
                FixtureRow {
                    timestamp_start_unix_ms: 0,
                    prompt: "turn one",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 200,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
                FixtureRow {
                    timestamp_start_unix_ms: 1_000,
                    prompt: "turn two",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 100,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
            ],
        );
        let mut options = Map::new();
        options.insert("open_loop_replay".into(), json!(false));
        let dataset = build(path, options).await;
        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns[0].timestamp_ms, Some(0.0));
        assert_eq!(turns[0].delay_ms, None);
        // gap=1000, prior turn's recorded service time=200 -> delay=800.
        assert_eq!(turns[1].timestamp_ms, None);
        assert_eq!(turns[1].delay_ms, Some(800.0));
    }

    #[tokio::test]
    async fn open_loop_strict_explodes_sessions_into_independent_conversations() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_fixture(
            directory.path(),
            &[
                FixtureRow {
                    timestamp_start_unix_ms: 0,
                    prompt: "turn one",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 50,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
                FixtureRow {
                    timestamp_start_unix_ms: 500,
                    prompt: "turn two",
                    input_tokens: 5,
                    output_tokens: 5,
                    provided_session_id: "shared",
                    duration_e2e_ms: 50,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                },
            ],
        );
        let mut options = Map::new();
        options.insert("open_loop_strict".into(), json!(true));
        let dataset = build(path, options).await;
        assert_eq!(dataset.conversations().len(), 2);
        for conversation in dataset.conversations() {
            assert_eq!(conversation.turns.len(), 1);
        }
    }

    #[test]
    fn reflow_idle_gaps_caps_large_gaps_and_preserves_order_and_ties() {
        // Ports the Python docstring example semantics: a >cap gap shortens to
        // the cap; ties keep input order (stable); no cap is the identity.
        let timestamps = [0.0, 100_000.0, 100_500.0];
        let reflowed = reflow_idle_gaps(&timestamps, Some(1_000.0));
        assert_eq!(reflowed, vec![0.0, 1_000.0, 1_500.0]);
        assert_eq!(reflow_idle_gaps(&timestamps, None), timestamps.to_vec());
    }

    #[test]
    fn choose_session_key_prefers_the_stronger_repeated_signal() {
        let rows = |provided: &[Option<&str>], poor_man: &[Option<&str>]| -> Vec<BasetenRow> {
            provided
                .iter()
                .zip(poor_man)
                .map(|(p, m)| BasetenRow {
                    prompt: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                    total_hashes: Vec::new(),
                    provided_session_id: p.map(str::to_string),
                    poor_man_session_id: m.map(str::to_string),
                    duration_e2e_ms: None,
                    duration_ttft_ms: None,
                    cached_tokens_reference: None,
                    block_size: None,
                    timestamp: None,
                    delay: None,
                    output_length: 1,
                })
                .collect()
        };
        // provided has one repeated pair, poor_man has none -> provided wins.
        let strong_provided = rows(&[Some("a"), Some("a"), Some("b")], &[None, None, None]);
        assert_eq!(
            choose_session_key(&strong_provided),
            Some(SessionKey::Provided)
        );

        // Neither column repeats -> no session key found.
        let no_signal = rows(&[Some("a"), Some("b")], &[Some("x"), Some("y")]);
        assert_eq!(choose_session_key(&no_signal), None);

        // Equal (both have one repeated pair) -> provided wins ties.
        let tied = rows(&[Some("a"), Some("a")], &[Some("x"), Some("x")]);
        assert_eq!(choose_session_key(&tied), Some(SessionKey::Provided));
    }

    #[test]
    fn sample_trace_sessions_keeps_whole_sessions_deterministically() {
        let mut rows = vec![
            BasetenRow {
                prompt: "a1".into(),
                input_tokens: 1,
                output_tokens: 1,
                total_hashes: Vec::new(),
                provided_session_id: Some("a".into()),
                poor_man_session_id: None,
                duration_e2e_ms: None,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
                block_size: None,
                timestamp: Some(0.0),
                delay: None,
                output_length: 1,
            },
            BasetenRow {
                prompt: "a2".into(),
                input_tokens: 1,
                output_tokens: 1,
                total_hashes: Vec::new(),
                provided_session_id: Some("a".into()),
                poor_man_session_id: None,
                duration_e2e_ms: None,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
                block_size: None,
                timestamp: Some(10.0),
                delay: None,
                output_length: 1,
            },
            BasetenRow {
                prompt: "b1".into(),
                input_tokens: 1,
                output_tokens: 1,
                total_hashes: Vec::new(),
                provided_session_id: Some("b".into()),
                poor_man_session_id: None,
                duration_e2e_ms: None,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
                block_size: None,
                timestamp: Some(20.0),
                delay: None,
                output_length: 1,
            },
            BasetenRow {
                prompt: "c1".into(),
                input_tokens: 1,
                output_tokens: 1,
                total_hashes: Vec::new(),
                provided_session_id: Some("c".into()),
                poor_man_session_id: None,
                duration_e2e_ms: None,
                duration_ttft_ms: None,
                cached_tokens_reference: None,
                block_size: None,
                timestamp: Some(30.0),
                delay: None,
                output_length: 1,
            },
        ];
        sample_trace_sessions(
            &mut rows,
            Some(SessionKey::Provided),
            0.01,
            crate::rng::RngRoot::new(Some(7)),
        )
        .unwrap();
        // Tiny ratio keeps at least one whole session; both turns of "a" stay together
        // when that session is chosen.
        assert!(!rows.is_empty());
        let sessions: std::collections::HashSet<_> = rows
            .iter()
            .map(|row| row.provided_session_id.as_deref().unwrap())
            .collect();
        assert!(!sessions.is_empty());
        for session in &sessions {
            let count = rows
                .iter()
                .filter(|row| row.provided_session_id.as_deref() == Some(session))
                .count();
            if *session == "a" {
                assert_eq!(count, 2, "session a must stay intact when kept");
            }
        }
    }

    #[test]
    fn metadata_sampling_is_deterministic_and_keeps_whole_sessions() {
        let metadata = [
            TraceMetadata {
                timestamp: 0.0,
                session_id: Some("a".into()),
            },
            TraceMetadata {
                timestamp: 1.0,
                session_id: Some("a".into()),
            },
            TraceMetadata {
                timestamp: 2.0,
                session_id: Some("b".into()),
            },
            TraceMetadata {
                timestamp: 3.0,
                session_id: Some("b".into()),
            },
        ];
        let first =
            sampled_metadata_mask(&metadata, Some(0.5), true, RngRoot::new(Some(9))).unwrap();
        let second =
            sampled_metadata_mask(&metadata, Some(0.5), true, RngRoot::new(Some(9))).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0], first[1]);
        assert_eq!(first[2], first[3]);
        assert!(first.iter().any(|is_kept| *is_kept));
    }

    #[test]
    fn metadata_sampling_is_disabled_without_a_session_column() {
        let metadata = [
            TraceMetadata {
                timestamp: 0.0,
                session_id: None,
            },
            TraceMetadata {
                timestamp: 1.0,
                session_id: None,
            },
        ];
        assert_eq!(
            sampled_metadata_mask(&metadata, Some(0.0), false, RngRoot::new(Some(9))).unwrap(),
            vec![true, true]
        );
    }

    #[test]
    fn session_column_policy_covers_preference_configuration_fallback_and_absence() {
        assert_eq!(
            resolve_session_column(true, true, None).unwrap(),
            Some(COL_SESSION)
        );
        assert_eq!(
            resolve_session_column(true, true, Some(COL_POOR_MAN_SESSION)).unwrap(),
            Some(COL_POOR_MAN_SESSION)
        );
        assert_eq!(
            resolve_session_column(false, true, None).unwrap(),
            Some(COL_POOR_MAN_SESSION)
        );
        assert_eq!(
            resolve_session_column(true, false, Some(COL_POOR_MAN_SESSION)).unwrap(),
            Some(COL_SESSION)
        );
        assert_eq!(resolve_session_column(false, false, None).unwrap(), None);
        let error = resolve_session_column(true, true, Some("unknown")).unwrap_err();
        assert!(error.to_string().contains("got unknown"));
    }

    #[tokio::test]
    async fn max_rows_precedes_seeded_session_sampling_for_direct_and_registry_loads() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_session_policy_fixture(directory.path());
        let mut options = Map::new();
        options.insert("session_sample_ratio".into(), json!(0.5));
        let mut direct_config =
            LoadConfig::new(DatasetSource::Path(path.clone())).with_rng_root(RngRoot::new(Some(9)));
        direct_config.max_rows = Some(4);
        direct_config.options = options.clone();
        let direct_rows = BasetenTraceDatasetLoader
            .load(&direct_config)
            .await
            .unwrap();
        assert!(!direct_rows.is_empty());
        assert!(direct_rows.len() <= 4);
        let direct_groups = direct_rows
            .iter()
            .map(|row| row.group_key.as_deref().unwrap())
            .collect::<std::collections::HashSet<_>>();
        assert!(
            direct_groups
                .iter()
                .all(|group| matches!(*group, "p1" | "p2"))
        );
        for group in &direct_groups {
            assert_eq!(
                direct_rows
                    .iter()
                    .filter(|row| row.group_key.as_deref() == Some(*group))
                    .count(),
                2
            );
        }

        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(BasetenTraceDatasetLoader),
                Arc::new(BasetenTraceComposer),
            ))
            .unwrap();
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(9)));
        compose.format_options = options;
        let dataset = registry
            .build_dataset(
                Some("baseten_trace"),
                &direct_config,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), direct_groups.len());
        assert_eq!(
            dataset
                .conversations()
                .iter()
                .map(|conversation| conversation.turns.len())
                .sum::<usize>(),
            direct_rows.len()
        );
    }
}
