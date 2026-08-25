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
use std::path::Path;

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
fn read_parquet_rows(path: &Path) -> Result<Vec<Value>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = std::fs::File::open(path).map_err(|error| {
        DatasetError::Validation(format!("failed to open {}: {error}", path.display()))
    })?;
    let reader = SerializedFileReader::new(file).map_err(|error| {
        DatasetError::Validation(format!(
            "failed to open {} as Parquet: {error}",
            path.display()
        ))
    })?;
    let rows = reader.get_row_iter(None).map_err(|error| {
        DatasetError::Validation(format!(
            "failed to read Parquet rows from {}: {error}",
            path.display()
        ))
    })?;
    rows.map(|row| {
        row.map(|row| row.to_json_value()).map_err(|error| {
            DatasetError::Validation(format!(
                "failed to decode a Parquet row from {}: {error}",
                path.display()
            ))
        })
    })
    .collect()
}

#[cfg(not(feature = "parquet"))]
fn read_parquet_rows(path: &Path) -> Result<Vec<Value>> {
    Err(DatasetError::Validation(format!(
        "baseten_trace dataset {} requires an aiperf runner built with the `parquet` feature",
        path.display()
    )))
}

#[cfg(feature = "parquet")]
fn parquet_schema_has_columns(path: &Path, columns: &[&str]) -> bool {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(reader) = SerializedFileReader::new(file) else {
        return false;
    };
    let names: std::collections::HashSet<&str> = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .iter()
        .map(|field| field.name())
        .collect();
    columns.iter().all(|column| names.contains(column))
}

#[cfg(not(feature = "parquet"))]
fn parquet_schema_has_columns(_path: &Path, _columns: &[&str]) -> bool {
    false
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
        if path.extension().and_then(|extension| extension.to_str()) != Some("parquet") {
            return false;
        }
        parquet_schema_has_columns(path, &required_columns())
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let (raw_rows, origin_label) = match &config.source {
            DatasetSource::Path(path) => {
                let rows = read_parquet_rows(path)?;
                (rows, path.display().to_string())
            }
            DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => {
                // Public / `--hf-dataset --hf-format baseten_trace` path: reuse the
                // shared remote parquet/jsonl acquisition, then apply baseten
                // replay normalization below.
                let rows = crate::dataset::loader::public::load_raw_rows(config).await?;
                let values = rows.into_iter().map(|row| row.value).collect();
                (values, config.source.label())
            }
            _ => {
                return Err(DatasetError::Validation(
                    "baseten_trace requires a Parquet file path, URL, or Hugging Face source"
                        .into(),
                ));
            }
        };
        let mut rows = raw_rows
            .into_iter()
            .map(|value| parse_row(&value, &origin_label))
            .collect::<Result<Vec<_>>>()?;

        let min_timestamp = rows
            .iter()
            .filter_map(|row| row.timestamp)
            .fold(f64::INFINITY, f64::min);
        let replay = ReplayOptions::from_options(&config.options)?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows.drain(..) {
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
            let wire = serde_json::to_vec(&row_to_value(&row)).map_err(DatasetError::from)?;
            out.push(RawRow {
                value: row_to_value(&row),
                wire: Some(Bytes::from(wire)),
                session_id: None,
                group_key: None,
                origin: RowOrigin::FileLine {
                    path: std::path::PathBuf::from(&origin_label),
                    line: 0,
                },
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
        let mut parsed: Vec<BasetenRow> = rows
            .into_iter()
            .map(|row| row_from_value(&row.value))
            .collect::<Result<Vec<_>>>()?;

        // Open-loop idle-gap reflow runs on every timed row, before grouping,
        // so a sparse (sampled) trace does not idle through dead air.
        // Closed-loop defers the reflow until after back-pressure clears
        // continuation timestamps, so it only touches session-start turns.
        if replay.open_loop && replay.max_idle_gap_cap_ms.is_some() {
            apply_idle_gap_cap(&mut parsed, replay.max_idle_gap_cap_ms);
        }

        let session_key = choose_session_key(&parsed);
        if let Some(ratio) = replay.session_sample_ratio {
            sample_trace_sessions(&mut parsed, session_key, ratio, config.rng_root)?;
        }

        let mut groups: HashMap<String, Vec<BasetenRow>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        let mut next_ordinal: u64 = 0;
        for row in parsed.drain(..) {
            let session_id = match session_key {
                Some(SessionKey::Provided) => row.provided_session_id.clone(),
                Some(SessionKey::PoorMan) => row.poor_man_session_id.clone(),
                None => None,
            }
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
}
