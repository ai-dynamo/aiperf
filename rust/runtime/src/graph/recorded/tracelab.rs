// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native TraceLab recorded-agent trace conversion.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};

use chrono::{DateTime, NaiveDateTime};
use flate2::read::MultiGzDecoder;
use serde_json::{Map, Value, json};

use crate::dataset::{DatasetSource, LoadConfig, TextTokenizer, load_raw_rows};
use crate::graph::input::GraphInputBundle;

use super::{RecordedTraceError, RecordedTraceInputConfig, compile_weka_trace_input};

const DEFAULT_BLOCK_SIZE: usize = 64;
const DEFAULT_MIN_SPAWN_MS: u64 = 10_000;

#[derive(Clone, Copy)]
struct TraceLabOptions {
    block_size: usize,
    is_subagent_join_enabled: bool,
    is_codex_join_enabled: bool,
    min_spawn_ms: u64,
}

impl Default for TraceLabOptions {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            is_subagent_join_enabled: true,
            is_codex_join_enabled: true,
            min_spawn_ms: DEFAULT_MIN_SPAWN_MS,
        }
    }
}

impl TraceLabOptions {
    fn parse(values: &Map<String, Value>) -> Result<Self, RecordedTraceError> {
        let mut options = Self::default();
        for (name, value) in values {
            match name.as_str() {
                "block_size" => {
                    options.block_size = value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            RecordedTraceError(
                                "TraceLab block_size must be a positive integer".into(),
                            )
                        })?;
                }
                "subagent_join" => {
                    options.is_subagent_join_enabled = value.as_bool().ok_or_else(|| {
                        RecordedTraceError("TraceLab subagent_join must be boolean".into())
                    })?;
                }
                "codex_subagent_join" => {
                    options.is_codex_join_enabled = value.as_bool().ok_or_else(|| {
                        RecordedTraceError("TraceLab codex_subagent_join must be boolean".into())
                    })?;
                }
                "min_spawn_ms" => {
                    options.min_spawn_ms = value.as_u64().ok_or_else(|| {
                        RecordedTraceError(
                            "TraceLab min_spawn_ms must be a non-negative integer".into(),
                        )
                    })?;
                }
                other => {
                    return Err(RecordedTraceError(format!(
                        "tracelab Graph-IR input does not support loader option {other:?}"
                    )));
                }
            }
        }
        Ok(options)
    }
}

#[derive(Clone)]
struct Session {
    id: String,
    rows: Vec<Value>,
}

struct TimedRound<'a> {
    submitted: f64,
    api_time: Option<f64>,
    row: &'a Value,
}

#[derive(Clone)]
struct Spawn {
    parent_id: String,
    child_id: String,
    start: f64,
    end: f64,
    duration_ms: u64,
}

struct HashIdMinter {
    next: i64,
}

impl HashIdMinter {
    fn new() -> Self {
        Self { next: 1 }
    }

    fn take(&mut self, count: usize) -> Result<Vec<i64>, RecordedTraceError> {
        let count = i64::try_from(count).map_err(|_| {
            RecordedTraceError("TraceLab hash-id count exceeds the signed identifier space".into())
        })?;
        let first = self.next;
        let end = first
            .checked_add(count)
            .ok_or_else(|| RecordedTraceError("TraceLab hash-id namespace is exhausted".into()))?;
        self.next = end;
        Ok((first..end).collect())
    }
}

/// Parse, convert, select, and lower one TraceLab corpus through native WEKA Graph-IR.
pub async fn compile_tracelab_trace_input(
    mut config: RecordedTraceInputConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphInputBundle, RecordedTraceError> {
    config.validate()?;
    let options = TraceLabOptions::parse(&config.load.options)?;
    let rows = load_rows(&config.load).await?;
    let traces = convert_rows(rows, options)?;
    let trace_order = traces
        .iter()
        .enumerate()
        .filter_map(|(index, trace)| {
            trace
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_owned(), index))
        })
        .collect::<HashMap<_, _>>();
    config.load.source = DatasetSource::Inline(Value::Array(traces));
    config.load.options.clear();
    let mut bundle = compile_weka_trace_input(config, tokenizer).await?;
    bundle.programs.sort_by_key(|program| {
        trace_order
            .get(&program.profiling.trace.id)
            .copied()
            .unwrap_or(usize::MAX)
    });
    bundle.metadata.format = "tracelab".into();
    Ok(bundle)
}

async fn load_rows(config: &LoadConfig) -> Result<Vec<Value>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) => {
            if path.is_dir() {
                return Err(RecordedTraceError(format!(
                    "TraceLab source {} must be a JSONL file, not a directory",
                    path.display()
                )));
            }
            let file = File::open(path).map_err(|error| {
                RecordedTraceError(format!(
                    "Cannot read TraceLab file {}: {error}",
                    path.display()
                ))
            })?;
            let label = path.display().to_string();
            if path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case("gz"))
            {
                parse_json_lines(BufReader::new(MultiGzDecoder::new(file)), &label)
            } else {
                parse_json_lines(BufReader::new(file), &label)
            }
        }
        DatasetSource::Bytes(bytes) if bytes.starts_with(&[0x1f, 0x8b]) => parse_json_lines(
            BufReader::new(MultiGzDecoder::new(Cursor::new(bytes))),
            "in-memory TraceLab gzip",
        ),
        DatasetSource::Bytes(bytes) => parse_json_lines(
            BufReader::new(Cursor::new(bytes)),
            "in-memory TraceLab JSONL",
        ),
        DatasetSource::Inline(Value::Array(values)) => Ok(values.clone()),
        DatasetSource::Inline(value) => Ok(vec![value.clone()]),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| rows.into_iter().map(|row| row.value).collect())
            .map_err(Into::into),
    }
}

fn parse_json_lines(
    mut reader: impl BufRead,
    label: &str,
) -> Result<Vec<Value>, RecordedTraceError> {
    let mut rows = Vec::new();
    let mut bytes = Vec::new();
    let mut line_number = 0_usize;
    loop {
        bytes.clear();
        let read = reader.read_until(b'\n', &mut bytes).map_err(|error| {
            RecordedTraceError(format!("Cannot read TraceLab file {label}: {error}"))
        })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        let line = std::str::from_utf8(&bytes).map_err(|error| {
            RecordedTraceError(format!("Cannot read TraceLab file {label}: {error}"))
        })?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = serde_json::from_str(line).map_err(|error| {
            RecordedTraceError(format!(
                "Invalid JSON in TraceLab file {label} at line {line_number}: {error}"
            ))
        })?;
        rows.push(row);
    }
    Ok(rows)
}

fn convert_rows(
    rows: Vec<Value>,
    options: TraceLabOptions,
) -> Result<Vec<Value>, RecordedTraceError> {
    let sessions = group_sessions(rows);
    if sessions.is_empty() {
        return Err(RecordedTraceError(
            "TraceLab source contains no sessions".into(),
        ));
    }
    let links = if options.is_subagent_join_enabled {
        build_join_index(&sessions, options)?
    } else {
        HashMap::new()
    };
    let session_ids = sessions
        .iter()
        .map(|session| session.id.as_str())
        .collect::<HashSet<_>>();
    let mut children = HashMap::<String, Vec<Spawn>>::new();
    for session in &sessions {
        let Some(spawn) = links.get(&session.id) else {
            continue;
        };
        if session_ids.contains(spawn.parent_id.as_str()) && !links.contains_key(&spawn.parent_id) {
            children
                .entry(spawn.parent_id.clone())
                .or_default()
                .push(spawn.clone());
        }
    }
    let nested = children
        .values()
        .flatten()
        .map(|spawn| spawn.child_id.as_str())
        .collect::<HashSet<_>>();
    let rows_by_id = sessions
        .iter()
        .map(|session| (session.id.as_str(), session.rows.as_slice()))
        .collect::<HashMap<_, _>>();
    let mut placed = HashSet::new();
    let mut traces = Vec::new();
    let mut trace_ids = HashSet::new();
    for session in &sessions {
        if nested.contains(session.id.as_str()) {
            continue;
        }
        let child_rows = children
            .get(&session.id)
            .into_iter()
            .flatten()
            .filter_map(|spawn| {
                rows_by_id
                    .get(spawn.child_id.as_str())
                    .map(|rows| (spawn, *rows))
            })
            .collect::<Vec<_>>();
        if let Some(trace) = build_trace(session, &child_rows, options.block_size, &mut placed)
            .map_err(|error| session_conversion_error(&session.id, error))?
        {
            insert_trace(trace, &mut trace_ids, &mut traces)?;
        }
    }
    for session in &sessions {
        if nested.contains(session.id.as_str())
            && !placed.contains(session.id.as_str())
            && let Some(trace) = build_trace(session, &[], options.block_size, &mut placed)
                .map_err(|error| session_conversion_error(&session.id, error))?
        {
            insert_trace(trace, &mut trace_ids, &mut traces)?;
        }
    }
    if traces.is_empty() {
        return Err(RecordedTraceError(
            "TraceLab source contains no dated sessions".into(),
        ));
    }
    Ok(traces)
}

fn session_conversion_error(session_id: &str, error: RecordedTraceError) -> RecordedTraceError {
    RecordedTraceError(format!(
        "converting TraceLab session {session_id:?}: {error}"
    ))
}

fn insert_trace(
    trace: Value,
    ids: &mut HashSet<String>,
    traces: &mut Vec<Value>,
) -> Result<(), RecordedTraceError> {
    let id = trace["id"].as_str().unwrap_or_default().to_string();
    if !ids.insert(id.clone()) {
        return Err(RecordedTraceError(format!(
            "TraceLab source contains duplicate converted trace id {id:?}"
        )));
    }
    traces.push(trace);
    Ok(())
}

fn group_sessions(rows: Vec<Value>) -> Vec<Session> {
    let mut sessions = Vec::<Session>::new();
    let mut positions = HashMap::<String, usize>::new();
    for row in rows {
        let Some(id) = row
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
        else {
            continue;
        };
        let position = match positions.get(&id).copied() {
            Some(position) => position,
            None => {
                let position = sessions.len();
                positions.insert(id.clone(), position);
                sessions.push(Session {
                    id: id.clone(),
                    rows: Vec::new(),
                });
                position
            }
        };
        sessions[position].rows.push(row);
    }
    sessions
}

fn parse_timestamp(value: &str) -> Result<f64, RecordedTraceError> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(timestamp.timestamp_micros() as f64 / 1_000_000.0);
    }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").map_err(|error| {
        RecordedTraceError(format!("invalid TraceLab timestamp {value:?}: {error}"))
    })?;
    Ok(naive.and_utc().timestamp_micros() as f64 / 1_000_000.0)
}

fn timestamps<'a>(values: impl Iterator<Item = &'a Value>) -> Result<Vec<f64>, RecordedTraceError> {
    values
        .filter_map(|value| value.as_str())
        .map(parse_timestamp)
        .collect()
}

fn round_timing(row: &Value) -> Result<Option<(f64, Option<f64>)>, RecordedTraceError> {
    let events = row
        .get("timing_events")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let inputs = timestamps(events.iter().filter_map(|event| {
        matches!(
            event.get("event_type").and_then(Value::as_str),
            Some("user_message" | "tool_result")
        )
        .then(|| event.get("timestamp"))
        .flatten()
    }))?;
    let outputs = timestamps(events.iter().filter_map(|event| {
        matches!(
            event.get("event_type").and_then(Value::as_str),
            Some("text" | "reasoning" | "tool_call")
        )
        .then(|| event.get("timestamp"))
        .flatten()
    }))?;
    let submitted = inputs
        .iter()
        .copied()
        .reduce(f64::max)
        .or_else(|| outputs.iter().copied().reduce(f64::min));
    let Some(submitted) = submitted else {
        return Ok(None);
    };
    let api_time = outputs
        .iter()
        .copied()
        .reduce(f64::max)
        .map(|last| (last - submitted).max(0.0));
    Ok(Some((submitted, api_time)))
}

fn ordered_rounds(rows: &[Value]) -> Result<Vec<TimedRound<'_>>, RecordedTraceError> {
    let mut timed = Vec::new();
    for row in rows {
        if let Some((submitted, api_time)) = round_timing(row)? {
            timed.push(TimedRound {
                submitted,
                api_time,
                row,
            });
        }
    }
    timed.sort_by(|left, right| {
        left.submitted.total_cmp(&right.submitted).then_with(|| {
            left.row
                .get("round_index")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                .cmp(
                    &right
                        .row
                        .get("round_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                )
        })
    });
    Ok(timed)
}

fn session_span(rows: &[Value]) -> Result<Option<(f64, f64)>, RecordedTraceError> {
    let mut stamps = Vec::new();
    for row in rows {
        if let Some(events) = row.get("timing_events").and_then(Value::as_array) {
            stamps.extend(timestamps(
                events.iter().filter_map(|event| event.get("timestamp")),
            )?);
        }
        if let Some(tools) = row.get("tools").and_then(Value::as_array) {
            stamps.extend(timestamps(tools.iter().flat_map(|tool| {
                [tool.get("emitted_at"), tool.get("result_at")]
                    .into_iter()
                    .flatten()
            }))?);
        }
    }
    Ok(stamps
        .iter()
        .copied()
        .reduce(f64::min)
        .zip(stamps.into_iter().reduce(f64::max)))
}

fn identity(session: &Session) -> (Option<&str>, Option<&str>) {
    let first = session.rows.first();
    (
        first
            .and_then(|row| row.get("user"))
            .and_then(Value::as_str),
        first
            .and_then(|row| row.get("project"))
            .and_then(Value::as_str),
    )
}

fn build_join_index(
    sessions: &[Session],
    options: TraceLabOptions,
) -> Result<HashMap<String, Spawn>, RecordedTraceError> {
    let mut spans = HashMap::<&str, (f64, f64)>::new();
    for session in sessions {
        if let Some(span) = session_span(&session.rows).map_err(|error| {
            RecordedTraceError(format!(
                "converting TraceLab session {:?}: {error}",
                session.id
            ))
        })? {
            spans.insert(&session.id, span);
        }
    }
    let mut windows = Vec::<(f64, f64, &str, u64)>::new();
    for session in sessions {
        let is_codex = session
            .rows
            .first()
            .and_then(|row| row.get("provider"))
            .and_then(Value::as_str)
            == Some("codex");
        if is_codex {
            if !options.is_codex_join_enabled {
                continue;
            }
            let mut spawns = Vec::new();
            let mut waits = Vec::new();
            for tool in session
                .rows
                .iter()
                .filter_map(|row| row.get("tools").and_then(Value::as_array))
                .flatten()
            {
                match tool.get("tool_name").and_then(Value::as_str) {
                    Some("spawn_agent") => {
                        if let Some(value) = tool.get("emitted_at").and_then(Value::as_str) {
                            spawns.push(
                                parse_timestamp(value).map_err(|error| {
                                    session_conversion_error(&session.id, error)
                                })?,
                            );
                        }
                    }
                    Some("wait_agent") => {
                        if let Some(value) = tool.get("result_at").and_then(Value::as_str) {
                            waits.push(
                                parse_timestamp(value).map_err(|error| {
                                    session_conversion_error(&session.id, error)
                                })?,
                            );
                        }
                    }
                    _ => {}
                }
            }
            if let Some((start, end)) = spawns
                .into_iter()
                .reduce(f64::min)
                .zip(waits.into_iter().reduce(f64::max))
                .filter(|(start, end)| end > start)
            {
                windows.push((start, end, &session.id, ((end - start) * 1000.0) as u64));
            }
            continue;
        }
        for tool in session
            .rows
            .iter()
            .filter_map(|row| row.get("tools").and_then(Value::as_array))
            .flatten()
        {
            if !matches!(
                tool.get("tool_name").and_then(Value::as_str),
                Some("Agent" | "Task")
            ) {
                continue;
            }
            let Some(duration_ms) = tool.get("tool_wall_latency_ms").and_then(Value::as_u64) else {
                continue;
            };
            if duration_ms < options.min_spawn_ms {
                continue;
            }
            let Some(start) = tool.get("emitted_at").and_then(Value::as_str) else {
                continue;
            };
            let Some(end) = tool.get("result_at").and_then(Value::as_str) else {
                continue;
            };
            windows.push((
                parse_timestamp(start)
                    .map_err(|error| session_conversion_error(&session.id, error))?,
                parse_timestamp(end)
                    .map_err(|error| session_conversion_error(&session.id, error))?,
                &session.id,
                duration_ms,
            ));
        }
    }
    let sessions_by_id = sessions
        .iter()
        .map(|session| (session.id.as_str(), session))
        .collect::<HashMap<_, _>>();
    let mut links = HashMap::new();
    for child in sessions {
        let Some((child_start, child_end)) = spans.get(child.id.as_str()).copied() else {
            continue;
        };
        let mut candidates = windows
            .iter()
            .filter(|(start, end, parent_id, _)| {
                *parent_id != child.id
                    && child_start >= *start
                    && child_end <= *end
                    && sessions_by_id
                        .get(*parent_id)
                        .is_some_and(|parent| identity(parent) == identity(child))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            (left.1 - left.0)
                .total_cmp(&(right.1 - right.0))
                .then_with(|| left.0.total_cmp(&right.0))
                .then_with(|| left.2.cmp(right.2))
        });
        if let Some((start, end, parent_id, duration_ms)) = candidates.first() {
            links.insert(
                child.id.clone(),
                Spawn {
                    parent_id: (*parent_id).to_string(),
                    child_id: child.id.clone(),
                    start: *start,
                    end: *end,
                    duration_ms: *duration_ms,
                },
            );
        }
    }
    Ok(links)
}

fn hash_chains(
    rounds: &[TimedRound<'_>],
    block_size: usize,
    minter: &mut HashIdMinter,
) -> Result<Vec<Vec<i64>>, RecordedTraceError> {
    let mut previous = Vec::new();
    rounds
        .iter()
        .map(|round| {
            let total = round
                .row
                .get("input_tokens_total")
                .and_then(Value::as_u64)
                .map(usize::try_from)
                .transpose()
                .map_err(|_| {
                    RecordedTraceError(
                        "TraceLab input_tokens_total exceeds the platform integer range".into(),
                    )
                })?
                .unwrap_or(0);
            let prefix = round
                .row
                .get("prefix_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .min(total as u64);
            let prefix = usize::try_from(prefix).map_err(|_| {
                RecordedTraceError(
                    "TraceLab prefix_tokens exceeds the platform integer range".into(),
                )
            })?;
            let blocks = total / block_size;
            let reused = (prefix / block_size).min(previous.len()).min(blocks);
            let mut current = previous[..reused].to_vec();
            current.extend(minter.take(blocks - reused)?);
            previous = current.clone();
            Ok(current)
        })
        .collect()
}

fn build_requests(
    rounds: &[TimedRound<'_>],
    hashes: Vec<Vec<i64>>,
    base: f64,
) -> (Vec<Value>, Vec<String>) {
    let mut requests = Vec::new();
    let mut models = Vec::<String>::new();
    let mut previous_end: Option<f64> = None;
    for (round, hash_ids) in rounds.iter().zip(hashes) {
        let model = round
            .row
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        if !models.contains(&model) {
            models.push(model.clone());
        }
        let timestamp = round.submitted - base;
        let think_time = previous_end.map(|end| (timestamp - end).max(0.0));
        let output = round
            .row
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(
                round
                    .row
                    .get("reasoning_output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            )
            .max(1);
        let input = round
            .row
            .get("input_tokens_total")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .max(1);
        let emitted_tool = round
            .row
            .get("timing_events")
            .and_then(Value::as_array)
            .is_some_and(|events| {
                events.iter().any(|event| {
                    event.get("event_type").and_then(Value::as_str) == Some("tool_call")
                })
            });
        requests.push(json!({
            "t": rounded(timestamp),
            "type": "n",
            "model": model,
            "in": input,
            "out": output,
            "hash_ids": hash_ids,
            "input_types": if round.row.get("first_input_event_type").and_then(Value::as_str) == Some("tool_result") { json!(["tool_result"]) } else { json!(["text"]) },
            "output_types": ["text"],
            "stop": if emitted_tool { "tool_use" } else { "end_turn" },
            "api_time": round.api_time.map(rounded),
            "think_time": think_time.map(rounded),
        }));
        previous_end = Some(timestamp + round.api_time.unwrap_or(0.0));
    }
    (requests, models)
}

fn build_trace(
    session: &Session,
    children: &[(&Spawn, &[Value])],
    block_size: usize,
    placed: &mut HashSet<String>,
) -> Result<Option<Value>, RecordedTraceError> {
    let rounds = ordered_rounds(&session.rows)?;
    let Some(first) = rounds.first() else {
        return Ok(None);
    };
    let base = first.submitted;
    let mut minter = HashIdMinter::new();
    let hashes = hash_chains(&rounds, block_size, &mut minter)?;
    let (mut requests, mut models) = build_requests(&rounds, hashes, base);
    let mut pending = HashMap::<usize, Vec<Value>>::new();
    let mut subagents = 0_usize;
    for (spawn, child_rows) in children {
        let child_rounds = ordered_rounds(child_rows)?;
        let Some(child_first) = child_rounds.first() else {
            continue;
        };
        let child_hashes = hash_chains(&child_rounds, block_size, &mut minter)?;
        let (mut inner, child_models) = build_requests(&child_rounds, child_hashes, base);
        let marker_time = spawn.start - base;
        for request in &mut inner {
            if request["t"].as_f64().is_some_and(|time| time < marker_time) {
                request["t"] = json!(rounded(marker_time));
            }
        }
        inner[0]["think_time"] = Value::Null;
        let child_end = child_rounds
            .iter()
            .map(|round| round.submitted + round.api_time.unwrap_or(0.0))
            .reduce(f64::max)
            .unwrap_or(child_first.submitted);
        let total_tokens = inner
            .iter()
            .map(|request| {
                request["in"].as_u64().unwrap_or(0) + request["out"].as_u64().unwrap_or(0)
            })
            .sum::<u64>();
        let tool_use_count = child_rows
            .iter()
            .filter_map(|row| row.get("tools").and_then(Value::as_array))
            .map(Vec::len)
            .sum::<usize>();
        let marker = json!({
            "t": rounded(marker_time),
            "type": "subagent",
            "agent_id": safe_id(&spawn.child_id),
            "subagent_type": child_models.first().map(String::as_str).unwrap_or("unknown"),
            "duration_ms": spawn.duration_ms,
            "total_tokens": total_tokens,
            "tool_use_count": tool_use_count,
            "status": if child_end <= spawn.end { "completed" } else { "incomplete" },
            "requests": inner,
            "models": child_models,
            "tool_tokens": 0,
            "system_tokens": 0,
        });
        let anchor = requests
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, request)| {
                request["t"]
                    .as_f64()
                    .is_some_and(|time| time <= marker_time)
                    .then_some(index)
            });
        let Some(anchor) = anchor else {
            continue;
        };
        if let Some(child_models) = marker.get("models").and_then(Value::as_array) {
            for model in child_models.iter().filter_map(Value::as_str) {
                if !models.iter().any(|known| known == model) {
                    models.push(model.to_string());
                }
            }
        }
        pending.entry(anchor).or_default().push(marker);
        placed.insert(spawn.child_id.clone());
        subagents += 1;
    }
    if !pending.is_empty() {
        let mut merged = Vec::with_capacity(requests.len() + subagents);
        for (index, request) in requests.into_iter().enumerate() {
            merged.push(request);
            if let Some(mut entries) = pending.remove(&index) {
                entries.sort_by(|left, right| {
                    left["t"]
                        .as_f64()
                        .unwrap_or(0.0)
                        .total_cmp(&right["t"].as_f64().unwrap_or(0.0))
                });
                merged.extend(entries);
            }
        }
        requests = merged;
    }
    let normal = requests.iter().filter(|request| request["type"] == "n");
    let rounds_count = normal.clone().count();
    let input_tokens = normal
        .clone()
        .map(|request| request["in"].as_u64().unwrap_or(0))
        .sum::<u64>();
    let output_tokens = normal
        .map(|request| request["out"].as_u64().unwrap_or(0))
        .sum::<u64>();
    Ok(Some(json!({
        "id": safe_id(&session.id),
        "models": models,
        "block_size": block_size,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
        "totals": {
            "rounds": rounds_count,
            "subagents": subagents,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "source": "tracelab"
        }
    })))
}

fn safe_id(value: &str) -> String {
    let mut output = String::new();
    let mut previous_was_unsafe = false;
    for character in value.chars() {
        let is_safe = character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-');
        if is_safe {
            previous_was_unsafe = false;
            output.push(character);
        } else if !previous_was_unsafe {
            previous_was_unsafe = true;
            output.push('_');
        }
        if output.len() >= 150 {
            break;
        }
    }
    while output.len() > 150 {
        output.pop();
    }
    output
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use serde_json::{Value, json};

    use crate::dataset::{DatasetSource, LoadConfig, TiktokenTokenizer};
    use crate::graph::recorded::{PromptCorpus, RecordedTraceInputConfig};

    use super::*;

    fn event(kind: &str, second: u32) -> Value {
        json!({
            "event_type": kind,
            "timestamp": format!("2026-05-31T12:00:{second:02}Z")
        })
    }

    fn row(session: &str, round: u32, submitted: u32, input: u64, prefix: u64) -> Value {
        json!({
            "provider": "claude",
            "project": "project-a",
            "user": "user-a",
            "session_id": session,
            "round_index": round,
            "model": if round == 1 { "model-b" } else { "model-a" },
            "input_tokens_total": input,
            "prefix_tokens": prefix,
            "newly_append_tokens": input.saturating_sub(prefix),
            "output_tokens": 10,
            "reasoning_output_tokens": if round == 1 { 5 } else { 0 },
            "first_input_event_type": if round == 1 { "tool_result" } else { "user_message" },
            "timing_events": [event(if round == 1 { "tool_result" } else { "user_message" }, submitted), event(if round == 1 { "tool_call" } else { "text" }, submitted + 1)],
            "tools": []
        })
    }

    fn tool(name: &str, emitted: u32, result: u32) -> Value {
        json!({
            "tool_name": name,
            "emitted_at": format!("2026-05-31T12:00:{emitted:02}Z"),
            "result_at": format!("2026-05-31T12:00:{result:02}Z"),
            "tool_wall_latency_ms": (result - emitted) * 1000
        })
    }

    fn config(
        source: DatasetSource,
        options: serde_json::Map<String, Value>,
    ) -> RecordedTraceInputConfig {
        let mut load = LoadConfig::new(source);
        load.options = options;
        RecordedTraceInputConfig {
            load,
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: None,
            prompt_corpus: PromptCorpus::Coding,
            content_root_seed: 7,
        }
    }

    #[test]
    fn conversion_preserves_timing_counts_models_and_compacting_hash_chains() {
        let rows = vec![
            row("claude:s", 2, 20, 128, 64),
            row("claude:s", 0, 0, 256, 0),
            row("claude:s", 1, 10, 384, 256),
        ];
        let converted = convert_rows(rows, TraceLabOptions::default()).unwrap();
        let trace = &converted[0];
        assert_eq!(trace["id"], "claude_s");
        assert_eq!(trace["models"], json!(["model-a", "model-b"]));
        assert_eq!(trace["block_size"], 64);
        let requests = trace["requests"].as_array().unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|r| r["t"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![0.0, 10.0, 20.0]
        );
        assert_eq!(requests[0]["hash_ids"], json!([1, 2, 3, 4]));
        assert_eq!(requests[1]["hash_ids"], json!([1, 2, 3, 4, 5, 6]));
        assert_eq!(requests[2]["hash_ids"], json!([1, 7]));
        assert_eq!(requests[1]["out"], 15);
        assert_eq!(requests[1]["input_types"], json!(["tool_result"]));
        assert_eq!(requests[1]["stop"], "tool_use");
        assert_eq!(requests[1]["think_time"], 9.0);
        assert_eq!(requests[1]["api_time"], 1.0);
        assert_eq!(safe_id("claude:/unsafe id"), "claude_unsafe_id");
        assert_eq!(safe_id(&"a".repeat(151)).len(), 150);

        let output_only = json!({
            "timing_events": [event("text", 4)]
        });
        let (_, api_time) = round_timing(&output_only).unwrap().unwrap();
        assert_eq!(api_time, Some(0.0));

        let mut wider_blocks = TraceLabOptions::default();
        wider_blocks.block_size = 128;
        let converted = convert_rows(vec![row("claude:wide", 0, 0, 256, 0)], wider_blocks).unwrap();
        assert_eq!(converted[0]["requests"][0]["hash_ids"], json!([1, 2]));
    }

    #[test]
    fn conversion_recovers_claude_and_codex_children_and_keeps_grandchildren() {
        let mut parent = vec![
            row("claude:root", 0, 0, 128, 0),
            row("claude:root", 1, 4, 128, 64),
        ];
        parent[1]["tools"] = json!([tool("Agent", 5, 50), tool("Task", 8, 30)]);
        let child = vec![
            row("claude:child", 0, 10, 128, 0),
            row("claude:child", 1, 20, 192, 128),
        ];
        let mut mid = row("claude:mid", 0, 11, 128, 0);
        mid["tools"] = json!([tool("Agent", 12, 25)]);
        let leaf = row("claude:leaf", 0, 15, 128, 0);
        let mut codex_parent = vec![
            row("codex:root", 0, 0, 128, 0),
            row("codex:root", 1, 55, 128, 64),
        ];
        codex_parent[0]["provider"] = json!("codex");
        codex_parent[1]["provider"] = json!("codex");
        codex_parent[0]["project"] = json!("project-codex");
        codex_parent[1]["project"] = json!("project-codex");
        codex_parent[0]["tools"] =
            json!([{"tool_name":"spawn_agent","emitted_at":"2026-05-31T12:00:05Z"}]);
        codex_parent[1]["tools"] =
            json!([{"tool_name":"wait_agent","result_at":"2026-05-31T12:00:50Z"}]);
        let mut codex_child = row("codex:child", 0, 20, 128, 0);
        codex_child["provider"] = json!("codex");
        codex_child["project"] = json!("project-codex");

        let rows: Vec<Value> = parent
            .into_iter()
            .chain(child)
            .chain([mid, leaf])
            .chain(codex_parent)
            .chain([codex_child])
            .collect();
        let mut without_join = TraceLabOptions::default();
        without_join.is_subagent_join_enabled = false;
        let standalone = convert_rows(rows.clone(), without_join).unwrap();
        assert!(standalone.iter().any(|trace| trace["id"] == "claude_child"));
        assert!(standalone.iter().all(|trace| {
            trace["requests"]
                .as_array()
                .is_some_and(|requests| requests.iter().all(|entry| entry["type"] != "subagent"))
        }));

        let mut without_codex = TraceLabOptions::default();
        without_codex.is_codex_join_enabled = false;
        let claude_only = convert_rows(rows.clone(), without_codex).unwrap();
        assert!(claude_only.iter().any(|trace| trace["id"] == "codex_child"));
        assert!(
            !claude_only
                .iter()
                .any(|trace| trace["id"] == "claude_child")
        );

        let converted = convert_rows(rows, TraceLabOptions::default()).unwrap();
        let ids = converted
            .iter()
            .map(|trace| trace["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"claude_root"));
        assert!(ids.contains(&"claude_leaf"));
        assert!(ids.contains(&"codex_root"));
        assert!(!ids.contains(&"claude_child"));
        assert!(!ids.contains(&"codex_child"));
        let root = converted
            .iter()
            .find(|trace| trace["id"] == "claude_root")
            .unwrap();
        let nested = root["requests"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["type"] == "subagent")
            .unwrap();
        assert_eq!(nested["agent_id"], "claude_child");
        assert_eq!(nested["duration_ms"], 22_000);
        let codex = converted
            .iter()
            .find(|trace| trace["id"] == "codex_root")
            .unwrap();
        assert!(
            codex["requests"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry["agent_id"] == "codex_child")
        );
    }

    #[tokio::test]
    async fn compiler_reads_plain_and_gzip_and_reports_source_errors() {
        let directory = tempfile::tempdir().unwrap();
        let rows = [
            row("claude:z", 0, 0, 128, 0),
            row("claude:z", 1, 10, 192, 128),
            row("claude:a", 0, 20, 64, 0),
        ];
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap() + "\n")
            .collect::<String>();
        let plain = directory.path().join("trace.jsonl");
        std::fs::write(&plain, &bytes).unwrap();
        let gzip = directory.path().join("trace.jsonl.gz");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes.as_bytes()).unwrap();
        std::fs::write(&gzip, encoder.finish().unwrap()).unwrap();

        let plain_bundle = compile_tracelab_trace_input(
            config(DatasetSource::Path(plain), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
        let gzip_bundle = compile_tracelab_trace_input(
            config(DatasetSource::Path(gzip), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
        assert_eq!(plain_bundle.metadata.format, "tracelab");
        assert_eq!(plain_bundle.metadata.root_count, 2);
        assert_eq!(plain_bundle.metadata.node_count, 3);
        assert_eq!(plain_bundle.metadata, gzip_bundle.metadata);
        assert_eq!(
            plain_bundle
                .programs
                .iter()
                .map(|program| program.profiling.trace.id.as_str())
                .collect::<Vec<_>>(),
            ["claude_z", "claude_a"]
        );

        let empty = directory.path().join("empty.jsonl");
        std::fs::write(&empty, "").unwrap();
        let error = match compile_tracelab_trace_input(
            config(DatasetSource::Path(empty), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        {
            Ok(_) => panic!("empty TraceLab source must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("no sessions"));

        let bad = directory.path().join("bad.jsonl");
        std::fs::write(&bad, "{}\n{oops\n").unwrap();
        let error = match compile_tracelab_trace_input(
            config(DatasetSource::Path(bad), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        {
            Ok(_) => panic!("invalid TraceLab JSON must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("line 2"));

        let non_utf8 = directory.path().join("non-utf8.jsonl");
        std::fs::write(&non_utf8, b"{\"session_id\":\"s\"}\n\xff").unwrap();
        let error = match compile_tracelab_trace_input(
            config(DatasetSource::Path(non_utf8), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        {
            Ok(_) => panic!("non-UTF-8 TraceLab source must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Cannot read TraceLab file"));

        let corrupt_gzip = directory.path().join("corrupt.jsonl.gz");
        std::fs::write(&corrupt_gzip, [0x1f, 0x8b, 0x08, 0x00]).unwrap();
        let error = match compile_tracelab_trace_input(
            config(DatasetSource::Path(corrupt_gzip), Default::default()),
            &TiktokenTokenizer::builtin(),
        )
        .await
        {
            Ok(_) => panic!("corrupt TraceLab gzip must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Cannot read TraceLab file"));

        let mut zero_block_size = Map::new();
        zero_block_size.insert("block_size".into(), json!(0));
        let error = match compile_tracelab_trace_input(
            config(
                DatasetSource::Inline(row("claude:s", 0, 0, 64, 0)),
                zero_block_size,
            ),
            &TiktokenTokenizer::builtin(),
        )
        .await
        {
            Ok(_) => panic!("zero TraceLab block size must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("positive integer"));
    }
}
