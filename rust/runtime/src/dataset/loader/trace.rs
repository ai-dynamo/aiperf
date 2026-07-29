// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake, Bailian, BurstGPT, and SageMaker trace formats.
//!
//! Literal payload/message slices are
//! retained with [`RawValue`] rather than decoded and serialized again.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::endpoints::extract_payload;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use chrono::{DateTime, NaiveDateTime};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use smallvec::smallvec;

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::dataset::model::{Conversation, ConversationContextMode, SessionId, Turn};
use crate::dataset::segment::{Handle, Role, SegmentPool};
use crate::dataset::synthesis::{PrefixTraceSynthesizer, TraceSynthesisRecord, TraceSynthesizer};
use crate::dataset::tokenizer::TextTokenizer;

/// Mooncake JSONL loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct MooncakeTraceDatasetLoader;
/// Mooncake trace composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct MooncakeTraceComposer;
/// Alibaba Bailian JSONL loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct BailianTraceDatasetLoader;
/// Bailian trace composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct BailianTraceComposer;
/// BurstGPT CSV loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct BurstGptTraceDatasetLoader;
/// BurstGPT trace composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct BurstGptTraceComposer;
/// SageMaker Data Capture loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct SageMakerDataCaptureDatasetLoader;
/// SageMaker capture composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SageMakerDataCaptureComposer;

#[derive(Debug, Deserialize)]
struct MooncakeRaw<'a> {
    #[serde(default)]
    input_length: Option<u64>,
    #[serde(default)]
    text_input: Option<String>,
    #[serde(default, borrow)]
    messages: Option<&'a RawValue>,
    #[serde(default, borrow)]
    tools: Option<&'a RawValue>,
    #[serde(default, borrow)]
    payload: Option<&'a RawValue>,
    #[serde(default)]
    output_length: Option<u32>,
    #[serde(default)]
    hash_ids: Vec<i64>,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    delay: Option<f64>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default, borrow)]
    extra: Option<&'a RawValue>,
}

impl MooncakeRaw<'_> {
    fn validate(&self, origin: &impl std::fmt::Display) -> Result<()> {
        let modes = [
            self.input_length.is_some(),
            self.text_input.is_some(),
            self.messages.is_some(),
            self.payload.is_some(),
        ]
        .into_iter()
        .filter(|mode| *mode)
        .count();
        if modes != 1 {
            return Err(DatasetError::Validation(format!(
                "{origin}: mooncake requires exactly one of input_length, text_input, messages, or payload"
            )));
        }
        if !self.hash_ids.is_empty() && self.input_length.is_none() {
            return Err(DatasetError::Validation(format!(
                "{origin}: mooncake hash_ids require input_length mode"
            )));
        }
        if self.tools.is_some() && self.messages.is_none() {
            return Err(DatasetError::Validation(format!(
                "{origin}: mooncake tools require messages mode"
            )));
        }
        if self.input_length == Some(0) || self.output_length == Some(0) {
            return Err(DatasetError::Validation(format!(
                "{origin}: trace token lengths must be greater than zero"
            )));
        }
        validate_time(self.timestamp, "timestamp", origin)?;
        validate_nonnegative(self.delay, "delay", origin)?;
        validate_raw_array(self.messages, "messages", origin)?;
        validate_raw_array(self.tools, "tools", origin)?;
        validate_raw_object(self.payload, "payload", origin, true)?;
        validate_raw_object(self.extra, "extra", origin, false)?;
        Ok(())
    }
}

#[async_trait]
impl DatasetLoader for MooncakeTraceDatasetLoader {
    fn name(&self) -> &str {
        "mooncake_trace"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        let Some(value) = &probe.value else {
            return false;
        };
        if value.get("chat_id").is_some()
            || value.get("captureData").is_some()
            || value.get("turns").is_some()
            || is_speed_bench(value)
        {
            return false;
        }
        serde_json::from_value::<MooncakeProbe>(value.clone())
            .ok()
            .is_some_and(|probe| probe.valid())
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let mut rows = jsonl_rows(&config.source)?;
        rows.retain(|row| {
            row.value
                .get("timestamp")
                .and_then(Value::as_f64)
                .is_none_or(|timestamp| in_window(timestamp, config))
        });
        let mut retained = Vec::with_capacity(rows.len());
        for row in rows {
            let wire = row.wire.as_ref().ok_or_else(|| {
                DatasetError::Validation(format!(
                    "{}: trace wire bytes were not retained",
                    row.origin
                ))
            })?;
            let parsed: MooncakeRaw<'_> = serde_json::from_slice(wire).map_err(|error| {
                DatasetError::Validation(format!("{}: invalid mooncake row: {error}", row.origin))
            })?;
            parsed.validate(&row.origin)?;
            if config
                .max_input_tokens
                .is_some_and(|cap| parsed.input_length.is_some_and(|length| length > cap))
            {
                continue;
            }
            retained.push(row);
        }
        Ok(retained)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

#[derive(Debug, Deserialize)]
struct MooncakeProbe {
    #[serde(default)]
    input_length: Option<u64>,
    #[serde(default)]
    text_input: Option<String>,
    #[serde(default)]
    messages: Option<Value>,
    #[serde(default)]
    payload: Option<Value>,
}

impl MooncakeProbe {
    fn valid(&self) -> bool {
        [
            self.input_length.is_some(),
            self.text_input.is_some(),
            self.messages.is_some(),
            self.payload.is_some(),
        ]
        .into_iter()
        .filter(|mode| *mode)
        .count()
            == 1
    }
}

impl Composer for MooncakeTraceComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let rows = synthesize_mooncake_rows(rows, config)?;
        let block_size = format_usize(config, "block_size", 512)?;
        let mut prompt_generator = config.prompt_generator.create(tokenizer, config.rng_root)?;
        let mut finalizer = config.finalizer()?;
        let mut session_ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut conversations = Vec::<Conversation>::new();
        let mut positions = HashMap::<SessionId, usize>::new();
        let mut modes = Vec::<Option<MooncakeMode>>::new();
        let mut parents = Vec::<Option<Handle>>::new();

        for row in rows {
            let wire = row.wire.as_ref().expect("loader retains trace wire");
            let parsed: MooncakeRaw<'_> = serde_json::from_slice(wire)?;
            parsed.validate(&row.origin)?;
            let session_id = parsed
                .session_id
                .as_deref()
                .map(SessionId::from)
                .unwrap_or_else(|| session_ids.next_id());
            let position = match positions.get(&session_id).copied() {
                Some(position) => position,
                None => {
                    let position = conversations.len();
                    conversations.push(Conversation::new(session_id.clone()));
                    positions.insert(session_id, position);
                    modes.push(None);
                    parents.push(None);
                    position
                }
            };
            let mode = if parsed.payload.is_some() {
                MooncakeMode::Payload
            } else if parsed.messages.is_some() {
                MooncakeMode::Messages
            } else {
                MooncakeMode::Delta
            };
            if let Some(previous) = modes[position]
                && previous != mode
            {
                return Err(DatasetError::Validation(format!(
                    "mooncake session {:?} mixes {previous:?} and {mode:?} input modes",
                    conversations[position].session_id.as_str()
                )));
            }
            modes[position] = Some(mode);
            let trace_hash_ids = parsed
                .input_length
                .map(|_| {
                    segments.intern_trace_hash_ids(
                        parsed.hash_ids.clone().into_boxed_slice(),
                        block_size,
                    )
                })
                .transpose()?;
            let mut turn = Turn {
                timestamp_ms: parsed.timestamp,
                delay_ms: parsed.delay,
                max_tokens: cap_output(parsed.output_length, config.max_output_tokens),
                trace_hash_ids,
                ..Turn::default()
            };
            match mode {
                MooncakeMode::Payload => {
                    let payload = parsed.payload.expect("mode proves payload");
                    let value: Value = serde_json::from_str(payload.get())?;
                    turn.input_tokens = Some(input_tokens(&value, tokenizer)?);
                    let handle = segments.intern_raw(
                        parents[position],
                        Bytes::copy_from_slice(payload.get().as_bytes()),
                    )?;
                    turn.body = Turn::dispatch_body(Some(handle), None, &[]);
                    parents[position] = Some(handle);
                }
                MooncakeMode::Messages => {
                    let messages = parsed.messages.expect("mode proves messages");
                    turn.raw_messages = Some(segments.intern_raw(
                        parents[position],
                        Bytes::copy_from_slice(messages.get().as_bytes()),
                    )?);
                    parents[position] = turn.raw_messages;
                    if let Some(tools) = parsed.tools {
                        turn.tools = Some(segments.intern_raw(
                            parents[position],
                            Bytes::copy_from_slice(tools.get().as_bytes()),
                        )?);
                        parents[position] = turn.tools;
                    }
                    let body = message_payload(messages, parsed.tools)?;
                    let extracted = extract_payload(&body);
                    turn.tool_tokens = extracted
                        .tool_texts
                        .iter()
                        .try_fold(0_u64, |count, text| add_tokens(count, text, tokenizer))?;
                    turn.input_tokens = Some(input_tokens_excluding_tools(&extracted, tokenizer)?);
                }
                MooncakeMode::Delta => {
                    if config
                        .trace_prompt_storage
                        .stores_generated_prompt(&parsed.hash_ids)
                    {
                        let generated = match parsed.text_input {
                            Some(text) => {
                                let tokens = tokenizer.encode(&text)?;
                                crate::dataset::prompt::GeneratedPrompt { text, tokens }
                            }
                            None => prompt_generator.generate(
                                parsed.input_length.expect("delta input mode has length") as usize,
                                &parsed.hash_ids,
                                block_size,
                            )?,
                        };
                        turn.input_tokens = Some(generated.tokens.len() as u64);
                        let handle = segments.intern_text(
                            parents[position],
                            Role::from("user"),
                            Bytes::from(generated.text),
                            generated.tokens.into_boxed_slice(),
                        )?;
                        parents[position] = Some(handle);
                        turn.content.push(crate::dataset::model::ContentGroup {
                            kind: crate::dataset::model::MediaKind::Text,
                            name: "text".into(),
                            handles: smallvec![handle],
                            uuids: smallvec![],
                        });
                    } else {
                        turn.input_tokens = Some(
                            parsed
                                .input_length
                                .expect("hash-backed delta mode has an authored input length"),
                        );
                    }
                }
            }
            if let Some(extra) = parsed.extra {
                turn.extra_body = Some(
                    segments.intern_raw(None, Bytes::copy_from_slice(extra.get().as_bytes()))?,
                );
            }
            finalizer.finalize_turn(&mut turn)?;
            conversations[position].turns.push(turn);
        }
        for (conversation, mode) in conversations.iter_mut().zip(modes) {
            if matches!(mode, Some(MooncakeMode::Messages | MooncakeMode::Payload)) {
                conversation.context_mode =
                    Some(ConversationContextMode::MessageArrayWithResponses);
            }
        }
        Ok(conversations)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MooncakeMode {
    Delta,
    Messages,
    Payload,
}

#[derive(Debug, Clone, Deserialize)]
struct BailianRow {
    chat_id: i64,
    #[serde(default = "root_parent")]
    parent_chat_id: i64,
    timestamp: f64,
    input_length: u64,
    output_length: u32,
    #[serde(default)]
    turn: i64,
    #[serde(default)]
    hash_ids: Vec<i64>,
}

const fn root_parent() -> i64 {
    -1
}

impl BailianRow {
    fn parse(value: Value, origin: &impl std::fmt::Display) -> Result<Self> {
        let row: Self = serde_json::from_value(value).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid Bailian row: {error}"))
        })?;
        if row.input_length == 0 || row.output_length == 0 || !row.timestamp.is_finite() {
            return Err(DatasetError::Validation(format!(
                "{origin}: Bailian lengths must be positive and timestamp finite"
            )));
        }
        Ok(row)
    }
}

#[async_trait]
impl DatasetLoader for BailianTraceDatasetLoader {
    fn name(&self) -> &str {
        "bailian_trace"
    }
    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe.value.as_ref().is_some_and(|value| {
            value.get("chat_id").is_some()
                && value.get("parent_chat_id").is_some()
                && value.get("turn").is_some()
                && BailianRow::parse(value.clone(), &"probe").is_ok()
        })
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let mut rows = Vec::new();
        for row in jsonl_rows(&config.source)? {
            let parsed = BailianRow::parse(row.value.clone(), &row.origin)?;
            let timestamp_ms = parsed.timestamp * 1000.0;
            if !in_window(timestamp_ms, config)
                || config
                    .max_input_tokens
                    .is_some_and(|cap| parsed.input_length > cap)
            {
                continue;
            }
            rows.push(row);
        }
        Ok(rows)
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

impl Composer for BailianTraceComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let mut parsed = rows
            .into_iter()
            .map(|row| BailianRow::parse(row.value, &row.origin))
            .collect::<Result<Vec<_>>>()?;
        let lookup = parsed
            .iter()
            .map(|row| (row.chat_id, row.parent_chat_id))
            .collect::<HashMap<_, _>>();
        let mut root_cache = HashMap::new();
        let mut groups = Vec::<(i64, Vec<BailianRow>)>::new();
        let mut group_index = HashMap::new();
        for row in parsed.drain(..) {
            let root = bailian_root(row.chat_id, &lookup, &mut root_cache);
            let index = *group_index.entry(root).or_insert_with(|| {
                groups.push((root, Vec::new()));
                groups.len() - 1
            });
            groups[index].1.push(row);
        }
        for (_, rows) in &mut groups {
            rows.sort_by_key(|row| row.turn);
        }
        synthesize_bailian_groups(&mut groups, config)?;
        let block_size = format_usize(config, "block_size", 16)?;
        let mut generator = config.prompt_generator.create(tokenizer, config.rng_root)?;
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(groups.len());
        for (root, rows) in groups {
            let mut conversation = Conversation::new(root.to_string());
            let mut parent = None;
            for row in rows {
                let content = if config
                    .trace_prompt_storage
                    .stores_generated_prompt(&row.hash_ids)
                {
                    let prompt =
                        generator.generate(row.input_length as usize, &row.hash_ids, block_size)?;
                    let handle = segments.intern_text(
                        parent,
                        "user",
                        Bytes::from(prompt.text),
                        prompt.tokens.into_boxed_slice(),
                    )?;
                    parent = Some(handle);
                    smallvec![crate::dataset::model::ContentGroup {
                        kind: crate::dataset::model::MediaKind::Text,
                        name: "text".into(),
                        handles: smallvec![handle],
                        uuids: smallvec![],
                    }]
                } else {
                    smallvec![]
                };
                let trace_hash_ids = segments
                    .intern_trace_hash_ids(row.hash_ids.clone().into_boxed_slice(), block_size)?;
                let mut turn = Turn {
                    timestamp_ms: Some(row.timestamp * 1000.0),
                    max_tokens: cap_output(Some(row.output_length), config.max_output_tokens),
                    input_tokens: Some(row.input_length),
                    trace_hash_ids: Some(trace_hash_ids),
                    content,
                    ..Turn::default()
                };
                finalizer.finalize_turn(&mut turn)?;
                conversation.turns.push(turn);
            }
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

fn bailian_root(chat_id: i64, lookup: &HashMap<i64, i64>, cache: &mut HashMap<i64, i64>) -> i64 {
    if let Some(root) = cache.get(&chat_id) {
        return *root;
    }
    let mut current = chat_id;
    let mut path = Vec::new();
    let mut seen = HashSet::new();
    while let Some(parent) = lookup.get(&current).copied() {
        if parent == -1
            || parent == current
            || !lookup.contains_key(&parent)
            || !seen.insert(current)
        {
            break;
        }
        path.push(current);
        current = parent;
    }
    for id in path {
        cache.insert(id, current);
    }
    cache.insert(chat_id, current);
    current
}

#[derive(Debug, Clone, Deserialize)]
struct BurstRow {
    timestamp: f64,
    input_length: u64,
    output_length: u32,
}

#[async_trait]
impl DatasetLoader for BurstGptTraceDatasetLoader {
    fn name(&self) -> &str {
        "burst_gpt_trace"
    }
    fn can_load(&self, probe: &DatasetProbe) -> bool {
        let Some(path) = &probe.path else {
            return false;
        };
        if path.extension().and_then(|extension| extension.to_str()) != Some("csv") {
            return false;
        }
        csv::Reader::from_path(path).ok().is_some_and(|mut reader| {
            reader.headers().ok().is_some_and(|headers| {
                ["Timestamp", "Request tokens", "Response tokens"]
                    .into_iter()
                    .all(|required| headers.iter().any(|header| header == required))
            })
        })
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let path = match &config.source {
            DatasetSource::Path(path) => path,
            _ => {
                return Err(DatasetError::Validation(
                    "BurstGPT requires a CSV path".into(),
                ));
            }
        };
        let mut reader = csv::Reader::from_path(path).map_err(|error| {
            DatasetError::Validation(format!("{}: invalid CSV: {error}", path.display()))
        })?;
        let headers = reader
            .headers()
            .map_err(|error| {
                DatasetError::Validation(format!("{}: invalid CSV header: {error}", path.display()))
            })?
            .clone();
        let column = |name: &str| {
            headers
                .iter()
                .position(|header| header == name)
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{}: missing CSV column {name:?}",
                        path.display()
                    ))
                })
        };
        let timestamp_col = column("Timestamp")?;
        let input_col = column("Request tokens")?;
        let output_col = column("Response tokens")?;
        let mut rows = Vec::new();
        for (index, record) in reader.records().enumerate() {
            let record = record.map_err(|error| {
                DatasetError::Validation(format!("{}:{}: {error}", path.display(), index + 2))
            })?;
            let parsed = (|| {
                Some(BurstRow {
                    timestamp: record.get(timestamp_col)?.parse().ok()?,
                    input_length: record.get(input_col)?.parse().ok()?,
                    output_length: record.get(output_col)?.parse().ok()?,
                })
            })();
            let Some(parsed) = parsed else { continue };
            let timestamp_ms = parsed.timestamp * 1000.0;
            if parsed.input_length == 0
                || parsed.output_length == 0
                || !parsed.timestamp.is_finite()
                || !in_window(timestamp_ms, config)
                || config
                    .max_input_tokens
                    .is_some_and(|cap| parsed.input_length > cap)
            {
                continue;
            }
            rows.push(RawRow {
                value: serde_json::json!({
                    "timestamp": parsed.timestamp,
                    "input_length": parsed.input_length,
                    "output_length": parsed.output_length,
                }),
                wire: None,
                session_id: None,
                group_key: None,
                origin: RowOrigin::FileLine {
                    path: path.clone(),
                    line: index + 2,
                },
            });
        }
        Ok(rows)
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

impl Composer for BurstGptTraceComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let mut parsed = rows
            .into_iter()
            .map(|row| serde_json::from_value::<BurstRow>(row.value).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        synthesize_burst_rows(&mut parsed, config)?;
        let mut generator = config.prompt_generator.create(tokenizer, config.rng_root)?;
        let mut finalizer = config.finalizer()?;
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut conversations = Vec::with_capacity(parsed.len());
        for row in parsed {
            let prompt = generator.generate(row.input_length as usize, &[], 1)?;
            let handle = segments.intern_text(
                None,
                "user",
                Bytes::from(prompt.text),
                prompt.tokens.into_boxed_slice(),
            )?;
            let mut turn = Turn {
                timestamp_ms: Some(row.timestamp * 1000.0),
                max_tokens: cap_output(Some(row.output_length), config.max_output_tokens),
                input_tokens: Some(row.input_length),
                content: smallvec![crate::dataset::model::ContentGroup {
                    kind: crate::dataset::model::MediaKind::Text,
                    name: "text".into(),
                    handles: smallvec![handle],
                    uuids: smallvec![],
                }],
                ..Turn::default()
            };
            finalizer.finalize_turn(&mut turn)?;
            let mut conversation = Conversation::new(ids.next_id());
            conversation.turns.push(turn);
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

fn synthesize_mooncake_rows(rows: Vec<RawRow>, config: &ComposeConfig) -> Result<Vec<RawRow>> {
    let Some(synthesis) = config
        .trace_synthesis
        .as_ref()
        .filter(|synthesis| synthesis.has_structural_transform())
    else {
        return Ok(rows);
    };
    let mut positions = HashMap::<String, usize>::new();
    let mut groups = Vec::<Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        let key = row
            .value
            .get("session_id")
            .and_then(Value::as_str)
            .map_or_else(
                || format!("__aiperf_row_{index}"),
                |value| format!("session:{value}"),
            );
        let position = *positions.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[position].push(index);
    }
    let order = groups.into_iter().flatten().collect::<Vec<_>>();
    let mut records = order
        .iter()
        .map(|index| synthesis_record(&rows[*index].value))
        .collect::<Result<Vec<_>>>()?;
    PrefixTraceSynthesizer::new(synthesis.clone(), config.rng_root)?.synthesize(&mut records)?;

    let mut slots = rows.into_iter().map(Some).collect::<Vec<_>>();
    order
        .into_iter()
        .zip(records)
        .map(|(index, synthesized)| {
            let mut row = slots[index]
                .take()
                .expect("grouped Mooncake indices are unique");
            apply_synthesis_record(&mut row.value, &synthesized)?;
            row.wire = Some(Bytes::from(serde_json::to_vec(&row.value)?));
            Ok(row)
        })
        .collect()
}

fn synthesize_bailian_groups(
    groups: &mut [(i64, Vec<BailianRow>)],
    config: &ComposeConfig,
) -> Result<()> {
    let Some(synthesis) = config
        .trace_synthesis
        .as_ref()
        .filter(|synthesis| synthesis.has_structural_transform())
    else {
        return Ok(());
    };
    let positions = groups
        .iter()
        .enumerate()
        .flat_map(|(group, (_, rows))| (0..rows.len()).map(move |row| (group, row)))
        .collect::<Vec<_>>();
    let mut records = positions
        .iter()
        .map(|(group, row)| {
            let row = &groups[*group].1[*row];
            TraceSynthesisRecord {
                hash_ids: row.hash_ids.clone(),
                input_length: row.input_length,
                timestamp_ms: Some(row.timestamp * 1_000.0),
                output_length: Some(row.output_length),
            }
        })
        .collect::<Vec<_>>();
    PrefixTraceSynthesizer::new(synthesis.clone(), config.rng_root)?.synthesize(&mut records)?;
    for ((group, row), synthesized) in positions.into_iter().zip(records) {
        let row = &mut groups[group].1[row];
        row.hash_ids = synthesized.hash_ids;
        row.input_length = synthesized.input_length;
        row.timestamp = synthesized
            .timestamp_ms
            .expect("Bailian synthesis retains timestamps")
            / 1_000.0;
        row.output_length = synthesized
            .output_length
            .expect("Bailian synthesis retains output lengths");
    }
    Ok(())
}

fn synthesize_burst_rows(rows: &mut [BurstRow], config: &ComposeConfig) -> Result<()> {
    let Some(synthesis) = config
        .trace_synthesis
        .as_ref()
        .filter(|synthesis| synthesis.has_structural_transform())
    else {
        return Ok(());
    };
    let mut records = rows
        .iter()
        .map(|row| TraceSynthesisRecord {
            hash_ids: Vec::new(),
            input_length: row.input_length,
            timestamp_ms: Some(row.timestamp * 1_000.0),
            output_length: Some(row.output_length),
        })
        .collect::<Vec<_>>();
    PrefixTraceSynthesizer::new(synthesis.clone(), config.rng_root)?.synthesize(&mut records)?;
    for (row, synthesized) in rows.iter_mut().zip(records) {
        row.input_length = synthesized.input_length;
        row.timestamp = synthesized
            .timestamp_ms
            .expect("BurstGPT synthesis retains timestamps")
            / 1_000.0;
        row.output_length = synthesized
            .output_length
            .expect("BurstGPT synthesis retains output lengths");
    }
    Ok(())
}

fn synthesis_record(value: &Value) -> Result<TraceSynthesisRecord> {
    let object = value.as_object().ok_or_else(|| {
        DatasetError::Validation("Mooncake synthesis row must be an object".into())
    })?;
    let hash_ids = object
        .get("hash_ids")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value.as_i64().ok_or_else(|| {
                        DatasetError::Validation(
                            "Mooncake synthesis hash_ids must contain i64 values".into(),
                        )
                    })
                })
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(TraceSynthesisRecord {
        hash_ids,
        input_length: object
            .get("input_length")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        timestamp_ms: object.get("timestamp").and_then(Value::as_f64),
        output_length: object
            .get("output_length")
            .and_then(Value::as_u64)
            .map(|value| {
                u32::try_from(value).map_err(|_| {
                    DatasetError::Validation("Mooncake synthesis output_length exceeds u32".into())
                })
            })
            .transpose()?,
    })
}

fn apply_synthesis_record(value: &mut Value, record: &TraceSynthesisRecord) -> Result<()> {
    let object = value.as_object_mut().ok_or_else(|| {
        DatasetError::Validation("Mooncake synthesis row must be an object".into())
    })?;
    if object.contains_key("input_length") {
        object.insert("input_length".into(), Value::from(record.input_length));
    }
    if record.hash_ids.is_empty() {
        object.remove("hash_ids");
    } else {
        object.insert(
            "hash_ids".into(),
            Value::Array(record.hash_ids.iter().copied().map(Value::from).collect()),
        );
    }
    if object.contains_key("timestamp") {
        object.insert(
            "timestamp".into(),
            record.timestamp_ms.map_or(Value::Null, Value::from),
        );
    }
    if object.contains_key("output_length") {
        object.insert(
            "output_length".into(),
            record.output_length.map_or(Value::Null, Value::from),
        );
    }
    Ok(())
}

#[async_trait]
impl DatasetLoader for SageMakerDataCaptureDatasetLoader {
    fn name(&self) -> &str {
        "sagemaker_data_capture"
    }
    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe.value.as_ref().is_some_and(|value| {
            value.get("captureData").is_some() && value.get("eventMetadata").is_some()
        }) || probe
            .path
            .as_deref()
            .is_some_and(directory_starts_with_capture)
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let raw_rows = match &config.source {
            DatasetSource::Path(path) if path.is_dir() => {
                let mut files = recursive_jsonl(path)?;
                files.sort();
                let mut rows = Vec::new();
                for file in files {
                    rows.extend(jsonl_rows(&DatasetSource::Path(file))?);
                }
                rows
            }
            source => jsonl_rows(source)?,
        };
        let mut decoded = Vec::<(f64, RawRow)>::new();
        for row in raw_rows {
            let capture = parse_capture(&row.value, &row.origin)?;
            decoded.push((
                capture.timestamp_ms,
                RawRow {
                    value: capture.value,
                    wire: Some(capture.input_wire),
                    session_id: None,
                    group_key: None,
                    origin: row.origin,
                },
            ));
        }
        let start = decoded
            .iter()
            .map(|(timestamp, _)| *timestamp)
            .reduce(f64::min)
            .unwrap_or(0.0);
        for (timestamp, row) in &mut decoded {
            *timestamp -= start;
            row.value
                .as_object_mut()
                .expect("capture parser returns object")
                .insert("timestamp".into(), Value::from(*timestamp));
        }
        decoded.retain(|(timestamp, row)| {
            in_window(*timestamp, config)
                && !config.max_input_tokens.is_some_and(|cap| {
                    row.value
                        .get("input_length")
                        .and_then(Value::as_u64)
                        .is_some_and(|length| length > cap)
                })
        });
        decoded.sort_by(|left, right| left.0.total_cmp(&right.0));
        Ok(decoded.into_iter().map(|(_, row)| row).collect())
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::MessageArrayWithResponses)
    }
}

impl Composer for SageMakerDataCaptureComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let wire = row.wire.ok_or_else(|| {
                DatasetError::Validation(format!("{}: missing captured input bytes", row.origin))
            })?;
            let raw: CapturedInputRaw<'_> = serde_json::from_slice(&wire)?;
            let messages = raw.messages.ok_or_else(|| {
                DatasetError::Validation(format!("{}: captured input has no messages", row.origin))
            })?;
            validate_raw_array(Some(messages), "messages", &row.origin)?;
            validate_raw_array(raw.tools, "tools", &row.origin)?;
            let messages_handle =
                segments.intern_raw(None, Bytes::copy_from_slice(messages.get().as_bytes()))?;
            let tools_handle = raw
                .tools
                .map(|tools| {
                    segments.intern_raw(
                        Some(messages_handle),
                        Bytes::copy_from_slice(tools.get().as_bytes()),
                    )
                })
                .transpose()?;
            let body: Value = serde_json::from_slice(&wire)?;
            let extracted = extract_payload(&body);
            let input_tokens = row
                .value
                .get("input_length")
                .and_then(Value::as_u64)
                .unwrap_or(input_tokens_excluding_tools(&extracted, tokenizer)?);
            let tool_tokens = extracted
                .tool_texts
                .iter()
                .try_fold(0_u64, |count, text| add_tokens(count, text, tokenizer))?;
            let mut turn = Turn {
                timestamp_ms: row.value.get("timestamp").and_then(Value::as_f64),
                max_tokens: cap_output(
                    row.value
                        .get("output_length")
                        .and_then(Value::as_u64)
                        .and_then(|value| u32::try_from(value).ok()),
                    config.max_output_tokens,
                ),
                input_tokens: Some(input_tokens),
                tool_tokens,
                raw_messages: Some(messages_handle),
                tools: tools_handle,
                ..Turn::default()
            };
            finalizer.finalize_turn(&mut turn)?;
            let mut conversation = Conversation::new(ids.next_id());
            conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
            conversation.turns.push(turn);
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

#[derive(Debug, Deserialize)]
struct CapturedInputRaw<'a> {
    #[serde(default, borrow)]
    messages: Option<&'a RawValue>,
    #[serde(default, borrow)]
    tools: Option<&'a RawValue>,
}

struct ParsedCapture {
    timestamp_ms: f64,
    input_wire: Bytes,
    value: Value,
}

fn parse_capture(value: &Value, origin: &impl std::fmt::Display) -> Result<ParsedCapture> {
    let event = value
        .get("eventMetadata")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            DatasetError::Validation(format!("{origin}: missing eventMetadata object"))
        })?;
    let timestamp = event
        .get("inferenceTime")
        .and_then(Value::as_str)
        .ok_or_else(|| DatasetError::Validation(format!("{origin}: missing inferenceTime")))?;
    let timestamp_ms = parse_iso8601_ms(timestamp).map_err(|message| {
        DatasetError::Validation(format!("{origin}: invalid inferenceTime: {message}"))
    })?;
    let capture = value
        .get("captureData")
        .and_then(Value::as_object)
        .ok_or_else(|| DatasetError::Validation(format!("{origin}: missing captureData object")))?;
    let input_wire = decode_capture_entry(capture.get("endpointInput"), origin, "endpointInput")?
        .ok_or_else(|| {
        DatasetError::Validation(format!("{origin}: unsupported endpointInput"))
    })?;
    let input: Value = serde_json::from_slice(&input_wire).map_err(|error| {
        DatasetError::Validation(format!("{origin}: invalid captured request JSON: {error}"))
    })?;
    if !input.get("messages").is_some_and(Value::is_array) {
        return Err(DatasetError::Validation(format!(
            "{origin}: captured request has no messages array"
        )));
    }
    let output = decode_capture_entry(capture.get("endpointOutput"), origin, "endpointOutput")?
        .and_then(|wire| serde_json::from_slice::<Value>(&wire).ok());
    let input_length = output
        .as_ref()
        .and_then(|value| value.get("usage"))
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let output_length = input
        .get("max_tokens")
        .or_else(|| input.get("max_completion_tokens"))
        .and_then(Value::as_u64);
    Ok(ParsedCapture {
        timestamp_ms,
        input_wire,
        value: serde_json::json!({
            "timestamp": timestamp_ms,
            "input_length": input_length,
            "output_length": output_length,
        }),
    })
}

fn decode_capture_entry(
    entry: Option<&Value>,
    origin: &impl std::fmt::Display,
    field: &str,
) -> Result<Option<Bytes>> {
    let Some(entry) = entry.and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(data) = entry.get("data").and_then(Value::as_str) else {
        return Ok(None);
    };
    match entry
        .get("encoding")
        .and_then(Value::as_str)
        .unwrap_or("BASE64")
    {
        "JSON" => Ok(Some(Bytes::copy_from_slice(data.as_bytes()))),
        "BASE64" => STANDARD
            .decode(data)
            .map(Bytes::from)
            .map(Some)
            .map_err(|error| {
                DatasetError::Validation(format!("{origin}: invalid {field} base64: {error}"))
            }),
        _ => Ok(None),
    }
}

fn parse_iso8601_ms(value: &str) -> std::result::Result<f64, String> {
    if let Ok(datetime) = DateTime::parse_from_rfc3339(value) {
        return Ok(datetime.timestamp_micros() as f64 / 1_000.0);
    }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f")
        .map_err(|error| error.to_string())?;
    Ok(naive.and_utc().timestamp_micros() as f64 / 1_000.0)
}

fn directory_starts_with_capture(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let Some(file) = recursive_jsonl(path)
        .ok()
        .and_then(|files| files.into_iter().next())
    else {
        return false;
    };
    let Ok(bytes) = std::fs::read(file) else {
        return false;
    };
    crate::dataset::loader::first_json_value(&bytes)
        .ok()
        .flatten()
        .is_some_and(|value| {
            value.get("captureData").is_some() && value.get("eventMetadata").is_some()
        })
}

fn recursive_jsonl(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn message_payload(messages: &RawValue, tools: Option<&RawValue>) -> Result<Value> {
    let messages: Value = serde_json::from_str(messages.get())?;
    let mut object = Map::new();
    object.insert("messages".into(), messages);
    if let Some(tools) = tools {
        object.insert("tools".into(), serde_json::from_str(tools.get())?);
    }
    Ok(Value::Object(object))
}

fn input_tokens(value: &Value, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    let extracted = extract_payload(value);
    let mut count = extracted.pretokenised_token_count;
    for text in extracted.texts {
        count = add_tokens(count, &text, tokenizer)?;
    }
    Ok(count)
}

fn input_tokens_excluding_tools(
    extracted: &crate::endpoints::ExtractedPayload,
    tokenizer: &dyn TextTokenizer,
) -> Result<u64> {
    let mut tool_counts = HashMap::<&str, usize>::new();
    for text in &extracted.tool_texts {
        *tool_counts.entry(text).or_default() += 1;
    }
    let mut count = extracted.pretokenised_token_count;
    for text in &extracted.texts {
        if let Some(remaining) = tool_counts.get_mut(text.as_str())
            && *remaining > 0
        {
            *remaining -= 1;
            continue;
        }
        count = add_tokens(count, text, tokenizer)?;
    }
    Ok(count)
}

fn add_tokens(count: u64, text: &str, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    count
        .checked_add(tokenizer.encode(text)?.len() as u64)
        .ok_or_else(|| DatasetError::Validation("trace token count overflowed u64".into()))
}

fn validate_raw_array(
    value: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
) -> Result<()> {
    if let Some(value) = value {
        let decoded: Value = serde_json::from_str(value.get())?;
        if !decoded.is_array() || decoded.as_array().is_some_and(Vec::is_empty) {
            return Err(DatasetError::Validation(format!(
                "{origin}: {field} must be a non-empty array"
            )));
        }
    }
    Ok(())
}

fn validate_raw_object(
    value: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    nonempty: bool,
) -> Result<()> {
    if let Some(value) = value {
        let decoded: Value = serde_json::from_str(value.get())?;
        if !decoded.is_object() || (nonempty && decoded.as_object().is_some_and(Map::is_empty)) {
            return Err(DatasetError::Validation(format!(
                "{origin}: {field} must be {}JSON object",
                if nonempty { "a non-empty " } else { "a " }
            )));
        }
    }
    Ok(())
}

fn validate_time(value: Option<f64>, field: &str, origin: &impl std::fmt::Display) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite()) {
        return Err(DatasetError::Validation(format!(
            "{origin}: {field} must be finite"
        )));
    }
    Ok(())
}

fn validate_nonnegative(
    value: Option<f64>,
    field: &str,
    origin: &impl std::fmt::Display,
) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(DatasetError::Validation(format!(
            "{origin}: {field} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn in_window(timestamp: f64, config: &LoadConfig) -> bool {
    config
        .start_offset_ms
        .is_none_or(|start| timestamp >= start)
        && config.end_offset_ms.is_none_or(|end| timestamp <= end)
}

fn cap_output(value: Option<u32>, cap: Option<u32>) -> Option<u32> {
    match (value, cap) {
        (Some(value), Some(cap)) => Some(value.min(cap)),
        (value, _) => value,
    }
}

fn format_usize(config: &ComposeConfig, key: &str, default: usize) -> Result<usize> {
    config
        .format_options
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    DatasetError::Validation(format!("trace option {key} must be positive usize"))
                })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn is_speed_bench(value: &Value) -> bool {
    value.get("task_id").is_some()
        && value.get("prompt").is_some()
        && value.get("sampling_params").is_some()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::dataset::request::{ConversationSession, EndpointRequestMaterializer};
    use crate::dataset::tokenizer::TiktokenTokenizer;
    use crate::dataset::{Overrides, RequestMaterializer};
    use crate::endpoints::{ChatEndpoint, CreditPhase, EndpointConfig, ModelEndpoint};

    async fn build(
        loader: Arc<dyn DatasetLoader>,
        composer: Arc<dyn Composer>,
        source: DatasetSource,
    ) -> crate::dataset::Dataset {
        let mut registry = LoaderRegistry::new();
        let name = loader.name().to_string();
        registry
            .register(DatasetFormatRegistration::new(loader, composer))
            .unwrap();
        registry
            .build_dataset(
                Some(&name),
                &LoadConfig::new(source),
                &ComposeConfig::new("model", RngRoot::new(Some(9))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn mooncake_preserves_raw_payload_and_message_mode_and_hash_prefixes() {
        let payload_wire =
            b"{\"session_id\":\"raw\",\"payload\":{ \"prompt\" : \"x\", \"max_tokens\":2 }}\n";
        let dataset = build(
            Arc::new(MooncakeTraceDatasetLoader),
            Arc::new(MooncakeTraceComposer),
            DatasetSource::Bytes(Bytes::copy_from_slice(payload_wire)),
        )
        .await;
        let mut session =
            ConversationSession::new(Arc::new(dataset), SessionId::from("raw")).unwrap();
        session.advance_to(0).unwrap();
        let request = EndpointRequestMaterializer
            .materialize(
                &session,
                &ChatEndpoint,
                &ModelEndpoint {
                    primary_model_name: "m".into(),
                    endpoint: EndpointConfig::default(),
                },
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(
            request.body,
            b"{ \"prompt\" : \"x\", \"max_tokens\":2 }"[..]
        );

        let synthetic = build(
            Arc::new(MooncakeTraceDatasetLoader),
            Arc::new(MooncakeTraceComposer),
            DatasetSource::Inline(json!([
                {"session_id":"a","input_length":1023,"hash_ids":[1,2],"output_length":2},
                {"session_id":"b","input_length":1024,"hash_ids":[1,3],"output_length":2}
            ])),
        )
        .await;
        let first = synthetic.conversations()[0].turns[0].content[0].handles[0];
        let second = synthetic.conversations()[1].turns[0].content[0].handles[0];
        assert_ne!(
            synthetic.segments().id(first).unwrap(),
            synthetic.segments().id(second).unwrap()
        );
        let first_turn = &synthetic.conversations()[0].turns[0];
        assert_eq!(first_turn.input_tokens, Some(1023));
        assert!(matches!(
            synthetic
                .segments()
                .get(first_turn.trace_hash_ids.unwrap())
                .unwrap(),
            crate::dataset::Payload::TraceHashIds { hash_ids, block_size }
                if hash_ids.as_ref() == [1, 2] && *block_size == 512
        ));
    }

    #[tokio::test]
    async fn mooncake_can_store_hash_identity_without_generated_prompt_material() {
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(MooncakeTraceDatasetLoader),
                Arc::new(MooncakeTraceComposer),
            ))
            .unwrap();
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(9)));
        compose.trace_prompt_storage = Arc::new(crate::dataset::HashIdentityTracePromptStorage);
        let dataset = registry
            .build_dataset(
                Some("mooncake_trace"),
                &LoadConfig::new(DatasetSource::Inline(json!([
                    {"input_length":128,"hash_ids":[11],"output_length":4}
                ]))),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.input_tokens, Some(128));
        assert!(turn.content.is_empty());
        assert!(matches!(
            dataset.segments().get(turn.trace_hash_ids.unwrap()).unwrap(),
            crate::dataset::Payload::TraceHashIds { hash_ids, block_size: 512 }
                if hash_ids.as_ref() == [11]
        ));
    }

    #[tokio::test]
    async fn mooncake_applies_native_prefix_time_and_length_synthesis() {
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(MooncakeTraceDatasetLoader),
                Arc::new(MooncakeTraceComposer),
            ))
            .unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!([
            {"session_id":"a","timestamp":100.0,"input_length":10,"hash_ids":[1,2],"output_length":2},
            {"session_id":"b","timestamp":201.0,"input_length":10,"hash_ids":[1,3],"output_length":3}
        ])));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(7)));
        compose
            .format_options
            .insert("block_size".into(), Value::from(4));
        compose.trace_synthesis = Some(crate::dataset::TraceSynthesisConfig {
            speedup_ratio: 2.0,
            prefix_len_multiplier: 2.0,
            prompt_len_multiplier: 1.5,
            output_len_multiplier: 1.5,
            block_size: 4,
            ..crate::dataset::TraceSynthesisConfig::default()
        });

        let dataset = registry
            .build_dataset(
                Some("mooncake_trace"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();

        assert_eq!(dataset.conversations().len(), 2);
        let first = &dataset.conversations()[0].turns[0];
        let second = &dataset.conversations()[1].turns[0];
        assert_eq!(first.input_tokens, Some(17));
        assert_eq!(second.input_tokens, Some(17));
        assert_eq!(first.timestamp_ms, Some(50.0));
        assert_eq!(second.timestamp_ms, Some(100.0));
        assert_eq!(first.max_tokens, Some(3));
        assert_eq!(second.max_tokens, Some(4));
        assert!(matches!(
            dataset
                .segments()
                .get(first.trace_hash_ids.unwrap())
                .unwrap(),
            crate::dataset::Payload::TraceHashIds { block_size: 4, .. }
        ));
    }

    #[tokio::test]
    async fn bailian_groups_parent_chain_and_sorts_turns() {
        let dataset = build(
            Arc::new(BailianTraceDatasetLoader),
            Arc::new(BailianTraceComposer),
            DatasetSource::Inline(json!([
                {"chat_id":2,"parent_chat_id":1,"timestamp":2.0,"input_length":4,"output_length":2,"turn":2,"hash_ids":[4]},
                {"chat_id":1,"parent_chat_id":-1,"timestamp":1.0,"input_length":4,"output_length":2,"turn":1,"hash_ids":[3]}
            ])),
        )
        .await;
        assert_eq!(dataset.conversations().len(), 1);
        assert_eq!(dataset.conversations()[0].session_id.as_str(), "1");
        assert_eq!(
            dataset.conversations()[0].turns[0].timestamp_ms,
            Some(1000.0)
        );
    }

    #[test]
    fn burst_gpt_can_load_detects_csv_header_not_json() {
        // Structural detection must key on the CSV header columns, never on
        // JSON-parsing the first line (the CSV header is not JSON). A canonical
        // BurstGPT header with extra columns is recognized; a JSONL file, a CSV
        // missing a required column, and a value-less probe are all rejected.
        let directory = tempfile::tempdir().unwrap();
        let burst = directory.path().join("burst_gpt.csv");
        std::fs::write(
            &burst,
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n0.0,ChatGPT,472,18,490,Conversation log\n",
        )
        .unwrap();
        let probe = DatasetProbe {
            value: None,
            path: Some(burst.clone()),
        };
        assert!(BurstGptTraceDatasetLoader.can_load(&probe));

        let missing_column = directory.path().join("partial.csv");
        std::fs::write(&missing_column, "Timestamp,Request tokens\n0.0,472\n").unwrap();
        assert!(!BurstGptTraceDatasetLoader.can_load(&DatasetProbe {
            value: None,
            path: Some(missing_column),
        }));

        let jsonl = directory.path().join("prompts.jsonl");
        std::fs::write(&jsonl, "{\"text\":\"hi\"}\n").unwrap();
        assert!(!BurstGptTraceDatasetLoader.can_load(&DatasetProbe {
            value: None,
            path: Some(jsonl),
        }));

        assert!(!BurstGptTraceDatasetLoader.can_load(&DatasetProbe {
            value: Some(json!({"timestamp": 1.0})),
            path: None,
        }));
    }

    #[tokio::test]
    async fn burst_gpt_parses_csv_skips_invalid_rows_and_converts_seconds() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("burst.csv");
        std::fs::write(
            &path,
            "Timestamp,Request tokens,Response tokens,Model\n1.5,4,3,gpt\nbad,row,data,gpt\n2.0,0,2,gpt\n",
        )
        .unwrap();
        let dataset = build(
            Arc::new(BurstGptTraceDatasetLoader),
            Arc::new(BurstGptTraceComposer),
            DatasetSource::Path(path),
        )
        .await;
        assert_eq!(dataset.conversations().len(), 1);
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.timestamp_ms, Some(1500.0));
        assert_eq!(turn.input_tokens, Some(4));
        assert_eq!(turn.max_tokens, Some(3));
        assert_eq!(
            dataset
                .segments()
                .get(turn.content[0].handles[0])
                .unwrap()
                .token_count(),
            Some(4)
        );
    }

    #[tokio::test]
    async fn sagemaker_decodes_base64_zero_aligns_and_keeps_messages() {
        let request = r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":9}"#;
        let response = r#"{"usage":{"prompt_tokens":4}}"#;
        let source = DatasetSource::Inline(json!([{
            "captureData": {
                "endpointInput": {"data": STANDARD.encode(request), "encoding":"BASE64"},
                "endpointOutput": {"data": response, "encoding":"JSON"}
            },
            "eventMetadata": {"eventId":"e", "inferenceTime":"2026-04-29T00:03:18Z"}
        }]));
        let dataset = build(
            Arc::new(SageMakerDataCaptureDatasetLoader),
            Arc::new(SageMakerDataCaptureComposer),
            source,
        )
        .await;
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.timestamp_ms, Some(0.0));
        assert_eq!(turn.input_tokens, Some(4));
        assert_eq!(turn.max_tokens, Some(9));
        assert!(turn.raw_messages.is_some());
    }
}
