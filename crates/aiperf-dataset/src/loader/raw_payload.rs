// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verbatim raw-payload JSONL and AIPerf `inputs.json` loaders.
//!
//! Raw JSONL rows retain the exact trimmed line bytes. `inputs.json` uses
//! `serde_json::value::RawValue`, retaining each nested payload object's exact
//! source slice rather than decoding and reserializing it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;

use aiperf_endpoints::extract_payload;

use crate::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::error::{DatasetError, Result};
use crate::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::model::{Conversation, ConversationContextMode, ModelId, SessionId, Turn};
use crate::segment::{Role, SegmentPool};
use crate::tokenizer::TextTokenizer;

/// Loader for exact JSON request objects, one row per request.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawPayloadDatasetLoader;

/// Loader for `{"data":[{"session_id", "payloads":[...]}]}` files.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputsJsonPayloadLoader;

/// Shared composer for both exact-payload formats.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawPayloadComposer;

#[async_trait]
impl DatasetLoader for RawPayloadDatasetLoader {
    fn name(&self) -> &str {
        "raw_payload"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        if let Some(value) = &probe.value {
            return is_raw_payload(value);
        }
        probe.path.as_deref().is_some_and(directory_has_raw_payload)
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        match &config.source {
            DatasetSource::Path(path) if path.is_dir() => load_raw_directory(path),
            source => {
                let mut rows = jsonl_rows(source)?;
                for (index, row) in rows.iter_mut().enumerate() {
                    validate_raw_payload(&row.value, &row.origin)?;
                    row.group_key = Some(format!("row:{index}"));
                }
                Ok(rows)
            }
        }
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }

    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::MessageArrayWithResponses)
    }
}

#[async_trait]
impl DatasetLoader for InputsJsonPayloadLoader {
    fn name(&self) -> &str {
        "inputs_json"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe.value.as_ref().is_some_and(is_inputs_json)
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let (bytes, path) = source_bytes(&config.source)?;
        parse_inputs_json(&bytes, path.as_deref())
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }

    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::MessageArrayWithResponses)
    }
}

impl Composer for RawPayloadComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let mut state = config.state(tokenizer, segments)?;
        let mut generator = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut conversations = Vec::<Conversation>::new();
        let mut positions = HashMap::<String, usize>::new();
        let mut parents = Vec::<Option<crate::Handle>>::new();

        for (row_index, row) in rows.into_iter().enumerate() {
            validate_payload_object(&row.value, &row.origin)?;
            let group = row
                .session_id
                .as_ref()
                .map(|id| format!("session:{}", id.as_str()))
                .or(row.group_key.clone())
                .unwrap_or_else(|| format!("row:{row_index}"));
            let position = match positions.get(&group).copied() {
                Some(position) => position,
                None => {
                    let session_id = row
                        .session_id
                        .clone()
                        .unwrap_or_else(|| generator.next_id());
                    let mut conversation = Conversation::new(session_id);
                    conversation.context_mode =
                        Some(ConversationContextMode::MessageArrayWithResponses);
                    let position = conversations.len();
                    conversations.push(conversation);
                    parents.push(None);
                    positions.insert(group, position);
                    position
                }
            };
            let wire = row.wire.ok_or_else(|| {
                DatasetError::Validation(format!(
                    "{}: raw payload loader did not retain exact wire bytes",
                    row.origin
                ))
            })?;
            let handle = state.segments.intern_raw(parents[position], wire)?;
            parents[position] = Some(handle);
            let mut turn = Turn {
                role: Some(Role::from("user")),
                model: row
                    .value
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ModelId::from),
                max_tokens: raw_max_tokens(&row.value),
                streaming: row.value.get("stream").and_then(Value::as_bool),
                raw_payload: Some(handle),
                input_tokens: raw_input_tokens(&row.value, state.tokenizer)?,
                ..Turn::default()
            };
            state.finalize_turn(&mut turn)?;
            conversations[position].turns.push(turn);
        }
        Ok(conversations)
    }
}

fn is_raw_payload(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.contains_key("conversation_id") || object.get("data").is_some_and(Value::is_array) {
        return false;
    }
    if object.contains_key("question_id") && object.contains_key("category") {
        return false;
    }
    object.get("messages").is_some_and(Value::is_array)
}

fn validate_raw_payload(value: &Value, origin: &impl std::fmt::Display) -> Result<()> {
    validate_payload_object(value, origin)?;
    if !value.get("messages").is_some_and(Value::is_array) {
        return Err(DatasetError::Validation(format!(
            "{origin}: raw_payload row is missing the required messages array"
        )));
    }
    Ok(())
}

fn validate_payload_object(value: &Value, origin: &impl std::fmt::Display) -> Result<()> {
    if !value.is_object() {
        return Err(DatasetError::Validation(format!(
            "{origin}: payload must be a JSON object"
        )));
    }
    Ok(())
}

fn raw_input_tokens(value: &Value, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    let extracted = extract_payload(value);
    let mut count = extracted.pretokenised_token_count;
    for text in extracted.texts {
        count = count
            .checked_add(tokenizer.encode(&text)?.len() as u64)
            .ok_or_else(|| {
                DatasetError::Validation("raw input token count overflowed u64".into())
            })?;
    }
    Ok(count)
}

fn directory_has_raw_payload(directory: &Path) -> bool {
    sorted_jsonl_files(directory).into_iter().any(|path| {
        std::fs::read(path)
            .ok()
            .and_then(|bytes| {
                bytes
                    .split(|byte| *byte == b'\n')
                    .map(crate::loader::trim_ascii)
                    .find(|line| !line.is_empty())
                    .and_then(|line| serde_json::from_slice::<Value>(line).ok())
            })
            .as_ref()
            .is_some_and(is_raw_payload)
    })
}

fn load_raw_directory(directory: &Path) -> Result<Vec<RawRow>> {
    let mut rows = Vec::new();
    for (file_index, path) in sorted_jsonl_files(directory).into_iter().enumerate() {
        let source = DatasetSource::Path(path.clone());
        let mut file_rows = jsonl_rows(&source)?;
        for row in &mut file_rows {
            validate_raw_payload(&row.value, &row.origin)?;
            row.group_key = Some(format!("file:{file_index}"));
        }
        rows.extend(file_rows);
    }
    Ok(rows)
}

fn sorted_jsonl_files(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();
    paths
}

fn is_inputs_json(value: &Value) -> bool {
    value
        .get("data")
        .and_then(Value::as_array)
        .and_then(|sessions| sessions.first())
        .and_then(Value::as_object)
        .and_then(|session| session.get("payloads"))
        .is_some_and(Value::is_array)
}

#[derive(Deserialize)]
struct InputsFileRaw {
    data: Vec<InputsSessionRaw>,
}

#[derive(Deserialize)]
struct InputsSessionRaw {
    session_id: String,
    payloads: Vec<Box<RawValue>>,
}

fn parse_inputs_json(bytes: &[u8], path: Option<&Path>) -> Result<Vec<RawRow>> {
    let parsed: InputsFileRaw = serde_json::from_slice(bytes).map_err(|error| {
        DatasetError::Validation(format!(
            "{}: invalid inputs.json: {error}",
            path.map_or_else(
                || "in-memory input".into(),
                |path| path.display().to_string()
            )
        ))
    })?;
    let mut session_ids = HashSet::new();
    let mut rows = Vec::new();
    for (session_index, session) in parsed.data.into_iter().enumerate() {
        if session.session_id.is_empty() {
            return Err(DatasetError::Validation(format!(
                "data[{session_index}] has an empty session_id"
            )));
        }
        if !session_ids.insert(session.session_id.clone()) {
            return Err(DatasetError::DuplicateSession(session.session_id));
        }
        if session.payloads.is_empty() {
            return Err(DatasetError::Validation(format!(
                "data[{session_index}].payloads must not be empty"
            )));
        }
        for (payload_index, payload) in session.payloads.into_iter().enumerate() {
            let origin = RowOrigin::JsonPointer {
                path: path.map(Path::to_path_buf),
                pointer: format!("/data/{session_index}/payloads/{payload_index}"),
            };
            let value: Value = serde_json::from_str(payload.get()).map_err(|error| {
                DatasetError::Validation(format!("{origin}: invalid payload: {error}"))
            })?;
            validate_payload_object(&value, &origin)?;
            rows.push(RawRow {
                value,
                wire: Some(Bytes::copy_from_slice(payload.get().as_bytes())),
                session_id: Some(SessionId::from(session.session_id.as_str())),
                group_key: None,
                origin,
            });
        }
    }
    Ok(rows)
}

fn source_bytes(source: &DatasetSource) -> Result<(Bytes, Option<PathBuf>)> {
    match source {
        DatasetSource::Path(path) => Ok((Bytes::from(std::fs::read(path)?), Some(path.clone()))),
        DatasetSource::Bytes(bytes) => Ok((bytes.clone(), None)),
        DatasetSource::Inline(value) => Ok((Bytes::from(serde_json::to_vec(value)?), None)),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => Err(DatasetError::Validation(
            "remote sources must be consumed by a remote-capable loader".into(),
        )),
    }
}

fn raw_max_tokens(value: &Value) -> Option<u32> {
    ["max_tokens", "max_completion_tokens", "max_output_tokens"]
        .into_iter()
        .find_map(|field| value.get(field).and_then(Value::as_u64))
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aiperf_rng::RngRoot;

    use super::*;
    use crate::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::segment::Payload;
    use crate::tokenizer::TiktokenTokenizer;

    #[tokio::test]
    async fn raw_jsonl_preserves_authored_bytes_and_one_session_per_line() {
        let input = Bytes::from_static(
            b" { \"messages\": [ {\"role\":\"user\",\"content\":\"a\"} ], \"z\": 1 } \n{\"messages\":[],\"model\":\"m\"}\n",
        );
        let loader = RawPayloadDatasetLoader;
        let rows = loader
            .load(&LoadConfig::new(DatasetSource::Bytes(input)))
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].wire.as_deref(),
            Some(&b"{ \"messages\": [ {\"role\":\"user\",\"content\":\"a\"} ], \"z\": 1 }"[..])
        );
        assert_ne!(rows[0].group_key, rows[1].group_key);
    }

    #[tokio::test]
    async fn inputs_json_retains_nested_raw_object_slices() {
        let input = Bytes::from_static(
            br#"{"data":[{"session_id":"s","payloads":[ { "messages" : [], "z" : 1 } ]}]}"#,
        );
        let rows = InputsJsonPayloadLoader
            .load(&LoadConfig::new(DatasetSource::Bytes(input)))
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].wire.as_deref(),
            Some(&b"{ \"messages\" : [], \"z\" : 1 }"[..])
        );
        assert_eq!(rows[0].session_id.as_ref().unwrap().as_str(), "s");
    }

    #[tokio::test]
    async fn full_pipeline_freezes_raw_payload_and_returns_it_byte_identically() {
        let input = Bytes::from_static(
            br#"{ "messages":[], "model":"authored", "max_tokens":9, "vendor":{"x":true} }"#,
        );
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(RawPayloadDatasetLoader),
                Arc::new(RawPayloadComposer),
            ))
            .unwrap();
        let dataset = registry
            .build_dataset(
                Some("raw_payload"),
                &LoadConfig::new(DatasetSource::Bytes(input.clone())),
                &ComposeConfig::new("fallback", RngRoot::new(Some(0))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.model.as_ref().unwrap().as_str(), "authored");
        assert_eq!(turn.max_tokens, Some(9));
        let handle = turn.raw_payload.unwrap();
        assert!(matches!(
            dataset.segments().get(handle).unwrap(),
            Payload::Raw { .. }
        ));
        assert_eq!(
            dataset
                .segments()
                .build_body(&[handle], &crate::Overrides::new())
                .unwrap(),
            input
        );
    }
}
