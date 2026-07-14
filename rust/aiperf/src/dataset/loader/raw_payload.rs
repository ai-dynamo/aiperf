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

use crate::endpoints::extract_payload;

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::dataset::model::{Conversation, ConversationContextMode, ModelId, SessionId, Turn};
use crate::dataset::segment::{Role, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

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
        let mut parents = Vec::<Option<crate::dataset::Handle>>::new();

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
            let raw_token_ids = raw_token_ids(&row.value, &row.origin)?;
            let (raw_payload, raw_token_handle, extra_body) = if config.requires_raw_token_ids {
                validate_token_native_fields(&row.value, &row.origin)?;
                let token_ids = raw_token_ids.as_ref().ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{}: selected endpoint requires a non-empty token_ids array",
                        row.origin
                    ))
                })?;
                let token_handle = state
                    .segments
                    .intern_token_ids(parents[position], token_ids.clone())?;
                let extra = token_native_extra_body(&row.value)?;
                let extra_handle = (!extra.is_empty())
                    .then(|| {
                        serde_json::to_vec(&Value::Object(extra))
                            .map(Bytes::from)
                            .map_err(DatasetError::from)
                            .and_then(|wire| state.segments.intern_raw(Some(token_handle), wire))
                    })
                    .transpose()?;
                parents[position] = extra_handle.or(Some(token_handle));
                (None, Some(token_handle), extra_handle)
            } else {
                let wire = row.wire.ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{}: raw payload loader did not retain exact wire bytes",
                        row.origin
                    ))
                })?;
                let handle = state.segments.intern_raw(parents[position], wire)?;
                let token_handle = raw_token_ids
                    .as_ref()
                    .map(|token_ids| {
                        state
                            .segments
                            .intern_token_ids(Some(handle), token_ids.clone())
                    })
                    .transpose()?;
                parents[position] = token_handle.or(Some(handle));
                (Some(handle), token_handle, None)
            };
            let mut turn = Turn {
                role: Some(Role::from("user")),
                model: row
                    .value
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ModelId::from),
                max_tokens: if config.requires_raw_token_ids {
                    token_native_max_tokens(&row.value)
                } else {
                    raw_max_tokens(&row.value)
                },
                streaming: row.value.get("stream").and_then(Value::as_bool),
                body: Turn::dispatch_body(raw_payload, raw_token_handle, &[]),
                extra_body,
                input_tokens: raw_token_ids.as_ref().map_or_else(
                    || raw_input_tokens(&row.value, state.tokenizer),
                    |token_ids| {
                        u64::try_from(token_ids.len()).map_err(|_| {
                            DatasetError::Validation("raw token count exceeds u64".into())
                        })
                    },
                )?,
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
    object.get("messages").is_some_and(Value::is_array) || object.contains_key("token_ids")
}

fn validate_raw_payload(value: &Value, origin: &impl std::fmt::Display) -> Result<()> {
    validate_payload_object(value, origin)?;
    if !value.get("messages").is_some_and(Value::is_array) && value.get("token_ids").is_none() {
        return Err(DatasetError::Validation(format!(
            "{origin}: raw_payload row requires either messages or token_ids"
        )));
    }
    raw_token_ids(value, origin)?;
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

fn token_native_extra_body(value: &Value) -> Result<serde_json::Map<String, Value>> {
    let mut extra = value.as_object().cloned().ok_or_else(|| {
        DatasetError::Validation("token-native raw payload must be a JSON object".into())
    })?;
    for field in [
        "token_ids",
        "model",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    ] {
        extra.remove(field);
    }
    Ok(extra)
}

fn validate_token_native_fields(value: &Value, origin: &impl std::fmt::Display) -> Result<()> {
    let object = value.as_object().ok_or_else(|| {
        DatasetError::Validation(format!("{origin}: token-native payload must be an object"))
    })?;
    if let Some(model) = object.get("model")
        && !model.is_null()
        && model.as_str().is_none_or(|model| model.trim().is_empty())
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: model must be a non-empty string when configured"
        )));
    }
    if let Some(stream) = object.get("stream")
        && !stream.is_null()
        && stream.as_bool() != Some(false)
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: token-native stream must be false"
        )));
    }
    for field in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
        let value = object.get(field);
        validate_optional_positive_u32(value, origin, field)?;
    }
    if let Some(sampling_params) = object.get("sampling_params") {
        match sampling_params {
            Value::Null => {}
            Value::Object(sampling_params) => {
                let value = sampling_params.get("max_tokens");
                validate_optional_positive_u32(value, origin, "sampling_params.max_tokens")?;
            }
            _ => {
                return Err(DatasetError::Validation(format!(
                    "{origin}: sampling_params must be an object when configured"
                )));
            }
        }
    }
    Ok(())
}

fn validate_optional_positive_u32(
    value: Option<&Value>,
    origin: &impl std::fmt::Display,
    field: &str,
) -> Result<()> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(());
    };
    if value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .is_none_or(|value| value == 0)
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: {field} must be a positive unsigned 32-bit integer"
        )));
    }
    Ok(())
}

fn raw_token_ids(value: &Value, origin: &impl std::fmt::Display) -> Result<Option<Vec<u32>>> {
    let Some(value) = value.get("token_ids") else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| {
            DatasetError::Validation(format!(
                "{origin}: token_ids must be a non-empty list of unsigned 32-bit integers"
            ))
        })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{origin}: token_ids[{index}] must be an unsigned 32-bit integer"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn directory_has_raw_payload(directory: &Path) -> bool {
    sorted_jsonl_files(directory).into_iter().any(|path| {
        std::fs::read(path)
            .ok()
            .and_then(|bytes| {
                bytes
                    .split(|byte| *byte == b'\n')
                    .map(crate::dataset::loader::trim_ascii)
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
        .or_else(|| {
            value
                .get("sampling_params")
                .and_then(Value::as_object)
                .and_then(|sampling| sampling.get("max_tokens"))
                .and_then(Value::as_u64)
        })
        .and_then(|value| u32::try_from(value).ok())
}

fn token_native_max_tokens(value: &Value) -> Option<u32> {
    value
        .get("sampling_params")
        .and_then(Value::as_object)
        .and_then(|sampling| sampling.get("max_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            ["max_tokens", "max_completion_tokens", "max_output_tokens"]
                .into_iter()
                .find_map(|field| value.get(field).and_then(Value::as_u64))
        })
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::dataset::segment::Payload;
    use crate::dataset::tokenizer::TiktokenTokenizer;

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
        let handle = *turn.body.first().expect("raw body handle");
        assert!(matches!(
            dataset.segments().get(handle).unwrap(),
            Payload::Raw { .. }
        ));
        assert_eq!(
            dataset
                .segments()
                .build_body(&[handle], &crate::dataset::Overrides::new())
                .unwrap(),
            input
        );
    }

    #[tokio::test]
    async fn token_native_composition_frees_raw_bytes_and_retains_typed_fields() {
        let input = Bytes::from_static(
            br#"{"model":"authored","token_ids":[7,8,9],"max_tokens":5,"sampling_params":{"temperature":0,"max_tokens":7},"stream":false,"request_id":"r-1"}"#,
        );
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(RawPayloadDatasetLoader),
                Arc::new(RawPayloadComposer),
            ))
            .unwrap();
        let mut compose = ComposeConfig::new("fallback", RngRoot::new(Some(0)));
        compose.requires_raw_token_ids = true;

        let dataset = registry
            .build_dataset(
                Some("raw_payload"),
                &LoadConfig::new(DatasetSource::Bytes(input)),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        // Token-native composition frees the raw body: the only body handle is
        // the token-native segment, never a raw one.
        let token_handle = *turn.body.first().expect("token handle");
        assert!(!matches!(
            dataset.segments().get(token_handle).unwrap(),
            Payload::Raw { .. }
        ));
        assert_eq!(turn.input_tokens, 3);
        assert_eq!(turn.max_tokens, Some(7));
        let Payload::TokenIds { token_ids } = dataset.segments().get(token_handle).unwrap() else {
            panic!("raw payload token IDs must be interned in the token domain")
        };
        assert_eq!(&**token_ids, &[7, 8, 9]);
        let Payload::Raw { wire } = dataset
            .segments()
            .get(turn.extra_body.expect("sampling extra"))
            .unwrap()
        else {
            panic!("non-canonical fields must remain available to the endpoint")
        };
        let extra: Value = serde_json::from_slice(wire).unwrap();
        assert_eq!(extra["sampling_params"]["temperature"], 0);
        assert_eq!(extra["sampling_params"]["max_tokens"], 7);
        assert_eq!(extra["request_id"], "r-1");
        assert!(extra.get("token_ids").is_none());
        assert!(extra.get("model").is_none());
        assert!(extra.get("stream").is_none());
        assert!(extra.get("max_tokens").is_none());

        for invalid in [
            json!({"token_ids": [1], "model": 7}),
            json!({"token_ids": [1], "stream": true}),
            json!({"token_ids": [1], "stream": "false"}),
            json!({"token_ids": [1], "max_tokens": 0}),
            json!({"token_ids": [1], "sampling_params": []}),
            json!({"token_ids": [1], "sampling_params": {"max_tokens": -1}}),
        ] {
            let result = registry
                .build_dataset(
                    Some("raw_payload"),
                    &LoadConfig::new(DatasetSource::Inline(json!([invalid]))),
                    &compose,
                    &TiktokenTokenizer::builtin(),
                )
                .await;
            assert!(result.is_err(), "invalid token-native field was accepted");
        }
    }
}
