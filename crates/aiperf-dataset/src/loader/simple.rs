// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical single-turn and multi-turn JSONL loaders.
//!
//! Ported from `src/aiperf/dataset/loader/{single_turn,multi_turn}.py` and
//! `src/aiperf/dataset/loader/mixins.py`, including modality validation,
//! session grouping, per-turn timing/output overrides, named batches, local
//! media encoding, and insertion-order preservation.

use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Map, Value};
use smallvec::SmallVec;

use crate::compose::{ComposeConfig, ComposeState, Composer, SessionIdGenerator};
use crate::error::{DatasetError, Result};
use crate::loader::{DatasetLoader, DatasetProbe, LoadConfig, RawRow, jsonl_rows};
use crate::model::{
    ContentGroup, Conversation, ConversationContextMode, MediaKind, SessionId, Turn,
};
use crate::segment::{Handle, Role, SegmentPool};
use crate::tokenizer::TextTokenizer;

/// Loader for one request per JSONL row, optionally grouped by `session_id`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleTurnDatasetLoader;

/// Composer paired with [`SingleTurnDatasetLoader`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleTurnComposer;

/// Loader for JSONL rows containing a non-empty `turns` array.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiTurnDatasetLoader;

/// Composer paired with [`MultiTurnDatasetLoader`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiTurnComposer;

#[derive(Debug, Clone, Deserialize)]
struct AuthoredGroup {
    #[serde(default)]
    name: String,
    contents: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum AuthoredBatch {
    Strings(Vec<String>),
    Groups(Vec<AuthoredGroup>),
}

#[derive(Debug, Clone, Deserialize)]
struct SingleTurnRow {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    texts: Option<AuthoredBatch>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    images: Option<AuthoredBatch>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    audios: Option<AuthoredBatch>,
    #[serde(default)]
    video: Option<String>,
    #[serde(default)]
    videos: Option<AuthoredBatch>,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    delay: Option<f64>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    output_length: Option<u32>,
    #[serde(default)]
    extra: Option<Map<String, Value>>,
}

impl SingleTurnRow {
    fn parse(value: Value, origin: &impl std::fmt::Display) -> Result<Self> {
        let row: Self = serde_json::from_value(value).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid single_turn row: {error}"))
        })?;
        row.validate(origin)?;
        Ok(row)
    }

    fn validate(&self, origin: &impl std::fmt::Display) -> Result<()> {
        for (singular, plural, names) in [
            (self.text.is_some(), self.texts.is_some(), "text and texts"),
            (
                self.image.is_some(),
                self.images.is_some(),
                "image and images",
            ),
            (
                self.audio.is_some(),
                self.audios.is_some(),
                "audio and audios",
            ),
            (
                self.video.is_some(),
                self.videos.is_some(),
                "video and videos",
            ),
        ] {
            if singular && plural {
                return Err(DatasetError::Validation(format!(
                    "{origin}: {names} cannot be set together"
                )));
            }
        }
        if self.timestamp.is_some() && self.delay.is_some() {
            return Err(DatasetError::Validation(format!(
                "{origin}: timestamp and delay cannot be set together"
            )));
        }
        if self.output_length == Some(0) {
            return Err(DatasetError::Validation(format!(
                "{origin}: output_length must be greater than zero"
            )));
        }
        if self
            .delay
            .is_some_and(|delay| !delay.is_finite() || delay < 0.0)
        {
            return Err(DatasetError::Validation(format!(
                "{origin}: delay must be finite and non-negative"
            )));
        }
        if self
            .timestamp
            .is_some_and(|timestamp| !timestamp.is_finite())
        {
            return Err(DatasetError::Validation(format!(
                "{origin}: timestamp must be finite"
            )));
        }
        let populated = [
            self.text.as_ref().is_some_and(|value| !value.is_empty()),
            batch_has_content(self.texts.as_ref()),
            self.image.as_ref().is_some_and(|value| !value.is_empty()),
            batch_has_content(self.images.as_ref()),
            self.audio.as_ref().is_some_and(|value| !value.is_empty()),
            batch_has_content(self.audios.as_ref()),
            self.video.as_ref().is_some_and(|value| !value.is_empty()),
            batch_has_content(self.videos.as_ref()),
        ]
        .into_iter()
        .any(|value| value);
        if !populated {
            return Err(DatasetError::Validation(format!(
                "{origin}: at least one modality must be provided"
            )));
        }
        Ok(())
    }
}

fn batch_has_content(batch: Option<&AuthoredBatch>) -> bool {
    match batch {
        Some(AuthoredBatch::Strings(values)) => values.iter().any(|value| !value.is_empty()),
        Some(AuthoredBatch::Groups(groups)) => groups
            .iter()
            .any(|group| group.contents.iter().any(|value| !value.is_empty())),
        None => false,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct MultiTurnRow {
    #[serde(default)]
    session_id: Option<String>,
    turns: Vec<SingleTurnRow>,
}

impl MultiTurnRow {
    fn parse(value: Value, origin: &impl std::fmt::Display) -> Result<Self> {
        let row: Self = serde_json::from_value(value).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid multi_turn row: {error}"))
        })?;
        if row.turns.is_empty() {
            return Err(DatasetError::Validation(format!(
                "{origin}: at least one turn must be provided"
            )));
        }
        for turn in &row.turns {
            turn.validate(origin)?;
        }
        Ok(row)
    }
}

#[async_trait]
impl DatasetLoader for SingleTurnDatasetLoader {
    fn name(&self) -> &str {
        "single_turn"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        let Some(value) = &probe.value else {
            return false;
        };
        value.get("turns").is_none() && SingleTurnRow::parse(value.clone(), &"probe").is_ok()
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let rows = jsonl_rows(&config.source)?;
        for row in &rows {
            SingleTurnRow::parse(row.value.clone(), &row.origin)?;
        }
        Ok(rows)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

#[async_trait]
impl DatasetLoader for MultiTurnDatasetLoader {
    fn name(&self) -> &str {
        "multi_turn"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe
            .value
            .as_ref()
            .is_some_and(|value| MultiTurnRow::parse(value.clone(), &"probe").is_ok())
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let rows = jsonl_rows(&config.source)?;
        for row in &rows {
            MultiTurnRow::parse(row.value.clone(), &row.origin)?;
        }
        Ok(rows)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

impl Composer for SingleTurnComposer {
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
        let mut positions = HashMap::<SessionId, usize>::new();
        let mut parents = Vec::<Option<Handle>>::new();

        for raw in rows {
            let row = SingleTurnRow::parse(raw.value, &raw.origin)?;
            let session_id = row
                .session_id
                .as_deref()
                .map(SessionId::from)
                .unwrap_or_else(|| generator.next_id());
            let position = match positions.get(&session_id).copied() {
                Some(position) => position,
                None => {
                    let position = conversations.len();
                    let (conversation, parent) =
                        start_conversation(session_id.clone(), position, &mut state)?;
                    conversations.push(conversation);
                    parents.push(parent);
                    positions.insert(session_id, position);
                    position
                }
            };
            let turn = compose_simple_turn(row, &mut parents[position], &mut state)?;
            conversations[position].turns.push(turn);
        }
        for conversation in &mut conversations {
            if conversation.turns.len() > 1 {
                conversation.context_mode =
                    Some(ConversationContextMode::MessageArrayWithResponses);
            }
        }
        Ok(conversations)
    }
}

impl Composer for MultiTurnComposer {
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
        let mut positions = HashMap::<SessionId, usize>::new();
        let mut parents = Vec::<Option<Handle>>::new();

        for raw in rows {
            let row = MultiTurnRow::parse(raw.value, &raw.origin)?;
            let session_id = row
                .session_id
                .as_deref()
                .map(SessionId::from)
                .unwrap_or_else(|| generator.next_id());
            let position = match positions.get(&session_id).copied() {
                Some(position) => position,
                None => {
                    let position = conversations.len();
                    let (conversation, parent) =
                        start_conversation(session_id.clone(), position, &mut state)?;
                    conversations.push(conversation);
                    parents.push(parent);
                    positions.insert(session_id, position);
                    position
                }
            };
            for authored_turn in row.turns {
                let turn = compose_simple_turn(authored_turn, &mut parents[position], &mut state)?;
                conversations[position].turns.push(turn);
            }
        }
        Ok(conversations)
    }
}

fn start_conversation(
    session_id: SessionId,
    position: usize,
    state: &mut ComposeState<'_>,
) -> Result<(Conversation, Option<Handle>)> {
    let mut conversation = Conversation::new(session_id);
    let mut parent = None;
    if let Some(system) = &state.config.shared_system_prompt {
        let tokens = state.tokenizer.encode(system)?;
        let handle = state.segments.intern_text(
            parent,
            "system",
            Bytes::copy_from_slice(system.as_bytes()),
            tokens.into_boxed_slice(),
        )?;
        conversation.system = Some(handle);
        parent = Some(handle);
    }
    if !state.config.user_context_prompts.is_empty() {
        let context = state
            .config
            .user_context_prompts
            .get(position)
            .ok_or_else(|| {
                DatasetError::Validation(format!(
                    "missing user context prompt for conversation index {position}"
                ))
            })?;
        let tokens = state.tokenizer.encode(context)?;
        let handle = state.segments.intern_text(
            parent,
            "user",
            Bytes::copy_from_slice(context.as_bytes()),
            tokens.into_boxed_slice(),
        )?;
        conversation.user_context = Some(handle);
        parent = Some(handle);
    }
    Ok((conversation, parent))
}

fn compose_simple_turn(
    row: SingleTurnRow,
    parent: &mut Option<Handle>,
    state: &mut ComposeState<'_>,
) -> Result<Turn> {
    let role = row.role.clone().unwrap_or_else(|| "user".to_string());
    let mut turn = Turn {
        role: row.role.map(Role::new),
        max_tokens: row.output_length,
        timestamp_ms: row.timestamp,
        delay_ms: row.delay,
        ..Turn::default()
    };
    append_modality(
        &mut turn,
        MediaKind::Text,
        row.text,
        row.texts,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Image,
        row.image,
        row.images,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Audio,
        row.audio,
        row.audios,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Video,
        row.video,
        row.videos,
        &role,
        parent,
        state,
    )?;
    if let Some(extra) = row.extra {
        let wire = serde_json::to_vec(&Value::Object(extra))?;
        turn.extra_body = Some(state.segments.intern_raw(None, Bytes::from(wire))?);
    }
    state.finalize_turn(&mut turn)?;
    Ok(turn)
}

#[allow(clippy::too_many_arguments)]
fn append_modality(
    turn: &mut Turn,
    kind: MediaKind,
    singular: Option<String>,
    plural: Option<AuthoredBatch>,
    role: &str,
    parent: &mut Option<Handle>,
    state: &mut ComposeState<'_>,
) -> Result<()> {
    let groups = match (singular, plural) {
        (Some(value), None) => vec![AuthoredGroup {
            name: String::new(),
            contents: vec![value],
        }],
        (None, Some(AuthoredBatch::Strings(contents))) => vec![AuthoredGroup {
            name: String::new(),
            contents,
        }],
        (None, Some(AuthoredBatch::Groups(groups))) => groups,
        (None, None) => return Ok(()),
        (Some(_), Some(_)) => unreachable!("row validation rejects singular + plural"),
    };

    for group in groups {
        let mut handles = SmallVec::new();
        for content in group.contents {
            let handle = if kind == MediaKind::Text {
                let tokens = state.tokenizer.encode(&content)?;
                turn.input_tokens = turn
                    .input_tokens
                    .checked_add(tokens.len() as u64)
                    .ok_or_else(|| {
                        DatasetError::Validation("turn input token count overflowed u64".into())
                    })?;
                state.segments.intern_text(
                    *parent,
                    role,
                    Bytes::from(content),
                    tokens.into_boxed_slice(),
                )?
            } else {
                let bytes = state.config.media_resolver.resolve(kind, &content)?;
                state.segments.intern_media(*parent, kind, bytes)?
            };
            *parent = Some(handle);
            handles.push(handle);
        }
        turn.content.push(ContentGroup {
            kind,
            name: group.name,
            handles,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aiperf_rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::loader::{DatasetFormatRegistration, DatasetSource, LoaderRegistry};
    use crate::segment::Payload;
    use crate::tokenizer::TiktokenTokenizer;

    fn config() -> ComposeConfig {
        ComposeConfig::new("model", RngRoot::new(Some(1)))
    }

    #[tokio::test]
    async fn single_turn_groups_sessions_and_interns_named_content() {
        let source = DatasetSource::Inline(json!([
            {"session_id":"s", "text":"first", "output_length":7},
            {"session_id":"s", "texts":[{"name":"query","contents":["second"]}]},
            {"text":"standalone"}
        ]));
        let load = LoadConfig::new(source);
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ))
            .unwrap();
        let dataset = registry
            .build_dataset(
                Some("single-turn"),
                &load,
                &config(),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 2);
        assert_eq!(dataset.conversations()[0].turns.len(), 2);
        assert_eq!(
            dataset.conversations()[0].context_mode,
            Some(ConversationContextMode::MessageArrayWithResponses)
        );
        assert_eq!(dataset.conversations()[0].turns[0].max_tokens, Some(7));
        let handle = dataset.conversations()[0].turns[1].content[0].handles[0];
        assert!(matches!(
            dataset.segments().get(handle).unwrap(),
            Payload::Text { .. }
        ));
    }

    #[tokio::test]
    async fn multi_turn_concatenates_duplicate_session_rows_and_applies_context() {
        let source = DatasetSource::Inline(json!([
            {"session_id":"s", "turns":[{"text":"one"}]},
            {"session_id":"s", "turns":[{"text":"two","delay":5}]}
        ]));
        let mut compose = config();
        compose.shared_system_prompt = Some("system".into());
        compose.user_context_prompts = vec!["context".into()];
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(MultiTurnDatasetLoader),
                Arc::new(MultiTurnComposer),
            ))
            .unwrap();
        let dataset = registry
            .build_dataset(
                None,
                &LoadConfig::new(source),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let conversation = &dataset.conversations()[0];
        assert_eq!(conversation.turns.len(), 2);
        assert_eq!(conversation.turns[1].delay_ms, Some(5.0));
        assert!(conversation.system.is_some());
        assert!(conversation.user_context.is_some());
    }

    #[test]
    fn validation_rejects_mutual_exclusion_and_empty_modalities() {
        for value in [
            json!({"text":"a", "texts":["b"]}),
            json!({"timestamp":0, "delay":1, "text":"a"}),
            json!({"texts":[]}),
            json!({"text":"a", "output_length":0}),
        ] {
            assert!(SingleTurnRow::parse(value, &"fixture").is_err());
        }
    }
}
