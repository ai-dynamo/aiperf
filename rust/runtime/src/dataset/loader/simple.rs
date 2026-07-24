// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical single-turn and multi-turn JSONL loaders.
//!
//! These loaders provide modality validation,
//! session grouping, per-turn timing/output overrides, named batches, local
//! media encoding, and insertion-order preservation.

use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Map, Value};
use smallvec::SmallVec;

use crate::dataset::compose::{ComposeConfig, ComposeState, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{DatasetLoader, DatasetProbe, LoadConfig, RawRow, jsonl_rows};
use crate::dataset::model::{
    ContentGroup, Conversation, ConversationContextMode, MediaKind, ModelId, SessionId, Turn,
};
use crate::dataset::segment::{Handle, Role, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

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
    model: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    streaming: Option<bool>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    texts: Option<AuthoredBatch>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    images: Option<AuthoredBatch>,
    /// Cache UUIDs aligned 1:1 with string-form `images`. Only supported
    /// when `images` is `AuthoredBatch::Strings`; for grouped batches, use
    /// per-group content directly. The singular `image` field is not
    /// supported. vLLM-extension only: opaque IDs that let the server reuse
    /// cached image embeddings across requests. See `Media::uuids`.
    #[serde(default)]
    image_uuids: Option<Vec<String>>,
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
    #[serde(default, alias = "token_ids")]
    raw_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    extra: Option<Map<String, Value>>,
    #[serde(default)]
    extra_headers: Option<Map<String, Value>>,
    #[serde(default)]
    request_parameters: Option<Map<String, Value>>,
}

impl SingleTurnRow {
    fn parse(value: Value, origin: &impl std::fmt::Display) -> Result<Self> {
        let mut row: Self = serde_json::from_value(value).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid single_turn row: {error}"))
        })?;
        row.normalize_image_uuids(origin)?;
        row.validate(origin)?;
        Ok(row)
    }

    /// Normalize UUID-only images and reject ambiguous UUID mappings.
    ///
    /// Ports Python's `SingleTurn.validate_image_uuids_alignment`.
    fn normalize_image_uuids(&mut self, origin: &impl std::fmt::Display) -> Result<()> {
        let Some(uuids) = &self.image_uuids else {
            return Ok(());
        };
        if self.image.is_some() {
            return Err(DatasetError::Validation(format!(
                "{origin}: image_uuids cannot be used with the singular image field"
            )));
        }
        if self.images.is_none() {
            self.images = Some(AuthoredBatch::Strings(vec![String::new(); uuids.len()]));
        }
        let Some(AuthoredBatch::Strings(contents)) = &self.images else {
            return Err(DatasetError::Validation(format!(
                "{origin}: image_uuids cannot be set when images is provided as grouped \
                 batches; use per-group content directly instead"
            )));
        };
        if uuids.len() != contents.len() {
            return Err(DatasetError::Validation(format!(
                "{origin}: image_uuids length ({}) must match images length ({})",
                uuids.len(),
                contents.len()
            )));
        }
        if uuids.iter().any(String::is_empty) {
            return Err(DatasetError::Validation(format!(
                "{origin}: image_uuids must not contain empty strings"
            )));
        }
        Ok(())
    }

    fn validate(&self, origin: &impl std::fmt::Display) -> Result<()> {
        let extra_has_token_ids = self.extra.as_ref().is_some_and(|extra| {
            extra.contains_key("raw_token_ids") || extra.contains_key("token_ids")
        });
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
        if self.raw_token_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(DatasetError::Validation(format!(
                "{origin}: raw_token_ids must be a non-empty list of unsigned 32-bit integers"
            )));
        }
        for (field, value) in [("model", &self.model), ("endpoint", &self.endpoint)] {
            if value.as_ref().is_some_and(|value| value.trim().is_empty()) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: {field} must be non-empty when configured"
                )));
            }
        }
        for (field, values) in [
            ("extra_headers", self.extra_headers.as_ref()),
            ("request_parameters", self.request_parameters.as_ref()),
        ] {
            if values.is_some_and(|values| values.values().any(|value| !value.is_string())) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: {field} values must be strings"
                )));
            }
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
            self.raw_token_ids.is_some(),
            extra_has_token_ids,
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
        if self.raw_token_ids.is_some()
            && [
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
            .any(|value| value)
        {
            return Err(DatasetError::Validation(format!(
                "{origin}: raw_token_ids cannot be combined with text or media fields"
            )));
        }
        if extra_has_token_ids
            && [
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
            .any(|value| value)
        {
            return Err(DatasetError::Validation(format!(
                "{origin}: extra token_ids cannot be combined with text or media fields"
            )));
        }
        if !populated {
            return Err(DatasetError::Validation(format!(
                "{origin}: at least one modality must be provided"
            )));
        }
        Ok(())
    }
}

/// Drop image content for UUIDs repeated within one session (`--uuid-and-strip`).
///
/// Images repeated within one row keep their payload (the server resolves
/// that request's cache misses before populating its cache); only UUIDs
/// whose content this loader observed in an *earlier* row for the same
/// session are stripped. Explicit cache-only references (empty content
/// authored directly) pass through regardless of local history. Ports
/// Python's `SingleTurnDatasetLoader._dedup_repeated_images_inplace`,
/// applied per-row here since one `single_turn` row is one turn.
fn dedup_repeated_images_inplace(
    row: &mut SingleTurnRow,
    seen: &mut std::collections::HashSet<String>,
) {
    let Some(uuids) = &row.image_uuids else {
        return;
    };
    let Some(AuthoredBatch::Strings(contents)) = &mut row.images else {
        return;
    };
    let mut new_uuids = std::collections::HashSet::new();
    for (content, uuid) in contents.iter_mut().zip(uuids) {
        if seen.contains(uuid) {
            content.clear();
        } else if !content.is_empty() {
            new_uuids.insert(uuid.clone());
        }
    }
    seen.extend(new_uuids);
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
        if config
            .options
            .get("uuid_and_strip")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(DatasetError::Validation(
                "--uuid-and-strip is not supported with --custom-dataset-type multi_turn. \
                 Load-time dedup of repeated images is only implemented for the \
                 single_turn loader. Use --custom-dataset-type single_turn (with \
                 session_id-grouped rows) for cache-reuse benchmarks."
                    .into(),
            ));
        }
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
        let uuid_and_strip = config
            .format_options
            .get("uuid_and_strip")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut seen_uuids = HashMap::<SessionId, std::collections::HashSet<String>>::new();

        for raw in rows {
            let mut row = SingleTurnRow::parse(raw.value, &raw.origin)?;
            let session_id = row
                .session_id
                .as_deref()
                .map(SessionId::from)
                .unwrap_or_else(|| generator.next_id());
            if uuid_and_strip {
                dedup_repeated_images_inplace(
                    &mut row,
                    seen_uuids.entry(session_id.clone()).or_default(),
                );
            }
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
        // A leading authored `system` turn eligible for hoisting is held here
        // (not yet interned) until the NEXT turn of its session decides its
        // fate: a following non-system turn commits the hoist into
        // `conversation.system`; a second consecutive leading `system` turn, or
        // no further turn at all, un-hoists it back to a normal dispatched turn
        // (matching Python, which only merges `system_message` as a leading
        // rendered system message during warmup). Deferring avoids interning
        // then having to unwind the segment parent chain. Indexed by conversation
        // position, parallel to `parents`.
        let mut pending_system = Vec::<Option<SingleTurnRow>>::new();

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
                    pending_system.push(None);
                    positions.insert(session_id, position);
                    position
                }
            };
            for authored_turn in row.turns {
                if state.config.hoist_leading_system_message {
                    if let Some(pending) = pending_system[position].take() {
                        // A leading system turn is deferred; this turn resolves it.
                        if authored_turn.role.as_deref() == Some("system") {
                            // Second consecutive leading system turn: un-hoist the
                            // deferred one as a normal turn, then fall through so
                            // this one dispatches normally too.
                            let turn =
                                compose_simple_turn(pending, &mut parents[position], &mut state)?;
                            conversations[position].turns.push(turn);
                        } else {
                            // Commit the hoist: intern the deferred system text at
                            // the conversation level, then fall through to dispatch
                            // this (non-system) turn normally.
                            commit_system_hoist(
                                &mut conversations[position],
                                &pending,
                                &mut parents[position],
                                &mut state,
                            )?;
                        }
                    } else if conversations[position].turns.is_empty()
                        && conversations[position].system.is_none()
                        && hoistable_system_text(&authored_turn).is_some()
                    {
                        // Defer this leading, text-only system turn.
                        pending_system[position] = Some(authored_turn);
                        continue;
                    }
                }
                let turn = compose_simple_turn(authored_turn, &mut parents[position], &mut state)?;
                conversations[position].turns.push(turn);
            }
        }

        // Flush any still-deferred leading system turn: it was the session's only
        // turn, so a conversation-level system message would leave it
        // undispatchable. Restore it as a normal turn (pre-hoist behavior).
        for position in 0..conversations.len() {
            if let Some(pending) = pending_system[position].take() {
                let turn = compose_simple_turn(pending, &mut parents[position], &mut state)?;
                conversations[position].turns.push(turn);
            }
        }
        Ok(conversations)
    }
}

/// Plain text of a leading authored `system` turn eligible for hoisting into
/// `conversation.system`, or `None` when the row does not qualify. Mirrors the
/// guard in Python `MultiTurnDatasetLoader._try_hoist_system_message`: role is
/// `system`, text only (no image/audio/video, no raw token ids), and no
/// dispatch-time metadata (`timestamp`, `delay`, `output_length`, `extra`) —
/// a conversation-level system message has no turn to carry those, so a system
/// turn that sets any of them falls through to normal handling rather than
/// silently dropping it.
fn hoistable_system_text(row: &SingleTurnRow) -> Option<String> {
    if row.role.as_deref() != Some("system")
        || row.timestamp.is_some()
        || row.delay.is_some()
        || row.output_length.is_some()
        || row.extra.is_some()
        || row.raw_token_ids.is_some()
        || row.image.is_some()
        || row.images.is_some()
        || row.audio.is_some()
        || row.audios.is_some()
        || row.video.is_some()
        || row.videos.is_some()
    {
        return None;
    }
    let mut parts = Vec::<String>::new();
    if let Some(text) = &row.text {
        parts.push(text.clone());
    }
    match &row.texts {
        Some(AuthoredBatch::Strings(strings)) => parts.extend(strings.iter().cloned()),
        Some(AuthoredBatch::Groups(groups)) => {
            for group in groups {
                parts.extend(group.contents.iter().cloned());
            }
        }
        None => {}
    }
    let text = parts.join("\n");
    (!text.is_empty()).then_some(text)
}

/// Intern a deferred leading system turn's text as the conversation-level system
/// message, mirroring how `start_conversation` interns `shared_system_prompt`.
/// The interned segment becomes the new parent so subsequent turns chain onto it.
fn commit_system_hoist(
    conversation: &mut Conversation,
    row: &SingleTurnRow,
    parent: &mut Option<Handle>,
    state: &mut ComposeState<'_>,
) -> Result<()> {
    let text = hoistable_system_text(row)
        .expect("commit_system_hoist called only for a hoistable system turn");
    let tokens = state.tokenizer.encode(&text)?;
    let handle = state.segments.intern_text(
        *parent,
        "system",
        Bytes::copy_from_slice(text.as_bytes()),
        tokens.into_boxed_slice(),
    )?;
    conversation.system = Some(handle);
    *parent = Some(handle);
    Ok(())
}

fn start_conversation(
    session_id: SessionId,
    position: usize,
    state: &mut ComposeState<'_>,
) -> Result<(Conversation, Option<Handle>)> {
    let mut conversation = Conversation::new(session_id);
    if state.config.requires_raw_token_ids {
        conversation.context_mode = Some(ConversationContextMode::MessageArrayWithResponses);
    }
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
    mut row: SingleTurnRow,
    parent: &mut Option<Handle>,
    state: &mut ComposeState<'_>,
) -> Result<Turn> {
    let sampling_max_tokens = if state.config.requires_raw_token_ids {
        token_native_sampling_max_tokens(row.extra.as_ref())?
    } else {
        row.extra
            .as_ref()
            .and_then(|extra| extra.get("sampling_params"))
            .and_then(Value::as_object)
            .and_then(|sampling| sampling.get("max_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
    };
    let max_tokens = if state.config.requires_raw_token_ids {
        sampling_max_tokens.or(row.output_length)
    } else {
        row.output_length.or(sampling_max_tokens)
    };
    let promoted_token_ids = if state.config.requires_raw_token_ids {
        take_extra_token_ids(row.extra.as_mut())?
    } else {
        None
    };
    let raw_token_ids = match (row.raw_token_ids.take(), promoted_token_ids) {
        (Some(_), Some(_)) => {
            return Err(DatasetError::Validation(
                "raw token IDs were authored in both raw_token_ids and extra.token_ids".into(),
            ));
        }
        (Some(token_ids), None) | (None, Some(token_ids)) => Some(token_ids),
        (None, None) => None,
    };
    let role = row.role.clone().unwrap_or_else(|| "user".to_string());
    let mut turn = Turn {
        role: row.role.map(Role::new),
        model: row.model.map(ModelId::from),
        endpoint: row.endpoint,
        streaming: row.streaming,
        max_tokens,
        timestamp_ms: row.timestamp,
        delay_ms: row.delay,
        ..Turn::default()
    };
    append_modality(
        &mut turn,
        MediaKind::Text,
        row.text,
        row.texts,
        None,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Image,
        row.image,
        row.images,
        row.image_uuids,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Audio,
        row.audio,
        row.audios,
        None,
        &role,
        parent,
        state,
    )?;
    append_modality(
        &mut turn,
        MediaKind::Video,
        row.video,
        row.videos,
        None,
        &role,
        parent,
        state,
    )?;
    if let Some(token_ids) = raw_token_ids {
        if !turn.content.is_empty() {
            return Err(DatasetError::Validation(
                "raw_token_ids cannot be combined with text or media fields".into(),
            ));
        }
        turn.input_tokens = Some(
            u64::try_from(token_ids.len())
                .map_err(|_| DatasetError::Validation("raw token count exceeds u64".into()))?,
        );
        let handle = state
            .segments
            .intern_token_ids(*parent, token_ids.into_boxed_slice())?;
        *parent = Some(handle);
        turn.body = Turn::dispatch_body(None, Some(handle), &[]);
    }
    let request_parent = *parent;
    if let Some(extra) = row.extra {
        let wire = serde_json::to_vec(&Value::Object(extra))?;
        turn.extra_body = Some(
            state
                .segments
                .intern_raw(request_parent, Bytes::from(wire))?,
        );
    }
    if let Some(headers) = row.extra_headers {
        turn.extra_headers = Some(state.segments.intern_raw(
            request_parent,
            Bytes::from(serde_json::to_vec(&Value::Object(headers))?),
        )?);
    }
    if let Some(parameters) = row.request_parameters {
        turn.request_parameters = Some(state.segments.intern_raw(
            request_parent,
            Bytes::from(serde_json::to_vec(&Value::Object(parameters))?),
        )?);
    }
    state.finalize_turn(&mut turn)?;
    Ok(turn)
}

fn token_native_sampling_max_tokens(extra: Option<&Map<String, Value>>) -> Result<Option<u32>> {
    let Some(value) = extra.and_then(|extra| extra.get("sampling_params")) else {
        return Ok(None);
    };
    let Some(sampling) = value.as_object() else {
        if value.is_null() {
            return Ok(None);
        }
        return Err(DatasetError::Validation(
            "sampling_params must be an object for a token-native row".into(),
        ));
    };
    let Some(value) = sampling.get("max_tokens") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or_else(|| {
            DatasetError::Validation(
                "sampling_params.max_tokens must be a positive unsigned 32-bit integer".into(),
            )
        })
}

fn take_extra_token_ids(extra: Option<&mut Map<String, Value>>) -> Result<Option<Vec<u32>>> {
    let Some(extra) = extra else {
        return Ok(None);
    };
    let value = match (extra.remove("raw_token_ids"), extra.remove("token_ids")) {
        (Some(_), Some(_)) => {
            return Err(DatasetError::Validation(
                "extra cannot contain both raw_token_ids and token_ids".into(),
            ));
        }
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => return Ok(None),
    };
    parse_token_ids(&value, "extra.token_ids").map(Some)
}

fn parse_token_ids(value: &Value, field: &str) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| {
            DatasetError::Validation(format!(
                "{field} must be a non-empty list of unsigned 32-bit integers"
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
                        "{field}[{index}] must be an unsigned 32-bit integer"
                    ))
                })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn append_modality(
    turn: &mut Turn,
    kind: MediaKind,
    singular: Option<String>,
    plural: Option<AuthoredBatch>,
    uuids: Option<Vec<String>>,
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
    // `image_uuids` is only accepted alongside the plain `AuthoredBatch::Strings`
    // form (see `normalize_image_uuids`), which always lowers to exactly one
    // group above, so attaching it to the first (only) group is unambiguous.
    let mut uuids = uuids;

    for group in groups {
        let mut handles = SmallVec::new();
        for content in group.contents {
            let handle = if kind == MediaKind::Text {
                let tokens = state.tokenizer.encode(&content)?;
                turn.input_tokens = Some(
                    turn.input_tokens
                        .unwrap_or(0)
                        .checked_add(tokens.len() as u64)
                        .ok_or_else(|| {
                            DatasetError::Validation("turn input token count overflowed u64".into())
                        })?,
                );
                state.segments.intern_text(
                    *parent,
                    role,
                    Bytes::from(content),
                    tokens.into_boxed_slice(),
                )?
            } else if content.is_empty() {
                // A UUID-only cache reference: no payload to resolve as a
                // local file/URL, just an empty content-addressed slot that
                // reads back as "" in `content_string`.
                state.segments.intern_media(*parent, kind, Bytes::new())?
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
            uuids: uuids.take().map(SmallVec::from_vec).unwrap_or_default(),
            handles,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::endpoints::{CreditPhase, EndpointConfig, ModelEndpoint};
    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, DatasetSource, LoaderRegistry};
    use crate::dataset::request::{
        BuiltinEndpointResolver, ConversationSession, EndpointRequestMaterializer, EndpointResolver,
    };
    use crate::dataset::segment::Payload;
    use crate::dataset::tokenizer::TiktokenTokenizer;
    use crate::dataset::{Overrides, RequestMaterializer};

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
    async fn image_uuids_normalize_and_thread_through_content_group() {
        let source = DatasetSource::Inline(json!([
            {"session_id":"s", "images":["http://a/img1.png", ""], "image_uuids":["uuid-1","uuid-2"]}
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
        let turn = &dataset.conversations()[0].turns[0];
        let group = &turn.content[0];
        assert_eq!(group.kind, MediaKind::Image);
        assert_eq!(group.uuids.as_slice(), ["uuid-1", "uuid-2"]);
        // The uuid-only slot (empty content) interns as an empty, valid segment
        // rather than erroring as a bad local-file path.
        let empty_handle = group.handles[1];
        match dataset.segments().get(empty_handle).unwrap() {
            Payload::Media { bytes, .. } => assert!(bytes.is_empty()),
            other => panic!("expected Media payload, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_uuids_reject_singular_image_field() {
        let source = DatasetSource::Inline(json!([
            {"session_id":"s", "image":"http://a/img.png", "image_uuids":["uuid-1"]}
        ]));
        let load = LoadConfig::new(source);
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ))
            .unwrap();
        let error = registry
            .build_dataset(
                Some("single-turn"),
                &load,
                &config(),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("singular image field"));
    }

    #[tokio::test]
    async fn uuid_and_strip_dedups_repeated_uuid_within_a_session_but_not_within_one_row() {
        let source = DatasetSource::Inline(json!([
            {"session_id":"s", "images":["http://a/img.png","http://a/img.png"], "image_uuids":["dup","dup"]},
            {"session_id":"s", "images":["http://a/img.png"], "image_uuids":["dup"]},
        ]));
        let load = LoadConfig::new(source);
        let mut compose = config();
        compose
            .format_options
            .insert("uuid_and_strip".into(), json!(true));
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
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turns = &dataset.conversations()[0].turns;
        // Both repeats within the first row keep their payload -- the server
        // resolves that request's cache misses before populating its cache.
        for handle in &turns[0].content[0].handles {
            match dataset.segments().get(*handle).unwrap() {
                Payload::Media { bytes, .. } => assert!(!bytes.is_empty()),
                other => panic!("expected Media payload, got {other:?}"),
            }
        }
        // The second row's repeat of a uuid observed in an earlier row is stripped.
        let handle = turns[1].content[0].handles[0];
        match dataset.segments().get(handle).unwrap() {
            Payload::Media { bytes, .. } => assert!(bytes.is_empty()),
            other => panic!("expected Media payload, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn uuid_and_strip_is_rejected_for_multi_turn_loader() {
        let source = DatasetSource::Inline(json!([{"session_id":"s","turns":[{"text":"hi"}]}]));
        let mut load = LoadConfig::new(source);
        load.options.insert("uuid_and_strip".into(), json!(true));
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(MultiTurnDatasetLoader),
                Arc::new(MultiTurnComposer),
            ))
            .unwrap();
        let error = registry
            .build_dataset(None, &load, &config(), &TiktokenTokenizer::builtin())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("uuid-and-strip"));
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

    #[tokio::test]
    async fn custom_turn_dispatch_fields_reach_the_resolved_endpoint_request() {
        let source = DatasetSource::Inline(json!([{
            "session_id":"dispatch",
            "text":"hello",
            "model":"turn-model",
            "endpoint":"responses",
            "streaming":false,
            "output_length":7,
            "extra_headers":{"x-custom":"yes"},
            "request_parameters":{"api-version":"2026-07"}
        }]));
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ))
            .unwrap();
        let dataset = Arc::new(
            registry
                .build_dataset(
                    Some("single_turn"),
                    &LoadConfig::new(source),
                    &config(),
                    &TiktokenTokenizer::builtin(),
                )
                .await
                .unwrap(),
        );
        let mut session = ConversationSession::new(dataset, SessionId::from("dispatch")).unwrap();
        session.advance_to(0).unwrap();
        let resolver = BuiltinEndpointResolver::default();
        let endpoint = resolver
            .resolve(session.endpoint_override().unwrap())
            .unwrap();
        let mut endpoint_config = EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        };
        endpoint_config.endpoint_type = endpoint.descriptor().legacy_type().expect("endpoint type");
        let request = EndpointRequestMaterializer
            .materialize(
                &session,
                endpoint.as_ref(),
                &ModelEndpoint {
                    primary_model_name: "default-model".into(),
                    endpoint: endpoint_config,
                },
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"], "turn-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_output_tokens"], 7);
        assert_eq!(request.endpoint_path.as_deref(), Some("/v1/responses"));
        assert_eq!(request.headers["x-custom"], "yes");
        assert_eq!(request.parameters["api-version"], "2026-07");
    }

    #[tokio::test]
    async fn streaming_path_survives_a_formatter_without_a_stream_body_field() {
        let source = DatasetSource::Inline(json!([{
            "session_id":"tgi",
            "endpoint":"huggingface_generate",
            "text":"hello",
            "streaming":true,
            "output_length":2
        }]));
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ))
            .unwrap();
        let dataset = Arc::new(
            registry
                .build_dataset(
                    Some("single_turn"),
                    &LoadConfig::new(source),
                    &config(),
                    &TiktokenTokenizer::builtin(),
                )
                .await
                .unwrap(),
        );
        let mut session = ConversationSession::new(dataset, SessionId::from("tgi")).unwrap();
        session.advance_to(0).unwrap();
        let endpoint = BuiltinEndpointResolver::default()
            .resolve(session.endpoint_override().unwrap())
            .unwrap();
        let request = EndpointRequestMaterializer
            .materialize(
                &session,
                endpoint.as_ref(),
                &ModelEndpoint {
                    primary_model_name: "model".into(),
                    endpoint: EndpointConfig {
                        streaming: true,
                        ..EndpointConfig::default()
                    },
                },
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert!(body.get("stream").is_none());
        assert!(request.streaming);
        assert_eq!(request.endpoint_path.as_deref(), Some("/generate_stream"));
    }

    #[tokio::test]
    async fn token_native_composition_promotes_and_validates_extra_token_ids() {
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ))
            .unwrap();
        let mut compose = config();
        compose.requires_raw_token_ids = true;
        let dataset = registry
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([{
                    "extra": {
                        "token_ids": [1, 2, 3],
                        "sampling_params": {"temperature": 0, "max_tokens": 9}
                    },
                    "output_length": 7
                }]))),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();

        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.input_tokens, Some(3));
        assert_eq!(turn.max_tokens, Some(9));
        let token_handle = *turn.body.first().expect("token handle");
        let Payload::TokenIds { token_ids } = dataset.segments().get(token_handle).unwrap() else {
            panic!("promoted token IDs must use the token segment domain")
        };
        assert_eq!(&**token_ids, &[1, 2, 3]);
        let extra = turn.extra_body.expect("remaining extra body");
        let Payload::Raw { wire } = dataset.segments().get(extra).unwrap() else {
            panic!("remaining sampling parameters must be raw JSON")
        };
        let extra: Value = serde_json::from_slice(wire).unwrap();
        assert!(extra.get("token_ids").is_none());
        assert_eq!(extra["sampling_params"]["temperature"], 0);
        assert_eq!(extra["sampling_params"]["max_tokens"], 9);

        for invalid in [
            json!([]),
            json!([true]),
            json!([-1]),
            json!([4_294_967_296_u64]),
        ] {
            let result = registry
                .build_dataset(
                    Some("single_turn"),
                    &LoadConfig::new(DatasetSource::Inline(json!([{
                        "extra": {"token_ids": invalid}
                    }]))),
                    &compose,
                    &TiktokenTokenizer::builtin(),
                )
                .await;
            assert!(result.is_err());
        }
        for invalid in [
            json!({"extra": {"token_ids": [1], "sampling_params": []}}),
            json!({
                "extra": {
                    "token_ids": [1],
                    "sampling_params": {"max_tokens": 0}
                }
            }),
        ] {
            let result = registry
                .build_dataset(
                    Some("single_turn"),
                    &LoadConfig::new(DatasetSource::Inline(json!([invalid]))),
                    &compose,
                    &TiktokenTokenizer::builtin(),
                )
                .await;
            assert!(result.is_err());
        }
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
