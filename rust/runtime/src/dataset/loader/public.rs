// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public/Hugging Face, ShareGPT, and accuracy dataset formats.
//!
//! Remote Hugging Face rows use the
//! documented Dataset Viewer `/rows` API in pages of at most 100. Revision-pinned
//! sources resolve the Hub revision to an immutable commit and decode the
//! repository's Parquet/JSON/JSONL/CSV artifacts directly.

use crate::endpoints::extract_payload;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use image::ColorType;
use image::codecs::jpeg::JpegEncoder;
#[cfg(feature = "parquet")]
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use serde_json::{Map, Value};
use smallvec::{SmallVec, smallvec};

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::dataset::model::{
    AccuracyAssociation, ContentGroup, Conversation, CorrelationId, MediaKind, SessionId, Turn,
};
use crate::dataset::segment::{Role, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

/// Accuracy benchmark problem loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct AccuracyDatasetLoader;
/// Accuracy problem composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct AccuracyComposer;
/// ShareGPT JSON loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShareGptDatasetLoader;
/// ShareGPT prompt/completion-pair composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShareGptComposer;
/// Configurable flat Hugging Face instruction-row loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfInstructionDatasetLoader;
/// Flat Hugging Face instruction-row composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfInstructionComposer;
/// Hugging Face conversation-array loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfConversationDatasetLoader;
/// Hugging Face conversation-array composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfConversationComposer;
/// MT-Bench prompt-list loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtBenchDatasetLoader;
/// MT-Bench multi-turn composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtBenchComposer;
/// MMVU video-question loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmvuDatasetLoader;
/// MMVU video-question composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmvuComposer;
/// Spec-Bench JSONL loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecBenchDatasetLoader;
/// Spec-Bench composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecBenchComposer;
/// SPEED-Bench JSONL loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeedBenchDatasetLoader;
/// SPEED-Bench composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeedBenchComposer;

macro_rules! public_loader {
    ($ty:ty, $name:literal, $probe:expr) => {
        #[async_trait]
        impl DatasetLoader for $ty {
            fn name(&self) -> &str {
                $name
            }

            fn can_load(&self, probe: &DatasetProbe) -> bool {
                probe.value.as_ref().is_some_and($probe)
            }

            async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
                load_public_rows(config).await
            }

            fn preferred_sampling_strategy(&self) -> &str {
                "sequential"
            }
        }
    };
}

public_loader!(AccuracyDatasetLoader, "accuracy", |value: &Value| {
    value.get("prompt").is_some() && value.get("task").is_some()
});
public_loader!(ShareGptDatasetLoader, "sharegpt", |value: &Value| {
    value.get("conversations").is_some_and(Value::is_array)
});
public_loader!(
    HfInstructionDatasetLoader,
    "hf_instruction_response",
    |value: &Value| { value.get("__aiperf_hf_instruction").is_some() }
);
public_loader!(
    HfConversationDatasetLoader,
    "hf_conversation",
    |value: &Value| { value.get("__aiperf_hf_conversation").is_some() }
);
public_loader!(MtBenchDatasetLoader, "mt_bench", |value: &Value| {
    value.get("prompt").is_some_and(Value::is_array)
});
public_loader!(MmvuDatasetLoader, "mmvu", |value: &Value| {
    value.get("question").is_some() && value.get("choices").is_some()
});
public_loader!(SpecBenchDatasetLoader, "spec_bench", |value: &Value| {
    value.get("turns").is_some_and(Value::is_array) && value.get("session_id").is_none()
});
public_loader!(SpeedBenchDatasetLoader, "speed_bench", |value: &Value| {
    is_speed_bench(value)
});

impl Composer for AccuracyComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        if rows.is_empty() {
            return Err(DatasetError::Validation(
                "accuracy benchmark returned zero problems".into(),
            ));
        }
        let system_prompt = string_option(config, "accuracy_system_prompt");
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(rows.len());

        // Tokenizing the prompt (and re-tokenizing it inside the composed
        // `messages` body for `input_tokens`) is the dominant per-row cost and
        // touches only `row.value`/`system_prompt`, not the shared, sequentially
        // interned `SegmentPool` or the session-id generator - precompute it in
        // parallel, then do the (necessarily sequential, since ids/segments are
        // shared mutable state) main pass using the precomputed values.
        struct Prepared {
            message_wire: Bytes,
            prompt_text: String,
            prompt_tokens: Vec<u32>,
            input_tokens: u64,
            generation_size: u32,
        }
        let prepared: Vec<Prepared> = rows
            .par_iter()
            .map(|row| -> Result<Prepared> {
                let object = require_object(&row.value, &row.origin)?;
                let prompt = required_string(object, "prompt", &row.origin)?;
                let mut messages = object
                    .get("raw_messages")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([{"role":"user","content":prompt}]));
                let messages_array = messages.as_array_mut().ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{}: raw_messages must be an array",
                        row.origin
                    ))
                })?;
                if let Some(system) = &system_prompt {
                    messages_array.insert(0, serde_json::json!({"role":"system","content":system}));
                }
                validate_message_values(messages_array, &row.origin)?;
                let message_wire = Bytes::from(serde_json::to_vec(&messages)?);
                let prompt_text = system_prompt.as_ref().map_or_else(
                    || prompt.to_string(),
                    |system| format!("{system}\n\n{prompt}"),
                );
                let prompt_tokens = tokenizer.encode(&prompt_text)?;
                let body = serde_json::json!({"messages": messages});
                let input_tokens = extracted_token_count(&body, tokenizer)?;
                let generation_size = object
                    .get("metadata")
                    .and_then(Value::as_object)
                    .and_then(|metadata| metadata.get("generation_size"))
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .or_else(|| {
                        object
                            .get("generation_size")
                            .and_then(Value::as_u64)
                            .and_then(|value| u32::try_from(value).ok())
                    })
                    .unwrap_or(100);
                if generation_size == 0 {
                    return Err(DatasetError::Validation(format!(
                        "{}: generation_size must be positive",
                        row.origin
                    )));
                }
                Ok(Prepared {
                    message_wire,
                    prompt_text,
                    prompt_tokens,
                    input_tokens,
                    generation_size,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        for (row, prepared) in rows.into_iter().zip(prepared) {
            let object = require_object(&row.value, &row.origin)?;
            let task = required_string(object, "task", &row.origin)?;
            let session_id = object
                .get("session_id")
                .and_then(Value::as_str)
                .map(SessionId::from)
                .unwrap_or_else(|| ids.next_id());
            let correlation_id = object
                .get("correlation_id")
                .and_then(Value::as_str)
                .map(CorrelationId::from)
                .unwrap_or_else(|| CorrelationId::from(session_id.as_str()));
            let raw_messages = segments.intern_raw(None, prepared.message_wire)?;
            let text = segments.intern_text(
                Some(raw_messages),
                "user",
                Bytes::from(prepared.prompt_text),
                prepared.prompt_tokens.into_boxed_slice(),
            )?;
            let extra_body = object
                .get("extra_body")
                .map(|value| {
                    if !value.is_object() {
                        return Err(DatasetError::Validation(format!(
                            "{}: extra_body must be an object",
                            row.origin
                        )));
                    }
                    segments.intern_raw(Some(text), Bytes::from(serde_json::to_vec(value)?))
                })
                .transpose()?;
            let input_tokens = Some(prepared.input_tokens);
            let generation_size = prepared.generation_size;
            let mut turn = Turn {
                role: Some(Role::from("user")),
                max_tokens: Some(generation_size),
                input_tokens,
                raw_messages: Some(raw_messages),
                extra_body,
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                    uuids: smallvec![],
                }],
                ..Turn::default()
            };
            finalizer.finalize_turn(&mut turn)?;
            let mut conversation = Conversation::new(session_id);
            conversation.turns.push(turn);
            conversation.accuracy = Some(AccuracyAssociation {
                correlation_id,
                task: task.to_string(),
            });
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

impl Composer for ShareGptComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let min_length = usize_option(config, "min_sequence_tokens", 4)?;
        let max_prompt = usize_option(config, "max_prompt_tokens", 1024)?;
        let max_total = usize_option(config, "max_total_tokens", 2048)?;
        let skip_min_output = config.output_length_distribution.is_some();
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();

        // Tokenizing every prompt/completion pair (and the min/max-length
        // validity check that depends on those token counts) touches only the
        // row's own JSON, not the shared segment pool or the session-id
        // generator - precompute the whole per-row pair list (or `None` for a
        // row that gets skipped) in parallel, then do the sequential
        // segments/ids pass using the precomputed result.
        let prepared_rows: Vec<Option<Vec<(String, Vec<u32>, u32)>>> = rows
            .par_iter()
            .map(|row| -> Result<Option<Vec<(String, Vec<u32>, u32)>>> {
                let Some(messages) = row.value.get("conversations").and_then(Value::as_array)
                else {
                    return Ok(None);
                };
                let pairs = sharegpt_pairs(messages);
                if pairs.is_empty() {
                    return Ok(None);
                }
                let mut prepared = Vec::with_capacity(pairs.len());
                for (prompt, completion) in pairs {
                    let prompt_tokens = tokenizer.encode(&prompt)?;
                    let completion_tokens = tokenizer.encode(&completion)?;
                    if prompt_tokens.len() < min_length
                        || prompt_tokens.len() > max_prompt
                        || (!skip_min_output && completion_tokens.len() < min_length)
                        || prompt_tokens.len() + completion_tokens.len() > max_total
                    {
                        return Ok(None);
                    }
                    let completion_tokens =
                        u32::try_from(completion_tokens.len()).map_err(|_| {
                            DatasetError::Validation(
                                "ShareGPT completion length exceeds the u32 request limit".into(),
                            )
                        })?;
                    prepared.push((prompt, prompt_tokens, completion_tokens));
                }
                if prepared.is_empty() {
                    return Ok(None);
                }
                Ok(Some(prepared))
            })
            .collect::<Result<Vec<_>>>()?;

        for prepared in prepared_rows.into_iter().flatten() {
            let mut conversation = Conversation::new(ids.next_id());
            let mut parent = None;
            for (prompt, tokens, output_tokens) in prepared {
                let input_tokens = Some(tokens.len() as u64);
                let handle = segments.intern_text(
                    parent,
                    "user",
                    Bytes::from(prompt),
                    tokens.into_boxed_slice(),
                )?;
                parent = Some(handle);
                let mut turn = Turn {
                    max_tokens: Some(output_tokens.max(1)),
                    input_tokens,
                    content: smallvec![ContentGroup {
                        kind: MediaKind::Text,
                        name: String::new(),
                        handles: smallvec![handle],
                        uuids: smallvec![],
                    }],
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

impl Composer for HfInstructionComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let prompt_column = string_option(config, "prompt_column")
            .ok_or_else(|| DatasetError::Validation("prompt_column is required".into()))?;
        let prompt_template = string_option(config, "prompt_template");
        let image_column = string_option(config, "image_column");
        let video_column = string_option(config, "video_column");
        let audio_column = string_option(config, "audio_column");
        if prompt_template.is_none()
            && let Some(first) = rows.first()
        {
            let object = require_object(&first.value, &first.origin)?;
            if !object.contains_key(&prompt_column) {
                return Err(DatasetError::Validation(format!(
                    "{}: prompt column {prompt_column:?} is missing; available columns: {}",
                    first.origin,
                    object.keys().cloned().collect::<Vec<_>>().join(", ")
                )));
            }
        }
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();
        for row in rows {
            let object = require_object(&row.value, &row.origin)?;
            let prompt = match &prompt_template {
                Some(template) => format_template(template, object, &row.origin)?,
                None => object
                    .get(&prompt_column)
                    .filter(|value| !value.is_null())
                    .map(value_text)
                    .transpose()?
                    .unwrap_or_default(),
            };
            if prompt.trim().is_empty() {
                continue;
            }
            let mut groups = vec![AuthoredMedia {
                kind: MediaKind::Text,
                name: String::new(),
                contents: vec![prompt],
            }];
            if let Some(column) = &image_column {
                groups.extend(media_from_value(object.get(column), MediaKind::Image)?);
            }
            if let Some(column) = &video_column {
                groups.extend(media_from_value(object.get(column), MediaKind::Video)?);
            }
            if let Some(column) = &audio_column {
                groups.extend(media_from_value(object.get(column), MediaKind::Audio)?);
            }
            let turn =
                compose_media_turn(groups, config, tokenizer, segments, &mut finalizer, None)?;
            let mut conversation = Conversation::new(ids.next_id());
            conversation.turns.push(turn);
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

/// Arbitrary-Hugging-Face auto-detecting loader (format id `hf`). Fetches rows
/// via the shared public path; the paired [`HfAutoComposer`] does the column
/// inference. `can_load` never matches (HF sources present an empty probe), so
/// this format is only ever reached by an explicit `format: "hf"`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfAutoDatasetLoader;

#[async_trait]
impl DatasetLoader for HfAutoDatasetLoader {
    fn name(&self) -> &str {
        "hf"
    }
    fn can_load(&self, _probe: &DatasetProbe) -> bool {
        false
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        load_public_rows(config).await
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

/// Layout-inferring composer for arbitrary HF datasets. Handles the flat,
/// joined, and message prompt shapes reported by [`super::hf_detect`], sizing
/// the output from the reference completion (or a fixed `output_len` override)
/// and filtering rows by token budget.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfAutoComposer;

impl Composer for HfAutoComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        use super::hf_detect::{RowLayout, infer_row_layout};

        let first = rows
            .first()
            .ok_or_else(|| DatasetError::Validation("HF dataset returned zero rows".into()))?;
        let override_col =
            string_option(config, "text_column").or_else(|| string_option(config, "prompt_column"));
        let output_override = string_option(config, "output_column");
        let layout = infer_row_layout(&first.value, override_col.as_deref())
            .map_err(DatasetError::Validation)?;

        // `output_len` accepts either a JSON number or a numeric string.
        let fixed_output = string_option(config, "output_len")
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| {
                usize_option(config, "output_len", 0)
                    .ok()
                    .filter(|n| *n > 0)
                    .map(|n| n as u32)
            });
        let min_length = usize_option(config, "min_sequence_tokens", 4)?;
        let max_prompt = usize_option(config, "max_prompt_tokens", 1024)?;
        let max_total = usize_option(config, "max_total_tokens", 2048)?;

        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();

        for row in &rows {
            let Some(obj) = row.value.as_object() else {
                continue;
            };
            let (prompt, completion): (String, Option<String>) = match &layout {
                RowLayout::Prompt {
                    prompt_field,
                    completion_field,
                } => {
                    let raw = obj.get(prompt_field.as_str());
                    let prompt = if prompt_field == "turns" {
                        raw.and_then(Value::as_array)
                            .and_then(|a| a.first())
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string()
                    } else {
                        raw.and_then(Value::as_str).unwrap_or_default().to_string()
                    };
                    let completion = output_override
                        .as_ref()
                        .or(completion_field.as_ref())
                        .and_then(|c| {
                            obj.get(c.as_str()).and_then(|v| {
                                v.as_array()
                                    .and_then(|a| a.first())
                                    .and_then(Value::as_str)
                                    .or_else(|| v.as_str())
                                    .map(str::to_string)
                            })
                        });
                    (prompt, completion)
                }
                RowLayout::Joined {
                    fields,
                    completion_field,
                } => {
                    let parts: Vec<String> = fields
                        .iter()
                        .filter_map(|c| {
                            obj.get(c.as_str())
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .collect();
                    if parts.is_empty() {
                        continue;
                    }
                    let completion = output_override
                        .as_ref()
                        .or(completion_field.as_ref())
                        .and_then(|c| {
                            obj.get(c.as_str())
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        });
                    (parts.join("\n\n"), completion)
                }
                RowLayout::Messages(field) => {
                    let msgs = obj
                        .get(field.as_str())
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    match super::hf_detect::first_user_message(&msgs) {
                        Some(p) => (p, super::hf_detect::first_assistant_message(&msgs)),
                        None => continue,
                    }
                }
            };

            if prompt.trim().is_empty() {
                continue;
            }
            let prompt_tokens = tokenizer.encode(&prompt)?;
            if prompt_tokens.len() < min_length || prompt_tokens.len() > max_prompt {
                continue;
            }

            let output_tokens: u32 = if let Some(n) = fixed_output {
                n
            } else if let Some(comp) = &completion {
                let c = tokenizer.encode(comp)?.len();
                if c == 0 { 128 } else { c as u32 }
            } else {
                128
            };
            if prompt_tokens.len() + output_tokens as usize > max_total {
                continue;
            }

            let input_tokens = Some(prompt_tokens.len() as u64);
            let handle = segments.intern_text(
                None,
                "user",
                Bytes::from(prompt),
                prompt_tokens.into_boxed_slice(),
            )?;
            let mut turn = Turn {
                max_tokens: Some(output_tokens.max(1)),
                input_tokens,
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
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

        if conversations.is_empty() {
            return Err(DatasetError::Validation(
                "no valid samples after processing HF dataset; try --hf-text-column or a different subset/split".into(),
            ));
        }
        Ok(conversations)
    }
}

impl Composer for HfConversationComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let column = string_option(config, "conversation_column")
            .ok_or_else(|| DatasetError::Validation("conversation_column is required".into()))?;
        let content_key =
            string_option(config, "message_content_key").unwrap_or_else(|| "content".into());
        let image_column = string_option(config, "image_column");
        let multi_turn = bool_option(config, "multi_turn", false)?;
        let min_length = usize_option(config, "min_sequence_tokens", 4)?;
        let max_prompt = usize_option(config, "max_prompt_tokens", 1024)?;
        let max_total = usize_option(config, "max_total_tokens", 2048)?;
        let skip_min_output = config.output_length_distribution.is_some();
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();

        // The `multi_turn` prompt/completion-pair tokenization + validity
        // check below touches only the row's own JSON, not the shared segment
        // pool or the session-id generator - precompute it in parallel for
        // every row up front (`single_turn` rows are cheap and stay serial).
        let multi_turn_prompts: Vec<Option<Vec<(String, Option<u32>)>>> = if multi_turn {
            rows.par_iter()
                .map(|row| -> Result<Option<Vec<(String, Option<u32>)>>> {
                    let Some(object) = row.value.as_object() else {
                        return Ok(None);
                    };
                    let Some(messages) = object.get(&column).and_then(Value::as_array) else {
                        return Ok(None);
                    };
                    let normalized = normalize_hf_messages(messages);
                    let mut prepared = Vec::new();
                    for (prompt, completion) in hf_message_pairs(&normalized, &content_key) {
                        let prompt_tokens = tokenizer.encode(&prompt)?.len();
                        let completion_tokens = tokenizer.encode(&completion)?.len();
                        if prompt_tokens < min_length
                            || prompt_tokens > max_prompt
                            || (!skip_min_output && completion_tokens < min_length)
                            || prompt_tokens + completion_tokens > max_total
                        {
                            return Ok(None);
                        }
                        prepared.push((
                            prompt,
                            Some(u32::try_from(completion_tokens).map_err(|_| {
                                DatasetError::Validation(
                                    "HF conversation completion length exceeds u32".into(),
                                )
                            })?),
                        ));
                    }
                    Ok(Some(prepared))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        for (row_index, row) in rows.into_iter().enumerate() {
            let object = require_object(&row.value, &row.origin)?;
            let Some(messages) = object.get(&column).and_then(Value::as_array) else {
                continue;
            };
            let normalized = normalize_hf_messages(messages);
            let prompts = if multi_turn {
                multi_turn_prompts[row_index].clone().unwrap_or_default()
            } else {
                first_user_message(&normalized, &content_key)
                    .into_iter()
                    .map(|prompt| (prompt, None))
                    .collect()
            };
            if prompts.is_empty() {
                continue;
            }
            let first_images = match &image_column {
                Some(column) => media_from_value(object.get(column), MediaKind::Image)?,
                None => Vec::new(),
            };
            let mut conversation = Conversation::new(ids.next_id());
            for (index, (prompt, max_tokens)) in prompts.into_iter().enumerate() {
                let mut groups = vec![AuthoredMedia {
                    kind: MediaKind::Text,
                    name: String::new(),
                    contents: vec![prompt],
                }];
                if index == 0 {
                    groups.extend(first_images.clone());
                }
                conversation.turns.push(compose_media_turn(
                    groups,
                    config,
                    tokenizer,
                    segments,
                    &mut finalizer,
                    max_tokens,
                )?);
            }
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

impl Composer for MtBenchComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        if let Some(first) = rows.first() {
            let object = require_object(&first.value, &first.origin)?;
            if !object.contains_key("prompt") {
                return Err(DatasetError::Validation(format!(
                    "{}: MT-Bench prompt column is missing; available columns: {}",
                    first.origin,
                    object.keys().cloned().collect::<Vec<_>>().join(", ")
                )));
            }
        }
        compose_prompt_lists(rows, "prompt", true, config, tokenizer, segments)
    }
}

impl Composer for SpecBenchComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let multi_turn = bool_option(config, "multi_turn", false)?;
        compose_prompt_lists(rows, "turns", multi_turn, config, tokenizer, segments)
    }
}

impl Composer for MmvuComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let video_column = string_option(config, "video_column").unwrap_or_else(|| "video".into());
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();
        for row in rows {
            let object = require_object(&row.value, &row.origin)?;
            let question = object
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let choices = object
                .get("choices")
                .and_then(Value::as_object)
                .map(|choices| {
                    choices
                        .iter()
                        .filter_map(|(key, value)| {
                            value.as_str().map(|value| format!("{key}.{value}"))
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let prompt = match (question.is_empty(), choices.is_empty()) {
                (false, false) => format!("{question} {choices}"),
                (false, true) => question.to_string(),
                (true, false) => choices,
                (true, true) => continue,
            };
            let videos = media_from_value(object.get(&video_column), MediaKind::Video)?;
            if videos.is_empty() {
                continue;
            }
            let mut groups = vec![AuthoredMedia {
                kind: MediaKind::Text,
                name: String::new(),
                contents: vec![prompt],
            }];
            groups.extend(videos);
            let mut conversation = Conversation::new(ids.next_id());
            conversation.turns.push(compose_media_turn(
                groups,
                config,
                tokenizer,
                segments,
                &mut finalizer,
                None,
            )?);
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

impl Composer for SpeedBenchComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let category = string_option(config, "category");
        let multi_turn = bool_option(config, "multi_turn", true)?;
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();

        // Tokenizing every selected message is the dominant per-row cost and
        // touches only the row's own JSON, not the shared segment pool (there
        // is no session-id generator here - `id` comes straight from the row)
        // - precompute the ordered (role, content, tokens) list for every row
        // in parallel, then do the sequential intern pass.
        struct Prepared<'a> {
            id: &'a str,
            turns: Vec<(&'a str, &'a str, Vec<u32>)>,
        }
        let prepared_rows: Vec<Option<Prepared>> = rows
            .par_iter()
            .map(|row| -> Result<Option<Prepared>> {
                if !is_speed_bench(&row.value) {
                    return Err(DatasetError::Validation(format!(
                        "{}: invalid SPEED-Bench row",
                        row.origin
                    )));
                }
                if category.as_ref().is_some_and(|category| {
                    row.value.get("category").and_then(Value::as_str) != Some(category)
                }) {
                    return Ok(None);
                }
                let id = row.value["question_id"].as_str().unwrap();
                let messages = row.value["messages"].as_array().unwrap();
                let selected = if multi_turn {
                    messages.as_slice()
                } else {
                    &messages[..1]
                };
                let mut turns = Vec::with_capacity(selected.len());
                for message in selected {
                    let role = message["role"].as_str().unwrap();
                    let content = message["content"].as_str().unwrap();
                    let tokens = tokenizer.encode(content)?;
                    turns.push((role, content, tokens));
                }
                Ok(Some(Prepared { id, turns }))
            })
            .collect::<Result<Vec<_>>>()?;

        for prepared in prepared_rows.into_iter().flatten() {
            let mut conversation = Conversation::new(prepared.id);
            let mut parent = None;
            for (role, content, tokens) in prepared.turns {
                let input_tokens = Some(tokens.len() as u64);
                let handle = segments.intern_text(
                    parent,
                    role,
                    Bytes::copy_from_slice(content.as_bytes()),
                    tokens.into_boxed_slice(),
                )?;
                parent = Some(handle);
                let mut turn = Turn {
                    role: Some(Role::from(role)),
                    input_tokens,
                    content: smallvec![ContentGroup {
                        kind: MediaKind::Text,
                        name: String::new(),
                        handles: smallvec![handle],
                        uuids: smallvec![],
                    }],
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

/// Acquire one raw local or remote row stream without selecting a linear composer.
///
/// Direct Graph-IR input adapters use this source seam for URL and Hugging Face
/// acquisition, then retain sole ownership of their format-specific validation
/// and lowering. No [`DatasetLoader`] registry or linear [`Conversation`]
/// representation is introduced on that path.
pub async fn load_raw_rows(config: &LoadConfig) -> Result<Vec<RawRow>> {
    config.validate()?;
    load_public_rows(config).await
}

pub(crate) async fn load_public_rows(config: &LoadConfig) -> Result<Vec<RawRow>> {
    match &config.source {
        DatasetSource::Url(url) => {
            let bytes = config
                .fetcher
                .fetch(url, url, config.bearer_token.as_deref())
                .await?;
            rows_from_remote_bytes(bytes, url, None)
        }
        DatasetSource::HuggingFace {
            dataset,
            config: subset,
            split,
            max_rows,
            revision,
        } => match revision {
            Some(revision) => {
                load_hugging_face_revision_rows(config, dataset, subset, split, *max_rows, revision)
                    .await
            }
            None => {
                let (subset, split) =
                    resolve_hf_coordinates(config, dataset, subset, split).await?;
                load_hugging_face_rows(config, dataset, &subset, &split, *max_rows).await
            }
        },
        source => jsonl_or_json_rows(source),
    }
}

/// Pick concrete (config, split) from a parsed `/info` `dataset_info` map,
/// resolving empty user inputs. Config: explicit if present (else `default`, else
/// first key). Split: explicit if present (else train > test > validation > first).
fn pick_hf_coordinates(
    dataset_info: &serde_json::Map<String, Value>,
    want_subset: &str,
    want_split: &str,
) -> Result<(String, String)> {
    let config = if !want_subset.is_empty() {
        if !dataset_info.contains_key(want_subset) {
            return Err(DatasetError::Validation(format!(
                "config {want_subset:?} not found; available: {}",
                dataset_info.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        want_subset.to_string()
    } else if dataset_info.contains_key("default") {
        "default".to_string()
    } else {
        dataset_info
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| DatasetError::Validation("dataset has no configs".into()))?
    };

    let splits = dataset_info
        .get(&config)
        .and_then(|c| c.get("splits"))
        .and_then(Value::as_object);
    let split = if !want_split.is_empty() {
        if let Some(s) = splits
            && !s.contains_key(want_split)
        {
            return Err(DatasetError::Validation(format!(
                "split {want_split:?} not found in config {config:?}; available: {}",
                s.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        want_split.to_string()
    } else if let Some(s) = splits {
        ["train", "test", "validation"]
            .iter()
            .find(|p| s.contains_key(**p))
            .map(|p| (*p).to_string())
            .or_else(|| s.keys().next().cloned())
            .unwrap_or_else(|| "train".to_string())
    } else {
        "train".to_string()
    };
    Ok((config, split))
}

/// Resolve HF coordinates, calling the Dataset Viewer `/info` endpoint only when
/// subset or split is empty. Pinned revisions never reach this path.
async fn resolve_hf_coordinates(
    config: &LoadConfig,
    dataset: &str,
    subset: &str,
    split: &str,
) -> Result<(String, String)> {
    if !subset.is_empty() && !split.is_empty() {
        return Ok((subset.to_string(), split.to_string()));
    }
    let encoded: String = url::form_urlencoded::byte_serialize(dataset.as_bytes()).collect();
    let info_url = format!("https://datasets-server.huggingface.co/info?dataset={encoded}");
    let bytes = config
        .fetcher
        .fetch(
            &info_url,
            &format!("hf-info:{dataset}"),
            config.bearer_token.as_deref(),
        )
        .await?;
    let info: Value = serde_json::from_slice(&bytes)
        .map_err(|e| DatasetError::Validation(format!("HF /info for {dataset} is invalid: {e}")))?;
    let dataset_info = info
        .get("dataset_info")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            DatasetError::Validation(format!(
                "HF /info for {dataset} has no dataset_info (missing/gated?)"
            ))
        })?;
    pick_hf_coordinates(dataset_info, subset, split)
}

async fn load_hugging_face_revision_rows(
    config: &LoadConfig,
    dataset: &str,
    subset: &str,
    split: &str,
    max_rows: Option<usize>,
    revision: &str,
) -> Result<Vec<RawRow>> {
    validate_hugging_face_coordinates(dataset, subset, split, max_rows)?;
    if revision.is_empty() {
        return Err(DatasetError::Validation(
            "Hugging Face revision cannot be empty".into(),
        ));
    }
    let info_url = hugging_face_url(&["api", "datasets", dataset, "revision", revision])?;
    let cache_key = format!("hf-revision:{dataset}:{revision}");
    let body = config
        .fetcher
        .fetch(
            info_url.as_str(),
            &cache_key,
            config.bearer_token.as_deref(),
        )
        .await?;
    let info: Value = serde_json::from_slice(&body).map_err(|error| {
        DatasetError::Validation(format!(
            "Hugging Face revision metadata for {dataset}@{revision} is invalid: {error}"
        ))
    })?;
    let commit = info.get("sha").and_then(Value::as_str).ok_or_else(|| {
        DatasetError::Validation(format!(
            "Hugging Face revision metadata for {dataset}@{revision} has no commit sha"
        ))
    })?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DatasetError::Validation(format!(
            "Hugging Face returned invalid commit sha {commit:?} for {dataset}@{revision}"
        )));
    }
    let siblings = info
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DatasetError::Validation(format!(
                "Hugging Face revision metadata for {dataset}@{revision} has no siblings"
            ))
        })?;
    // Prefer the dataset's declared `cardData.data_files` mapping: it maps
    // (config, split) -> file path directly, so files not named after the split
    // (e.g. a single root `traces.jsonl` mapped to split `train`) resolve. Fall
    // back to matching the split by file name for datasets that omit the mapping.
    let mut files = card_data_files(&info, subset, split);
    if files.is_empty() {
        files = siblings
            .iter()
            .filter_map(|sibling| sibling.get("rfilename").and_then(Value::as_str))
            .filter(|path| supported_tabular_extension(path).is_some())
            .filter(|path| path_matches_split(path, split))
            .map(str::to_string)
            .collect::<Vec<_>>();
        if subset != "default" {
            let subset_files = files
                .iter()
                .filter(|path| path_matches_subset(path, subset))
                .cloned()
                .collect::<Vec<_>>();
            if !subset_files.is_empty() {
                files = subset_files;
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(DatasetError::Validation(format!(
            "Hugging Face revision {dataset}@{commit} contains no supported Parquet/JSON/JSONL/CSV files for config {subset:?}, split {split:?}"
        )));
    }

    let mut rows = Vec::new();
    for path in files {
        if max_rows.is_some_and(|cap| rows.len() >= cap) {
            break;
        }
        let remaining = max_rows.map(|cap| cap.saturating_sub(rows.len()));
        let label = format!("hf://{dataset}@{commit}/{path}");

        // A shard (multi-GB parquet like voxpopuli, or one big `traces.jsonl` like
        // weka) can dwarf the few rows a run needs. Match Python's `datasets`: pull
        // the file through `hf-hub` (cached, resumable, xet-accelerated, shared with
        // the HF cache) and read only the first `remaining` rows from the local file
        // — `BufReader` lines for jsonl, seeked row groups for parquet — never
        // loading or parsing the whole shard. Injected fetchers opt out so tests
        // and custom caches stay the exclusive download path.
        let streamed = if config.fetcher.allows_hf_hub_streaming() {
            load_hf_file_streaming(dataset, commit, &path, remaining, &label).await?
        } else {
            None
        };
        match streamed {
            Some(streamed) => rows.extend(streamed),
            None => {
                let url = hugging_face_resolve_url(dataset, commit, &path)?;
                let key = format!("hf-file:{dataset}:{commit}:{path}");
                let body = config
                    .fetcher
                    .fetch(url.as_str(), &key, config.bearer_token.as_deref())
                    .await?;
                rows.extend(rows_from_remote_bytes(body, &label, remaining)?);
            }
        }
    }
    if let Some(cap) = max_rows {
        rows.truncate(cap);
    }
    Ok(rows)
}

// Load an unpinned Hugging Face dataset through the Dataset Viewer, falling
// back to revision artifacts at main when the viewer is unavailable.
async fn load_hugging_face_rows(
    config: &LoadConfig,
    dataset: &str,
    subset: &str,
    split: &str,
    max_rows: Option<usize>,
) -> Result<Vec<RawRow>> {
    match load_datasets_server_rows(config, dataset, subset, split, max_rows).await {
        Ok(rows) => Ok(rows),
        Err(rows_error) => {
            load_hugging_face_revision_rows(config, dataset, subset, split, max_rows, "main")
                .await
                .map_err(|parquet_error| {
                    DatasetError::Validation(format!(
                        "Hugging Face dataset {dataset:?} (config {subset:?}, split {split:?}) \
                         failed via the /rows API ({rows_error}) and the Parquet fallback \
                         ({parquet_error})"
                    ))
                })
        }
    }
}

async fn load_datasets_server_rows(
    config: &LoadConfig,
    dataset: &str,
    subset: &str,
    split: &str,
    max_rows: Option<usize>,
) -> Result<Vec<RawRow>> {
    validate_hugging_face_coordinates(dataset, subset, split, max_rows)?;
    let mut values = Vec::new();
    let mut offset = 0_usize;
    let mut total = usize::MAX;
    while offset < total && max_rows.is_none_or(|cap| offset < cap) {
        let length = max_rows
            .map(|cap| cap.saturating_sub(offset).min(100))
            .unwrap_or(100);
        if length == 0 {
            break;
        }
        let mut url = url::Url::parse("https://datasets-server.huggingface.co/rows")
            .expect("constant Hugging Face URL");
        url.query_pairs_mut()
            .append_pair("dataset", dataset)
            .append_pair("config", subset)
            .append_pair("split", split)
            .append_pair("offset", &offset.to_string())
            .append_pair("length", &length.to_string());
        let cache_key = format!("hf-rows:{dataset}:{subset}:{split}:{offset}:{length}");
        let bytes = config
            .fetcher
            .fetch(url.as_str(), &cache_key, config.bearer_token.as_deref())
            .await?;
        let response: Value = serde_json::from_slice(&bytes).map_err(|error| {
            DatasetError::Validation(format!("Hugging Face rows response is invalid: {error}"))
        })?;
        total = response
            .get("num_rows_total")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                DatasetError::Validation("Hugging Face rows response has no num_rows_total".into())
            })?;
        let rows = response
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                DatasetError::Validation("Hugging Face rows response has no rows array".into())
            })?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let value = row.get("row").cloned().ok_or_else(|| {
                DatasetError::Validation("Hugging Face row has no row object".into())
            })?;
            values.push(value);
        }
        offset += rows.len();
    }
    if let Some(cap) = max_rows {
        values.truncate(cap);
    }
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(index, value)| RawRow {
            wire: serde_json::to_vec(&value).ok().map(Bytes::from),
            value,
            session_id: None,
            group_key: None,
            origin: RowOrigin::JsonPointer {
                path: None,
                pointer: format!("hf://{dataset}/{subset}/{split}/{index}"),
            },
        })
        .collect())
}

/// Download an HF revision file through `hf-hub` (cached, shared with the HF
/// cache) and read only the first `max_rows` rows from the local file — jsonl by
/// line, parquet by seeked row groups — so a giant shard is never fully loaded or
/// parsed. Returns `None` for formats this streaming reader does not handle
/// (`json`, `csv`), so the caller falls back to a full fetch.
async fn load_hf_file_streaming(
    dataset: &str,
    commit: &str,
    path: &str,
    max_rows: Option<usize>,
    label: &str,
) -> Result<Option<Vec<RawRow>>> {
    let is_jsonl = supported_tabular_extension(path) == Some("jsonl");
    let is_parquet = supported_tabular_extension(path) == Some("parquet");
    if !is_jsonl && !is_parquet {
        return Ok(None);
    }
    #[cfg(not(feature = "parquet"))]
    if is_parquet {
        // Lite build has no Parquet reader; fall back to the full-fetch path,
        // which surfaces the clear "requires the parquet feature" error.
        return Ok(None);
    }
    let (dataset, commit, path, label_owned) = (
        dataset.to_owned(),
        commit.to_owned(),
        path.to_owned(),
        label.to_owned(),
    );
    let values = tokio::task::spawn_blocking(move || -> Result<Vec<Value>> {
        let local = hf_hub_download_dataset_file(&dataset, &commit, &path)?;
        if is_jsonl {
            read_jsonl_head(&local, max_rows, &label_owned)
        } else {
            read_parquet_head(&local, max_rows, &label_owned)
        }
    })
    .await
    .map_err(|error| {
        DatasetError::Validation(format!("streaming dataset file task failed: {error}"))
    })??;
    Ok(Some(rows_from_values(values, label)?))
}

/// Download one revision-pinned file from a HuggingFace *dataset* repo via
/// `hf-hub`, returning the local cache path. Reuses the standard `~/.cache/huggingface`
/// cache (shared with Python) and hf-hub's resumable/xet transfer.
fn hf_hub_download_dataset_file(
    dataset: &str,
    commit: &str,
    path: &str,
) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};
    let api = ApiBuilder::from_env()
        .with_retries(3)
        .build()
        .map_err(|error| DatasetError::Validation(format!("configuring hf-hub client: {error}")))?;
    api.repo(Repo::with_revision(
        dataset.to_string(),
        RepoType::Dataset,
        commit.to_string(),
    ))
    .get(path)
    .map_err(|error| {
        DatasetError::Validation(format!(
            "downloading hf://{dataset}@{commit}/{path}: {error}"
        ))
    })
}

/// Read at most `max_rows` JSON values from a local JSONL file, one per line,
/// without loading the whole file into memory.
fn read_jsonl_head(
    path: &std::path::Path,
    max_rows: Option<usize>,
    label: &str,
) -> Result<Vec<Value>> {
    use std::io::BufRead;
    let reader = std::io::BufReader::new(std::fs::File::open(path).map_err(DatasetError::Io)?);
    let mut values = Vec::new();
    for line in reader.lines() {
        if max_rows.is_some_and(|cap| values.len() >= cap) {
            break;
        }
        let line = line.map_err(DatasetError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(serde_json::from_str(&line).map_err(|error| {
            DatasetError::Validation(format!("{label}: invalid JSONL line: {error}"))
        })?);
    }
    Ok(values)
}

/// Read at most `max_rows` rows from a local Parquet file. `SerializedFileReader`
/// over a `File` seeks, so only the row groups the iterator touches are read.
#[cfg(feature = "parquet")]
fn read_parquet_head(
    path: &std::path::Path,
    max_rows: Option<usize>,
    label: &str,
) -> Result<Vec<Value>> {
    let reader = SerializedFileReader::new(std::fs::File::open(path).map_err(DatasetError::Io)?)
        .map_err(|error| DatasetError::Validation(format!("opening Parquet {label}: {error}")))?;
    reader
        .get_row_iter(None)
        .map_err(|error| DatasetError::Validation(format!("reading Parquet {label}: {error}")))?
        .take(max_rows.unwrap_or(usize::MAX))
        .map(|row| {
            row.map(|row| row.to_json_value()).map_err(|error| {
                DatasetError::Validation(format!("decoding a Parquet row from {label}: {error}"))
            })
        })
        .collect()
}

#[cfg(not(feature = "parquet"))]
fn read_parquet_head(
    _path: &std::path::Path,
    _max_rows: Option<usize>,
    _label: &str,
) -> Result<Vec<Value>> {
    Err(DatasetError::Validation(
        "reading Parquet requires the `parquet` feature".into(),
    ))
}

fn jsonl_or_json_rows(source: &DatasetSource) -> Result<Vec<RawRow>> {
    match source {
        DatasetSource::Path(path)
            if supported_tabular_extension(path.to_string_lossy().as_ref()) == Some("parquet") =>
        {
            rows_from_remote_bytes(
                Bytes::from(std::fs::read(path)?),
                &path.display().to_string(),
                None,
            )
        }
        DatasetSource::Path(path)
            if supported_tabular_extension(path.to_string_lossy().as_ref()) == Some("csv") =>
        {
            rows_from_remote_bytes(
                Bytes::from(std::fs::read(path)?),
                &path.display().to_string(),
                None,
            )
        }
        DatasetSource::Path(path)
            if path.extension().and_then(|extension| extension.to_str()) == Some("json") =>
        {
            rows_from_public_bytes(&std::fs::read(path)?, &path.display().to_string())
        }
        DatasetSource::Bytes(bytes) => rows_from_public_bytes(bytes, "in-memory public dataset"),
        DatasetSource::Inline(Value::Object(object))
            if object.get("data").is_some_and(Value::is_array) =>
        {
            rows_from_values(object["data"].as_array().unwrap().clone(), "inline data")
        }
        source => jsonl_rows(source),
    }
}

fn rows_from_remote_bytes(
    bytes: Bytes,
    label: &str,
    max_rows: Option<usize>,
) -> Result<Vec<RawRow>> {
    let values = match supported_tabular_extension(label) {
        Some("parquet") => decode_parquet(bytes, label, max_rows)?,
        Some("csv") => decode_csv(&bytes, label, max_rows)?,
        _ => {
            let mut rows = rows_from_public_bytes(&bytes, label)?;
            if let Some(cap) = max_rows {
                rows.truncate(cap);
            }
            return Ok(rows);
        }
    };
    rows_from_values(values, label)
}

/// Reject `.parquet` inputs when the `parquet` feature is compiled out (lite
/// build). The `arrow`/`parquet` decode stack is ~2.6 MiB of `.text`; a lite
/// nightly drops it and surfaces a clear error instead of silently linking it.
#[cfg(not(feature = "parquet"))]
fn decode_parquet(_bytes: Bytes, label: &str, _max_rows: Option<usize>) -> Result<Vec<Value>> {
    Err(DatasetError::Validation(format!(
        "Parquet dataset {label} requires an aiperf runner built with the `parquet` \
         feature; this build has it disabled"
    )))
}

#[cfg(feature = "parquet")]
fn decode_parquet(bytes: Bytes, label: &str, max_rows: Option<usize>) -> Result<Vec<Value>> {
    let reader = SerializedFileReader::new(bytes).map_err(|error| {
        DatasetError::Validation(format!("failed to open {label} as Parquet: {error}"))
    })?;
    let rows = reader.get_row_iter(None).map_err(|error| {
        DatasetError::Validation(format!("failed to read Parquet rows from {label}: {error}"))
    })?;
    rows.take(max_rows.unwrap_or(usize::MAX))
        .map(|row| {
            row.map(|row| row.to_json_value()).map_err(|error| {
                DatasetError::Validation(format!(
                    "failed to decode a Parquet row from {label}: {error}"
                ))
            })
        })
        .collect()
}

fn decode_csv(bytes: &[u8], label: &str, max_rows: Option<usize>) -> Result<Vec<Value>> {
    let mut reader = csv::Reader::from_reader(bytes);
    let headers = reader
        .headers()
        .map_err(|error| DatasetError::Validation(format!("invalid CSV {label}: {error}")))?
        .clone();
    reader
        .records()
        .take(max_rows.unwrap_or(usize::MAX))
        .enumerate()
        .map(|(index, record)| {
            let record = record.map_err(|error| {
                DatasetError::Validation(format!("invalid CSV {label} row {}: {error}", index + 2))
            })?;
            Ok(Value::Object(
                headers
                    .iter()
                    .zip(record.iter())
                    .map(|(key, value)| (key.to_string(), Value::String(value.to_string())))
                    .collect(),
            ))
        })
        .collect()
}

fn validate_hugging_face_coordinates(
    dataset: &str,
    subset: &str,
    split: &str,
    max_rows: Option<usize>,
) -> Result<()> {
    if dataset.split('/').count() != 2
        || dataset.split('/').any(str::is_empty)
        || subset.is_empty()
        || split.is_empty()
    {
        return Err(DatasetError::Validation(
            "Hugging Face dataset must be namespace/name and config/split must be non-empty".into(),
        ));
    }
    if max_rows == Some(0) {
        return Err(DatasetError::Validation(
            "Hugging Face max_rows must be positive when configured".into(),
        ));
    }
    Ok(())
}

fn supported_tabular_extension(path_or_url: &str) -> Option<&'static str> {
    let extension = match url::Url::parse(path_or_url) {
        Ok(url) => url.path().rsplit_once('.')?.1.to_ascii_lowercase(),
        Err(_) => path_or_url.rsplit_once('.')?.1.to_ascii_lowercase(),
    };
    match extension.as_str() {
        "parquet" => Some("parquet"),
        "json" => Some("json"),
        "jsonl" | "ndjson" => Some("jsonl"),
        "csv" => Some("csv"),
        _ => None,
    }
}

/// Resolve concrete file paths for one config/split from the dataset's
/// `cardData.configs[].data_files` mapping, when the dataset declares one.
///
/// Datasets frequently store data under names unrelated to the split (a single
/// root `traces.jsonl` mapped to `train`, say); the mapping is authoritative for
/// those. Only concrete tabular paths are returned — globs (e.g.
/// `data/*.parquet`) are left to the name-based fallback.
fn card_data_files(info: &Value, config: &str, split: &str) -> Vec<String> {
    let Some(configs) = info
        .get("cardData")
        .and_then(|card| card.get("configs"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for entry in configs {
        let name = entry
            .get("config_name")
            .and_then(Value::as_str)
            .unwrap_or("default");
        if name != config {
            continue;
        }
        let Some(data_files) = entry.get("data_files").and_then(Value::as_array) else {
            continue;
        };
        for data_file in data_files {
            let file_split = data_file
                .get("split")
                .and_then(Value::as_str)
                .unwrap_or("train");
            if file_split != split {
                continue;
            }
            match data_file.get("path") {
                Some(Value::String(path)) => paths.push(path.clone()),
                Some(Value::Array(globs)) => {
                    paths.extend(globs.iter().filter_map(Value::as_str).map(str::to_string));
                }
                _ => {}
            }
        }
    }
    paths.retain(|path| !path.contains('*') && supported_tabular_extension(path).is_some());
    paths
}

fn path_matches_split(path: &str, split: &str) -> bool {
    let split = split.to_ascii_lowercase();
    let path = path.to_ascii_lowercase();
    let stem = path
        .rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.').map(|(stem, _)| stem))
        .unwrap_or("");
    path.split('/').any(|component| component == split)
        || stem == split
        || stem.starts_with(&format!("{split}-"))
        || stem.contains(&format!("-{split}-"))
}

fn path_matches_subset(path: &str, subset: &str) -> bool {
    let subset = subset.to_ascii_lowercase();
    let path = path.to_ascii_lowercase();
    path.split('/').any(|component| component == subset)
        || path
            .rsplit('/')
            .next()
            .is_some_and(|name| name.starts_with(&format!("{subset}-")))
}

fn hugging_face_url(parts: &[&str]) -> Result<url::Url> {
    let mut url =
        url::Url::parse("https://huggingface.co/").expect("constant Hugging Face base URL");
    let mut segments = url.path_segments_mut().map_err(|_| {
        DatasetError::Validation("Hugging Face base URL cannot accept path segments".into())
    })?;
    segments.clear();
    for part in parts {
        if *part == parts.get(2).copied().unwrap_or("") && part.contains('/') {
            segments.extend(part.split('/'));
        } else {
            segments.push(part);
        }
    }
    drop(segments);
    Ok(url)
}

fn hugging_face_resolve_url(dataset: &str, commit: &str, path: &str) -> Result<url::Url> {
    let mut url =
        url::Url::parse("https://huggingface.co/").expect("constant Hugging Face base URL");
    let mut segments = url.path_segments_mut().map_err(|_| {
        DatasetError::Validation("Hugging Face base URL cannot accept path segments".into())
    })?;
    segments.clear();
    segments.push("datasets");
    segments.extend(dataset.split('/'));
    segments.push("resolve");
    segments.push(commit);
    segments.extend(path.split('/'));
    drop(segments);
    Ok(url)
}

fn rows_from_public_bytes(bytes: &[u8], label: &str) -> Result<Vec<RawRow>> {
    if let Ok(value) = serde_json::from_slice::<Value>(bytes) {
        let values = match value {
            Value::Array(values) => values,
            Value::Object(mut object) if object.get("data").is_some_and(Value::is_array) => {
                match object.remove("data") {
                    Some(Value::Array(values)) => values,
                    // Unreachable: the arm guard already proved `data` is an array.
                    _ => unreachable!("data guaranteed to be an array by arm guard"),
                }
            }
            value => vec![value],
        };
        return rows_from_values(values, label);
    }
    crate::dataset::loader::rows_from_bytes(bytes, None)
}

fn rows_from_values(values: Vec<Value>, label: &str) -> Result<Vec<RawRow>> {
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(index, value)| RawRow {
            wire: serde_json::to_vec(&value).ok().map(Bytes::from),
            value,
            session_id: None,
            group_key: None,
            origin: RowOrigin::JsonPointer {
                path: None,
                pointer: format!("{label}#/{index}"),
            },
        })
        .collect())
}

fn compose_prompt_lists(
    rows: Vec<RawRow>,
    field: &str,
    multi_turn: bool,
    config: &ComposeConfig,
    tokenizer: &dyn TextTokenizer,
    segments: &mut SegmentPool,
) -> Result<Vec<Conversation>> {
    let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
    let mut finalizer = config.finalizer()?;
    let mut conversations = Vec::new();

    // Tokenizing every non-empty text turn is the dominant per-row cost and
    // touches only the row's own JSON, not the shared segment pool or the
    // session-id generator - precompute the ordered (text, tokens) list for
    // every row in parallel (order within a row is preserved so the
    // `parent`-chaining intern pass below stays correct), then do the
    // sequential intern pass.
    let prepared_rows: Vec<Vec<(String, Vec<u32>)>> = rows
        .par_iter()
        .map(|row| -> Result<Vec<(String, Vec<u32>)>> {
            let Some(turns) = row.value.get(field).and_then(Value::as_array) else {
                return Ok(Vec::new());
            };
            let selected = if multi_turn {
                turns.as_slice()
            } else {
                &turns[..turns.len().min(1)]
            };
            let mut prepared = Vec::new();
            for value in selected {
                let text = value_text(value)?.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                let tokens = tokenizer.encode(&text)?;
                prepared.push((text, tokens));
            }
            Ok(prepared)
        })
        .collect::<Result<Vec<_>>>()?;

    for prepared in prepared_rows {
        if prepared.is_empty() {
            continue;
        }
        let mut conversation = Conversation::new(ids.next_id());
        let mut parent = None;
        for (text, tokens) in prepared {
            let input_tokens = Some(tokens.len() as u64);
            let handle = segments.intern_text(
                parent,
                "user",
                Bytes::from(text),
                tokens.into_boxed_slice(),
            )?;
            parent = Some(handle);
            let mut turn = Turn {
                input_tokens,
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![handle],
                    uuids: smallvec![],
                }],
                ..Turn::default()
            };
            finalizer.finalize_turn(&mut turn)?;
            conversation.turns.push(turn);
        }
        if !conversation.turns.is_empty() {
            conversations.push(conversation);
        }
    }
    Ok(conversations)
}

#[derive(Debug, Clone)]
struct AuthoredMedia {
    kind: MediaKind,
    name: String,
    contents: Vec<String>,
}

fn compose_media_turn(
    groups: Vec<AuthoredMedia>,
    config: &ComposeConfig,
    tokenizer: &dyn TextTokenizer,
    segments: &mut SegmentPool,
    finalizer: &mut crate::dataset::compose::TurnFinalizer<'_>,
    max_tokens: Option<u32>,
) -> Result<Turn> {
    let mut turn = Turn {
        max_tokens,
        ..Turn::default()
    };
    let mut parent = None;
    for group in groups {
        let mut handles = SmallVec::new();
        for content in group.contents {
            let handle = if group.kind == MediaKind::Text {
                let tokens = tokenizer.encode(&content)?;
                turn.input_tokens = Some(
                    turn.input_tokens
                        .unwrap_or(0)
                        .checked_add(tokens.len() as u64)
                        .ok_or_else(|| DatasetError::Validation("input token overflow".into()))?,
                );
                segments.intern_text(
                    parent,
                    "user",
                    Bytes::from(content),
                    tokens.into_boxed_slice(),
                )?
            } else {
                let resolved = config.media_resolver.resolve(group.kind, &content)?;
                segments.intern_media(parent, group.kind, resolved)?
            };
            parent = Some(handle);
            handles.push(handle);
        }
        if !handles.is_empty() {
            turn.content.push(ContentGroup {
                kind: group.kind,
                name: group.name,
                handles,
                uuids: SmallVec::new(),
            });
        }
    }
    finalizer.finalize_turn(&mut turn)?;
    Ok(turn)
}

fn media_from_value(value: Option<&Value>, kind: MediaKind) -> Result<Vec<AuthoredMedia>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .map_or_else(|| vec![value], |values| values.iter().collect());
    let contents = items
        .into_iter()
        .map(|value| media_content(value, kind))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok((!contents.is_empty())
        .then_some(AuthoredMedia {
            kind,
            name: String::new(),
            contents,
        })
        .into_iter()
        .collect())
}

fn media_content(value: &Value, kind: MediaKind) -> Result<Option<String>> {
    match value {
        Value::String(value) if !value.is_empty() => Ok(Some(value.clone())),
        Value::Object(object) => {
            if kind == MediaKind::Audio
                && object.get("array").is_some()
                && object.get("sampling_rate").is_some()
            {
                return encode_hf_audio_array(object).map(Some);
            }
            if let Some(bytes) = object.get("bytes")
                && !bytes.is_null()
            {
                let raw = decode_hf_bytes(bytes)?;
                return encode_hf_media_bytes(
                    kind,
                    &raw,
                    object.get("path").and_then(Value::as_str),
                )
                .map(Some);
            }
            Ok(object
                .get("src")
                .or_else(|| object.get("url"))
                .or_else(|| object.get("path"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string))
        }
        _ => Ok(None),
    }
}

fn decode_hf_bytes(value: &Value) -> Result<Vec<u8>> {
    match value {
        Value::String(encoded) => {
            let encoded = encoded
                .split_once(',')
                .filter(|(prefix, _)| prefix.contains("base64"))
                .map_or(encoded.as_str(), |(_, encoded)| encoded);
            STANDARD.decode(encoded).map_err(|error| {
                DatasetError::Validation(format!("invalid Hugging Face media bytes: {error}"))
            })
        }
        Value::Array(bytes) => bytes
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| {
                        DatasetError::Validation(
                            "Hugging Face media byte arrays must contain u8 values".into(),
                        )
                    })
            })
            .collect(),
        _ => Err(DatasetError::Validation(
            "Hugging Face media bytes must be base64 text or a u8 array".into(),
        )),
    }
}

fn encode_hf_media_bytes(kind: MediaKind, raw: &[u8], path: Option<&str>) -> Result<String> {
    match kind {
        MediaKind::Text => std::str::from_utf8(raw)
            .map(str::to_string)
            .map_err(|error| {
                DatasetError::Validation(format!("invalid UTF-8 text bytes: {error}"))
            }),
        MediaKind::Image => {
            let image = image::load_from_memory(raw).map_err(|error| {
                DatasetError::Validation(format!("invalid Hugging Face image bytes: {error}"))
            })?;
            let rgb = image.to_rgb8();
            let mut jpeg = Vec::new();
            JpegEncoder::new_with_quality(&mut jpeg, 85)
                .encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    ColorType::Rgb8.into(),
                )
                .map_err(|error| {
                    DatasetError::Validation(format!(
                        "failed to encode Hugging Face image as JPEG: {error}"
                    ))
                })?;
            Ok(format!("data:image/jpeg;base64,{}", STANDARD.encode(jpeg)))
        }
        MediaKind::Audio => {
            let (wav, _) = crate::dataset::generator::transcode_audio_to_wav(raw)?;
            Ok(format!("wav,{}", STANDARD.encode(wav)))
        }
        MediaKind::Video => {
            let mime = path
                .and_then(|path| path.rsplit_once('.').map(|(_, extension)| extension))
                .filter(|extension| extension.eq_ignore_ascii_case("webm"))
                .map_or("video/mp4", |_| "video/webm");
            Ok(format!("data:{mime};base64,{}", STANDARD.encode(raw)))
        }
    }
}

fn encode_hf_audio_array(object: &Map<String, Value>) -> Result<String> {
    let sample_rate = object
        .get("sampling_rate")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            DatasetError::Validation(
                "Hugging Face audio sampling_rate must be a positive u32".into(),
            )
        })?;
    let values = object
        .get("array")
        .and_then(Value::as_array)
        .ok_or_else(|| DatasetError::Validation("Hugging Face audio array is invalid".into()))?;
    let (channels, samples) = if values.iter().all(Value::is_number) {
        (1_u16, values.iter().collect::<Vec<_>>())
    } else {
        let frames = values
            .iter()
            .map(|frame| frame.as_array())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                DatasetError::Validation(
                    "Hugging Face audio array must contain numbers or numeric frames".into(),
                )
            })?;
        let channels = frames.first().map_or(0, |frame| frame.len());
        if channels == 0 || frames.iter().any(|frame| frame.len() != channels) {
            return Err(DatasetError::Validation(
                "Hugging Face audio frames must have a consistent non-zero channel count".into(),
            ));
        }
        (
            u16::try_from(channels).map_err(|_| {
                DatasetError::Validation("Hugging Face audio channel count exceeds u16".into())
            })?,
            frames.into_iter().flatten().collect(),
        )
    };
    let data_len = samples
        .len()
        .checked_mul(2)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| DatasetError::Validation("Hugging Face WAV exceeds 4 GiB".into()))?;
    let block_align = channels
        .checked_mul(2)
        .ok_or_else(|| DatasetError::Validation("Hugging Face WAV alignment overflow".into()))?;
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| DatasetError::Validation("Hugging Face WAV byte-rate overflow".into()))?;
    let riff_len = 36_u32
        .checked_add(data_len)
        .ok_or_else(|| DatasetError::Validation("Hugging Face WAV size overflow".into()))?;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_len.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let sample = sample
            .as_f64()
            .filter(|sample| sample.is_finite())
            .ok_or_else(|| {
                DatasetError::Validation("Hugging Face audio samples must be finite numbers".into())
            })?;
        let pcm = (sample.clamp(-1.0, 1.0) * f64::from(i16::MAX)).round() as i16;
        wav.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(format!("wav,{}", STANDARD.encode(wav)))
}

fn sharegpt_pairs(messages: &[Value]) -> Vec<(String, String)> {
    let messages = messages
        .iter()
        .filter_map(Value::as_object)
        .collect::<Vec<_>>();
    if messages.len() < 2 {
        return Vec::new();
    }
    let uses_roles = messages.iter().any(|message| {
        matches!(
            message.get("from").and_then(Value::as_str),
            Some("human" | "gpt")
        )
    });
    if !uses_roles {
        return match (
            messages[0].get("value").and_then(Value::as_str),
            messages[1].get("value").and_then(Value::as_str),
        ) {
            (Some(prompt), Some(completion)) if !prompt.is_empty() && !completion.is_empty() => {
                vec![(prompt.to_string(), completion.to_string())]
            }
            _ => Vec::new(),
        };
    }
    adjacent_pairs(&messages, "from", "human", "gpt", "value")
}

fn normalize_hf_messages(messages: &[Value]) -> Vec<Map<String, Value>> {
    messages
        .iter()
        .filter_map(|value| {
            let value = value
                .as_array()
                .and_then(|values| values.first())
                .unwrap_or(value);
            value
                .as_object()
                .filter(|object| !object.is_empty())
                .cloned()
        })
        .collect()
}

fn first_user_message(messages: &[Map<String, Value>], content_key: &str) -> Option<String> {
    messages
        .iter()
        .find(|message| {
            matches!(
                message
                    .get("role")
                    .or_else(|| message.get("from"))
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("user" | "human")
            )
        })
        .or_else(|| messages.first())
        .and_then(|message| message_text(message, content_key))
}

fn hf_message_pairs(messages: &[Map<String, Value>], content_key: &str) -> Vec<(String, String)> {
    let references = messages.iter().collect::<Vec<_>>();
    if messages.iter().any(|message| {
        matches!(
            message.get("from").and_then(Value::as_str),
            Some("human" | "gpt")
        )
    }) {
        adjacent_pairs(&references, "from", "human", "gpt", content_key)
    } else {
        adjacent_pairs(&references, "role", "user", "assistant", content_key)
    }
}

fn adjacent_pairs(
    messages: &[&Map<String, Value>],
    role_key: &str,
    prompt_role: &str,
    completion_role: &str,
    content_key: &str,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut index = 0;
    while index + 1 < messages.len() {
        let first = messages[index];
        let second = messages[index + 1];
        if first.get(role_key).and_then(Value::as_str) == Some(prompt_role)
            && second.get(role_key).and_then(Value::as_str) == Some(completion_role)
            && let (Some(prompt), Some(completion)) = (
                message_text(first, content_key),
                message_text(second, content_key),
            )
        {
            pairs.push((prompt, completion));
            index += 2;
        } else {
            index += 1;
        }
    }
    pairs
}

fn message_text(message: &Map<String, Value>, content_key: &str) -> Option<String> {
    message
        .get(content_key)
        .and_then(Value::as_str)
        .map(|text| text.replace("<image>", "").trim().to_string())
        .filter(|text| !text.is_empty())
}

fn format_template(
    template: &str,
    values: &Map<String, Value>,
    origin: &impl std::fmt::Display,
) -> Result<String> {
    let mut output = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        output.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after.find('}').ok_or_else(|| {
            DatasetError::Validation(format!("{origin}: unmatched '{{' in prompt_template"))
        })?;
        let key = &after[..close];
        if key.is_empty() || key.contains(['{', '}']) {
            return Err(DatasetError::Validation(format!(
                "{origin}: invalid prompt_template field {key:?}"
            )));
        }
        let value = values.get(key).ok_or_else(|| {
            DatasetError::Validation(format!(
                "{origin}: prompt_template references missing column {key:?}"
            ))
        })?;
        output.push_str(&value_text(value)?);
        rest = &after[close + 1..];
    }
    if rest.contains('}') {
        return Err(DatasetError::Validation(format!(
            "{origin}: unmatched '}}' in prompt_template"
        )));
    }
    output.push_str(rest);
    Ok(output)
}

fn value_text(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Null => Ok(String::new()),
        value => serde_json::to_string(value).map_err(DatasetError::from),
    }
}

fn extracted_token_count(value: &Value, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    let extracted = extract_payload(value);
    extracted
        .texts
        .into_iter()
        .try_fold(extracted.pretokenised_token_count, |count, text| {
            count
                .checked_add(tokenizer.encode(&text)?.len() as u64)
                .ok_or_else(|| DatasetError::Validation("input token overflow".into()))
        })
}

fn validate_message_values(messages: &[Value], origin: &impl std::fmt::Display) -> Result<()> {
    if messages.is_empty()
        || messages.iter().any(|message| {
            !message.is_object() || !message.get("role").is_some_and(Value::is_string)
        })
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: messages must be non-empty objects with roles"
        )));
    }
    Ok(())
}

fn is_speed_bench(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object
        .get("question_id")
        .and_then(Value::as_str)
        .is_some_and(|id| id.len() == 32)
        && object
            .get("category")
            .and_then(Value::as_str)
            .is_some_and(|category| !category.is_empty())
        && object
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                !messages.is_empty()
                    && messages.iter().all(|message| {
                        message
                            .get("role")
                            .and_then(Value::as_str)
                            .is_some_and(|role| !role.trim().is_empty())
                            && message
                                .get("content")
                                .and_then(Value::as_str)
                                .is_some_and(|content| {
                                    !content.trim().is_empty()
                                        && content
                                            != "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
                                })
                    })
            })
}

fn require_object<'a>(
    value: &'a Value,
    origin: &impl std::fmt::Display,
) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| DatasetError::Validation(format!("{origin}: dataset row must be an object")))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    origin: &impl std::fmt::Display,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            DatasetError::Validation(format!("{origin}: {field} must be a non-empty string"))
        })
}

fn string_option(config: &ComposeConfig, key: &str) -> Option<String> {
    config
        .format_options
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn bool_option(config: &ComposeConfig, key: &str, default: bool) -> Result<bool> {
    config
        .format_options
        .get(key)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                DatasetError::Validation(format!("format option {key} must be boolean"))
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn usize_option(config: &ComposeConfig, key: &str, default: usize) -> Result<usize> {
    config
        .format_options
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    DatasetError::Validation(format!("format option {key} must be usize"))
                })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::rng::RngRoot;
    #[cfg(feature = "parquet")]
    use parquet::data_type::{ByteArray, ByteArrayType};
    #[cfg(feature = "parquet")]
    use parquet::file::writer::SerializedFileWriter;
    #[cfg(feature = "parquet")]
    use parquet::schema::parser::parse_message_type;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::dataset::tokenizer::TiktokenTokenizer;

    struct MockRevisionFetcher {
        info: Bytes,
        file: Bytes,
        urls: Mutex<Vec<String>>,
    }

    struct StaticFetcher(Bytes);

    #[async_trait]
    impl crate::dataset::fetch::DatasetFetcher for StaticFetcher {
        async fn fetch(
            &self,
            _url: &str,
            _cache_key: &str,
            _bearer_token: Option<&str>,
        ) -> Result<Bytes> {
            Ok(self.0.clone())
        }
    }

    #[async_trait]
    impl crate::dataset::fetch::DatasetFetcher for MockRevisionFetcher {
        async fn fetch(
            &self,
            url: &str,
            _cache_key: &str,
            _bearer_token: Option<&str>,
        ) -> Result<Bytes> {
            self.urls.lock().unwrap().push(url.to_string());
            if url.contains("/revision/") {
                Ok(self.info.clone())
            } else if url.contains("/resolve/") {
                Ok(self.file.clone())
            } else {
                Err(DatasetError::Validation(format!(
                    "unexpected mock URL {url}"
                )))
            }
        }
    }

    async fn build(
        loader: Arc<dyn DatasetLoader>,
        composer: Arc<dyn Composer>,
        source: Value,
        options: Map<String, Value>,
    ) -> Result<crate::dataset::Dataset> {
        let mut registry = LoaderRegistry::new();
        let name = loader.name().to_string();
        registry.register(DatasetFormatRegistration::new(loader, composer))?;
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(5)));
        compose.format_options = options;
        registry
            .build_dataset(
                Some(&name),
                &LoadConfig::new(DatasetSource::Inline(source)),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
    }

    #[tokio::test]
    async fn accuracy_carries_real_correlation_and_both_chat_completion_views() {
        let mut options = Map::new();
        options.insert(
            "accuracy_system_prompt".into(),
            Value::String("brief".into()),
        );
        let dataset = build(
            Arc::new(AccuracyDatasetLoader),
            Arc::new(AccuracyComposer),
            json!([{"prompt":"Q?","task":"math","correlation_id":"problem-1","metadata":{"generation_size":5},"extra_body":{"temperature":0.2,"stop":["Q:"]}}]),
            options,
        ).await.unwrap();
        let conversation = &dataset.conversations()[0];
        assert_eq!(
            conversation
                .accuracy
                .as_ref()
                .unwrap()
                .correlation_id
                .as_str(),
            "problem-1"
        );
        assert!(conversation.turns[0].raw_messages.is_some());
        assert_eq!(conversation.turns[0].max_tokens, Some(5));
        assert!(!conversation.turns[0].content.is_empty());
        assert!(conversation.turns[0].extra_body.is_some());
    }

    #[tokio::test]
    async fn accuracy_rejects_non_object_generation_extras() {
        let error = build(
            Arc::new(AccuracyDatasetLoader),
            Arc::new(AccuracyComposer),
            json!([{"prompt":"Q?","task":"math","extra_body":[]}]),
            Map::new(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("extra_body must be an object"));
    }

    #[tokio::test]
    async fn sharegpt_uses_every_adjacent_human_gpt_pair() {
        let dataset = build(
            Arc::new(ShareGptDatasetLoader),
            Arc::new(ShareGptComposer),
            json!([{"conversations":[
                {"from":"human","value":"one two three four"},
                {"from":"gpt","value":"a b c d"},
                {"from":"human","value":"five six seven eight"},
                {"from":"gpt","value":"e f g h"}
            ]}]),
            Map::new(),
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations()[0].turns.len(), 2);
    }

    fn hf_auto_options(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn pick_hf_coordinates_picks_default_config_and_train_split() {
        let info = json!({"default": {"splits": {"train": {}, "test": {}}}});
        let (c, s) = pick_hf_coordinates(info.as_object().unwrap(), "", "").unwrap();
        assert_eq!((c.as_str(), s.as_str()), ("default", "train"));
    }

    #[test]
    fn pick_hf_coordinates_falls_back_to_test_when_no_train() {
        let info = json!({"main": {"splits": {"validation": {}, "test": {}}}});
        let (c, s) = pick_hf_coordinates(info.as_object().unwrap(), "", "").unwrap();
        assert_eq!((c.as_str(), s.as_str()), ("main", "test"));
    }

    #[test]
    fn pick_hf_coordinates_honors_explicit_subset_and_split() {
        let info = json!({"a": {"splits": {"train": {}}}, "b": {"splits": {"x": {}}}});
        let (c, s) = pick_hf_coordinates(info.as_object().unwrap(), "b", "x").unwrap();
        assert_eq!((c.as_str(), s.as_str()), ("b", "x"));
    }

    #[test]
    fn pick_hf_coordinates_unknown_explicit_subset_errors() {
        let info = json!({"a": {"splits": {"train": {}}}});
        assert!(pick_hf_coordinates(info.as_object().unwrap(), "missing", "").is_err());
    }

    #[tokio::test]
    async fn hf_auto_composer_text_format_single_turn() {
        let dataset = build(
            Arc::new(HfAutoDatasetLoader),
            Arc::new(HfAutoComposer),
            json!([
                {"prompt": "Explain how photosynthesis converts sunlight into chemical energy in plants.",
                 "completion": "Plants use chlorophyll to capture light."},
                {"prompt": "Describe the structure of DNA and how genetic information is encoded within it.",
                 "completion": "DNA is a double helix of base pairs."}
            ]),
            hf_auto_options(json!({})),
        )
        .await
        .unwrap();
        let convos = dataset.conversations();
        assert_eq!(convos.len(), 2);
        assert!(convos.iter().all(|c| c.turns.len() == 1));
        assert!(convos.iter().all(|c| c.turns[0].max_tokens.unwrap() > 0));
    }

    #[tokio::test]
    async fn hf_auto_composer_output_len_override() {
        let dataset = build(
            Arc::new(HfAutoDatasetLoader),
            Arc::new(HfAutoComposer),
            json!([{
                "question": "What are the key differences between supervised and unsupervised learning here?",
                "answer": "Supervised uses labels."
            }]),
            hf_auto_options(json!({"output_len": 77})),
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations()[0].turns[0].max_tokens, Some(77));
    }

    #[tokio::test]
    async fn hf_auto_composer_chat_format_uses_user_and_assistant() {
        let dataset = build(
            Arc::new(HfAutoDatasetLoader),
            Arc::new(HfAutoComposer),
            json!([{
                "conversation": [
                    {"role": "user", "content": "What is the meaning of life, the universe, and everything today?"},
                    {"role": "assistant", "content": "42, per Douglas Adams and his famous novel."}
                ]
            }]),
            hf_auto_options(json!({})),
        )
        .await
        .unwrap();
        let convos = dataset.conversations();
        assert_eq!(convos.len(), 1);
        assert_eq!(convos[0].turns.len(), 1);
        assert!(convos[0].turns[0].max_tokens.unwrap() > 0);
    }

    #[tokio::test]
    async fn hf_instruction_template_and_media_are_composed() {
        let mut options = Map::new();
        options.insert("prompt_column".into(), Value::String("question".into()));
        options.insert(
            "prompt_template".into(),
            Value::String("Code: {code}; Q: {question}".into()),
        );
        options.insert("image_column".into(), Value::String("image".into()));
        let dataset = build(
            Arc::new(HfInstructionDatasetLoader),
            Arc::new(HfInstructionComposer),
            json!([{"code":"x=1","question":"why?","image":{"src":"https://example.com/i.png"}}]),
            options,
        )
        .await
        .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert!(
            turn.content
                .iter()
                .any(|group| group.kind == MediaKind::Text)
        );
        assert!(
            turn.content
                .iter()
                .any(|group| group.kind == MediaKind::Image)
        );
    }

    #[tokio::test]
    async fn hf_and_mt_bench_report_missing_required_columns() {
        let mut hf_options = Map::new();
        hf_options.insert("prompt_column".into(), Value::String("prompt".into()));
        let hf_error = build(
            Arc::new(HfInstructionDatasetLoader),
            Arc::new(HfInstructionComposer),
            json!([{"wrong":"value"}]),
            hf_options,
        )
        .await
        .unwrap_err();
        assert!(hf_error.to_string().contains("prompt column"));

        let mt_error = build(
            Arc::new(MtBenchDatasetLoader),
            Arc::new(MtBenchComposer),
            json!([{"wrong":[]}]),
            Map::new(),
        )
        .await
        .unwrap_err();
        assert!(mt_error.to_string().contains("MT-Bench prompt column"));
    }

    #[tokio::test]
    async fn hf_decoded_image_audio_and_video_values_are_inlined_at_compose_time() {
        let image = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
            2,
            1,
            image::Rgb([10, 20, 30]),
        ));
        let mut png = Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let mut options = Map::new();
        options.insert("prompt_column".into(), Value::String("prompt".into()));
        options.insert("image_column".into(), Value::String("image".into()));
        options.insert("audio_column".into(), Value::String("audio".into()));
        options.insert("video_column".into(), Value::String("video".into()));
        let dataset = build(
            Arc::new(HfInstructionDatasetLoader),
            Arc::new(HfInstructionComposer),
            json!([{
                "prompt":"describe the media",
                "image":{"bytes":STANDARD.encode(png.into_inner()),"path":"pixel.png"},
                "audio":{"array":[0.0,0.5,-0.5],"sampling_rate":8000},
                "video":{"bytes":STANDARD.encode([1_u8,2,3,4]),"path":"clip.webm"}
            }]),
            options,
        )
        .await
        .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.content.len(), 4);
        let image = content_bytes(&dataset, turn, MediaKind::Image);
        assert!(image.starts_with(b"data:image/jpeg;base64,"));
        let audio = content_bytes(&dataset, turn, MediaKind::Audio);
        let wav = STANDARD
            .decode(
                std::str::from_utf8(audio)
                    .unwrap()
                    .split_once(',')
                    .unwrap()
                    .1,
            )
            .unwrap();
        assert!(wav.starts_with(b"RIFF"));
        let video = content_bytes(&dataset, turn, MediaKind::Video);
        assert!(video.starts_with(b"data:video/webm;base64,"));
    }

    #[tokio::test]
    async fn hf_conversation_multiturn_validates_pairs_and_keeps_completion_lengths() {
        let mut options = Map::new();
        options.insert(
            "conversation_column".into(),
            Value::String("messages".into()),
        );
        options.insert("multi_turn".into(), Value::Bool(true));
        options.insert("min_sequence_tokens".into(), Value::from(1));
        let tokenizer = TiktokenTokenizer::builtin();
        let first_completion = "first authored answer";
        let second_completion = "second authored answer with detail";
        let dataset = build(
            Arc::new(HfConversationDatasetLoader),
            Arc::new(HfConversationComposer),
            json!([{"messages":[
                {"role":"user","content":"first question"},
                {"role":"assistant","content":first_completion},
                {"role":"user","content":"second question"},
                {"role":"assistant","content":second_completion}
            ]}]),
            options,
        )
        .await
        .unwrap();
        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[0].max_tokens,
            Some(tokenizer.count(first_completion).unwrap() as u32)
        );
        assert_eq!(
            turns[1].max_tokens,
            Some(tokenizer.count(second_completion).unwrap() as u32)
        );
    }

    #[tokio::test]
    async fn mt_bench_keeps_every_prompt_as_a_prefix_chained_turn() {
        let dataset = build(
            Arc::new(MtBenchDatasetLoader),
            Arc::new(MtBenchComposer),
            json!([
                {"prompt":["first user turn", "second user turn"]},
                {"prompt":[]}
            ]),
            Map::new(),
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns.len(), 2);
        let first = turns[0].content[0].handles[0];
        let second = turns[1].content[0].handles[0];
        assert_eq!(
            dataset.segments().segment(second).unwrap().parent,
            Some(first)
        );
    }

    #[tokio::test]
    async fn spec_bench_defaults_to_first_turn_and_supports_full_conversations() {
        let source = json!([{"turns":["first", "second", "third"]}]);
        let single = build(
            Arc::new(SpecBenchDatasetLoader),
            Arc::new(SpecBenchComposer),
            source.clone(),
            Map::new(),
        )
        .await
        .unwrap();
        assert_eq!(single.conversations()[0].turns.len(), 1);

        let mut options = Map::new();
        options.insert("multi_turn".into(), Value::Bool(true));
        let multi = build(
            Arc::new(SpecBenchDatasetLoader),
            Arc::new(SpecBenchComposer),
            source,
            options,
        )
        .await
        .unwrap();
        assert_eq!(multi.conversations()[0].turns.len(), 3);
    }

    #[tokio::test]
    async fn mmvu_formats_choices_and_requires_video_content() {
        let dataset = build(
            Arc::new(MmvuDatasetLoader),
            Arc::new(MmvuComposer),
            json!([
                {
                    "question":"Which option?",
                    "choices":{"A":"alpha", "B":"beta"},
                    "video":{"url":"https://example.com/video.mp4"}
                },
                {"question":"no media", "choices":{"A":"skip"}}
            ]),
            Map::new(),
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.content.len(), 2);
        let text = dataset.segments().get(turn.content[0].handles[0]).unwrap();
        assert!(
            matches!(text, crate::dataset::Payload::Text { bytes, .. } if bytes == "Which option? A.alpha B.beta")
        );
        let video = dataset.segments().get(turn.content[1].handles[0]).unwrap();
        assert!(
            matches!(video, crate::dataset::Payload::Media { kind: MediaKind::Video, bytes } if bytes == "https://example.com/video.mp4")
        );
    }

    #[tokio::test]
    async fn speed_bench_preserves_session_roles_and_category_filtering() {
        let id = "0123456789abcdef0123456789abcdef";
        let mut options = Map::new();
        options.insert("category".into(), Value::String("coding".into()));
        let dataset = build(
            Arc::new(SpeedBenchDatasetLoader),
            Arc::new(SpeedBenchComposer),
            json!([
                {
                    "question_id":id,
                    "category":"coding",
                    "messages":[
                        {"role":"user", "content":"write code"},
                        {"role":"assistant", "content":"authored response"}
                    ]
                },
                {
                    "question_id":"fedcba9876543210fedcba9876543210",
                    "category":"math",
                    "messages":[{"role":"user", "content":"skip me"}]
                }
            ]),
            options,
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
        assert_eq!(dataset.conversations()[0].session_id.as_str(), id);
        let turns = &dataset.conversations()[0].turns;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role.as_ref().unwrap().as_str(), "user");
        assert_eq!(turns[1].role.as_ref().unwrap().as_str(), "assistant");
    }

    #[tokio::test]
    async fn pinned_hugging_face_source_resolves_commit_and_never_uses_rows_api() {
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let fetcher = Arc::new(MockRevisionFetcher {
            info: Bytes::from(
                serde_json::to_vec(&json!({
                    "sha": commit,
                    "siblings": [{"rfilename":"data/train-00000.jsonl"}]
                }))
                .unwrap(),
            ),
            file: Bytes::from_static(b"{\"prompt\":\"first\"}\n{\"prompt\":\"second\"}\n"),
            urls: Mutex::new(Vec::new()),
        });
        let mut load = LoadConfig::new(DatasetSource::HuggingFace {
            dataset: "owner/repository".into(),
            config: "default".into(),
            split: "train".into(),
            max_rows: Some(1),
            revision: Some("reviewed".into()),
        });
        load.fetcher = fetcher.clone();
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(2)));
        compose
            .format_options
            .insert("prompt_column".into(), Value::String("prompt".into()));
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("hf_instruction_response"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
        let urls = fetcher.urls.lock().unwrap();
        assert_eq!(urls.len(), 2);
        assert!(urls[0].contains("/revision/reviewed"));
        assert!(urls[1].contains(&format!("/resolve/{commit}/data/train-00000.jsonl")));
        assert!(urls.iter().all(|url| !url.contains("datasets-server")));
    }

    #[tokio::test]
    async fn hugging_face_rows_never_exceed_the_authored_cap() {
        let fetcher = Arc::new(StaticFetcher(Bytes::from(
            serde_json::to_vec(&json!({
                "num_rows_total":2,
                "rows":[
                    {"row":{"prompt":"first"}},
                    {"row":{"prompt":"second"}}
                ]
            }))
            .unwrap(),
        )));
        let mut load = LoadConfig::new(DatasetSource::HuggingFace {
            dataset: "owner/repository".into(),
            config: "default".into(),
            split: "train".into(),
            max_rows: Some(1),
            revision: None,
        });
        load.fetcher = fetcher;
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(2)));
        compose
            .format_options
            .insert("prompt_column".into(), Value::String("prompt".into()));
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("hf_instruction_response"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_decoder_preserves_nested_capable_json_scalars() {
        let schema = Arc::new(
            parse_message_type(
                "message schema { REQUIRED BYTE_ARRAY prompt (UTF8); REQUIRED INT64 tokens; }",
            )
            .unwrap(),
        );
        let mut bytes = Vec::new();
        let mut writer = SerializedFileWriter::new(&mut bytes, schema, Default::default()).unwrap();
        let mut row_group = writer.next_row_group().unwrap();
        let mut prompt = row_group.next_column().unwrap().unwrap();
        prompt
            .typed::<ByteArrayType>()
            .write_batch(
                &[
                    ByteArray::from("hello".as_bytes().to_vec()),
                    ByteArray::from("world".as_bytes().to_vec()),
                ],
                None,
                None,
            )
            .unwrap();
        prompt.close().unwrap();
        let mut tokens = row_group.next_column().unwrap().unwrap();
        tokens
            .typed::<parquet::data_type::Int64Type>()
            .write_batch(&[3, 4], None, None)
            .unwrap();
        tokens.close().unwrap();
        row_group.close().unwrap();
        writer.close().unwrap();
        let values = decode_parquet(Bytes::from(bytes), "fixture.parquet", Some(1)).unwrap();
        assert_eq!(values, vec![json!({"prompt":"hello","tokens":3})]);
    }

    fn content_bytes<'a>(
        dataset: &'a crate::dataset::Dataset,
        turn: &Turn,
        kind: MediaKind,
    ) -> &'a [u8] {
        let handle = turn
            .content
            .iter()
            .find(|group| group.kind == kind)
            .unwrap()
            .handles[0];
        match dataset.segments().get(handle).unwrap() {
            crate::dataset::Payload::Media { bytes, .. } => bytes,
            payload => panic!("expected media, got {}", payload.kind_name()),
        }
    }
}
