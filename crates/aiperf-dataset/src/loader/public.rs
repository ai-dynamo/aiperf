// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public/Hugging Face, ShareGPT, and accuracy dataset formats.
//!
//! Converters are grounded in `src/aiperf/dataset/loader/sharegpt.py:24-217`,
//! `hf_instruction_response.py:14-128`, `hf_conversation.py:14-282`,
//! `mt_bench.py:13-89`, `mmvu.py:13-98`, `spec_bench.py:15-97`, and
//! `accuracy_dataset_loader.py:21-150`. Remote Hugging Face rows use the
//! documented Dataset Viewer `/rows` API in pages of at most 100. Revision-pinned
//! sources resolve the Hub revision to an immutable commit and decode the
//! repository's Parquet/JSON/JSONL/CSV artifacts directly.

use aiperf_endpoints::extract_payload;
use async_trait::async_trait;
use bytes::Bytes;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::{Map, Value};
use smallvec::{SmallVec, smallvec};

use crate::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::error::{DatasetError, Result};
use crate::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::model::{
    AccuracyGroundTruth, ContentGroup, Conversation, CorrelationId, MediaKind, SessionId, Turn,
};
use crate::segment::{Role, SegmentPool};
use crate::tokenizer::TextTokenizer;

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
    value.get("ground_truth").is_some() && value.get("task").is_some()
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
        for row in rows {
            let object = require_object(&row.value, &row.origin)?;
            let prompt = required_string(object, "prompt", &row.origin)?;
            let ground_truth = required_string(object, "ground_truth", &row.origin)?;
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
            let mut messages = object
                .get("raw_messages")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([{"role":"user","content":prompt}]));
            let messages_array = messages.as_array_mut().ok_or_else(|| {
                DatasetError::Validation(format!("{}: raw_messages must be an array", row.origin))
            })?;
            if let Some(system) = &system_prompt {
                messages_array.insert(0, serde_json::json!({"role":"system","content":system}));
            }
            validate_message_values(messages_array, &row.origin)?;
            let message_wire = Bytes::from(serde_json::to_vec(&messages)?);
            let raw_messages = segments.intern_raw(None, message_wire)?;
            let prompt_text = system_prompt.as_ref().map_or_else(
                || prompt.to_string(),
                |system| format!("{system}\n\n{prompt}"),
            );
            let prompt_tokens = tokenizer.encode(&prompt_text)?;
            let text = segments.intern_text(
                Some(raw_messages),
                "user",
                Bytes::from(prompt_text),
                prompt_tokens.clone().into_boxed_slice(),
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
                }],
                ..Turn::default()
            };
            finalizer.finalize_turn(&mut turn)?;
            let mut conversation = Conversation::new(session_id);
            conversation.turns.push(turn);
            conversation.accuracy = Some(AccuracyGroundTruth {
                correlation_id,
                ground_truth: ground_truth.to_string(),
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
        for row in rows {
            let Some(messages) = row.value.get("conversations").and_then(Value::as_array) else {
                continue;
            };
            let pairs = sharegpt_pairs(messages);
            if pairs.is_empty() {
                continue;
            }
            let mut prepared = Vec::with_capacity(pairs.len());
            let mut valid = true;
            for (prompt, completion) in pairs {
                let prompt_tokens = tokenizer.encode(&prompt)?;
                let completion_tokens = tokenizer.encode(&completion)?;
                if prompt_tokens.len() < min_length
                    || prompt_tokens.len() > max_prompt
                    || (!skip_min_output && completion_tokens.len() < min_length)
                    || prompt_tokens.len() + completion_tokens.len() > max_total
                {
                    valid = false;
                    break;
                }
                prepared.push((prompt, prompt_tokens, completion_tokens.len() as u32));
            }
            if !valid || prepared.is_empty() {
                continue;
            }
            let mut conversation = Conversation::new(ids.next_id());
            let mut parent = None;
            for (prompt, tokens, output_tokens) in prepared {
                let handle = segments.intern_text(
                    parent,
                    "user",
                    Bytes::from(prompt),
                    tokens.clone().into_boxed_slice(),
                )?;
                parent = Some(handle);
                let mut turn = Turn {
                    max_tokens: Some(output_tokens.max(1)),
                    input_tokens: tokens.len() as u64,
                    content: smallvec![ContentGroup {
                        kind: MediaKind::Text,
                        name: String::new(),
                        handles: smallvec![handle],
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
                groups.extend(media_from_value(object.get(column), MediaKind::Image));
            }
            if let Some(column) = &video_column {
                groups.extend(media_from_value(object.get(column), MediaKind::Video));
            }
            if let Some(column) = &audio_column {
                groups.extend(media_from_value(object.get(column), MediaKind::Audio));
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
        for row in rows {
            let object = require_object(&row.value, &row.origin)?;
            let Some(messages) = object.get(&column).and_then(Value::as_array) else {
                continue;
            };
            let normalized = normalize_hf_messages(messages);
            let prompts = if multi_turn {
                let mut prepared = Vec::new();
                let mut valid = true;
                for (prompt, completion) in hf_message_pairs(&normalized, &content_key) {
                    let prompt_tokens = tokenizer.encode(&prompt)?.len();
                    let completion_tokens = tokenizer.encode(&completion)?.len();
                    if prompt_tokens < min_length
                        || prompt_tokens > max_prompt
                        || (!skip_min_output && completion_tokens < min_length)
                        || prompt_tokens + completion_tokens > max_total
                    {
                        valid = false;
                        break;
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
                if valid { prepared } else { Vec::new() }
            } else {
                first_user_message(&normalized, &content_key)
                    .into_iter()
                    .map(|prompt| (prompt, None))
                    .collect()
            };
            if prompts.is_empty() {
                continue;
            }
            let first_images = image_column
                .as_ref()
                .map(|column| media_from_value(object.get(column), MediaKind::Image))
                .unwrap_or_default();
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
            let videos = media_from_value(object.get(&video_column), MediaKind::Video);
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
        for row in rows {
            if !is_speed_bench(&row.value) {
                return Err(DatasetError::Validation(format!(
                    "{}: invalid SPEED-Bench row",
                    row.origin
                )));
            }
            if category.as_ref().is_some_and(|category| {
                row.value.get("category").and_then(Value::as_str) != Some(category)
            }) {
                continue;
            }
            let id = row.value["question_id"].as_str().unwrap();
            let messages = row.value["messages"].as_array().unwrap();
            let selected = if multi_turn {
                messages.as_slice()
            } else {
                &messages[..1]
            };
            let mut conversation = Conversation::new(id);
            let mut parent = None;
            for message in selected {
                let role = message["role"].as_str().unwrap();
                let content = message["content"].as_str().unwrap();
                let tokens = tokenizer.encode(content)?;
                let handle = segments.intern_text(
                    parent,
                    role,
                    Bytes::copy_from_slice(content.as_bytes()),
                    tokens.clone().into_boxed_slice(),
                )?;
                parent = Some(handle);
                let mut turn = Turn {
                    role: Some(Role::from(role)),
                    input_tokens: tokens.len() as u64,
                    content: smallvec![ContentGroup {
                        kind: MediaKind::Text,
                        name: String::new(),
                        handles: smallvec![handle],
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
            None => load_hugging_face_rows(config, dataset, subset, split, *max_rows).await,
        },
        source => jsonl_or_json_rows(source),
    }
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
    let mut files = siblings
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
        let url = hugging_face_resolve_url(dataset, commit, &path)?;
        let key = format!("hf-file:{dataset}:{commit}:{path}");
        let body = config
            .fetcher
            .fetch(url.as_str(), &key, config.bearer_token.as_deref())
            .await?;
        let remaining = max_rows.map(|cap| cap.saturating_sub(rows.len()));
        rows.extend(rows_from_remote_bytes(
            body,
            &format!("hf://{dataset}@{commit}/{path}"),
            remaining,
        )?);
    }
    if let Some(cap) = max_rows {
        rows.truncate(cap);
    }
    Ok(rows)
}

async fn load_hugging_face_rows(
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
                object.remove("data").unwrap().as_array().unwrap().clone()
            }
            value => vec![value],
        };
        return rows_from_values(values, label);
    }
    crate::loader::rows_from_bytes(bytes, None)
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
    for row in rows {
        let Some(turns) = row.value.get(field).and_then(Value::as_array) else {
            continue;
        };
        let selected = if multi_turn {
            turns.as_slice()
        } else {
            &turns[..turns.len().min(1)]
        };
        let mut conversation = Conversation::new(ids.next_id());
        let mut parent = None;
        for value in selected {
            let text = value_text(value)?.trim().to_string();
            if text.is_empty() {
                continue;
            }
            let tokens = tokenizer.encode(&text)?;
            let handle = segments.intern_text(
                parent,
                "user",
                Bytes::from(text),
                tokens.clone().into_boxed_slice(),
            )?;
            parent = Some(handle);
            let mut turn = Turn {
                input_tokens: tokens.len() as u64,
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![handle],
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
    finalizer: &mut crate::compose::TurnFinalizer<'_>,
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
                turn.input_tokens = turn
                    .input_tokens
                    .checked_add(tokens.len() as u64)
                    .ok_or_else(|| DatasetError::Validation("input token overflow".into()))?;
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
            });
        }
    }
    finalizer.finalize_turn(&mut turn)?;
    Ok(turn)
}

fn media_from_value(value: Option<&Value>, kind: MediaKind) -> Vec<AuthoredMedia> {
    let Some(value) = value else {
        return Vec::new();
    };
    let items = value
        .as_array()
        .map_or_else(|| vec![value], |values| values.iter().collect());
    let contents = items
        .into_iter()
        .filter_map(|value| match value {
            Value::String(value) if !value.is_empty() => Some(value.clone()),
            Value::Object(object) => object
                .get("src")
                .or_else(|| object.get("url"))
                .or_else(|| object.get("path"))
                .and_then(Value::as_str)
                .map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!contents.is_empty())
        .then_some(AuthoredMedia {
            kind,
            name: String::new(),
            contents,
        })
        .into_iter()
        .collect()
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
    use std::sync::Arc;
    use std::sync::Mutex;

    use aiperf_rng::RngRoot;
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use serde_json::json;

    use super::*;
    use crate::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::tokenizer::TiktokenTokenizer;

    struct MockRevisionFetcher {
        info: Bytes,
        file: Bytes,
        urls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl crate::fetch::DatasetFetcher for MockRevisionFetcher {
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
    ) -> Result<crate::Dataset> {
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
            json!([{"prompt":"Q?","ground_truth":"A","task":"math","correlation_id":"problem-1","metadata":{"generation_size":5},"extra_body":{"temperature":0.2,"stop":["Q:"]}}]),
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
            json!([{"prompt":"Q?","ground_truth":"A","task":"math","extra_body":[]}]),
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
            matches!(text, crate::Payload::Text { bytes, .. } if bytes == "Which option? A.alpha B.beta")
        );
        let video = dataset.segments().get(turn.content[1].handles[0]).unwrap();
        assert!(
            matches!(video, crate::Payload::Media { kind: MediaKind::Video, bytes } if bytes == "https://example.com/video.mp4")
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
}
