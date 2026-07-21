// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent-pool multimodal sampling.
//!
//! Each file or named inline collection is a pool. With unit batch sizes one
//! entry is sampled with replacement from every pool and merged; with any
//! non-unit batch size, modality associations are deliberately flattened before
//! sampling.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::rng::RandomGenerator;
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::Value;
use smallvec::SmallVec;

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::{
    DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow, RowOrigin, jsonl_rows,
};
use crate::dataset::model::{ContentGroup, Conversation, MediaKind, Turn};
use crate::dataset::segment::{Handle, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

/// Loader for one or more independent random content pools.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomPoolDatasetLoader;

/// Composer that samples random-pool rows with replacement.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomPoolComposer;

#[derive(Debug, Clone, Deserialize)]
struct NamedContents {
    #[serde(default)]
    name: String,
    contents: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Contents {
    Strings(Vec<String>),
    Named(Vec<NamedContents>),
}

#[derive(Debug, Clone, Deserialize)]
struct PoolEntry {
    #[serde(default, rename = "type")]
    _type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    texts: Option<Contents>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    images: Option<Contents>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    audios: Option<Contents>,
    #[serde(default)]
    video: Option<String>,
    #[serde(default)]
    videos: Option<Contents>,
}

impl PoolEntry {
    fn parse(value: Value, origin: &impl std::fmt::Display) -> Result<Self> {
        let entry: Self = serde_json::from_value(value).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid random_pool row: {error}"))
        })?;
        for (single, multiple, label) in [
            (&entry.text, &entry.texts, "text/texts"),
            (&entry.image, &entry.images, "image/images"),
            (&entry.audio, &entry.audios, "audio/audios"),
            (&entry.video, &entry.videos, "video/videos"),
        ] {
            if single.is_some() && multiple.is_some() {
                return Err(DatasetError::Validation(format!(
                    "{origin}: {label} are mutually exclusive"
                )));
            }
        }
        let populated = [
            entry.text.as_ref().is_some_and(|value| !value.is_empty()),
            has_contents(entry.texts.as_ref()),
            entry.image.as_ref().is_some_and(|value| !value.is_empty()),
            has_contents(entry.images.as_ref()),
            entry.audio.as_ref().is_some_and(|value| !value.is_empty()),
            has_contents(entry.audios.as_ref()),
            entry.video.as_ref().is_some_and(|value| !value.is_empty()),
            has_contents(entry.videos.as_ref()),
        ]
        .into_iter()
        .any(|value| value);
        if !populated {
            return Err(DatasetError::Validation(format!(
                "{origin}: random_pool row requires at least one modality"
            )));
        }
        Ok(entry)
    }

    fn groups(&self, kind: MediaKind, default_name: &str) -> Vec<(String, Vec<String>)> {
        let (single, multiple) = match kind {
            MediaKind::Text => (&self.text, &self.texts),
            MediaKind::Image => (&self.image, &self.images),
            MediaKind::Audio => (&self.audio, &self.audios),
            MediaKind::Video => (&self.video, &self.videos),
        };
        match (single, multiple) {
            (Some(value), None) => vec![(default_name.to_string(), vec![value.clone()])],
            (None, Some(Contents::Strings(values))) => {
                vec![(default_name.to_string(), values.clone())]
            }
            (None, Some(Contents::Named(groups))) => groups
                .iter()
                .map(|group| (group.name.clone(), group.contents.clone()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn has_contents(contents: Option<&Contents>) -> bool {
    match contents {
        Some(Contents::Strings(values)) => values.iter().any(|value| !value.is_empty()),
        Some(Contents::Named(groups)) => groups
            .iter()
            .flat_map(|group| &group.contents)
            .any(|value| !value.is_empty()),
        None => false,
    }
}

#[async_trait]
impl DatasetLoader for RandomPoolDatasetLoader {
    fn name(&self) -> &str {
        "random_pool"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        if let Some(path) = probe.path.as_deref().filter(|path| path.is_dir()) {
            return directory_has_random_pool(path);
        }
        probe.value.as_ref().is_some_and(|value| {
            value.get("type").and_then(Value::as_str) == Some("random_pool")
                && PoolEntry::parse(value.clone(), &"probe").is_ok()
        })
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let mut rows = Vec::new();
        match &config.source {
            DatasetSource::Path(path) if path.is_dir() => {
                let mut files = std::fs::read_dir(path)?
                    .filter_map(std::result::Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.is_file())
                    .collect::<Vec<_>>();
                files.sort();
                for file in files {
                    let group = file
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("pool")
                        .to_string();
                    for mut row in jsonl_rows(&DatasetSource::Path(file))? {
                        PoolEntry::parse(row.value.clone(), &row.origin)?;
                        row.group_key = Some(group.clone());
                        rows.push(row);
                    }
                }
            }
            DatasetSource::Inline(Value::Object(pools))
                if pools.values().all(Value::is_array)
                    && !pools.contains_key("text")
                    && !pools.contains_key("image") =>
            {
                for (pool, values) in pools {
                    for (index, value) in values
                        .as_array()
                        .expect("guarded as arrays")
                        .iter()
                        .cloned()
                        .enumerate()
                    {
                        let origin = RowOrigin::JsonPointer {
                            path: None,
                            pointer: format!("/{pool}/{index}"),
                        };
                        PoolEntry::parse(value.clone(), &origin)?;
                        rows.push(RawRow {
                            value,
                            wire: None,
                            session_id: None,
                            group_key: Some(pool.clone()),
                            origin,
                        });
                    }
                }
            }
            source => {
                rows = jsonl_rows(source)?;
                for row in &mut rows {
                    PoolEntry::parse(row.value.clone(), &row.origin)?;
                    row.group_key = Some("<inline>".into());
                }
            }
        }
        if rows.is_empty() {
            return Err(DatasetError::Validation(
                "random_pool source contains no rows".into(),
            ));
        }
        Ok(rows)
    }
}

fn directory_has_random_pool(directory: &Path) -> bool {
    std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .any(|path| {
            std::fs::read(path).ok().is_some_and(|bytes| {
                bytes
                    .split(|byte| *byte == b'\n')
                    .map(crate::dataset::loader::trim_ascii)
                    .find(|line| !line.is_empty())
                    .and_then(|line| serde_json::from_slice::<Value>(line).ok())
                    .is_some_and(|value| PoolEntry::parse(value, &"probe").is_ok())
            })
        })
}

impl Composer for RandomPoolComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let num_conversations = option_usize(config, "num_conversations", 1)?;
        let batches = BatchSizes {
            text: option_usize(config, "text_batch_size", 1)?,
            image: option_usize(config, "image_batch_size", 1)?,
            audio: option_usize(config, "audio_batch_size", 1)?,
            video: option_usize(config, "video_batch_size", 1)?,
        };
        let mut pools = BTreeMap::<String, Vec<PoolEntry>>::new();
        for row in rows {
            let pool = row.group_key.clone().unwrap_or_else(|| "<inline>".into());
            pools
                .entry(pool)
                .or_default()
                .push(PoolEntry::parse(row.value, &row.origin)?);
        }
        let mut rng = RandomGenerator::from_seed(
            config
                .rng_root
                .derive_seed("dataset.loader.random_pool.sampling"),
        );
        let sampled = if batches.all_unit() {
            sample_associated(&pools, num_conversations, &mut rng)?
        } else {
            sample_flattened(&pools, num_conversations, batches, &mut rng)?
        };
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(sampled.len());
        for groups in sampled {
            let mut turn = Turn::default();
            let mut parent = None;
            for group in groups {
                intern_group(&mut turn, &mut parent, group, config, tokenizer, segments)?;
            }
            finalizer.finalize_turn(&mut turn)?;
            let mut conversation = Conversation::new(ids.next_id());
            conversation.turns.push(turn);
            conversations.push(conversation);
        }
        Ok(conversations)
    }
}

#[derive(Debug, Clone, Copy)]
struct BatchSizes {
    text: usize,
    image: usize,
    audio: usize,
    video: usize,
}

impl BatchSizes {
    fn all_unit(self) -> bool {
        [self.text, self.image, self.audio, self.video]
            .into_iter()
            .all(|size| size == 1)
    }

    fn get(self, kind: MediaKind) -> usize {
        match kind {
            MediaKind::Text => self.text,
            MediaKind::Image => self.image,
            MediaKind::Audio => self.audio,
            MediaKind::Video => self.video,
        }
    }
}

#[derive(Debug, Clone)]
struct SampledGroup {
    kind: MediaKind,
    name: String,
    contents: Vec<String>,
}

fn sample_associated(
    pools: &BTreeMap<String, Vec<PoolEntry>>,
    count: usize,
    rng: &mut RandomGenerator,
) -> Result<Vec<Vec<SampledGroup>>> {
    let mut output = vec![Vec::new(); count];
    for (pool_name, entries) in pools {
        if entries.is_empty() {
            return Err(DatasetError::Validation(format!(
                "random pool {pool_name:?} is empty"
            )));
        }
        let default_name = Path::new(pool_name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(pool_name);
        for groups in &mut output {
            let entry = rng
                .choice(entries)
                .expect("empty pools rejected before sampling");
            for kind in [
                MediaKind::Text,
                MediaKind::Image,
                MediaKind::Audio,
                MediaKind::Video,
            ] {
                groups.extend(entry.groups(kind, default_name).into_iter().map(
                    |(name, contents)| SampledGroup {
                        kind,
                        name,
                        contents,
                    },
                ));
            }
        }
    }
    Ok(output)
}

fn sample_flattened(
    pools: &BTreeMap<String, Vec<PoolEntry>>,
    count: usize,
    batches: BatchSizes,
    rng: &mut RandomGenerator,
) -> Result<Vec<Vec<SampledGroup>>> {
    let mut flat = HashMap::<MediaKind, Vec<String>>::new();
    for entries in pools.values() {
        for entry in entries {
            for kind in [
                MediaKind::Text,
                MediaKind::Image,
                MediaKind::Audio,
                MediaKind::Video,
            ] {
                for (_, contents) in entry.groups(kind, "") {
                    flat.entry(kind).or_default().extend(contents);
                }
            }
        }
    }
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        let mut groups = Vec::new();
        for kind in [
            MediaKind::Text,
            MediaKind::Image,
            MediaKind::Audio,
            MediaKind::Video,
        ] {
            let Some(pool) = flat.get(&kind) else {
                continue;
            };
            let batch_size = batches.get(kind);
            if batch_size == 0 {
                continue;
            }
            let mut contents = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                contents.push(
                    rng.choice(pool)
                        .expect("flattened pool contains authored content")
                        .clone(),
                );
            }
            groups.push(SampledGroup {
                kind,
                name: String::new(),
                contents,
            });
        }
        output.push(groups);
    }
    Ok(output)
}

fn intern_group(
    turn: &mut Turn,
    parent: &mut Option<Handle>,
    group: SampledGroup,
    config: &ComposeConfig,
    tokenizer: &dyn TextTokenizer,
    segments: &mut SegmentPool,
) -> Result<()> {
    let mut handles = SmallVec::new();
    for content in group.contents {
        let handle = if group.kind == MediaKind::Text {
            let tokens = tokenizer.encode(&content)?;
            turn.input_tokens = turn
                .input_tokens
                .checked_add(tokens.len() as u64)
                .ok_or_else(|| {
                    DatasetError::Validation("input token count overflowed u64".into())
                })?;
            segments.intern_text(
                *parent,
                "user",
                Bytes::from(content),
                tokens.into_boxed_slice(),
            )?
        } else {
            let bytes = config.media_resolver.resolve(group.kind, &content)?;
            segments.intern_media(*parent, group.kind, bytes)?
        };
        *parent = Some(handle);
        handles.push(handle);
    }
    if !handles.is_empty() {
        turn.content.push(ContentGroup {
            kind: group.kind,
            name: group.name,
            handles,
            uuids: smallvec::smallvec![],
        });
    }
    Ok(())
}

fn option_usize(config: &ComposeConfig, key: &str, default: usize) -> Result<usize> {
    config
        .format_options
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    DatasetError::Validation(format!("random_pool option {key} must be a usize"))
                })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, LoaderRegistry};
    use crate::dataset::tokenizer::TiktokenTokenizer;

    #[tokio::test]
    async fn every_named_pool_contributes_and_sampling_reproduces() {
        let source = DatasetSource::Inline(json!({
            "queries": [{"text":"q1"},{"text":"q2"}],
            "images": [{"image":"https://example.com/1.png"}]
        }));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(7)));
        compose
            .format_options
            .insert("num_conversations".into(), Value::from(3));
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(RandomPoolDatasetLoader),
                Arc::new(RandomPoolComposer),
            ))
            .unwrap();
        let first = registry
            .build_dataset(
                Some("random_pool"),
                &LoadConfig::new(source.clone()),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let second = registry
            .build_dataset(
                Some("random_pool"),
                &LoadConfig::new(source),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(first.conversations(), second.conversations());
        assert_eq!(first.conversations().len(), 3);
        assert!(first.conversations().iter().all(|conversation| {
            let turn = &conversation.turns[0];
            turn.content
                .iter()
                .any(|group| group.kind == MediaKind::Text)
                && turn
                    .content
                    .iter()
                    .any(|group| group.kind == MediaKind::Image)
        }));
    }
}
