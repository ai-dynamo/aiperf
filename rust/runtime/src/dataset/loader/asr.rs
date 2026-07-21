// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hugging Face automatic-speech-recognition dataset format.
//!
//! Provides a configurable audio column, decode-disabled raw bytes, WAV
//! normalization, exact duration metadata, the fixed transcription prompt, a
//! 30-second default cap, and row skipping for absent/invalid/long audio. Remote
//! audio assets use the injected dataset fetcher and therefore the same
//! Clock-injected hyper stack as row downloads.

use std::path::Path;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::Value;
use smallvec::smallvec;

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::generator::transcode_audio_to_wav;
use crate::dataset::loader::public::load_public_rows;
use crate::dataset::loader::{DatasetLoader, DatasetProbe, LoadConfig, RawRow};
use crate::dataset::model::{ContentGroup, Conversation, MediaKind, Turn};
use crate::dataset::segment::{Role, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

const ASR_PROMPT: &str = "Transcribe this audio.";
const DEFAULT_MAX_DURATION_SECONDS: f64 = 30.0;
const NORMALIZED_AUDIO_FIELD: &str = "__aiperf_asr_wav";
const DURATION_FIELD: &str = "__aiperf_asr_duration_seconds";

/// Hugging Face ASR loader and audio normalizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfAsrDatasetLoader;

/// ASR row-to-conversation composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfAsrComposer;

#[async_trait]
impl DatasetLoader for HfAsrDatasetLoader {
    fn name(&self) -> &str {
        "hf_asr"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe
            .value
            .as_ref()
            .and_then(|value| value.get("__aiperf_hf_asr"))
            .and_then(Value::as_bool)
            == Some(true)
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let audio_column = config
            .options
            .get("audio_column")
            .and_then(Value::as_str)
            .unwrap_or("audio");
        let max_duration = config
            .options
            .get("max_audio_duration_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_MAX_DURATION_SECONDS);
        if !max_duration.is_finite() || max_duration <= 0.0 {
            return Err(DatasetError::Validation(
                "max_audio_duration_seconds must be finite and positive".into(),
            ));
        }
        let max_conversations = config
            .options
            .get("max_conversations")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok());
        let rows = load_public_rows(config).await?;
        let mut normalized = Vec::new();
        let mut skipped = 0_usize;
        for mut row in rows {
            if max_conversations.is_some_and(|cap| normalized.len() >= cap) {
                break;
            }
            let Some(audio) = row.value.get(audio_column) else {
                skipped += 1;
                continue;
            };
            let raw = match resolve_audio_bytes(audio, config).await {
                Ok(Some(raw)) => raw,
                Ok(None) | Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let (wav, duration) = match transcode_audio_to_wav(&raw) {
                Ok(value) => value,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if duration > max_duration {
                skipped += 1;
                continue;
            }
            let object = row.value.as_object_mut().ok_or_else(|| {
                DatasetError::Validation(format!("{}: ASR row must be an object", row.origin))
            })?;
            object.insert(
                NORMALIZED_AUDIO_FIELD.into(),
                Value::String(STANDARD.encode(wav)),
            );
            object.insert(DURATION_FIELD.into(), Value::from(duration));
            normalized.push(row);
        }
        if normalized.is_empty() {
            return Err(DatasetError::Validation(format!(
                "ASR dataset produced no conversations; skipped {skipped} absent, invalid, or over-{max_duration}s audio rows from column {audio_column:?}"
            )));
        }
        Ok(normalized)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

impl Composer for HfAsrComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let prompt_tokens = tokenizer.encode(ASR_PROMPT)?;
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let object = row.value.as_object().ok_or_else(|| {
                DatasetError::Validation(format!("{}: ASR row must be an object", row.origin))
            })?;
            let wav = object
                .get(NORMALIZED_AUDIO_FIELD)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "{}: ASR loader did not normalize audio",
                        row.origin
                    ))
                })?;
            let wav = STANDARD.decode(wav).map_err(|error| {
                DatasetError::Validation(format!(
                    "{}: invalid normalized audio: {error}",
                    row.origin
                ))
            })?;
            let duration = object
                .get(DURATION_FIELD)
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    DatasetError::Validation(format!("{}: ASR duration is missing", row.origin))
                })?;
            let prompt = segments.intern_text(
                None,
                "user",
                Bytes::from_static(ASR_PROMPT.as_bytes()),
                prompt_tokens.clone().into_boxed_slice(),
            )?;
            let audio = segments.intern_media(
                Some(prompt),
                MediaKind::Audio,
                Bytes::from(format!("wav,{}", STANDARD.encode(wav))),
            )?;
            let mut turn = Turn {
                role: Some(Role::from("user")),
                input_tokens: prompt_tokens.len() as u64,
                audio_duration_seconds: Some(duration),
                content: smallvec![
                    ContentGroup {
                        kind: MediaKind::Text,
                        name: "text".into(),
                        handles: smallvec![prompt],
                        uuids: smallvec![],
                    },
                    ContentGroup {
                        kind: MediaKind::Audio,
                        name: "input_audio".into(),
                        handles: smallvec![audio],
                        uuids: smallvec![],
                    }
                ],
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

async fn resolve_audio_bytes(value: &Value, config: &LoadConfig) -> Result<Option<Bytes>> {
    // The HuggingFace datasets-server serves an Audio column as a single-element
    // array (`[{"src": <signed url>}]`); unwrap it to the inner audio object.
    let value = match value {
        Value::Array(items) => match items.first() {
            Some(item) => item,
            None => return Ok(None),
        },
        other => other,
    };
    let reference = match value {
        Value::String(value) => Some(value.as_str()),
        Value::Object(object) => {
            if let Some(bytes) = object.get("bytes").and_then(Value::as_str) {
                return decode_embedded_audio(bytes).map(Some);
            }
            object
                .get("src")
                .or_else(|| object.get("path"))
                .and_then(Value::as_str)
        }
        _ => None,
    };
    let Some(reference) = reference else {
        return Ok(None);
    };
    if reference.starts_with("data:") || looks_like_base64_audio(reference) {
        return decode_embedded_audio(reference).map(Some);
    }
    if reference.contains("://") {
        let cache_key = format!("hf-audio:{}", blake3::hash(reference.as_bytes()).to_hex());
        return config
            .fetcher
            .fetch(reference, &cache_key, config.bearer_token.as_deref())
            .await
            .map(Some);
    }
    let path = Path::new(reference);
    if path.is_file() {
        return std::fs::read(path)
            .map(Bytes::from)
            .map(Some)
            .map_err(Into::into);
    }
    Ok(None)
}

fn decode_embedded_audio(value: &str) -> Result<Bytes> {
    let encoded = if value.starts_with("data:") {
        value
            .split_once(',')
            .map(|(_, encoded)| encoded)
            .ok_or_else(|| {
                DatasetError::Validation("audio data URI has no comma separator".into())
            })?
    } else {
        value
            .split_once(',')
            .filter(|(format, _)| matches!(format.to_ascii_lowercase().as_str(), "wav" | "mp3"))
            .map_or(value, |(_, encoded)| encoded)
    };
    STANDARD
        .decode(encoded)
        .map(Bytes::from)
        .map_err(|error| DatasetError::Validation(format!("invalid base64 audio: {error}")))
}

fn looks_like_base64_audio(value: &str) -> bool {
    value.starts_with("wav,") || value.starts_with("mp3,")
}

#[cfg(test)]
mod tests {
    use crate::rng::{RngRoot, SamplingDistribution};
    use serde_json::json;

    use super::*;
    use crate::dataset::generator::{
        NativeAudioGenerator, SyntheticAudioConfig, SyntheticMediaGenerator,
    };
    use crate::dataset::loader::{DatasetSource, LoaderRegistry};
    use crate::dataset::segment::Payload;
    use crate::dataset::tokenizer::TiktokenTokenizer;

    #[tokio::test]
    async fn asr_normalizes_audio_skips_long_rows_and_carries_duration() {
        let mut generator = NativeAudioGenerator::new(
            SyntheticAudioConfig {
                batch_size: 1,
                duration_seconds: SamplingDistribution::fixed(0.1).unwrap(),
                ..SyntheticAudioConfig::default()
            },
            RngRoot::new(Some(8)),
        )
        .unwrap();
        let audio = generator.generate().unwrap();
        let encoded = std::str::from_utf8(&audio.wire).unwrap().to_string();
        let source = DatasetSource::Inline(json!([
            {"__aiperf_hf_asr": true, "sound": {"bytes": encoded}},
            {"__aiperf_hf_asr": true, "sound": null}
        ]));
        let mut load = LoadConfig::new(source);
        load.options
            .insert("audio_column".into(), Value::String("sound".into()));
        let compose = ComposeConfig::new("whisper", RngRoot::new(Some(8)));
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("hf_asr"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.audio_duration_seconds, Some(0.1));
        let audio = turn.content[1].handles[0];
        assert!(
            matches!(dataset.segments().get(audio).unwrap(), Payload::Media { kind: MediaKind::Audio, bytes } if bytes.starts_with(b"wav,"))
        );
    }
}
