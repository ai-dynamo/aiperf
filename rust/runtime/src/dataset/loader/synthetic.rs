// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic conversation and rankings formats.
//!
//! Prompt generation, paired ISL/OSL sampling,
//! reusable prefixes, per-session context, multimodal batches, turn delays, and
//! final model/max-token selection are all resolved before the pool is frozen.
//! Token-native composition uses a no-decode branch: exact IDs enter the segment
//! arena directly and no temporary text payload is constructed.

use crate::rng::{ConfiguredRandomGenerator, RandomGenerator, RngRoot};
use async_trait::async_trait;
use bytes::Bytes;
use smallvec::SmallVec;

use crate::dataset::compose::{ComposeConfig, Composer, SessionIdGenerator};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::generator::{
    SyntheticDatasetConfig, SyntheticMediaGenerator, SyntheticPrefixConfig, SyntheticPromptConfig,
};
use crate::dataset::loader::{DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow};
use crate::dataset::model::{ContentGroup, Conversation, ConversationContextMode, MediaKind, Turn};
use crate::dataset::prompt::{GeneratedPrompt, PromptGenerator};
use crate::dataset::segment::{Handle, Role, SegmentPool};
use crate::dataset::tokenizer::TextTokenizer;

/// Pure synthetic source marker loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticDatasetLoader;

/// Synthetic multimodal conversation composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticComposer;

/// Pure synthetic rankings source marker loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticRankingsDatasetLoader;

/// Synthetic query/passage rankings composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticRankingsComposer;

#[async_trait]
impl DatasetLoader for SyntheticDatasetLoader {
    fn name(&self) -> &str {
        "synthetic"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe
            .value
            .as_ref()
            .and_then(|value| value.get("__aiperf_synthetic"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        require_inline_synthetic_source(&config.source)?;
        Ok(Vec::new())
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

#[async_trait]
impl DatasetLoader for SyntheticRankingsDatasetLoader {
    fn name(&self) -> &str {
        "synthetic_rankings"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        probe
            .value
            .as_ref()
            .and_then(|value| value.get("__aiperf_synthetic_rankings"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        require_inline_synthetic_source(&config.source)?;
        Ok(Vec::new())
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
}

impl Composer for SyntheticComposer {
    fn compose(
        &self,
        _rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let shape = config.synthetic_config.as_ref().ok_or_else(|| {
            DatasetError::Validation(
                "synthetic format requires ComposeConfig.synthetic_config".into(),
            )
        })?;
        validate_shape(shape)?;
        validate_raw_token_shape(config, shape)?;

        let mut prompt_generator = if shape.prompts.is_some() || has_prefixes(&shape.prefixes) {
            Some(config.prompt_generator.create(tokenizer, config.rng_root)?)
        } else {
            None
        };
        let mut prefix_rng: ConfiguredRandomGenerator = config
            .rng_root
            .derive_generator("dataset.prompt.prefix.selection");
        let prefix_pool = if shape.prefixes.pool_size.is_some() {
            let generator = prompt_generator.as_mut().ok_or_else(|| {
                DatasetError::Validation("synthetic prefixes require a tokenizer".into())
            })?;
            build_prefix_pool(
                &shape.prefixes,
                generator.as_mut(),
                config.requires_raw_token_ids,
            )?
        } else {
            Vec::new()
        };
        let mut prefix_reuse = shape
            .prompts
            .as_ref()
            .and_then(|prompt| PrefixReuse::from_prompt(prompt, config.rng_root));
        let generated_system = if let Some(tokens) = shape.prefixes.shared_system_tokens {
            let generator = prompt_generator.as_mut().ok_or_else(|| {
                DatasetError::Validation("synthetic context prompts require a tokenizer".into())
            })?;
            Some(generated_context(generator.as_mut(), tokens, i64::MIN)?)
        } else {
            None
        };

        let mut image_generator = shape
            .images
            .as_ref()
            .filter(|image| image.batch_size > 0)
            .map(|image| config.media_generator_factory.image(image, config.rng_root))
            .transpose()?;
        let mut audio_generator = shape
            .audio
            .as_ref()
            .filter(|audio| audio.batch_size > 0)
            .map(|audio| config.media_generator_factory.audio(audio, config.rng_root))
            .transpose()?;
        let mut video_generator = shape
            .video
            .as_ref()
            .filter(|video| video.batch_size > 0)
            .map(|video| config.media_generator_factory.video(video, config.rng_root))
            .transpose()?;

        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut turn_rng = component_rng(config.rng_root, "composer.conversation.turn_count");
        let mut delay_rng = component_rng(config.rng_root, "composer.conversation.turn_delay");
        let mut length_rng = component_rng(config.rng_root, "composer.turn.sequence_length");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(shape.entries);
        let mut random_range_ordinal = 0_usize;
        let mut has_warned_random_range_exhausted = false;

        for conversation_index in 0..shape.entries {
            let mut conversation = Conversation::new(ids.next_id());
            if config.requires_raw_token_ids {
                conversation.context_mode =
                    Some(ConversationContextMode::MessageArrayWithResponses);
            }
            let mut parent = None;
            let system = config
                .shared_system_prompt
                .as_deref()
                .or(generated_system.as_deref());
            if let Some(system) = system {
                conversation.system =
                    Some(intern_text(segments, parent, "system", system, tokenizer)?);
                parent = conversation.system;
            }
            let explicit_context = config.user_context_prompts.get(conversation_index);
            let generated_context = if explicit_context.is_none()
                && let Some(tokens) = shape.prefixes.user_context_tokens
            {
                let generator = prompt_generator.as_mut().ok_or_else(|| {
                    DatasetError::Validation("synthetic context prompts require a tokenizer".into())
                })?;
                Some(generated_context(
                    generator.as_mut(),
                    tokens,
                    i64::try_from(conversation_index)
                        .map_err(|_| DatasetError::Validation("too many conversations".into()))?,
                )?)
            } else {
                None
            };
            if let Some(context) = explicit_context
                .map(String::as_str)
                .or(generated_context.as_deref())
            {
                conversation.user_context =
                    Some(intern_text(segments, parent, "user", context, tokenizer)?);
                parent = conversation.user_context;
            }

            let turns = shape
                .turns
                .sample_int(&mut turn_rng)
                .map_err(|error| DatasetError::Validation(error.to_string()))?;
            let turns = usize::try_from(turns).map_err(|_| {
                DatasetError::Validation(format!("sampled turn count {turns} exceeds usize"))
            })?;
            for turn_index in 0..turns {
                let mut turn = Turn {
                    role: Some(Role::from("user")),
                    ..Turn::default()
                };
                if turn_index > 0 && shape.turn_delay_ms.expected_value() > 0.0 {
                    let delay = shape
                        .turn_delay_ms
                        .sample(&mut delay_rng)
                        .map_err(|error| DatasetError::Validation(error.to_string()))?
                        .ceil()
                        * shape.turn_delay_ratio;
                    turn.delay_ms = Some(delay);
                }

                let paired_lengths = if let Some(plan) = &config.random_range_plan {
                    let pair = if let Some(pair) = plan.pair(random_range_ordinal) {
                        pair
                    } else {
                        if !has_warned_random_range_exhausted {
                            tracing::warn!(
                                component = "dataset.random_range",
                                preseed_size = plan.inputs().len(),
                                turn_ordinal = random_range_ordinal,
                                "random range cache exhausted; falling back to deterministic sampling"
                            );
                            has_warned_random_range_exhausted = true;
                        }
                        plan.fallback_pair(&mut length_rng)?
                    };
                    random_range_ordinal = random_range_ordinal.saturating_add(1);
                    Some(pair)
                } else {
                    config
                        .sequence_length_distribution
                        .as_ref()
                        .map(|distribution| distribution.sample(&mut length_rng))
                        .transpose()
                        .map_err(|error| DatasetError::Validation(error.to_string()))?
                };
                if let Some((_, output)) = paired_lengths {
                    turn.max_tokens = Some(u32::try_from(output).map_err(|_| {
                        DatasetError::Validation(format!("sampled OSL {output} exceeds u32"))
                    })?);
                }

                if let Some(prompt) = &shape.prompts {
                    let input_tokens = match paired_lengths {
                        Some((input, _)) => usize::try_from(input).map_err(|_| {
                            DatasetError::Validation(format!("sampled ISL {input} exceeds usize"))
                        })?,
                        None => {
                            let sampled = usize::try_from(
                                prompt
                                    .input_tokens
                                    .sample_int(&mut length_rng)
                                    .map_err(|error| DatasetError::Validation(error.to_string()))?,
                            )
                            .map_err(|_| {
                                DatasetError::Validation("sampled ISL exceeds usize".into())
                            })?;
                            sampled
                                .saturating_sub(prompt.input_token_subtraction)
                                .max(1)
                        }
                    };
                    let generator = prompt_generator.as_deref_mut().ok_or_else(|| {
                        DatasetError::Validation("synthetic prompts require a tokenizer".into())
                    })?;
                    if config.requires_raw_token_ids {
                        let token_ids = if let Some(reuse) = prefix_reuse.as_mut() {
                            reuse.prompt_token_ids(generator, input_tokens)?
                        } else {
                            let selected_prefix = (turn_index == 0 && !prefix_pool.is_empty())
                                .then(|| prefix_rng.choice(&prefix_pool).ok())
                                .flatten();
                            if let Some(prefix) = selected_prefix {
                                generator
                                    .generate_token_ids_with_prefix(input_tokens, &prefix.tokens)?
                            } else {
                                generator.generate_token_ids(input_tokens, &[], 1)?
                            }
                        };
                        turn.input_tokens = Some(token_ids.len() as u64);
                        let handle = segments.intern_token_ids(parent, token_ids)?;
                        parent = Some(handle);
                        turn.body = Turn::dispatch_body(None, Some(handle), &[]);
                    } else if let Some(reuse) = prefix_reuse.as_mut() {
                        let mut handles = SmallVec::new();
                        for _ in 0..prompt.batch_size {
                            // Token-native reuse: the shared prefix and unique
                            // suffix are assembled as exact ids, so `input_tokens`
                            // is hit exactly and the decoded text carries a
                            // byte-identical leading run across warm prompts.
                            let tokens = reuse.prompt_tokens(generator, input_tokens)?;
                            turn.input_tokens = Some(
                                turn.input_tokens
                                    .unwrap_or(0)
                                    .checked_add(tokens.len() as u64)
                                    .ok_or_else(|| {
                                        DatasetError::Validation(
                                            "input token count overflow".into(),
                                        )
                                    })?,
                            );
                            let text = tokenizer.decode(&tokens)?;
                            let handle = segments.intern_text(
                                parent,
                                "user",
                                Bytes::from(text),
                                tokens.into_boxed_slice(),
                            )?;
                            parent = Some(handle);
                            handles.push(handle);
                        }
                        turn.content.push(ContentGroup {
                            kind: MediaKind::Text,
                            name: "text".into(),
                            handles,
                            uuids: SmallVec::new(),
                        });
                    } else {
                        let mut handles = SmallVec::new();
                        for _ in 0..prompt.batch_size {
                            let selected_prefix = (turn_index == 0 && !prefix_pool.is_empty())
                                .then(|| prefix_rng.choice(&prefix_pool).ok().cloned())
                                .flatten();
                            let generated = if let Some(prefix) = selected_prefix.as_ref() {
                                generator.generate_with_prefix(input_tokens, &prefix.tokens)?
                            } else {
                                generator.generate(input_tokens, &[], 1)?
                            };
                            turn.input_tokens = Some(
                                turn.input_tokens
                                    .unwrap_or(0)
                                    .checked_add(generated.tokens.len() as u64)
                                    .ok_or_else(|| {
                                        DatasetError::Validation(
                                            "input token count overflow".into(),
                                        )
                                    })?,
                            );
                            // The full authored text remains one endpoint value, while
                            // the hidden prefix parent makes reuse visible in the
                            // content-addressed chain without changing wire shape.
                            let content_parent = if let Some(prefix) = selected_prefix {
                                Some(segments.intern_text(
                                    parent,
                                    "user",
                                    Bytes::from(prefix.text),
                                    prefix.tokens.into_boxed_slice(),
                                )?)
                            } else {
                                parent
                            };
                            let handle = segments.intern_text(
                                content_parent,
                                "user",
                                Bytes::from(generated.text),
                                generated.tokens.into_boxed_slice(),
                            )?;
                            parent = Some(handle);
                            handles.push(handle);
                        }
                        turn.content.push(ContentGroup {
                            kind: MediaKind::Text,
                            name: "text".into(),
                            handles,
                            uuids: SmallVec::new(),
                        });
                    }
                }

                if let Some(generator) = image_generator.as_mut() {
                    append_media_batch(
                        &mut turn,
                        &mut parent,
                        segments,
                        generator.as_mut(),
                        shape.images.as_ref().map_or(0, |value| value.batch_size),
                        "image_url",
                    )?;
                }
                if let Some(generator) = audio_generator.as_mut() {
                    append_media_batch(
                        &mut turn,
                        &mut parent,
                        segments,
                        generator.as_mut(),
                        shape.audio.as_ref().map_or(0, |value| value.batch_size),
                        "input_audio",
                    )?;
                }
                if let Some(generator) = video_generator.as_mut() {
                    append_media_batch(
                        &mut turn,
                        &mut parent,
                        segments,
                        generator.as_mut(),
                        shape.video.as_ref().map_or(0, |value| value.batch_size),
                        "video_url",
                    )?;
                }
                if turn.content.is_empty() && turn.body.is_empty() {
                    return Err(DatasetError::Validation(
                        "synthetic turn generated no text, image, audio, or video content".into(),
                    ));
                }
                finalizer.finalize_turn(&mut turn)?;
                conversation.turns.push(turn);
            }
            conversations.push(conversation);
            // Streaming build: each conversation's prompts/media are interned into
            // `segments` and its heavy intermediates dropped before the next, so a
            // throttled progress line is the only per-conversation work retained.
            crate::dataset::dataset::report_build_progress(
                "synthetic",
                conversation_index + 1,
                shape.entries,
            );
        }
        Ok(conversations)
    }
}

impl Composer for SyntheticRankingsComposer {
    fn compose(
        &self,
        _rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let shape = config.synthetic_config.as_ref().ok_or_else(|| {
            DatasetError::Validation(
                "synthetic_rankings requires ComposeConfig.synthetic_config".into(),
            )
        })?;
        if shape.entries == 0 {
            return Err(DatasetError::Validation(
                "synthetic rankings entries must be positive".into(),
            ));
        }
        let rankings = shape.rankings.clone().unwrap_or_default();
        let mut generator = config.prompt_generator.create(tokenizer, config.rng_root)?;
        let mut count_rng = component_rng(config.rng_root, "dataset.rankings.passages");
        let mut query_rng = component_rng(config.rng_root, "dataset.rankings.query.tokens");
        let mut passage_rng = component_rng(config.rng_root, "dataset.rankings.passages.tokens");
        let mut ids = SessionIdGenerator::new(config.rng_root.seed(), "session");
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::with_capacity(shape.entries);
        for _ in 0..shape.entries {
            let passage_count = usize::try_from(
                rankings
                    .passages
                    .sample_int(&mut count_rng)
                    .map_err(|error| DatasetError::Validation(error.to_string()))?,
            )
            .map_err(|_| DatasetError::Validation("passage count exceeds usize".into()))?;
            let query_tokens = usize::try_from(
                rankings
                    .query_tokens
                    .sample_int(&mut query_rng)
                    .map_err(|error| DatasetError::Validation(error.to_string()))?,
            )
            .map_err(|_| DatasetError::Validation("query length exceeds usize".into()))?;
            let query = generator.generate(query_tokens, &[], 1)?;
            let query_handle = segments.intern_text(
                None,
                "user",
                Bytes::from(query.text),
                query.tokens.into_boxed_slice(),
            )?;
            let mut passage_handles = SmallVec::new();
            let mut parent = Some(query_handle);
            let mut input_tokens = query_tokens as u64;
            for _ in 0..passage_count {
                let length = usize::try_from(
                    rankings
                        .passage_tokens
                        .sample_int(&mut passage_rng)
                        .map_err(|error| DatasetError::Validation(error.to_string()))?,
                )
                .map_err(|_| DatasetError::Validation("passage length exceeds usize".into()))?;
                let passage = generator.generate(length, &[], 1)?;
                let handle = segments.intern_text(
                    parent,
                    "user",
                    Bytes::from(passage.text),
                    passage.tokens.into_boxed_slice(),
                )?;
                parent = Some(handle);
                passage_handles.push(handle);
                input_tokens = input_tokens.checked_add(length as u64).ok_or_else(|| {
                    DatasetError::Validation("ranking input token count overflow".into())
                })?;
            }
            let mut turn = Turn {
                role: Some(Role::from("user")),
                input_tokens: Some(input_tokens),
                content: smallvec::smallvec![
                    ContentGroup {
                        kind: MediaKind::Text,
                        name: "query".into(),
                        handles: smallvec::smallvec![query_handle],
                        uuids: SmallVec::new(),
                    },
                    ContentGroup {
                        kind: MediaKind::Text,
                        name: "passages".into(),
                        handles: passage_handles,
                        uuids: SmallVec::new(),
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

fn require_inline_synthetic_source(source: &DatasetSource) -> Result<()> {
    if matches!(source, DatasetSource::Inline(_)) {
        Ok(())
    } else {
        Err(DatasetError::Validation(
            "synthetic formats use an inline marker source and ComposeConfig.synthetic_config"
                .into(),
        ))
    }
}

fn validate_shape(shape: &SyntheticDatasetConfig) -> Result<()> {
    if shape.entries == 0 || shape.turns.expected_value() < 1.0 {
        return Err(DatasetError::Validation(
            "synthetic entries and expected turn count must be at least one".into(),
        ));
    }
    if !shape.turn_delay_ratio.is_finite() || shape.turn_delay_ratio < 0.0 {
        return Err(DatasetError::Validation(
            "synthetic turn-delay ratio must be finite and non-negative".into(),
        ));
    }
    if shape
        .prompts
        .as_ref()
        .is_some_and(|prompt| prompt.batch_size == 0)
    {
        return Err(DatasetError::Validation(
            "synthetic prompt batch_size must be positive when prompts are enabled".into(),
        ));
    }
    if let Some(prompt) = shape.prompts.as_ref() {
        validate_prefix_reuse(prompt, &shape.prefixes)?;
    }
    let enabled = shape.prompts.is_some()
        || shape
            .images
            .as_ref()
            .is_some_and(|value| value.batch_size > 0)
        || shape
            .audio
            .as_ref()
            .is_some_and(|value| value.batch_size > 0)
        || shape
            .video
            .as_ref()
            .is_some_and(|value| value.batch_size > 0);
    if !enabled {
        return Err(DatasetError::Validation(
            "all synthetic text, image, audio, and video inputs are disabled".into(),
        ));
    }
    validate_prefixes(&shape.prefixes)
}

fn validate_raw_token_shape(config: &ComposeConfig, shape: &SyntheticDatasetConfig) -> Result<()> {
    if !config.requires_raw_token_ids {
        return Ok(());
    }
    let prompt = shape.prompts.as_ref().ok_or_else(|| {
        DatasetError::Validation(
            "raw-token synthetic datasets require text prompts to be enabled".into(),
        )
    })?;
    if prompt.batch_size != 1 {
        return Err(DatasetError::Validation(
            "raw-token synthetic datasets require prompt batch_size=1".into(),
        ));
    }
    if shape.prefixes.shared_system_tokens.is_some()
        || shape.prefixes.user_context_tokens.is_some()
        || config.shared_system_prompt.is_some()
        || !config.user_context_prompts.is_empty()
    {
        return Err(DatasetError::Validation(
            "raw-token synthetic datasets do not support system or user-context text".into(),
        ));
    }
    if shape
        .images
        .as_ref()
        .is_some_and(|value| value.batch_size > 0)
        || shape
            .audio
            .as_ref()
            .is_some_and(|value| value.batch_size > 0)
        || shape
            .video
            .as_ref()
            .is_some_and(|value| value.batch_size > 0)
    {
        return Err(DatasetError::Validation(
            "raw-token synthetic datasets do not support image, audio, or video inputs".into(),
        ));
    }
    Ok(())
}

fn validate_prefix_reuse(
    prompt: &crate::dataset::generator::SyntheticPromptConfig,
    prefixes: &SyntheticPrefixConfig,
) -> Result<()> {
    if !(prompt.prefix_reuse_fraction.is_finite()
        && (0.0..=1.0).contains(&prompt.prefix_reuse_fraction))
    {
        return Err(DatasetError::Validation(
            "synthetic prompt prefix_reuse_fraction must be within [0, 1]".into(),
        ));
    }
    if !(prompt.prefix_reuse_ratio.is_finite() && (0.0..=1.0).contains(&prompt.prefix_reuse_ratio))
    {
        return Err(DatasetError::Validation(
            "synthetic prompt prefix_reuse_ratio must be within [0, 1]".into(),
        ));
    }
    if prompt.prefix_reuse_fraction > 0.0 && prefixes.pool_size.is_some() {
        return Err(DatasetError::Validation(
            "synthetic prompt prefix reuse and the reusable prefix pool are mutually exclusive"
                .into(),
        ));
    }
    Ok(())
}

fn validate_prefixes(prefixes: &SyntheticPrefixConfig) -> Result<()> {
    let pool = prefixes.pool_size.is_some() || prefixes.prefix_tokens.is_some();
    let context = prefixes.shared_system_tokens.is_some() || prefixes.user_context_tokens.is_some();
    if pool && context {
        return Err(DatasetError::Validation(
            "prefix pool and shared-system/user-context modes are mutually exclusive".into(),
        ));
    }
    if prefixes.pool_size.is_some() != prefixes.prefix_tokens.is_some() {
        return Err(DatasetError::Validation(
            "prefix pool_size and prefix_tokens must be configured together".into(),
        ));
    }
    if [
        prefixes.pool_size,
        prefixes.prefix_tokens,
        prefixes.shared_system_tokens,
        prefixes.user_context_tokens,
    ]
    .into_iter()
    .flatten()
    .any(|value| value == 0)
    {
        return Err(DatasetError::Validation(
            "synthetic prefix lengths and pool size must be positive".into(),
        ));
    }
    Ok(())
}

fn has_prefixes(prefixes: &SyntheticPrefixConfig) -> bool {
    prefixes.pool_size.is_some()
        || prefixes.shared_system_tokens.is_some()
        || prefixes.user_context_tokens.is_some()
}

fn build_prefix_pool(
    prefixes: &SyntheticPrefixConfig,
    generator: &mut dyn PromptGenerator,
    requires_raw_token_ids: bool,
) -> Result<Vec<GeneratedPrompt>> {
    let (Some(size), Some(tokens)) = (prefixes.pool_size, prefixes.prefix_tokens) else {
        return Ok(Vec::new());
    };
    (0..size)
        .map(|_| {
            if requires_raw_token_ids {
                Ok(GeneratedPrompt {
                    text: String::new(),
                    tokens: generator.generate_prefix_token_ids(tokens)?,
                })
            } else {
                generator.generate_prefix(tokens)
            }
        })
        .collect()
}

/// Deterministic shared-prefix targeting layered over the corpus block generator.
///
/// A configured share of prompts is steered onto a single reusable leading token
/// run so an upstream KV cache registers real prefix hits; the rest keep their
/// fully independent prompts. The reusable run is extended lazily and its already
/// materialized ids are never rewritten, so every reusing prompt exposes the same
/// leading token sequence no matter what input length it sampled. Because each
/// prompt is stitched together from concrete token ids, the requested
/// `input_tokens` target is met exactly.
struct PrefixReuse {
    /// Probability, in `[0, 1]`, that a prompt is steered onto the reusable run.
    fraction: f64,
    /// Portion of a reusing prompt's input length taken from the reusable run.
    ratio: f64,
    /// Selection stream deciding reuse, kept apart from the corpus sampling draw.
    decision: ConfiguredRandomGenerator,
    /// Reusable prefix ids, held stable once materialized so every hit lines up.
    shared: Vec<u32>,
}

impl PrefixReuse {
    /// Prepare reuse state when the prompt requests a non-zero reuse fraction.
    fn from_prompt(prompt: &SyntheticPromptConfig, root: RngRoot) -> Option<Self> {
        (prompt.prefix_reuse_fraction > 0.0).then(|| Self {
            fraction: prompt.prefix_reuse_fraction,
            ratio: prompt.prefix_reuse_ratio,
            decision: root.derive_generator("dataset.prompt.prefix.reuse"),
            shared: Vec::new(),
        })
    }

    /// Build one exact-length prompt, leading it with the reusable run for the
    /// deterministically selected reusing share of prompts.
    fn prompt_tokens(
        &mut self,
        generator: &mut dyn PromptGenerator,
        input_tokens: usize,
    ) -> Result<Vec<u32>> {
        if self.decision.random() >= self.fraction {
            return Ok(generator.generate(input_tokens, &[], 1)?.tokens);
        }
        let prefix_len = ((input_tokens as f64) * self.ratio).round() as usize;
        let prefix_len = prefix_len.min(input_tokens);
        self.grow_shared(generator, prefix_len)?;
        let mut tokens = self.shared[..prefix_len].to_vec();
        let suffix = input_tokens - prefix_len;
        if suffix > 0 {
            tokens.extend_from_slice(&generator.generate(suffix, &[], 1)?.tokens);
        }
        Ok(tokens)
    }

    /// Build one exact-length raw-token prompt under the same shared-prefix
    /// selection policy, never decoding the sampled ids to text.
    fn prompt_token_ids(
        &mut self,
        generator: &mut dyn PromptGenerator,
        input_tokens: usize,
    ) -> Result<Vec<u32>> {
        if self.decision.random() >= self.fraction {
            return generator.generate_token_ids(input_tokens, &[], 1);
        }
        let prefix_len = ((input_tokens as f64) * self.ratio).round() as usize;
        let prefix_len = prefix_len.min(input_tokens);
        self.grow_shared_ids(generator, prefix_len)?;
        let mut tokens = self.shared[..prefix_len].to_vec();
        let suffix = input_tokens - prefix_len;
        if suffix > 0 {
            tokens.extend_from_slice(&generator.generate_token_ids(suffix, &[], 1)?);
        }
        Ok(tokens)
    }

    /// Grow the reusable run up to at least `needed` tokens, leaving the ids
    /// already materialized untouched so every reusing prompt stays aligned.
    fn grow_shared(&mut self, generator: &mut dyn PromptGenerator, needed: usize) -> Result<()> {
        while self.shared.len() < needed {
            let delta = needed - self.shared.len();
            self.shared
                .extend_from_slice(&generator.generate(delta, &[], 1)?.tokens);
        }
        Ok(())
    }

    /// Grow the reusable raw-token prefix without passing through a text decode.
    fn grow_shared_ids(
        &mut self,
        generator: &mut dyn PromptGenerator,
        needed: usize,
    ) -> Result<()> {
        while self.shared.len() < needed {
            let delta = needed - self.shared.len();
            self.shared
                .extend_from_slice(&generator.generate_token_ids(delta, &[], 1)?);
        }
        Ok(())
    }
}

fn generated_context(
    generator: &mut dyn PromptGenerator,
    tokens: usize,
    hash: i64,
) -> Result<String> {
    Ok(generator.generate(tokens, &[hash], tokens)?.text)
}

fn intern_text(
    segments: &mut SegmentPool,
    parent: Option<Handle>,
    role: &str,
    text: &str,
    tokenizer: &dyn TextTokenizer,
) -> Result<Handle> {
    segments.intern_text(
        parent,
        role,
        Bytes::copy_from_slice(text.as_bytes()),
        tokenizer.encode(text)?.into_boxed_slice(),
    )
}

fn append_media_batch(
    turn: &mut Turn,
    parent: &mut Option<Handle>,
    segments: &mut SegmentPool,
    generator: &mut dyn SyntheticMediaGenerator,
    batch_size: usize,
    name: &str,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut handles = SmallVec::new();
    let mut audio_duration = 0.0;
    let mut kind = None;
    for _ in 0..batch_size {
        let generated = generator.generate()?;
        kind.get_or_insert(generated.kind);
        if kind != Some(generated.kind) {
            return Err(DatasetError::Validation(
                "one synthetic media batch produced mixed media kinds".into(),
            ));
        }
        if generated.kind == MediaKind::Audio {
            audio_duration += generated.duration_seconds.unwrap_or(0.0);
        }
        let handle = segments.intern_media(*parent, generated.kind, generated.wire)?;
        *parent = Some(handle);
        handles.push(handle);
    }
    let kind = kind.expect("positive batch size generates at least one value");
    if kind == MediaKind::Audio {
        turn.audio_duration_seconds = Some(audio_duration);
    }
    turn.content.push(ContentGroup {
        kind,
        name: name.into(),
        handles,
        uuids: SmallVec::new(),
    });
    Ok(())
}

fn component_rng(root: RngRoot, namespace: &str) -> ConfiguredRandomGenerator {
    root.derive_generator(namespace)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::rng::SamplingDistribution;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::LoaderRegistry;
    use crate::dataset::prompt::{GeneratedPrompt, PromptGenerator, PromptGeneratorFactory};
    use crate::dataset::segment::Payload;
    use crate::dataset::tokenizer::{TextTokenizer, TiktokenTokenizer};

    struct StaticPromptGenerator {
        prompt: GeneratedPrompt,
    }

    impl PromptGenerator for StaticPromptGenerator {
        fn generate_token_ids(
            &mut self,
            num_tokens: usize,
            _hash_ids: &[i64],
            _block_size: usize,
        ) -> Result<Vec<u32>> {
            if num_tokens != self.prompt.tokens.len() {
                return Err(DatasetError::Validation(format!(
                    "expected {} tokens, got {num_tokens}",
                    self.prompt.tokens.len()
                )));
            }
            Ok(self.prompt.tokens.clone())
        }

        fn generate(
            &mut self,
            num_tokens: usize,
            _hash_ids: &[i64],
            _block_size: usize,
        ) -> Result<GeneratedPrompt> {
            if num_tokens != self.prompt.tokens.len() {
                return Err(DatasetError::Validation(format!(
                    "expected {} tokens, got {num_tokens}",
                    self.prompt.tokens.len()
                )));
            }
            Ok(self.prompt.clone())
        }
    }

    struct StaticPromptGeneratorFactory {
        prompt: GeneratedPrompt,
    }

    impl PromptGeneratorFactory for StaticPromptGeneratorFactory {
        fn create<'a>(
            &self,
            _tokenizer: &'a dyn TextTokenizer,
            _root: RngRoot,
        ) -> Result<Box<dyn PromptGenerator + 'a>> {
            Ok(Box::new(StaticPromptGenerator {
                prompt: self.prompt.clone(),
            }))
        }
    }

    struct RejectGeneratedTextTokenizer {
        inner: TiktokenTokenizer,
    }

    impl RejectGeneratedTextTokenizer {
        fn new() -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
            }
        }
    }

    impl TextTokenizer for RejectGeneratedTextTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            if text == "generated prompt" {
                return Err(DatasetError::Tokenizer(
                    "unexpected synthetic prompt re-encode".into(),
                ));
            }
            self.inner.encode(text)
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String> {
            self.inner.decode(token_ids)
        }

        fn bos_token_id(&self) -> Option<u32> {
            self.inner.bos_token_id()
        }

        fn eos_token_id(&self) -> Option<u32> {
            self.inner.eos_token_id()
        }

        fn vocab_size(&self) -> Option<u32> {
            self.inner.vocab_size()
        }

        fn name(&self) -> &str {
            self.inner.name()
        }
    }

    #[test]
    fn prefix_reuse_shares_exact_token_ids_and_holds_length() {
        use crate::dataset::prompt::{CorpusPromptGeneratorFactory, PromptGeneratorFactory};

        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::default();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(19))).unwrap();
        // A unit reuse fraction forces every prompt onto the warm path, so the
        // deterministic shared prefix is exercised directly.
        let mut reuse = PrefixReuse {
            fraction: 1.0,
            ratio: 0.5,
            decision: RngRoot::new(Some(19)).derive_generator("test.reuse"),
            shared: Vec::new(),
        };
        let first = reuse.prompt_tokens(generator.as_mut(), 10).unwrap();
        let second = reuse.prompt_tokens(generator.as_mut(), 10).unwrap();
        assert_eq!(first.len(), 10);
        assert_eq!(second.len(), 10);
        // ratio 0.5 of a length-10 prompt reserves a 5-token shared prefix that is
        // byte-for-byte identical across warm prompts, while the suffix stays unique.
        assert_eq!(&first[..5], &second[..5]);
        assert_eq!(&first[..5], &reuse.shared[..5]);
        assert_ne!(&first[5..], &second[5..]);
    }

    #[test]
    fn prefix_reuse_holds_shared_run_across_differing_input_lengths() {
        use crate::dataset::prompt::{CorpusPromptGeneratorFactory, PromptGeneratorFactory};

        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::default();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(23))).unwrap();
        // A unit fraction steers every prompt onto the reusable run. Each prompt
        // asks for a different input length, so the reusable run must grow once
        // and freeze: the shared portion stays byte-identical even though the
        // reserved prefix length rises with the (larger) target input length.
        let mut reuse = PrefixReuse {
            fraction: 1.0,
            ratio: 0.5,
            decision: RngRoot::new(Some(23)).derive_generator("test.reuse"),
            shared: Vec::new(),
        };
        // ratio 0.5 reserves prefixes of 4, 8, and 12 tokens for these lengths.
        let short = reuse.prompt_tokens(generator.as_mut(), 8).unwrap();
        let medium = reuse.prompt_tokens(generator.as_mut(), 16).unwrap();
        let long = reuse.prompt_tokens(generator.as_mut(), 24).unwrap();
        assert_eq!(short.len(), 8);
        assert_eq!(medium.len(), 16);
        assert_eq!(long.len(), 24);
        // The shortest reserved prefix (4 tokens) is a common leading run of all
        // three prompts even though every prompt targeted a different length.
        assert_eq!(&short[..4], &medium[..4]);
        assert_eq!(&short[..4], &long[..4]);
        // The reusable run only ever grew; a length seen earlier is a strict
        // prefix of one reserved later, never rewritten.
        assert_eq!(&medium[..8], &long[..8]);
        assert_eq!(&short[..4], &reuse.shared[..4]);
    }

    #[tokio::test]
    async fn prefix_reuse_fraction_targets_warm_share_and_keeps_isl_exact() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(41)));
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 30,
            turns: SamplingDistribution::fixed(1.0).unwrap(),
            prompts: Some(SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(12.0).unwrap(),
                input_token_subtraction: 0,
                batch_size: 1,
                prefix_reuse_fraction: 0.5,
                prefix_reuse_ratio: 0.5,
            }),
            ..SyntheticDatasetConfig::default()
        });
        let tokenizer = TiktokenTokenizer::builtin();
        let dataset = registry
            .build_dataset(Some("synthetic"), &load, &compose, &tokenizer)
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 30);

        // The shared prefix is the first six tokens (ratio 0.5 of ISL 12). Warm
        // prompts collapse onto one identical prefix; cold prompts almost surely
        // differ, so the dominant prefix group counts the warm fraction.
        let mut counts: std::collections::HashMap<Vec<u32>, usize> =
            std::collections::HashMap::new();
        for conversation in dataset.conversations() {
            let handle = conversation.turns[0].content[0].handles[0];
            let Payload::Text {
                bytes, token_count, ..
            } = dataset.segments().get(handle).unwrap()
            else {
                panic!("synthetic prompt must be text");
            };
            // Token-native assembly keeps the input length exactly on target.
            assert_eq!(*token_count, 12);
            let prefix: Vec<u32> = tokenizer
                .encode(std::str::from_utf8(bytes).unwrap())
                .unwrap()[..6]
                .to_vec();
            *counts.entry(prefix).or_default() += 1;
        }
        let warm = counts.values().copied().max().unwrap();
        assert!(
            (9..=21).contains(&warm),
            "warm share {warm} should be near half of 30"
        );
    }

    #[tokio::test]
    async fn verbatim_system_prompt_is_additive_to_synthetic_user_isl() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(41)));
        compose.verbatim_system_prompt = Some("exact system text".into());
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            turns: SamplingDistribution::fixed(1.0).unwrap(),
            prompts: Some(SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(12.0).unwrap(),
                batch_size: 1,
                ..SyntheticPromptConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });
        let tokenizer = TiktokenTokenizer::builtin();

        let dataset = registry
            .build_dataset(Some("synthetic"), &load, &compose, &tokenizer)
            .await
            .unwrap();

        let conversation = &dataset.conversations()[0];
        let system = conversation.system.unwrap();
        let Payload::Text { bytes, .. } = dataset.segments().get(system).unwrap() else {
            panic!("system prompt must be text");
        };
        assert_eq!(bytes, "exact system text");
        assert_eq!(conversation.turns[0].input_tokens, Some(12));
    }

    #[tokio::test]
    async fn prefix_reuse_default_leaves_prompts_unique() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(7)));
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 8,
            turns: SamplingDistribution::fixed(1.0).unwrap(),
            prompts: Some(SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(12.0).unwrap(),
                batch_size: 1,
                ..SyntheticPromptConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });
        let tokenizer = TiktokenTokenizer::builtin();
        let dataset = registry
            .build_dataset(Some("synthetic"), &load, &compose, &tokenizer)
            .await
            .unwrap();
        let mut prefixes = std::collections::HashSet::new();
        for conversation in dataset.conversations() {
            let handle = conversation.turns[0].content[0].handles[0];
            let Payload::Text { bytes, .. } = dataset.segments().get(handle).unwrap() else {
                panic!("synthetic prompt must be text");
            };
            let head: Vec<u32> = tokenizer
                .encode(std::str::from_utf8(bytes).unwrap())
                .unwrap()[..6]
                .to_vec();
            prefixes.insert(head);
        }
        // With the default fraction 0.0 no shared prefix is drawn, so leading
        // token runs stay distinct across prompts.
        assert_eq!(prefixes.len(), 8);
    }

    #[tokio::test]
    async fn synthetic_prompts_reuse_generated_tokens_without_reencoding_text() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(29)));
        compose.prompt_generator = Arc::new(StaticPromptGeneratorFactory {
            prompt: GeneratedPrompt {
                text: "generated prompt".into(),
                tokens: vec![11, 12, 13],
            },
        });
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            turns: SamplingDistribution::fixed(1.0).unwrap(),
            prompts: Some(SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(3.0).unwrap(),
                batch_size: 1,
                ..SyntheticPromptConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });

        let dataset = registry
            .build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                &RejectGeneratedTextTokenizer::new(),
            )
            .await
            .expect("dataset should not re-encode generated prompt text");
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.input_tokens, Some(3));
        let handle = turn.content[0].handles[0];
        let Payload::Text {
            bytes, token_count, ..
        } = dataset.segments().get(handle).unwrap()
        else {
            panic!("synthetic prompt must be text");
        };
        assert_eq!(std::str::from_utf8(bytes).unwrap(), "generated prompt");
        assert_eq!(*token_count, 3);
    }

    #[tokio::test]
    async fn full_synthetic_pipeline_generates_multiturn_multimodal_context() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let source = DatasetSource::Inline(json!({"__aiperf_synthetic": true}));
        let load = LoadConfig::new(source);
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(11)));
        compose.output_length_distribution = Some(SamplingDistribution::fixed(9.0).unwrap());
        compose.max_output_tokens = Some(7);
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 2,
            turns: SamplingDistribution::fixed(2.0).unwrap(),
            turn_delay_ms: SamplingDistribution::fixed(5.0).unwrap(),
            prefixes: SyntheticPrefixConfig {
                shared_system_tokens: Some(4),
                user_context_tokens: Some(3),
                ..SyntheticPrefixConfig::default()
            },
            images: Some(crate::dataset::generator::SyntheticImageConfig {
                batch_size: 1,
                width: SamplingDistribution::fixed(4.0).unwrap(),
                height: SamplingDistribution::fixed(3.0).unwrap(),
                format: crate::dataset::generator::SyntheticImageFormat::Png,
                ..crate::dataset::generator::SyntheticImageConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });
        let dataset = registry
            .build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 2);
        for conversation in dataset.conversations() {
            assert!(conversation.system.is_some());
            assert!(conversation.user_context.is_some());
            assert_eq!(conversation.turns.len(), 2);
            assert_eq!(conversation.turns[0].max_tokens, Some(7));
            assert_eq!(conversation.turns[1].delay_ms, Some(5.0));
            assert_eq!(conversation.turns[0].content.len(), 2);
            let image = conversation.turns[0].content[1].handles[0];
            assert!(matches!(
                dataset.segments().get(image).unwrap(),
                Payload::Media {
                    kind: MediaKind::Image,
                    ..
                }
            ));
        }
    }

    #[tokio::test]
    async fn rankings_generate_exact_query_and_passage_lengths() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(
            json!({"__aiperf_synthetic_rankings": true}),
        ));
        let mut compose = ComposeConfig::new("reranker", RngRoot::new(Some(3)));
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            rankings: Some(crate::dataset::generator::SyntheticRankingsConfig {
                passages: SamplingDistribution::fixed(3.0).unwrap(),
                passage_tokens: SamplingDistribution::fixed(5.0).unwrap(),
                query_tokens: SamplingDistribution::fixed(4.0).unwrap(),
            }),
            ..SyntheticDatasetConfig::default()
        });
        let dataset = registry
            .build_dataset(
                Some("synthetic_rankings"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.content[0].handles.len(), 1);
        assert_eq!(turn.content[1].handles.len(), 3);
        assert_eq!(turn.input_tokens, Some(19));
    }

    #[tokio::test]
    async fn reusable_prefix_is_a_shared_parent_of_the_first_turn_only() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(17)));
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            turns: SamplingDistribution::fixed(2.0).unwrap(),
            prompts: Some(crate::dataset::generator::SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(6.0).unwrap(),
                batch_size: 1,
                ..crate::dataset::generator::SyntheticPromptConfig::default()
            }),
            prefixes: SyntheticPrefixConfig {
                pool_size: Some(1),
                prefix_tokens: Some(4),
                ..SyntheticPrefixConfig::default()
            },
            ..SyntheticDatasetConfig::default()
        });
        let dataset = registry
            .build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let conversation = &dataset.conversations()[0];
        let first = conversation.turns[0].content[0].handles[0];
        let second = conversation.turns[1].content[0].handles[0];
        let prefix = dataset.segments().segment(first).unwrap().parent.unwrap();
        assert_eq!(
            dataset.segments().segment(second).unwrap().parent,
            Some(first)
        );
        let Payload::Text {
            bytes: prefix_bytes,
            ..
        } = dataset.segments().get(prefix).unwrap()
        else {
            panic!("synthetic prefix parent must be text");
        };
        let Payload::Text {
            bytes: first_bytes, ..
        } = dataset.segments().get(first).unwrap()
        else {
            panic!("first synthetic prompt must be text");
        };
        assert!(first_bytes.starts_with(prefix_bytes));
    }

    #[tokio::test]
    async fn independently_sampled_isl_subtracts_server_special_tokens() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(17)));
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            prompts: Some(SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(8.0).unwrap(),
                input_token_subtraction: 2,
                batch_size: 1,
                ..SyntheticPromptConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });
        let dataset = registry
            .build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations()[0].turns[0].input_tokens, Some(6));
    }

    #[tokio::test]
    async fn raw_token_synthetic_composition_never_decodes_text() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(23)));
        compose.requires_raw_token_ids = true;
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            turns: SamplingDistribution::fixed(2.0).unwrap(),
            prompts: Some(crate::dataset::generator::SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(8.0).unwrap(),
                batch_size: 1,
                ..crate::dataset::generator::SyntheticPromptConfig::default()
            }),
            ..SyntheticDatasetConfig::default()
        });

        let dataset = registry
            .build_dataset(Some("synthetic"), &load, &compose, &NoDecodeTokenizer)
            .await
            .unwrap();
        let conversation = &dataset.conversations()[0];
        assert_eq!(conversation.turns.len(), 2);
        assert_eq!(
            conversation.context_mode,
            Some(ConversationContextMode::MessageArrayWithResponses)
        );
        let turn = &conversation.turns[0];
        assert!(turn.content.is_empty());
        assert_eq!(turn.input_tokens, Some(8));
        let handle = *turn.body.first().expect("raw token handle");
        let Payload::TokenIds { token_ids } = dataset.segments().get(handle).unwrap() else {
            panic!("raw-token synthetic prompt must be stored as token IDs");
        };
        assert_eq!(token_ids.len(), 8);
        assert!(!token_ids.contains(&9));
    }

    #[tokio::test]
    async fn raw_token_prefix_pool_is_composed_without_decoding() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(23)));
        compose.requires_raw_token_ids = true;
        compose.prompt_generator = Arc::new(crate::dataset::CorpusPromptGeneratorFactory::random());
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 1,
            prompts: Some(crate::dataset::generator::SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(8.0).unwrap(),
                batch_size: 1,
                ..crate::dataset::generator::SyntheticPromptConfig::default()
            }),
            prefixes: SyntheticPrefixConfig {
                pool_size: Some(1),
                prefix_tokens: Some(4),
                ..SyntheticPrefixConfig::default()
            },
            ..SyntheticDatasetConfig::default()
        });

        let dataset = registry
            .build_dataset(Some("synthetic"), &load, &compose, &NoDecodeTokenizer)
            .await
            .unwrap();
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.input_tokens, Some(12));
        let handle = *turn.body.first().expect("raw token handle");
        let Payload::TokenIds { token_ids } = dataset.segments().get(handle).unwrap() else {
            panic!("raw-token synthetic prompt must be stored as token IDs");
        };
        assert_eq!(token_ids.len(), 12);
    }

    #[tokio::test]
    async fn raw_token_prefix_reuse_shares_leading_token_run() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let load = LoadConfig::new(DatasetSource::Inline(json!({"__aiperf_synthetic": true})));
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(37)));
        compose.requires_raw_token_ids = true;
        compose.prompt_generator = Arc::new(crate::dataset::CorpusPromptGeneratorFactory::random());
        compose.synthetic_config = Some(SyntheticDatasetConfig {
            entries: 6,
            turns: SamplingDistribution::fixed(1.0).unwrap(),
            prompts: Some(crate::dataset::generator::SyntheticPromptConfig {
                input_tokens: SamplingDistribution::fixed(12.0).unwrap(),
                input_token_subtraction: 0,
                batch_size: 1,
                prefix_reuse_fraction: 1.0,
                prefix_reuse_ratio: 0.5,
            }),
            ..SyntheticDatasetConfig::default()
        });

        let dataset = registry
            .build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 6);

        let expected_prefix_len = 6;
        let first = &dataset.conversations()[0].turns[0];
        let first_handle = *first.body.first().expect("raw token handle");
        let Payload::TokenIds {
            token_ids: first_tokens,
        } = dataset.segments().get(first_handle).unwrap()
        else {
            panic!("raw-token synthetic prompt must be stored as token IDs");
        };
        assert_eq!(first_tokens.len(), 12);

        for conversation in &dataset.conversations()[1..] {
            let turn = &conversation.turns[0];
            let handle = *turn.body.first().expect("raw token handle");
            let Payload::TokenIds { token_ids } = dataset.segments().get(handle).unwrap() else {
                panic!("raw-token synthetic prompt must be stored as token IDs");
            };
            assert_eq!(token_ids.len(), 12);
            assert_eq!(
                &token_ids[..expected_prefix_len],
                &first_tokens[..expected_prefix_len]
            );
        }
    }
}
