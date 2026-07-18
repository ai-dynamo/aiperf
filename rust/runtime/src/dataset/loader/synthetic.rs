// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic conversation and rankings formats.
//!
//! Prompt generation, paired ISL/OSL sampling,
//! reusable prefixes, per-session context, multimodal batches, turn delays, and
//! final model/max-token selection are all resolved before the pool is frozen.
//! Token-native composition uses a no-decode branch: exact IDs enter the segment
//! arena directly and no temporary text payload is constructed.

use crate::rng::{RandomGenerator, RngRoot};
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
use crate::dataset::prompt::PromptGenerator;
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
        let mut prefix_rng = RandomGenerator::from_seed(
            config
                .rng_root
                .derive_seed("dataset.prompt.prefix.selection"),
        );
        let prefix_pool = if shape.prefixes.pool_size.is_some() {
            let generator = prompt_generator.as_mut().ok_or_else(|| {
                DatasetError::Validation("synthetic prefixes require a tokenizer".into())
            })?;
            build_prefix_pool(&shape.prefixes, generator.as_mut())?
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

                let paired_lengths = config
                    .sequence_length_distribution
                    .as_ref()
                    .map(|distribution| distribution.sample(&mut length_rng))
                    .transpose()
                    .map_err(|error| DatasetError::Validation(error.to_string()))?;
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
                        None => usize::try_from(
                            prompt
                                .input_tokens
                                .sample_int(&mut length_rng)
                                .map_err(|error| DatasetError::Validation(error.to_string()))?,
                        )
                        .map_err(|_| {
                            DatasetError::Validation("sampled ISL exceeds usize".into())
                        })?,
                    };
                    let generator = prompt_generator.as_deref_mut().ok_or_else(|| {
                        DatasetError::Validation("synthetic prompts require a tokenizer".into())
                    })?;
                    if config.requires_raw_token_ids {
                        let token_ids = generator.generate_token_ids(input_tokens, &[], 1)?;
                        turn.input_tokens = token_ids.len() as u64;
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
                            turn.input_tokens =
                                turn.input_tokens.checked_add(tokens.len() as u64).ok_or_else(
                                    || DatasetError::Validation("input token count overflow".into()),
                                )?;
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
                        });
                    } else {
                        let mut handles = SmallVec::new();
                        for _ in 0..prompt.batch_size {
                            let generated = generator.generate(input_tokens, &[], 1)?;
                            let selected_prefix = (turn_index == 0 && !prefix_pool.is_empty())
                                .then(|| prefix_rng.choice(&prefix_pool).ok().cloned())
                                .flatten();
                            let text = selected_prefix.as_ref().map_or_else(
                                || generated.text.clone(),
                                |prefix| format!("{prefix} {}", generated.text),
                            );
                            let tokens = tokenizer.encode(&text)?;
                            turn.input_tokens = turn
                                .input_tokens
                                .checked_add(tokens.len() as u64)
                                .ok_or_else(|| {
                                    DatasetError::Validation("input token count overflow".into())
                                })?;
                            // The full authored text remains one endpoint value, while
                            // the hidden prefix parent makes reuse visible in the
                            // content-addressed chain without changing wire shape.
                            let content_parent = if let Some(prefix) = selected_prefix {
                                Some(segments.intern_text(
                                    parent,
                                    "user",
                                    Bytes::from(prefix.clone()),
                                    tokenizer.encode(&prefix)?.into_boxed_slice(),
                                )?)
                            } else {
                                parent
                            };
                            let handle = segments.intern_text(
                                content_parent,
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
                input_tokens,
                content: smallvec::smallvec![
                    ContentGroup {
                        kind: MediaKind::Text,
                        name: "query".into(),
                        handles: smallvec::smallvec![query_handle],
                    },
                    ContentGroup {
                        kind: MediaKind::Text,
                        name: "passages".into(),
                        handles: passage_handles,
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
    if prompt.prefix_reuse_fraction > 0.0 {
        return Err(DatasetError::Validation(
            "raw-token synthetic datasets do not support prompt prefix reuse".into(),
        ));
    }
    if has_prefixes(&shape.prefixes)
        || config.shared_system_prompt.is_some()
        || !config.user_context_prompts.is_empty()
    {
        return Err(DatasetError::Validation(
            "raw-token synthetic datasets do not support prefix, system, or user-context text"
                .into(),
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
    if !(prompt.prefix_reuse_fraction.is_finite() && (0.0..=1.0).contains(&prompt.prefix_reuse_fraction))
    {
        return Err(DatasetError::Validation(
            "synthetic prompt prefix_reuse_fraction must be within [0, 1]".into(),
        ));
    }
    if !(prompt.prefix_reuse_ratio.is_finite() && (0.0..=1.0).contains(&prompt.prefix_reuse_ratio)) {
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
) -> Result<Vec<String>> {
    let (Some(size), Some(tokens)) = (prefixes.pool_size, prefixes.prefix_tokens) else {
        return Ok(Vec::new());
    };
    (0..size)
        .map(|index| {
            let hash = i64::try_from(index)
                .map_err(|_| DatasetError::Validation("prefix pool exceeds i64".into()))?;
            Ok(generator.generate(tokens, &[hash], tokens)?.text)
        })
        .collect()
}

/// Deterministic bimodal prefix reuse layered on the corpus block generator.
///
/// A configured fraction of prompts draw an identical leading token run — the
/// shared prefix — so a server KV cache observes genuine prefix hits, while the
/// remaining prompts stay fully unique. The shared prefix is grown on demand and
/// never rewritten, so every reusing prompt shares a byte-identical leading token
/// sequence regardless of its own sampled input length, and exact-length input
/// targeting is preserved because the prompt is assembled from exact token ids.
struct PrefixReuse {
    /// Probability, in `[0, 1]`, that a prompt draws the shared prefix.
    fraction: f64,
    /// Fraction of a reusing prompt's input length occupied by the shared prefix.
    ratio: f64,
    /// Seeded warm/cold selector, independent of the corpus sampling stream.
    decision: RandomGenerator,
    /// The shared prefix token ids, frozen once grown so warm prompts match.
    shared: Vec<u32>,
}

impl PrefixReuse {
    /// Prepare reuse state when the prompt requests a non-zero reuse fraction.
    fn from_prompt(prompt: &SyntheticPromptConfig, root: RngRoot) -> Option<Self> {
        (prompt.prefix_reuse_fraction > 0.0).then(|| Self {
            fraction: prompt.prefix_reuse_fraction,
            ratio: prompt.prefix_reuse_ratio,
            decision: RandomGenerator::from_seed(root.derive_seed("dataset.prompt.prefix.reuse")),
            shared: Vec::new(),
        })
    }

    /// Assemble one exact-length prompt, prepending the shared prefix for the
    /// deterministically selected warm fraction of prompts.
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

    /// Extend the shared prefix to at least `needed` tokens without disturbing the
    /// tokens already committed, keeping every warm prefix byte-identical.
    fn grow_shared(&mut self, generator: &mut dyn PromptGenerator, needed: usize) -> Result<()> {
        while self.shared.len() < needed {
            let delta = needed - self.shared.len();
            self.shared
                .extend_from_slice(&generator.generate(delta, &[], 1)?.tokens);
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
    });
    Ok(())
}

fn component_rng(root: RngRoot, namespace: &str) -> RandomGenerator {
    RandomGenerator::from_seed(root.derive_seed(namespace))
}

#[cfg(test)]
mod tests {
    use crate::rng::SamplingDistribution;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::LoaderRegistry;
    use crate::dataset::segment::Payload;
    use crate::dataset::tokenizer::TiktokenTokenizer;

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
        assert_eq!(turn.input_tokens, 19);
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
        assert_eq!(turn.input_tokens, 8);
        let handle = *turn.body.first().expect("raw token handle");
        let Payload::TokenIds { token_ids } = dataset.segments().get(handle).unwrap() else {
            panic!("raw-token synthetic prompt must be stored as token IDs");
        };
        assert_eq!(token_ids.len(), 8);
        assert!(!token_ids.contains(&9));
    }
}
