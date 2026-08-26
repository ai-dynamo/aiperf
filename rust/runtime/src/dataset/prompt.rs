// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-length synthetic prompt generation for trace and synthetic datasets.
//!
//! The block-cache algorithm works as follows: each hash id owns one token
//! block, the final block may be shorter, a BOS/EOS separator occupies the first
//! token when available, and sampled corpus tokens wrap at the corpus boundary.
//! The no-decode token path: raw-token endpoints receive the sampled IDs
//! directly, with EOS replaced before engine admission.

use std::collections::HashMap;
use std::sync::Arc;

use crate::rng::namespace::DATASET_PROMPT_CORPUS;
use crate::rng::{ConfiguredRandomGenerator, RngRoot, RuntimeRandomGenerator};

use crate::dataset::corpus::{SHAKESPEARE_CORPUS, tokenize_corpus_chunked};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::random_range::RandomCorpusStyle;
use crate::dataset::tokenizer::TextTokenizer;

/// Generated text paired with the authoritative token sequence used to build it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedPrompt {
    /// Decoded prompt text.
    pub text: String,
    /// Exact token ids used for content addressing and input accounting.
    pub tokens: Vec<u32>,
}

/// Stateful exact-length prompt generator.
pub trait PromptGenerator {
    /// Generate exact raw token IDs without decoding them to text.
    fn generate_token_ids(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<Vec<u32>>;

    /// Generate `num_tokens`, reusing blocks identified by `hash_ids` when
    /// supplied. `block_size` must frame those identifiers exactly.
    fn generate(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<GeneratedPrompt>;
}

/// Factory seam giving every composition pass independent deterministic state.
pub trait PromptGeneratorFactory: Send + Sync {
    /// Create a generator for one tokenizer and RNG root.
    fn create<'a>(
        &self,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Result<Box<dyn PromptGenerator + 'a>>;
}

/// Text body a [`CorpusPromptGeneratorFactory`] samples prompts from.
///
/// A trait-free enum is deliberate: the corpus choice is authored data, not a
/// separate behavior interface.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CorpusSource {
    /// The embedded Shakespeare ("sonnet") product-default corpus.
    Sonnet,
    /// The seeded procedural coding/tool corpus shared with recorded traces.
    Coding,
    /// Tokenizer-driven synthetic random generation.
    Random,
    /// A caller-supplied corpus body (custom-corpus configs and tests).
    Custom(String),
}

/// Corpus-token generator with prefix block reuse.
///
/// The default Shakespeare corpus is tokenized once through the character-chunked
/// [`tokenize_corpus_chunked`] policy, then sampled with wrap-around and prefix
/// block reuse to produce exact-length synthetic prompts.
#[derive(Debug, Clone)]
pub struct CorpusPromptGeneratorFactory {
    corpus: CorpusSource,
    reference_random: Option<ReferenceRandomPromptConfig>,
}

#[derive(Debug, Clone)]
struct ReferenceRandomPromptConfig {
    style: RandomCorpusStyle,
    offsets: Arc<[usize]>,
}

impl CorpusPromptGeneratorFactory {
    /// Use the embedded Shakespeare ("sonnet") product-default corpus.
    pub fn sonnet() -> Self {
        Self {
            corpus: CorpusSource::Sonnet,
            reference_random: None,
        }
    }

    /// Use the seeded procedural coding/tool corpus.
    pub fn coding() -> Self {
        Self {
            corpus: CorpusSource::Coding,
            reference_random: None,
        }
    }

    /// Use tokenizer-driven synthetic random generation.
    pub fn random() -> Self {
        Self {
            corpus: CorpusSource::Random,
            reference_random: None,
        }
    }

    /// Use a reference style and preseeded conversation offsets.
    pub fn random_reference(style: RandomCorpusStyle, offsets: Arc<[usize]>) -> Self {
        Self {
            corpus: CorpusSource::Random,
            reference_random: Some(ReferenceRandomPromptConfig { style, offsets }),
        }
    }

    /// Use a caller-provided non-empty corpus body.
    ///
    /// The corpus is tokenized through the same character-chunked policy as the
    /// sonnet corpus, so a custom source stays on the identical reproducibility
    /// contract.
    pub fn new(corpus: impl Into<String>) -> Result<Self> {
        let corpus = corpus.into();
        if corpus.trim().is_empty() {
            return Err(DatasetError::Validation(
                "prompt generator corpus cannot be empty".into(),
            ));
        }
        Ok(Self {
            corpus: CorpusSource::Custom(corpus),
            reference_random: None,
        })
    }

    /// Tokenize this corpus once and return a factory that creates generators
    /// without re-tokenizing.
    ///
    /// Default [`PromptGeneratorFactory::create`] still tokenizes on every call
    /// so production callers that never prepare keep today's behavior. Benchmarks
    /// and multi-compose pipelines that share one tokenizer should call this
    /// before the timed or hot region, then inject the prepared factory into
    /// [`crate::dataset::compose::ComposeConfig::prompt_generator`].
    pub fn prepare(
        &self,
        tokenizer: &dyn TextTokenizer,
    ) -> Result<PreparedCorpusPromptGeneratorFactory> {
        Ok(PreparedCorpusPromptGeneratorFactory {
            source: match &self.corpus {
                CorpusSource::Random => PreparedPromptGeneratorSource::Random,
                _ => PreparedPromptGeneratorSource::Corpus(tokenize_corpus_arc(
                    &self.corpus,
                    tokenizer,
                    None,
                )?),
            },
        })
    }
}

impl Default for CorpusPromptGeneratorFactory {
    /// The default corpus is the Shakespeare sonnet text.
    fn default() -> Self {
        Self::sonnet()
    }
}

impl PromptGeneratorFactory for CorpusPromptGeneratorFactory {
    fn create<'a>(
        &self,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Result<Box<dyn PromptGenerator + 'a>> {
        match &self.corpus {
            CorpusSource::Random => Ok(Box::new(RandomPromptGenerator::new(
                tokenizer,
                root,
                self.reference_random.clone(),
            )?)),
            _ => {
                // Tokenize once into Arc and build the generator directly — avoid
                // the prepare→create detour that would Arc::clone a temporary
                // factory.
                Ok(Box::new(CorpusPromptGenerator::from_corpus_tokens(
                    tokenize_corpus_arc(&self.corpus, tokenizer, Some(root))?,
                    tokenizer,
                    root,
                )))
            }
        }
    }
}

/// Prepared corpus tokens shared across many cheap generator creations.
///
/// Constructed by [`CorpusPromptGeneratorFactory::prepare`]. Each
/// [`PromptGeneratorFactory::create`] clones an [`Arc`] of the token stream and
/// builds fresh RNG / block-cache state; the corpus is not re-tokenized.
#[derive(Debug, Clone)]
pub struct PreparedCorpusPromptGeneratorFactory {
    source: PreparedPromptGeneratorSource,
}

#[derive(Debug, Clone)]
enum PreparedPromptGeneratorSource {
    Corpus(Arc<[u32]>),
    Random,
}

impl PromptGeneratorFactory for PreparedCorpusPromptGeneratorFactory {
    fn create<'a>(
        &self,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Result<Box<dyn PromptGenerator + 'a>> {
        match &self.source {
            PreparedPromptGeneratorSource::Corpus(corpus_tokens) => {
                Ok(Box::new(CorpusPromptGenerator::from_corpus_tokens(
                    Arc::clone(corpus_tokens),
                    tokenizer,
                    root,
                )))
            }
            PreparedPromptGeneratorSource::Random => {
                Ok(Box::new(RandomPromptGenerator::new(tokenizer, root, None)?))
            }
        }
    }
}

/// Build or tokenize one corpus into a non-empty shared token stream.
fn tokenize_corpus_arc(
    corpus: &CorpusSource,
    tokenizer: &dyn TextTokenizer,
    root: Option<RngRoot>,
) -> Result<Arc<[u32]>> {
    let corpus_tokens = match corpus {
        CorpusSource::Sonnet => tokenize_corpus_chunked(SHAKESPEARE_CORPUS, tokenizer)?,
        CorpusSource::Coding => {
            let root = root.ok_or_else(|| {
                DatasetError::Validation(
                    "coding corpus preparation requires generator creation with a run root".into(),
                )
            })?;
            crate::dataset::coding::build_coding_corpus(
                tokenizer,
                root.derive_seed_or_entropy("dataset.prompt.coding"),
            )
            .map_err(|error| DatasetError::Validation(error.to_string()))?
        }
        CorpusSource::Random => {
            return Err(DatasetError::Validation(
                "random prompt generation does not use a reusable corpus token stream".into(),
            ));
        }
        CorpusSource::Custom(text) => tokenize_corpus_chunked(text, tokenizer)?,
    };
    if corpus_tokens.is_empty() {
        return Err(DatasetError::Validation(
            "prompt generator corpus encoded to zero tokens".into(),
        ));
    }
    Ok(Arc::from(corpus_tokens))
}

struct CorpusPromptGenerator<'a> {
    corpus_tokens: Arc<[u32]>,
    block_separator: Option<u32>,
    tokenizer: &'a dyn TextTokenizer,
    rng: ConfiguredRandomGenerator,
    blocks: HashMap<i64, Vec<u32>>,
}

impl<'a> CorpusPromptGenerator<'a> {
    fn from_corpus_tokens(
        corpus_tokens: Arc<[u32]>,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Self {
        Self {
            corpus_tokens,
            block_separator: tokenizer.block_separation_token_id(),
            tokenizer,
            rng: root.derive_generator(DATASET_PROMPT_CORPUS),
            blocks: HashMap::new(),
        }
    }
}

impl PromptGenerator for CorpusPromptGenerator<'_> {
    fn generate_token_ids(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = self.build_token_ids(num_tokens, hash_ids, block_size)?;
        if let Some(eos) = self.tokenizer.eos_token_id()
            && tokens.contains(&eos)
        {
            let replacement = match self.tokenizer.vocab_size() {
                Some(vocab_size) if vocab_size > 1 && eos < vocab_size => {
                    Some(eos.wrapping_add(1) % vocab_size)
                }
                Some(vocab_size) => self
                    .corpus_tokens
                    .iter()
                    .copied()
                    .find(|token| *token != eos && *token < vocab_size),
                None => self
                    .corpus_tokens
                    .iter()
                    .copied()
                    .find(|token| *token != eos),
            }
            .ok_or_else(|| {
                DatasetError::Tokenizer(format!(
                    "tokenizer {:?} has no valid non-EOS token for synthetic raw input",
                    self.tokenizer.name()
                ))
            })?;
            for token in &mut tokens {
                if *token == eos {
                    *token = replacement;
                }
            }
        }
        Ok(tokens)
    }

    fn generate(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<GeneratedPrompt> {
        let tokens = self.build_token_ids(num_tokens, hash_ids, block_size)?;
        Ok(GeneratedPrompt {
            text: self.tokenizer.decode(&tokens)?,
            tokens,
        })
    }
}

impl CorpusPromptGenerator<'_> {
    fn build_token_ids(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<Vec<u32>> {
        if num_tokens == 0 {
            return Err(DatasetError::Validation(
                "synthetic prompt length must be greater than zero".into(),
            ));
        }
        let tokens = if hash_ids.is_empty() {
            self.sample_tokens(num_tokens)?
        } else {
            if block_size == 0 {
                return Err(DatasetError::Validation(
                    "hash-id block_size must be greater than zero".into(),
                ));
            }
            let prefix = hash_ids
                .len()
                .saturating_sub(1)
                .checked_mul(block_size)
                .ok_or_else(|| DatasetError::Validation("hash block length overflow".into()))?;
            let final_size = num_tokens
                .checked_sub(prefix)
                .filter(|size| *size <= block_size);
            let final_size = final_size.ok_or_else(|| {
                DatasetError::Validation(format!(
                    "input length {num_tokens}, {} hash ids, and block size {block_size} are incompatible",
                    hash_ids.len()
                ))
            })?;
            let mut tokens = Vec::with_capacity(num_tokens);
            for (index, hash_id) in hash_ids.iter().enumerate() {
                let size = if index + 1 == hash_ids.len() {
                    final_size
                } else {
                    block_size
                };
                if !self.blocks.contains_key(hash_id) {
                    let mut block = Vec::with_capacity(size);
                    if let Some(separator) = self.block_separator {
                        block.push(separator);
                        block.extend(self.sample_tokens(size.saturating_sub(1))?);
                    } else {
                        block.extend(self.sample_tokens(size)?);
                    }
                    self.blocks.insert(*hash_id, block);
                }
                let block = self.blocks.get(hash_id).expect("inserted above");
                if block.len() != size {
                    return Err(DatasetError::Validation(format!(
                        "hash id {hash_id} was first materialized with {} tokens but is now requested with {size}",
                        block.len()
                    )));
                }
                tokens.extend_from_slice(block);
            }
            tokens
        };
        debug_assert_eq!(tokens.len(), num_tokens);
        Ok(tokens)
    }

    fn sample_tokens(&mut self, count: usize) -> Result<Vec<u32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let start =
            self.rng
                .randrange_u64(0, self.corpus_tokens.len() as u64)
                .map_err(|error| DatasetError::Validation(error.to_string()))? as usize;
        Ok((0..count)
            .map(|offset| self.corpus_tokens[(start + offset) % self.corpus_tokens.len()])
            .collect())
    }
}

const RANDOM_TEXT_REPAIR_ATTEMPTS: usize = 16;

#[derive(Clone, Copy)]
enum RandomGenerationMode {
    Raw,
    Text,
}

struct RandomPromptGenerator<'a> {
    tokenizer: &'a dyn TextTokenizer,
    block_separator: Option<u32>,
    rng: ConfiguredRandomGenerator,
    raw_blocks: HashMap<i64, Vec<u32>>,
    text_blocks: HashMap<i64, Vec<u32>>,
    allowed_token_ids: Option<Arc<[u32]>>,
    vocab_size: Option<u32>,
    eos_token_id: Option<u32>,
    replacement_token: u32,
    offsets: Option<Arc<[usize]>>,
    offset_index: usize,
    request_index: usize,
    has_warned_offsets_exhausted: bool,
}

impl<'a> RandomPromptGenerator<'a> {
    fn new(
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
        reference: Option<ReferenceRandomPromptConfig>,
    ) -> Result<Self> {
        let style = reference
            .as_ref()
            .map_or(RandomCorpusStyle::Vllm, |config| config.style);
        let allowed_token_ids = match style {
            RandomCorpusStyle::Vllm => tokenizer.allowed_random_token_ids(),
            RandomCorpusStyle::Sglang => None,
        };
        if allowed_token_ids
            .as_ref()
            .is_some_and(|tokens| tokens.is_empty())
        {
            return Err(DatasetError::Tokenizer(format!(
                "tokenizer {:?} exposes no allowed token ids for synthetic random prompts",
                tokenizer.name()
            )));
        }
        let vocab_size = tokenizer.vocab_size().filter(|size| *size > 0);
        if allowed_token_ids.is_none() && vocab_size.is_none() {
            return Err(DatasetError::Tokenizer(format!(
                "tokenizer {:?} does not expose a usable vocabulary for synthetic random prompts",
                tokenizer.name()
            )));
        }
        let eos_token_id = tokenizer
            .eos_token_id()
            .filter(|eos| vocab_size.is_none_or(|size| *eos < size));
        let replacement_token = allowed_token_ids
            .as_deref()
            .into_iter()
            .flatten()
            .copied()
            .chain(vocab_size.iter().copied().flat_map(|size| 0..size))
            .find(|token| Some(*token) != eos_token_id)
            .ok_or_else(|| {
                DatasetError::Tokenizer(format!(
                    "tokenizer {:?} has no valid non-EOS token for synthetic random input",
                    tokenizer.name()
                ))
            })?;
        Ok(Self {
            tokenizer,
            block_separator: tokenizer.block_separation_token_id(),
            rng: root.derive_generator(DATASET_PROMPT_CORPUS),
            raw_blocks: HashMap::new(),
            text_blocks: HashMap::new(),
            allowed_token_ids,
            vocab_size,
            eos_token_id,
            replacement_token,
            offsets: reference.map(|config| config.offsets),
            offset_index: 0,
            request_index: 0,
            has_warned_offsets_exhausted: false,
        })
    }

    fn build_token_ids(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
        mode: RandomGenerationMode,
    ) -> Result<Vec<u32>> {
        if num_tokens == 0 {
            return Err(DatasetError::Validation(
                "synthetic prompt length must be greater than zero".into(),
            ));
        }
        let tokens = if hash_ids.is_empty() {
            self.materialize_tokens(num_tokens, false, mode)?
        } else {
            if block_size == 0 {
                return Err(DatasetError::Validation(
                    "hash-id block_size must be greater than zero".into(),
                ));
            }
            let prefix = hash_ids
                .len()
                .saturating_sub(1)
                .checked_mul(block_size)
                .ok_or_else(|| DatasetError::Validation("hash block length overflow".into()))?;
            let final_size = num_tokens
                .checked_sub(prefix)
                .filter(|size| *size <= block_size);
            let final_size = final_size.ok_or_else(|| {
                DatasetError::Validation(format!(
                    "input length {num_tokens}, {} hash ids, and block size {block_size} are incompatible",
                    hash_ids.len()
                ))
            })?;
            let mut tokens = Vec::with_capacity(num_tokens);
            for (index, hash_id) in hash_ids.iter().enumerate() {
                let size = if index + 1 == hash_ids.len() {
                    final_size
                } else {
                    block_size
                };
                if !self.has_block(mode, *hash_id) {
                    let block = self.materialize_tokens(size, true, mode)?;
                    self.insert_block(mode, *hash_id, block);
                }
                let block = self
                    .get_block(mode, *hash_id)
                    .expect("block inserted above or already present");
                if block.len() != size {
                    return Err(DatasetError::Validation(format!(
                        "hash id {hash_id} was first materialized with {} tokens but is now requested with {size}",
                        block.len()
                    )));
                }
                tokens.extend_from_slice(block);
            }
            tokens
        };
        debug_assert_eq!(tokens.len(), num_tokens);
        Ok(tokens)
    }

    fn materialize_tokens(
        &mut self,
        count: usize,
        include_separator: bool,
        mode: RandomGenerationMode,
    ) -> Result<Vec<u32>> {
        match mode {
            RandomGenerationMode::Raw => self.sample_raw_candidate(count, include_separator),
            RandomGenerationMode::Text => self.sample_text_exact_tokens(count, include_separator),
        }
    }

    fn sample_raw_candidate(&mut self, count: usize, include_separator: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(count);
        if include_separator
            && count > 0
            && let Some(separator) = self.block_separator
        {
            tokens.push(self.valid_token(separator));
        }
        let remaining = count.saturating_sub(tokens.len());
        tokens.extend(self.sample_raw_tokens(remaining)?);
        Ok(tokens)
    }

    fn sample_text_exact_tokens(
        &mut self,
        count: usize,
        include_separator: bool,
    ) -> Result<Vec<u32>> {
        let candidate = self.sample_raw_candidate(count, include_separator)?;
        self.repair_exact_text_tokens(candidate, count)
    }

    fn repair_exact_text_tokens(
        &mut self,
        mut candidate: Vec<u32>,
        target_len: usize,
    ) -> Result<Vec<u32>> {
        if target_len == 0 {
            return Ok(Vec::new());
        }
        self.replace_eos_in_place(&mut candidate);
        for _ in 0..RANDOM_TEXT_REPAIR_ATTEMPTS {
            let text = self.tokenizer.decode_lossy(&candidate)?;
            let mut encoded = self.tokenizer.encode(&text)?;
            self.replace_eos_in_place(&mut encoded);
            if encoded.len() == target_len {
                return Ok(encoded);
            }
            candidate = encoded;
            if candidate.len() > target_len {
                candidate.truncate(target_len);
            } else {
                let missing = target_len - candidate.len();
                candidate.extend(self.sample_independent_tokens(missing)?);
            }
        }
        Err(DatasetError::Validation(format!(
            "tokenizer {:?} could not repair synthetic random prompt to exact length {target_len}",
            self.tokenizer.name()
        )))
    }

    fn sample_raw_tokens(&mut self, count: usize) -> Result<Vec<u32>> {
        if self.offsets.is_none() {
            return self.sample_independent_tokens(count);
        }
        let pool_len = self.allowed_token_ids.as_ref().map_or_else(
            || usize::try_from(self.vocab_size.unwrap_or(0)).unwrap_or(0),
            |tokens| tokens.len(),
        );
        if pool_len == 0 {
            return Err(DatasetError::Validation(
                "random prompt token pool cannot be empty".into(),
            ));
        }
        let cached_offset = self
            .offsets
            .as_ref()
            .and_then(|offsets| offsets.get(self.offset_index))
            .copied();
        if cached_offset.is_none() && !self.has_warned_offsets_exhausted {
            self.has_warned_offsets_exhausted = true;
            tracing::warn!(
                cached_offsets = self.offsets.as_ref().map_or(0, |offsets| offsets.len()),
                "preseeded random prompt offsets exhausted; subsequent prompts no longer match the reference stream"
            );
        }
        let offset = cached_offset
            .map_or_else(
                || {
                    self.rng
                        .randrange_u64(0, self.vocab_size.unwrap_or(pool_len as u32) as u64)
                        .map(|value| value as usize)
                },
                Ok,
            )
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        self.offset_index = self.offset_index.saturating_add(1);
        let request_index = self.request_index;
        self.request_index = self.request_index.saturating_add(1);
        Ok((0..count)
            .map(|index| {
                let pool_index = ((offset as u128 + request_index as u128 + index as u128)
                    % pool_len as u128) as usize;
                self.allowed_token_ids
                    .as_ref()
                    .map_or(pool_index as u32, |tokens| tokens[pool_index])
            })
            .collect())
    }

    fn sample_independent_tokens(&mut self, count: usize) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(count);
        for _ in 0..count {
            let sampled = if let Some(allowed) = &self.allowed_token_ids {
                let index = self
                    .rng
                    .randrange_u64(0, allowed.len() as u64)
                    .map_err(|error| DatasetError::Validation(error.to_string()))?
                    as usize;
                allowed[index]
            } else {
                self.rng
                    .randrange_u64(0, u64::from(self.vocab_size.expect("validated above")))
                    .map_err(|error| DatasetError::Validation(error.to_string()))?
                    as u32
            };
            tokens.push(self.valid_token(sampled));
        }
        Ok(tokens)
    }

    fn replace_eos_in_place(&self, tokens: &mut [u32]) {
        if let Some(eos) = self.eos_token_id {
            for token in tokens {
                if *token == eos {
                    *token = self.replacement_token;
                }
            }
        }
    }

    fn valid_token(&self, token: u32) -> u32 {
        if Some(token) == self.eos_token_id {
            self.replacement_token
        } else {
            token
        }
    }

    fn has_block(&self, mode: RandomGenerationMode, hash_id: i64) -> bool {
        match mode {
            RandomGenerationMode::Raw => self.raw_blocks.contains_key(&hash_id),
            RandomGenerationMode::Text => self.text_blocks.contains_key(&hash_id),
        }
    }

    fn insert_block(&mut self, mode: RandomGenerationMode, hash_id: i64, block: Vec<u32>) {
        match mode {
            RandomGenerationMode::Raw => {
                self.raw_blocks.insert(hash_id, block);
            }
            RandomGenerationMode::Text => {
                self.text_blocks.insert(hash_id, block);
            }
        }
    }

    fn get_block(&self, mode: RandomGenerationMode, hash_id: i64) -> Option<&[u32]> {
        match mode {
            RandomGenerationMode::Raw => self.raw_blocks.get(&hash_id).map(Vec::as_slice),
            RandomGenerationMode::Text => self.text_blocks.get(&hash_id).map(Vec::as_slice),
        }
    }
}

impl PromptGenerator for RandomPromptGenerator<'_> {
    fn generate_token_ids(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<Vec<u32>> {
        self.build_token_ids(num_tokens, hash_ids, block_size, RandomGenerationMode::Raw)
    }

    fn generate(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<GeneratedPrompt> {
        let tokens =
            self.build_token_ids(num_tokens, hash_ids, block_size, RandomGenerationMode::Text)?;
        Ok(GeneratedPrompt {
            text: self.tokenizer.decode(&tokens)?,
            tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::rng::RngRoot;

    use super::*;
    use crate::dataset::tokenizer::TiktokenTokenizer;

    /// Counts `encode` calls so tests can prove corpus tokenization happens in
    /// [`CorpusPromptGeneratorFactory::prepare`] rather than every `create`.
    struct CountingTokenizer {
        inner: TiktokenTokenizer,
        encodes: AtomicUsize,
    }

    impl CountingTokenizer {
        fn new() -> Self {
            Self {
                inner: TiktokenTokenizer::builtin(),
                encodes: AtomicUsize::new(0),
            }
        }

        fn encode_count(&self) -> usize {
            self.encodes.load(Ordering::SeqCst)
        }
    }

    impl TextTokenizer for CountingTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            self.encodes.fetch_add(1, Ordering::SeqCst);
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
            "counting"
        }
    }

    struct AllowedOnlyTokenizer;

    impl TextTokenizer for AllowedOnlyTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            if text.chars().all(|c| c == 'a') {
                Ok(vec![1; text.chars().count()])
            } else {
                Err(DatasetError::Tokenizer(format!(
                    "unexpected text for allowed-only tokenizer: {text:?}"
                )))
            }
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String> {
            if token_ids.iter().all(|id| *id == 1) {
                Ok("a".repeat(token_ids.len()))
            } else {
                Err(DatasetError::Tokenizer(format!(
                    "disallowed token ids in decode: {token_ids:?}"
                )))
            }
        }

        fn decode_lossy(&self, token_ids: &[u32]) -> Result<String> {
            self.decode(token_ids)
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(9)
        }

        fn vocab_size(&self) -> Option<u32> {
            Some(10)
        }

        fn allowed_random_token_ids(&self) -> Option<Arc<[u32]>> {
            Some(Arc::from(vec![1_u32]))
        }

        fn name(&self) -> &str {
            "allowed-only"
        }
    }

    #[test]
    fn prepared_factory_create_does_not_reencode_corpus() {
        let tokenizer = CountingTokenizer::new();
        let factory = CorpusPromptGeneratorFactory::new("alpha beta gamma delta epsilon").unwrap();
        let prepared = factory.prepare(&tokenizer).unwrap();
        let encodes_after_prepare = tokenizer.encode_count();
        assert!(
            encodes_after_prepare > 0,
            "prepare must tokenize the corpus"
        );

        let mut generator = prepared.create(&tokenizer, RngRoot::new(Some(7))).unwrap();
        assert_eq!(
            tokenizer.encode_count(),
            encodes_after_prepare,
            "create must not re-tokenize a prepared corpus"
        );
        let prompt = generator.generate(4, &[], 1).unwrap();
        assert_eq!(prompt.tokens.len(), 4);
        assert_eq!(
            tokenizer.encode_count(),
            encodes_after_prepare,
            "generate must sample prepared tokens without corpus encode"
        );
    }

    #[test]
    fn prepared_and_cold_factories_agree_on_generated_prompts() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::new("alpha beta gamma delta epsilon").unwrap();
        let prepared = factory.prepare(&tokenizer).unwrap();

        let cold = factory
            .create(&tokenizer, RngRoot::new(Some(11)))
            .unwrap()
            .generate(8, &[1, 2], 4)
            .unwrap();
        let warm = prepared
            .create(&tokenizer, RngRoot::new(Some(11)))
            .unwrap()
            .generate(8, &[1, 2], 4)
            .unwrap();
        assert_eq!(cold, warm);
    }

    #[test]
    fn prepare_rejects_empty_token_stream() {
        struct EmptyTokenizer;
        impl TextTokenizer for EmptyTokenizer {
            fn encode(&self, _text: &str) -> Result<Vec<u32>> {
                Ok(Vec::new())
            }
            fn decode(&self, _token_ids: &[u32]) -> Result<String> {
                Ok(String::new())
            }
            fn bos_token_id(&self) -> Option<u32> {
                None
            }
            fn eos_token_id(&self) -> Option<u32> {
                None
            }
            fn name(&self) -> &str {
                "empty"
            }
        }

        let factory = CorpusPromptGeneratorFactory::new("fixture").unwrap();
        assert!(factory.prepare(&EmptyTokenizer).is_err());
    }

    #[test]
    fn exact_lengths_reuse_hash_prefixes() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::default();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(3))).unwrap();
        let first = generator.generate(7, &[10, 11], 4).unwrap();
        let second = generator.generate(8, &[10, 12], 4).unwrap();
        assert_eq!(first.tokens.len(), 7);
        assert_eq!(second.tokens.len(), 8);
        assert_eq!(&first.tokens[..4], &second.tokens[..4]);
        assert_eq!(tokenizer.encode(&first.text).unwrap(), first.tokens);
    }

    #[test]
    fn sonnet_prompts_are_natural_language_and_seed_deterministic() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::sonnet();

        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(11))).unwrap();
        let prompt = generator.generate(64, &[], 1).unwrap();
        assert_eq!(prompt.tokens.len(), 64);
        let alphabetic = prompt.text.chars().filter(|c| c.is_alphabetic()).count();
        assert!(
            alphabetic >= prompt.text.len() / 2,
            "expected natural-language corpus text, got {:?}",
            prompt.text
        );

        let mut same = factory.create(&tokenizer, RngRoot::new(Some(11))).unwrap();
        assert_eq!(same.generate(64, &[], 1).unwrap(), prompt);
        let mut other = factory.create(&tokenizer, RngRoot::new(Some(12))).unwrap();
        assert_ne!(other.generate(64, &[], 1).unwrap().tokens, prompt.tokens);
    }

    #[test]
    fn coding_prompts_are_seeded_and_distinct_from_sonnet() {
        let tokenizer = TiktokenTokenizer::builtin();
        let coding = CorpusPromptGeneratorFactory::coding();
        let sonnet = CorpusPromptGeneratorFactory::sonnet();

        let coding_prompt = coding
            .create(&tokenizer, RngRoot::new(Some(17)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();
        let repeated = coding
            .create(&tokenizer, RngRoot::new(Some(17)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();
        let sonnet_prompt = sonnet
            .create(&tokenizer, RngRoot::new(Some(17)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();

        assert_eq!(coding_prompt, repeated);
        assert_eq!(coding_prompt.tokens.len(), 32);
        assert_ne!(coding_prompt.tokens, sonnet_prompt.tokens);
    }

    #[test]
    fn random_prompts_are_seeded_and_reencode_to_exact_length() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::random();

        let prompt = factory
            .create(&tokenizer, RngRoot::new(Some(17)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();
        let repeated = factory
            .create(&tokenizer, RngRoot::new(Some(17)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();
        let other = factory
            .create(&tokenizer, RngRoot::new(Some(18)))
            .unwrap()
            .generate(32, &[], 1)
            .unwrap();

        assert_eq!(prompt, repeated);
        assert_eq!(prompt.tokens.len(), 32);
        assert_eq!(tokenizer.encode(&prompt.text).unwrap(), prompt.tokens);
        assert_ne!(prompt.tokens, other.tokens);
    }

    #[test]
    fn random_raw_token_generation_avoids_eos() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let tokenizer = NoDecodeTokenizer;
        let factory = CorpusPromptGeneratorFactory::random();
        let token_ids = factory
            .create(&tokenizer, RngRoot::new(Some(3)))
            .unwrap()
            .generate_token_ids(8, &[1, 2], 4)
            .unwrap();

        assert_eq!(token_ids.len(), 8);
        assert!(!token_ids.contains(&9));
    }

    #[test]
    fn random_generation_respects_allowed_token_filter_for_raw_and_text_modes() {
        let tokenizer = AllowedOnlyTokenizer;
        let factory = CorpusPromptGeneratorFactory::random();
        let mut generator = factory
            .create(&tokenizer, RngRoot::new(Some(5)))
            .expect("random generator");

        let raw = generator
            .generate_token_ids(8, &[], 1)
            .expect("raw token ids");
        assert_eq!(raw, vec![1; 8]);

        let text = generator.generate(8, &[], 1).expect("text prompt");
        assert_eq!(text.tokens, vec![1; 8]);
        assert_eq!(text.text, "aaaaaaaa");
    }

    #[test]
    fn reference_random_offsets_add_request_ordinal_and_style_selects_pool() {
        let tokenizer = AllowedOnlyTokenizer;
        let vllm = CorpusPromptGeneratorFactory::random_reference(
            RandomCorpusStyle::Vllm,
            Arc::from([2_usize, 2]),
        );
        let mut generator = vllm.create(&tokenizer, RngRoot::new(Some(5))).unwrap();
        assert_eq!(generator.generate_token_ids(4, &[], 1).unwrap(), vec![1; 4]);

        let sglang = CorpusPromptGeneratorFactory::random_reference(
            RandomCorpusStyle::Sglang,
            Arc::from([2_usize, 2]),
        );
        let mut generator = sglang.create(&tokenizer, RngRoot::new(Some(5))).unwrap();
        assert_eq!(
            generator.generate_token_ids(4, &[], 1).unwrap(),
            vec![2, 3, 4, 5]
        );
        assert_eq!(
            generator.generate_token_ids(4, &[], 1).unwrap(),
            vec![3, 4, 5, 6]
        );
    }

    #[test]
    fn custom_and_sonnet_factories_produce_distinct_corpora() {
        let tokenizer = TiktokenTokenizer::builtin();
        let custom = CorpusPromptGeneratorFactory::new("alpha beta gamma delta epsilon").unwrap();
        let sonnet = CorpusPromptGeneratorFactory::sonnet();
        let custom_prompt = custom
            .create(&tokenizer, RngRoot::new(Some(5)))
            .unwrap()
            .generate(4, &[], 1)
            .unwrap();
        let sonnet_prompt = sonnet
            .create(&tokenizer, RngRoot::new(Some(5)))
            .unwrap()
            .generate(4, &[], 1)
            .unwrap();
        assert_eq!(custom_prompt.tokens.len(), 4);
        assert_eq!(sonnet_prompt.tokens.len(), 4);
        assert_ne!(custom_prompt.tokens, sonnet_prompt.tokens);
    }

    #[test]
    fn empty_corpus_is_rejected() {
        assert!(CorpusPromptGeneratorFactory::new("   \n\t  ").is_err());
    }

    #[test]
    fn incompatible_hash_geometry_is_rejected() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::default();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(3))).unwrap();
        assert!(generator.generate(4, &[1, 2], 4).is_err());
    }

    #[test]
    fn raw_token_generation_skips_decode_and_replaces_eos() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let tokenizer = NoDecodeTokenizer;
        let factory = CorpusPromptGeneratorFactory::new("fixture").unwrap();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(3))).unwrap();
        let token_ids = generator.generate_token_ids(8, &[1, 2], 4).unwrap();

        assert_eq!(token_ids.len(), 8);
        assert!(!token_ids.contains(&9));
    }
}
