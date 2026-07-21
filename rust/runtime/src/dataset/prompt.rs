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
use crate::rng::{RngRoot, RustRandomGenerator};

use crate::dataset::corpus::{SHAKESPEARE_CORPUS, tokenize_corpus_chunked};
use crate::dataset::error::{DatasetError, Result};
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
/// A trait-free enum is deliberate: the corpus is data, not behavior, and both
/// variants flow through the identical chunk-tokenization policy in
/// [`tokenize_corpus_chunked`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum CorpusSource {
    /// The embedded Shakespeare ("sonnet") product-default corpus.
    Sonnet,
    /// A caller-supplied corpus body (custom-corpus configs and tests).
    Custom(String),
}

impl CorpusSource {
    fn text(&self) -> &str {
        match self {
            Self::Sonnet => SHAKESPEARE_CORPUS,
            Self::Custom(text) => text,
        }
    }
}

/// Corpus-token generator with prefix block reuse.
///
/// The default Shakespeare corpus is tokenized once through the character-chunked
/// [`tokenize_corpus_chunked`] policy, then sampled with wrap-around and prefix
/// block reuse to produce exact-length synthetic prompts.
#[derive(Debug, Clone)]
pub struct CorpusPromptGeneratorFactory {
    corpus: CorpusSource,
}

impl CorpusPromptGeneratorFactory {
    /// Use the embedded Shakespeare ("sonnet") product-default corpus.
    pub fn sonnet() -> Self {
        Self {
            corpus: CorpusSource::Sonnet,
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
            corpus_tokens: tokenize_corpus_arc(self.corpus.text(), tokenizer)?,
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
        // Tokenize once into Arc and build the generator directly — avoid the
        // prepare→create detour that would Arc::clone a temporary factory.
        Ok(Box::new(CorpusPromptGenerator::from_corpus_tokens(
            tokenize_corpus_arc(self.corpus.text(), tokenizer)?,
            tokenizer,
            root,
        )))
    }
}

/// Prepared corpus tokens shared across many cheap generator creations.
///
/// Constructed by [`CorpusPromptGeneratorFactory::prepare`]. Each
/// [`PromptGeneratorFactory::create`] clones an [`Arc`] of the token stream and
/// builds fresh RNG / block-cache state; the corpus is not re-tokenized.
#[derive(Debug, Clone)]
pub struct PreparedCorpusPromptGeneratorFactory {
    corpus_tokens: Arc<[u32]>,
}

impl PromptGeneratorFactory for PreparedCorpusPromptGeneratorFactory {
    fn create<'a>(
        &self,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Result<Box<dyn PromptGenerator + 'a>> {
        Ok(Box::new(CorpusPromptGenerator::from_corpus_tokens(
            Arc::clone(&self.corpus_tokens),
            tokenizer,
            root,
        )))
    }
}

/// Tokenize a corpus body into a non-empty shared token stream.
fn tokenize_corpus_arc(corpus: &str, tokenizer: &dyn TextTokenizer) -> Result<Arc<[u32]>> {
    let corpus_tokens = tokenize_corpus_chunked(corpus, tokenizer)?;
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
    rng: RustRandomGenerator,
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
            rng: RustRandomGenerator::from_seed(root.derive_seed(DATASET_PROMPT_CORPUS)),
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

#[cfg(test)]
mod tests {
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
