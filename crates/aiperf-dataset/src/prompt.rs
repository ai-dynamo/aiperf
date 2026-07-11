// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-length synthetic prompt generation for trace and synthetic datasets.
//!
//! The block-cache algorithm follows
//! `src/aiperf/dataset/generator/prompt.py:174-343`: each hash id owns one token
//! block, the final block may be shorter, a BOS/EOS separator occupies the first
//! token when available, and sampled corpus tokens wrap at the corpus boundary.

use std::collections::HashMap;

use aiperf_rng::{RandomGenerator, RngRoot};

use crate::error::{DatasetError, Result};
use crate::tokenizer::TextTokenizer;

const DEFAULT_CORPUS: &str = "To benchmark inference faithfully, deterministic prompts preserve shared prefixes while varied continuations exercise the complete serving path. ";

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

/// Corpus-token generator with prefix block reuse.
#[derive(Debug, Clone)]
pub struct CorpusPromptGeneratorFactory {
    corpus: String,
}

impl CorpusPromptGeneratorFactory {
    /// Use a caller-provided non-empty corpus.
    pub fn new(corpus: impl Into<String>) -> Result<Self> {
        let corpus = corpus.into();
        if corpus.is_empty() {
            return Err(DatasetError::Validation(
                "prompt generator corpus cannot be empty".into(),
            ));
        }
        Ok(Self { corpus })
    }
}

impl Default for CorpusPromptGeneratorFactory {
    fn default() -> Self {
        Self {
            corpus: DEFAULT_CORPUS.repeat(32),
        }
    }
}

impl PromptGeneratorFactory for CorpusPromptGeneratorFactory {
    fn create<'a>(
        &self,
        tokenizer: &'a dyn TextTokenizer,
        root: RngRoot,
    ) -> Result<Box<dyn PromptGenerator + 'a>> {
        let corpus_tokens = tokenizer.encode(&self.corpus)?;
        if corpus_tokens.is_empty() {
            return Err(DatasetError::Validation(
                "prompt generator corpus encoded to zero tokens".into(),
            ));
        }
        Ok(Box::new(CorpusPromptGenerator {
            corpus_tokens,
            block_separator: tokenizer.block_separation_token_id(),
            tokenizer,
            rng: RandomGenerator::from_seed(root.derive_seed("dataset.prompt.corpus")),
            blocks: HashMap::new(),
        }))
    }
}

struct CorpusPromptGenerator<'a> {
    corpus_tokens: Vec<u32>,
    block_separator: Option<u32>,
    tokenizer: &'a dyn TextTokenizer,
    rng: RandomGenerator,
    blocks: HashMap<i64, Vec<u32>>,
}

impl PromptGenerator for CorpusPromptGenerator<'_> {
    fn generate(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<GeneratedPrompt> {
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
        Ok(GeneratedPrompt {
            text: self.tokenizer.decode(&tokens)?,
            tokens,
        })
    }
}

impl CorpusPromptGenerator<'_> {
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
    use aiperf_rng::RngRoot;

    use super::*;
    use crate::tokenizer::TiktokenTokenizer;

    #[test]
    fn exact_lengths_and_hash_prefix_reuse_are_proven() {
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
    fn incompatible_hash_geometry_is_rejected() {
        let tokenizer = TiktokenTokenizer::builtin();
        let factory = CorpusPromptGeneratorFactory::default();
        let mut generator = factory.create(&tokenizer, RngRoot::new(Some(3))).unwrap();
        assert!(generator.generate(4, &[1, 2], 4).is_err());
    }
}
