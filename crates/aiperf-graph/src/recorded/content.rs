// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Corpus sampling for recorded trace reconstruction through `aiperf-rng`.
//!
//! Block/tail selection follows
//! `../aiperf-graph-ir/src/aiperf/dataset/graph/adapters/shared/content.py:241-433`.
//! Stream derivation deliberately uses AIPerf's canonical BLAKE3/PCG64
//! substrate, so recorded adapters obey the same native reproducibility
//! contract as dataset composition and graph scheduling.

use std::collections::HashMap;

use aiperf_dataset::TextTokenizer;
use aiperf_rng::{RandomGenerator, RngRoot, derive_seed_u64, namespace};
use num_bigint::BigInt;

use super::{PromptCorpus, RecordedTraceError};

const SHAKESPEARE: &str =
    include_str!("../../../../src/aiperf/dataset/generator/assets/shakespeare.txt");

/// Content extension seam consumed by the shared trie lowerer.
pub(crate) trait RecordedContentSynthesizer {
    /// Decode full cache blocks in order under a local or global hash namespace.
    fn block_tokens(
        &mut self,
        hashes: &[BigInt],
        block_size: usize,
        trace_scope: Option<&str>,
    ) -> Result<Vec<u32>, RecordedTraceError>;

    /// Sample a deterministic non-wrapping partial-tail window.
    fn tail_tokens(&self, count: usize, seed: &str) -> Vec<u32>;

    /// Decode exact token IDs through the selected run tokenizer.
    fn decode(&self, tokens: &[u32]) -> Result<String, RecordedTraceError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BlockCacheKey {
    scope: String,
    hash: BigInt,
    block_size: usize,
}

/// Corpus-backed implementation shared by WEKA and Dynamo.
pub(crate) struct CorpusContentSynthesizer<'a> {
    tokenizer: &'a dyn TextTokenizer,
    corpus: Vec<u32>,
    hash_seed: u64,
    blocks: HashMap<BlockCacheKey, Vec<u32>>,
}

impl<'a> CorpusContentSynthesizer<'a> {
    pub(crate) fn new(
        tokenizer: &'a dyn TextTokenizer,
        corpus: PromptCorpus,
        root_seed: u64,
    ) -> Result<Self, RecordedTraceError> {
        let (tokens, hash_namespace) = match corpus {
            PromptCorpus::Sonnet => (
                build_sonnet_corpus(tokenizer)?,
                namespace::DATASET_PROMPT_CORPUS,
            ),
            PromptCorpus::Coding => (
                super::coding::build_coding_corpus(tokenizer, root_seed)?,
                namespace::DATASET_CODING_CONTENT_CORPUS,
            ),
        };
        if tokens.is_empty() {
            return Err(RecordedTraceError(
                "recorded content corpus tokenized to an empty sequence".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            corpus: tokens,
            hash_seed: RngRoot::new(Some(root_seed))
                .derive_seed(hash_namespace)
                .expect("seeded root always derives a seed"),
            blocks: HashMap::new(),
        })
    }
}

impl RecordedContentSynthesizer for CorpusContentSynthesizer<'_> {
    fn block_tokens(
        &mut self,
        hashes: &[BigInt],
        block_size: usize,
        trace_scope: Option<&str>,
    ) -> Result<Vec<u32>, RecordedTraceError> {
        let scope = trace_scope.unwrap_or_default();
        let mut out = Vec::with_capacity(hashes.len().saturating_mul(block_size));
        for hash in hashes {
            let key = BlockCacheKey {
                scope: scope.to_string(),
                hash: hash.clone(),
                block_size,
            };
            let block = if let Some(cached) = self.blocks.get(&key) {
                cached
            } else {
                let seed = derive_seed_u64(&format!("{}:{scope}:{hash}", self.hash_seed));
                let mut random = RandomGenerator::from_seed(Some(seed));
                let upper = i64::try_from(self.corpus.len()).map_err(|_| {
                    RecordedTraceError("recorded content corpus exceeds i64 range".into())
                })?;
                let start = usize::try_from(
                    random
                        .randrange(0, upper, 1)
                        .map_err(|error| RecordedTraceError(error.to_string()))?,
                )
                .expect("non-negative corpus offset");
                let block = wrapping_window(&self.corpus, start, block_size);
                self.blocks.insert(key.clone(), block);
                self.blocks.get(&key).expect("inserted block cache entry")
            };
            out.extend_from_slice(block);
        }
        Ok(out)
    }

    fn tail_tokens(&self, count: usize, seed: &str) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        let modulus = self.corpus.len().saturating_sub(count).max(1);
        let offset = (derive_seed_u64(seed) % modulus as u64) as usize;
        self.corpus[offset..self.corpus.len().min(offset.saturating_add(count))].to_vec()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, RecordedTraceError> {
        self.tokenizer
            .decode(tokens)
            .map_err(|error| RecordedTraceError(error.to_string()))
    }
}

fn build_sonnet_corpus(tokenizer: &dyn TextTokenizer) -> Result<Vec<u32>, RecordedTraceError> {
    const MAX_CHARS_PER_CHUNK: usize = 10_000;
    let mut corpus = Vec::new();
    let mut buffer = Vec::<&str>::new();
    let mut chars = 0_usize;
    for raw in SHAKESPEARE.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        buffer.push(line);
        chars = chars.saturating_add(line.chars().count());
        if chars >= MAX_CHARS_PER_CHUNK {
            corpus.extend(
                tokenizer
                    .encode(&buffer.join(" "))
                    .map_err(|error| RecordedTraceError(error.to_string()))?,
            );
            buffer.clear();
            chars = 0;
        }
    }
    if !buffer.is_empty() {
        corpus.extend(
            tokenizer
                .encode(&buffer.join(" "))
                .map_err(|error| RecordedTraceError(error.to_string()))?,
        );
    }
    Ok(corpus)
}

fn wrapping_window(corpus: &[u32], start: usize, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let mut position = start;
    while out.len() < count {
        let available = corpus.len() - position;
        let take = available.min(count - out.len());
        out.extend_from_slice(&corpus[position..position + take]);
        position = 0;
    }
    out
}
