// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Corpus sampling for recorded trace reconstruction through `aiperf-rng`.
//!
//! Stream derivation deliberately uses AIPerf's canonical BLAKE3/PCG64
//! substrate, so recorded adapters obey the same native reproducibility
//! contract as dataset composition and graph scheduling.

use std::collections::HashMap;

use crate::dataset::TextTokenizer;
use crate::rng::{RandomGenerator, RngRoot, derive_seed_u64, namespace};

use super::BlockHash;

use super::{PromptCorpus, RecordedTraceError};

/// Content extension seam consumed by the shared trie lowerer.
pub(crate) trait RecordedContentSynthesizer {
    /// Decode full cache blocks in order under a local or global hash namespace.
    fn block_tokens(
        &mut self,
        hashes: &[BlockHash],
        block_size: usize,
        trace_scope: Option<&str>,
    ) -> Result<Vec<u32>, RecordedTraceError>;

    /// Sample a deterministic non-wrapping partial-tail window.
    fn tail_tokens(&self, count: usize, seed: &str) -> Vec<u32>;

    /// Decode exact token IDs through the selected run tokenizer.
    fn decode(&self, tokens: &[u32]) -> Result<String, RecordedTraceError>;
}

/// Corpus-backed implementation shared by WEKA and Dynamo.
pub(crate) struct CorpusContentSynthesizer<'a> {
    tokenizer: &'a dyn TextTokenizer,
    corpus: Vec<u32>,
    hash_seed: u64,
    // Two-level so a cache-hit probe on the lowering hot path allocates nothing:
    // the scope `String` is owned only once per newly seen scope, and per-block
    // lookups key off the `Copy` `(hash, block_size)` tuple.
    blocks: HashMap<String, HashMap<(BlockHash, usize), Vec<u32>>>,
}

impl<'a> CorpusContentSynthesizer<'a> {
    pub(crate) fn new(
        tokenizer: &'a dyn TextTokenizer,
        corpus: PromptCorpus,
        root_seed: u64,
    ) -> Result<Self, RecordedTraceError> {
        let (tokens, hash_namespace) = match corpus {
            PromptCorpus::Sonnet => (
                crate::dataset::corpus::tokenize_sonnet_corpus(tokenizer)
                    .map_err(|error| RecordedTraceError(error.to_string()))?,
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
        hashes: &[BlockHash],
        block_size: usize,
        trace_scope: Option<&str>,
    ) -> Result<Vec<u32>, RecordedTraceError> {
        let scope = trace_scope.unwrap_or_default();
        // Own the scope string at most once per newly seen scope; subsequent
        // probes borrow it via `&str` and never re-allocate.
        if !self.blocks.contains_key(scope) {
            self.blocks.insert(scope.to_string(), HashMap::new());
        }
        let mut out = Vec::with_capacity(hashes.len().saturating_mul(block_size));
        for hash in hashes {
            let key = (*hash, block_size);
            if !self.blocks[scope].contains_key(&key) {
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
                self.blocks
                    .get_mut(scope)
                    .expect("scope cache present")
                    .insert(key, block);
            }
            out.extend_from_slice(&self.blocks[scope][&key]);
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

#[cfg(test)]
mod tests {
    use crate::dataset::TiktokenTokenizer;

    use super::*;

    #[test]
    fn blake3_hash_streams_preserve_global_and_trace_local_namespaces() {
        let tokenizer = TiktokenTokenizer::builtin();
        let mut content =
            CorpusContentSynthesizer::new(&tokenizer, PromptCorpus::Sonnet, 42).unwrap();
        let hash: BlockHash = "184467440737095516170".parse().unwrap();

        let global = content
            .block_tokens(std::slice::from_ref(&hash), 16, None)
            .unwrap();
        let global_again = content
            .block_tokens(std::slice::from_ref(&hash), 16, None)
            .unwrap();
        let local_a = content
            .block_tokens(std::slice::from_ref(&hash), 16, Some("trace-a"))
            .unwrap();
        let local_a_again = content
            .block_tokens(std::slice::from_ref(&hash), 16, Some("trace-a"))
            .unwrap();
        let local_b = content
            .block_tokens(std::slice::from_ref(&hash), 16, Some("trace-b"))
            .unwrap();

        assert_eq!(global, global_again);
        assert_eq!(local_a, local_a_again);
        assert_ne!(global, local_a);
        assert_ne!(local_a, local_b);
    }
}
