// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact corpus sampling for recorded trace reconstruction.
//!
//! Each KV block follows `HashIdRandomGenerator` and `sample_tokens_from_corpus`
//! (`dataset/loader/hash_ids_synthesis.py`, `dataset/generator/prompt.py`,
//! `common/hash_id_random_generator.py`): a per-block CPython MT19937 seeded from
//! `sha256(f"{corpus_child_seed}:{trace_id}:{hash_id}")[:8]` draws a start offset
//! via `randrange(corpus_len)`, and the block is `[sep] + corpus[start..]`
//! (wrapping), where `sep` is the tokenizer's block-separation token (BOS/EOS)
//! and consumes one slot. `corpus_child_seed` is
//! `rng.derive("dataset.coding_content.corpus")` — `sha256(f"{root}:{id}")[:8]`.
//!
//! Truncating a full `block_size` window to a message's partial-tail length is
//! prefix-stable (the same seed yields the same start, so a shorter window is a
//! prefix of the longer one).

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::dataset::TextTokenizer;
use crate::rng::{PythonMt19937, PythonRandomGenerator, derive_seed_u64, namespace};

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
    /// Block-separation token prepended to every block
    /// (`tokenizer.block_separation_token_id`, i.e. BOS or EOS), consuming one
    /// slot of the window. `None` when the tokenizer exposes neither.
    sep_token: Option<u32>,
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
            // Hash synthesis uses the SHA-256-derived corpus child seed, not
            // the BLAKE3 RngRoot algebra.
            hash_seed: PythonRandomGenerator::derive_child_seed(root_seed, hash_namespace),
            // WEKA per-turn windows carry no block
            // separator: each block is a full `block_size`-token window (verified
            // against the real sent prompt — the reconstruction is byte-exact only
            // with no sep, whereas a BOS-consuming window is one token short per
            // block). `sample_block` keeps the sep seam for the tested
            // `sample_tokens_from_corpus` contract, but the weka path drives it None.
            sep_token: None,
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
                let hash_str = hash.to_string();
                let block = sample_block(
                    &self.corpus,
                    self.hash_seed,
                    scope,
                    &hash_str,
                    block_size,
                    self.sep_token,
                );
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
        // `parallel_decode` uses `skip_special_tokens=True`, so the per-block
        // separator (BOS/EOS) is consumed as a
        // window slot but never appears in the sent text. Strip it here to match.
        if let Some(sep) = self.sep_token
            && tokens.contains(&sep)
        {
            let filtered: Vec<u32> = tokens.iter().copied().filter(|t| *t != sep).collect();
            return self
                .tokenizer
                .decode(&filtered)
                .map_err(|error| RecordedTraceError(error.to_string()));
        }
        self.tokenizer
            .decode(tokens)
            .map_err(|error| RecordedTraceError(error.to_string()))
    }
}

/// Draw one KV block's tokens byte-exactly: seed a CPython MT from
/// `sha256(f"{hash_seed}:{scope}:{hash}")[:8]` (big-endian), optionally prepend
/// `sep` (consuming one slot), then take a wrapping `window_len`-token window
/// starting at `randrange(corpus_len)`.
fn sample_block(
    corpus: &[u32],
    hash_seed: u64,
    scope: &str,
    hash: &str,
    block_size: usize,
    sep: Option<u32>,
) -> Vec<u32> {
    let mut hasher = Sha256::new();
    hasher.update(format!("{hash_seed}:{scope}:{hash}").as_bytes());
    let digest = hasher.finalize();
    let mut low8 = [0u8; 8];
    low8.copy_from_slice(&digest[..8]);
    let seed = u64::from_be_bytes(low8);
    let mut mt = PythonMt19937::from_u64_seed(seed);

    let mut block = Vec::with_capacity(block_size);
    let mut window_len = block_size;
    if let Some(sep_token) = sep {
        block.push(sep_token);
        window_len = window_len.saturating_sub(1);
    }
    let start = mt.randbelow(corpus.len() as u64) as usize;
    block.extend_from_slice(&wrapping_window(corpus, start, window_len));
    block
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

    #[derive(serde::Deserialize)]
    struct BlockCase {
        hash_id: u64,
        block_size: usize,
        sep: Option<u32>,
        tokens: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    struct BlockGolden {
        base_seed: u64,
        trace_id: String,
        pool_len: usize,
        blocks: Vec<BlockCase>,
    }

    #[test]
    fn per_block_window_matches_agentx_hash_id_sampling() {
        // The golden vectors cover HashIdRandomGenerator reseeding.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/agentx_block_golden.json"
        );
        let raw = std::fs::read_to_string(path).expect("read agentx block golden");
        let golden: BlockGolden = serde_json::from_str(&raw).expect("parse agentx block golden");
        let corpus: Vec<u32> = (0..golden.pool_len as u32).collect();
        for case in &golden.blocks {
            let got = sample_block(
                &corpus,
                golden.base_seed,
                &golden.trace_id,
                &case.hash_id.to_string(),
                case.block_size,
                case.sep,
            );
            assert_eq!(
                got, case.tokens,
                "hash_id {} block_size {} sep {:?}",
                case.hash_id, case.block_size, case.sep
            );
        }
    }

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
