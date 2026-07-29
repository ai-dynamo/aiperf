// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Corpus-backed [`TokenSynth`] implementation: the token-generation bridge
//! between the hash-id RNG ([`super::rng`]) and the reconstructor
//! ([`super::synth`]).
//!
//! Ports the token-sampling behavior of `WekaTraceLoader._decode_block_tokens`
//! and `HashIdsPromptSynthesisMixin.sample_partial_tail_tokens`
//! (`src/aiperf/dataset/loader/weka_trace.py`, `hash_ids_synthesis.py`):
//!
//! - **block tokens**: for each `hash_id`, reseed the RNG for that id, draw
//!   `randrange(corpus_size)` as a start offset, and take `block_size` tokens
//!   from the tokenized corpus, wrapping at the end. Cached per id within a
//!   trace scope (the `hash_id_scope: "local"` contract).
//! - **partial tail**: `sha256(seed)`, first 8 bytes big-endian mod
//!   `max(corpus_size - n, 1)` as an offset, take `n` tokens (no wrap).
//!
//! `decode_tokens_to_text` delegates to an injected decoder (the Qwen tokenizer
//! in production; a stub in tests), so this module carries no tokenizer itself.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::agentx::rng::HashIdRandomGenerator;
use crate::agentx::synth::TokenSynth;

/// A tokenized-corpus-backed token generator matching the Python Weka loader.
pub struct CorpusTokenSynth<F: Fn(&[u32]) -> String> {
    corpus: Vec<u32>,
    block_size: i64,
    rng: HashIdRandomGenerator,
    cache: HashMap<i64, Vec<u32>>,
    decode: F,
}

impl<F: Fn(&[u32]) -> String> CorpusTokenSynth<F> {
    /// Construct from the tokenized corpus, block size, RNG base seed, trace id
    /// scope, and a token→text decoder.
    pub fn new(
        corpus: Vec<u32>,
        block_size: i64,
        base_seed: u64,
        trace_id: impl Into<String>,
        decode: F,
    ) -> Self {
        let mut rng = HashIdRandomGenerator::new(base_seed);
        rng.set_trace_id(trace_id);
        Self {
            corpus,
            block_size,
            rng,
            cache: HashMap::new(),
            decode,
        }
    }

    /// Re-scope to a new trace id, clearing the per-scope block cache. The Python
    /// loader clears `pg._cache` and sets the RNG trace id before each
    /// conversation scope; the `local` hash scope means only one scope's cache
    /// need be alive at a time.
    pub fn set_scope(&mut self, trace_id: impl Into<String>) {
        self.rng.set_trace_id(trace_id);
        self.cache.clear();
    }

    fn corpus_size(&self) -> i64 {
        self.corpus.len() as i64
    }

    fn block_for(&mut self, h: i64) -> Vec<u32> {
        if let Some(cached) = self.cache.get(&h) {
            return cached.clone();
        }
        let cs = self.corpus_size();
        self.rng.reseed_for_hash_id(h);
        let start = self.rng.randrange(cs) as usize;
        let bs = self.block_size as usize;
        let end = start + bs;
        let mut block: Vec<u32> = if end <= self.corpus.len() {
            self.corpus[start..end].to_vec()
        } else {
            // Wrap: tail of the corpus followed by its head (Python slice + wrap).
            let mut b = self.corpus[start..].to_vec();
            b.extend_from_slice(&self.corpus[..end - self.corpus.len()]);
            b
        };
        block.truncate(bs);
        self.cache.insert(h, block.clone());
        block
    }
}

impl<F: Fn(&[u32]) -> String> TokenSynth for CorpusTokenSynth<F> {
    fn decode_block_tokens(&mut self, hash_ids: &[i64]) -> Vec<u32> {
        let mut out = Vec::with_capacity(hash_ids.len() * self.block_size.max(0) as usize);
        for &h in hash_ids {
            out.extend(self.block_for(h));
        }
        out
    }

    fn sample_partial_tail_tokens(&mut self, n: usize, seed: &str) -> Vec<u32> {
        if n == 0 {
            return Vec::new();
        }
        let cs = self.corpus.len();
        let digest = Sha256::digest(seed.as_bytes());
        let mut first8 = [0u8; 8];
        first8.copy_from_slice(&digest[..8]);
        let raw = u64::from_be_bytes(first8);
        let modulus = (cs.saturating_sub(n)).max(1) as u64;
        let offset = (raw % modulus) as usize;
        let end = (offset + n).min(cs);
        self.corpus[offset..end].to_vec()
    }

    fn decode_tokens_to_text(&self, tokens: &[u32]) -> String {
        (self.decode)(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids_text(t: &[u32]) -> String {
        t.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn block_wrap_and_cache() {
        // corpus [0..10), bs=4. Force a start near the end to exercise wrap.
        let corpus: Vec<u32> = (0..10).collect();
        let mut s = CorpusTokenSynth::new(corpus, 4, 42, "t", ids_text);
        let a = s.decode_block_tokens(&[7]);
        assert_eq!(a.len(), 4);
        // Cached: second call returns identical block.
        let b = s.decode_block_tokens(&[7]);
        assert_eq!(a, b);
        // Every token is within the corpus range.
        assert!(a.iter().all(|&t| t < 10));
    }

    /// Golden values captured from the real Python `HashIdRandomGenerator` +
    /// the `_decode_block_tokens` / `sample_partial_tail_tokens` formulas over a
    /// `0..corpus_size` integer corpus.
    #[test]
    fn matches_python_block_and_tail_golden() {
        // block_tokens(base_seed=42, tid="t", corpus_size=50, bs=4, [7,1,7,3])
        let mut s = CorpusTokenSynth::new((0..50).collect(), 4, 42, "t", ids_text);
        assert_eq!(
            s.decode_block_tokens(&[7, 1, 7, 3]),
            vec![
                22, 23, 24, 25, 34, 35, 36, 37, 22, 23, 24, 25, 23, 24, 25, 26
            ]
        );
        // block_tokens(1234567890, "trace_0012", 30, 3, [99999, 0])
        let mut s2 =
            CorpusTokenSynth::new((0..30).collect(), 3, 1234567890, "trace_0012", ids_text);
        assert_eq!(
            s2.decode_block_tokens(&[99999, 0]),
            vec![9, 10, 11, 9, 10, 11]
        );
        // tail(corpus_size=100, n=5, "seed-x")
        let mut s3 = CorpusTokenSynth::new((0..100).collect(), 4, 0, "t", ids_text);
        assert_eq!(
            s3.sample_partial_tail_tokens(5, "seed-x"),
            vec![84, 85, 86, 87, 88]
        );
        // tail(64, 10, "call_turn_3")
        let mut s4 = CorpusTokenSynth::new((0..64).collect(), 4, 0, "t", ids_text);
        assert_eq!(
            s4.sample_partial_tail_tokens(10, "call_turn_3"),
            vec![35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        );
    }

    /// The AgentX corpus-equivalence gate: `dataset::coding::build_coding_corpus`
    /// must reproduce Python `CodingContentGenerator._tokenized_corpus` (the weka
    /// `coding` corpus) byte-for-byte at seed 42 with Qwen3-0.6B. Golden captured
    /// from the real Python generator. Skips when Qwen is not in the local HF cache.
    #[test]
    fn build_coding_corpus_matches_python_agentx_corpus() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let base = format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        let snap = match std::fs::read_dir(&base) {
            Ok(d) => d
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("tokenizer.json").exists()),
            Err(_) => {
                eprintln!("skip: Qwen3-0.6B not in local HF cache");
                return;
            }
        };
        let snap = match snap {
            Some(s) => s,
            None => {
                eprintln!("skip: no Qwen snapshot with tokenizer.json");
                return;
            }
        };
        let tok = crate::dataset::tokenizer::HuggingFaceTokenizer::from_directory(&snap)
            .expect("load Qwen tokenizer");
        let corpus =
            crate::dataset::coding::build_coding_corpus(&tok, 42).expect("build coding corpus");
        assert_eq!(corpus.len(), 276439, "corpus length");
        assert_eq!(
            &corpus[..20],
            &[
                90, 515, 220, 330, 2829, 788, 330, 562, 756, 220, 330, 691, 788, 80505, 262, 330,
                17437, 788, 330, 69134
            ],
            "corpus prefix"
        );
        assert_eq!(
            &corpus[100000..100020],
            &[
                741, 262, 220, 20, 19, 1572, 262, 220, 20, 20, 2760, 3312, 707, 49304, 1193, 11,
                897, 25, 1140, 8
            ],
            "corpus mid"
        );
    }

    #[test]
    fn partial_tail_is_deterministic_and_sized() {
        let corpus: Vec<u32> = (0..100).collect();
        let mut s = CorpusTokenSynth::new(corpus, 4, 1, "t", ids_text);
        let a = s.sample_partial_tail_tokens(5, "seed-x");
        let b = s.sample_partial_tail_tokens(5, "seed-x");
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        let c = s.sample_partial_tail_tokens(5, "seed-y");
        assert_ne!(a, c); // different seed -> different offset (overwhelmingly)
    }
}
