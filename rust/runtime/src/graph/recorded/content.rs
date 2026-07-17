// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact corpus sampling for recorded trace reconstruction.
//!
//! Each KV block uses a per-block CPython MT19937 seeded from
//! `sha256(f"{corpus_child_seed}:{trace_id}:{hash_id}")[:8]` draws a start offset
//! via `randrange(corpus_len)`, and the block is `[sep] + corpus[start..]`
//! (wrapping), where `sep` is the tokenizer's block-separation token (BOS/EOS)
//! and consumes one slot. `corpus_child_seed` is
//! `rng.derive("dataset.coding_content.corpus")` — `sha256(f"{root}:{id}")[:8]`.
//!
//! Truncating a full `block_size` window to a message's partial-tail length is
//! prefix-stable (the same seed yields the same start, so a shorter window is a
//! prefix of the longer one).

use rustc_hash::FxHashMap;
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

/// Immutable corpus state shared across every per-trace synthesizer.
///
/// Tokenizing the corpus and deriving the hash seed is done once; the result is
/// `Send + Sync` (a token vector, a `u64`, an `Option<u32>`, and a `Send + Sync`
/// tokenizer reference), so many [`CorpusContentSynthesizer`]s can borrow it
/// concurrently from a rayon fan-out — each owning only its private block cache.
pub(crate) struct CorpusShared<'a> {
    tokenizer: &'a dyn TextTokenizer,
    corpus: Vec<u32>,
    hash_seed: u64,
    /// Block-separation token prepended to every block
    /// (`tokenizer.block_separation_token_id`, i.e. BOS or EOS), consuming one
    /// slot of the window. `None` when the tokenizer exposes neither.
    sep_token: Option<u32>,
    /// Draw the corpus window start with a CPython MT19937 exactly as the Python
    /// reference does, so reconstructed content is byte-identical to Python. This
    /// is the **default** (full byte parity). Seeding a fresh 624-word Mersenne
    /// state per block is measurable but not the bottleneck, so the parity path
    /// stays on unless `AIPERF_WEKA_FAST_CONTENT` opts into deriving the offset
    /// directly from a BLAKE3 digest instead — that keeps window length, prefix
    /// sharing, token counts, and graph structure identical but changes only the
    /// exact filler bytes, trading Python byte parity for a little less work.
    python_parity: bool,
}

impl<'a> CorpusShared<'a> {
    /// Tokenize the corpus and derive the hash seed exactly once.
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
            // Byte parity with Python by default; opt out for a little more speed.
            python_parity: std::env::var_os("AIPERF_WEKA_FAST_CONTENT").is_none(),
        })
    }

    /// Open a fresh per-trace synthesizer with an empty block cache. The block
    /// cache is a pure memoization keyed by `(scope, hash, block_size)`, so a
    /// per-trace instance produces byte-identical output to a shared one — it
    /// merely rebuilds the cache privately, which is what makes the fan-out safe.
    pub(crate) fn synthesizer(&self) -> CorpusContentSynthesizer<'_> {
        CorpusContentSynthesizer {
            shared: self,
            blocks: FxHashMap::default(),
        }
    }
}

/// Corpus-backed implementation shared by WEKA and Dynamo.
pub(crate) struct CorpusContentSynthesizer<'a> {
    shared: &'a CorpusShared<'a>,
    // Two-level so a cache-hit probe on the lowering hot path allocates nothing:
    // the scope `String` is owned only once per newly seen scope, and per-block
    // lookups key off the `Copy` `(hash, block_size)` tuple.
    blocks: FxHashMap<String, FxHashMap<(BlockHash, usize), Vec<u32>>>,
}

impl<'a> CorpusContentSynthesizer<'a> {
    /// Build a self-contained synthesizer that owns its corpus (sequential path).
    pub(crate) fn new(
        tokenizer: &'a dyn TextTokenizer,
        corpus: PromptCorpus,
        root_seed: u64,
    ) -> Result<OwnedCorpusContentSynthesizer<'a>, RecordedTraceError> {
        Ok(OwnedCorpusContentSynthesizer {
            shared: CorpusShared::new(tokenizer, corpus, root_seed)?,
        })
    }
}

/// Owning wrapper for callers that build and drive one synthesizer inline.
///
/// It holds the [`CorpusShared`] and hands out a borrowing
/// [`CorpusContentSynthesizer`] via [`Self::as_synthesizer`], keeping the single
/// self-contained construction ergonomic for the sequential Dynamo and
/// `aiperf_trace` paths while the shared corpus stays borrow-friendly for WEKA.
pub(crate) struct OwnedCorpusContentSynthesizer<'a> {
    shared: CorpusShared<'a>,
}

impl<'a> OwnedCorpusContentSynthesizer<'a> {
    /// Borrow a fresh per-instance synthesizer over the owned corpus.
    pub(crate) fn as_synthesizer(&self) -> CorpusContentSynthesizer<'_> {
        self.shared.synthesizer()
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
            self.blocks.insert(scope.to_string(), FxHashMap::default());
        }
        let mut out = Vec::with_capacity(hashes.len().saturating_mul(block_size));
        for hash in hashes {
            let key = (*hash, block_size);
            if !self.blocks[scope].contains_key(&key) {
                let hash_str = hash.to_string();
                let block = sample_block(
                    &self.shared.corpus,
                    self.shared.hash_seed,
                    scope,
                    &hash_str,
                    block_size,
                    self.shared.sep_token,
                    self.shared.python_parity,
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
        let corpus = &self.shared.corpus;
        let modulus = corpus.len().saturating_sub(count).max(1);
        let offset = (derive_seed_u64(seed) % modulus as u64) as usize;
        corpus[offset..corpus.len().min(offset.saturating_add(count))].to_vec()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, RecordedTraceError> {
        // `parallel_decode` uses `skip_special_tokens=True`, so the per-block
        // separator (BOS/EOS) is consumed as a
        // window slot but never appears in the sent text. Strip it here to match.
        if let Some(sep) = self.shared.sep_token
            && tokens.contains(&sep)
        {
            let filtered: Vec<u32> = tokens.iter().copied().filter(|t| *t != sep).collect();
            return self
                .shared
                .tokenizer
                .decode_lossy(&filtered)
                .map_err(|error| RecordedTraceError(error.to_string()));
        }
        self.shared
            .tokenizer
            .decode_lossy(tokens)
            .map_err(|error| RecordedTraceError(error.to_string()))
    }
}

/// Draw one KV block's tokens.
///
/// With `python_parity`, the window start reproduces the Python reference byte
/// for byte: seed a CPython MT from `sha256(f"{hash_seed}:{scope}:{hash}")[:8]`
/// (big-endian) and take `randrange(corpus_len)`. By default the start is derived
/// directly from a BLAKE3 digest of the same key (`blake3(...)[:8] % corpus_len`),
/// which avoids seeding a fresh 624-word Mersenne state per block. Both paths take
/// the same wrapping `window_len`-token window and optionally prepend `sep`, so
/// only the exact window offset — never the length, prefix sharing, or token
/// count — differs between them.
fn sample_block(
    corpus: &[u32],
    hash_seed: u64,
    scope: &str,
    hash: &str,
    block_size: usize,
    sep: Option<u32>,
    python_parity: bool,
) -> Vec<u32> {
    let mut block = Vec::with_capacity(block_size);
    let mut window_len = block_size;
    if let Some(sep_token) = sep {
        block.push(sep_token);
        window_len = window_len.saturating_sub(1);
    }
    let start = if python_parity {
        let mut hasher = Sha256::new();
        hasher.update(format!("{hash_seed}:{scope}:{hash}").as_bytes());
        let digest = hasher.finalize();
        let mut low8 = [0u8; 8];
        low8.copy_from_slice(&digest[..8]);
        let seed = u64::from_be_bytes(low8);
        PythonMt19937::from_u64_seed(seed).randbelow(corpus.len() as u64) as usize
    } else {
        let digest = blake3::hash(format!("{hash_seed}:{scope}:{hash}").as_bytes());
        let low8 = <[u8; 8]>::try_from(&digest.as_bytes()[..8]).expect("blake3 digest is 32 bytes");
        (u64::from_le_bytes(low8) % corpus.len() as u64) as usize
    };
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
                // The golden vectors encode the exact CPython MT19937 draw.
                true,
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
        let owned = CorpusContentSynthesizer::new(&tokenizer, PromptCorpus::Sonnet, 42).unwrap();
        let mut content = owned.as_synthesizer();
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
