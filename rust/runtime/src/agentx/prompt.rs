// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone Weka prompt-token composer (hash blocks + sha256-keyed tail),
//! ported from `src/aiperf/dataset/loader/weka_prompt_compose.py`.
//!
//! The same hash-id-seeded RNG used by the conversation reconstructor for LCP
//! segments is reused here for the prompt itself, so workers produce both
//! byte-deterministically without a separate decode pool. Token generation is
//! injected via two callbacks so this module stays pure and testable:
//!
//! - `decode_block_tokens(hash_ids)` — the exact token block sequence for the
//!   given hash ids (hash-id-seeded).
//! - `sample_partial_tail_tokens(n, seed)` — an `n`-token sha256-keyed sample,
//!   position-deterministic and reproducible across processes.

/// Build the prompt token sequence for a Weka turn (three ISL layouts).
///
/// - `hash_ids` empty: prompt is entirely a sha256-keyed sample of length
///   `input_length`.
/// - `input_length <= block tokens`: exact-tile / last-block-partial — truncate
///   the hashed prefix to `input_length`.
/// - `input_length > block tokens`: prefix-only — append a sha256-keyed partial
///   tail.
///
/// `input_length` is clamped at zero (a non-positive recorded length yields an
/// empty prompt rather than a panic; callers pass recorded `in[k] >= 0`).
pub fn compose_weka_prompt_tokens<D, S>(
    hash_ids: &[i64],
    input_length: i64,
    decode_block_tokens: D,
    sample_partial_tail_tokens: S,
    seed: &str,
) -> Vec<u32>
where
    D: FnOnce(&[i64]) -> Vec<u32>,
    S: FnOnce(usize, &str) -> Vec<u32>,
{
    let want = input_length.max(0) as usize;
    if hash_ids.is_empty() {
        return sample_partial_tail_tokens(want, seed);
    }
    let block_tokens = decode_block_tokens(hash_ids);
    if want <= block_tokens.len() {
        let mut t = block_tokens;
        t.truncate(want);
        return t;
    }
    let tail = want - block_tokens.len();
    let mut out = block_tokens;
    out.extend(sample_partial_tail_tokens(tail, seed));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic stub token generators standing in for the tokenizer-backed
    // callbacks: block hash `h` decodes to a run of `[h*10, h*10+1, ...]` per
    // 4-token block; the tail sampler emits `[900, 901, ...]`.
    fn stub_blocks(hash_ids: &[i64]) -> Vec<u32> {
        hash_ids
            .iter()
            .flat_map(|&h| (0..4).map(move |i| (h as u32) * 10 + i))
            .collect()
    }
    fn stub_tail(n: usize, _seed: &str) -> Vec<u32> {
        (0..n as u32).map(|i| 900 + i).collect()
    }

    #[test]
    fn empty_hash_ids_is_all_tail() {
        let out = compose_weka_prompt_tokens(&[], 3, stub_blocks, stub_tail, "s");
        assert_eq!(out, vec![900, 901, 902]);
    }

    #[test]
    fn exact_or_partial_truncates_prefix() {
        // 2 blocks -> 8 tokens; want 5 -> truncate to first 5 block tokens.
        let out = compose_weka_prompt_tokens(&[1, 2], 5, stub_blocks, stub_tail, "s");
        assert_eq!(out, vec![10, 11, 12, 13, 20]);
    }

    #[test]
    fn prefix_only_appends_tail() {
        // 1 block -> 4 tokens; want 6 -> 4 block tokens + 2 tail tokens.
        let out = compose_weka_prompt_tokens(&[3], 6, stub_blocks, stub_tail, "s");
        assert_eq!(out, vec![30, 31, 32, 33, 900, 901]);
    }
}
