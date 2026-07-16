// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared "sonnet" (Shakespeare) synthetic-prompt corpus and its byte-exact
//! chunk tokenization.
//!
//! This is the single owner of the embedded Shakespeare corpus used by both the
//! config-driven synthetic prompt generator ([`crate::dataset::prompt`]) and the
//! recorded-trace content synthesizer ([`crate::graph::recorded`]). Keeping one
//! embed and one tokenization routine here prevents the two corpus consumers
//! from drifting apart and avoids embedding the ~5 MB asset twice.
//!
//! The chunk tokenization reproduces the inherited Python
//! `PromptGenerator._initialize_corpus`
//! (`src/aiperf/dataset/generator/prompt.py:89-153`) exactly: lines are
//! stripped, empty lines dropped, accumulated into character-bounded chunks,
//! each chunk joined with a single space and tokenized independently, and the
//! per-chunk token vectors concatenated. Character-based (not CPU-based)
//! chunking is what makes the token boundaries — and therefore every sampled
//! prompt — reproducible across machines and across the Python/Rust split.

use crate::dataset::error::Result;
use crate::dataset::tokenizer::TextTokenizer;

/// The embedded Shakespeare corpus (genai-perf's canonical synthetic source).
///
/// Sole embed site for this asset; see the module docs for why it lives here.
pub const SHAKESPEARE_CORPUS: &str =
    include_str!("../../../../src/aiperf/dataset/generator/assets/shakespeare.txt");

/// Fixed per-chunk character budget matching Python's `MAX_CHARS_PER_CHUNK`
/// (`src/aiperf/dataset/generator/prompt.py:120`). Fixed — never CPU-derived —
/// so tokenization boundaries are identical on every host.
pub const MAX_CHARS_PER_CHUNK: usize = 10_000;

/// Tokenize the embedded Shakespeare corpus with the run's tokenizer.
///
/// Equivalent to constructing a Python `PromptGenerator` with the default
/// `DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"`.
pub fn tokenize_sonnet_corpus(tokenizer: &dyn TextTokenizer) -> Result<Vec<u32>> {
    tokenize_corpus_chunked(SHAKESPEARE_CORPUS, tokenizer)
}

/// Tokenize an arbitrary corpus body using Python's character-chunked algorithm.
///
/// Applying the identical stripping/chunking/join policy to custom corpora keeps
/// a caller-supplied corpus on the same reproducibility contract as the default
/// Shakespeare source rather than encoding the raw string wholesale (which would
/// merge line boundaries and cross-chunk BPE merges differently).
pub fn tokenize_corpus_chunked(corpus: &str, tokenizer: &dyn TextTokenizer) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    let mut buffer: Vec<&str> = Vec::new();
    let mut chars = 0_usize;

    for raw_line in corpus.lines() {
        // Python: `line.strip()` then drop empties (prompt.py:109).
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        buffer.push(line);
        // Python `len(line)` counts Unicode code points, matching `chars().count()`.
        chars = chars.saturating_add(line.chars().count());
        if chars >= MAX_CHARS_PER_CHUNK {
            tokens.extend(tokenizer.encode(&buffer.join(" "))?);
            buffer.clear();
            chars = 0;
        }
    }
    // Trailing partial chunk (prompt.py:137-138).
    if !buffer.is_empty() {
        tokens.extend(tokenizer.encode(&buffer.join(" "))?);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tokenizer::TiktokenTokenizer;

    #[test]
    fn sonnet_corpus_is_embedded_and_substantial() {
        assert!(SHAKESPEARE_CORPUS.len() > 4_000_000);
        // The genai-perf asset opens with the sonnet collection header.
        assert!(SHAKESPEARE_CORPUS.trim_start().starts_with("THE SONNETS"));
    }

    #[test]
    fn sonnet_corpus_tokenizes_to_a_large_reproducible_stream() {
        let tokenizer = TiktokenTokenizer::builtin();
        let first = tokenize_sonnet_corpus(&tokenizer).unwrap();
        let second = tokenize_sonnet_corpus(&tokenizer).unwrap();
        // Tokenization is a pure function of (corpus, tokenizer).
        assert_eq!(first, second);
        // ~5 MB of English is on the order of a million tokens; guard the floor.
        assert!(
            first.len() > 1_000_000,
            "sonnet corpus produced only {} tokens",
            first.len()
        );
    }

    #[test]
    fn chunk_boundaries_follow_character_budget_not_wholesale_encode() {
        // Two lines whose combined stripped length crosses the chunk budget must
        // be tokenized as two independent chunks; a wholesale encode of the
        // joined string would differ at the boundary for a real BPE tokenizer.
        let tokenizer = TiktokenTokenizer::builtin();
        let long_line = "word ".repeat(MAX_CHARS_PER_CHUNK); // > budget on its own
        let corpus = format!("{long_line}\ntail line");
        let chunked = tokenize_corpus_chunked(&corpus, &tokenizer).unwrap();

        let first_chunk = tokenizer.encode(long_line.trim()).unwrap();
        let second_chunk = tokenizer.encode("tail line").unwrap();
        let mut expected = first_chunk;
        expected.extend(second_chunk);
        assert_eq!(chunked, expected);
    }

    #[test]
    fn blank_lines_are_dropped_before_joining() {
        let tokenizer = TiktokenTokenizer::builtin();
        let with_blanks = tokenize_corpus_chunked("alpha\n\n   \nbeta", &tokenizer).unwrap();
        let without_blanks = tokenizer.encode("alpha beta").unwrap();
        assert_eq!(with_blanks, without_blanks);
    }
}
