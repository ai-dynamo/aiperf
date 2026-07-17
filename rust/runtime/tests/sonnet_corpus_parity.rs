// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-language parity: the Rust sonnet-corpus tokenization must match the
//! Python `PromptGenerator._initialize_corpus`
//! (`src/aiperf/dataset/generator/prompt.py`) byte-for-byte.
//!
//! The reference fixture `tests/data/sonnet_corpus_parity.json` is produced by
//! `tools/gen_sonnet_corpus_parity.py`, which runs the *real* Python
//! `PromptGenerator` with the built-in tiktoken `o200k_base` tokenizer and
//! records the tokenized-corpus digest. This test rebuilds the corpus through
//! `aiperf_runtime::dataset::corpus::tokenize_sonnet_corpus` with the equivalent Rust
//! `TiktokenTokenizer::builtin()` and asserts an identical token count,
//! head/tail window, and SHA-256 of the little-endian `u32` token stream. Any
//! drift in the embedded asset, the chunking policy, or the tokenizer backends
//! fails here.

use aiperf_runtime::dataset::TiktokenTokenizer;
use aiperf_runtime::dataset::corpus::tokenize_sonnet_corpus;
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &str = include_str!("data/sonnet_corpus_parity.json");

fn u32_vec(value: &Value, key: &str) -> Vec<u32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {key} is not an array"))
        .iter()
        .map(|entry| {
            u32::try_from(entry.as_u64().expect("token id is a u64")).expect("token fits u32")
        })
        .collect()
}

#[test]
fn rust_sonnet_corpus_matches_python_reference() {
    let reference: Value = serde_json::from_str(FIXTURE).expect("valid fixture json");
    assert_eq!(
        reference["tokenizer"].as_str(),
        Some("o200k_base"),
        "fixture must be generated with the built-in o200k_base tokenizer"
    );

    let tokenizer = TiktokenTokenizer::builtin();
    let corpus = tokenize_sonnet_corpus(&tokenizer).expect("sonnet corpus tokenizes");

    let expected_count = reference["corpus_token_count"]
        .as_u64()
        .expect("corpus_token_count is a u64") as usize;
    assert_eq!(
        corpus.len(),
        expected_count,
        "Rust corpus token count {} != Python {}",
        corpus.len(),
        expected_count
    );

    assert_eq!(
        &corpus[..16],
        u32_vec(&reference, "first_16").as_slice(),
        "leading 16 corpus tokens diverge from Python"
    );
    assert_eq!(
        &corpus[corpus.len() - 16..],
        u32_vec(&reference, "last_16").as_slice(),
        "trailing 16 corpus tokens diverge from Python"
    );

    let mut hasher = Sha256::new();
    for token in &corpus {
        hasher.update(token.to_le_bytes());
    }
    let digest = hasher.finalize();
    let digest_hex = digest.iter().fold(String::new(), |mut acc, byte| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{byte:02x}");
        acc
    });
    assert_eq!(
        digest_hex,
        reference["sha256_le_u32"]
            .as_str()
            .expect("sha256 hex string"),
        "full corpus SHA-256 diverges from Python reference"
    );
}
