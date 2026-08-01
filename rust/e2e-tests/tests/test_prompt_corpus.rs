// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::collections::HashSet;

use aiperf_runtime::dataset::TextTokenizer;
use aiperf_runtime::dataset::TiktokenTokenizer;

// `--prompt-corpus` selects which body of text synthetic prompts are drawn from.
// The selector must not leak into the length contract: whichever corpus is named,
// every served prompt still has to hit the exact requested input length.

const REQUEST_COUNT: u32 = 8;
const ISL: usize = 48;
const SEED: u32 = 4321;

/// Run the same synthetic workload under one named corpus and return the served
/// user prompts, in record order.
async fn served_prompts(corpus: &str, seed: u32) -> Vec<String> {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --tokenizer o200k_base \
         --url {} \
         --endpoint-type chat \
         --prompt-corpus {corpus} \
         --synthetic-input-tokens-mean {ISL} \
         --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean 1 \
         --random-seed {seed} \
         --request-count {REQUEST_COUNT} \
         --concurrency 1 \
         --export-level raw \
         --ui none",
        h.mock.url
    ));
    assert!(
        r.success(),
        "`--prompt-corpus {corpus}` run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let raw = r.artifacts.raw_records();
    assert_eq!(
        raw.len(),
        REQUEST_COUNT as usize,
        "corpus {corpus}: expected {REQUEST_COUNT} raw records, got {}",
        raw.len()
    );
    raw.iter()
        .enumerate()
        .map(|(index, record)| {
            record["payload"]["messages"][0]["content"]
                .as_str()
                .unwrap_or_else(|| {
                    panic!("corpus {corpus} record {index} has no user content: {record}")
                })
                .to_string()
        })
        .collect()
}

/// Every corpus must hold the input length exactly, and the corpora must actually
/// differ — a selector that silently fell back to one default would serve the same
/// prompts under all three names and still pass a length-only assertion.
#[tokio::test]
async fn each_prompt_corpus_holds_input_length_and_serves_distinct_text() {
    if cfg!(target_os = "macos") {
        return;
    }

    let tokenizer = TiktokenTokenizer::builtin();
    let mut per_corpus: Vec<(&str, Vec<String>)> = Vec::new();
    for corpus in ["sonnet", "coding", "random"] {
        let prompts = served_prompts(corpus, SEED).await;
        for (index, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer.encode(prompt).expect("re-encode served prompt");
            assert_eq!(
                tokens.len(),
                ISL,
                "corpus {corpus} prompt {index} served {} tokens, expected exactly {ISL}: \
                 {prompt:?}",
                tokens.len()
            );
        }
        per_corpus.push((corpus, prompts));
    }

    for (left_index, (left_name, left)) in per_corpus.iter().enumerate() {
        for (right_name, right) in per_corpus.iter().skip(left_index + 1) {
            let left_set: HashSet<&String> = left.iter().collect();
            let shared = right.iter().filter(|p| left_set.contains(p)).count();
            assert_eq!(
                shared, 0,
                "corpora {left_name} and {right_name} served {shared} identical prompts; \
                 the selector is not reaching the generator"
            );
        }
    }
}

/// A named corpus is still seeded content: the same seed must reproduce the same
/// prompts, and a different seed must not.
#[tokio::test]
async fn prompt_corpus_content_is_seed_reproducible() {
    if cfg!(target_os = "macos") {
        return;
    }

    let first = served_prompts("coding", SEED).await;
    let repeat = served_prompts("coding", SEED).await;
    assert_eq!(
        first, repeat,
        "the same seed must reproduce the same coding-corpus prompts"
    );

    let reseeded = served_prompts("coding", SEED + 1).await;
    assert_ne!(
        first, reseeded,
        "a different seed must not reproduce the same coding-corpus prompts"
    );
}
