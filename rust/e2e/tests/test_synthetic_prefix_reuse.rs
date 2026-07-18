// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::collections::HashMap;

use aiperf_runtime::dataset::TextTokenizer;
use aiperf_runtime::dataset::TiktokenTokenizer;

const REQUEST_COUNT: u32 = 12;
const ISL: usize = 64;
// `--prefix-reuse-ratio 0.5` reserves the first half of every reusing prompt as
// the shared prefix.
const SHARED_PREFIX_TOKENS: usize = ISL / 2;

/// A synthetic run with a direct prefix-reuse target must serve prompts whose
/// warm fraction share a byte-identical leading token run while every prompt
/// still hits the exact input length.
#[tokio::test]
async fn synthetic_prefix_reuse_shares_identical_served_prefixes() {
    if cfg!(target_os = "macos") {
        return;
    }

    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model openai/gpt-oss-120b \
         --tokenizer o200k_base \
         --url {} \
         --endpoint-type chat \
         --synthetic-input-tokens-mean {ISL} \
         --synthetic-input-tokens-stddev 0 \
         --prefix-reuse-fraction 0.5 \
         --prefix-reuse-ratio 0.5 \
         --random-seed 1234 \
         --request-count {REQUEST_COUNT} \
         --concurrency 1 \
         --export-level raw \
         --ui simple",
        h.mock.url
    ));

    let raw_records = r.artifacts.raw_records();
    assert_eq!(
        raw_records.len(),
        REQUEST_COUNT as usize,
        "expected {REQUEST_COUNT} raw records, got {}\nstdout:\n{}\nstderr:\n{}",
        raw_records.len(),
        r.stdout,
        r.stderr,
    );

    let tokenizer = TiktokenTokenizer::builtin();
    // Group served prompts by their leading shared-prefix token run: warm prompts
    // collapse onto one identical prefix, cold prompts almost surely differ.
    let mut prefix_groups: HashMap<Vec<u32>, usize> = HashMap::new();
    for (index, record) in raw_records.iter().enumerate() {
        let content = record["payload"]["messages"][0]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("record {index} has no user message content: {record}"));
        let tokens = tokenizer.encode(content).expect("re-encode served prompt");
        assert_eq!(
            tokens.len(),
            ISL,
            "record {index} served {} tokens, expected exactly {ISL}",
            tokens.len()
        );
        let prefix = tokens[..SHARED_PREFIX_TOKENS].to_vec();
        *prefix_groups.entry(prefix).or_default() += 1;
    }

    let warm = prefix_groups.values().copied().max().unwrap();
    assert!(
        warm >= 2,
        "expected a reused shared prefix across multiple prompts, got groups {prefix_groups:?}"
    );
    assert!(
        prefix_groups.len() >= 2,
        "expected some cold prompts with distinct prefixes, got groups {prefix_groups:?}"
    );
    assert!(
        warm < REQUEST_COUNT as usize,
        "expected a cold fraction, but every prompt shared one prefix"
    );
}
