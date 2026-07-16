// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use aiperf_runtime::dataset::TextTokenizer;
use aiperf_runtime::dataset::TiktokenTokenizer;
use aiperf_runtime::dataset::corpus::tokenize_sonnet_corpus;

// End-to-end proof that the native synthetic prompt synthesizer serves
// Shakespeare ("sonnet") corpus text of the requested length.
//
// A full `python -m aiperf profile` run drives the Rust runner against the
// in-process mock server with `--export-level raw`, which dumps every request's
// exact wire payload to `profile_export_raw.jsonl`. We then re-tokenize each
// served prompt with the same built-in `o200k_base` tokenizer and assert:
//   * the prompt is exactly the requested input-sequence length, and
//   * its token window appears verbatim in the embedded Shakespeare corpus,
// proving the bytes on the wire were sampled from the sonnet corpus by the
// native `CorpusPromptGeneratorFactory`.

const REQUEST_COUNT: u32 = 5;
const ISL: usize = 128;

fn contains_window(corpus: &[u32], needle: &[u32]) -> bool {
    if needle.is_empty() || needle.len() > corpus.len() {
        return false;
    }
    corpus.windows(needle.len()).any(|window| window == needle)
}

#[tokio::test]
async fn synthetic_sonnet_prompts_are_served_end_to_end() {
    // Skip on macOS: the e2e subprocess harness is flaky on GitHub's macOS CI.
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

    // Same tokenizer as `--tokenizer o200k_base`; rebuild the embedded corpus so
    // provenance can be checked against the exact token stream the runner sees.
    let tokenizer = TiktokenTokenizer::builtin();
    let corpus = tokenize_sonnet_corpus(&tokenizer).expect("sonnet corpus tokenizes");

    for (index, record) in raw_records.iter().enumerate() {
        let content = record["payload"]["messages"][0]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("record {index} has no user message content: {record}"));

        // Real natural-language corpus text, not an empty or placeholder body.
        let alphabetic = content.chars().filter(|c| c.is_alphabetic()).count();
        assert!(
            alphabetic >= content.len() / 2,
            "record {index} content is not natural language: {content:?}"
        );

        // Exact requested input-sequence length under the run's tokenizer.
        let tokens = tokenizer.encode(content).expect("re-encode served prompt");
        assert_eq!(
            tokens.len(),
            ISL,
            "record {index} served {} tokens, expected exactly {ISL}",
            tokens.len()
        );

        // Provenance: a mid-prompt window is a verbatim slice of the sonnet
        // corpus (leading/trailing tokens can shift under decode/encode
        // whitespace normalization, so probe an interior window).
        let window = &tokens[2..2 + 16.min(tokens.len().saturating_sub(2))];
        assert!(
            contains_window(&corpus, window),
            "record {index} prompt window is not present in the sonnet corpus: {window:?}"
        );
    }
}
