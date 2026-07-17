// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-parity of [`HuggingFaceTokenizer`] against a Python `transformers`
//! reference over an adversarial input battery (encode + chat-template).
//!
//! `tools/parity/dump_tokenizer_parity.py` defines the per-model tokenizer
//! directories and reference IDs. The test requires locally cached model files
//! and is `#[ignore]` by default. Run it with:
//!
//! ```text
//! HF_HUB_OFFLINE=1 python3 tools/parity/dump_tokenizer_parity.py /tmp/tok_parity.json
//! cargo test -p aiperf-runtime --test tokenizer_parity -- --ignored --nocapture
//! ```
//!
//! Override the golden path with `TOKENIZER_PARITY_GOLDEN`.

use std::path::Path;

use aiperf_runtime::dataset::{HuggingFaceTokenizer, TextTokenizer};
use serde_json::Value;

fn ids_of(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// First index where two id streams diverge, with a small context window.
fn first_divergence(want: &[u32], got: &[u32]) -> String {
    let n = want.len().min(got.len());
    let at = (0..n).find(|&i| want[i] != got[i]).unwrap_or(n);
    let lo = at.saturating_sub(3);
    format!(
        "diverge@{at} (len want={} got={}) want[..]={:?} got[..]={:?}",
        want.len(),
        got.len(),
        &want[lo..(at + 4).min(want.len())],
        &got[lo..(at + 4).min(got.len())],
    )
}

#[test]
#[ignore = "needs tools/parity/dump_tokenizer_parity.py output + locally cached model files"]
fn tokenizer_matches_python_reference() {
    let path = std::env::var("TOKENIZER_PARITY_GOLDEN")
        .unwrap_or_else(|_| "/tmp/tok_parity.json".to_string());
    let golden: Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {path:?}: {e}")),
    )
    .unwrap();

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (repo, entry) in golden.as_object().unwrap() {
        let dir = entry["dir"].as_str().unwrap();
        assert!(
            Path::new(dir).join("tokenizer.json").is_file(),
            "{repo}: missing {dir}/tokenizer.json (re-run the Python generator)"
        );
        let tok = HuggingFaceTokenizer::from_directory(dir)
            .unwrap_or_else(|e| panic!("load {repo} from {dir}: {e}"));

        for case in entry["encode"].as_array().unwrap() {
            total += 1;
            let name = case["name"].as_str().unwrap();
            let want = ids_of(&case["ids"]);
            let got = tok.encode(case["text"].as_str().unwrap()).unwrap();
            if got != want {
                failures.push(format!(
                    "encode {repo}/{name}: {}",
                    first_divergence(&want, &got)
                ));
            }
        }

        for case in entry["chat"].as_array().unwrap() {
            total += 1;
            let name = case["name"].as_str().unwrap();
            let messages: Vec<Value> = case["messages"].as_array().unwrap().clone();
            let add_gen = case["add_generation_prompt"].as_bool().unwrap();
            let want = ids_of(&case["ids"]);
            match tok.apply_chat_template(&messages, add_gen).unwrap() {
                Some(got) if got == want => {}
                Some(got) => {
                    failures.push(format!(
                        "chat {repo}/{name}: {}",
                        first_divergence(&want, &got)
                    ));
                }
                None => failures.push(format!(
                    "chat {repo}/{name}: renderer produced nothing (Python had {} ids)",
                    want.len()
                )),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{}/{} parity cases failed:\n{}",
        failures.len(),
        total,
        failures.join("\n")
    );
    eprintln!("tokenizer parity: {total}/{total} byte-identical vs Python");
}
