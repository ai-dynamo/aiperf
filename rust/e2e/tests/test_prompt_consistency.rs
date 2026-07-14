// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for random prompt generation consistency.
//!
//! This test ensures that randomly generated prompt texts remain consistent across
//! different configuration changes when using the same seed. The goal is to verify
//! that the random text generation is decoupled from other configuration parameters.

mod common;
use common::*;

const CONSISTENCY_SEED: u32 = 12345;
const WORKERS_MAX: u32 = 1;

/// Extract all prompt text content from payloads in `inputs.json`.
///
/// Returns the list of all text content from prompts in order.
fn extract_prompt_texts(inputs: &serde_json::Value) -> Vec<String> {
    let mut texts = Vec::new();
    let sessions = match inputs.get("data").and_then(|d| d.as_array()) {
        Some(s) => s,
        None => return texts,
    };
    for session in sessions {
        let payloads = match session.get("payloads").and_then(|p| p.as_array()) {
            Some(p) => p,
            None => continue,
        };
        for payload in payloads {
            if let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) {
                // Chat format
                for message in messages {
                    match message.get("content") {
                        Some(serde_json::Value::String(s)) => texts.push(s.clone()),
                        Some(serde_json::Value::Array(items)) => {
                            // Multimodal content
                            for item in items {
                                if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                                    if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                        texts.push(t.to_string());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            } else if let Some(prompt) = payload.get("prompt").and_then(|p| p.as_str()) {
                // Completions format
                texts.push(prompt.to_string());
            }
        }
    }
    texts
}

/// Verify prompt texts are identical when adding audio/images.
///
/// Adding multimodal content (audio/images) should not affect the randomly
/// generated text portions of prompts.
#[tokio::test]
async fn test_prompt_consistency_with_multimodal_additions() {
    // Run without multimodal content
    let h_text_only = AIPerfHarness::new().await;
    let result_text_only = h_text_only.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 10 \
         --concurrency 2 \
         --random-seed {CONSISTENCY_SEED} \
         --prompt-input-tokens-mean 90 \
         --prompt-input-tokens-stddev 8 \
         --num-dataset-entries 10 \
         --workers-max {WORKERS_MAX} \
         --ui simple",
        h_text_only.mock.url
    ));
    assert!(result_text_only.success());

    // Run with audio and images
    let h_multimodal = AIPerfHarness::new().await;
    let result_multimodal = h_multimodal.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 10 \
         --concurrency 2 \
         --random-seed {CONSISTENCY_SEED} \
         --prompt-input-tokens-mean 90 \
         --prompt-input-tokens-stddev 8 \
         --num-dataset-entries 10 \
         --image-width-mean 128 \
         --image-height-mean 128 \
         --audio-length-mean 0.1 \
         --audio-length-stddev 0.02 \
         --workers-max {WORKERS_MAX} \
         --ui simple",
        h_multimodal.mock.url
    ));
    assert!(result_multimodal.success());

    let texts_text_only = extract_prompt_texts(&result_text_only.artifacts.inputs());
    let texts_multimodal = extract_prompt_texts(&result_multimodal.artifacts.inputs());

    assert_eq!(
        texts_text_only.len(),
        texts_multimodal.len(),
        "Prompt count should be identical"
    );
    assert_eq!(
        texts_text_only, texts_multimodal,
        "Prompt texts should be identical even when audio/images are added"
    );
}
