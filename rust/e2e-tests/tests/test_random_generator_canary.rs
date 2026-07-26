// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use serde_json::Value;

const CANARY_SEED: u32 = 42;

const REFERENCE_JSON: &str = include_str!("assets/canary_reference_inputs.json");

fn assert_inputs_match(reference: &Value, current: &Value) {
    let ref_sessions = reference["data"]
        .as_array()
        .expect("reference missing 'data' array");
    let cur_sessions = current["data"]
        .as_array()
        .expect("current missing 'data' array");

    assert_eq!(
        ref_sessions.len(),
        cur_sessions.len(),
        "Session count mismatch: expected {}, got {}",
        ref_sessions.len(),
        cur_sessions.len()
    );

    for (i, (ref_session, cur_session)) in ref_sessions.iter().zip(cur_sessions.iter()).enumerate()
    {
        assert_eq!(
            ref_session["payloads"], cur_session["payloads"],
            "Session {i}: payload mismatch.\n\
             Reference: {}\n\
             Current:   {}",
            ref_session["payloads"], cur_session["payloads"]
        );
    }
}

#[tokio::test]
async fn test_random_generator_canary() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model-names \"openai/gpt-oss-20b,openai/gpt-oss-120b\" \
         --model-selection-strategy random \
         --url {} \
         --endpoint-type chat \
         --request-count 20 \
         --concurrency 2 \
         --random-seed {CANARY_SEED} \
         --prompt-input-tokens-mean 100 \
         --prompt-input-tokens-stddev 10 \
         --num-dataset-entries 20 \
         --prompt-output-tokens-mean 50 \
         --prompt-output-tokens-stddev 5 \
         --audio-length-mean 0.05 \
         --audio-length-stddev 0.01 \
         --workers-max 2 \
         --ui simple",
        h.mock.url
    ));

    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 20);

    let current = r.artifacts.inputs();
    assert!(!current.is_null(), "inputs.json should not be null");

    let reference: Value =
        serde_json::from_str(REFERENCE_JSON).expect("reference inputs JSON should parse");

    assert_inputs_match(&reference, &current);
}
