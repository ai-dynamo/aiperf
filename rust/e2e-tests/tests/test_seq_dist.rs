// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_sequence_length_distribution() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model Qwen/Qwen3-0.6B --endpoint-type chat --endpoint /v1/chat/completions \
             --streaming --url {} \
             --sequence-distribution \"64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10\" \
             --ui simple",
            h.mock.url
        ),
        120,
    );
    assert!(r.success());

    for request in r.artifacts.jsonl() {
        let isl = &request["metrics"]["input_sequence_length"];
        let osl = &request["metrics"]["output_sequence_length"];
        assert!(!isl.is_null());
        assert!(!osl.is_null());
        assert!(isl["value"].as_f64().unwrap() > 0.0);
        assert!(osl["value"].as_f64().unwrap() > 0.0);
    }
}
