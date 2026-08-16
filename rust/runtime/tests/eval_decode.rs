// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{RewardDocument, ScoreVersion};

#[test]
fn reward_and_score_json_reject_constructor_invalid_values() {
    assert!(serde_json::from_str::<RewardDocument>(r#"{"metrics":{}}"#).is_err());
    assert!(serde_json::from_str::<RewardDocument>(r#"{"metrics":{"":1.0}}"#).is_err());
    assert!(
        serde_json::from_str::<ScoreVersion>(
            r#"{
            "attempt":"attempt-1",
            "version":0,
            "evaluator":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "evidence":[],
            "metric":"reward",
            "value":1.0,
            "rationale":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "predecessor":null
        }"#,
        )
        .is_err()
    );
}
