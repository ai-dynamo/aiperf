// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, DeclaredArtifactTransfer, RegradeRequest, VerifierResult,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn verifier_json_rejects_empty_evidence_and_blank_regrade_metric() {
    assert!(serde_json::from_str::<VerifierResult>(
        r#"{
            "attempt":"attempt-1",
            "verifier":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "evidence":[],
            "reward":{"metrics":{"reward":1.0}},
            "rationale":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }"#,
    )
    .is_err());
    assert!(serde_json::from_str::<RegradeRequest>(
        r#"{
            "previous":{
                "attempt":"attempt-1","version":0,
                "evaluator":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "evidence":["blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                "metric":"reward","value":1.0,
                "rationale":"blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "predecessor":null
            },
            "result":{
                "attempt":"attempt-1",
                "verifier":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "evidence":["blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                "reward":{"metrics":{"reward":1.0}},
                "rationale":"blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "metric":" "
        }"#,
    )
    .is_err());
}

#[test]
fn artifact_aliases_are_rejected_before_verifier_materialization() {
    assert!(DeclaredArtifactTransfer::new(vec![
        ("/results/patch.diff", digest('a')),
        ("/results//patch.diff", digest('b')),
    ])
    .is_err());
}
