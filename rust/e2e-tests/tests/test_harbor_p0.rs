// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, DeclaredArtifactTransfer, RegradeRequest, RewardDocument,
    ScoreVersion, VerifierResult, regrade,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn native_harbor_p0_regrades_declared_artifact_evidence_without_harbor_runtime() {
    let attempt = AttemptId::new("p0-attempt").unwrap();
    let transfer = DeclaredArtifactTransfer::new(vec![("/results/patch.diff", digest('a'))]).unwrap();
    let original = ScoreVersion::initial(
        attempt.clone(), digest('b'), vec![digest('c')], "reward", 0.0, digest('d'),
    ).unwrap();
    let result = VerifierResult::new(
        attempt, digest('e'), transfer.artifacts().iter().map(|(_, digest)| digest.clone()).collect(),
        RewardDocument::parse(Some(br#"{"reward":1.0}"#), None).unwrap(), digest('f'),
    ).unwrap();

    let score = regrade(RegradeRequest::new(original, result, "reward").unwrap()).unwrap();

    assert_eq!(score.value, 1.0);
    assert_eq!(score.evidence, vec![digest('a')]);
}
