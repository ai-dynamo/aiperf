// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, EvalTaskRef, ImportDisposition, ImportReport, ModelIdentity,
    PolicyIdentity, RuntimeIdentity, TrialBudget, TrialSpec,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

fn trial(seed: u64) -> TrialSpec {
    TrialSpec::new(
        EvalTaskRef::new("task-1", digest('a')).unwrap(),
        AgentVariantRef::new("external-agent").unwrap(),
        ModelIdentity::new("provider", "model").unwrap(),
        seed,
        PolicyIdentity::new(digest('b')),
        TrialBudget::new(1.0, 2.0).unwrap(),
        digest('c'),
        digest('d'),
        RuntimeIdentity::new("runtime-v1").unwrap(),
    )
    .unwrap()
}

#[test]
fn resolved_trial_digest_changes_with_seed() {
    assert_eq!(trial(7).identity_digest(), trial(7).identity_digest());
    assert_ne!(trial(7).identity_digest(), trial(8).identity_digest());
}

#[test]
fn import_report_rejects_unknown_disposition() {
    let report = r#"{
        "source_digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "normalized_digest":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "disposition":"bridge"
    }"#;
    assert!(serde_json::from_str::<ImportReport>(report).is_err());
    assert_eq!(ImportDisposition::Unsupported.as_str(), "unsupported");
}
