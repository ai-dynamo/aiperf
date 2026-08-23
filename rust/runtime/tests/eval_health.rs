// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "engine")]

use aiperf_runtime::eval::{ArtifactDigest, TaskHealthRecord, TaskVerdict};

#[test]
fn broken_task_requires_evidence_before_quarantine() {
    let digest = ArtifactDigest::parse(format!("blake3:{}", "a".repeat(64))).unwrap();
    assert!(TaskHealthRecord::new(TaskVerdict::Broken, vec![]).is_err());
    assert_eq!(
        TaskHealthRecord::new(TaskVerdict::Broken, vec![digest])
            .unwrap()
            .verdict,
        TaskVerdict::Broken
    );
}
