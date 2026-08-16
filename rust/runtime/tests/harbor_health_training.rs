// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind, TaskHealthRecord, TaskVerdict,
    TrajectoryExportManifest,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn broken_health_record_quarantines_a_task_without_quarantining_a_valid_task() {
    let valid = TaskHealthRecord::new(TaskVerdict::Valid, vec![digest('a')]).unwrap();
    let broken = TaskHealthRecord::new(TaskVerdict::Broken, vec![digest('b')]).unwrap();

    assert!(!valid.is_quarantined());
    assert!(broken.is_quarantined());
}

#[test]
fn health_record_rejects_duplicate_evidence_as_nonindependent_support() {
    let error = TaskHealthRecord::new(TaskVerdict::Broken, vec![digest('a'), digest('a')])
        .unwrap_err();

    assert_eq!(error.to_string(), "task health evidence must be unique");
}

#[test]
fn trajectory_export_revalidates_its_ordered_attempt_evidence() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let events = vec![
        EvidenceEvent::new(
            attempt.clone(),
            3,
            EvidenceKind::Agent,
            digest('a'),
            None,
        ),
        EvidenceEvent::new(
            attempt.clone(),
            5,
            EvidenceKind::Tool,
            digest('b'),
            Some(digest('a')),
        ),
    ];

    let manifest = TrajectoryExportManifest::from_events(attempt, &events).unwrap();

    assert!(manifest.validate_against(&events).is_ok());
    assert!(manifest.validate_against(&[events[1].clone(), events[0].clone()]).is_err());
}
