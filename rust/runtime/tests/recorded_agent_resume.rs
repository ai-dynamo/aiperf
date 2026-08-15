// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Contract tests for persisted recorded-agent replay progress.

use std::collections::BTreeMap;

use aiperf_runtime::graph::driver::ReplayTaskIdentity;
use aiperf_runtime::graph::replay::{
    CompletedReplayTask, ReplayCheckpoint, ReplayResumeError, ReplayRunIdentity,
};

fn task() -> ReplayTaskIdentity {
    ReplayTaskIdentity {
        adapter: "pinchbench".into(),
        family: "pinchbench-openclaw".into(),
        task_id: "task_meeting_council_budget".into(),
        primary_role: None,
    }
}

#[test]
fn resume_requires_same_root_manifest_recordings_profiles_and_namespace() {
    let run = ReplayRunIdentity::for_checkpoint(
        "opaque-run-id",
        "replay-root-digest",
        "manifest-digest",
        BTreeMap::from([(
            "pinchbench:task_meeting_council_budget".into(),
            "recording-a".into(),
        )]),
        BTreeMap::from([(
            "pinchbench:task_meeting_council_budget".into(),
            "profile-a".into(),
        )]),
        "secret cache namespace",
    );
    let checkpoint = ReplayCheckpoint::new(run.clone(), "manifest-digest").with_completed(
        task(),
        CompletedReplayTask::successful(0, "recording-a", "profile-a", "env-a", 4),
    );

    checkpoint
        .validate_resume(&run)
        .expect("identical replay identity resumes");
    for changed in [
        ReplayRunIdentity::for_checkpoint(
            "opaque-run-id",
            "other-root",
            "manifest-digest",
            run.recording_digests().clone(),
            run.request_profile_digests().clone(),
            "secret cache namespace",
        ),
        ReplayRunIdentity::for_checkpoint(
            "opaque-run-id",
            "replay-root-digest",
            "other-manifest",
            run.recording_digests().clone(),
            run.request_profile_digests().clone(),
            "secret cache namespace",
        ),
        ReplayRunIdentity::for_checkpoint(
            "opaque-run-id",
            "replay-root-digest",
            "manifest-digest",
            BTreeMap::from([(
                "pinchbench:task_meeting_council_budget".into(),
                "other-recording".into(),
            )]),
            run.request_profile_digests().clone(),
            "secret cache namespace",
        ),
        ReplayRunIdentity::for_checkpoint(
            "opaque-run-id",
            "replay-root-digest",
            "manifest-digest",
            run.recording_digests().clone(),
            BTreeMap::from([(
                "pinchbench:task_meeting_council_budget".into(),
                "other-profile".into(),
            )]),
            "secret cache namespace",
        ),
        ReplayRunIdentity::for_checkpoint(
            "opaque-run-id",
            "replay-root-digest",
            "manifest-digest",
            run.recording_digests().clone(),
            run.request_profile_digests().clone(),
            "other namespace",
        ),
    ] {
        assert!(matches!(
            checkpoint.validate_resume(&changed),
            Err(ReplayResumeError::IdentityMismatch(_))
        ));
    }
}

#[test]
fn resume_skips_only_verified_successful_exact_tasks() {
    let run = ReplayRunIdentity::for_checkpoint(
        "opaque-run-id",
        "root",
        "manifest",
        BTreeMap::from([(
            "pinchbench:task_meeting_council_budget".into(),
            "recording".into(),
        )]),
        BTreeMap::from([(
            "pinchbench:task_meeting_council_budget".into(),
            "profile".into(),
        )]),
        "secret",
    );
    let successful = CompletedReplayTask::successful(0, "recording", "profile", "env", 4);
    let partial = CompletedReplayTask::partial(0, "recording", "profile", "env", 3);
    assert!(successful.is_verified_success_for(&task(), 0, "recording", "profile", 4));
    assert!(!partial.is_verified_success_for(&task(), 0, "recording", "profile", 4));
    let checkpoint = ReplayCheckpoint::new(run, "manifest").with_completed(task(), successful);
    assert!(checkpoint.should_skip(&task(), 0, "recording", "profile", 4));
}
