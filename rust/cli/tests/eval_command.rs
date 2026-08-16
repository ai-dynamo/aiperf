// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

#[test]
fn native_eval_command_runs_a_local_harbor_package() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        package_path.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}
