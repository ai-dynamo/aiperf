// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of native `aiperf speed-bench-report` CSV output vs the
//! Python oracle (`tools/parity/speed_bench/golden.csv`).

use std::process::Command;

#[test]
fn speed_bench_csv_matches_python_oracle() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let bin = format!("{dir}/../../target/debug/aiperf");
    let artifacts = format!("{dir}/../../tools/parity/speed_bench/artifacts/");
    let golden = format!("{dir}/../../tools/parity/speed_bench/golden.csv");
    let out = std::env::temp_dir().join("aiperf_sb_parity.csv");

    let status = Command::new(&bin)
        .args([
            "speed-bench-report",
            &artifacts,
            "--format",
            "csv",
            "--output",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("run");
    assert!(status.success());

    let got = std::fs::read(&out).expect("native csv");
    let want = std::fs::read(&golden).expect("golden csv");
    assert_eq!(
        got, want,
        "speed-bench-report CSV diverges from the Python oracle"
    );
}
