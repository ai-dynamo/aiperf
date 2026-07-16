// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of native `aiperf analyze-trace` JSON output vs the Python
//! `PrefixAnalyzer` oracle (`tools/parity/analyze_trace/golden.json`, produced by
//! `python -m aiperf analyze-trace ... --output-file`).

use std::process::Command;

#[test]
fn analyze_trace_json_matches_python_oracle() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let bin = format!("{dir}/../../target/debug/aiperf");
    let trace = format!("{dir}/../../tools/parity/analyze_trace/trace.jsonl");
    let golden = format!("{dir}/../../tools/parity/analyze_trace/golden.json");
    let out = std::env::temp_dir().join("aiperf_at_parity.json");

    let status = Command::new(&bin)
        .args([
            "analyze-trace",
            &trace,
            "--block-size",
            "512",
            "--output-file",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("run native analyze-trace");
    assert!(status.success(), "analyze-trace exited nonzero");

    let got = std::fs::read_to_string(&out).expect("read native output");
    let want = std::fs::read_to_string(&golden).expect("read golden");
    assert_eq!(
        got, want,
        "analyze-trace JSON diverges from the Python oracle"
    );
}
