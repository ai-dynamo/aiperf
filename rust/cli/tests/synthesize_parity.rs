// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of native `aiperf synthesize agentic-code` vs the Python
//! oracle. The committed goldens under `tools/parity/synthesize/` were produced
//! by `python -m aiperf synthesize agentic-code --seed S --num-sessions N`; this
//! test runs the native binary into a temp dir and asserts the generated
//! `dataset.jsonl` is byte-identical to the golden.
//!
//! Regenerate goldens with, e.g.:
//!   python -m aiperf synthesize agentic-code --num-sessions 50 --seed 42 --output OUT
//!   cp OUT/default_50s_seed42_*/dataset.jsonl tools/parity/synthesize/seed42_n50.jsonl

use std::process::Command;

fn bin() -> String {
    format!("{}/../../target/debug/aiperf", env!("CARGO_MANIFEST_DIR"))
}

fn golden(name: &str) -> String {
    format!(
        "{}/../../tools/parity/synthesize/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Run the native synthesizer and return the generated `dataset.jsonl` bytes.
fn run_native(seed: u64, num_sessions: usize) -> Vec<u8> {
    let out_dir = std::env::temp_dir().join(format!(
        "aiperf_synth_test_{seed}_{num_sessions}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&out_dir);
    let status = Command::new(bin())
        .args([
            "synthesize",
            "agentic-code",
            "--num-sessions",
            &num_sessions.to_string(),
            "--seed",
            &seed.to_string(),
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("spawn native aiperf");
    assert!(status.success(), "native synthesize failed");

    // The run dir carries a non-deterministic timestamp; find the single child.
    let run_dir = std::fs::read_dir(&out_dir)
        .expect("read out dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("run dir");
    let jsonl = run_dir.join("dataset.jsonl");
    let bytes = std::fs::read(&jsonl).expect("read dataset.jsonl");
    let _ = std::fs::remove_dir_all(&out_dir);
    bytes
}

fn assert_byte_exact(seed: u64, num_sessions: usize, golden_name: &str) {
    let got = run_native(seed, num_sessions);
    let want = std::fs::read(golden(golden_name)).expect("read golden");
    assert_eq!(
        got.len(),
        want.len(),
        "seed={seed} n={num_sessions}: byte length differs ({} vs golden {})",
        got.len(),
        want.len()
    );
    if got != want {
        // Locate the first differing line for a useful message.
        let got_s = String::from_utf8_lossy(&got);
        let want_s = String::from_utf8_lossy(&want);
        for (i, (a, b)) in got_s.lines().zip(want_s.lines()).enumerate() {
            assert_eq!(a, b, "seed={seed} n={num_sessions}: line {} differs", i + 1);
        }
        panic!("seed={seed} n={num_sessions}: dataset.jsonl differs from golden");
    }
}

#[test]
fn synthesize_seed42_n50_byte_exact() {
    assert_byte_exact(42, 50, "seed42_n50.jsonl");
}

#[test]
fn synthesize_seed7_n50_byte_exact() {
    assert_byte_exact(7, 50, "seed7_n50.jsonl");
}

#[test]
fn synthesize_seed42_n5_byte_exact() {
    assert_byte_exact(42, 5, "seed42_n5.jsonl");
}
