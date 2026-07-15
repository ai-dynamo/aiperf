// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of the native `BenchmarkRun` against Python golden vectors.
//!
//! The native type is both the domain object and the wire request. This suite
//! asserts two things per golden:
//!
//! 1. The unchanged `aiperf-runner` input type (`RunnerEnvelopeV2`) accepts the
//!    golden bytes — structural validity against the real consumer.
//! 2. Each already-ported `cfg` section round-trips byte-exact: deserializing
//!    the golden through the native `BenchmarkRun` (which drops not-yet-ported
//!    keys) and re-serializing reproduces that section's subtree exactly.
//!
//! As sections land, add their key to `PORTED_CFG_SECTIONS`. A section not in
//! the list is intentionally dropped by the native type and not asserted yet.

use aiperf::runner_protocol::protocol_v2::RunnerEnvelopeV2;
use aiperf_cli::model::BenchmarkRun;

/// `cfg` sections the native type currently models; asserted for byte-exact
/// round-trip. Extend as each section is ported.
const PORTED_CFG_SECTIONS: &[&str] = &[
    "endpoint",
    "models",
    "tokenizer",
    "transport",
    "runtime",
    "metrics",
    "artifacts",
    "datasets",
    "phases",
];

/// Run-level (non-`cfg`) fields the native type currently models byte-exact.
const PORTED_RUN_FIELDS: &[&str] = &["resolved"];

/// Load a golden request JSON (paths are relative to the crate dir `rust/cli`).
fn load_golden(name: &str) -> serde_json::Value {
    let path = format!("../../tools/parity/golden/{name}.request.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden is valid JSON")
}

/// Re-serialize the golden `run` through the native `BenchmarkRun` type.
fn runner_view(golden: &serde_json::Value) -> serde_json::Value {
    let run: BenchmarkRun =
        serde_json::from_value(golden["run"].clone()).expect("golden run deserializes as native");
    serde_json::to_value(&run).expect("native run serializes")
}

#[test]
fn golden_minimal_chat_is_valid_runner_input() {
    let golden = load_golden("minimal_chat");
    // The unchanged runner input type must accept the golden bytes.
    let _: RunnerEnvelopeV2 =
        serde_json::from_value(golden.clone()).expect("golden is valid RunnerEnvelopeV2");
}

#[test]
fn golden_minimal_chat_ported_sections_roundtrip() {
    let golden = load_golden("minimal_chat");
    let view = runner_view(&golden);
    for section in PORTED_CFG_SECTIONS {
        let want = &golden["run"]["cfg"][section];
        // `runner_view` serializes the run object directly (not the envelope).
        let got = &view["cfg"][section];
        assert!(!want.is_null(), "golden is missing cfg.{section}");
        assert_eq!(
            got, want,
            "cfg.{section} diverges from golden\n got: {got:#}\nwant: {want:#}"
        );
    }
    for field in PORTED_RUN_FIELDS {
        let want = &golden["run"][field];
        let got = &view[field];
        assert_eq!(
            got, want,
            "run.{field} diverges from golden\n got: {got:#}\nwant: {want:#}"
        );
    }
}
