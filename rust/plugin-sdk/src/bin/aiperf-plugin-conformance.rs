// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-plugin-conformance` — run the plugin conformance suite against a
//! built cdylib artifact.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let artifact = match args.first() {
        Some(a) if a != "--help" && a != "-h" => PathBuf::from(a),
        _ => {
            println!("aiperf-plugin-conformance <ARTIFACT>");
            println!();
            println!("Run the conformance suite against a plugin cdylib.");
            println!("Exits 0 if all checks pass, 1 otherwise.");
            return;
        }
    };

    let report = match aiperf_plugin_sdk::conformance::run_conformance(&artifact) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    for check in &report.passed {
        println!("PASS  {check}");
    }
    for failure in &report.failed {
        println!("FAIL  {} — {}", failure.test_name, failure.reason);
    }

    if report.failed.is_empty() {
        println!("conformance: OK ({} checks)", report.passed.len());
    } else {
        println!(
            "conformance: FAIL ({} passed, {} failed)",
            report.passed.len(),
            report.failed.len()
        );
        std::process::exit(1);
    }
}
