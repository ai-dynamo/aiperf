// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-plugin-inspect` — inspect the exported symbol table and embedded
//! build record of a plugin cdylib.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let artifact = match args.first() {
        Some(a) if a != "--help" && a != "-h" => PathBuf::from(a),
        _ => {
            println!("aiperf-plugin-inspect <ARTIFACT>");
            println!();
            println!("Inspect the exported symbols and embedded build record of a plugin cdylib.");
            return;
        }
    };

    let report = match aiperf_plugin_sdk::inspect::inspect_artifact(&artifact) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if report.symbols_ok {
        println!("symbols: OK");
    } else {
        println!("symbols: FAIL");
        for sym in &report.missing_symbols {
            println!("  missing: {sym}");
        }
        for sym in &report.extra_symbols {
            println!("  unexpected: {sym}");
        }
    }

    match &report.embedded_record {
        Some(rec) => {
            println!("embedded record: present");
            println!("  canonical_digest: {}", rec.canonical_digest);
        }
        None => {
            println!("embedded record: absent");
        }
    }

    if !report.symbols_ok {
        std::process::exit(1);
    }
}
