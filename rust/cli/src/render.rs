// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Terminal presentation for the native flow.
//!
//! The runner reserves stdout for one JSON line and writes the human summary to
//! `profile_export_console.txt` on disk (never stdout). The Python frontend used
//! to echo that table; on the pure-Rust path `aiperf-cli` echoes it itself so a
//! user sees results without opening a file.

use std::path::Path;

/// Print the runner's console summary (the sibling `profile_export_console.txt`
/// of the given `native-v2.json` report path) to stdout. Best-effort: silently
/// does nothing when the file is absent (e.g. DynoSim/sketch paths).
pub fn print_console_summary(report_path: &str) {
    let report = Path::new(report_path);
    let Some(dir) = report.parent() else {
        return;
    };
    let console = dir.join("profile_export_console.txt");
    if let Ok(text) = std::fs::read_to_string(&console) {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    }
}
