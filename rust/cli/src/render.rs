// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Terminal presentation.
//!
//! Execution reserves child stdout for one JSON line and writes the human
//! summary to `profile_export_console.txt`; the parent prints that file.

use std::path::Path;

/// Print the report's sibling `profile_export_console.txt` when present.
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
