// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Enforces clock-only time access in the HTTP transport.

use std::fs;
use std::path::Path;

const FORBIDDEN: &[&str] = &[
    "Instant::now",
    "SystemTime::now",
    "tokio::time::sleep",
    "tokio::time::timeout",
    "tokio::time::interval",
    "tokio::time::Instant",
];

fn scan(dir: &Path, hits: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            scan(&path, hits);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = fs::read_to_string(&path).unwrap();
            for line in src.lines() {
                let code = line.split("//").next().unwrap_or("");
                for pat in FORBIDDEN {
                    if code.contains(pat) {
                        hits.push(format!("{}: {}", path.display(), line.trim()));
                    }
                }
            }
        }
    }
}

#[test]
fn no_direct_time_access_in_src() {
    // Other runtime modules legitimately use wall-clock APIs, so scope this
    // constraint to the HTTP transport.
    let mut hits = Vec::new();
    scan(Path::new("src/transport::http"), &mut hits);
    assert!(
        hits.is_empty(),
        "direct time access found (use Clock):\n{}",
        hits.join("\n")
    );
}
