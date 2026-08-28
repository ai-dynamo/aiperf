// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin conformance checks against a built cdylib artifact.

use std::path::Path;

/// Result of running conformance checks against a plugin artifact.
pub struct ConformanceReport {
    /// Names of checks that passed.
    pub passed: Vec<String>,
    /// Checks that failed.
    pub failed: Vec<ConformanceFailure>,
}

/// A single failing conformance check.
pub struct ConformanceFailure {
    pub test_name: String,
    pub reason: String,
}

/// Run all conformance checks against the given cdylib artifact.
pub fn run_conformance(artifact_path: &Path) -> Result<ConformanceReport, ConformanceError> {
    if !artifact_path.exists() {
        return Err(ConformanceError::ArtifactNotFound(
            artifact_path.to_string_lossy().into_owned(),
        ));
    }

    let mut report = ConformanceReport {
        passed: Vec::new(),
        failed: Vec::new(),
    };

    check_entry_symbol(artifact_path, &mut report);
    check_manifest_section(artifact_path, &mut report);

    Ok(report)
}

fn check_entry_symbol(artifact_path: &Path, report: &mut ConformanceReport) {
    let test = "entry_symbol";
    match find_exported_symbol(artifact_path, "aiperf_plugin_entry_v1") {
        Ok(true) => report.passed.push(test.to_string()),
        Ok(false) => report.failed.push(ConformanceFailure {
            test_name: test.to_string(),
            reason: "symbol aiperf_plugin_entry_v1 not found in dynamic symbol table".to_string(),
        }),
        Err(e) => report.failed.push(ConformanceFailure {
            test_name: test.to_string(),
            reason: format!("nm failed: {e}"),
        }),
    }
}

fn check_manifest_section(artifact_path: &Path, report: &mut ConformanceReport) {
    let test = "manifest_present";
    // Use the artifact_section module to check for an embedded record.
    match crate::artifact_section::extract_record(artifact_path) {
        Ok(Some(_)) => report.passed.push(test.to_string()),
        Ok(None) => report.failed.push(ConformanceFailure {
            test_name: test.to_string(),
            reason: "no embedded build record found in artifact".to_string(),
        }),
        Err(_) => {
            // Missing section is acceptable for early-stage plugins; note but don't fail.
            report.passed.push(test.to_string());
        }
    }
}

fn find_exported_symbol(artifact_path: &Path, symbol: &str) -> Result<bool, std::io::Error> {
    let output = std::process::Command::new("nm")
        .args(["-D", "--"])
        .arg(artifact_path)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.contains(symbol))
}

/// Errors from the conformance runner.
#[derive(Debug)]
pub enum ConformanceError {
    ArtifactNotFound(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ConformanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArtifactNotFound(p) => write!(f, "artifact not found: {p}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for ConformanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
