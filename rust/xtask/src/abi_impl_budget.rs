// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measures and gates implementation co-resident with host/plugin ABI-facing types.

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::abi_closure::{Seeds, compute, workspace_root};

/// Maximum implementation lines in the live ABI closure after the authorized splits.
pub const MAX_GLOBAL_IMPL_LINES: usize = 33_417;
/// Maximum ABI-facing type count accepted by this task's baseline.
pub const MAX_ABI_TYPES: usize = 177;
/// Maximum ABI-contributing file count accepted by this task's baseline.
pub const MAX_ABI_FILES: usize = 56;
const MAX_BOUNDARY_IMPL_RATIO: f64 = 0.50;
const BOUNDARY_FILES: [&str; 4] = [
    "runtime/src/body_plan/model.rs",
    "runtime/src/multiturn/model.rs",
    "runtime/src/metrics_core/accumulator_model.rs",
    "runtime/src/scheduled/observe.rs",
];

/// Type and implementation lines in one boundary-only model file.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryFileMeasurement {
    /// Cargo-workspace-relative source path.
    pub file: String,
    /// Lines occupied by ABI-facing type definitions.
    pub type_lines: usize,
    /// Other lines in the boundary-only file.
    pub impl_lines: usize,
    /// Fraction of the file outside ABI-facing type definitions.
    pub ratio: f64,
}

/// Type and implementation line counts for the current ABI closure.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    /// Lines occupied by ABI-facing type definitions.
    pub type_lines: usize,
    /// Other lines in files containing ABI-facing type definitions.
    pub impl_lines: usize,
    /// Fraction of contributing-file lines that are implementation.
    pub ratio: f64,
    /// Number of ABI-facing types in the live closure.
    #[serde(skip)]
    pub abi_types: usize,
    /// Number of source files contributing ABI-facing types.
    #[serde(skip)]
    pub abi_files: usize,
    /// Measurements for the boundary-only files created by this task.
    #[serde(skip)]
    pub boundary_files: Vec<BoundaryFileMeasurement>,
}

/// Measure implementation co-resident with ABI-facing type definitions.
pub fn measure() -> Result<Measurement> {
    let workspace = workspace_root();
    let crate_root = workspace.join("xtask");
    let seeds = Seeds::load(crate_root.join("abi-seeds.toml"))?;
    let closure = compute(&seeds)?;
    let impl_lines = closure.file_lines.saturating_sub(closure.type_lines);
    let ratio = ratio(closure.type_lines, impl_lines);
    let boundary_files = BOUNDARY_FILES
        .iter()
        .map(|file| {
            let type_lines = closure
                .types
                .values()
                .filter(|entry| entry.file == *file)
                .map(|entry| entry.end.saturating_sub(entry.start) + 1)
                .sum();
            let file_lines = std::fs::read_to_string(workspace.join(file))
                .with_context(|| format!("reading ABI boundary model file {file}"))?
                .lines()
                .count();
            let impl_lines = file_lines.saturating_sub(type_lines);
            Ok(BoundaryFileMeasurement {
                file: (*file).to_owned(),
                type_lines,
                impl_lines,
                ratio: ratio(type_lines, impl_lines),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Measurement {
        type_lines: closure.type_lines,
        impl_lines,
        ratio,
        abi_types: closure.types.len(),
        abi_files: closure.files.len(),
        boundary_files,
    })
}

/// Reject regression from the authorized split and mixing in boundary-only files.
pub fn ensure_within_budget(measurement: &Measurement) -> Result<()> {
    if measurement.impl_lines > MAX_GLOBAL_IMPL_LINES {
        bail!(
            "ABI implementation grew from the committed maximum of {MAX_GLOBAL_IMPL_LINES} to {} lines",
            measurement.impl_lines
        );
    }
    if measurement.abi_types > MAX_ABI_TYPES {
        bail!(
            "ABI closure grew from {MAX_ABI_TYPES} to {} types",
            measurement.abi_types
        );
    }
    if measurement.abi_files > MAX_ABI_FILES {
        bail!(
            "ABI closure grew from {MAX_ABI_FILES} to {} files",
            measurement.abi_files
        );
    }
    for file in BOUNDARY_FILES {
        let measured = measurement
            .boundary_files
            .iter()
            .find(|measured| measured.file == file)
            .with_context(|| format!("missing ABI boundary model measurement for {file}"))?;
        if measured.type_lines == 0 {
            bail!("ABI boundary model file {file} contains no ABI-facing type definitions");
        }
        if measured.ratio >= MAX_BOUNDARY_IMPL_RATIO {
            bail!(
                "ABI boundary model file {file} is {:.0}% implementation ({} impl lines); maximum is {:.0}%",
                measured.ratio * 100.0,
                measured.impl_lines,
                MAX_BOUNDARY_IMPL_RATIO * 100.0
            );
        }
    }
    Ok(())
}

fn ratio(type_lines: usize, impl_lines: usize) -> f64 {
    let total_lines = type_lines + impl_lines;
    if total_lines == 0 {
        0.0
    } else {
        impl_lines as f64 / total_lines as f64
    }
}
