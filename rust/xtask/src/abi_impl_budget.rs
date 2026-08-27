// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measures implementation co-resident with host/plugin ABI-facing types.

use anyhow::Result;
use serde::Serialize;

use crate::abi_closure::{Seeds, compute, workspace_root};

/// Maximum accepted fraction of implementation lines in ABI-contributing files.
pub const MAX_IMPL_RATIO: f64 = 0.50;

/// Type and implementation line counts for the current ABI closure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Measurement {
    /// Lines occupied by ABI-facing type definitions.
    pub type_lines: usize,
    /// Other lines in files containing ABI-facing type definitions.
    pub impl_lines: usize,
    /// Fraction of contributing-file lines that are implementation.
    pub ratio: f64,
}

/// Measure implementation co-resident with ABI-facing type definitions.
pub fn measure() -> Result<Measurement> {
    let crate_root = workspace_root().join("xtask");
    let seeds = Seeds::load(crate_root.join("abi-seeds.toml"))?;
    let closure = compute(&seeds)?;
    let impl_lines = closure.file_lines.saturating_sub(closure.type_lines);
    let total_lines = closure.type_lines + impl_lines;
    let ratio = if total_lines == 0 {
        0.0
    } else {
        impl_lines as f64 / total_lines as f64
    };
    Ok(Measurement {
        type_lines: closure.type_lines,
        impl_lines,
        ratio,
    })
}
