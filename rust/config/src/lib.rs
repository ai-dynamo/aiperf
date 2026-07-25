// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared typed Config-v2 model for AIPerf.
//!
//! This crate is the leaf that later refactors build the unified Config-v2 wire
//! type onto. For now it exposes a single placeholder anchor, [`schema_version`],
//! so the crate has a testable symbol; it is removed once `AiperfConfig` lands.

pub mod model;

/// Returns the Config-v2 schema version string this crate targets.
///
/// Placeholder anchor for the leaf crate; superseded when the unified
/// `AiperfConfig` model is introduced.
pub fn schema_version() -> &'static str {
    "2.0"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_is_2_0() {
        assert_eq!(schema_version(), "2.0");
    }
}
