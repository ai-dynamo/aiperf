// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared typed Config-v2 model for AIPerf.
//!
//! This module holds the typed benchmark domain object and its protocol-v2 wire
//! projection, shared by the runtime and the CLI. It exposes a schema-version
//! anchor, [`schema_version`], alongside the [`model`] submodules.

pub mod model;
pub mod validate;

/// Returns the Config-v2 schema version string this module targets.
///
/// Placeholder anchor; superseded when the unified `AiperfConfig` model is
/// introduced.
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
