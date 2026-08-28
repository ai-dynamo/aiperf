// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermetic build sandbox enforcement utilities.
//!
//! Verifies that a plugin build environment uses only the allowed
//! environment variables and does not access the network.

/// The environment variables a hermetic plugin build may observe.
pub const ALLOWED_ENV_VARS: &[&str] = &[
    "PATH",
    "HOME",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CARGO_TARGET_DIR",
    "TMPDIR",
    "TMP",
    "TEMP",
    "OUT_DIR",
];

/// Check that the given environment variable name is on the allowlist.
pub fn is_env_var_allowed(name: &str) -> bool {
    // Also allow any CARGO_* and RUSTC_* variables set by the Cargo driver.
    ALLOWED_ENV_VARS.contains(&name)
        || name.starts_with("CARGO_")
        || name.starts_with("RUSTC_")
        || name.starts_with("DEP_")
}

/// Filter a set of environment variables to the allowed set.
pub fn filter_env(
    vars: impl Iterator<Item = (String, String)>,
) -> impl Iterator<Item = (String, String)> {
    vars.filter(|(k, _)| is_env_var_allowed(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_is_allowed() {
        assert!(is_env_var_allowed("PATH"));
    }

    #[test]
    fn cargo_star_allowed() {
        assert!(is_env_var_allowed("CARGO_PKG_NAME"));
        assert!(is_env_var_allowed("CARGO_MANIFEST_DIR"));
    }

    #[test]
    fn arbitrary_var_rejected() {
        assert!(!is_env_var_allowed("MY_SECRET_TOKEN"));
        assert!(!is_env_var_allowed("OPENAI_API_KEY"));
    }
}
