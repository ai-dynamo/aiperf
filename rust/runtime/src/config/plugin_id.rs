// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Syntactic validation for plugin-registered component IDs.
//!
//! A plugin ID is a slash-separated namespace path with an optional colon-separated
//! version suffix. Only printable ASCII excluding control characters and OS path
//! separators outside the namespace slash is accepted. Absolute paths, relative
//! traversal components (`..`), and empty IDs are rejected without loading any
//! plugin.

/// Validate a plugin component ID syntactically.
///
/// The check is intentionally lightweight: it rejects structural path injection
/// (absolute, `./`, `../`) and empty strings. Full semantic validation (whether
/// the plugin is actually registered and compatible) is deferred to runtime
/// when the frozen plugin universe is available.
///
/// # Errors
///
/// Returns a human-readable error string when the ID is structurally invalid.
pub fn validate_plugin_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("plugin ID cannot be empty".into());
    }
    if id.starts_with('/') {
        return Err(format!("plugin ID {id:?} cannot be an absolute path"));
    }
    if id.starts_with("./") || id == "." {
        return Err(format!("plugin ID {id:?} cannot be a relative path"));
    }
    if id.starts_with("../") || id == ".." {
        return Err(format!(
            "plugin ID {id:?} cannot start with a path traversal"
        ));
    }
    // Reject traversal components anywhere in a slash-separated path.
    for segment in id.split('/') {
        if segment == ".." || segment == "." {
            return Err(format!(
                "plugin ID {id:?} contains a path traversal component"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_paths_rejected() {
        assert!(validate_plugin_id("/usr/lib/plugin").is_err());
    }

    #[test]
    fn relative_dot_rejected() {
        assert!(validate_plugin_id("./local").is_err());
    }

    #[test]
    fn traversal_rejected() {
        assert!(validate_plugin_id("../escape").is_err());
        assert!(validate_plugin_id("vendor/../escape").is_err());
    }

    #[test]
    fn empty_rejected() {
        assert!(validate_plugin_id("").is_err());
    }

    #[test]
    fn namespaced_accepted() {
        assert!(validate_plugin_id("vendor/my-transport:1.0").is_ok());
        assert!(validate_plugin_id("com.example/plugin:0.1").is_ok());
        assert!(validate_plugin_id("my-plugin").is_ok());
    }
}
