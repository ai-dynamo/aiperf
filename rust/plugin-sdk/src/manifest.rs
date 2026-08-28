// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin manifest schema and validation.

use serde::{Deserialize, Serialize};

/// Parsed representation of a `plugin.toml` manifest file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    /// Must be `1`.
    pub schema_version: u32,
    /// Plugin identity section.
    pub plugin: PluginEntry,
    /// Host-requirement section.
    pub requires: Requirements,
}

/// Plugin identity fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginEntry {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Host requirement constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    /// SemVer requirement on `aiperf-plugin-sdk` (e.g. `">=0.13.0"`).
    pub aiperf_sdk: String,
    /// Required target triple (e.g. `"x86_64-unknown-linux-gnu"`).
    pub target: String,
}

/// Serialize a manifest to TOML bytes.
pub fn serialize_manifest(manifest: &PluginManifest) -> Result<Vec<u8>, ManifestError> {
    toml::to_string(manifest)
        .map(|s| s.into_bytes())
        .map_err(|e| ManifestError::Serialize(e.to_string()))
}

/// Parse a manifest from TOML bytes.
pub fn parse_manifest(bytes: &[u8]) -> Result<PluginManifest, ManifestError> {
    let s = std::str::from_utf8(bytes).map_err(|e| ManifestError::Utf8(e.to_string()))?;
    toml::from_str(s).map_err(|e| ManifestError::Parse(e.to_string()))
}

/// Validate a parsed manifest's semantic constraints.
pub fn validate_manifest(manifest: &PluginManifest) -> Result<(), ManifestError> {
    if manifest.schema_version != 1 {
        return Err(ManifestError::InvalidSchemaVersion(manifest.schema_version));
    }
    if manifest.plugin.name.is_empty() {
        return Err(ManifestError::EmptyField("plugin.name"));
    }
    // Validate plugin.version as a canonical SemVer version.
    semver::Version::parse(&manifest.plugin.version).map_err(|e| ManifestError::InvalidSemver {
        field: "plugin.version",
        value: manifest.plugin.version.clone(),
        reason: e.to_string(),
    })?;
    if manifest.requires.aiperf_sdk.is_empty() {
        return Err(ManifestError::EmptyField("requires.aiperf_sdk"));
    }
    // Validate requires.aiperf_sdk as a SemVer requirement expression.
    semver::VersionReq::parse(&manifest.requires.aiperf_sdk).map_err(|e| {
        ManifestError::InvalidSemver {
            field: "requires.aiperf_sdk",
            value: manifest.requires.aiperf_sdk.clone(),
            reason: e.to_string(),
        }
    })?;
    if manifest.requires.target.is_empty() {
        return Err(ManifestError::EmptyField("requires.target"));
    }
    Ok(())
}

/// Errors arising from manifest handling.
#[derive(Debug)]
pub enum ManifestError {
    InvalidSchemaVersion(u32),
    EmptyField(&'static str),
    /// A SemVer version or requirement string failed to parse.
    InvalidSemver {
        field: &'static str,
        value: String,
        reason: String,
    },
    Parse(String),
    Serialize(String),
    Utf8(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSchemaVersion(v) => {
                write!(f, "invalid schema_version {v}; expected 1")
            }
            Self::EmptyField(field) => write!(f, "manifest field {field} must not be empty"),
            Self::InvalidSemver {
                field,
                value,
                reason,
            } => write!(f, "invalid SemVer in {field} ({value:?}): {reason}"),
            Self::Parse(e) => write!(f, "manifest parse error: {e}"),
            Self::Serialize(e) => write!(f, "manifest serialize error: {e}"),
            Self::Utf8(e) => write!(f, "manifest UTF-8 error: {e}"),
        }
    }
}

impl std::error::Error for ManifestError {}
