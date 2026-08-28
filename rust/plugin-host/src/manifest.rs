// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict schema-2.0 plugin manifest DTOs.
//!
//! All structs use `#[serde(deny_unknown_fields)]` so that any field not
//! defined here is a hard parse error, preventing silent data loss when the
//! manifest contains fields intended for a newer schema version.

use serde::{Deserialize, Serialize};

/// Root of a native AIPerf plugin manifest file.
///
/// `schema_version` must be exactly `"2.0"`.  Version `"1.0"` is the Python
/// plugin format and returns the stable error code
/// `python-plugin-manifest-not-native`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifestV2 {
    pub schema_version: String,
    pub packages: Vec<PluginPackageManifestV2>,
}

/// Per-package metadata and artifact table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginPackageManifestV2 {
    pub id: String,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub baseline: Option<BaselineRequirementV2>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRecordV2>,
    pub categories: Vec<PluginCategoryEntryV2>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub depends_on: Vec<DependencyEdgeV2>,
}

/// Minimum host/allocator baseline the plugin was built and tested against.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineRequirementV2 {
    pub aiperf_version: String,
    #[serde(default)]
    pub allocator_digest: Option<String>,
}

/// One platform artifact entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecordV2 {
    pub target: String,
    pub path: String,
    pub digest: String,
    #[serde(default)]
    pub build_id: Option<String>,
}

/// A category registration entry.  The `category` tag selects the variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum PluginCategoryEntryV2 {
    Exporter(CategoryRegistrationV2),
    Endpoint(CategoryRegistrationV2),
    Transport(CategoryRegistrationV2),
    Workload(CategoryRegistrationV2),
    Dataset(CategoryRegistrationV2),
    Sampler(CategoryRegistrationV2),
    Actuator(CategoryRegistrationV2),
}

impl PluginCategoryEntryV2 {
    /// Returns the canonical ID for this category registration.
    pub fn id(&self) -> &str {
        match self {
            Self::Exporter(r)
            | Self::Endpoint(r)
            | Self::Transport(r)
            | Self::Workload(r)
            | Self::Dataset(r)
            | Self::Sampler(r)
            | Self::Actuator(r) => &r.id,
        }
    }

    /// Returns the aliases for this category registration.
    pub fn aliases(&self) -> &[String] {
        match self {
            Self::Exporter(r)
            | Self::Endpoint(r)
            | Self::Transport(r)
            | Self::Workload(r)
            | Self::Dataset(r)
            | Self::Sampler(r)
            | Self::Actuator(r) => &r.aliases,
        }
    }
}

/// Fields common to every category variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoryRegistrationV2 {
    pub id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// A directed dependency edge to another plugin package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct DependencyEdgeV2 {
    pub id: String,
    pub version: String,
}
