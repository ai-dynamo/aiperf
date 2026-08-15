// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Serializable trace-program driver and replay environment specifications.
//!
//! These DTOs travel with a complete graph program at placement boundaries. They
//! deliberately describe a driver or environment recipe without carrying an
//! opened process, host path, or trait object.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable replay task identity shared by input discovery and graph programs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ReplayTaskIdentity {
    /// Supported source adapter.
    pub adapter: String,
    /// Task-family identifier.
    pub family: String,
    /// Upstream task identifier.
    pub task_id: String,
    /// Optional descriptive workload role.
    #[serde(default)]
    pub primary_role: Option<String>,
}

/// One validated environment recipe selected before placement preflight.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceEnvironmentSpec {
    /// Registered environment-recipe identifier.
    pub kind: String,
    /// Recipe-specific, transportable configuration validated by that recipe.
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
}

/// Source facts retained for recorded replay without credentials.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayTraceMetadata {
    /// Zero-based ordinal of this task in its source manifest.
    pub manifest_ordinal: usize,
    /// Stable identity of the replay task.
    pub identity: ReplayTaskIdentity,
    /// BLAKE3 digest of the decompressed source recording.
    pub source_digest: String,
    /// Optional digest of normalization targets derived from the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_target_digest: Option<String>,
    /// Per-call target output lengths in source order.
    #[serde(default)]
    pub target_output_tokens: Vec<u64>,
    /// Expected model-call count from the source recording.
    pub expected_llm_node_count: u64,
    /// Expected completed tool-command count from the source recording.
    pub expected_tool_node_count: u64,
    /// Resolved request-profile identity used to lower recorded calls.
    pub request_profile_identity: String,
    /// Stable comparability labels retained in result provenance.
    #[serde(default)]
    pub comparability_annotations: BTreeMap<String, Value>,
}

/// Serializable selector for one registered trace-program driver.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceDriverSpec {
    /// Registered driver identifier.
    pub kind: String,
    /// Driver-specific, validated configuration.
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
}

impl TraceDriverSpec {
    /// Build the built-in static graph driver specification.
    pub fn static_graph() -> Self {
        Self {
            kind: "static_graph".into(),
            data: BTreeMap::new(),
        }
    }

    /// Whether this is the built-in static graph driver with no extra settings.
    pub fn is_static_graph(&self) -> bool {
        self.kind == "static_graph" && self.data.is_empty()
    }
}
