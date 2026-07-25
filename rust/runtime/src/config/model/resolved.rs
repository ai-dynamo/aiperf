// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed facts resolved before runner execution.
//!
//! The runner reads GPU fields for custom telemetry and DCGM source selection.
//! Every field is present in the wire object, including nulls.

use serde::{Deserialize, Serialize};

/// Resolution facts attached to a run. Fields with a genuinely-uncertain shape
/// (custom metric bags, comm config) stay as `Value`; the rest are typed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resolved {
    /// Whether the artifact directory was created during resolution.
    pub artifact_dir_created: bool,
    /// Opaque communication configuration unused by native execution.
    pub comm_config: Option<serde_json::Value>,
    /// Concrete dataset file paths, when resolved.
    pub dataset_file_paths: Option<Vec<String>>,
    /// Whether the dataset carries timing data.
    pub dataset_has_timing_data: Option<bool>,
    /// Whether the dataset forks (graph programs).
    pub dataset_is_forking: Option<bool>,
    /// Number of dataset roots.
    pub dataset_root_count: Option<u64>,
    /// Per-dataset sampling strategy ids.
    pub dataset_sampling_strategies: Option<Vec<String>>,
    /// Number of conversation sessions.
    pub dataset_session_count: Option<u64>,
    /// Total dataset records.
    pub dataset_total_records: Option<u64>,
    /// Dataset kind ids.
    pub dataset_types: Option<Vec<String>>,
    /// Custom GPU metric definitions (name→unit), read from the telemetry CSV.
    pub gpu_custom_metrics: Option<serde_json::Value>,
    /// DCGM field mappings for the telemetry source.
    pub gpu_dcgm_mappings: Option<serde_json::Value>,
    /// GPU telemetry source mode (e.g. `summary`).
    pub gpu_telemetry_mode: String,
    /// Scenario-lock outcome (agentic/graph programs); `null` on the scheduled path.
    pub scenario_outcome: Option<serde_json::Value>,
    /// Resolved tokenizer identities.
    pub tokenizer_names: Option<Vec<String>>,
    /// Total expected run duration, seconds.
    pub total_expected_duration: Option<f64>,
}

impl Default for Resolved {
    fn default() -> Self {
        Self {
            artifact_dir_created: false,
            comm_config: None,
            dataset_file_paths: None,
            dataset_has_timing_data: None,
            dataset_is_forking: None,
            dataset_root_count: None,
            dataset_sampling_strategies: None,
            dataset_session_count: None,
            dataset_total_records: None,
            dataset_types: None,
            gpu_custom_metrics: None,
            gpu_dcgm_mappings: None,
            gpu_telemetry_mode: "summary".to_string(),
            scenario_outcome: None,
            tokenizer_names: None,
            total_expected_duration: None,
        }
    }
}
