// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WEKA-loader tunables, ported from the `Environment.DATASET.WEKA_*` fields in
//! `src/aiperf/common/environment.py`. [`WekaConfig::default`] reproduces the
//! Python defaults exactly; callers override per run.

/// Configuration for WEKA trace reconstruction (chain detection + classification).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WekaConfig {
    /// Split flattened agent fan-outs into per-agent chains (`WEKA_SPLIT_FLATTENED_AGENTS`).
    pub split_flattened_agents: bool,
    /// Emit tool-shaped messages (`WEKA_TOOL_SHAPED_MESSAGES`).
    pub tool_shaped_messages: bool,
    /// Seam-join max gap seconds (`WEKA_SEAM_MAX_GAP_SECONDS`).
    pub seam_max_gap_seconds: f64,
    /// Seam-join min overlap ratio (`WEKA_SEAM_MIN_OVERLAP_RATIO`).
    pub seam_min_overlap_ratio: f64,
    /// Max requests for aux one-shot reclassification (`WEKA_AUX_MAX_REQUESTS`).
    pub aux_max_requests: usize,
    /// Aux small-fresh-context ISL ratio (`WEKA_AUX_ISL_RATIO`).
    pub aux_isl_ratio: f64,
    /// Aux ISL floor (`WEKA_AUX_ISL_FLOOR`).
    pub aux_isl_floor: i64,
    /// Aux cross-model arm enabled (`WEKA_AUX_CROSS_MODEL`).
    pub aux_cross_model: bool,
    /// Reduction max output length (`WEKA_AUX_REDUCTION_OSL_MAX`).
    pub aux_reduction_osl_max: i64,
    /// Reduction input/output ratio (`WEKA_AUX_REDUCTION_RATIO`).
    pub aux_reduction_ratio: f64,
    /// Minimum members for a parallel worker group (`WEKA_WORKER_GROUP_MIN`).
    pub worker_group_min: i64,
}

impl Default for WekaConfig {
    fn default() -> Self {
        Self {
            split_flattened_agents: true,
            tool_shaped_messages: false,
            seam_max_gap_seconds: 3600.0,
            seam_min_overlap_ratio: 0.5,
            aux_max_requests: 1,
            aux_isl_ratio: 0.10,
            aux_isl_floor: 16384,
            aux_cross_model: true,
            aux_reduction_osl_max: 4000,
            aux_reduction_ratio: 20.0,
            worker_group_min: 3,
        }
    }
}

/// Title-generation preamble output-token ceiling (`_TITLE_GEN_MAX_OUTPUT_TOKENS`).
pub const TITLE_GEN_MAX_OUTPUT_TOKENS: i64 = 64;

/// Join epsilon in seconds (`_JOIN_EPSILON_SECONDS`).
pub const JOIN_EPSILON_SECONDS: f64 = 1e-6;
