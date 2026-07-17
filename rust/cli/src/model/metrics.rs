// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed metrics policy.
//!
//! `slos` is caller-defined; optional fields are omitted when unset.

use serde::{Deserialize, Serialize};

/// The typed native metrics policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// Goodput SLO thresholds keyed by metric tag (open bag; empty by default).
    #[serde(default)]
    pub slos: serde_json::Map<String, serde_json::Value>,
    /// Timeslice window, in seconds (present only when a slice is requested).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_duration_seconds: Option<f64>,
    /// Bounded-memory sketch retention (present only when enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sketch: Option<bool>,
}
