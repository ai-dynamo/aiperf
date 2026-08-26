// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed worker and cell runtime policy.
//!
//! `workers` and `workers_min` emit null when unset; `workers_max` is omitted.

use crate::config::model::{DispatchMode, HopRouting};
use serde::{Deserialize, Serialize};

fn default_cells() -> u32 {
    1
}

/// The typed worker/cell runtime policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Runtime {
    /// Fixed worker count (`null` = runner auto-selects).
    pub workers: Option<u32>,
    /// Minimum worker count for adaptive worker scaling (`null` = unset).
    pub workers_min: Option<u32>,
    /// Maximum worker count for adaptive worker scaling (present only when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers_max: Option<u32>,
    /// Cellular (multi-process) cell count; `1` is the single-process path.
    #[serde(default = "default_cells")]
    pub cells: u32,
    /// Admission strategy for `workers>1` scheduled execution (`runtime.dispatch`).
    /// Absent (`None`) omits the wire field, which the runner decodes as
    /// [`DispatchMode::default`] (`Global`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<DispatchMode>,
    /// Worker-assignment policy for the single-coordinator modes (`dispatch ==
    /// global-hop` or `global-push`) with `workers > 1` (`runtime.hop_routing`).
    /// Absent (`None`) omits the wire field, decoded as
    /// [`HopRouting::default`] (`RoundRobin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hop_routing: Option<HopRouting>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            workers: None,
            workers_min: None,
            workers_max: None,
            cells: 1,
            dispatch: None,
            hop_routing: None,
        }
    }
}
