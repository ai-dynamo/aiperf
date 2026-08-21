// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hierarchical cellular aggregation refusal.

use anyhow::{Result, bail};

/// Env var requesting a cellular aggregation fanout.
pub const CELL_AGG_FANOUT_ENV: &str = "AIPERF_CELL_AGG_FANOUT";

/// Whether the configured fanout requests a hierarchy rather than the flat star.
pub fn is_hierarchy_requested(cell_count: u32) -> bool {
    std::env::var(CELL_AGG_FANOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .is_some_and(|fanout| (2..cell_count).contains(&fanout))
}

/// Refuses the unavailable hierarchical aggregation role before it reads credentials
/// or binds a listener.
#[cfg(feature = "cellular")]
pub async fn run_aggregator(_envelope: &serde_json::Value) -> Result<()> {
    bail!(
        "hierarchical cellular aggregation is unavailable until every tree edge has controller-provisioned role security"
    )
}
