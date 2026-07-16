// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `models` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_authored_models`.
//! Serializing [`Models`] yields the exact `cfg.models` subtree the runner
//! consumes.

use serde::{Deserialize, Serialize};

/// Model-selection strategy across a multi-model item list. Closed set
/// (`ModelSelectionStrategy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStrategy {
    /// Rotate through models in order.
    RoundRobin,
    /// Pick uniformly at random.
    Random,
    /// Weighted random by item `weight`.
    Weighted,
}

/// One selectable model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelItem {
    /// Model name as sent to the server.
    pub name: String,
    /// Selection weight (only projected when set, for `weighted`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
}

/// The typed `models` section.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Models {
    /// Selection strategy.
    pub strategy: ModelStrategy,
    /// Ordered model items.
    pub items: Vec<ModelItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_wire_spellings() {
        assert_eq!(
            serde_json::to_value(ModelStrategy::RoundRobin).unwrap(),
            serde_json::json!("round_robin")
        );
    }

    #[test]
    fn weight_omitted_when_absent() {
        let m = ModelItem {
            name: "x".into(),
            weight: None,
        };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v.get("weight"), None);
    }
}
