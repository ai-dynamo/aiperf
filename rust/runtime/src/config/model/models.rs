// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed model-selection policy.

use serde::{Deserialize, Serialize};

/// Model-selection strategy across multiple models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStrategy {
    /// Rotate through models in order.
    #[default]
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
    /// Selection strategy. Omission must resolve to the same `round_robin` the
    /// protocol-v2 `ModelsSpec` and `resolve.rs` already default to; the typed
    /// model is re-serialized into the protocol-v2 request, so a missing default
    /// here hard-rejects an otherwise-valid config.
    #[serde(default)]
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
