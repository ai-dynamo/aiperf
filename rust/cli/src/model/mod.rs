// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed benchmark domain object and protocol-v2 wire request.
//!
//! Closed string sets use enums; caller-defined bags remain JSON values.
//! Wire-present optional fields serialize as explicit nulls.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub mod artifacts;
pub mod config;
pub mod dataset;
pub mod endpoint;
pub mod export;
pub mod metrics;
pub mod models;
pub mod phase;
pub mod public_catalog;
pub mod rate_series;
pub mod resolved;
pub mod runtime;
pub mod telemetry;
pub mod tokenizer;
pub mod transport;

pub use config::BenchmarkConfig;
pub use resolved::Resolved;

/// A benchmark run serialized directly as the runner request body.
///
/// Computed or caller-defined open bags remain JSON values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkRun {
    /// Stable benchmark identifier.
    pub benchmark_id: String,
    /// Runner-owned artifact directory.
    pub artifact_dir: PathBuf,
    /// Canonical benchmark configuration.
    pub cfg: BenchmarkConfig,
    /// Redacted invoking command line (invocation-derived).
    #[serde(default)]
    pub cli_command: Option<String>,
    /// Human-readable run label.
    #[serde(default)]
    pub label: String,
    /// Deterministic root seed when authored.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Outer sweep identifier (absent for a bare single run).
    #[serde(default)]
    pub sweep_id: Option<String>,
    /// Zero-based trial number.
    #[serde(default)]
    pub trial: u32,
    /// Sweep variation metadata (absent for a bare single run). Open bag.
    #[serde(default)]
    pub variation: Option<serde_json::Value>,
    /// Resolution facts (gpu custom metrics, comm config, …), typed.
    #[serde(default)]
    pub resolved: Resolved,
    /// Envelope-level template variables. Open bag.
    #[serde(default)]
    pub variables: serde_json::Map<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_optionals_emit_null_not_absent() {
        let run = BenchmarkRun {
            benchmark_id: "b".into(),
            artifact_dir: "/tmp/x".into(),
            cfg: BenchmarkConfig::default(),
            cli_command: None,
            label: String::new(),
            random_seed: None,
            sweep_id: None,
            trial: 0,
            variation: None,
            resolved: Resolved::default(),
            variables: serde_json::Map::new(),
        };
        let v = serde_json::to_value(&run).unwrap();
        assert!(v.get("random_seed").is_some());
        assert_eq!(v["random_seed"], serde_json::Value::Null);
        assert_eq!(v["sweep_id"], serde_json::Value::Null);
    }
}
