// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The native, fully-typed `BenchmarkRun` — simultaneously the CLI's domain
//! object AND the runner wire request.
//!
//! Architecture (per the user's directive: "one native BenchmarkRun IS the
//! domain+wire"): unlike Python — which keeps a rich `BenchmarkConfig` domain
//! and a separate `rust_wire` projection into a JSON request — the Rust CLI has
//! ONE typed object. The pre-translation layer parses CLI flags + YAML *into*
//! this object (see `crate::flags` / the loader), and serializing it *is* the
//! protocol-v2 request the unchanged `aiperf` consumes. There is no
//! separate wire DTO and no config→wire projection step.
//!
//! Typing discipline (maximal): every closed string set is an `enum`, units are
//! newtypes, optional-vs-absent is `Option`. `serde_json::Value` appears ONLY
//! for genuinely open bags whose schema is caller-defined (`resolved` facts,
//! `variables`, `endpoint.extra`, `headers`). Each `cfg` section is added as a
//! fully-typed struct as it is ported; [`BenchmarkConfig`] intentionally omits
//! `deny_unknown_fields` so a Python golden deserialized through it drops the
//! not-yet-ported sections symmetrically (the parity filter).
//!
//! Byte-exact serialization: fields use serde defaults that EMIT `null` for
//! `None` (Python dumps with `exclude_none=False`), so we never add
//! `skip_serializing_if` on wire-present optionals.

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
pub mod resolved;
pub mod runtime;
pub mod telemetry;
pub mod tokenizer;
pub mod transport;

pub use config::BenchmarkConfig;
pub use resolved::Resolved;

/// Requested runner operation. Wire spelling matches `RunnerOperationV2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Side-effect-free structural + static validation.
    Validate,
    /// Validate, prepare, execute, and commit the report.
    Execute,
}

/// The protocol-v2 envelope: exactly what is written to the runner's stdin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunnerRequest {
    /// Wire protocol discriminator (always `2`).
    pub protocol_version: u32,
    /// Requested operation.
    pub operation: Operation,
    /// The native benchmark run.
    pub run: BenchmarkRun,
}

impl RunnerRequest {
    /// The protocol version this CLI speaks.
    pub const PROTOCOL_V2: u32 = 2;

    /// Wrap one run in an execute/validate envelope.
    pub fn new(operation: Operation, run: BenchmarkRun) -> Self {
        Self {
            protocol_version: Self::PROTOCOL_V2,
            operation,
            run,
        }
    }
}

/// One fully-typed native benchmark run (domain object == wire request body).
///
/// Mirrors the runner-consumed shape of `BenchmarkRunWireV2`, fully typed. Open
/// bags (`resolved`, `variables`, `variation`) stay as `Value` because their
/// schema is computed/caller-defined rather than a closed set.
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
    fn operation_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(Operation::Execute).unwrap(),
            serde_json::json!("execute")
        );
    }

    #[test]
    fn none_optionals_emit_null_not_absent() {
        // Python dumps with exclude_none=False; wire-present optionals must
        // serialize as explicit null so the runner sees the same shape.
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
