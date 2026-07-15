// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The CLI's own protocol-v2 wire DTOs — the runner-consumed request schema.
//!
//! These types are hand-written to mirror **only** the fields the unchanged
//! `aiperf-runner` deserializes and acts on (see
//! `rust/aiperf/src/runner_protocol/protocol_v2.rs`). They deliberately ignore
//! any additional keys Python's full `model_dump` carries, so they serve three
//! roles at once (see the plan's parity mechanism):
//!
//! 1. **Output** — the projection builds and serializes them into the request.
//! 2. **Golden filter** — deserializing a Python golden through them strips the
//!    runner-ignored passthrough keys symmetrically.
//! 3. **Gate** — the parity test compares `to_value` of both sides.
//!
//! They intentionally do NOT use `#[serde(deny_unknown_fields)]`: ignoring
//! unknown keys is what makes role (2) work.
//!
//! This is the seed of the schema; concrete `cfg` sections are added as each
//! projection section is ported. Until then `run.cfg` is retained as an opaque
//! [`serde_json::Value`] so the envelope round-trips losslessly.

use serde::{Deserialize, Serialize};

/// Requested runner operation. Matches `RunnerOperationV2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Side-effect-free structural + static validation.
    Validate,
    /// Validate, prepare, execute, and commit the report.
    Execute,
}

/// One protocol-v2 request envelope, reduced to the runner-consumed schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CliRequest {
    /// Wire protocol discriminator (always `2`).
    pub protocol_version: u32,
    /// Requested operation.
    pub operation: Operation,
    /// The benchmark run. Typed section-by-section as projection lands; opaque
    /// for now so the envelope round-trips.
    pub run: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_serializes_snake_case() {
        let v = serde_json::to_value(Operation::Execute).unwrap();
        assert_eq!(v, serde_json::json!("execute"));
    }
}
