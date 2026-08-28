// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! In-process static validation of one resolved run.
//!
//! `aiperf config validate` used to stop at "does this YAML resolve?", which
//! cannot see an unregistered component ID, an incompatible streaming selection,
//! or a forbidden resource for the selected workload — every one of which needs
//! the frozen registry. This module submits the same run the CLI would execute
//! through `OperationV2::Validate`, which is the coordinator's side-effect-free
//! path: it opens no socket, no dataset, and no stream.

use aiperf_runtime::config::resolve::Inputs;
use aiperf_runtime::engine::application::Application;
use aiperf_runtime::engine::coordinator::ResponseV2;
use aiperf_runtime::engine::distribution_identity::current_distribution_id;
use aiperf_runtime::engine::protocol_v2::{
    DiagnosticV2, EnvelopeV2, OperationV2, PROTOCOL_V2, resolved_run_bytes,
};

/// One static-validation outcome rendered for the CLI.
#[derive(Debug)]
pub struct StaticValidation {
    /// Whether every static rule passed.
    pub is_valid: bool,
    /// Typed diagnostics in coordinator order; empty on success.
    pub errors: Vec<DiagnosticV2>,
    /// Checks the coordinator deferred to execution preparation.
    pub deferred: Vec<String>,
}

/// Validate one set of authoring inputs against the linked distribution.
///
/// Composes the stock application exactly as `--execute` would, then submits the
/// run as a `validate` operation. No subprocess is spawned: the operation is
/// side-effect free by construction, so process isolation buys nothing here.
pub fn validate_statically(inputs: &Inputs) -> anyhow::Result<StaticValidation> {
    #[derive(serde::Serialize)]
    struct AuthoringWire<'a> {
        authoring: &'a Inputs,
    }

    let payload = serde_json::to_vec(&AuthoringWire { authoring: inputs })
        .map_err(|error| anyhow::anyhow!("failed to serialize the run for validation: {error}"))?;
    // Resolution is the runtime's, not the CLI's: `resolved_run_bytes` is the
    // same single authoritative `Inputs -> BenchmarkRun` step `--execute` uses,
    // and it preserves factory-owned `RawValue` config that a `Value` round-trip
    // would destroy.
    let resolved = resolved_run_bytes(&payload)?;
    let run = serde_json::from_slice(&resolved)
        .map_err(|error| anyhow::anyhow!("resolved run failed the wire contract: {error}"))?;

    let distribution_id = current_distribution_id()
        .map_err(|error| anyhow::anyhow!("failed to identify aiperf distribution: {error}"))?;
    let application = Application::stock(distribution_id)
        .map_err(|error| anyhow::anyhow!("failed to compose aiperf distribution: {error:#}"))?;

    let result = application.handle_v2(EnvelopeV2 {
        protocol_version: PROTOCOL_V2,
        operation: OperationV2::Validate,
        run,
    });
    match result.response {
        ResponseV2::Validation(validation) => Ok(StaticValidation {
            is_valid: validation.success,
            errors: validation.errors,
            deferred: validation
                .deferred_checks
                .into_iter()
                .map(|check| format!("{} ({})", check.path, check.reason))
                .collect(),
        }),
        // The coordinator answers a `Validate` operation with a validation
        // response on every path, including its failure paths.
        ResponseV2::Terminal(_) => Err(anyhow::anyhow!(
            "validate operation returned a terminal response"
        )),
    }
}
