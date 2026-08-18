// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! CLI-side pre-launch validation and reset-kv-cache execution.
//!
//! The outer `aiperf profile` loop owns one logical runner invocation per single run,
//! sweep cell, or search probe. This module validates that endpoint-local control
//! hooks are compatible with the selected transport before the child launch and, for
//! single-process runs, executes `endpoint.reset_kv_cache` exactly once before the
//! runner child starts.

use std::rc::Rc;

use aiperf_runtime::clock::{Clock, RealClock, RealClockAnchor};
use aiperf_runtime::engine::control_hooks::{
    prepare_endpoint_control_hooks_from_profile_value, run_reset_kv_cache,
};
use aiperf_runtime::engine::control_plane_http::{
    ControlPlaneClientPolicy, ControlPlaneHttpProviderFactory,
    NativeControlPlaneHttpProviderFactory,
};
use anyhow::{Context, Result, bail};

use crate::model::BenchmarkRun;
use crate::model::transport::Transport;

/// Validate the selected transport for any configured endpoint-local control hooks.
pub(crate) fn validate_supported_control_hook_transport(run: &BenchmarkRun) -> Result<()> {
    if !control_hooks_requested(run) {
        return Ok(());
    }
    match run.cfg.transport.as_ref() {
        None | Some(Transport::Http) | Some(Transport::Grpc) | Some(Transport::Websocket(_)) => {
            Ok(())
        }
        Some(Transport::DryRun(_)) => bail!(
            "endpoint.reset_kv_cache / endpoint.server_profiler require a live HTTP or gRPC target; \
             the dry_run transport has no server control plane"
        ),
        Some(Transport::DynosimOffline(_)) | Some(Transport::DynosimOnline(_)) => bail!(
            "endpoint.reset_kv_cache / endpoint.server_profiler are unsupported for dynosim transports; \
             they require a live server control plane"
        ),
    }
}

/// Run `endpoint.reset_kv_cache` exactly once before a single-process child launch.
pub(crate) fn run_reset_kv_cache_before_run(run: &BenchmarkRun) -> Result<()> {
    validate_supported_control_hook_transport(run)?;
    let runtime_cells = run.cfg.runtime.as_ref().map_or(1, |runtime| runtime.cells);
    if runtime_cells > 1 {
        return Ok(());
    }
    let Some(endpoint) = run.cfg.endpoint.as_ref() else {
        return Ok(());
    };
    if endpoint.reset_kv_cache.is_none() {
        return Ok(());
    }

    let endpoint_value =
        serde_json::to_value(endpoint).context("serializing endpoint control-hook profile")?;
    let clock: Rc<dyn Clock> = RealClock::from_anchor(RealClockAnchor::now());
    let provider = NativeControlPlaneHttpProviderFactory::default()
        .prepare(clock.clone(), ControlPlaneClientPolicy::default());
    let hooks = prepare_endpoint_control_hooks_from_profile_value(
        clock,
        provider.as_ref(),
        &endpoint_value,
    )
    .context("preparing endpoint-local control hooks")?;
    if let Some(reset) = hooks.reset_kv_cache.as_ref() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building reset_kv_cache runtime")?;
        let local = tokio::task::LocalSet::new();
        local
            .block_on(&runtime, run_reset_kv_cache(reset))
            .context("executing endpoint.reset_kv_cache before runner launch")?;
    }
    Ok(())
}

fn control_hooks_requested(run: &BenchmarkRun) -> bool {
    run.cfg.endpoint.as_ref().is_some_and(|endpoint| {
        endpoint.reset_kv_cache.is_some() || endpoint.server_profiler.is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::BenchmarkConfig;
    use crate::model::endpoint::{
        ConnectionReuse, Endpoint, EndpointType, ResetKvCacheConfig, ServerProfilerConfig,
    };
    use crate::model::runtime::Runtime;

    fn run_with_endpoint(endpoint: Endpoint) -> BenchmarkRun {
        BenchmarkRun {
            benchmark_id: "bench".to_owned(),
            artifact_dir: std::path::PathBuf::from("/tmp/aiperf-control-hooks-test"),
            cfg: BenchmarkConfig {
                endpoint: Some(endpoint),
                ..BenchmarkConfig::default()
            },
            cli_command: None,
            label: String::new(),
            random_seed: None,
            sweep_id: None,
            trial: 0,
            variation: None,
            resolved: crate::model::Resolved::default(),
            variables: serde_json::Map::new(),
        }
    }

    fn endpoint() -> Endpoint {
        Endpoint {
            urls: vec!["http://127.0.0.1:8000".to_owned()],
            endpoint_type: EndpointType("chat".to_owned()),
            streaming: false,
            use_legacy_max_tokens: false,
            use_server_token_count: false,
            timeout_seconds: 30.0,
            connection_reuse: ConnectionReuse::Pooled,
            ssl_verify: true,
            uds_path: None,
            connection_limit: 4,
            keepalive_timeout: 15.0,
            download_video_content: false,
            extra: serde_json::Map::new(),
            headers: std::collections::BTreeMap::new(),
            http2: false,
            wait_for_model_timeout: 0.0,
            wait_for_model_interval: 5.0,
            wait_for_model_mode: crate::model::endpoint::WaitForModelMode::Inference,
            path: None,
            api_key: None,
            session_header: None,
            request_content_type: None,
            template: None,
            response_field: None,
            reset_kv_cache: Some(ResetKvCacheConfig {
                timeout_seconds: None,
                path: None,
            }),
            server_profiler: Some(ServerProfilerConfig {
                timeout_seconds: None,
                start_path: None,
                stop_path: None,
            }),
            proxy: None,
            proxy_from_env: false,
        }
    }

    #[test]
    fn dry_run_transport_rejects_control_hooks() {
        let mut run = run_with_endpoint(endpoint());
        run.cfg.transport = Some(Transport::DryRun(Default::default()));
        let error = validate_supported_control_hook_transport(&run)
            .expect_err("dry_run should reject endpoint-local control hooks");
        assert!(error.to_string().contains("dry_run transport"));
    }

    #[test]
    fn cellular_runs_skip_local_reset_execution() {
        let mut run = run_with_endpoint(endpoint());
        run.cfg.runtime = Some(Runtime {
            cells: 4,
            ..Runtime::default()
        });
        run_reset_kv_cache_before_run(&run).expect("cellular run defers reset to the controller");
    }
}
