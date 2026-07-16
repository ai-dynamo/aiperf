// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 run-level validation for the native online gRPC transport.
//!
//! There is no protocol-v1 request projection, endpoint adapter, or
//! transport×workload pair object in this module. The scheduled and graph
//! workload factories resolve the gRPC transport by id and call
//! [`validate_grpc_run`] for the transport-specific endpoint checks before
//! lowering into the common prepared native plan; execution then flows through
//! the shared `PreparedTurn`/graph placement over the gRPC dispatcher.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::transport_grpc::GrpcBindingRegistry;
use crate::transport_http::config::ClientConfig;
use anyhow::{Context, Result, ensure};
use url::Url;

use crate::engine::graph_execution::GraphTransportKind;
use crate::engine::grpc_turn_execution::GrpcExecutionFactory;
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{NativeTransportExecution, RunnerRunContext};
use crate::engine::turn_execution::RequestExecutorFactory;

/// Native execution binding for the built-in `grpc` transport.
///
/// gRPC drives the same `RequestExecutor` seam as HTTP; the binding owns its
/// own [`GrpcExecutionFactory`] (it is not a named field of the
/// process execution-factory set), so gRPC is a transport the workloads treat
/// identically to any other. Readiness polling is skipped (no gRPC server-ready
/// probe today) and graph nodes dispatch over the Tonic sink.
#[derive(Debug, Default)]
pub struct GrpcNativeExecution;

impl GrpcNativeExecution {
    /// Construct the stateless gRPC execution binding.
    pub fn new() -> Self {
        Self
    }
}

impl NativeTransportExecution for GrpcNativeExecution {
    fn executor_factory(&self) -> Arc<dyn RequestExecutorFactory> {
        Arc::new(GrpcExecutionFactory::default())
    }

    fn readiness_enabled(&self) -> bool {
        false
    }

    fn graph_transport_kind(&self) -> Result<GraphTransportKind> {
        Ok(GraphTransportKind::Grpc)
    }

    fn validate_run(&self, run: &AuthoredRunSpecV2, context: &RunnerRunContext) -> Result<()> {
        validate_grpc_run(run, context)
    }

    fn provenance(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("transport".to_owned(), "grpc".to_owned())])
    }
}

/// Validate the gRPC-specific endpoint policy for one authored run.
///
/// Called by the scheduled and graph workload factories once the gRPC transport
/// is resolved by id. Confirms every endpoint profile has a built-in gRPC
/// binding, `grpc://`/`grpcs://` URLs, no HTTP-only client policy, no readiness
/// retries (unsupported over gRPC today), one routing/session policy across
/// profiles, and no online sidecars.
pub(crate) fn validate_grpc_run(run: &AuthoredRunSpecV2, context: &RunnerRunContext) -> Result<()> {
    let default = context.default_endpoint_profile()?;
    let default_http_client = ClientConfig::default();
    let bindings = GrpcBindingRegistry::builtin()?;
    for (profile_id, profile) in context.endpoint_profiles() {
        bindings
            .prepare(&profile.endpoint_id)
            .with_context(|| format!("endpoint profile {profile_id:?} gRPC binding"))?;
        ensure!(
            !profile.config.urls.is_empty(),
            "endpoint profile {profile_id:?} has no gRPC URL"
        );
        let schemes = profile
            .config
            .urls
            .iter()
            .map(|url| {
                let parsed = Url::parse(url)
                    .with_context(|| format!("parsing endpoint profile {profile_id:?} URL"))?;
                let scheme = parsed.scheme().to_ascii_lowercase();
                ensure!(
                    matches!(scheme.as_str(), "grpc" | "grpcs"),
                    "grpc endpoint profile {profile_id:?} requires grpc:// or grpcs:// URLs, got {url:?}"
                );
                Ok(scheme)
            })
            .collect::<Result<BTreeSet<_>>>()?;
        ensure!(
            schemes.len() == 1,
            "endpoint profile {profile_id:?} mixes grpc:// and grpcs:// URLs"
        );
        // `ssl_verify` is intentionally NOT part of this guard: it is honored by
        // the gRPC TLS builder (grpcs cert verification toggle), so an authored
        // `ssl_verify=false` is supported rather than rejected as HTTP-specific.
        ensure!(
            profile.client.http_version == default_http_client.http_version
                && profile.client.keepalive_ns == default_http_client.keepalive_ns
                && profile.client.max_connections_per_origin
                    == default_http_client.max_connections_per_origin,
            "endpoint profile {profile_id:?} authors HTTP-specific client policy unsupported by grpc"
        );
        ensure!(
            profile.config.wait_for_model_timeout <= 0.0,
            "protocol-v2 grpc execution does not yet run readiness retries; endpoint profile {profile_id:?} enables one"
        );
        // The common prepared execution factory currently constructs one
        // transport policy for the default profile. Fail closed rather than
        // silently applying it to a differently routed secondary profile.
        ensure!(
            profile.config.urls == default.config.urls,
            "grpc currently requires every endpoint profile to share the default profile URL list"
        );
        ensure!(
            profile.connection_reuse == default.connection_reuse,
            "grpc currently requires one connection_reuse policy across endpoint profiles"
        );
        ensure!(
            profile.session_header == default.session_header,
            "grpc currently requires one session_header across endpoint profiles"
        );
        ensure!(
            profile.config.timeout_seconds == default.config.timeout_seconds,
            "grpc currently requires one request timeout across endpoint profiles"
        );
    }
    ensure!(
        run.sidecars.content_server.is_none()
            && run.sidecars.gpu_telemetry.is_none()
            && run.sidecars.network_latency.is_none()
            && run.sidecars.server_metrics.is_none()
            && run.sidecars.live_streaming.is_none(),
        "protocol-v2 grpc execution has no registered sidecar adapter"
    );
    Ok(())
}
