// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run-level validation and binding for the native gRPC transport.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::transport::grpc::GrpcBindingRegistry;
use crate::transport::http::config::ClientConfig;
use anyhow::{Context, Result, ensure};
use url::Url;

use crate::engine::grpc_turn_execution::GrpcExecutionFactory;
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{NativeTransportExecution, RunContext};
use crate::engine::turn_execution::RequestExecutorFactory;

/// Native execution binding for the built-in `grpc` transport.
///
/// Owns the gRPC executor and graph dispatcher; readiness polling is disabled.
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

    fn build_graph_dispatcher(
        &self,
        clock: Rc<dyn crate::clock::Clock>,
        run_origin_ns: i64,
        urls: &[String],
        model: &str,
        transport_config: crate::transport::http::TransportSinkConfig,
        endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
        capture_raw: bool,
    ) -> Result<Rc<dyn crate::transport::core::Dispatcher>> {
        Ok(Rc::new(
            crate::engine::grpc_turn_execution::grpc_sink_with_endpoints(
                clock,
                run_origin_ns,
                urls,
                model.to_string(),
                transport_config,
                GrpcBindingRegistry::builtin()?,
                endpoints,
                capture_raw,
            )?,
        ))
    }

    fn graph_transport_label(&self) -> &'static str {
        "grpc"
    }

    fn validate_run(&self, run: &AuthoredRunSpecV2, context: &RunContext) -> Result<()> {
        validate_grpc_run(run, context)
    }

    fn run_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("transport".to_owned(), "grpc".to_owned())])
    }
}

/// Validate the gRPC-specific endpoint policy for one authored run.
///
/// Requires gRPC bindings and schemes, shared routing policy, and no sidecars.
pub(crate) fn validate_grpc_run(run: &AuthoredRunSpecV2, context: &RunContext) -> Result<()> {
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
            "protocol-v2 grpc execution rejects readiness retries; endpoint profile {profile_id:?} enables one"
        );
        // The prepared execution factory has one transport policy, so secondary
        // profiles must match the default rather than silently diverge.
        ensure!(
            profile.config.urls == default.config.urls,
            "grpc requires every endpoint profile to share the default profile URL list"
        );
        ensure!(
            profile.connection_reuse == default.connection_reuse,
            "grpc requires one connection_reuse policy across endpoint profiles"
        );
        ensure!(
            profile.session_header == default.session_header,
            "grpc requires one session_header across endpoint profiles"
        );
        ensure!(
            profile.config.timeout_seconds == default.config.timeout_seconds,
            "grpc requires one request timeout across endpoint profiles"
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
