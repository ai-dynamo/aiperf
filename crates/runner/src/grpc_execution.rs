// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2-only scheduled pair for the native online gRPC transport.
//!
//! There is no protocol-v1 request projection or endpoint adapter in this
//! module. Authored Config v2 is validated through the open runner, endpoint,
//! dataset, and gRPC binding registries, then lowered once into the common
//! prepared scheduled plan.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use aiperf::metrics_core::ReportPairRunFacts;
use aiperf::transport_grpc::GrpcBindingRegistry;
use aiperf::transport_http::config::ClientConfig;
use anyhow::{Context, Result, anyhow, bail, ensure};
use url::Url;

use crate::execute::execute_prepared_native_plan_uncommitted_with_factories;
use crate::execution_factories::RunnerExecutionFactories;
use crate::online_execution::{
    NativeOnlineTokenizerSourceResolver, OnlineTokenizerSourceResolver, lower_scheduled,
    validate_authored_tokenizer,
};
use crate::protocol_v2::AuthoredRunSpecV2;
use crate::registry::{
    OnlineGrpcTransportConfigV2, PreparedRunOutcome, PreparedRunnerOperation, RunnerPairFactory,
    RunnerRegistryBuilder, RunnerRunContext, ScheduledWorkloadConfigV2, ValidatedTransportConfig,
    ValidatedWorkloadConfig,
};

const TRANSPORT_ID: &str = "grpc";
const WORKLOAD_ID: &str = "scheduled";

/// Register the native gRPC scheduled pair.
pub fn register_grpc_pairs(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_pair(Arc::new(OnlineGrpcScheduledPair {
        tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
    }))
}

#[derive(Clone)]
struct OnlineGrpcScheduledPair {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineGrpcScheduledPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineGrpcScheduledPair")
    }
}

impl RunnerPairFactory for OnlineGrpcScheduledPair {
    fn transport_id(&self) -> &'static str {
        TRANSPORT_ID
    }

    fn workload_id(&self) -> &'static str {
        WORKLOAD_ID
    }

    fn validate_pair(
        &self,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        grpc_transport(transport)?;
        scheduled_workload(workload)?;
        Ok(())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(transport, workload)?;
        validate_grpc_run(run, context)?;
        validate_authored_tokenizer(&scheduled_workload(workload)?.tokenizer)
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _transport: Box<dyn ValidatedTransportConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!("{TRANSPORT_ID} preparation requires the coordinator-owned RunnerRunContext")
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        transport: Box<dyn ValidatedTransportConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        grpc_transport(transport.as_ref())?;
        let workload = scheduled_workload(workload.as_ref())?;
        let plan = lower_scheduled(run, context, workload, self.tokenizers.as_ref())?;
        Ok(Box::new(PreparedGrpcScheduledOperation {
            plan,
            product_registry: context.product_registry_handle(),
            execution_factories: context.execution_factories_handle(),
        }))
    }
}

fn grpc_transport(config: &dyn ValidatedTransportConfig) -> Result<&OnlineGrpcTransportConfigV2> {
    ValidatedTransportConfig::as_any(config)
        .downcast_ref::<OnlineGrpcTransportConfigV2>()
        .ok_or_else(|| anyhow!("online gRPC pair received a different transport config type"))
}

fn scheduled_workload(config: &dyn ValidatedWorkloadConfig) -> Result<&ScheduledWorkloadConfigV2> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<ScheduledWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("online gRPC pair received a different workload config type"))
}

fn validate_grpc_run(run: &AuthoredRunSpecV2, context: &RunnerRunContext) -> Result<()> {
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
        ensure!(
            profile.client.http_version == default_http_client.http_version
                && profile.client.ssl_verify == default_http_client.ssl_verify
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

struct PreparedGrpcScheduledOperation {
    plan: crate::execute::NativeRunSpec,
    product_registry: Arc<aiperf::extensions::AiperfRegistry>,
    execution_factories: RunnerExecutionFactories,
}

impl fmt::Debug for PreparedGrpcScheduledOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedGrpcScheduledOperation")
            .field("benchmark_id", &self.plan.benchmark_id)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedGrpcScheduledOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let native_report = execute_prepared_native_plan_uncommitted_with_factories(
            self.plan,
            self.execution_factories.grpc(),
            self.execution_factories.graph(),
            self.product_registry.as_ref(),
        )?;
        Ok(PreparedRunOutcome {
            native_report,
            report_facts: ReportPairRunFacts::new(),
            provenance: BTreeMap::from([("transport".into(), "grpc".into())]),
            report_commit: None,
        })
    }
}
