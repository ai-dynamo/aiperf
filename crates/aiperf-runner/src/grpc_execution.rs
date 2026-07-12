// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2-only scheduled pair for the native online gRPC backend.
//!
//! There is no protocol-v1 request projection or endpoint adapter in this
//! module. Authored Config v2 is validated through the open runner, endpoint,
//! dataset, and gRPC binding registries, then lowered once into the common
//! prepared scheduled plan.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use aiperf_metrics::ReportPairRunFacts;
use aiperf_transport_grpc::GrpcBindingRegistry;
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
    OnlineGrpcBackendConfigV2, PreparedRunOutcome, PreparedRunnerOperation, RunnerPairFactory,
    RunnerRegistryBuilder, RunnerRunContext, ScheduledWorkloadConfigV2, ValidatedBackendConfig,
    ValidatedWorkloadConfig,
};

const BACKEND_ID: &str = "online_grpc";
const WORKLOAD_ID: &str = "scheduled";

/// Register the native gRPC scheduled pair.
pub fn register_online_grpc_pairs(builder: &mut RunnerRegistryBuilder) -> Result<()> {
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
    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    fn workload_id(&self) -> &'static str {
        WORKLOAD_ID
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        grpc_backend(backend)?;
        scheduled_workload(workload)?;
        Ok(())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        validate_grpc_run(run, context)?;
        validate_authored_tokenizer(&scheduled_workload(workload)?.tokenizer)
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!("{BACKEND_ID} preparation requires the coordinator-owned RunnerRunContext")
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        grpc_backend(backend.as_ref())?;
        let workload = scheduled_workload(workload.as_ref())?;
        let plan = lower_scheduled(run, context, workload, self.tokenizers.as_ref())?;
        Ok(Box::new(PreparedGrpcScheduledOperation {
            plan,
            product_registry: context.product_registry_handle(),
            execution_factories: context.execution_factories_handle(),
        }))
    }
}

fn grpc_backend(config: &dyn ValidatedBackendConfig) -> Result<&OnlineGrpcBackendConfigV2> {
    ValidatedBackendConfig::as_any(config)
        .downcast_ref::<OnlineGrpcBackendConfigV2>()
        .ok_or_else(|| anyhow!("online gRPC pair received a different backend config type"))
}

fn scheduled_workload(config: &dyn ValidatedWorkloadConfig) -> Result<&ScheduledWorkloadConfigV2> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<ScheduledWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("online gRPC pair received a different workload config type"))
}

fn validate_grpc_run(run: &AuthoredRunSpecV2, context: &RunnerRunContext) -> Result<()> {
    let default = context.default_endpoint_profile()?;
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
                    "online_grpc endpoint profile {profile_id:?} requires grpc:// or grpcs:// URLs, got {url:?}"
                );
                Ok(scheme)
            })
            .collect::<Result<BTreeSet<_>>>()?;
        ensure!(
            schemes.len() == 1,
            "endpoint profile {profile_id:?} mixes grpc:// and grpcs:// URLs"
        );
        ensure!(
            !profile.http2,
            "endpoint profile {profile_id:?}.http2 is HTTP-specific and unsupported by online_grpc"
        );
        ensure!(
            profile.config.wait_for_model_timeout <= 0.0,
            "protocol-v2 online_grpc execution does not yet run readiness retries; endpoint profile {profile_id:?} enables one"
        );
        // The common prepared execution factory currently constructs one
        // transport policy for the default profile. Fail closed rather than
        // silently applying it to a differently routed secondary profile.
        ensure!(
            profile.config.urls == default.config.urls,
            "online_grpc currently requires every endpoint profile to share the default profile URL list"
        );
        ensure!(
            profile.connection_reuse == default.connection_reuse,
            "online_grpc currently requires one connection_reuse policy across endpoint profiles"
        );
        ensure!(
            profile.session_header == default.session_header,
            "online_grpc currently requires one session_header across endpoint profiles"
        );
        ensure!(
            profile.config.timeout_seconds == default.config.timeout_seconds,
            "online_grpc currently requires one request timeout across endpoint profiles"
        );
    }
    ensure!(
        run.sidecars.gpu_telemetry.is_none()
            && run.sidecars.network_latency.is_none()
            && run.sidecars.server_metrics.is_none()
            && run.sidecars.live_streaming.is_none(),
        "protocol-v2 online_grpc execution has no registered sidecar adapter"
    );
    Ok(())
}

struct PreparedGrpcScheduledOperation {
    plan: crate::execute::NativeRunPlan,
    product_registry: Arc<aiperf_extensions::AiperfRegistry>,
    execution_factories: RunnerExecutionFactories,
}

impl fmt::Debug for PreparedGrpcScheduledOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedGrpcScheduledOperation")
            .field("benchmark_id", &self.plan.run.benchmark_id)
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
