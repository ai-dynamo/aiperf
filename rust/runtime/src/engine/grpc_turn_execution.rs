// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport sink for protocol-v2 turn execution.

use std::rc::Rc;
use std::sync::Arc;

use crate::clock::Clock;
use crate::dispatch::sink::RequestObserver;
use crate::endpoints::PreparedEndpointTable;
use crate::metrics_core::InferenceDimensions;
use crate::multiturn::TurnToSend;
use crate::scheduled::TurnResponseObserver;
use crate::transport::core::ConnectionReuseStrategy;
use crate::transport::core::{DispatchResult, MeasuredContext};
use crate::transport::core::{PreparedTurn, RequestExecutor};
use crate::transport::grpc::{
    ConnectionReuseStrategy as GrpcConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig,
};
use crate::transport::grpc::{GrpcTransportSink, GrpcTransportSinkConfig};
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;

use crate::engine::turn_execution::{
    ExecutionBackendConfig, ExecutionSinkBuilder, PreparedEndpointTableFactory,
    RequestExecutorFactory, WorkerSink, build_native,
};

/// Native gRPC execution factory.
#[derive(Clone, Debug, Default)]
pub struct GrpcExecutionFactory {
    bindings: Option<GrpcBindingRegistry>,
}

impl GrpcExecutionFactory {
    /// Construct with the built-in gRPC binding registry.
    pub fn builtin() -> Result<Self> {
        Ok(Self {
            bindings: Some(GrpcBindingRegistry::builtin()?),
        })
    }

    /// Construct with a distribution-composed binding registry.
    pub fn new(bindings: GrpcBindingRegistry) -> Self {
        Self {
            bindings: Some(bindings),
        }
    }
}

impl RequestExecutorFactory for GrpcExecutionFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        ensure!(
            config.workers > 0,
            "gRPC execution workers must be positive"
        );
        ensure!(
            config.prepared_endpoints.is_some(),
            "native gRPC execution requires protocol-v2 prepared endpoints"
        );
        let bindings = self
            .bindings
            .clone()
            .map(Ok)
            .unwrap_or_else(GrpcBindingRegistry::builtin)?;
        let workers = config.workers;
        let coordinator_clock = config.coordinator_clock.clone();
        let anchor = config.real_clock_anchor;
        let hop_routing = config.hop_routing;
        let worker_labels = config.worker_labels.clone();
        build_native(
            GrpcSinkBuilder::from_config(&config, bindings),
            workers,
            coordinator_clock,
            anchor,
            hop_routing,
            worker_labels,
        )
    }
}

/// Constructs worker-local gRPC transport sinks.
pub struct GrpcSinkBuilder {
    base_urls: Vec<String>,
    model: String,
    transport: crate::engine::turn_execution::ExecutionTransportPolicy,
    prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
    bindings: GrpcBindingRegistry,
}

impl GrpcSinkBuilder {
    pub fn from_config(config: &ExecutionBackendConfig, bindings: GrpcBindingRegistry) -> Self {
        Self {
            base_urls: config.base_urls.clone(),
            model: config.model.clone(),
            transport: config.transport.clone(),
            prepared_endpoints: config.prepared_endpoints.clone(),
            bindings,
        }
    }
}

impl ExecutionSinkBuilder for GrpcSinkBuilder {
    type Sink = GrpcTransportSink;

    fn label(&self) -> &'static str {
        "grpc"
    }

    fn build_sink(&self, clock: Rc<dyn Clock>, _worker_id: usize) -> Result<GrpcTransportSink> {
        prepare_grpc_sink(
            clock,
            0,
            &self.base_urls,
            self.model.clone(),
            self.transport.clone(),
            self.prepared_endpoints.as_deref(),
            self.bindings.clone(),
        )
    }
}

#[async_trait(?Send)]
impl WorkerSink for GrpcTransportSink {
    fn set_run_origin(&self, origin_ns: i64) {
        GrpcTransportSink::set_run_origin(self, origin_ns);
    }

    fn clock(&self) -> &dyn Clock {
        GrpcTransportSink::clock(self)
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <GrpcTransportSink as RequestExecutor>::inference_dimensions(self, turn)
    }

    fn supports_response_streaming(&self) -> bool {
        // The gRPC sink reports only terminal responses.
        false
    }

    async fn dispatch_measured(
        &self,
        observer: &dyn RequestObserver,
        turn: PreparedTurn,
        _context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        _responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        GrpcTransportSink::dispatch_collect(self, turn, observer, on_first_token).await
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_grpc_sink(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: crate::engine::turn_execution::ExecutionTransportPolicy,
    prepared_endpoints: Option<&dyn PreparedEndpointTableFactory>,
    bindings: GrpcBindingRegistry,
) -> Result<GrpcTransportSink> {
    let endpoints = prepared_endpoints
        .ok_or_else(|| anyhow!("gRPC execution requires a prepared endpoint table factory"))?
        .prepare_worker()?;
    grpc_sink_with_endpoints(
        clock,
        start_ns,
        base_urls,
        model,
        transport,
        bindings,
        Rc::new(endpoints),
    )
}

/// Assemble a gRPC sink from a worker-local prepared endpoint table.
///
/// `capture_raw` is the request payload's only consumer; with it off the sink
/// skips building the canonical payload entirely.
///
#[allow(clippy::too_many_arguments)]
pub(crate) fn grpc_sink_with_endpoints(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: crate::engine::turn_execution::ExecutionTransportPolicy,
    bindings: GrpcBindingRegistry,
    endpoints: Rc<PreparedEndpointTable>,
) -> Result<GrpcTransportSink> {
    GrpcTransportSink::new(
        clock,
        start_ns,
        base_urls,
        model,
        grpc_sink_config(transport),
        bindings,
    )?
    .with_prepared_endpoints(endpoints)
}

fn grpc_sink_config(
    transport: crate::engine::turn_execution::ExecutionTransportPolicy,
) -> GrpcTransportSinkConfig {
    GrpcTransportSinkConfig {
        client: GrpcClientConfig {
            total_timeout_ns: transport.total_timeout_ns,
            trace_chunks: false,
            ssl_verify: transport.ssl_verify,
            max_connect_retries: transport.max_connect_retries,
            connect_retry_backoff_ns: transport.connect_retry_backoff_ns,
            ..GrpcClientConfig::default()
        },
        connection_reuse: grpc_reuse(transport.connection_reuse),
        session_header: transport.session_header,
        capture_raw: transport.raw_capture,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::turn_execution::ExecutionTransportPolicy;

    #[test]
    fn grpc_binds_all_neutral_execution_policy_fields() {
        let config = grpc_sink_config(ExecutionTransportPolicy {
            total_timeout_ns: Some(19),
            ssl_verify: false,
            max_connect_retries: 4,
            connect_retry_backoff_ns: 23,
            connection_reuse: ConnectionReuseStrategy::StickyUserSessions,
            session_header: Some("x-session".into()),
            raw_capture: true,
        });

        assert_eq!(config.client.total_timeout_ns, Some(19));
        assert!(!config.client.ssl_verify);
        assert_eq!(config.client.max_connect_retries, 4);
        assert_eq!(config.client.connect_retry_backoff_ns, 23);
        assert_eq!(
            config.connection_reuse,
            GrpcConnectionReuseStrategy::StickyUserSessions
        );
        assert_eq!(config.session_header.as_deref(), Some("x-session"));
        assert!(config.capture_raw);
    }
}

fn grpc_reuse(reuse: ConnectionReuseStrategy) -> GrpcConnectionReuseStrategy {
    match reuse {
        ConnectionReuseStrategy::Pooled => GrpcConnectionReuseStrategy::Pooled,
        ConnectionReuseStrategy::Never => GrpcConnectionReuseStrategy::Never,
        ConnectionReuseStrategy::StickyUserSessions => {
            GrpcConnectionReuseStrategy::StickyUserSessions
        }
    }
}
