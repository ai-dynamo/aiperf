// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport for protocol-v2 turn execution.
//!
//! gRPC contributes only a [`WorkerSink`] (its [`GrpcTransportSink`]) and the
//! [`GrpcSinkBuilder`] that constructs one per worker. It owns NO execution
//! machinery: worker threads, measurement, drain, cancellation, and streaming
//! all live once in [`crate::engine::turn_execution`] and are shared with HTTP
//! (and any future socket transport) via [`build_native`]. This is the Rust
//! analogue of the Python `BaseTransport.send_request` seam — a transport is a
//! sink, not an execution model.

use std::rc::Rc;
use std::sync::Arc;

use crate::clock::Clock;
use crate::endpoints::PreparedEndpointTable;
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::InferenceDimensions;
use crate::multiturn::TurnToSend;
use crate::scheduled::TurnResponseObserver;
use crate::transport::core::ConnectionReuseStrategy;
use crate::transport::core::{DispatchResult, MeasuredContext};
use crate::transport::grpc::{
    ConnectionReuseStrategy as GrpcConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig,
};
use crate::transport::grpc::{GrpcTransportSink, GrpcTransportSinkConfig};
use crate::transport::http::{PreparedTurn, RequestExecutor, TransportSinkConfig};
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;

use crate::engine::turn_execution::{
    ExecutionBackendConfig, ExecutionSinkBuilder, PreparedEndpointTableFactory,
    RequestExecutorFactory, WorkerSink, build_native,
};

/// V2-only native gRPC execution factory.
///
/// A thin adapter: it validates gRPC-specific prerequisites, then hands a
/// [`GrpcSinkBuilder`] to the shared [`build_native`] executor. There is no gRPC
/// worker loop.
#[derive(Clone, Debug, Default)]
pub struct GrpcExecutionFactory {
    bindings: Option<GrpcBindingRegistry>,
}

impl GrpcExecutionFactory {
    /// Construct with the built-in open gRPC binding registry.
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
        build_native(
            GrpcSinkBuilder::from_config(&config, bindings),
            workers,
            coordinator_clock,
            anchor,
        )
    }
}

/// Built-in gRPC sink builder: constructs a worker-local [`GrpcTransportSink`].
pub struct GrpcSinkBuilder {
    base_urls: Vec<String>,
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
    bindings: GrpcBindingRegistry,
}

impl GrpcSinkBuilder {
    /// Capture the per-run gRPC sink inputs from the resolved backend config.
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

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <GrpcTransportSink as RequestExecutor>::inference_dimensions(self, turn)
    }

    fn supports_response_streaming(&self) -> bool {
        // gRPC does not relay intermediate responses to a live observer today.
        false
    }

    async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        _responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        // `_responses` is unused: `supports_response_streaming` is false, so the
        // shared worker loop never opens a live response channel for gRPC.
        GrpcTransportSink::dispatch_measured(self, observer, turn, context, on_first_token).await
    }
}

fn prepare_grpc_sink(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: crate::transport::http::TransportSinkConfig,
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

/// Assemble a v2 gRPC sink from an already-built worker-local prepared endpoint
/// table, translating the transport-neutral [`TransportSinkConfig`] into the
/// gRPC client/reuse/session policy and preparing dense per-endpoint bindings.
///
/// This is the shared sink-construction core reused both by the scheduled gRPC
/// worker path (via [`prepare_grpc_sink`], which resolves the table from a
/// [`PreparedEndpointTableFactory`]) and by the graph endpoint runtime
/// factory (which owns its `PreparedEndpointTable` directly). Keeping one core
/// keeps the gRPC client/reuse/session mapping in a single place.
///
/// [`TransportSinkConfig`]: crate::transport::http::TransportSinkConfig
pub(crate) fn grpc_sink_with_endpoints(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: crate::transport::http::TransportSinkConfig,
    bindings: GrpcBindingRegistry,
    endpoints: Rc<PreparedEndpointTable>,
) -> Result<GrpcTransportSink> {
    GrpcTransportSink::new(
        clock,
        start_ns,
        base_urls,
        model,
        GrpcTransportSinkConfig {
            client: GrpcClientConfig {
                total_timeout_ns: transport.client.total_timeout_ns,
                trace_chunks: transport.client.collect_trace_chunks,
                // Carry the endpoint's ssl_verify to the gRPC TLS builder so a
                // `grpcs://` run can disable cert verification (self-signed test
                // servers), mirroring the HTTP transport.
                ssl_verify: transport.client.ssl_verify,
                ..GrpcClientConfig::default()
            },
            connection_reuse: grpc_reuse(transport.connection_reuse),
            session_header: transport.session_header,
        },
        bindings,
    )?
    .with_prepared_endpoints(endpoints)
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
