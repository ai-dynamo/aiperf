// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral placement for turn execution.
//!
//! Each transport supplies a worker-local [`WorkerSink`] through an
//! [`ExecutionSinkBuilder`]. Scheduling, measurement, streaming, cancellation,
//! and drain remain transport-independent.

use std::rc::Rc;
use std::sync::Arc;

use crate::clock::{Clock, RealClockAnchor};
use crate::endpoints::PreparedEndpointTable;
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::InferenceDimensions;
use crate::multiturn::TurnToSend;
use crate::scheduled::TurnResponseObserver;
use crate::transport::core::{DispatchResult, MeasuredContext, PreparedTurn, RequestExecutor};
use crate::transport::http::{TransportSink, TransportSinkConfig};
use anyhow::{Result, ensure};
use async_trait::async_trait;

/// Inputs for one execution backend.
pub struct ExecutionBackendConfig {
    /// Number of requested execution workers.
    pub workers: usize,
    /// Coordinator-local execution clock.
    pub coordinator_clock: Rc<dyn Clock>,
    /// Origin shared by worker-local clocks.
    pub real_clock_anchor: RealClockAnchor,
    /// Ordered inference endpoint list.
    pub base_urls: Vec<String>,
    /// Effective primary model.
    pub model: String,
    /// Fully resolved transport policy.
    pub transport: TransportSinkConfig,
    /// Optional worker-local endpoint preparation.
    ///
    /// Each worker receives an independent dense-key table.
    pub prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
}

/// Constructs worker-local prepared endpoint tables.
pub trait PreparedEndpointTableFactory: Send + Sync {
    /// Build one deterministic dense-key table.
    fn prepare_worker(&self) -> Result<PreparedEndpointTable>;
}

/// Constructs the request executor for a run.
pub trait RequestExecutorFactory: Send + Sync {
    /// Construct the backend used by the run's dispatcher.
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>>;
}

/// Worker-facing contract for a transport sink.
#[async_trait(?Send)]
pub trait WorkerSink {
    /// Anchor the sink's timestamp origin to the run origin (shared with the
    /// worker observer so TTFT/ITL are not offset by setup duration).
    fn set_run_origin(&self, origin_ns: i64);

    /// Report inference dimensions without performing I/O.
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions;

    /// Whether this sink can stream intermediate responses to a live observer.
    fn supports_response_streaming(&self) -> bool;

    /// Drive one prepared turn to terminal, recording into `observer`. When the
    /// sink streams and `responses` is `Some`, intermediate parsed responses are
    /// forwarded live; otherwise `responses` is ignored.
    async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult>;

    /// Optional one-shot warm round-trip discarded before timed issuance. The
    /// default is a no-op for transports that do not warm connections.
    async fn prewarm(&self, _turn: PreparedTurn) -> Result<()> {
        Ok(())
    }
}

#[async_trait(?Send)]
impl WorkerSink for TransportSink {
    fn set_run_origin(&self, origin_ns: i64) {
        TransportSink::set_run_origin(self, origin_ns);
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <TransportSink as crate::scheduled::TurnDispatcher>::inference_dimensions(self, turn)
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        TransportSink::dispatch_measured(self, observer, turn, context, on_first_token, responses)
            .await
    }

    async fn prewarm(&self, turn: PreparedTurn) -> Result<()> {
        <TransportSink as RequestExecutor>::prewarm(self, turn).await
    }
}

/// Constructs a `!Send` transport sink inside each worker reactor.
pub trait ExecutionSinkBuilder: Send + Sync + 'static {
    /// The worker-local sink this transport drives.
    type Sink: WorkerSink + RequestExecutor + 'static;

    /// Short worker-thread name infix (e.g. `"http"`, `"grpc"`).
    fn label(&self) -> &'static str;

    /// Build one worker-local sink on `clock` for `worker_id`.
    fn build_sink(&self, clock: Rc<dyn Clock>, worker_id: usize) -> Result<Self::Sink>;
}

/// Constructs worker-local HTTP transport sinks.
pub struct HttpSinkBuilder {
    base_urls: Vec<String>,
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
}

impl HttpSinkBuilder {
    pub fn from_config(config: &ExecutionBackendConfig) -> Self {
        Self {
            base_urls: config.base_urls.clone(),
            model: config.model.clone(),
            transport: config.transport.clone(),
            prepared_endpoints: config.prepared_endpoints.clone(),
        }
    }
}

impl ExecutionSinkBuilder for HttpSinkBuilder {
    type Sink = TransportSink;

    fn label(&self) -> &'static str {
        "http"
    }

    fn build_sink(&self, clock: Rc<dyn Clock>, _worker_id: usize) -> Result<TransportSink> {
        prepare_transport_sink(
            clock,
            0,
            &self.base_urls,
            self.model.clone(),
            self.transport.clone(),
            self.prepared_endpoints.as_deref(),
        )
    }
}

/// Build one co-located transport sink.
///
/// Thread-per-core execution constructs one backend per shard, so this function
/// accepts exactly one worker.
pub(crate) fn build_native<B: ExecutionSinkBuilder>(
    builder: B,
    workers: usize,
    coordinator_clock: Rc<dyn Clock>,
    _real_clock_anchor: RealClockAnchor,
) -> Result<Rc<dyn RequestExecutor>> {
    ensure!(workers > 0, "execution workers must be positive");
    ensure!(
        workers == 1,
        "the transport factory builds a single co-located sink; thread-per-core \
         parallelism is provided by the sharded scheduled runtime above the transport, \
         not by a cross-thread transport hop (got workers = {workers})"
    );
    Ok(Rc::new(builder.build_sink(coordinator_clock, 0)?))
}

/// Native HTTP execution factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct HttpExecutionFactory;

impl RequestExecutorFactory for HttpExecutionFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        let workers = config.workers;
        let coordinator_clock = config.coordinator_clock.clone();
        let anchor = config.real_clock_anchor;
        build_native(
            HttpSinkBuilder::from_config(&config),
            workers,
            coordinator_clock,
            anchor,
        )
    }
}

/// Construct one worker-local HTTP [`TransportSink`].
fn prepare_transport_sink(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<&dyn PreparedEndpointTableFactory>,
) -> Result<TransportSink> {
    let sink = TransportSink::new_multi_configured(clock, start_ns, base_urls, model, transport)?;
    match prepared_endpoints {
        Some(factory) => Ok(sink.with_prepared_endpoints(Rc::new(factory.prepare_worker()?))),
        None => Ok(sink),
    }
}
