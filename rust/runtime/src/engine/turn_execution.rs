// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral placement for turn execution.
//!
//! The benchmark has one scheduler and one logical `TurnDispatcher`. Each
//! transport contributes only a [`WorkerSink`] (the Rust analogue of Python's
//! `BaseTransport.send_request`) built through an [`ExecutionSinkBuilder`]; HTTP,
//! gRPC, and any future socket transport share the one dispatch/measure/stream
//! path written once against this seam and never bring their own execution model.
//!
//! There is ONE thread-per-core execution mechanism, and it lives ABOVE the
//! transport: [`run_sharded_scheduled`](crate::engine::sharded_scheduled) runs the
//! whole scheduled pipeline — scheduler, admission, dispatch, transport, and
//! capture — independently on each sub-cell OS thread, so a transport sink is
//! always co-located on its own reactor with no cross-thread per-request hop. This
//! module therefore builds only a single co-located sink; the factory and sink
//! traits remain the injection point for a future cross-node transport.

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

/// Inputs available to an execution-placement factory for one benchmark run.
pub struct ExecutionBackendConfig {
    /// Number of execution workers requested by resolved Config v2.
    pub workers: usize,
    /// Coordinator-local clock used by direct execution.
    pub coordinator_clock: Rc<dyn Clock>,
    /// Copyable origin used to construct worker-local clocks on one timeline.
    pub real_clock_anchor: RealClockAnchor,
    /// Ordered inference endpoint list.
    pub base_urls: Vec<String>,
    /// Effective primary model.
    pub model: String,
    /// Fully resolved transport policy.
    pub transport: TransportSinkConfig,
    /// Optional worker-local open endpoint preparation.
    ///
    /// The factory runs independently on every native worker, preserving the
    /// same dense-key table contract a future remote placement can implement.
    pub prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
}

/// Worker-local prepared endpoint table construction.
///
/// Implementations retain registry/factory state only. Each placement worker
/// calls this seam before accepting commands, so prepared endpoint objects and
/// credentials never cross a thread or remote execution boundary.
pub trait PreparedEndpointTableFactory: Send + Sync {
    /// Build one complete deterministic dense-key table for a worker.
    fn prepare_worker(&self) -> Result<PreparedEndpointTable>;
}

/// Composition seam for local (co-located) or a future remote execution placement.
pub trait RequestExecutorFactory: Send + Sync {
    /// Construct the backend used below the run's single logical dispatcher.
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>>;
}

/// The worker-facing contract of a transport sink — the Rust analogue of the
/// Python `BaseTransport.send_request` seam.
///
/// This is the ONLY thing that differs between native transports: given a
/// prepared turn, drive it to terminal against a real server and feed the
/// worker-local observer. The dispatch/measure/stream path is written once over
/// this trait, so HTTP, gRPC, and any future socket transport share one
/// measurement path, one drain, one cancellation, and one streaming relay — a
/// transport never brings its own execution model, only its sink. Thread-per-core
/// parallelism is provided by the sharded scheduled runtime above the transport.
#[async_trait(?Send)]
pub trait WorkerSink {
    /// Anchor the sink's timestamp origin to the run origin (shared with the
    /// worker observer so TTFT/ITL are not offset by setup duration).
    fn set_run_origin(&self, origin_ns: i64);

    /// Report the coordinator-known inference dimensions for one turn (used by
    /// the dimension probe; no IO).
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

/// Worker-local sink construction — the transport-specific half of execution.
///
/// One implementation per transport (HTTP, gRPC). Everything above it —
/// measurement, drain, cancellation, streaming — is generic over this builder, so
/// adding a transport means writing a builder, never a second execution backend.
/// The builder is `Send + Sync` (moved onto each sharded sub-cell OS thread) and
/// constructs the `!Send` sink inside that thread's reactor.
pub trait ExecutionSinkBuilder: Send + Sync + 'static {
    /// The worker-local sink this transport drives.
    type Sink: WorkerSink + RequestExecutor + 'static;

    /// Short worker-thread name infix (e.g. `"http"`, `"grpc"`).
    fn label(&self) -> &'static str;

    /// Build one worker-local sink on `clock` for `worker_id`.
    fn build_sink(&self, clock: Rc<dyn Clock>, worker_id: usize) -> Result<Self::Sink>;
}

/// Built-in HTTP sink builder: constructs a worker-local [`TransportSink`].
pub struct HttpSinkBuilder {
    base_urls: Vec<String>,
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
}

impl HttpSinkBuilder {
    /// Capture the per-run HTTP sink inputs from the resolved backend config.
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

/// Build the native execution backend for any transport sink builder.
///
/// There is exactly ONE thread-per-core execution mechanism, and it lives ABOVE
/// the transport: [`run_sharded_scheduled`](crate::engine::sharded_scheduled) runs
/// the whole scheduled pipeline — scheduler, admission, dispatch, transport, and
/// capture — independently on each sub-cell OS thread, so every worker's transport
/// sink is co-located on its own reactor. This factory therefore only ever builds a
/// single co-located sink; `workers > 1` never reaches here because every workload
/// shape shards (`shardable == workers > 1` in
/// [`execute_native_inner`](crate::engine::execute), including static-accuracy),
/// and the sharded runtime hands each of its threads `workers == 1`. A `workers > 1`
/// request is a wiring bug rather than a second execution model, so it fails closed.
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

/// Native local execution factory.
///
/// Builds one co-located transport sink on the coordinator reactor; OS-thread
/// parallelism for `workers > 1` runs comes from the sharded scheduled runtime
/// above the transport, not from this factory.
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

/// Construct one worker-local HTTP [`TransportSink`], optionally binding a
/// worker-local prepared endpoint table so prepared endpoint objects and
/// credentials never cross a thread boundary.
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
