// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native local and thread-per-core placement for protocol-v2 gRPC turns.
//!
//! Scheduling remains on one coordinator reactor. Two or more configured
//! workers receive owned, scheduler-free commands over bounded queues and each
//! own a current-thread Tokio runtime, Tonic channels, prepared endpoint table,
//! and dense gRPC binding table. No transport object or hot-path observer is
//! shared across worker threads.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;

use aiperf::grpc::{GrpcTransportSink, GrpcTransportSinkConfig};
use aiperf::http::{
    HttpTurnDispatchResult, HttpTurnExecutionBackend, MeasuredTurnContext, MeasuredTurnOutcome,
    PreparedHttpTurn,
};
use aiperf::metrics::NativeMetricsObserver;
use aiperf::multiturn::TurnToSend;
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_metrics::{InferenceDimensions, MetricsConfig, RecordIngest};
use aiperf_transport_grpc::{
    ConnectionReuseStrategy as GrpcConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig,
};
use aiperf_transport_http::models::ConnectionReuseStrategy;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, HttpPreparedEndpointTableFactory,
};

const WORKER_QUEUE_CAPACITY: usize = 256;

/// V2-only native gRPC execution factory.
#[derive(Clone, Debug, Default)]
pub struct NativeGrpcExecutionBackendFactory {
    bindings: Option<GrpcBindingRegistry>,
}

impl NativeGrpcExecutionBackendFactory {
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

impl HttpExecutionBackendFactory for NativeGrpcExecutionBackendFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        ensure!(
            config.workers > 0,
            "gRPC execution workers must be positive"
        );
        ensure!(
            config.prepared_endpoints.is_some(),
            "native gRPC execution requires protocol-v2 prepared endpoints"
        );
        if config.workers == 1 {
            let bindings = self
                .bindings
                .clone()
                .map(Ok)
                .unwrap_or_else(GrpcBindingRegistry::builtin)?;
            return Ok(Rc::new(prepare_grpc_sink(
                config.coordinator_clock,
                0,
                &config.base_urls,
                config.model,
                config.transport,
                config.prepared_endpoints.as_deref(),
                bindings,
            )?));
        }
        let bindings = self
            .bindings
            .clone()
            .map(Ok)
            .unwrap_or_else(GrpcBindingRegistry::builtin)?;
        Ok(Rc::new(ThreadPerCoreGrpcExecutionBackend::new(
            config, bindings,
        )?))
    }
}

#[derive(Debug)]
enum ObserverEvent {
    Admit {
        uuid: Uuid,
        at_ms: f64,
        reused_input_tokens: usize,
    },
    Token {
        uuid: Uuid,
        at_ms: f64,
    },
    ClassifiedToken {
        uuid: Uuid,
        at_ms: f64,
        kind: ObservedTokenKind,
    },
    Usage {
        uuid: Uuid,
        usage: ObservedUsage,
    },
    EndpointMetrics {
        uuid: Uuid,
        metrics: ObservedEndpointMetrics,
    },
    Terminal {
        uuid: Uuid,
        status: ReplayTerminalStatus,
    },
}

impl ObserverEvent {
    fn replay(self, observer: &dyn RequestObserver, origin_ms: f64) {
        match self {
            Self::Admit {
                uuid,
                at_ms,
                reused_input_tokens,
            } => observer.on_admit(uuid, at_ms - origin_ms, reused_input_tokens),
            Self::Token { uuid, at_ms } => observer.on_token(uuid, at_ms - origin_ms),
            Self::ClassifiedToken { uuid, at_ms, kind } => {
                observer.on_classified_token(uuid, at_ms - origin_ms, kind);
            }
            Self::Usage { uuid, usage } => observer.on_usage(uuid, usage),
            Self::EndpointMetrics { uuid, metrics } => {
                observer.on_endpoint_metrics(uuid, metrics);
            }
            Self::Terminal { uuid, status } => observer.on_terminal(uuid, status),
        }
    }
}

#[derive(Default)]
struct BufferedObserver {
    events: RefCell<Vec<ObserverEvent>>,
}

impl BufferedObserver {
    fn take(&self) -> Vec<ObserverEvent> {
        self.events.take()
    }
}

impl RequestObserver for BufferedObserver {
    fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}

    fn on_admit(&self, uuid: Uuid, at_ms: f64, reused_input_tokens: usize) {
        self.events.borrow_mut().push(ObserverEvent::Admit {
            uuid,
            at_ms,
            reused_input_tokens,
        });
    }

    fn on_token(&self, uuid: Uuid, at_ms: f64) {
        self.events
            .borrow_mut()
            .push(ObserverEvent::Token { uuid, at_ms });
    }

    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        self.events
            .borrow_mut()
            .push(ObserverEvent::ClassifiedToken { uuid, at_ms, kind });
    }

    fn on_usage(&self, uuid: Uuid, usage: ObservedUsage) {
        self.events
            .borrow_mut()
            .push(ObserverEvent::Usage { uuid, usage });
    }

    fn on_endpoint_metrics(&self, uuid: Uuid, metrics: ObservedEndpointMetrics) {
        self.events
            .borrow_mut()
            .push(ObserverEvent::EndpointMetrics { uuid, metrics });
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        self.events
            .borrow_mut()
            .push(ObserverEvent::Terminal { uuid, status });
    }
}

struct WorkerReply {
    result: Result<HttpTurnDispatchResult>,
    /// Buffered observations for the legacy `execute_turn(observer)` replay path;
    /// empty for the worker-local measured path.
    events: Vec<ObserverEvent>,
    /// Non-consuming cloned record for a live sink, when requested.
    live_record: Option<RecordIngest>,
}

struct WorkerCommand {
    turn: PreparedHttpTurn,
    /// Present for the worker-local measured path; absent for the buffered path.
    context: Option<MeasuredTurnContext>,
    first_token: oneshot::Sender<i64>,
    completed: oneshot::Sender<WorkerReply>,
}

/// Control-plane message multiplexed onto each gRPC worker's command channel.
enum WorkerMessage {
    Configure {
        config: MetricsConfig,
        origin_ns: i64,
    },
    Command(Box<WorkerCommand>),
    Drain {
        end_ns: i64,
        reply: std::sync::mpsc::SyncSender<Vec<(Uuid, RecordIngest)>>,
    },
}

struct ThreadPerCoreGrpcExecutionBackend {
    senders: RefCell<Option<Vec<mpsc::Sender<WorkerMessage>>>>,
    threads: RefCell<Vec<JoinHandle<Result<()>>>>,
    next_worker: Cell<usize>,
    run_origin_ns: Cell<Option<i64>>,
    dimension_sink: GrpcTransportSink,
}

impl ThreadPerCoreGrpcExecutionBackend {
    fn new(config: HttpExecutionBackendConfig, bindings: GrpcBindingRegistry) -> Result<Self> {
        ensure!(
            config.workers > 1,
            "thread-per-core gRPC execution requires at least two workers"
        );
        let dimension_sink = prepare_grpc_sink(
            config.coordinator_clock.clone(),
            0,
            &config.base_urls,
            config.model.clone(),
            config.transport.clone(),
            config.prepared_endpoints.as_deref(),
            bindings.clone(),
        )?;
        let mut senders = Vec::with_capacity(config.workers);
        let mut threads = Vec::with_capacity(config.workers);
        for worker_id in 0..config.workers {
            let (sender, receiver) = mpsc::channel::<WorkerMessage>(WORKER_QUEUE_CAPACITY);
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
            let base_urls = config.base_urls.clone();
            let model = config.model.clone();
            let transport = config.transport.clone();
            let endpoints = config.prepared_endpoints.clone();
            let bindings = bindings.clone();
            let anchor = config.real_clock_anchor;
            let thread = match std::thread::Builder::new()
                .name(format!("aiperf-grpc-{worker_id}"))
                .spawn(move || {
                    let result = run_worker_thread(
                        receiver, anchor, base_urls, model, transport, endpoints, bindings,
                        started_tx,
                    );
                    if let Err(error) = &result {
                        tracing::error!(worker_id, error = %error, "gRPC execution worker failed");
                    }
                    result
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error).context("spawning gRPC execution worker");
                }
            };
            match started_rx.recv() {
                Ok(Ok(())) => {
                    senders.push(sender);
                    threads.push(thread);
                }
                Ok(Err(message)) => {
                    drop(sender);
                    let _ = thread.join();
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(anyhow!(message))
                        .context(format!("starting gRPC execution worker {worker_id}"));
                }
                Err(error) => {
                    drop(sender);
                    let _ = thread.join();
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error)
                        .context(format!("receiving gRPC worker {worker_id} startup status"));
                }
            }
        }
        Ok(Self {
            senders: RefCell::new(Some(senders)),
            threads: RefCell::new(threads),
            next_worker: Cell::new(0),
            run_origin_ns: Cell::new(None),
            dimension_sink,
        })
    }

    fn shutdown_workers(&self) -> Result<()> {
        drop(self.senders.borrow_mut().take());
        join_worker_threads(self.threads.take())
    }
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for ThreadPerCoreGrpcExecutionBackend {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(
            self.run_origin_ns.replace(Some(start_ns)).is_none(),
            "gRPC execution run origin was configured more than once"
        );
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <GrpcTransportSink as HttpTurnExecutionBackend>::inference_dimensions(
            &self.dimension_sink,
            turn,
        )
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        let run_origin_ns = self.origin()?;
        let reply = self.execute_command(turn, None, on_first_token).await?;
        for event in reply.events {
            event.replay(observer, run_origin_ns as f64 / 1_000_000.0);
        }
        reply.result
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        let senders = self.senders.borrow();
        let senders = senders
            .as_ref()
            .ok_or_else(|| anyhow!("gRPC execution backend is shut down"))?;
        for sender in senders.iter() {
            sender
                .try_send(WorkerMessage::Configure {
                    config: config.clone(),
                    origin_ns,
                })
                .map_err(|_| anyhow!("gRPC execution worker rejected measurement configuration"))?;
        }
        Ok(())
    }

    async fn execute_turn_measured(
        &self,
        turn: PreparedHttpTurn,
        context: MeasuredTurnContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredTurnOutcome> {
        let reply = self
            .execute_command(turn, Some(context), on_first_token)
            .await?;
        Ok(MeasuredTurnOutcome {
            result: reply.result?,
            live_record: reply.live_record,
        })
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        let senders = {
            let senders = self.senders.borrow();
            senders
                .as_ref()
                .ok_or_else(|| anyhow!("gRPC execution backend is shut down"))?
                .clone()
        };
        let mut receivers = Vec::with_capacity(senders.len());
        for sender in &senders {
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            sender
                .try_send(WorkerMessage::Drain {
                    end_ns,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow!("gRPC execution worker rejected a drain request"))?;
            receivers.push(reply_rx);
        }
        let mut records = Vec::new();
        for receiver in receivers {
            let worker_records = receiver
                .recv()
                .map_err(|_| anyhow!("gRPC execution worker dropped before draining records"))?;
            records.extend(worker_records);
        }
        Ok(records)
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdown_workers()
    }
}

impl ThreadPerCoreGrpcExecutionBackend {
    fn origin(&self) -> Result<i64> {
        self.run_origin_ns
            .get()
            .ok_or_else(|| anyhow!("gRPC execution run origin is not configured"))
    }

    async fn execute_command(
        &self,
        turn: PreparedHttpTurn,
        context: Option<MeasuredTurnContext>,
        on_first_token: &dyn Fn(i64),
    ) -> Result<WorkerReply> {
        let _run_origin_ns = self.origin()?;
        let sender = {
            let senders = self.senders.borrow();
            let senders = senders
                .as_ref()
                .ok_or_else(|| anyhow!("gRPC execution backend is shut down"))?;
            let index = self.next_worker.get() % senders.len();
            self.next_worker.set(index.wrapping_add(1));
            senders[index].clone()
        };
        let (first_token_tx, mut first_token_rx) = oneshot::channel();
        let (completed_tx, mut completed_rx) = oneshot::channel();
        sender
            .send(WorkerMessage::Command(Box::new(WorkerCommand {
                turn,
                context,
                first_token: first_token_tx,
                completed: completed_tx,
            })))
            .await
            .map_err(|_| anyhow!("gRPC worker stopped before accepting a command"))?;
        let mut first_token_channel_done = false;
        let reply = loop {
            tokio::select! {
                biased;
                first = &mut first_token_rx, if !first_token_channel_done => {
                    first_token_channel_done = true;
                    if let Ok(ttft_ns) = first {
                        on_first_token(ttft_ns);
                    }
                }
                completed = &mut completed_rx => {
                    break completed.map_err(|_| {
                        anyhow!("gRPC worker dropped a command before completion")
                    })?;
                }
            }
        };
        if !first_token_channel_done && let Ok(ttft_ns) = first_token_rx.try_recv() {
            on_first_token(ttft_ns);
        }
        Ok(reply)
    }
}

impl Drop for ThreadPerCoreGrpcExecutionBackend {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown_workers() {
            tracing::error!(error = %error, "failed to shut down gRPC execution workers");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker_thread(
    receiver: mpsc::Receiver<WorkerMessage>,
    anchor: RealClockAnchor,
    base_urls: Vec<String>,
    model: String,
    transport: aiperf::http::TransportSinkConfig,
    prepared_endpoints: Option<Arc<dyn HttpPreparedEndpointTableFactory>>,
    bindings: GrpcBindingRegistry,
    started: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) -> Result<()> {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("creating gRPC worker Tokio runtime");
        }
    };
    let clock = RealClock::from_anchor(anchor);
    let sink = match prepare_grpc_sink(
        clock.clone(),
        0,
        &base_urls,
        model,
        transport,
        prepared_endpoints.as_deref(),
        bindings,
    ) {
        Ok(sink) => Rc::new(sink),
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("constructing worker-local gRPC transport");
        }
    };
    if started.send(Ok(())).is_err() {
        return Ok(());
    }
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, run_worker(receiver, sink, clock));
    Ok(())
}

fn prepare_grpc_sink(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: aiperf::http::TransportSinkConfig,
    prepared_endpoints: Option<&dyn HttpPreparedEndpointTableFactory>,
    bindings: GrpcBindingRegistry,
) -> Result<GrpcTransportSink> {
    let endpoints = prepared_endpoints
        .ok_or_else(|| anyhow!("gRPC execution requires a prepared endpoint table factory"))?
        .prepare_worker()?;
    GrpcTransportSink::new(
        clock,
        start_ns,
        base_urls,
        model,
        GrpcTransportSinkConfig {
            client: GrpcClientConfig {
                total_timeout_ns: transport.client.total_timeout_ns,
                trace_chunks: transport.client.collect_trace_chunks,
                ..GrpcClientConfig::default()
            },
            connection_reuse: grpc_reuse(transport.connection_reuse),
            session_header: transport.session_header,
        },
        bindings,
    )?
    .with_prepared_endpoints(Rc::new(endpoints))
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

async fn run_worker(
    mut receiver: mpsc::Receiver<WorkerMessage>,
    sink: Rc<GrpcTransportSink>,
    clock: Rc<dyn Clock>,
) {
    let mut jobs = JoinSet::new();
    let mut accepting = true;
    let mut observer: Option<Rc<NativeMetricsObserver>> = None;
    let mut pending_drain: Option<(i64, std::sync::mpsc::SyncSender<Vec<(Uuid, RecordIngest)>>)> =
        None;
    while accepting || !jobs.is_empty() {
        tokio::select! {
            message = receiver.recv(), if accepting => {
                match message {
                    Some(WorkerMessage::Configure { config, origin_ns }) => {
                        observer = Some(Rc::new(NativeMetricsObserver::new(
                            clock.clone(),
                            origin_ns,
                            config,
                        )));
                    }
                    Some(WorkerMessage::Command(command)) => {
                        let sink = sink.clone();
                        let observer = observer.clone();
                        jobs.spawn_local(async move {
                            execute_worker_command(sink, observer, *command).await;
                        });
                    }
                    Some(WorkerMessage::Drain { end_ns, reply }) => {
                        accepting = false;
                        pending_drain = Some((end_ns, reply));
                    }
                    None => accepting = false,
                }
            }
            completed = jobs.join_next(), if !jobs.is_empty() => {
                if let Some(Err(error)) = completed {
                    tracing::error!(error = %error, "gRPC execution task panicked");
                }
            }
        }
    }
    if let Some((end_ns, reply)) = pending_drain {
        let records = observer
            .map(|observer| observer.take_finalizer_at(end_ns).finish_with_records().records)
            .unwrap_or_default();
        let _ = reply.send(records);
    }
}

async fn execute_worker_command(
    sink: Rc<GrpcTransportSink>,
    worker_observer: Option<Rc<NativeMetricsObserver>>,
    command: WorkerCommand,
) {
    let WorkerCommand {
        turn,
        context,
        first_token,
        completed,
    } = command;
    let uuid = turn.request.uuid;
    let first_token = RefCell::new(Some(first_token));
    let on_first_token = |ttft_ns| {
        if let Some(sender) = first_token.borrow_mut().take() {
            let _ = sender.send(ttft_ns);
        }
    };
    let reply = match &context {
        Some(context) => match &worker_observer {
            Some(observer) => {
                let result = sink
                    .dispatch_prepared_turn_measured(observer, turn, context, &on_first_token)
                    .await;
                let live_record = context
                    .wants_live_record
                    .then(|| observer.snapshot_record(uuid, 0))
                    .flatten();
                WorkerReply {
                    result,
                    events: Vec::new(),
                    live_record,
                }
            }
            None => WorkerReply {
                result: Err(anyhow!(
                    "worker-local measurement was not configured before a measured command"
                )),
                events: Vec::new(),
                live_record: None,
            },
        },
        None => {
            let observer = Rc::new(BufferedObserver::default());
            let result = sink
                .dispatch_prepared_turn_collect_record(turn, observer.as_ref(), &on_first_token)
                .await;
            WorkerReply {
                result,
                events: observer.take(),
                live_record: None,
            }
        }
    };
    drop(first_token.borrow_mut().take());
    let _ = completed.send(reply);
}

fn join_worker_threads(threads: Vec<JoinHandle<Result<()>>>) -> Result<()> {
    let mut errors = Vec::new();
    for thread in threads {
        match thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => errors.push(format!("{error:#}")),
            Err(_) => errors.push("gRPC execution worker panicked".to_string()),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "{} gRPC execution worker(s) failed: {}",
            errors.len(),
            errors.join("; ")
        ))
    }
}
