// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable placement for HTTP turn execution.
//!
//! The benchmark has one scheduler and one logical `TurnDispatcher`. This
//! module places only its prepared HTTP commands: either directly on the
//! coordinator reactor or across thread-per-core reactors. The factory and
//! backend traits are the injection point for a future cross-node transport;
//! phase, workload, admission, observer, and reporting code never knows where a
//! command ran.

use std::cell::{Cell, RefCell};
use std::future::poll_fn;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context as TaskContext, Poll};
use std::thread::JoinHandle;

use aiperf::http::{
    HttpTurnDispatchResult, HttpTurnExecutionBackend, PreparedHttpTurn, TransportSink,
    TransportSinkConfig,
};
use aiperf::multiturn::TurnToSend;
use aiperf::scheduled::TurnResponseObserver;
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_endpoints::{ParsedResponse, PreparedEndpointTable};
use aiperf_metrics::InferenceDimensions;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::PollSender;
use uuid::Uuid;

const WORKER_QUEUE_CAPACITY: usize = 256;
const WORKER_RESPONSE_CAPACITY: usize = 256;

/// Inputs available to an execution-placement factory for one benchmark run.
pub struct HttpExecutionBackendConfig {
    /// Number of HTTP execution workers requested by resolved Config v2.
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
    pub prepared_endpoints: Option<Arc<dyn HttpPreparedEndpointTableFactory>>,
}

/// Worker-local prepared endpoint table construction.
///
/// Implementations retain registry/factory state only. Each placement worker
/// calls this seam before accepting commands, so prepared endpoint objects and
/// credentials never cross a thread or remote execution boundary.
pub trait HttpPreparedEndpointTableFactory: Send + Sync {
    /// Build one complete deterministic dense-key table for a worker.
    fn prepare_worker(&self) -> Result<PreparedEndpointTable>;
}

/// Composition seam for local, thread-per-core, or remote execution placement.
pub trait HttpExecutionBackendFactory: Send + Sync {
    /// Construct the backend used below the run's single logical dispatcher.
    fn build(&self, config: HttpExecutionBackendConfig)
    -> Result<Rc<dyn HttpTurnExecutionBackend>>;
}

/// Native local execution factory.
///
/// One worker keeps the transport on the coordinator reactor. Two or more
/// workers create one current-thread Tokio runtime and one transport stack per
/// OS thread; no worker owns scheduling or benchmark policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeHttpExecutionBackendFactory;

impl HttpExecutionBackendFactory for NativeHttpExecutionBackendFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        ensure!(
            config.workers > 0,
            "HTTP execution workers must be positive"
        );
        if config.workers == 1 {
            return Ok(Rc::new(prepare_transport_sink(
                config.coordinator_clock,
                0,
                &config.base_urls,
                config.model,
                config.transport,
                config.prepared_endpoints.as_deref(),
            )?));
        }
        Ok(Rc::new(ThreadPerCoreHttpExecutionBackend::new(config)?))
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
    fn on_arrival(
        &self,
        _uuid: Uuid,
        _arrival_ms: f64,
        _input_length: usize,
        _requested_output_length: usize,
    ) {
        // Arrival is owned by the one coordinator-side dispatcher.
    }

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
    events: Vec<ObserverEvent>,
}

struct WorkerCommand {
    turn: PreparedHttpTurn,
    first_token: oneshot::Sender<i64>,
    responses: Option<mpsc::Sender<ParsedResponse>>,
    completed: oneshot::Sender<WorkerReply>,
    cancellation: PlacementCancellation,
}

#[derive(Clone)]
struct PlacementCancellation {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl PlacementCancellation {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            self.notify.notify_waiters();
        }
    }

    async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.cancelled.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

struct PlacementCancellationGuard {
    cancellation: PlacementCancellation,
    armed: bool,
}

impl PlacementCancellationGuard {
    fn new(cancellation: PlacementCancellation) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PlacementCancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Local thread-per-core placement behind the single dispatcher.
struct ThreadPerCoreHttpExecutionBackend {
    senders: RefCell<Option<Vec<mpsc::Sender<WorkerCommand>>>>,
    threads: RefCell<Vec<JoinHandle<Result<()>>>>,
    next_worker: Cell<usize>,
    run_origin_ns: Cell<Option<i64>>,
    dimension_sink: TransportSink,
}

impl ThreadPerCoreHttpExecutionBackend {
    fn new(config: HttpExecutionBackendConfig) -> Result<Self> {
        ensure!(
            config.workers > 1,
            "thread-per-core execution requires at least two workers"
        );
        let dimension_sink = prepare_transport_sink(
            config.coordinator_clock.clone(),
            0,
            &config.base_urls,
            config.model.clone(),
            config.transport.clone(),
            config.prepared_endpoints.as_deref(),
        )?;
        let mut senders = Vec::with_capacity(config.workers);
        let mut threads = Vec::with_capacity(config.workers);

        for worker_id in 0..config.workers {
            let (sender, receiver) = mpsc::channel(WORKER_QUEUE_CAPACITY);
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
            let base_urls = config.base_urls.clone();
            let model = config.model.clone();
            let transport = config.transport.clone();
            let prepared_endpoints = config.prepared_endpoints.clone();
            let anchor = config.real_clock_anchor;
            let thread = match std::thread::Builder::new()
                .name(format!("aiperf-http-{worker_id}"))
                .spawn(move || {
                    let result = run_worker_thread(
                        receiver,
                        anchor,
                        base_urls,
                        model,
                        transport,
                        prepared_endpoints,
                        started_tx,
                    );
                    if let Err(error) = &result {
                        tracing::error!(worker_id, error = %error, "HTTP execution worker failed");
                    }
                    result
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error).context("spawning HTTP execution worker");
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
                        .context(format!("starting HTTP execution worker {worker_id}"));
                }
                Err(error) => {
                    drop(sender);
                    let _ = thread.join();
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error)
                        .context(format!("receiving HTTP worker {worker_id} startup status"));
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
        // Closing every sender lets each worker drain its bounded queue and
        // local JoinSet before the OS thread exits.
        drop(self.senders.borrow_mut().take());
        join_worker_threads(self.threads.take())
    }
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for ThreadPerCoreHttpExecutionBackend {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(
            self.run_origin_ns.replace(Some(start_ns)).is_none(),
            "HTTP execution run origin was configured more than once"
        );
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <TransportSink as aiperf::scheduled::TurnDispatcher>::inference_dimensions(
            &self.dimension_sink,
            turn,
        )
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        self.execute_command(turn, observer, on_first_token, None)
            .await
    }

    async fn execute_turn_streaming(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<HttpTurnDispatchResult> {
        self.execute_command(turn, observer, on_first_token, Some(responses))
            .await
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdown_workers()
    }
}

impl ThreadPerCoreHttpExecutionBackend {
    async fn execute_command(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<HttpTurnDispatchResult> {
        let run_origin_ns = self
            .run_origin_ns
            .get()
            .ok_or_else(|| anyhow!("HTTP execution run origin is not configured"))?;
        let sender = {
            let senders = self.senders.borrow();
            let senders = senders
                .as_ref()
                .ok_or_else(|| anyhow!("HTTP execution backend is shut down"))?;
            let index = self.next_worker.get() % senders.len();
            self.next_worker.set(index.wrapping_add(1));
            senders[index].clone()
        };
        let (first_token_tx, mut first_token_rx) = oneshot::channel();
        let (response_tx, mut response_rx) = mpsc::channel(WORKER_RESPONSE_CAPACITY);
        let (completed_tx, mut completed_rx) = oneshot::channel();
        let cancellation = PlacementCancellation::new();
        let mut cancellation_guard = PlacementCancellationGuard::new(cancellation.clone());
        sender
            .send(WorkerCommand {
                turn,
                first_token: first_token_tx,
                responses: responses.map(|_| response_tx),
                completed: completed_tx,
                cancellation,
            })
            .await
            .map_err(|_| anyhow!("HTTP execution worker stopped before accepting a command"))?;

        let mut first_token_channel_done = false;
        let mut response_channel_done = responses.is_none();
        let reply = loop {
            tokio::select! {
                biased;
                first = &mut first_token_rx, if !first_token_channel_done => {
                    first_token_channel_done = true;
                    if let Ok(ttft_ns) = first {
                        on_first_token(ttft_ns);
                    }
                }
                response = response_rx.recv(), if !response_channel_done => {
                    match response {
                        Some(response) => {
                            let responses = responses
                                .expect("response channel exists only for streaming dispatch");
                            poll_fn(|context| responses.poll_ready(context)).await?;
                            responses.start_send(response)?;
                        }
                        None => response_channel_done = true,
                    }
                }
                completed = &mut completed_rx => {
                    break completed.map_err(|_| {
                        anyhow!("HTTP execution worker dropped a command before completion")
                    })?;
                }
            }
        };
        if !first_token_channel_done && let Ok(ttft_ns) = first_token_rx.try_recv() {
            on_first_token(ttft_ns);
        }
        if let Some(responses) = responses {
            while let Ok(response) = response_rx.try_recv() {
                poll_fn(|context| responses.poll_ready(context)).await?;
                responses.start_send(response)?;
            }
        }
        for event in reply.events {
            event.replay(observer, run_origin_ns as f64 / 1_000_000.0);
        }
        cancellation_guard.disarm();
        reply.result
    }
}

impl Drop for ThreadPerCoreHttpExecutionBackend {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown_workers() {
            tracing::error!(error = %error, "failed to shut down HTTP execution workers");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker_thread(
    receiver: mpsc::Receiver<WorkerCommand>,
    anchor: RealClockAnchor,
    base_urls: Vec<String>,
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<Arc<dyn HttpPreparedEndpointTableFactory>>,
    started: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) -> Result<()> {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("creating HTTP worker Tokio runtime");
        }
    };
    let clock = RealClock::from_anchor(anchor);
    let sink = match prepare_transport_sink(
        clock,
        0,
        &base_urls,
        model,
        transport,
        prepared_endpoints.as_deref(),
    ) {
        Ok(sink) => Rc::new(sink),
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("constructing worker-local HTTP transport");
        }
    };
    if started.send(Ok(())).is_err() {
        return Ok(());
    }
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, run_worker(receiver, sink));
    Ok(())
}

fn prepare_transport_sink(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    base_urls: &[String],
    model: String,
    transport: TransportSinkConfig,
    prepared_endpoints: Option<&dyn HttpPreparedEndpointTableFactory>,
) -> Result<TransportSink> {
    let sink = TransportSink::new_multi_configured(clock, start_ns, base_urls, model, transport)?;
    match prepared_endpoints {
        Some(factory) => Ok(sink.with_prepared_endpoints(Rc::new(factory.prepare_worker()?))),
        None => Ok(sink),
    }
}

async fn run_worker(mut receiver: mpsc::Receiver<WorkerCommand>, sink: Rc<TransportSink>) {
    let mut jobs = JoinSet::new();
    let mut accepting = true;
    while accepting || !jobs.is_empty() {
        tokio::select! {
            command = receiver.recv(), if accepting => {
                match command {
                    Some(command) => {
                        let sink = sink.clone();
                        jobs.spawn_local(async move {
                            execute_worker_command(sink, command).await;
                        });
                    }
                    None => accepting = false,
                }
            }
            completed = jobs.join_next(), if !jobs.is_empty() => {
                if let Some(Err(error)) = completed {
                    tracing::error!(error = %error, "HTTP execution task panicked");
                }
            }
        }
    }
}

struct WorkerResponseObserver {
    sender: RefCell<PollSender<ParsedResponse>>,
}

impl WorkerResponseObserver {
    fn new(sender: mpsc::Sender<ParsedResponse>) -> Self {
        Self {
            sender: RefCell::new(PollSender::new(sender)),
        }
    }
}

impl TurnResponseObserver for WorkerResponseObserver {
    fn poll_ready(&self, context: &mut TaskContext<'_>) -> Poll<Result<()>> {
        self.sender
            .borrow_mut()
            .poll_reserve(context)
            .map(|result| {
                result.map_err(|_| {
                    anyhow!("HTTP execution response stream receiver closed before terminal")
                })
            })
    }

    fn start_send(&self, response: ParsedResponse) -> Result<()> {
        self.sender
            .borrow_mut()
            .send_item(response)
            .map_err(|_| anyhow!("HTTP execution response stream receiver closed before terminal"))
    }
}

async fn execute_worker_command(sink: Rc<TransportSink>, command: WorkerCommand) {
    let WorkerCommand {
        turn,
        first_token,
        responses,
        completed,
        cancellation,
    } = command;
    let observer = Rc::new(BufferedObserver::default());
    let first_token = RefCell::new(Some(first_token));
    let on_first_token = |ttft_ns| {
        if let Some(sender) = first_token.borrow_mut().take() {
            let _ = sender.send(ttft_ns);
        }
    };
    let response_observer = responses.map(WorkerResponseObserver::new);
    let dispatch = async {
        match response_observer.as_ref() {
            Some(responses) => {
                sink.dispatch_prepared_turn_collect_record_streaming(
                    turn,
                    observer.as_ref(),
                    &on_first_token,
                    responses,
                )
                .await
            }
            None => {
                sink.dispatch_prepared_turn_collect_record(turn, observer.as_ref(), &on_first_token)
                    .await
            }
        }
    };
    tokio::pin!(dispatch);
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            Err(anyhow!("HTTP execution command cancelled by its coordinator"))
        }
        result = &mut dispatch => result,
    };
    drop(first_token.borrow_mut().take());
    let _ = completed.send(WorkerReply {
        result,
        events: observer.take(),
    });
}

fn join_worker_threads(threads: Vec<JoinHandle<Result<()>>>) -> Result<()> {
    let mut errors = Vec::new();
    for thread in threads {
        match thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => errors.push(format!("{error:#}")),
            Err(_) => errors.push("HTTP execution worker panicked".to_string()),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "{} HTTP execution worker(s) failed: {}",
            errors.len(),
            errors.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use aiperf::http::{HttpRequest, PreparedHttpEndpoint};
    use aiperf::multiturn::PreparedEndpointReference;
    use aiperf_endpoints::{EndpointId, EndpointKey, EndpointRegistry, RawEndpointConfig};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[test]
    fn buffered_observer_events_replay_in_order() {
        #[derive(Default)]
        struct Observer {
            events: RefCell<Vec<&'static str>>,
        }
        impl RequestObserver for Observer {
            fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}
            fn on_admit(&self, _: Uuid, _: f64, _: usize) {
                self.events.borrow_mut().push("admit");
            }
            fn on_token(&self, _: Uuid, _: f64) {
                self.events.borrow_mut().push("token");
            }
            fn on_usage(&self, _: Uuid, _: ObservedUsage) {
                self.events.borrow_mut().push("usage");
            }
            fn on_terminal(&self, _: Uuid, _: ReplayTerminalStatus) {
                self.events.borrow_mut().push("terminal");
            }
        }

        let uuid = Uuid::nil();
        let buffered = BufferedObserver::default();
        buffered.on_admit(uuid, 1.0, 0);
        buffered.on_token(uuid, 2.0);
        buffered.on_usage(uuid, ObservedUsage::default());
        buffered.on_terminal(uuid, ReplayTerminalStatus::Completed);
        let observer = Observer::default();
        for event in buffered.take() {
            event.replay(&observer, 0.0);
        }
        assert_eq!(
            observer.events.borrow().as_slice(),
            &["admit", "token", "usage", "terminal"]
        );
    }

    #[derive(Clone)]
    struct StreamingEndpointTableFactory {
        registry: EndpointRegistry,
        url: String,
    }

    impl HttpPreparedEndpointTableFactory for StreamingEndpointTableFactory {
        fn prepare_worker(&self) -> Result<PreparedEndpointTable> {
            let endpoint = self.registry.prepare(
                &EndpointId::new("chat")?,
                RawEndpointConfig {
                    urls: vec![self.url.clone()],
                    streaming: true,
                    use_server_token_count: true,
                    ..RawEndpointConfig::default()
                },
            )?;
            let mut table = PreparedEndpointTable::new();
            assert_eq!(table.push(endpoint)?, EndpointKey::from_index(0));
            Ok(table)
        }
    }

    struct ResponseCollector {
        sender: RefCell<PollSender<ParsedResponse>>,
    }

    impl TurnResponseObserver for ResponseCollector {
        fn poll_ready(&self, context: &mut TaskContext<'_>) -> Poll<Result<()>> {
            self.sender
                .borrow_mut()
                .poll_reserve(context)
                .map(|result| result.map_err(|_| anyhow!("fixture response consumer closed")))
        }

        fn start_send(&self, response: ParsedResponse) -> Result<()> {
            self.sender
                .borrow_mut()
                .send_item(response)
                .map_err(|_| anyhow!("fixture response consumer closed"))
        }
    }

    fn streaming_backend(address: std::net::SocketAddr) -> Rc<dyn HttpTurnExecutionBackend> {
        let anchor = RealClockAnchor::now();
        let clock: Rc<dyn Clock> = RealClock::from_anchor(anchor);
        let url = format!("http://{address}");
        let table_factory = Arc::new(StreamingEndpointTableFactory {
            registry: EndpointRegistry::builtin().unwrap(),
            url: url.clone(),
        });
        let backend = NativeHttpExecutionBackendFactory
            .build(HttpExecutionBackendConfig {
                workers: 2,
                coordinator_clock: clock.clone(),
                real_clock_anchor: anchor,
                base_urls: vec![url],
                model: "fixture-model".to_string(),
                transport: TransportSinkConfig::default(),
                prepared_endpoints: Some(table_factory),
            })
            .unwrap();
        backend.set_run_origin(clock.now_ns()).unwrap();
        backend
    }

    fn streaming_turn() -> PreparedHttpTurn {
        PreparedHttpTurn {
            request: HttpRequest {
                uuid: Uuid::new_v4(),
                input_length: 1,
                max_output_tokens: 4,
                prompt_text: None,
                request_body: Some(serde_json::json!({
                    "model": "fixture-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                    "stream": true,
                    "stream_options": {"include_usage": true}
                })),
                request_body_bytes: None,
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: Some("evaluation-unit".to_string()),
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
            },
            model: "fixture-model".to_string(),
            endpoint: PreparedHttpEndpoint::Prepared(PreparedEndpointReference {
                key: EndpointKey::from_index(0),
                endpoint_id: EndpointId::new("chat").unwrap(),
            }),
            endpoint_aware: true,
        }
    }

    struct FirstResponseObserver {
        observed: Arc<Notify>,
    }

    impl TurnResponseObserver for FirstResponseObserver {
        fn poll_ready(&self, _context: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(&self, _response: ParsedResponse) -> Result<()> {
            self.observed.notify_one();
            Ok(())
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn thread_per_core_placement_forwards_live_normalized_sse_frames() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_sent_terminal = Arc::new(AtomicBool::new(false));
        let server_terminal = server_sent_terminal.clone();
        let first_observed = Arc::new(Notify::new());
        let server_first_observed = first_observed.clone();
        let release_burst = Arc::new(Notify::new());
        let server_release_burst = release_burst.clone();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 16 * 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let first = "data: {\"id\":\"response\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n";
            let mut terminal = String::new();
            for index in 0..300 {
                let finish_reason = if index == 299 { "\"stop\"" } else { "null" };
                terminal.push_str(&format!(
                    "data: {{\"id\":\"response\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"x\"}},\"finish_reason\":{finish_reason}}}]}}\n\n"
                ));
            }
            terminal.push_str(
                "data: {\"id\":\"response\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":300}}\n\ndata: [DONE]\n\n",
            );
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                first.len() + terminal.len(),
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(first.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            server_first_observed.notified().await;
            server_terminal.store(true, Ordering::SeqCst);
            socket.write_all(terminal.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            server_release_burst.notify_one();
        });

        let backend = streaming_backend(address);
        assert!(backend.supports_response_streaming());
        let turn = streaming_turn();
        let (response_tx, mut response_rx) = mpsc::channel(1);
        let responses = ResponseCollector {
            sender: RefCell::new(PollSender::new(response_tx)),
        };
        let collected = Arc::new(Mutex::new(Vec::new()));
        let collected_for_task = collected.clone();
        let saw_frame_before_terminal = Arc::new(AtomicBool::new(false));
        let saw_frame_for_task = saw_frame_before_terminal.clone();
        let consumer = tokio::spawn(async move {
            let mut count = 0_usize;
            while let Some(response) = response_rx.recv().await {
                count += 1;
                if count == 1 {
                    saw_frame_for_task.store(
                        !server_sent_terminal.load(Ordering::SeqCst),
                        Ordering::SeqCst,
                    );
                    first_observed.notify_one();
                } else if count == 2 {
                    release_burst.notified().await;
                }
                collected_for_task.lock().unwrap().push(response);
            }
        });
        let observer = BufferedObserver::default();
        let first_tokens = Cell::new(0_usize);
        let result = backend
            .execute_turn_streaming(
                turn,
                &observer,
                &|_| first_tokens.set(first_tokens.get() + 1),
                &responses,
            )
            .await
            .unwrap();
        drop(responses);
        consumer.await.unwrap();
        assert_eq!(
            result.outcome.response_text,
            format!("hel{}", "x".repeat(300))
        );
        assert_eq!(first_tokens.get(), 1);
        assert_eq!(collected.lock().unwrap().len(), 301);
        assert!(
            saw_frame_before_terminal.load(Ordering::SeqCst),
            "cross-thread placement buffered SSE until terminal"
        );
        backend.shutdown().unwrap();
        server.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropping_placement_dispatch_cancels_the_worker_transport() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let force_close = Arc::new(Notify::new());
        let server_force_close = force_close.clone();
        let (closed_tx, mut closed_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 16 * 1024];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"id\":\"response\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n",
                )
                .await
                .unwrap();
            socket.flush().await.unwrap();
            let mut probe = [0_u8; 1024];
            let closed = loop {
                tokio::select! {
                    read = socket.read(&mut probe) => {
                        match read {
                            Ok(0) | Err(_) => break true,
                            Ok(_) => continue,
                        }
                    }
                    () = server_force_close.notified() => break false,
                }
            };
            let _ = closed_tx.send(closed);
        });

        let backend = streaming_backend(address);
        let first_response = Arc::new(Notify::new());
        let responses = FirstResponseObserver {
            observed: first_response.clone(),
        };
        let observer = BufferedObserver::default();
        {
            let dispatch =
                backend.execute_turn_streaming(streaming_turn(), &observer, &|_| {}, &responses);
            tokio::pin!(dispatch);
            tokio::select! {
                biased;
                result = &mut dispatch => panic!("infinite SSE dispatch terminated before cancellation: {result:?}"),
                () = first_response.notified() => {}
            }
        }

        let mut worker_closed_socket = None;
        for _ in 0..10_000 {
            match closed_rx.try_recv() {
                Ok(closed) => {
                    worker_closed_socket = Some(closed);
                    break;
                }
                Err(oneshot::error::TryRecvError::Empty) => tokio::task::yield_now().await,
                Err(oneshot::error::TryRecvError::Closed) => break,
            }
        }
        if worker_closed_socket.is_none() {
            force_close.notify_one();
            worker_closed_socket = closed_rx.await.ok();
        }
        backend.shutdown().unwrap();
        server.await.unwrap();
        assert_eq!(
            worker_closed_socket,
            Some(true),
            "dropping coordinator dispatch did not cancel the worker HTTP request"
        );
    }
}
