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
use std::rc::Rc;
use std::sync::Arc;
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
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
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
        sender
            .send(WorkerCommand {
                turn,
                first_token: first_token_tx,
                responses: responses.map(|_| response_tx),
                completed: completed_tx,
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
                        Some(response) => responses
                            .expect("response channel exists only for streaming dispatch")
                            .on_response(response),
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
                responses.on_response(response);
            }
        }
        for event in reply.events {
            event.replay(observer, run_origin_ns as f64 / 1_000_000.0);
        }
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
    sender: mpsc::Sender<ParsedResponse>,
    failure: RefCell<Option<String>>,
}

impl WorkerResponseObserver {
    fn new(sender: mpsc::Sender<ParsedResponse>) -> Self {
        Self {
            sender,
            failure: RefCell::new(None),
        }
    }

    fn take_failure(&self) -> Option<String> {
        self.failure.borrow_mut().take()
    }
}

impl TurnResponseObserver for WorkerResponseObserver {
    fn on_response(&self, response: ParsedResponse) {
        if let Err(error) = self.sender.try_send(response) {
            self.failure
                .borrow_mut()
                .get_or_insert_with(|| match error {
                    mpsc::error::TrySendError::Full(_) => {
                        "HTTP execution response stream exceeded its bounded placement channel"
                            .to_string()
                    }
                    mpsc::error::TrySendError::Closed(_) => {
                        "HTTP execution response stream receiver closed before terminal".to_string()
                    }
                });
        }
    }
}

async fn execute_worker_command(sink: Rc<TransportSink>, command: WorkerCommand) {
    let observer = Rc::new(BufferedObserver::default());
    let first_token = RefCell::new(Some(command.first_token));
    let on_first_token = |ttft_ns| {
        if let Some(sender) = first_token.borrow_mut().take() {
            let _ = sender.send(ttft_ns);
        }
    };
    let response_observer = command.responses.map(WorkerResponseObserver::new);
    let mut result = match response_observer.as_ref() {
        Some(responses) => {
            sink.dispatch_prepared_turn_collect_record_streaming(
                command.turn,
                observer.as_ref(),
                &on_first_token,
                responses,
            )
            .await
        }
        None => {
            sink.dispatch_prepared_turn_collect_record(
                command.turn,
                observer.as_ref(),
                &on_first_token,
            )
            .await
        }
    };
    if let Some(failure) = response_observer.and_then(|responses| responses.take_failure())
        && result.is_ok()
    {
        result = Err(anyhow!(failure));
    }
    drop(first_token.borrow_mut().take());
    let _ = command.completed.send(WorkerReply {
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
        responses: RefCell<Vec<ParsedResponse>>,
        server_sent_terminal: Arc<AtomicBool>,
        saw_frame_before_terminal: Cell<bool>,
    }

    impl TurnResponseObserver for ResponseCollector {
        fn on_response(&self, response: ParsedResponse) {
            if !self.server_sent_terminal.load(Ordering::SeqCst) {
                self.saw_frame_before_terminal.set(true);
            }
            self.responses.borrow_mut().push(response);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn thread_per_core_placement_forwards_live_normalized_sse_frames() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_sent_terminal = Arc::new(AtomicBool::new(false));
        let server_terminal = server_sent_terminal.clone();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 16 * 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let first = "data: {\"id\":\"response\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n";
            let terminal = concat!(
                "data: {\"id\":\"response\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"id\":\"response\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n"
            );
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                first.len() + terminal.len(),
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(first.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            server_terminal.store(true, Ordering::SeqCst);
            socket.write_all(terminal.as_bytes()).await.unwrap();
        });

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
        assert!(backend.supports_response_streaming());
        let turn = PreparedHttpTurn {
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
        };
        let responses = ResponseCollector {
            responses: RefCell::new(Vec::new()),
            server_sent_terminal,
            saw_frame_before_terminal: Cell::new(false),
        };
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
        assert_eq!(result.outcome.response_text, "hello");
        assert_eq!(first_tokens.get(), 1);
        assert!(!responses.responses.borrow().is_empty());
        assert!(
            responses.saw_frame_before_terminal.get(),
            "cross-thread placement buffered SSE until terminal"
        );
        backend.shutdown().unwrap();
        server.await.unwrap();
    }
}
