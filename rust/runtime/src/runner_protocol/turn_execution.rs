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

use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::endpoints::{ParsedResponse, PreparedEndpointTable};
use crate::http::{
    DispatchResult, MeasuredContext, MeasuredOutcome, PreparedTurn, RequestExecutor, TransportSink,
    TransportSinkConfig,
};
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest};
use crate::multiturn::TurnToSend;
use crate::scheduled::TurnResponseObserver;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
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
pub trait RequestExecutorFactory: Send + Sync {
    /// Construct the backend used below the run's single logical dispatcher.
    fn build(&self, config: HttpExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>>;
}

/// Native local execution factory.
///
/// One worker keeps the transport on the coordinator reactor. Two or more
/// workers create one current-thread Tokio runtime and one transport stack per
/// OS thread; no worker owns scheduling or benchmark policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeRequestExecutorFactory;

impl RequestExecutorFactory for NativeRequestExecutorFactory {
    fn build(&self, config: HttpExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
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
        Ok(Rc::new(ThreadPerCoreRequestExecutor::new(config)?))
    }
}

struct WorkerReply {
    result: Result<DispatchResult>,
    /// Non-consuming cloned record for a live sink, when the measured command
    /// requested one; the authoritative record stays in the worker observer.
    live_record: Option<RecordIngest>,
}

struct WorkerCommand {
    turn: PreparedTurn,
    context: MeasuredContext,
    first_token: oneshot::Sender<i64>,
    responses: Option<mpsc::Sender<ParsedResponse>>,
    completed: oneshot::Sender<WorkerReply>,
    cancellation: PlacementCancellation,
}

/// Control-plane message multiplexed onto each worker's command channel.
enum WorkerMessage {
    /// Build the worker-local observer from the single resolved metrics
    /// configuration and run origin before any measured command.
    Configure {
        config: MetricsConfig,
        origin_ns: i64,
    },
    /// Execute one prepared turn (buffered or measured).
    Command(Box<WorkerCommand>),
    /// Warm this worker's sink with one discarded round-trip before timed
    /// issuance, then acknowledge so the coordinator can release all workers
    /// from a warmed state (the Rust-native "workers ready, go" barrier).
    Prewarm {
        turn: PreparedTurn,
        done: oneshot::Sender<()>,
    },
    /// Finalize the worker observer at `end_ns` and return its records, then
    /// exit. Sent once, after all commands for this worker have been enqueued.
    Drain {
        end_ns: i64,
        reply: std::sync::mpsc::SyncSender<Vec<(Uuid, RecordIngest)>>,
    },
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
struct ThreadPerCoreRequestExecutor {
    senders: RefCell<Option<Vec<mpsc::Sender<WorkerMessage>>>>,
    threads: RefCell<Vec<JoinHandle<Result<()>>>>,
    next_worker: Cell<usize>,
    run_origin_ns: Cell<Option<i64>>,
    dimension_sink: TransportSink,
}

impl ThreadPerCoreRequestExecutor {
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
            let (sender, receiver) = mpsc::channel::<WorkerMessage>(WORKER_QUEUE_CAPACITY);
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
impl RequestExecutor for ThreadPerCoreRequestExecutor {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(
            self.run_origin_ns.replace(Some(start_ns)).is_none(),
            "HTTP execution run origin was configured more than once"
        );
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <TransportSink as crate::scheduled::TurnDispatcher>::inference_dimensions(
            &self.dimension_sink,
            turn,
        )
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        let senders = self.senders.borrow();
        let senders = senders
            .as_ref()
            .ok_or_else(|| anyhow!("HTTP execution backend is shut down"))?;
        for sender in senders.iter() {
            sender
                .try_send(WorkerMessage::Configure {
                    config: config.clone(),
                    origin_ns,
                })
                .map_err(|_| anyhow!("HTTP execution worker rejected measurement configuration"))?;
        }
        Ok(())
    }

    async fn prewarm(&self, turn: PreparedTurn) -> Result<()> {
        // Broadcast one discarded warmup round-trip to every worker and wait for
        // all to finish, so timed issuance starts from a uniformly warmed state
        // (connections established, body/tokenizer/JIT paths hot). Non-fatal.
        let dones = {
            let senders = self.senders.borrow();
            let Some(senders) = senders.as_ref() else {
                return Ok(());
            };
            let mut dones = Vec::with_capacity(senders.len());
            for sender in senders.iter() {
                let (done, wait) = oneshot::channel();
                if sender
                    .try_send(WorkerMessage::Prewarm {
                        turn: turn.clone(),
                        done,
                    })
                    .is_ok()
                {
                    dones.push(wait);
                }
            }
            dones
        };
        for wait in dones {
            let _ = wait.await;
        }
        Ok(())
    }

    async fn execute_measured(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        let reply = self
            .execute_command(turn, context, on_first_token, None)
            .await?;
        Ok(MeasuredOutcome {
            result: reply.result?,
            live_record: reply.live_record,
        })
    }

    async fn execute_measured_streaming(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<MeasuredOutcome> {
        let reply = self
            .execute_command(turn, context, on_first_token, Some(responses))
            .await?;
        Ok(MeasuredOutcome {
            result: reply.result?,
            live_record: reply.live_record,
        })
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        let senders = {
            let senders = self.senders.borrow();
            senders
                .as_ref()
                .ok_or_else(|| anyhow!("HTTP execution backend is shut down"))?
                .clone()
        };
        // Each worker finalizes its observer once its in-flight jobs complete and
        // replies with its dense-local records; the coordinator concatenates them.
        let mut receivers = Vec::with_capacity(senders.len());
        for sender in &senders {
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            sender
                .try_send(WorkerMessage::Drain {
                    end_ns,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow!("HTTP execution worker rejected a drain request"))?;
            receivers.push(reply_rx);
        }
        let mut records = Vec::new();
        for receiver in receivers {
            let worker_records = receiver
                .recv()
                .map_err(|_| anyhow!("HTTP execution worker dropped before draining records"))?;
            records.extend(worker_records);
        }
        Ok(records)
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdown_workers()
    }
}

impl ThreadPerCoreRequestExecutor {
    fn origin(&self) -> Result<i64> {
        self.run_origin_ns
            .get()
            .ok_or_else(|| anyhow!("HTTP execution run origin is not configured"))
    }

    async fn execute_command(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<WorkerReply> {
        let _run_origin_ns = self.origin()?;
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
            .send(WorkerMessage::Command(Box::new(WorkerCommand {
                turn,
                context,
                first_token: first_token_tx,
                responses: responses.map(|_| response_tx),
                completed: completed_tx,
                cancellation,
            })))
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
        cancellation_guard.disarm();
        Ok(reply)
    }
}

impl Drop for ThreadPerCoreRequestExecutor {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown_workers() {
            tracing::error!(error = %error, "failed to shut down HTTP execution workers");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker_thread(
    receiver: mpsc::Receiver<WorkerMessage>,
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
        clock.clone(),
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
    local.block_on(&runtime, run_worker(receiver, sink, clock));
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

async fn run_worker(
    mut receiver: mpsc::Receiver<WorkerMessage>,
    sink: Rc<TransportSink>,
    clock: Rc<dyn Clock>,
) {
    let mut jobs = JoinSet::new();
    let mut accepting = true;
    // Built lazily by `Configure`; shared (`Rc`) into every measured task so all
    // of this worker's requests accumulate into one observer that is drained
    // once at end of run.
    let mut observer: Option<Rc<NativeMetricsObserver>> = None;
    // Set by `Drain`; the loop finalizes and replies once its JoinSet empties.
    let mut pending_drain: Option<(i64, std::sync::mpsc::SyncSender<Vec<(Uuid, RecordIngest)>>)> =
        None;
    while accepting || !jobs.is_empty() {
        tokio::select! {
            message = receiver.recv(), if accepting => {
                match message {
                    Some(WorkerMessage::Configure { config, origin_ns }) => {
                        // The worker sink was constructed with a placeholder
                        // run origin of 0 (the true origin is not known until
                        // after backend startup). Its `ms()` conversion for
                        // token-arrival timestamps must share the observer's
                        // `origin_ns`, or TTFT/ITL are offset by the setup
                        // duration. The workers==1 path already anchors both to
                        // the same origin via `set_run_origin`; do the same per
                        // worker here.
                        sink.set_run_origin(origin_ns);
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
                    Some(WorkerMessage::Prewarm { turn, done }) => {
                        let sink = sink.clone();
                        jobs.spawn_local(async move {
                            let _ = sink.prewarm(turn).await;
                            let _ = done.send(());
                        });
                    }
                    Some(WorkerMessage::Drain { end_ns, reply }) => {
                        // No more commands follow a drain; stop accepting and let
                        // the loop finalize once every in-flight job completes.
                        accepting = false;
                        pending_drain = Some((end_ns, reply));
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
    if let Some((end_ns, reply)) = pending_drain {
        let records = observer
            .map(|observer| {
                observer
                    .take_finalizer_at(end_ns)
                    .finish_with_records()
                    .records
            })
            .unwrap_or_default();
        let _ = reply.send(records);
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

async fn execute_worker_command(
    sink: Rc<TransportSink>,
    worker_observer: Option<Rc<NativeMetricsObserver>>,
    command: WorkerCommand,
) {
    let WorkerCommand {
        turn,
        context,
        first_token,
        responses,
        completed,
        cancellation,
    } = command;
    let uuid = turn.request.uuid;
    let first_token = RefCell::new(Some(first_token));
    let on_first_token = |ttft_ns| {
        if let Some(sender) = first_token.borrow_mut().take() {
            let _ = sender.send(ttft_ns);
        }
    };
    let response_observer = responses.map(WorkerResponseObserver::new);
    let reply = match &worker_observer {
        Some(observer) => {
            let dispatch = sink.dispatch_measured(
                observer,
                turn,
                &context,
                &on_first_token,
                response_observer
                    .as_ref()
                    .map(|responses| responses as &dyn TurnResponseObserver),
            );
            tokio::pin!(dispatch);
            let result = tokio::select! {
                biased;
                () = cancellation.cancelled() => {
                    Err(anyhow!("HTTP execution command cancelled by its coordinator"))
                }
                result = &mut dispatch => result,
            };
            let live_record = context
                .wants_live_record
                .then(|| {
                    // Metrics-only (sketch) mode moves the record out of the
                    // observer so its token storage is freed as the run streams;
                    // every other mode clones it for the end-of-run drain.
                    if context.consume_record {
                        observer.drain_terminal_record(uuid, 0)
                    } else {
                        observer.snapshot_record(uuid, 0)
                    }
                })
                .flatten();
            WorkerReply {
                result,
                live_record,
            }
        }
        None => {
            let _ = completed.send(WorkerReply {
                result: Err(anyhow!(
                    "worker-local measurement was not configured before a measured command"
                )),
                live_record: None,
            });
            return;
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

    use crate::endpoints::{EndpointId, EndpointKey, EndpointRegistry, RawEndpointConfig};
    use crate::http::{HttpRequest, PreparedHttpEndpoint};
    use crate::metrics::RequestMetricMetadata;
    use crate::multiturn::PreparedEndpointReference;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// Coordinator-known arrival facts for a fixture turn. `MeasuredContext`
    /// has no `Default`, so the tests build the same all-neutral context the
    /// coordinator would forward for a one-turn fixture dispatch.
    fn measured_context() -> MeasuredContext {
        MeasuredContext {
            arrival_ms: 0.0,
            input_length: 1,
            requested_output_length: 4,
            metadata: RequestMetricMetadata::default(),
            wants_live_record: false,
            consume_record: false,
        }
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

    fn streaming_backend(address: std::net::SocketAddr) -> Rc<dyn RequestExecutor> {
        let anchor = RealClockAnchor::now();
        let clock: Rc<dyn Clock> = RealClock::from_anchor(anchor);
        let url = format!("http://{address}");
        let table_factory = Arc::new(StreamingEndpointTableFactory {
            registry: EndpointRegistry::builtin().unwrap(),
            url: url.clone(),
        });
        let backend = NativeRequestExecutorFactory
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
        let origin_ns = clock.now_ns();
        backend.set_run_origin(origin_ns).unwrap();
        backend
            .configure_measurement(MetricsConfig::default(), origin_ns)
            .unwrap();
        backend
    }

    fn streaming_turn() -> PreparedTurn {
        PreparedTurn {
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
            data_policy: crate::multiturn::TurnDataPolicy::ordinary(),
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
        let first_tokens = Cell::new(0_usize);
        let outcome = backend
            .execute_measured_streaming(
                turn,
                measured_context(),
                &|_| first_tokens.set(first_tokens.get() + 1),
                &responses,
            )
            .await
            .unwrap();
        drop(responses);
        consumer.await.unwrap();
        assert_eq!(
            outcome.result.outcome.response_text,
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
        {
            let dispatch = backend.execute_measured_streaming(
                streaming_turn(),
                measured_context(),
                &|_| {},
                &responses,
            );
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
