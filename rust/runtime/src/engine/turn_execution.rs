// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral placement for turn execution.
//!
//! Each transport supplies a worker-local [`WorkerSink`] through an
//! [`ExecutionSinkBuilder`]. Scheduling, measurement, streaming, cancellation,
//! and drain remain transport-independent.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::poll_fn;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context as TaskContext, Poll};
use std::thread::JoinHandle;

use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::endpoints::{ParsedResponse, PreparedEndpointTable};
use crate::engine::protocol::HopRouting;
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest};
use crate::multiturn::TurnToSend;
use crate::scheduled::TurnResponseObserver;
use crate::transport::core::{
    DispatchResult, MeasuredContext, MeasuredOutcome, PreparedTurn, RequestExecutor,
};
use crate::transport::http::{TransportSink, TransportSinkConfig};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::PollSender;
use uuid::Uuid;

/// Bounded per-worker command queue depth for the thread-per-core hop executor.
const WORKER_QUEUE_CAPACITY: usize = 256;
/// Bounded per-command streaming-response relay depth.
const WORKER_RESPONSE_CAPACITY: usize = 256;

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
    /// Whether the run retains raw HTTP-exchange artifacts. The gRPC sink uses
    /// this to skip building the per-request HTTP-compatibility record when no
    /// raw artifact will consume it.
    pub raw_enabled: bool,
    /// Optional worker-local endpoint preparation.
    ///
    /// Each worker receives an independent dense-key table.
    pub prepared_endpoints: Option<Arc<dyn PreparedEndpointTableFactory>>,
    /// Worker-assignment policy applied by the `workers > 1` hop executor. Inert
    /// for `workers == 1` (co-located sink, no hop).
    pub hop_routing: HopRouting,
    /// Logical dry-run placement width captured before physical worker caps.
    pub virtual_worker_width: Option<usize>,
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

/// Build one execution backend for `workers` execution workers.
///
/// `workers == 1` keeps the transport sink co-located on the caller's reactor:
/// this is the placement every [`DispatchMode::Sharded`] and
/// [`DispatchMode::Global`] sub-cell thread uses (thread-per-core parallelism
/// is provided by the sharded scheduled runtime *above* the transport, one
/// co-located backend per shard).
///
/// `workers > 1` builds the [`ThreadPerCoreExecutor`] cross-thread hop: one
/// coordinator-owned scheduling loop above this backend issues every turn in
/// exact global order, and each [`RequestExecutor::execute_measured`] call is
/// round-robined to one of `workers` worker OS threads over a bounded mpsc
/// command queue, driven to terminal by that thread's worker-local
/// [`WorkerSink`], and returned over a oneshot reply. This is the
/// [`DispatchMode::GlobalHop`] placement (see
/// [`crate::engine::global_hop::run_global_hop`]); it reproduces exact
/// request-to-thread assignment order, which the `Global` shared-admission
/// model cannot guarantee.
///
/// [`DispatchMode::Sharded`]: crate::engine::protocol::DispatchMode::Sharded
/// [`DispatchMode::Global`]: crate::engine::protocol::DispatchMode::Global
/// [`DispatchMode::GlobalHop`]: crate::engine::protocol::DispatchMode::GlobalHop
pub(crate) fn build_native<B: ExecutionSinkBuilder>(
    builder: B,
    workers: usize,
    coordinator_clock: Rc<dyn Clock>,
    real_clock_anchor: RealClockAnchor,
    hop_routing: HopRouting,
) -> Result<Rc<dyn RequestExecutor>> {
    ensure!(workers > 0, "execution workers must be positive");
    if workers == 1 {
        return Ok(Rc::new(builder.build_sink(coordinator_clock, 0)?));
    }
    Ok(Rc::new(ThreadPerCoreExecutor::new(
        builder,
        workers,
        coordinator_clock,
        real_clock_anchor,
        hop_routing,
    )?))
}

/// Native HTTP execution factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct HttpExecutionFactory;

impl RequestExecutorFactory for HttpExecutionFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        let workers = config.workers;
        let coordinator_clock = config.coordinator_clock.clone();
        let anchor = config.real_clock_anchor;
        let hop_routing = config.hop_routing;
        build_native(
            HttpSinkBuilder::from_config(&config),
            workers,
            coordinator_clock,
            anchor,
            hop_routing,
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

/// One worker's reply for a single dispatched command.
struct WorkerReply {
    result: Result<DispatchResult>,
    /// Non-consuming cloned record for a live sink, when the measured command
    /// requested one; the authoritative record stays in the worker observer.
    live_record: Option<RecordIngest>,
}

/// One measured turn hopped from the coordinator loop to a worker thread.
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

/// Cross-thread cancellation handle for a single in-flight worker command.
///
/// Dropping the coordinator's dispatch future (e.g. the scheduler cancelled the
/// turn) fires this so the worker aborts its transport round-trip instead of
/// running to completion on a request nobody is waiting for.
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

/// Fires [`PlacementCancellation::cancel`] on drop unless disarmed after the
/// command completes normally.
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

/// Generic thread-per-core placement behind the single dispatcher.
///
/// One worker loop for every transport: the sink type is the only variable,
/// supplied by `B::Sink`. Worker threads, measurement, drain, cancellation, and
/// streaming are written once here. Round-robins each dispatched turn to the
/// next worker thread in issuance order, so the coordinator's single scheduling
/// loop maps turn `i` deterministically to worker `i % workers`.
struct ThreadPerCoreExecutor<B: ExecutionSinkBuilder> {
    senders: RefCell<Option<Vec<mpsc::Sender<WorkerMessage>>>>,
    threads: RefCell<Vec<JoinHandle<Result<()>>>>,
    /// Worker-assignment policy applied at the single pick site.
    routing: HopRouting,
    /// Round-robin cursor; also the fallback for correlation-less sticky turns.
    next_worker: Cell<usize>,
    /// Monotonic send counter stamped into [`WorkerLoad::last_sent`], so the LRU-like tiebreak
    /// orders by issuance rather than by a clock reading.
    send_seq: Cell<u64>,
    /// Per-worker load, read by [`HopRouting::LeastLoaded`]. Single-threaded
    /// coordinator, so plain `Cell`s.
    inflight: Vec<WorkerLoad>,
    /// `correlation_id` → bound worker, so [`HopRouting::LeastLoaded`]
    /// continuations stay on the worker their session was first placed on.
    sticky: RefCell<HashMap<String, usize>>,
    run_origin_ns: Cell<Option<i64>>,
    dimension_sink: B::Sink,
}

/// Decrements a worker's in-flight counter on drop, so a cancelled or
/// early-returning command still releases its [`HopRouting::LeastLoaded`] slot.
struct InflightGuard<'a> {
    slot: &'a Cell<usize>,
}

impl<'a> InflightGuard<'a> {
    fn new(slot: &'a Cell<usize>) -> Self {
        slot.set(slot.get().saturating_add(1));
        Self { slot }
    }
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.slot.set(self.slot.get().saturating_sub(1));
    }
}

/// Pure worker-assignment decision for one dispatched turn.
///
/// Factored out of [`ThreadPerCoreExecutor::execute_command`] so the routing
/// policy is unit-testable without a live executor. `rr_cursor` is the
/// round-robin cursor (advanced only when a round-robin pick is made), `inflight`
/// is the per-worker in-flight snapshot, and `sticky` holds
/// `correlation_id`→worker bindings for [`HopRouting::LeastLoaded`].
///
/// `LeastLoaded` mirrors Python's `StickyCreditRouter`: honour an existing binding, else take the
/// shallowest worker — ties broken by fewer credits routed so far, then by longest since last
/// send — and bind the correlation id to the winner.
///
/// Bindings follow Python's lifecycle: created only on a non-final turn (a single-turn session
/// never binds), and released on the owning session's final turn, decrementing that worker's
/// `active_sessions`.
///
/// One Python behaviour is deliberately absent: pinning DAG children to their parent's worker via
/// `parent_correlation_id`. Graph and DAG traces are placed by
/// [`ThreadPerCoreTracePlacement`](crate::graph::placement) and never reach this hop, so there is
/// no child here to pin.
pub(crate) fn pick_worker(
    routing: HopRouting,
    workers: usize,
    correlation: Option<&str>,
    is_final_turn: bool,
    inflight: &[WorkerLoad],
    sticky: &mut HashMap<String, usize>,
    rr_cursor: &mut usize,
) -> usize {
    debug_assert!(workers > 0, "worker count must be positive");
    match routing {
        HopRouting::RoundRobin => round_robin(workers, rr_cursor),
        HopRouting::Sticky => match correlation {
            Some(id) => (fnv1a64(id.as_bytes()) % workers as u64) as usize,
            None => round_robin(workers, rr_cursor),
        },
        HopRouting::LeastLoaded => {
            // An existing binding wins, exactly as Python's sticky lookup does.
            let bound = correlation.and_then(|id| sticky.get(id).copied());
            let worker = match bound {
                Some(worker) => worker,
                None => {
                    let worker = least_loaded(inflight);
                    // Python creates a binding only for a NON-final turn: a single-turn session
                    // would otherwise be inserted and evicted on the same call, churning the map
                    // and the `active_sessions` counter for nothing.
                    if let Some(id) = correlation
                        && !is_final_turn
                    {
                        sticky.insert(id.to_owned(), worker);
                        let load = &inflight[worker];
                        load.active_sessions.set(load.active_sessions.get() + 1);
                    }
                    worker
                }
            };
            // The session's final turn releases its binding. Without this the map grows for the
            // lifetime of the run — one entry per session, never reclaimed — and every worker's
            // `active_sessions` ratchets upward, biasing placement toward whichever worker
            // happened to take fewest sessions early on.
            if is_final_turn
                && let Some(id) = correlation
                && let Some(released) = sticky.remove(id)
            {
                let load = &inflight[released];
                load.active_sessions
                    .set(load.active_sessions.get().saturating_sub(1));
            }
            worker
        }
    }
}

/// Per-worker load signals consulted when several workers tie on in-flight depth.
///
/// Mirrors the fields Python's `StickyCreditRouter` tie-breaks on, minus
/// `active_sessions` — that one requires an end-of-session signal to decrement, which does not
/// reach this pick site (see [`pick_worker`]). `sent` is Python's `virtual_sent_credits` and
/// `last_sent` its `last_sent_at_ns`, kept as a monotonic sequence rather than a wall-clock
/// reading so placement stays reproducible run to run and under a `SimClock`.
///
/// Python seeds both to non-zero on worker registration to stop a late-joining worker from
/// attracting a thundering herd. That does not apply here: every worker thread is registered
/// before the first pick, so all start equal and a zero seed is the faithful choice.
#[derive(Default)]
pub(crate) struct WorkerLoad {
    /// In-flight commands: `+1` on send, `-1` on reply.
    pub(crate) inflight: Cell<usize>,
    /// Multi-turn sessions currently bound to this worker. Python's `active_sessions`:
    /// incremented when a binding is created, decremented when the session's final turn evicts it.
    pub(crate) active_sessions: Cell<usize>,
    /// Total credits ever routed here. Python's `virtual_sent_credits`.
    pub(crate) sent: Cell<u64>,
    /// Sequence number of this worker's most recent send. Python's `last_sent_at_ns`, as a
    /// counter — oldest wins, giving the same LRU-like fairness without reading a clock.
    pub(crate) last_sent: Cell<u64>,
}

/// Advance and return the round-robin worker index.
fn round_robin(workers: usize, rr_cursor: &mut usize) -> usize {
    let worker = *rr_cursor % workers;
    *rr_cursor = rr_cursor.wrapping_add(1);
    worker
}

/// Index of the shallowest in-flight worker, ties broken the way Python's router breaks them.
///
/// A bare `min_by_key` on depth alone resolves every tie to the lowest index, which at the start
/// of a run — when every worker is at zero — sends the first burst of sessions all to worker 0.
/// Python's order after depth is `active_sessions`, then credits routed so far, then longest
/// since last send; the index is kept as a final deterministic tiebreak so the choice is still
/// total and reproducible.
fn least_loaded(load: &[WorkerLoad]) -> usize {
    load.iter()
        .enumerate()
        .min_by_key(|(index, worker)| {
            (
                worker.inflight.get(),
                worker.active_sessions.get(),
                worker.sent.get(),
                worker.last_sent.get(),
                *index,
            )
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Fixed, seed-free FNV-1a 64-bit hash. Unlike
/// [`std::collections::hash_map::DefaultHasher`] this is stable across processes
/// and runs, so the same `correlation_id` always maps to the same worker.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

impl<B: ExecutionSinkBuilder> ThreadPerCoreExecutor<B> {
    fn new(
        builder: B,
        workers: usize,
        coordinator_clock: Rc<dyn Clock>,
        real_clock_anchor: RealClockAnchor,
        routing: HopRouting,
    ) -> Result<Self> {
        ensure!(
            workers > 1,
            "thread-per-core execution requires at least two workers"
        );
        let label = builder.label();
        let dimension_sink = builder.build_sink(coordinator_clock, 0)?;
        let builder = Arc::new(builder);
        let mut senders = Vec::with_capacity(workers);
        let mut threads = Vec::with_capacity(workers);

        for worker_id in 0..workers {
            let (sender, receiver) = mpsc::channel::<WorkerMessage>(WORKER_QUEUE_CAPACITY);
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
            let builder = builder.clone();
            let thread = match std::thread::Builder::new()
                .name(format!("aiperf-{label}-{worker_id}"))
                .spawn(move || {
                    let result = run_worker_thread(
                        receiver,
                        real_clock_anchor,
                        builder,
                        worker_id,
                        started_tx,
                    );
                    if let Err(error) = &result {
                        tracing::error!(worker_id, error = %error, "execution worker failed");
                    }
                    result
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error).context("spawning execution worker");
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
                        .context(format!("starting execution worker {worker_id}"));
                }
                Err(error) => {
                    drop(sender);
                    let _ = thread.join();
                    drop(senders);
                    join_worker_threads(threads)?;
                    return Err(error)
                        .context(format!("receiving worker {worker_id} startup status"));
                }
            }
        }

        Ok(Self {
            senders: RefCell::new(Some(senders)),
            threads: RefCell::new(threads),
            routing,
            next_worker: Cell::new(0),
            send_seq: Cell::new(0),
            inflight: (0..workers).map(|_| WorkerLoad::default()).collect(),
            sticky: RefCell::new(HashMap::new()),
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
impl<B: ExecutionSinkBuilder> RequestExecutor for ThreadPerCoreExecutor<B> {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(
            self.run_origin_ns.replace(Some(start_ns)).is_none(),
            "execution run origin was configured more than once"
        );
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        WorkerSink::inference_dimensions(&self.dimension_sink, turn)
    }

    fn supports_response_streaming(&self) -> bool {
        WorkerSink::supports_response_streaming(&self.dimension_sink)
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        let senders = self.senders.borrow();
        let senders = senders
            .as_ref()
            .ok_or_else(|| anyhow!("execution backend is shut down"))?;
        for sender in senders.iter() {
            sender
                .try_send(WorkerMessage::Configure {
                    config: config.clone(),
                    origin_ns,
                })
                .map_err(|_| anyhow!("execution worker rejected measurement configuration"))?;
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
                .ok_or_else(|| anyhow!("execution backend is shut down"))?
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
                .map_err(|_| anyhow!("execution worker rejected a drain request"))?;
            receivers.push(reply_rx);
        }
        let mut records = Vec::new();
        for receiver in receivers {
            let worker_records = receiver
                .recv()
                .map_err(|_| anyhow!("execution worker dropped before draining records"))?;
            records.extend(worker_records);
        }
        Ok(records)
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdown_workers()
    }
}

impl<B: ExecutionSinkBuilder> ThreadPerCoreExecutor<B> {
    fn origin(&self) -> Result<i64> {
        self.run_origin_ns
            .get()
            .ok_or_else(|| anyhow!("execution run origin is not configured"))
    }

    async fn execute_command(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<WorkerReply> {
        let _run_origin_ns = self.origin()?;
        let (sender, _inflight_guard) = {
            let senders = self.senders.borrow();
            let senders = senders
                .as_ref()
                .ok_or_else(|| anyhow!("execution backend is shut down"))?;
            let mut rr_cursor = self.next_worker.get();
            // Pass the live `Cell` slice, not an eager snapshot: only `LeastLoaded`
            // reads worker depths, so RoundRobin/Sticky avoid a per-request W-sized
            // heap allocation on the coordinator hot path.
            let index = pick_worker(
                self.routing,
                senders.len(),
                context.metadata.correlation_id.as_deref(),
                turn.request.is_final_turn,
                &self.inflight,
                &mut self.sticky.borrow_mut(),
                &mut rr_cursor,
            );
            self.next_worker.set(rr_cursor);
            // Hold the slot from send through reply; the guard decrements on any
            // return path (completion, cancellation, error) so LeastLoaded depth
            // stays accurate.
            // Record the send before the guard: `sent` and `last_sent` are what break a tie the
            // next time several workers sit at the same depth.
            let chosen = &self.inflight[index];
            chosen.sent.set(chosen.sent.get() + 1);
            self.send_seq.set(self.send_seq.get() + 1);
            chosen.last_sent.set(self.send_seq.get());
            let guard = InflightGuard::new(&chosen.inflight);
            (senders[index].clone(), guard)
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
            .map_err(|_| anyhow!("execution worker stopped before accepting a command"))?;

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
                        anyhow!("execution worker dropped a command before completion")
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

impl<B: ExecutionSinkBuilder> Drop for ThreadPerCoreExecutor<B> {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown_workers() {
            tracing::error!(error = %error, "failed to shut down execution workers");
        }
    }
}

fn run_worker_thread<B: ExecutionSinkBuilder>(
    receiver: mpsc::Receiver<WorkerMessage>,
    anchor: RealClockAnchor,
    builder: Arc<B>,
    worker_id: usize,
    started: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) -> Result<()> {
    // IO + time only, deliberately not enable_all(). enable_all() also starts
    // the signal driver, and tokio hangs child-process orphan reaping off it:
    // every park runs OrphanQueueImpl::reap_orphans. A worker runtime parks
    // once per idle turn of the request loop, so at benchmark rates that sweep
    // measured 4.35% of client CPU in a frame-pointer profile -- for a queue
    // that is always empty, because workers spawn no child processes and
    // handle no signals (the CLI process owns both).
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("creating worker Tokio runtime");
        }
    };
    let clock = RealClock::from_anchor(anchor);
    let sink = match builder.build_sink(clock.clone(), worker_id) {
        Ok(sink) => Rc::new(sink),
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error).context("constructing worker-local transport sink");
        }
    };
    if started.send(Ok(())).is_err() {
        return Ok(());
    }
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, run_worker(receiver, sink, clock, worker_id));
    Ok(())
}

async fn run_worker<S: WorkerSink + 'static>(
    mut receiver: mpsc::Receiver<WorkerMessage>,
    sink: Rc<S>,
    clock: Rc<dyn Clock>,
    worker_id: usize,
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
                    tracing::error!(error = %error, "execution task panicked");
                }
            }
        }
    }
    if let Some((end_ns, reply)) = pending_drain {
        let mut records = observer
            .map(|observer| {
                observer
                    .take_finalizer_at(end_ns)
                    .finish_with_records()
                    .records
            })
            .unwrap_or_default();
        // Stamp the executing worker identity into records the coordinator did
        // not already attribute. In the hop path the worker-local observer holds
        // exactly this thread's requests, so `worker_id` here is authoritative:
        // it makes per-worker routing (e.g. `HopRouting::Sticky` session
        // affinity) observable at the record boundary.
        for (_uuid, ingest) in &mut records {
            if ingest.worker_id.is_none() {
                ingest.worker_id = Some(worker_id.to_string());
            }
        }
        let _ = reply.send(records);
    }
}

/// Relays a worker's streamed parsed responses back to the coordinator dispatch
/// future over a bounded channel.
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
                    anyhow!("execution response stream receiver closed before terminal")
                })
            })
    }

    fn start_send(&self, response: ParsedResponse) -> Result<()> {
        self.sender
            .borrow_mut()
            .send_item(response)
            .map_err(|_| anyhow!("execution response stream receiver closed before terminal"))
    }
}

async fn execute_worker_command<S: WorkerSink + 'static>(
    sink: Rc<S>,
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
                    Err(anyhow!("execution command cancelled by its coordinator"))
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
            Err(_) => errors.push("execution worker panicked".to_string()),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "{} execution worker(s) failed: {}",
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
    use crate::metrics::RequestMetricMetadata;
    use crate::multiturn::PreparedEndpointReference;
    use crate::transport::core::{PreparedEndpointBinding, Request};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[test]
    fn a_session_releases_its_binding_on_the_final_turn() {
        // The leak this fixes: without eviction the map keeps one entry per session for the whole
        // run, and every worker's active_sessions ratchets upward and never comes back down.
        let load: Vec<WorkerLoad> = (0..2).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;

        let first = pick_worker(
            HopRouting::LeastLoaded,
            2,
            Some("conv"),
            false,
            &load,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(sticky.len(), 1);
        assert_eq!(load[first].active_sessions.get(), 1);

        let last = pick_worker(
            HopRouting::LeastLoaded,
            2,
            Some("conv"),
            true,
            &load,
            &mut sticky,
            &mut cursor,
        );
        // The final turn still runs on the bound worker, and only then is the binding released.
        assert_eq!(last, first);
        assert!(sticky.is_empty(), "binding outlived its session");
        assert_eq!(load[first].active_sessions.get(), 0);
    }

    #[test]
    fn a_single_turn_session_never_binds_at_all() {
        // Python creates the entry only for a non-final turn; binding then evicting on the same
        // call would churn the map and the counter for nothing.
        let load: Vec<WorkerLoad> = (0..2).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let worker = pick_worker(
            HopRouting::LeastLoaded,
            2,
            Some("one-shot"),
            true,
            &load,
            &mut sticky,
            &mut cursor,
        );
        assert!(sticky.is_empty());
        assert_eq!(load[worker].active_sessions.get(), 0);
    }

    #[test]
    fn the_sticky_map_stays_bounded_across_many_sessions() {
        // A long multi-session run must not accumulate state. Every session opens and closes, so
        // the map returns to empty however many pass through.
        let load: Vec<WorkerLoad> = (0..4).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        for session in 0..500 {
            let id = format!("conv-{session}");
            for turn in 0..3 {
                pick_worker(
                    HopRouting::LeastLoaded,
                    4,
                    Some(&id),
                    turn == 2,
                    &load,
                    &mut sticky,
                    &mut cursor,
                );
            }
        }
        assert!(sticky.is_empty(), "{} bindings leaked", sticky.len());
        for worker in &load {
            assert_eq!(worker.active_sessions.get(), 0);
        }
    }

    #[test]
    fn least_loaded_prefers_fewer_committed_sessions_before_credit_count() {
        // Python's order after depth is active_sessions, then sent, then last_sent. Worker 1 holds
        // more credits but fewer sessions, so it wins.
        let load: Vec<WorkerLoad> = (0..2).map(|_| WorkerLoad::default()).collect();
        load[0].active_sessions.set(3);
        load[0].sent.set(1);
        load[1].active_sessions.set(1);
        load[1].sent.set(99);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        assert_eq!(
            pick_worker(
                HopRouting::LeastLoaded,
                2,
                None,
                false,
                &load,
                &mut sticky,
                &mut cursor
            ),
            1
        );
    }

    #[test]
    fn least_loaded_spreads_the_opening_burst_instead_of_stacking_worker_zero() {
        // Every worker starts at depth zero. A bare min-by-depth resolves every tie to index 0 and
        // sends the whole opening burst there; Python breaks the tie on credits-sent, so the burst
        // fans out. This is the behaviour that tiebreak exists for.
        let load: Vec<WorkerLoad> = (0..4).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let mut picks = Vec::new();
        for i in 0..4 {
            let id = format!("session-{i}");
            let worker = pick_worker(
                HopRouting::LeastLoaded,
                4,
                Some(&id),
                false,
                &load,
                &mut sticky,
                &mut cursor,
            );
            // Stamp the send the way the executor does.
            load[worker].sent.set(load[worker].sent.get() + 1);
            load[worker].last_sent.set(i as u64 + 1);
            picks.push(worker);
        }
        picks.sort_unstable();
        assert_eq!(
            picks,
            vec![0, 1, 2, 3],
            "opening burst stacked on one worker"
        );
    }

    #[test]
    fn least_loaded_prefers_the_worker_idle_longest_when_depth_and_sends_tie() {
        // Same depth, same credits routed: Python takes the oldest last-send. Worker 2 here.
        let load: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        for worker in &load {
            worker.sent.set(5);
        }
        load[0].last_sent.set(90);
        load[1].last_sent.set(80);
        load[2].last_sent.set(10);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        assert_eq!(
            pick_worker(
                HopRouting::LeastLoaded,
                3,
                None,
                false,
                &load,
                &mut sticky,
                &mut cursor
            ),
            2
        );
    }

    #[test]
    fn least_loaded_still_puts_depth_first() {
        // The tiebreaks only apply among workers already tied on in-flight depth: a shallow worker
        // wins even when it has taken far more credits.
        let load: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        load[0].inflight.set(4);
        load[1].inflight.set(4);
        load[2].inflight.set(1);
        load[2].sent.set(1_000);
        load[2].last_sent.set(9_999);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        assert_eq!(
            pick_worker(
                HopRouting::LeastLoaded,
                3,
                None,
                false,
                &load,
                &mut sticky,
                &mut cursor
            ),
            2
        );
    }

    #[test]
    fn pick_worker_round_robin_cycles_in_issue_order() {
        let inflight: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let picks: Vec<usize> = (0..7)
            .map(|_| {
                pick_worker(
                    HopRouting::RoundRobin,
                    3,
                    Some("ignored-under-round-robin"),
                    false,
                    &inflight,
                    &mut sticky,
                    &mut cursor,
                )
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2, 0]);
        assert!(sticky.is_empty(), "round-robin never binds correlations");
    }

    #[test]
    fn pick_worker_sticky_maps_correlation_stably() {
        let inflight: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        // Pin the concrete FNV-1a placement so a hash change is caught.
        assert_eq!((fnv1a64(b"conv-A") % 3) as usize, 1);
        let first = pick_worker(
            HopRouting::Sticky,
            3,
            Some("conv-A"),
            false,
            &inflight,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(first, 1);
        // Same correlation → same worker, and repeated picks do not advance the
        // round-robin cursor.
        for _ in 0..4 {
            assert_eq!(
                pick_worker(
                    HopRouting::Sticky,
                    3,
                    Some("conv-A"),
                    false,
                    &inflight,
                    &mut sticky,
                    &mut cursor,
                ),
                1
            );
        }
        assert_eq!(cursor, 0, "sticky hits do not touch the round-robin cursor");
    }

    #[test]
    fn pick_worker_sticky_falls_back_to_round_robin_without_correlation() {
        let inflight: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let picks: Vec<usize> = (0..4)
            .map(|_| {
                pick_worker(
                    HopRouting::Sticky,
                    3,
                    None,
                    false,
                    &inflight,
                    &mut sticky,
                    &mut cursor,
                )
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0]);
    }

    #[test]
    fn pick_worker_least_loaded_picks_shallowest_then_binds() {
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        // Shallowest queue wins.
        let load: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        load[0].inflight.set(2);
        load[2].inflight.set(1);
        let worker = pick_worker(
            HopRouting::LeastLoaded,
            3,
            Some("conv-A"),
            false,
            &load,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(worker, 1);
        // Continuations stay bound to worker 1 even after it becomes the deepest.
        let deeper: Vec<WorkerLoad> = (0..3).map(|_| WorkerLoad::default()).collect();
        deeper[1].inflight.set(3);
        let worker = pick_worker(
            HopRouting::LeastLoaded,
            3,
            Some("conv-A"),
            false,
            &deeper,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(worker, 1);
    }

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

    impl PreparedEndpointTableFactory for StreamingEndpointTableFactory {
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
        let backend = HttpExecutionFactory
            .build(ExecutionBackendConfig {
                workers: 2,
                coordinator_clock: clock.clone(),
                real_clock_anchor: anchor,
                base_urls: vec![url],
                model: "fixture-model".to_string(),
                transport: TransportSinkConfig::default(),
                raw_enabled: false,
                prepared_endpoints: Some(table_factory),
                hop_routing: HopRouting::RoundRobin,
                virtual_worker_width: None,
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
            request: Request {
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
                image_count: None,
                recorded_api_time_ns: None,
                recorded_ttft_ns: None,
            },
            model: "fixture-model".to_string(),
            endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
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
