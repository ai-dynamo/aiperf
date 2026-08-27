// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The generic execution capsule every request transport plugin links.
//!
//! One worker loop for every transport: the sink type is the only variable,
//! supplied by [`ExecutionSinkBuilder::Sink`]. Worker threads, routing, drain,
//! and shutdown are written once here and monomorphized into each plugin, so
//! [`execute_worker_command`] calls
//! [`WorkerSink::dispatch_measured`] through a statically known type rather than
//! through `Box<dyn WorkerSink>`.
//!
//! Two placements, chosen by worker count:
//!
//! - `workers == 1` keeps the sink co-located on the caller's own reactor and
//!   returns it *directly* as the [`RequestExecutor`]. There is no wrapper, no
//!   channel, and no thread hop. This is the placement most runs take, which is
//!   why [`ExecutionSinkBuilder::Sink`] is bound `RequestExecutor` at all.
//! - `workers > 1` builds [`ThreadPerCoreExecutor`]: one coordinator-side
//!   scheduling loop issues every request in exact global order and places it on
//!   one of `workers` worker OS threads over a bounded mpsc command queue, driven
//!   to terminal by that thread's worker-local sink and returned over a oneshot.
//!   Exact request-to-thread assignment order is the property this placement has
//!   and a shared-admission model does not.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;

use aiperf_core::clock::Clock;
use aiperf_core::dispatch::{ReplayTerminalStatus, RequestObserver};
use aiperf_plugin_api::transport::{BoundaryRequest, BoundaryTerminal, RequestExecutor};
use aiperf_plugin_api::validation::ValidationError;
use anyhow::{Result, anyhow};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::direct::{ExecutionSinkBuilder, WorkerRequest, WorkerSink, WorkerTerminal};

/// Depth of one worker's bounded command queue.
///
/// Bounded on purpose: an unbounded queue would let the coordinator run ahead of
/// a stalled worker and turn backpressure into unbounded memory growth. The
/// admission gate above this capsule is what actually limits in-flight work, so
/// this only needs to absorb short scheduling jitter.
pub const WORKER_COMMAND_QUEUE_DEPTH: usize = 1024;

/// Where the coordinator places each request among the worker threads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HopRouting {
    /// Request `i` goes to worker `i % workers`. Deterministic and stateless.
    #[default]
    RoundRobin,
    /// A session hashes to a fixed worker; sessionless requests round-robin.
    Sticky,
    /// The shallowest worker wins, and a multi-turn session then stays there.
    LeastLoaded,
}

/// Per-worker load signals consulted when workers tie on in-flight depth.
#[derive(Debug, Default)]
pub struct WorkerLoad {
    /// Commands currently in flight on this worker.
    pub inflight: Cell<usize>,
    /// Total commands ever routed here; the first tiebreak.
    pub routed: Cell<u64>,
    /// Monotonic send sequence of the last command routed here.
    ///
    /// A counter rather than a clock reading, so the tiebreak orders by issuance
    /// and stays identical under `SimClock`.
    pub last_sent: Cell<u64>,
    /// Multi-turn sessions currently bound to this worker.
    pub active_sessions: Cell<usize>,
}

/// Build one execution backend for `workers` execution workers.
///
/// `workers == 1` returns the co-located sink itself; `workers > 1` returns a
/// [`ThreadPerCoreExecutor`]. The caller wraps the result in nothing.
pub fn build_native_request_executor<B: ExecutionSinkBuilder>(
    builder: B,
    workers: usize,
    coordinator_clock: Rc<dyn Clock>,
    worker_clocks: Arc<dyn WorkerClockFactory>,
    routing: HopRouting,
) -> Result<Rc<dyn RequestExecutor>, ValidationError> {
    if workers == 0 {
        return Err(ValidationError::Rejected(
            "execution workers must be positive".to_owned(),
        ));
    }
    if workers == 1 {
        // A co-located sink runs on the caller's own thread, so the caller — not
        // a worker — attributes the record.
        let sink = builder
            .build_sink(coordinator_clock, 0)
            .map_err(|error| ValidationError::Rejected(error.to_string()))?;
        return Ok(Rc::new(sink));
    }
    let executor = ThreadPerCoreExecutor::new(builder, workers, worker_clocks, routing)
        .map_err(|error| ValidationError::Rejected(error.to_string()))?;
    Ok(Rc::new(executor))
}

/// Rebuilds the run's clock on a worker thread.
///
/// `Rc<dyn Clock>` cannot cross a thread boundary, so each worker constructs its
/// own handle to the same time source. The factory is what carries the anchor
/// (the real clock's origin, or the simulation clock's shared event queue) across
/// the spawn.
pub trait WorkerClockFactory: Send + Sync + 'static {
    /// Build this worker's handle to the run's clock.
    fn build(&self, worker_id: usize) -> Rc<dyn Clock>;
}

/// Builds a worker's measurement observer on the worker thread.
///
/// Worker-local by construction: accumulating per worker and merging once at the
/// drain boundary is what keeps the per-token path free of shared-state
/// contention.
pub trait WorkerObserverFactory: Send + Sync + 'static {
    /// Build worker `worker_id`'s observer.
    fn build(&self, worker_id: usize) -> Rc<dyn RequestObserver>;
}

/// An observer that records nothing.
///
/// The default when a placement is built without a [`WorkerObserverFactory`],
/// used by prewarm and by transports whose measurement is applied above this
/// capsule.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopObserver;

impl RequestObserver for NoopObserver {
    fn on_arrival(&self, _uuid: Uuid, _arrival_ms: f64, _input: usize, _requested: usize) {}
    fn on_admit(&self, _uuid: Uuid, _admit_ms: f64, _reused_input_tokens: usize) {}
    fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}
    fn on_terminal(&self, _uuid: Uuid, _status: ReplayTerminalStatus) {}
}

/// One dispatched request plus the channel its terminal returns on.
pub struct WorkerCommand {
    /// The owned request to drive to terminal.
    pub request: WorkerRequest,
    /// Where this command's terminal is delivered.
    pub reply: oneshot::Sender<WorkerTerminal>,
}

/// What the coordinator sends down a worker's command queue.
pub enum WorkerMessage {
    /// Drive one request to terminal.
    Execute(Box<WorkerCommand>),
    /// Anchor the worker's measurement to the run origin.
    SetRunOrigin(i64),
    /// Warm connections without measuring.
    Prewarm(Box<WorkerRequest>),
    /// Materialize and drive an identity-only credit.
    ///
    /// The body is built worker-side, so the issuer sits in neither the request's
    /// lifetime nor its body construction.
    Credit(u64),
    /// Stop accepting work and release worker-local resources.
    Shutdown,
}

/// Pure worker-assignment decision for one request.
///
/// Factored out of [`ThreadPerCoreExecutor`] so the routing policy is testable
/// without live worker threads. `rr_cursor` advances only when a round-robin pick
/// is actually made, and `sticky` holds session-key bindings for
/// [`HopRouting::LeastLoaded`].
///
/// A binding is created only on a non-final turn — a single-turn session would
/// otherwise be inserted and evicted on the same call — and released on the
/// session's final turn. Without the release the map grows one entry per session
/// for the lifetime of the run and every worker's `active_sessions` ratchets
/// upward, biasing placement toward whichever worker took fewest sessions early.
pub fn pick_worker(
    routing: HopRouting,
    workers: usize,
    session_key: Option<&str>,
    is_final_turn: bool,
    inflight: &[WorkerLoad],
    sticky: &mut HashMap<String, usize>,
    rr_cursor: &mut usize,
) -> usize {
    debug_assert!(workers > 0, "worker count must be positive");
    match routing {
        HopRouting::RoundRobin => round_robin(workers, rr_cursor),
        HopRouting::Sticky => match session_key {
            Some(key) => (fnv1a64(key.as_bytes()) % workers as u64) as usize,
            None => round_robin(workers, rr_cursor),
        },
        HopRouting::LeastLoaded => {
            let bound = session_key.and_then(|key| sticky.get(key).copied());
            let worker = match bound {
                Some(worker) => worker,
                None => {
                    let worker = least_loaded(inflight);
                    if let Some(key) = session_key
                        && !is_final_turn
                    {
                        sticky.insert(key.to_owned(), worker);
                        let load = &inflight[worker];
                        load.active_sessions.set(load.active_sessions.get() + 1);
                    }
                    worker
                }
            };
            if is_final_turn
                && let Some(key) = session_key
                && let Some(released) = sticky.remove(key)
            {
                let load = &inflight[released];
                load.active_sessions
                    .set(load.active_sessions.get().saturating_sub(1));
            }
            worker
        }
    }
}

fn round_robin(workers: usize, rr_cursor: &mut usize) -> usize {
    let worker = *rr_cursor % workers;
    *rr_cursor = worker.wrapping_add(1);
    worker
}

fn least_loaded(inflight: &[WorkerLoad]) -> usize {
    let mut best = 0usize;
    for candidate in 1..inflight.len() {
        let current = &inflight[best];
        let other = &inflight[candidate];
        let is_better = (
            other.inflight.get(),
            other.routed.get(),
            other.last_sent.get(),
        ) < (
            current.inflight.get(),
            current.routed.get(),
            current.last_sent.get(),
        );
        if is_better {
            best = candidate;
        }
    }
    best
}

/// FNV-1a over the session key. A stable, allocation-free hash so the same
/// session lands on the same worker across processes and across runs.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Generic thread-per-core placement behind a single coordinator.
///
/// Single-threaded coordinator, so plain `Cell`/`RefCell` — there is no second
/// thread touching this state, and an `Arc<Mutex<_>>` here would add contention
/// to the issuance path for nothing.
pub struct ThreadPerCoreExecutor<B: ExecutionSinkBuilder> {
    senders: RefCell<Option<Vec<mpsc::Sender<WorkerMessage>>>>,
    threads: RefCell<Vec<JoinHandle<Result<()>>>>,
    routing: HopRouting,
    next_worker: Cell<usize>,
    send_seq: Cell<u64>,
    inflight: Vec<WorkerLoad>,
    sticky: RefCell<HashMap<String, usize>>,
    run_origin_ns: Cell<Option<i64>>,
    label: &'static str,
    builder: std::marker::PhantomData<fn() -> B>,
}

impl<B: ExecutionSinkBuilder> ThreadPerCoreExecutor<B> {
    /// Spawn `workers` worker threads, each owning one `B::Sink`.
    ///
    /// Every worker reports its construction result before this returns, so a
    /// sink that fails to build surfaces here rather than as a silently missing
    /// worker discovered on the first dispatch.
    pub fn new(
        builder: B,
        workers: usize,
        worker_clocks: Arc<dyn WorkerClockFactory>,
        routing: HopRouting,
    ) -> Result<Self> {
        if workers == 0 {
            return Err(anyhow!("execution workers must be positive"));
        }
        let label = builder.label();
        let builder = Arc::new(builder);
        let mut senders = Vec::with_capacity(workers);
        let mut threads = Vec::with_capacity(workers);
        let mut inflight = Vec::with_capacity(workers);

        for worker_id in 0..workers {
            let (sender, receiver) = mpsc::channel(WORKER_COMMAND_QUEUE_DEPTH);
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
            let worker_builder = Arc::clone(&builder);
            let clocks = Arc::clone(&worker_clocks);
            let handle = std::thread::Builder::new()
                .name(format!("aiperf-{label}-{worker_id}"))
                .spawn(move || {
                    run_worker_thread::<B>(receiver, worker_builder, clocks, worker_id, started_tx)
                })?;
            match started_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(anyhow!("worker {worker_id} failed to start: {error}")),
                Err(_) => return Err(anyhow!("worker {worker_id} exited before reporting startup")),
            }
            senders.push(sender);
            threads.push(handle);
            inflight.push(WorkerLoad::default());
        }

        Ok(Self {
            senders: RefCell::new(Some(senders)),
            threads: RefCell::new(threads),
            routing,
            next_worker: Cell::new(0),
            send_seq: Cell::new(0),
            inflight,
            sticky: RefCell::new(HashMap::new()),
            run_origin_ns: Cell::new(None),
            label,
            builder: std::marker::PhantomData,
        })
    }

    /// The builder's diagnostic label.
    pub const fn label(&self) -> &'static str {
        self.label
    }

    /// Anchor every worker's measurement to `origin_ns`.
    pub fn set_run_origin(&self, origin_ns: i64) {
        self.run_origin_ns.set(Some(origin_ns));
        let senders = self.senders.borrow();
        let Some(senders) = senders.as_ref() else {
            return;
        };
        for sender in senders {
            // A full queue at origin-set time means the worker is already busy
            // with a prior run's tail; dropping the anchor is worse than
            // blocking the coordinator, so use the blocking form.
            if let Err(error) = sender.blocking_send(WorkerMessage::SetRunOrigin(origin_ns)) {
                tracing::debug!(error = %error, component = "transport-sdk", "worker closed before run origin");
            }
        }
    }

    /// Route and dispatch one request, returning its terminal.
    pub async fn execute_command(&self, request: WorkerRequest) -> WorkerTerminal {
        let worker = {
            let mut sticky = self.sticky.borrow_mut();
            let mut cursor = self.next_worker.get();
            let picked = pick_worker(
                self.routing,
                self.inflight.len(),
                request.session_key(),
                request.is_final_turn(),
                &self.inflight,
                &mut sticky,
                &mut cursor,
            );
            self.next_worker.set(cursor);
            picked
        };
        let sequence = self.send_seq.get().wrapping_add(1);
        self.send_seq.set(sequence);
        let load = &self.inflight[worker];
        load.routed.set(load.routed.get().saturating_add(1));
        load.last_sent.set(sequence);

        let sender = {
            let senders = self.senders.borrow();
            match senders.as_ref() {
                Some(senders) => senders[worker].clone(),
                None => return WorkerTerminal::failed("placement_shut_down"),
            }
        };

        let (reply_tx, reply_rx) = oneshot::channel();
        let command = WorkerCommand {
            request,
            reply: reply_tx,
        };
        // Decrement on every exit path, including the send failure below, so a
        // cancelled command still releases its `LeastLoaded` slot.
        let _guard = InflightGuard::new(&load.inflight);
        if sender
            .send(WorkerMessage::Execute(Box::new(command)))
            .await
            .is_err()
        {
            return WorkerTerminal::failed("worker_unavailable");
        }
        match reply_rx.await {
            Ok(terminal) => terminal,
            Err(_) => WorkerTerminal::failed("worker_exited"),
        }
    }

    /// Close every command queue and join every worker thread.
    pub fn shutdown(&self) {
        let senders = self.senders.borrow_mut().take();
        if let Some(senders) = senders {
            for sender in &senders {
                let _ = sender.blocking_send(WorkerMessage::Shutdown);
            }
            drop(senders);
        }
        for handle in self.threads.borrow_mut().drain(..) {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::debug!(error = %error, component = "transport-sdk", "worker exited with error");
                }
                Err(_) => {
                    tracing::debug!(component = "transport-sdk", "worker thread panicked");
                }
            }
        }
    }
}

impl<B: ExecutionSinkBuilder> RequestExecutor for ThreadPerCoreExecutor<B> {
    fn execute<'a>(
        &'a self,
        request: &'a dyn BoundaryRequest,
    ) -> Pin<Box<dyn Future<Output = Box<dyn BoundaryTerminal>> + 'a>> {
        let owned = WorkerRequest::from_boundary(request);
        Box::pin(async move { Box::new(self.execute_command(owned).await) as Box<dyn BoundaryTerminal> })
    }
}

impl<B: ExecutionSinkBuilder> Drop for ThreadPerCoreExecutor<B> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Decrements a worker's in-flight counter on drop.
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

/// One worker OS thread: build the sink, then run the loop to completion.
///
/// The sink is built *here*, on the worker's own thread, so `!Send` worker-local
/// state never has to move between threads. `started` reports the build result
/// back to the coordinator before the loop begins.
pub fn run_worker_thread<B: ExecutionSinkBuilder>(
    receiver: mpsc::Receiver<WorkerMessage>,
    builder: Arc<B>,
    clocks: Arc<dyn WorkerClockFactory>,
    worker_id: usize,
    started: std::sync::mpsc::SyncSender<Result<(), String>>,
) -> Result<()> {
    let clock = clocks.build(worker_id);
    let sink = match builder.build_sink(clock.clone(), worker_id) {
        Ok(sink) => {
            let _ = started.send(Ok(()));
            Rc::new(sink)
        }
        Err(error) => {
            let _ = started.send(Err(error.to_string()));
            return Err(error);
        }
    };
    let materializer = builder.build_credit_materializer()?;
    // `Clock::drive` supplies the reactor discipline: a real clock parks on IO
    // and timers, a virtual clock advances event by event. Going through it is
    // what keeps a simulated run deterministic on a worker thread.
    clock.drive(Box::pin(async move {
        run_worker::<B::Sink>(receiver, sink, materializer, NoopObserver).await;
    }));
    Ok(())
}

/// The worker loop, monomorphized over the concrete sink type `S`.
///
/// `S` is a type parameter and not `Box<dyn WorkerSink>` on purpose: the call to
/// [`WorkerSink::dispatch_measured`] inside [`execute_worker_command`] must stay
/// statically dispatched, because it is on the per-request path.
pub async fn run_worker<S: WorkerSink + 'static>(
    mut receiver: mpsc::Receiver<WorkerMessage>,
    sink: Rc<S>,
    materializer: Option<Box<dyn crate::direct::CreditMaterializer>>,
    observer: impl RequestObserver + Clone + 'static,
) -> bool {
    let mut clean_exit = true;
    while let Some(message) = receiver.recv().await {
        match message {
            WorkerMessage::Execute(command) => {
                execute_worker_command::<S>(Rc::clone(&sink), *command, observer.clone()).await;
            }
            WorkerMessage::SetRunOrigin(origin_ns) => sink.set_run_origin(origin_ns),
            WorkerMessage::Prewarm(request) => {
                if let Err(error) = sink.prewarm(request.as_ref()).await {
                    tracing::debug!(error = %error, component = "transport-sdk", "prewarm failed");
                }
            }
            WorkerMessage::Credit(correlation_id) => {
                let Some(materializer) = materializer.as_ref() else {
                    tracing::debug!(
                        component = "transport-sdk",
                        "credit dispatched to a worker with no materializer"
                    );
                    clean_exit = false;
                    continue;
                };
                match materializer.materialize(correlation_id) {
                    Ok(request) => {
                        let (reply, _) = oneshot::channel();
                        let command = WorkerCommand { request, reply };
                        execute_worker_command::<S>(
                            Rc::clone(&sink),
                            command,
                            observer.clone(),
                        )
                        .await;
                    }
                    Err(error) => {
                        tracing::debug!(error = %error, component = "transport-sdk", "credit materialization failed");
                        clean_exit = false;
                    }
                }
            }
            WorkerMessage::Shutdown => break,
        }
    }
    if let Err(error) = sink.shutdown().await {
        tracing::debug!(error = %error, component = "transport-sdk", "sink shutdown failed");
        clean_exit = false;
    }
    clean_exit
}

/// Drive one command to terminal on this worker and answer its reply channel.
///
/// Measurement is applied exactly once, here, around the sink call. The sink's
/// own `dispatch_measured` deliberately does not measure; double-wrapping would
/// change recorded TTFT and latency.
pub async fn execute_worker_command<S: WorkerSink + 'static>(
    sink: Rc<S>,
    command: WorkerCommand,
    observer: impl RequestObserver,
) {
    let WorkerCommand { request, reply } = command;
    let terminal = crate::measure::measure_dispatch(
        &observer,
        sink.clock(),
        crate::measure::ArrivalFacts::from_request(&request),
        sink.dispatch_measured(&observer, &request),
    )
    .await;
    let terminal = match terminal {
        Ok(terminal) => WorkerTerminal::from_boundary(terminal.as_ref()),
        Err(error) => {
            tracing::debug!(error = %error, component = "transport-sdk", "dispatch failed");
            WorkerTerminal::failed("dispatch_error")
        }
    };
    // A dropped receiver means the coordinator abandoned this request; the work
    // is already done and there is nothing to recover.
    let _ = reply.send(terminal);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loads(depths: &[usize]) -> Vec<WorkerLoad> {
        depths
            .iter()
            .map(|&depth| {
                let load = WorkerLoad::default();
                load.inflight.set(depth);
                load
            })
            .collect()
    }

    #[test]
    fn round_robin_walks_every_worker_in_order() {
        let inflight = loads(&[0, 0, 0]);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let picks: Vec<usize> = (0..5)
            .map(|_| {
                pick_worker(
                    HopRouting::RoundRobin,
                    3,
                    None,
                    true,
                    &inflight,
                    &mut sticky,
                    &mut cursor,
                )
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn least_loaded_binds_a_multi_turn_session_and_releases_it() {
        let inflight = loads(&[2, 0]);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let first = pick_worker(
            HopRouting::LeastLoaded,
            2,
            Some("session-a"),
            false,
            &inflight,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(first, 1);
        assert_eq!(sticky.get("session-a"), Some(&1));
        assert_eq!(inflight[1].active_sessions.get(), 1);

        // The binding wins even after worker 1 becomes the deeper worker.
        inflight[1].inflight.set(9);
        let second = pick_worker(
            HopRouting::LeastLoaded,
            2,
            Some("session-a"),
            true,
            &inflight,
            &mut sticky,
            &mut cursor,
        );
        assert_eq!(second, 1);
        assert!(sticky.is_empty(), "final turn must release the binding");
        assert_eq!(inflight[1].active_sessions.get(), 0);
    }

    #[test]
    fn sticky_routing_is_stable_for_the_same_session_key() {
        let inflight = loads(&[0, 0, 0, 0]);
        let mut sticky = HashMap::new();
        let mut cursor = 0usize;
        let mut pick = || {
            pick_worker(
                HopRouting::Sticky,
                4,
                Some("conversation-7"),
                false,
                &inflight,
                &mut sticky,
                &mut cursor,
            )
        };
        let first = pick();
        assert_eq!(first, pick());
        assert!(first < 4);
    }
}
