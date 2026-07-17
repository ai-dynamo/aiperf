// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-trace placement implementations.
//!
//! The native pool assigns complete traces round-robin to OS threads. Each
//! worker owns a current-thread Tokio runtime, a `LocalSet`, and its backend,
//! preserving the crate's lock-free thread-per-core execution model.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{self, Display};
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinSet, LocalSet};

use crate::graph::errors::TraceError;
use crate::graph::execution::TracePlacement;
use crate::graph::model::GraphTracePlan;

/// Default number of complete trace commands buffered per placement worker.
pub const DEFAULT_GRAPH_WORKER_QUEUE_CAPACITY: usize = 256;

/// Builds one worker-local backend after entering its owning OS thread.
///
/// Factory state must be shareable, but the returned backend deliberately need
/// not be `Send` or `Sync`: it never leaves that worker's `LocalSet`.
pub trait TracePlacementFactory: Send + Sync {
    /// Construct the backend owned by `worker_id`.
    ///
    /// `clock` overrides the backend's time source. Thread-per-core workers pass
    /// `None` and let the factory build a `RealClock` bound to each worker's own
    /// reactor (the `!Send` clock cannot cross the thread boundary, so it is
    /// reconstructed from a `Send` anchor). The single-reactor inline placement
    /// passes `Some(injected_clock)` — the run's `SimClock` under a virtual run —
    /// so its `Clock::sleep`s are virtual-time events the idle-pump can advance
    /// rather than real timerfd sleeps (which would panic on the IO-less pump).
    fn create_backend(
        &self,
        worker_id: usize,
        clock: Option<Rc<dyn crate::clock::Clock>>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError>;
}

/// Native thread-per-core whole-trace placement backend.
pub struct ThreadPerCoreTracePlacement {
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    controls: Vec<mpsc::UnboundedSender<WorkerControl>>,
    next_worker: Cell<usize>,
    cancelled: Cell<bool>,
    prefill_shards: RefCell<Option<Vec<usize>>>,
    threads: RefCell<Vec<JoinHandle<()>>>,
}

impl ThreadPerCoreTracePlacement {
    /// Start `worker_count` current-thread runtimes and build one backend each.
    pub fn new(
        worker_count: usize,
        factory: Arc<dyn TracePlacementFactory>,
    ) -> Result<Self, GraphPlacementError> {
        Self::new_with_queue_capacity(worker_count, DEFAULT_GRAPH_WORKER_QUEUE_CAPACITY, factory)
    }

    /// Start workers with an explicit per-worker whole-trace queue capacity.
    pub fn new_with_queue_capacity(
        worker_count: usize,
        queue_capacity: usize,
        factory: Arc<dyn TracePlacementFactory>,
    ) -> Result<Self, GraphPlacementError> {
        if worker_count == 0 {
            return Err(GraphPlacementError(
                "graph placement requires at least one worker".into(),
            ));
        }
        if queue_capacity == 0 {
            return Err(GraphPlacementError(
                "graph placement queue capacity must be positive".into(),
            ));
        }

        let mut senders = Vec::with_capacity(worker_count);
        let mut controls = Vec::with_capacity(worker_count);
        let mut threads = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let (command_tx, command_rx) = mpsc::channel(queue_capacity);
            let (control_tx, control_rx) = mpsc::unbounded_channel();
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
            let worker_factory = factory.clone();
            let thread = match std::thread::Builder::new()
                .name(format!("aiperf-graph-{worker_id}"))
                .spawn(move || {
                    worker_thread(worker_id, worker_factory, command_rx, control_rx, ready_tx)
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    stop_workers(senders, controls, threads);
                    return Err(GraphPlacementError(format!(
                        "failed to spawn graph worker {worker_id}: {error}"
                    )));
                }
            };
            senders.push(command_tx);
            controls.push(control_tx);
            threads.push(thread);

            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    stop_workers(senders, controls, threads);
                    return Err(GraphPlacementError(error));
                }
                Err(error) => {
                    stop_workers(senders, controls, threads);
                    return Err(GraphPlacementError(format!(
                        "graph worker {worker_id} exited during startup: {error}"
                    )));
                }
            }
        }

        Ok(Self {
            senders,
            controls,
            next_worker: Cell::new(0),
            cancelled: Cell::new(false),
            prefill_shards: RefCell::new(None),
            threads: RefCell::new(threads),
        })
    }

    /// Number of native placement workers.
    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }
}

#[async_trait(?Send)]
impl TracePlacement for ThreadPerCoreTracePlacement {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        if self.cancelled.get() {
            return Err(cancelled_trace(&plan.trace.id));
        }
        let worker_id = self.next_execution_worker().ok_or_else(|| {
            TraceError::Other("graph placement has no worker with positive prefill capacity".into())
        })?;
        let (result_tx, result_rx) = oneshot::channel();
        self.senders[worker_id]
            .send(WorkerCommand::Execute {
                plan,
                result: result_tx,
            })
            .await
            .map_err(|_| {
                TraceError::Other(format!(
                    "graph placement worker {worker_id} is not available"
                ))
            })?;
        result_rx.await.map_err(|_| {
            TraceError::Other(format!(
                "graph placement worker {worker_id} exited before returning its trace"
            ))
        })?
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.cancelled.set(true);
        self.broadcast_control(|_worker_id, ack| WorkerControl::CancelInflight { ack })
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
        if limit == 0 {
            return Err(TraceError::Other(
                "graph prefill limit must be positive".into(),
            ));
        }
        let shards = shard_global_limit(limit, self.worker_count());
        self.broadcast_control(|worker_id, ack| WorkerControl::SetPrefillLimit {
            limit: shards[worker_id],
            ack,
        })?;
        *self.prefill_shards.borrow_mut() = Some(shards);
        Ok(())
    }
}

impl ThreadPerCoreTracePlacement {
    fn next_execution_worker(&self) -> Option<usize> {
        let worker_count = self.worker_count();
        let start = self.next_worker.get() % worker_count;
        let shards = self.prefill_shards.borrow();
        for offset in 0..worker_count {
            let worker_id = (start + offset) % worker_count;
            if shards.as_ref().is_none_or(|limits| limits[worker_id] > 0) {
                self.next_worker.set((worker_id + 1) % worker_count);
                return Some(worker_id);
            }
        }
        None
    }

    fn broadcast_control(
        &self,
        control: impl Fn(usize, std::sync::mpsc::SyncSender<Result<(), String>>) -> WorkerControl,
    ) -> Result<(), TraceError> {
        let mut errors = Vec::new();
        for (worker_id, sender) in self.controls.iter().enumerate() {
            let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
            if sender.send(control(worker_id, ack_tx)).is_err() {
                errors.push(format!(
                    "graph placement worker {worker_id} is unavailable for control updates"
                ));
                continue;
            }
            match ack_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => errors.push(format!(
                    "graph placement worker {worker_id} rejected its control update: {error}"
                )),
                Err(_) => errors.push(format!(
                        "graph placement worker {worker_id} exited before acknowledging its control update"
                )),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(TraceError::Other(format!(
                "graph placement control fanout failed: {}",
                errors.join("; ")
            )))
        }
    }
}

fn shard_global_limit(limit: usize, worker_count: usize) -> Vec<usize> {
    let quotient = limit / worker_count;
    let remainder = limit % worker_count;
    (0..worker_count)
        .map(|worker_id| quotient + usize::from(worker_id < remainder))
        .collect()
}

fn cancelled_trace(trace_id: &str) -> TraceError {
    TraceError::Cancelled(format!(
        "graph trace {trace_id:?} was rejected after placement cancellation"
    ))
}

impl Drop for ThreadPerCoreTracePlacement {
    fn drop(&mut self) {
        self.senders.clear();
        self.controls.clear();
        for thread in self.threads.get_mut().drain(..) {
            let _ = thread.join();
        }
    }
}

/// Single-reactor whole-trace placement: every trace runs inline on the
/// caller's current-thread reactor, spawning no worker OS threads.
///
/// The thread-per-core placement is the online default because it fans traces
/// across cores for throughput, but it is fundamentally incompatible with a
/// virtual [`crate::clock::SimClock`]: each worker thread owns its own reactor,
/// and the coordinator's `drive_sim` idle-pump can only advance the sleepers of
/// the *one* reactor it drives. Under sim, work placed on worker threads simply
/// never has its virtual-time arrivals advanced, so the replay stalls after the
/// first root node. This placement collapses all per-trace execution onto the
/// coordinator's single reactor — the same reactor `drive_sim` drives — so every
/// node's `Clock::sleep` is a schedulable virtual-time event. The coordinator
/// still `spawn_local`s each `execute_trace`, so independent traces overlap
/// concurrently on that one reactor exactly as they would across worker threads.
///
/// One backend is built eagerly on the current thread (it is `!Send`, so it can
/// never leave this reactor) and shared by every concurrent trace, matching how
/// a thread-per-core worker's single backend serves all traces routed to it.
pub struct LocalTracePlacement {
    backend: Rc<dyn TracePlacement>,
}

impl LocalTracePlacement {
    /// Build the current-thread backend from `factory` (worker slot `0`) over the
    /// injected `clock` — the run's single reactor clock, so a virtual `SimClock`
    /// drives the backend's sleeps in virtual time.
    pub fn new(
        factory: Arc<dyn TracePlacementFactory>,
        clock: Rc<dyn crate::clock::Clock>,
    ) -> Result<Self, GraphPlacementError> {
        Ok(Self {
            backend: factory.create_backend(0, Some(clock))?,
        })
    }
}

#[async_trait(?Send)]
impl TracePlacement for LocalTracePlacement {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        self.backend.execute_trace(plan).await
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.backend.cancel_inflight()
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
        self.backend.set_prefill_limit(limit)
    }
}

enum WorkerCommand {
    Execute {
        plan: GraphTracePlan,
        result: oneshot::Sender<Result<(), TraceError>>,
    },
}

enum WorkerControl {
    CancelInflight {
        ack: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    SetPrefillLimit {
        limit: usize,
        ack: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
}

fn worker_thread(
    worker_id: usize,
    factory: Arc<dyn TracePlacementFactory>,
    commands: mpsc::Receiver<WorkerCommand>,
    controls: mpsc::UnboundedReceiver<WorkerControl>,
    ready: std::sync::mpsc::SyncSender<Result<(), String>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready.send(Err(format!(
                "failed to create graph worker {worker_id} runtime: {error}"
            )));
            return;
        }
    };
    let local = LocalSet::new();
    runtime.block_on(local.run_until(async move {
        let backend = match factory.create_backend(worker_id, None) {
            Ok(backend) => backend,
            Err(error) => {
                let _ = ready.send(Err(format!(
                    "failed to initialize graph worker {worker_id}: {error}"
                )));
                return;
            }
        };
        if ready.send(Ok(())).is_err() {
            return;
        }

        worker_loop(backend, commands, controls).await;
    }));
}

async fn worker_loop(
    backend: Rc<dyn TracePlacement>,
    mut commands: mpsc::Receiver<WorkerCommand>,
    mut controls: mpsc::UnboundedReceiver<WorkerControl>,
) {
    let mut active = JoinSet::new();
    let mut cancelled = false;
    loop {
        tokio::select! {
            biased;
            control = controls.recv() => match control {
                Some(WorkerControl::CancelInflight { ack }) => {
                    cancelled = true;
                    let result = backend.cancel_inflight().map_err(|error| error.to_string());
                    reject_queued_commands(&mut commands);
                    let _ = ack.send(result);
                }
                Some(WorkerControl::SetPrefillLimit { limit, ack }) => {
                    let result = backend
                        .set_prefill_limit(limit)
                        .map_err(|error| error.to_string());
                    let _ = ack.send(result);
                }
                None => break,
            },
            command = commands.recv() => match command {
                Some(command) if cancelled => reject_command(command),
                Some(WorkerCommand::Execute { plan, result }) => {
                    let backend = backend.clone();
                    active.spawn_local(async move {
                        let _ = result.send(backend.execute_trace(plan).await);
                    });
                }
                None => break,
            },
            completed = active.join_next(), if !active.is_empty() => {
                let _ = completed;
            }
        }
    }
    while active.join_next().await.is_some() {}
}

fn reject_queued_commands(commands: &mut mpsc::Receiver<WorkerCommand>) {
    while let Ok(command) = commands.try_recv() {
        reject_command(command);
    }
}

fn reject_command(command: WorkerCommand) {
    let WorkerCommand::Execute { plan, result } = command;
    let _ = result.send(Err(cancelled_trace(&plan.trace.id)));
}

fn stop_workers(
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    controls: Vec<mpsc::UnboundedSender<WorkerControl>>,
    threads: Vec<JoinHandle<()>>,
) {
    drop(senders);
    drop(controls);
    for thread in threads {
        let _ = thread.join();
    }
}

/// Worker construction or placement setup failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPlacementError(pub String);

impl Display for GraphPlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GraphPlacementError {}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio::sync::Notify;

    use super::*;
    use crate::graph::model::{GraphRecord, TraceRecord};

    struct RecordingFactory {
        placements: Arc<Mutex<Vec<(usize, String)>>>,
        prefill_limits: Arc<Mutex<Vec<(usize, usize)>>>,
    }

    impl TracePlacementFactory for RecordingFactory {
        fn create_backend(
            &self,
            worker_id: usize,
            _clock: Option<Rc<dyn crate::clock::Clock>>,
        ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
            Ok(Rc::new(RecordingWorker {
                worker_id,
                placements: self.placements.clone(),
                prefill_limits: self.prefill_limits.clone(),
            }))
        }
    }

    struct RecordingWorker {
        worker_id: usize,
        placements: Arc<Mutex<Vec<(usize, String)>>>,
        prefill_limits: Arc<Mutex<Vec<(usize, usize)>>>,
    }

    struct FanoutFactory {
        cancellations: Arc<Mutex<Vec<usize>>>,
        failing_worker: usize,
    }

    impl TracePlacementFactory for FanoutFactory {
        fn create_backend(
            &self,
            worker_id: usize,
            _clock: Option<Rc<dyn crate::clock::Clock>>,
        ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
            Ok(Rc::new(FanoutWorker {
                worker_id,
                cancellations: self.cancellations.clone(),
                failing_worker: self.failing_worker,
            }))
        }
    }

    struct FanoutWorker {
        worker_id: usize,
        cancellations: Arc<Mutex<Vec<usize>>>,
        failing_worker: usize,
    }

    #[async_trait(?Send)]
    impl TracePlacement for FanoutWorker {
        async fn execute_trace(&self, _plan: GraphTracePlan) -> Result<(), TraceError> {
            Ok(())
        }

        fn cancel_inflight(&self) -> Result<(), TraceError> {
            self.cancellations.lock().unwrap().push(self.worker_id);
            if self.worker_id == self.failing_worker {
                Err(TraceError::Other(format!(
                    "intentional worker {} cancellation failure",
                    self.worker_id
                )))
            } else {
                Ok(())
            }
        }
    }

    #[derive(Default)]
    struct CancellableWorkerState {
        started: AtomicBool,
        cancelled: AtomicBool,
        terminal_cleanups: AtomicUsize,
        wake: Notify,
    }

    struct CancellableFactory {
        state: Arc<CancellableWorkerState>,
    }

    impl TracePlacementFactory for CancellableFactory {
        fn create_backend(
            &self,
            _worker_id: usize,
            _clock: Option<Rc<dyn crate::clock::Clock>>,
        ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
            Ok(Rc::new(CancellableWorker {
                state: self.state.clone(),
            }))
        }
    }

    struct CancellableWorker {
        state: Arc<CancellableWorkerState>,
    }

    #[async_trait(?Send)]
    impl TracePlacement for CancellableWorker {
        async fn execute_trace(&self, _plan: GraphTracePlan) -> Result<(), TraceError> {
            self.state.started.store(true, Ordering::SeqCst);
            self.state.wake.notify_waiters();
            loop {
                let notified = self.state.wake.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.state.cancelled.load(Ordering::SeqCst) {
                    self.state.terminal_cleanups.fetch_add(1, Ordering::SeqCst);
                    return Err(TraceError::Cancelled(
                        "worker completed graceful terminal cancellation".into(),
                    ));
                }
                notified.await;
            }
        }

        fn cancel_inflight(&self) -> Result<(), TraceError> {
            self.state.cancelled.store(true, Ordering::SeqCst);
            self.state.wake.notify_waiters();
            Ok(())
        }
    }

    struct QueueCancellationWorker {
        cancellations: Rc<Cell<usize>>,
    }

    #[async_trait(?Send)]
    impl TracePlacement for QueueCancellationWorker {
        async fn execute_trace(&self, _plan: GraphTracePlan) -> Result<(), TraceError> {
            panic!("a command arriving after cancellation must never execute")
        }

        fn cancel_inflight(&self) -> Result<(), TraceError> {
            self.cancellations.set(self.cancellations.get() + 1);
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl TracePlacement for RecordingWorker {
        async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
            self.placements
                .lock()
                .unwrap()
                .push((self.worker_id, plan.trace.id));
            Ok(())
        }

        fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
            self.prefill_limits
                .lock()
                .unwrap()
                .push((self.worker_id, limit));
            Ok(())
        }
    }

    fn plan(id: &str) -> GraphTracePlan {
        GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        }
    }

    #[test]
    fn thread_per_core_places_whole_traces_round_robin() {
        let placements = Arc::new(Mutex::new(Vec::new()));
        let factory = Arc::new(RecordingFactory {
            placements: placements.clone(),
            prefill_limits: Arc::new(Mutex::new(Vec::new())),
        });
        let backend = ThreadPerCoreTracePlacement::new(2, factory).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(LocalSet::new().run_until(async {
            for id in ["a", "b", "c", "d"] {
                backend.execute_trace(plan(id)).await.unwrap();
            }
        }));
        drop(backend);

        assert_eq!(
            *placements.lock().unwrap(),
            vec![
                (0, "a".into()),
                (1, "b".into()),
                (0, "c".into()),
                (1, "d".into())
            ]
        );
    }

    #[test]
    fn global_prefill_limit_shards_exactly_and_routes_only_to_positive_workers() {
        let placements = Arc::new(Mutex::new(Vec::new()));
        let prefill_limits = Arc::new(Mutex::new(Vec::new()));
        let factory = Arc::new(RecordingFactory {
            placements: placements.clone(),
            prefill_limits: prefill_limits.clone(),
        });
        let backend = ThreadPerCoreTracePlacement::new(3, factory).unwrap();
        backend.set_prefill_limit(2).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(LocalSet::new().run_until(async {
            for id in ["a", "b", "c", "d"] {
                backend.execute_trace(plan(id)).await.unwrap();
            }
            backend.set_prefill_limit(1).unwrap();
            backend.execute_trace(plan("e")).await.unwrap();
            backend.execute_trace(plan("f")).await.unwrap();
        }));
        drop(backend);

        let mut observed = prefill_limits.lock().unwrap().clone();
        observed.sort_unstable();
        assert_eq!(
            observed,
            vec![(0, 1), (0, 1), (1, 0), (1, 1), (2, 0), (2, 0)]
        );
        assert_eq!(
            *placements.lock().unwrap(),
            vec![
                (0, "a".into()),
                (1, "b".into()),
                (0, "c".into()),
                (1, "d".into()),
                (0, "e".into()),
                (0, "f".into()),
            ]
        );
    }

    #[test]
    fn public_placement_rejects_a_zero_global_prefill_limit() {
        let prefill_limits = Arc::new(Mutex::new(Vec::new()));
        let backend = ThreadPerCoreTracePlacement::new(
            2,
            Arc::new(RecordingFactory {
                placements: Arc::new(Mutex::new(Vec::new())),
                prefill_limits: prefill_limits.clone(),
            }),
        )
        .unwrap();

        let error = backend.set_prefill_limit(0).unwrap_err();
        drop(backend);

        assert_eq!(
            error,
            TraceError::Other("graph prefill limit must be positive".into())
        );
        assert!(prefill_limits.lock().unwrap().is_empty());
    }

    #[test]
    fn cancellation_fanout_reaches_all_workers_after_one_rejects_it() {
        let cancellations = Arc::new(Mutex::new(Vec::new()));
        let backend = ThreadPerCoreTracePlacement::new(
            3,
            Arc::new(FanoutFactory {
                cancellations: cancellations.clone(),
                failing_worker: 0,
            }),
        )
        .unwrap();

        let error = backend.cancel_inflight().unwrap_err().to_string();
        drop(backend);

        let mut observed = cancellations.lock().unwrap().clone();
        observed.sort_unstable();
        assert_eq!(observed, vec![0, 1, 2]);
        assert!(error.contains("worker 0") && error.contains("intentional"));
    }

    #[test]
    fn active_trace_reaches_terminal_cleanup_after_graceful_cancellation() {
        let state = Arc::new(CancellableWorkerState::default());
        let backend = Rc::new(
            ThreadPerCoreTracePlacement::new(
                1,
                Arc::new(CancellableFactory {
                    state: state.clone(),
                }),
            )
            .unwrap(),
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let backend_for_run = backend.clone();
        let state_for_run = state.clone();
        let error = runtime.block_on(LocalSet::new().run_until(async move {
            let executing = backend_for_run.clone();
            let task =
                tokio::task::spawn_local(
                    async move { executing.execute_trace(plan("active")).await },
                );
            while !state_for_run.started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
            backend_for_run.cancel_inflight().unwrap();
            task.await.unwrap().unwrap_err()
        }));
        drop(backend);

        assert!(matches!(error, TraceError::Cancelled(_)));
        assert_eq!(state.terminal_cleanups.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn worker_rejects_commands_arriving_after_cancellation_is_latched() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(LocalSet::new().run_until(async {
            let cancellations = Rc::new(Cell::new(0));
            let backend: Rc<dyn TracePlacement> = Rc::new(QueueCancellationWorker {
                cancellations: cancellations.clone(),
            });
            let (command_tx, command_rx) = mpsc::channel(1);
            let (control_tx, control_rx) = mpsc::unbounded_channel();
            let worker = tokio::task::spawn_local(worker_loop(backend, command_rx, control_rx));

            let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
            control_tx
                .send(WorkerControl::CancelInflight { ack: ack_tx })
                .unwrap();
            loop {
                match ack_rx.try_recv() {
                    Ok(result) => {
                        result.unwrap();
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        tokio::task::yield_now().await;
                    }
                    Err(error) => panic!("cancellation acknowledgment failed: {error}"),
                }
            }

            let (result_tx, result_rx) = oneshot::channel();
            command_tx
                .send(WorkerCommand::Execute {
                    plan: plan("late"),
                    result: result_tx,
                })
                .await
                .unwrap();
            let error = result_rx.await.unwrap().unwrap_err();
            assert!(matches!(error, TraceError::Cancelled(_)));
            assert_eq!(cancellations.get(), 1);

            drop(command_tx);
            drop(control_tx);
            worker.await.unwrap();
        }));
    }

    #[test]
    fn worker_drains_execute_already_queued_when_cancellation_arrives() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(LocalSet::new().run_until(async {
            let cancellations = Rc::new(Cell::new(0));
            let backend: Rc<dyn TracePlacement> = Rc::new(QueueCancellationWorker {
                cancellations: cancellations.clone(),
            });
            let (command_tx, command_rx) = mpsc::channel(1);
            let (control_tx, control_rx) = mpsc::unbounded_channel();
            let (result_tx, result_rx) = oneshot::channel();
            command_tx
                .send(WorkerCommand::Execute {
                    plan: plan("queued"),
                    result: result_tx,
                })
                .await
                .unwrap();
            let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
            control_tx
                .send(WorkerControl::CancelInflight { ack: ack_tx })
                .unwrap();

            let worker = tokio::task::spawn_local(worker_loop(backend, command_rx, control_rx));
            let error = result_rx.await.unwrap().unwrap_err();
            assert!(matches!(error, TraceError::Cancelled(_)));
            assert!(ack_rx.recv().unwrap().is_ok());
            assert_eq!(cancellations.get(), 1);

            drop(command_tx);
            drop(control_tx);
            worker.await.unwrap();
        }));
    }

    #[test]
    fn bounded_placement_rejects_a_zero_capacity() {
        let placements = Arc::new(Mutex::new(Vec::new()));
        let factory = Arc::new(RecordingFactory {
            placements,
            prefill_limits: Arc::new(Mutex::new(Vec::new())),
        });
        let error = ThreadPerCoreTracePlacement::new_with_queue_capacity(1, 0, factory)
            .err()
            .expect("zero capacity must fail closed");
        assert_eq!(
            error,
            GraphPlacementError("graph placement queue capacity must be positive".into())
        );
    }
}
