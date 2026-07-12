// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-trace placement implementations.
//!
//! The native pool assigns complete traces round-robin to OS threads. Each
//! worker owns a current-thread Tokio runtime, a `LocalSet`, and its backend,
//! preserving the crate's lock-free thread-per-core execution model. A future
//! cross-node implementation can implement the same execution trait using ZMQ
//! or another wire without teaching the coordinator about that transport.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{self, Display};
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinSet, LocalSet};

use crate::errors::TraceError;
use crate::execution::GraphTraceExecutionBackend;
use crate::model::GraphTracePlan;

/// Default number of complete trace commands buffered per placement worker.
pub const DEFAULT_GRAPH_WORKER_QUEUE_CAPACITY: usize = 256;

/// Builds one worker-local backend after entering its owning OS thread.
///
/// Factory state must be shareable, but the returned backend deliberately need
/// not be `Send` or `Sync`: it never leaves that worker's `LocalSet`.
pub trait GraphTraceExecutionBackendFactory: Send + Sync {
    /// Construct the backend owned by `worker_id`.
    fn create_backend(
        &self,
        worker_id: usize,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError>;
}

/// Native thread-per-core whole-trace placement backend.
pub struct ThreadPerCoreGraphTraceExecutionBackend {
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    controls: Vec<mpsc::UnboundedSender<WorkerControl>>,
    next_worker: Cell<usize>,
    threads: RefCell<Vec<JoinHandle<()>>>,
}

impl ThreadPerCoreGraphTraceExecutionBackend {
    /// Start `worker_count` current-thread runtimes and build one backend each.
    pub fn new(
        worker_count: usize,
        factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    ) -> Result<Self, GraphPlacementError> {
        Self::new_with_queue_capacity(worker_count, DEFAULT_GRAPH_WORKER_QUEUE_CAPACITY, factory)
    }

    /// Start workers with an explicit per-worker whole-trace queue capacity.
    pub fn new_with_queue_capacity(
        worker_count: usize,
        queue_capacity: usize,
        factory: Arc<dyn GraphTraceExecutionBackendFactory>,
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
            threads: RefCell::new(threads),
        })
    }

    /// Number of native placement workers.
    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }
}

#[async_trait(?Send)]
impl GraphTraceExecutionBackend for ThreadPerCoreGraphTraceExecutionBackend {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        let worker_id = self.next_worker.get();
        self.next_worker.set((worker_id + 1) % self.senders.len());
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
        self.broadcast_control(|ack| WorkerControl::CancelInflight { ack })
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
        if limit == 0 {
            return Err(TraceError::Other(
                "graph prefill limit must be positive".into(),
            ));
        }
        self.broadcast_control(|ack| WorkerControl::SetPrefillLimit { limit, ack })
    }
}

impl ThreadPerCoreGraphTraceExecutionBackend {
    fn broadcast_control(
        &self,
        control: impl Fn(std::sync::mpsc::SyncSender<Result<(), String>>) -> WorkerControl,
    ) -> Result<(), TraceError> {
        for (worker_id, sender) in self.controls.iter().enumerate() {
            let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
            sender.send(control(ack_tx)).map_err(|_| {
                TraceError::Other(format!(
                    "graph placement worker {worker_id} is unavailable for control updates"
                ))
            })?;
            ack_rx
                .recv()
                .map_err(|_| {
                    TraceError::Other(format!(
                        "graph placement worker {worker_id} exited before acknowledging its control update"
                    ))
                })?
                .map_err(|error| {
                    TraceError::Other(format!(
                        "graph placement worker {worker_id} rejected its control update: {error}"
                    ))
                })?;
        }
        Ok(())
    }
}

impl Drop for ThreadPerCoreGraphTraceExecutionBackend {
    fn drop(&mut self) {
        self.senders.clear();
        self.controls.clear();
        for thread in self.threads.get_mut().drain(..) {
            let _ = thread.join();
        }
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
    factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    mut commands: mpsc::Receiver<WorkerCommand>,
    mut controls: mpsc::UnboundedReceiver<WorkerControl>,
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
        let backend = match factory.create_backend(worker_id) {
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

        let mut active = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                control = controls.recv() => match control {
                    Some(WorkerControl::CancelInflight { ack }) => {
                        active.abort_all();
                        let _ = ack.send(Ok(()));
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
    }));
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
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::model::{GraphRecord, TraceRecord};

    struct RecordingFactory {
        placements: Arc<Mutex<Vec<(usize, String)>>>,
        prefill_limits: Arc<Mutex<Vec<(usize, usize)>>>,
    }

    impl GraphTraceExecutionBackendFactory for RecordingFactory {
        fn create_backend(
            &self,
            worker_id: usize,
        ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
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

    #[async_trait(?Send)]
    impl GraphTraceExecutionBackend for RecordingWorker {
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
        let backend = ThreadPerCoreGraphTraceExecutionBackend::new(2, factory).unwrap();
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
    fn placement_broadcasts_prefill_control_to_every_worker() {
        let placements = Arc::new(Mutex::new(Vec::new()));
        let prefill_limits = Arc::new(Mutex::new(Vec::new()));
        let factory = Arc::new(RecordingFactory {
            placements,
            prefill_limits: prefill_limits.clone(),
        });
        let backend = ThreadPerCoreGraphTraceExecutionBackend::new(2, factory).unwrap();
        backend.set_prefill_limit(7).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(LocalSet::new().run_until(async {
            backend.execute_trace(plan("a")).await.unwrap();
            backend.execute_trace(plan("b")).await.unwrap();
        }));
        drop(backend);

        let mut observed = prefill_limits.lock().unwrap().clone();
        observed.sort_unstable();
        assert_eq!(observed, vec![(0, 7), (1, 7)]);
    }

    #[test]
    fn bounded_placement_rejects_a_zero_capacity() {
        let placements = Arc::new(Mutex::new(Vec::new()));
        let factory = Arc::new(RecordingFactory {
            placements,
            prefill_limits: Arc::new(Mutex::new(Vec::new())),
        });
        let error = ThreadPerCoreGraphTraceExecutionBackend::new_with_queue_capacity(1, 0, factory)
            .err()
            .expect("zero capacity must fail closed");
        assert_eq!(
            error,
            GraphPlacementError("graph placement queue capacity must be positive".into())
        );
    }
}
