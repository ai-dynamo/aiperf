// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated bounded worker for synchronous vendor GPU APIs.

use std::rc::Rc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::{self, JoinHandle};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::clock::Clock;
use crate::gpu_telemetry::model::{GpuScrape, GpuTelemetryRecord};
use crate::gpu_telemetry::source::{GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};

const CHANNEL_CAPACITY: usize = 1;
const WORKER_THREAD_NAME: &str = "aiperf-gpu-vendor";

/// Synchronous vendor API owned and invoked exclusively by one worker thread.
pub(super) trait VendorWorker: Send + 'static {
    fn initialize(&mut self) -> Result<(), GpuTelemetryError>;
    fn scrape(&mut self, timestamp_ns: i64) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError>;
    fn shutdown(&mut self) -> Result<(), GpuTelemetryError>;
}

type WorkerReply<T> = SyncSender<Result<T, GpuTelemetryError>>;

enum WorkerCommand {
    Scrape {
        timestamp_ns: i64,
        reply: WorkerReply<Vec<GpuTelemetryRecord>>,
    },
    Shutdown {
        reply: WorkerReply<()>,
    },
}

struct WorkerState {
    commands: Option<SyncSender<WorkerCommand>>,
    thread: Option<JoinHandle<()>>,
    is_shutdown: bool,
}

/// GPU telemetry source that confines a synchronous vendor API to one OS thread.
pub(super) struct VendorWorkerSource {
    clock: Rc<dyn Clock>,
    endpoint_url: String,
    state: Mutex<WorkerState>,
}

impl VendorWorkerSource {
    /// Starts one vendor worker and returns only after initialization succeeds.
    pub(super) async fn spawn(
        clock: Rc<dyn Clock>,
        endpoint_url: impl Into<String>,
        worker: Box<dyn VendorWorker>,
    ) -> Result<Self, GpuTelemetryError> {
        let endpoint_url = endpoint_url.into();
        if endpoint_url.trim().is_empty() {
            return Err(GpuTelemetryError::Protocol(
                "vendor source endpoint_url must be non-empty".to_string(),
            ));
        }

        let (commands, command_receiver) = sync_channel(CHANNEL_CAPACITY);
        let (startup_reply, startup_receiver) = sync_channel(CHANNEL_CAPACITY);
        let worker_thread = thread::Builder::new()
            .name(WORKER_THREAD_NAME.to_string())
            .spawn(move || run_worker(worker, command_receiver, startup_reply))
            .map_err(|error| {
                GpuTelemetryError::Worker(format!("spawning vendor worker thread: {error}"))
            })?;

        let startup = receive_reply(startup_receiver, "initialization").await;
        if let Err(error) = startup {
            let join_result = join_worker(worker_thread).await;
            return match join_result {
                Ok(()) => Err(error),
                Err(join_error) => Err(join_error),
            };
        }

        Ok(Self {
            clock,
            endpoint_url,
            state: Mutex::new(WorkerState {
                commands: Some(commands),
                thread: Some(worker_thread),
                is_shutdown: false,
            }),
        })
    }

    async fn scrape_records(
        &self,
        timestamp_ns: i64,
    ) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
        let state = self.state.lock().await;
        if state.is_shutdown {
            return Err(GpuTelemetryError::Worker(
                "vendor scrape attempted after shutdown".to_string(),
            ));
        }
        let commands = state.commands.as_ref().ok_or_else(|| {
            GpuTelemetryError::Worker("vendor command channel is unavailable".to_string())
        })?;
        let (reply, receiver) = sync_channel(CHANNEL_CAPACITY);
        try_send_command(
            commands,
            WorkerCommand::Scrape {
                timestamp_ns,
                reply,
            },
        )?;
        receive_reply(receiver, "scrape").await
    }
}

#[async_trait(?Send)]
impl GpuTelemetrySource for VendorWorkerSource {
    fn endpoint_url(&self) -> &str {
        &self.endpoint_url
    }

    async fn scrape(&self, _mode: GpuScrapeMode) -> Result<Option<GpuScrape>, GpuTelemetryError> {
        let timestamp_ns = self.clock.now_ns();
        let records = self.scrape_records(timestamp_ns).await?;
        for record in &records {
            if record.timestamp_ns != timestamp_ns {
                return Err(GpuTelemetryError::Protocol(format!(
                    "vendor record timestamp {} does not match scrape timestamp {timestamp_ns}",
                    record.timestamp_ns
                )));
            }
            if record.endpoint_url != self.endpoint_url {
                return Err(GpuTelemetryError::Protocol(format!(
                    "vendor record endpoint {:?} does not match source {:?}",
                    record.endpoint_url, self.endpoint_url
                )));
            }
        }
        Ok(Some(GpuScrape {
            timestamp_ns,
            endpoint_url: self.endpoint_url.clone(),
            records,
        }))
    }

    async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        let mut state = self.state.lock().await;
        if state.is_shutdown {
            return Ok(());
        }
        let commands = state.commands.as_ref().ok_or_else(|| {
            GpuTelemetryError::Worker("vendor command channel is unavailable".to_string())
        })?;
        let (reply, receiver) = sync_channel(CHANNEL_CAPACITY);
        match commands.try_send(WorkerCommand::Shutdown { reply }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(GpuTelemetryError::Worker(
                    "vendor command channel is full".to_string(),
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                state.is_shutdown = true;
                state.commands.take();
                let worker_thread = state.thread.take();
                if let Some(worker_thread) = worker_thread {
                    let _ = join_worker(worker_thread).await;
                }
                return Err(GpuTelemetryError::Worker(
                    "vendor command channel disconnected".to_string(),
                ));
            }
        }
        state.is_shutdown = true;
        state.commands.take();
        let worker_thread = state.thread.take().ok_or_else(|| {
            GpuTelemetryError::Worker("vendor worker thread is unavailable".to_string())
        })?;

        let shutdown_result = receive_reply(receiver, "shutdown").await;
        let join_result = join_worker(worker_thread).await;
        shutdown_result.and(join_result)
    }
}

fn run_worker(
    mut worker: Box<dyn VendorWorker>,
    commands: Receiver<WorkerCommand>,
    startup_reply: WorkerReply<()>,
) {
    let initialization = worker.initialize();
    let is_initialized = initialization.is_ok();
    if send_worker_reply(&startup_reply, initialization, "initialization").is_err()
        || !is_initialized
    {
        return;
    }

    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Scrape {
                timestamp_ns,
                reply,
            } => {
                let result = worker.scrape(timestamp_ns);
                if send_worker_reply(&reply, result, "scrape").is_err() {
                    let _ = worker.shutdown();
                    return;
                }
            }
            WorkerCommand::Shutdown { reply } => {
                let result = worker.shutdown();
                let _ = send_worker_reply(&reply, result, "shutdown");
                return;
            }
        }
    }
    let _ = worker.shutdown();
}

fn try_send_command(
    commands: &SyncSender<WorkerCommand>,
    command: WorkerCommand,
) -> Result<(), GpuTelemetryError> {
    commands.try_send(command).map_err(|error| match error {
        TrySendError::Full(_) => {
            GpuTelemetryError::Worker("vendor command channel is full".to_string())
        }
        TrySendError::Disconnected(_) => {
            GpuTelemetryError::Worker("vendor command channel disconnected".to_string())
        }
    })
}

fn send_worker_reply<T>(
    reply: &WorkerReply<T>,
    result: Result<T, GpuTelemetryError>,
    operation: &str,
) -> Result<(), GpuTelemetryError> {
    reply.try_send(result).map_err(|error| match error {
        TrySendError::Full(_) => {
            GpuTelemetryError::Worker(format!("vendor {operation} reply channel is full"))
        }
        TrySendError::Disconnected(_) => {
            GpuTelemetryError::Worker(format!("vendor {operation} reply channel disconnected"))
        }
    })
}

async fn receive_reply<T: Send + 'static>(
    receiver: Receiver<Result<T, GpuTelemetryError>>,
    operation: &'static str,
) -> Result<T, GpuTelemetryError> {
    tokio::task::spawn_blocking(move || {
        receiver.recv().map_err(|_| {
            GpuTelemetryError::Worker(format!("vendor worker exited before the {operation} reply"))
        })?
    })
    .await
    .map_err(|error| {
        GpuTelemetryError::Worker(format!("waiting for vendor {operation} reply: {error}"))
    })?
}

async fn join_worker(worker_thread: JoinHandle<()>) -> Result<(), GpuTelemetryError> {
    tokio::task::spawn_blocking(move || worker_thread.join())
        .await
        .map_err(|error| GpuTelemetryError::Worker(format!("joining vendor worker: {error}")))?
        .map_err(|_| GpuTelemetryError::Worker("vendor worker thread panicked".to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::thread::ThreadId;

    use crate::clock::SimClock;
    use crate::gpu_telemetry::model::{GpuMetadata, UNKNOWN_GPU_TELEMETRY_PLATFORM};

    use super::*;

    const ENDPOINT: &str = "vendor://localhost";

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Action {
        Initialize(ThreadId),
        Scrape(i64, ThreadId),
        Shutdown(ThreadId),
    }

    struct FakeVendor {
        actions: Arc<StdMutex<Vec<Action>>>,
        initialize_error: Option<GpuTelemetryError>,
        records: Vec<GpuTelemetryRecord>,
        panic_on_scrape: bool,
    }

    impl FakeVendor {
        fn healthy(actions: Arc<StdMutex<Vec<Action>>>) -> Self {
            Self {
                actions,
                initialize_error: None,
                records: Vec::new(),
                panic_on_scrape: false,
            }
        }

        fn record_action(&self, action: Action) {
            self.actions.lock().unwrap().push(action);
        }
    }

    impl VendorWorker for FakeVendor {
        fn initialize(&mut self) -> Result<(), GpuTelemetryError> {
            self.record_action(Action::Initialize(thread::current().id()));
            if let Some(error) = &self.initialize_error {
                return Err(error.clone());
            }
            Ok(())
        }

        fn scrape(
            &mut self,
            timestamp_ns: i64,
        ) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
            self.record_action(Action::Scrape(timestamp_ns, thread::current().id()));
            assert!(!self.panic_on_scrape, "injected worker panic");
            Ok(self.records.clone())
        }

        fn shutdown(&mut self) -> Result<(), GpuTelemetryError> {
            self.record_action(Action::Shutdown(thread::current().id()));
            Ok(())
        }
    }

    fn clock() -> Rc<dyn Clock> {
        Rc::new(SimClock::new())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vendor_calls_run_on_owned_thread_and_empty_scrapes_are_some() {
        let caller = thread::current().id();
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            Box::new(FakeVendor::healthy(actions.clone())),
        )
        .await
        .unwrap();

        let scrape = source
            .scrape(GpuScrapeMode::Continuous)
            .await
            .unwrap()
            .unwrap();
        assert!(scrape.records.is_empty());
        source.shutdown().await.unwrap();

        let actions = actions.lock().unwrap();
        assert_eq!(actions.len(), 3);
        assert!(actions.iter().all(|action| match action {
            Action::Initialize(thread) | Action::Scrape(_, thread) | Action::Shutdown(thread) => {
                *thread != caller
            }
        }));
        let worker_threads = actions
            .iter()
            .map(|action| match action {
                Action::Initialize(thread)
                | Action::Scrape(_, thread)
                | Action::Shutdown(thread) => *thread,
            })
            .collect::<Vec<_>>();
        assert!(worker_threads.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_startup_is_returned_before_source_construction() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let mut worker = FakeVendor::healthy(actions.clone());
        worker.initialize_error = Some(GpuTelemetryError::Worker("initialize failed".to_string()));
        let result = VendorWorkerSource::spawn(clock(), ENDPOINT, Box::new(worker)).await;
        assert!(matches!(
            result,
            Err(GpuTelemetryError::Worker(message)) if message == "initialize failed"
        ));
        assert_eq!(actions.lock().unwrap().len(), 1);
    }

    #[test]
    fn full_command_and_reply_channels_are_typed_errors() {
        let (commands, _receiver) = sync_channel(CHANNEL_CAPACITY);
        let (first_reply, _first_receiver) = sync_channel(CHANNEL_CAPACITY);
        commands
            .try_send(WorkerCommand::Shutdown { reply: first_reply })
            .unwrap();
        let (second_reply, _second_receiver) = sync_channel(CHANNEL_CAPACITY);
        let error = try_send_command(
            &commands,
            WorkerCommand::Shutdown {
                reply: second_reply,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpuTelemetryError::Worker(message) if message.contains("command channel is full")
        ));

        let (reply, _receiver) = sync_channel(CHANNEL_CAPACITY);
        reply.try_send(Ok(())).unwrap();
        let error = send_worker_reply(&reply, Ok(()), "scrape").unwrap_err();
        assert!(matches!(
            error,
            GpuTelemetryError::Worker(message) if message.contains("scrape reply channel is full")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn worker_exit_before_reply_is_typed() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let mut worker = FakeVendor::healthy(actions);
        worker.panic_on_scrape = true;
        let source = VendorWorkerSource::spawn(clock(), ENDPOINT, Box::new(worker))
            .await
            .unwrap();
        let error = source.scrape(GpuScrapeMode::Continuous).await.unwrap_err();
        assert!(matches!(
            error,
            GpuTelemetryError::Worker(message) if message.contains("exited before the scrape reply")
        ));
        assert!(source.shutdown().await.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_is_idempotent_and_ordered_after_boundary() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            Box::new(FakeVendor::healthy(actions.clone())),
        )
        .await
        .unwrap();
        source.scrape(GpuScrapeMode::Continuous).await.unwrap();
        source.scrape(GpuScrapeMode::Boundary).await.unwrap();
        source.shutdown().await.unwrap();
        source.shutdown().await.unwrap();

        let actions = actions.lock().unwrap();
        assert!(matches!(
            actions.as_slice(),
            [
                Action::Initialize(_),
                Action::Scrape(0, _),
                Action::Scrape(0, _),
                Action::Shutdown(_),
            ]
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn record_identity_and_timestamp_are_enforced() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let mut worker = FakeVendor::healthy(actions);
        worker.records.push(GpuTelemetryRecord {
            timestamp_ns: 1,
            endpoint_url: "other://localhost".to_string(),
            metadata: GpuMetadata {
                gpu_index: 0,
                gpu_uuid: "gpu-0".to_string(),
                gpu_model_name: "fake".to_string(),
                pci_bus_id: None,
                device: None,
                hostname: None,
                namespace: None,
                pod_name: None,
                platform: UNKNOWN_GPU_TELEMETRY_PLATFORM.to_string(),
            },
            metrics: BTreeMap::new(),
        });
        let source = VendorWorkerSource::spawn(clock(), ENDPOINT, Box::new(worker))
            .await
            .unwrap();
        let error = source.scrape(GpuScrapeMode::Boundary).await.unwrap_err();
        assert!(matches!(error, GpuTelemetryError::Protocol(_)));
        source.shutdown().await.unwrap();
    }
}
