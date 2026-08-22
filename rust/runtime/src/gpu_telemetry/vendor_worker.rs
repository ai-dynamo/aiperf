// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated bounded worker for synchronous vendor GPU APIs.

use std::future::Future;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{
    Notify,
    mpsc::{OwnedPermit, Receiver as TokioReceiver, Sender, channel},
};

use crate::clock::Clock;
use crate::gpu_telemetry::model::{GpuScrape, GpuTelemetryRecord};
use crate::gpu_telemetry::source::{GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};

const CHANNEL_CAPACITY: usize = 1;
const REAPER_CHANNEL_CAPACITY: usize = 64;
const REAPER_POLL_INTERVAL: Duration = Duration::from_millis(1);
const DEFAULT_OPERATION_TIMEOUT_NS: i64 = 10_000_000_000;
const WORKER_THREAD_NAME: &str = "aiperf-gpu-vendor";
const REAPER_THREAD_NAME: &str = "aiperf-gpu-reaper";

/// Synchronous vendor API owned and invoked exclusively by one worker thread.
pub(super) trait VendorWorker: Send + 'static {
    fn initialize(&mut self) -> Result<(), GpuTelemetryError>;
    fn scrape(&mut self, timestamp_ns: i64) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError>;
    fn shutdown(&mut self) -> Result<(), GpuTelemetryError>;
}

type WorkerReply<T> = Sender<Result<T, GpuTelemetryError>>;
type WorkerResult = Result<(), GpuTelemetryError>;

enum WorkerCommand {
    Scrape {
        timestamp_ns: i64,
        reply: WorkerReply<Vec<GpuTelemetryRecord>>,
    },
    Shutdown,
}

#[derive(Default)]
struct CompletionState {
    result: Option<WorkerResult>,
    is_abandoned: bool,
    is_observed: bool,
    is_abandoned_error_reported: bool,
}

#[derive(Default)]
struct WorkerCompletion {
    state: StdMutex<CompletionState>,
    ready: Notify,
}

impl WorkerCompletion {
    fn finish(&self, result: WorkerResult) {
        let mut state = lock_unpoisoned(&self.state);
        state.result = Some(result);
        let abandoned_error = abandoned_error_to_report(&mut state);
        self.ready.notify_waiters();
        drop(state);
        if let Some(error) = abandoned_error {
            tracing::error!(error = %error, component = "gpu_vendor_worker", "dropped GPU vendor source cleanup failed");
        }
    }

    async fn wait(&self) -> WorkerResult {
        loop {
            let notified = self.ready.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = &lock_unpoisoned(&self.state).result {
                return result.clone();
            }
            notified.await;
        }
    }

    fn observe(&self) {
        lock_unpoisoned(&self.state).is_observed = true;
    }

    fn abandon(&self) {
        let mut state = lock_unpoisoned(&self.state);
        state.is_abandoned = true;
        let abandoned_error = abandoned_error_to_report(&mut state);
        drop(state);
        if let Some(error) = abandoned_error {
            tracing::error!(error = %error, component = "gpu_vendor_worker", "dropped GPU vendor source cleanup failed");
        }
    }
}

fn abandoned_error_to_report(state: &mut CompletionState) -> Option<GpuTelemetryError> {
    if !state.is_abandoned || state.is_observed || state.is_abandoned_error_reported {
        return None;
    }
    let error = state.result.as_ref()?.as_ref().err()?.clone();
    state.is_abandoned_error_reported = true;
    Some(error)
}

fn lock_unpoisoned<T>(mutex: &StdMutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct ReapRequest {
    thread: JoinHandle<WorkerResult>,
    completion: Arc<WorkerCompletion>,
}

struct WorkerReaper {
    sender: Sender<ReapRequest>,
    _thread: JoinHandle<()>,
}

static WORKER_REAPER: OnceLock<Result<WorkerReaper, GpuTelemetryError>> = OnceLock::new();

fn worker_reaper() -> Result<&'static WorkerReaper, GpuTelemetryError> {
    WORKER_REAPER
        .get_or_init(|| {
            let (sender, receiver) = channel(REAPER_CHANNEL_CAPACITY);
            let reaper_thread = thread::Builder::new()
                .name(REAPER_THREAD_NAME.to_string())
                .spawn(move || run_reaper(receiver))
                .map_err(|error| {
                    GpuTelemetryError::Worker(format!(
                        "spawning vendor worker reaper thread: {error}"
                    ))
                })?;
            Ok(WorkerReaper {
                sender,
                _thread: reaper_thread,
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

async fn reserve_reaper_slot() -> Result<OwnedPermit<ReapRequest>, GpuTelemetryError> {
    worker_reaper()?
        .sender
        .clone()
        .reserve_owned()
        .await
        .map_err(|_| GpuTelemetryError::Worker("vendor worker reaper exited".to_string()))
}

fn run_reaper(mut receiver: tokio::sync::mpsc::Receiver<ReapRequest>) {
    let mut workers: Vec<ReapRequest> = Vec::new();
    loop {
        if workers.is_empty() {
            match receiver.blocking_recv() {
                Some(worker) => workers.push(worker),
                None => return,
            }
        }
        while let Ok(worker) = receiver.try_recv() {
            workers.push(worker);
        }
        let mut index = 0;
        while index < workers.len() {
            if workers[index].thread.is_finished() {
                let worker = workers.swap_remove(index);
                let result = worker.thread.join().unwrap_or_else(|_| {
                    Err(GpuTelemetryError::Worker(
                        "vendor worker thread panicked".to_string(),
                    ))
                });
                worker.completion.finish(result);
            } else {
                index += 1;
            }
        }
        if !workers.is_empty() {
            thread::sleep(REAPER_POLL_INTERVAL);
        }
    }
}

enum WorkerLifecycle {
    Running(SyncSender<WorkerCommand>),
    ShutdownRequested,
}

struct WorkerState {
    lifecycle: WorkerLifecycle,
    completion: Arc<WorkerCompletion>,
}

/// GPU telemetry source that confines a synchronous vendor API to one OS thread.
pub(super) struct VendorWorkerSource {
    clock: Rc<dyn Clock>,
    endpoint_url: String,
    request_timeout_ns: i64,
    state: StdMutex<WorkerState>,
}

impl VendorWorkerSource {
    /// Starts one vendor worker with the default operation deadline.
    pub(super) async fn spawn<F>(
        clock: Rc<dyn Clock>,
        endpoint_url: impl Into<String>,
        factory: F,
    ) -> Result<Self, GpuTelemetryError>
    where
        F: FnOnce() -> Result<Box<dyn VendorWorker>, GpuTelemetryError> + Send + 'static,
    {
        Self::spawn_with_timeout(clock, endpoint_url, DEFAULT_OPERATION_TIMEOUT_NS, factory).await
    }

    /// Starts one vendor worker and bounds initialization by the supplied clock.
    pub(super) async fn spawn_with_timeout<F>(
        clock: Rc<dyn Clock>,
        endpoint_url: impl Into<String>,
        request_timeout_ns: i64,
        factory: F,
    ) -> Result<Self, GpuTelemetryError>
    where
        F: FnOnce() -> Result<Box<dyn VendorWorker>, GpuTelemetryError> + Send + 'static,
    {
        if request_timeout_ns <= 0 {
            return Err(GpuTelemetryError::Protocol(
                "vendor source request_timeout_ns must be positive".to_string(),
            ));
        }
        match wait_with_timeout(
            clock.clone(),
            request_timeout_ns,
            Self::spawn_inner(clock, endpoint_url.into(), request_timeout_ns, factory),
        )
        .await
        {
            DeadlineResult::Ready(result) => result,
            DeadlineResult::TimedOut => Err(timeout_error("initialization", request_timeout_ns)),
        }
    }

    async fn spawn_inner<F>(
        clock: Rc<dyn Clock>,
        endpoint_url: String,
        request_timeout_ns: i64,
        factory: F,
    ) -> Result<Self, GpuTelemetryError>
    where
        F: FnOnce() -> Result<Box<dyn VendorWorker>, GpuTelemetryError> + Send + 'static,
    {
        if endpoint_url.trim().is_empty() {
            return Err(GpuTelemetryError::Protocol(
                "vendor source endpoint_url must be non-empty".to_string(),
            ));
        }

        // Reserve cleanup ownership before the vendor thread exists. Once spawn
        // succeeds, the OS JoinHandle is handed to the reaper without an await.
        let reaper_slot = reserve_reaper_slot().await?;
        let (commands, command_receiver) = sync_channel(CHANNEL_CAPACITY);
        let (startup_reply, startup_receiver) = channel(CHANNEL_CAPACITY);
        let worker_thread = thread::Builder::new()
            .name(WORKER_THREAD_NAME.to_string())
            .spawn(move || run_worker(factory, command_receiver, startup_reply))
            .map_err(|error| {
                GpuTelemetryError::Worker(format!("spawning vendor worker thread: {error}"))
            })?;
        let completion = Arc::new(WorkerCompletion::default());
        reaper_slot.send(ReapRequest {
            thread: worker_thread,
            completion: completion.clone(),
        });

        let source = Self {
            clock,
            endpoint_url,
            request_timeout_ns,
            state: StdMutex::new(WorkerState {
                lifecycle: WorkerLifecycle::Running(commands),
                completion,
            }),
        };
        if let Err(startup_error) = receive_reply(startup_receiver, "initialization").await {
            let completion = source.request_shutdown_after_startup_failure();
            let worker_result = wait_for_completion(completion).await;
            return Err(merge_failures(
                startup_error,
                worker_result.err(),
                "vendor initialization",
            ));
        }
        Ok(source)
    }

    fn request_shutdown_after_startup_failure(&self) -> Arc<WorkerCompletion> {
        let mut state = lock_unpoisoned(&self.state);
        if let WorkerLifecycle::Running(commands) = &state.lifecycle {
            let _ = commands.try_send(WorkerCommand::Shutdown);
        }
        state.lifecycle = WorkerLifecycle::ShutdownRequested;
        state.completion.clone()
    }

    async fn scrape_records(
        &self,
        timestamp_ns: i64,
    ) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
        let commands = {
            let state = lock_unpoisoned(&self.state);
            match &state.lifecycle {
                WorkerLifecycle::Running(commands) => commands.clone(),
                WorkerLifecycle::ShutdownRequested => {
                    return Err(GpuTelemetryError::Worker(
                        "vendor scrape attempted after shutdown".to_string(),
                    ));
                }
            }
        };
        let (reply, receiver) = channel(CHANNEL_CAPACITY);
        try_send_command(
            &commands,
            WorkerCommand::Scrape {
                timestamp_ns,
                reply,
            },
        )?;
        match wait_with_timeout(
            self.clock.clone(),
            self.request_timeout_ns,
            receive_reply(receiver, "scrape"),
        )
        .await
        {
            DeadlineResult::Ready(result) => result,
            DeadlineResult::TimedOut => {
                self.request_shutdown_after_startup_failure();
                Err(timeout_error("scrape", self.request_timeout_ns))
            }
        }
    }

    fn begin_shutdown(&self) -> Result<Arc<WorkerCompletion>, GpuTelemetryError> {
        let mut state = lock_unpoisoned(&self.state);
        if let WorkerLifecycle::Running(commands) = &state.lifecycle {
            match commands.try_send(WorkerCommand::Shutdown) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => {
                    state.lifecycle = WorkerLifecycle::ShutdownRequested;
                }
                Err(TrySendError::Full(_)) => {
                    return Err(GpuTelemetryError::Worker(
                        "vendor command channel is full".to_string(),
                    ));
                }
            }
        }
        Ok(state.completion.clone())
    }
}

impl Drop for VendorWorkerSource {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let WorkerLifecycle::Running(commands) = &state.lifecycle {
            // If a scrape already occupies the bounded slot, dropping our
            // sender still makes the worker shut itself down after that scrape.
            let _ = commands.try_send(WorkerCommand::Shutdown);
            state.lifecycle = WorkerLifecycle::ShutdownRequested;
        }
        state.completion.abandon();
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
        let completion = self.begin_shutdown()?;
        match wait_with_timeout(
            self.clock.clone(),
            self.request_timeout_ns,
            wait_for_completion(completion),
        )
        .await
        {
            DeadlineResult::Ready(result) => result,
            DeadlineResult::TimedOut => Err(timeout_error("shutdown", self.request_timeout_ns)),
        }
    }
}

fn run_worker<F>(
    factory: F,
    commands: Receiver<WorkerCommand>,
    startup_reply: WorkerReply<()>,
) -> WorkerResult
where
    F: FnOnce() -> Result<Box<dyn VendorWorker>, GpuTelemetryError>,
{
    let mut worker = match factory() {
        Ok(worker) => worker,
        Err(error) => {
            let reply_result =
                send_worker_reply(&startup_reply, Err(error.clone()), "factory initialization");
            return Err(merge_failures(
                error,
                reply_result.err(),
                "reporting vendor factory failure",
            ));
        }
    };

    if let Err(initialize_error) = worker.initialize() {
        let cleanup_error = worker.shutdown().err();
        let failure = merge_failures(
            initialize_error,
            cleanup_error,
            "vendor initialization cleanup",
        );
        let reply_result =
            send_worker_reply(&startup_reply, Err(failure.clone()), "initialization");
        return Err(merge_failures(
            failure,
            reply_result.err(),
            "reporting vendor initialization failure",
        ));
    }
    if let Err(reply_error) = send_worker_reply(&startup_reply, Ok(()), "initialization") {
        return Err(merge_failures(
            reply_error,
            worker.shutdown().err(),
            "vendor startup cancellation cleanup",
        ));
    }

    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Scrape {
                timestamp_ns,
                reply,
            } => {
                let result = worker.scrape(timestamp_ns);
                if let Err(reply_error) = send_worker_reply(&reply, result, "scrape") {
                    return Err(merge_failures(
                        reply_error,
                        worker.shutdown().err(),
                        "vendor scrape cancellation cleanup",
                    ));
                }
            }
            WorkerCommand::Shutdown => return worker.shutdown(),
        }
    }
    worker.shutdown()
}

fn merge_failures(
    primary: GpuTelemetryError,
    secondary: Option<GpuTelemetryError>,
    context: &str,
) -> GpuTelemetryError {
    match secondary {
        None => primary,
        Some(secondary) if secondary == primary => primary,
        Some(secondary) => GpuTelemetryError::Worker(format!(
            "{context} failed: {primary}; additional failure: {secondary}"
        )),
    }
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
        tokio::sync::mpsc::error::TrySendError::Full(_) => {
            GpuTelemetryError::Worker(format!("vendor {operation} reply channel is full"))
        }
        tokio::sync::mpsc::error::TrySendError::Closed(_) => {
            GpuTelemetryError::Worker(format!("vendor {operation} reply channel disconnected"))
        }
    })
}

enum DeadlineResult<T> {
    Ready(T),
    TimedOut,
}

async fn wait_with_timeout<T>(
    clock: Rc<dyn Clock>,
    request_timeout_ns: i64,
    future: impl Future<Output = T>,
) -> DeadlineResult<T> {
    tokio::select! {
        biased;
        result = future => DeadlineResult::Ready(result),
        () = clock.sleep(request_timeout_ns) => DeadlineResult::TimedOut,
    }
}

fn timeout_error(operation: &str, request_timeout_ns: i64) -> GpuTelemetryError {
    GpuTelemetryError::Worker(format!("vendor {operation} timed out after {request_timeout_ns}ns"))
}

async fn receive_reply<T>(
    mut receiver: TokioReceiver<Result<T, GpuTelemetryError>>,
    operation: &'static str,
) -> Result<T, GpuTelemetryError> {
    receiver.recv().await.ok_or_else(|| {
        GpuTelemetryError::Worker(format!("vendor worker exited before the {operation} reply"))
    })?
}

async fn wait_for_completion(completion: Arc<WorkerCompletion>) -> Result<(), GpuTelemetryError> {
    let result = completion.wait().await;
    completion.observe();
    result
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc::Receiver as StdReceiver;
    use std::thread::ThreadId;
    use std::time::Duration;

    use crate::clock::SimClock;
    use crate::gpu_telemetry::model::{GpuMetadata, UNKNOWN_GPU_TELEMETRY_PLATFORM};

    use super::*;

    const ENDPOINT: &str = "vendor://localhost";

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Action {
        Construct(String, ThreadId),
        Initialize(String, ThreadId),
        Scrape(i64, String, ThreadId),
        Shutdown(String, ThreadId),
        Drop(String, ThreadId),
    }

    struct FakeVendor {
        actions: Arc<StdMutex<Vec<Action>>>,
        initialize_error: Option<GpuTelemetryError>,
        shutdown_error: Option<GpuTelemetryError>,
        records: Vec<GpuTelemetryRecord>,
        panic_on_scrape: bool,
        initialize_started: Option<SyncSender<()>>,
        initialize_release: Option<Receiver<()>>,
        shutdown_started: Option<SyncSender<()>>,
        shutdown_release: Option<Receiver<()>>,
        drop_reply: Option<SyncSender<ThreadId>>,
    }

    impl FakeVendor {
        fn healthy(actions: Arc<StdMutex<Vec<Action>>>) -> Self {
            Self {
                actions,
                initialize_error: None,
                shutdown_error: None,
                records: Vec::new(),
                panic_on_scrape: false,
                initialize_started: None,
                initialize_release: None,
                shutdown_started: None,
                shutdown_release: None,
                drop_reply: None,
            }
        }

        fn record_action(&self, action: Action) {
            self.actions.lock().unwrap().push(action);
        }
    }

    impl VendorWorker for FakeVendor {
        fn initialize(&mut self) -> Result<(), GpuTelemetryError> {
            self.record_action(Action::Initialize(
                current_thread_name(),
                thread::current().id(),
            ));
            if let Some(started) = &self.initialize_started {
                let _ = started.try_send(());
            }
            if let Some(release) = &self.initialize_release {
                let _ = release.recv();
            }
            if let Some(error) = &self.initialize_error {
                return Err(error.clone());
            }
            Ok(())
        }

        fn scrape(
            &mut self,
            timestamp_ns: i64,
        ) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
            self.record_action(Action::Scrape(
                timestamp_ns,
                current_thread_name(),
                thread::current().id(),
            ));
            assert!(!self.panic_on_scrape, "injected worker panic");
            Ok(self.records.clone())
        }

        fn shutdown(&mut self) -> Result<(), GpuTelemetryError> {
            self.record_action(Action::Shutdown(
                current_thread_name(),
                thread::current().id(),
            ));
            if let Some(started) = &self.shutdown_started {
                let _ = started.try_send(());
            }
            if let Some(release) = &self.shutdown_release {
                let _ = release.recv();
            }
            match &self.shutdown_error {
                Some(error) => Err(error.clone()),
                None => Ok(()),
            }
        }
    }

    impl Drop for FakeVendor {
        fn drop(&mut self) {
            let thread_id = thread::current().id();
            self.record_action(Action::Drop(current_thread_name(), thread_id));
            if let Some(reply) = &self.drop_reply {
                let _ = reply.try_send(thread_id);
            }
        }
    }

    fn current_thread_name() -> String {
        thread::current().name().unwrap_or("unnamed").to_string()
    }

    fn clock() -> Rc<dyn Clock> {
        Rc::new(SimClock::new())
    }

    fn fake_factory(
        actions: Arc<StdMutex<Vec<Action>>>,
        configure: impl FnOnce(&mut FakeVendor) + Send + 'static,
    ) -> impl FnOnce() -> Result<Box<dyn VendorWorker>, GpuTelemetryError> + Send + 'static {
        move || {
            actions.lock().unwrap().push(Action::Construct(
                current_thread_name(),
                thread::current().id(),
            ));
            let mut worker = FakeVendor::healthy(actions);
            configure(&mut worker);
            Ok(Box::new(worker))
        }
    }

    async fn receive_drop(receiver: StdReceiver<ThreadId>) -> ThreadId {
        tokio::task::spawn_blocking(move || receiver.recv_timeout(Duration::from_secs(2)))
            .await
            .unwrap()
            .unwrap()
    }

    async fn receive_signal(receiver: StdReceiver<()>) {
        tokio::task::spawn_blocking(move || receiver.recv_timeout(Duration::from_secs(2)))
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vendor_lifecycle_runs_on_one_named_worker_and_empty_scrapes_are_some() {
        let caller = thread::current().id();
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source =
            VendorWorkerSource::spawn(clock(), ENDPOINT, fake_factory(actions.clone(), |_| {}))
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
        assert_eq!(actions.len(), 5);
        assert!(actions.iter().all(|action| match action {
            Action::Construct(name, thread)
            | Action::Initialize(name, thread)
            | Action::Scrape(_, name, thread)
            | Action::Shutdown(name, thread)
            | Action::Drop(name, thread) => name == WORKER_THREAD_NAME && *thread != caller,
        }));
        let worker_threads = actions
            .iter()
            .map(|action| match action {
                Action::Construct(_, thread)
                | Action::Initialize(_, thread)
                | Action::Scrape(_, _, thread)
                | Action::Shutdown(_, thread)
                | Action::Drop(_, thread) => *thread,
            })
            .collect::<Vec<_>>();
        assert!(worker_threads.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_initialization_runs_shutdown_and_drop_on_worker() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let result = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions.clone(), |worker| {
                worker.initialize_error =
                    Some(GpuTelemetryError::Worker("initialize failed".to_string()));
            }),
        )
        .await;
        assert!(matches!(
            result,
            Err(GpuTelemetryError::Worker(message)) if message == "initialize failed"
        ));
        assert!(matches!(
            actions.lock().unwrap().as_slice(),
            [
                Action::Construct(name_1, thread_1),
                Action::Initialize(name_2, thread_2),
                Action::Shutdown(name_3, thread_3),
                Action::Drop(name_4, thread_4),
            ] if [name_1, name_2, name_3, name_4]
                .iter()
                .all(|name| name.as_str() == WORKER_THREAD_NAME)
                && thread_1 == thread_2
                && thread_2 == thread_3
                && thread_3 == thread_4
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn initialization_timeout_returns_at_clock_deadline() {
        let clock = Rc::new(SimClock::new());
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let (started_reply, started_receiver) = sync_channel(CHANNEL_CAPACITY);
        let (release_reply, release_receiver) = sync_channel(CHANNEL_CAPACITY);
        let (drop_reply, drop_receiver) = sync_channel(CHANNEL_CAPACITY);
        let spawn = VendorWorkerSource::spawn_with_timeout(
            clock.clone(),
            ENDPOINT,
            1,
            fake_factory(actions.clone(), move |worker| {
                worker.initialize_started = Some(started_reply);
                worker.initialize_release = Some(release_receiver);
                worker.drop_reply = Some(drop_reply);
            }),
        );
        tokio::pin!(spawn);
        tokio::select! {
            () = receive_signal(started_receiver) => {}
            _ = &mut spawn => panic!("initialization unexpectedly completed"),
        }

        clock.advance_to(1);
        assert!(matches!(
            spawn.await,
            Err(GpuTelemetryError::Worker(message)) if message == "vendor initialization timed out after 1ns"
        ));
        release_reply.try_send(()).unwrap();
        let dropped_on = receive_drop(drop_receiver).await;
        assert_ne!(dropped_on, thread::current().id());
        assert!(matches!(
            actions.lock().unwrap().as_slice(),
            [
                Action::Construct(_, worker_thread),
                Action::Initialize(_, initialize_thread),
                Action::Shutdown(_, shutdown_thread),
                Action::Drop(_, drop_thread),
            ] if worker_thread == initialize_thread
                && initialize_thread == shutdown_thread
                && shutdown_thread == drop_thread
                && *drop_thread == dropped_on
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn initialization_and_cleanup_failures_are_both_preserved() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let result = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions, |worker| {
                worker.initialize_error =
                    Some(GpuTelemetryError::Worker("initialize failed".to_string()));
                worker.shutdown_error =
                    Some(GpuTelemetryError::Worker("cleanup failed".to_string()));
            }),
        )
        .await;
        assert!(matches!(
            result,
            Err(GpuTelemetryError::Worker(message))
                if message.contains("initialize failed") && message.contains("cleanup failed")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn factory_runs_on_worker_and_factory_failure_is_typed() {
        let caller = thread::current().id();
        let result = VendorWorkerSource::spawn(clock(), ENDPOINT, move || {
            assert_eq!(thread::current().name(), Some(WORKER_THREAD_NAME));
            assert_ne!(thread::current().id(), caller);
            Err(GpuTelemetryError::Worker("factory failed".to_string()))
        })
        .await;
        assert!(matches!(
            result,
            Err(GpuTelemetryError::Worker(message)) if message == "factory failed"
        ));
    }

    #[test]
    fn full_command_and_reply_channels_are_typed_errors() {
        let (commands, _receiver) = sync_channel(CHANNEL_CAPACITY);
        commands.try_send(WorkerCommand::Shutdown).unwrap();
        let error = try_send_command(&commands, WorkerCommand::Shutdown).unwrap_err();
        assert!(matches!(
            error,
            GpuTelemetryError::Worker(message) if message.contains("command channel is full")
        ));

        let (reply, _receiver) = channel(CHANNEL_CAPACITY);
        reply.try_send(Ok(())).unwrap();
        let error = send_worker_reply(&reply, Ok(()), "scrape").unwrap_err();
        assert!(matches!(
            error,
            GpuTelemetryError::Worker(message) if message.contains("scrape reply channel is full")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn worker_panic_is_reported_by_shutdown_and_repeated_shutdown() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions, |worker| worker.panic_on_scrape = true),
        )
        .await
        .unwrap();
        let scrape_error = source.scrape(GpuScrapeMode::Continuous).await.unwrap_err();
        assert!(matches!(
            scrape_error,
            GpuTelemetryError::Worker(message) if message.contains("exited before the scrape reply")
        ));
        for _ in 0..2 {
            let shutdown_error = source.shutdown().await.unwrap_err();
            assert!(matches!(
                shutdown_error,
                GpuTelemetryError::Worker(message) if message.contains("thread panicked")
            ));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_shutdown_is_rejoined_by_repeated_shutdown() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let (started_reply, started_receiver) = sync_channel(CHANNEL_CAPACITY);
        let (release_reply, release_receiver) = sync_channel(CHANNEL_CAPACITY);
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions.clone(), move |worker| {
                worker.shutdown_started = Some(started_reply);
                worker.shutdown_release = Some(release_receiver);
            }),
        )
        .await
        .unwrap();

        {
            let shutdown = source.shutdown();
            tokio::pin!(shutdown);
            tokio::select! {
                () = receive_signal(started_receiver) => {}
                result = &mut shutdown => panic!("shutdown unexpectedly completed: {result:?}"),
            }
        }
        release_reply.try_send(()).unwrap();
        source.shutdown().await.unwrap();
        source.shutdown().await.unwrap();

        assert!(matches!(
            actions.lock().unwrap().as_slice(),
            [
                Action::Construct(_, _),
                Action::Initialize(_, _),
                Action::Shutdown(_, _),
                Action::Drop(_, _),
            ]
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_is_idempotent_and_ordered_after_boundary() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source =
            VendorWorkerSource::spawn(clock(), ENDPOINT, fake_factory(actions.clone(), |_| {}))
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
                Action::Construct(_, _),
                Action::Initialize(_, _),
                Action::Scrape(0, _, _),
                Action::Scrape(0, _, _),
                Action::Shutdown(_, _),
                Action::Drop(_, _),
            ]
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_shutdown_preserves_shutdown_failure() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions, |worker| {
                worker.shutdown_error =
                    Some(GpuTelemetryError::Worker("shutdown failed".to_string()));
            }),
        )
        .await
        .unwrap();
        for _ in 0..2 {
            assert!(matches!(
                source.shutdown().await,
                Err(GpuTelemetryError::Worker(message)) if message == "shutdown failed"
            ));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drop_requests_shutdown_and_reaper_joins_worker() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let (drop_reply, drop_receiver) = sync_channel(CHANNEL_CAPACITY);
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions.clone(), move |worker| {
                worker.drop_reply = Some(drop_reply);
            }),
        )
        .await
        .unwrap();
        drop(source);
        let dropped_on = receive_drop(drop_receiver).await;

        let actions = actions.lock().unwrap();
        assert!(matches!(
            actions.as_slice(),
            [
                Action::Construct(_, worker_thread),
                Action::Initialize(_, initialize_thread),
                Action::Shutdown(_, shutdown_thread),
                Action::Drop(_, drop_thread),
            ] if worker_thread == initialize_thread
                && initialize_thread == shutdown_thread
                && shutdown_thread == drop_thread
                && *drop_thread == dropped_on
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn record_identity_and_timestamp_are_enforced() {
        let actions = Arc::new(StdMutex::new(Vec::new()));
        let source = VendorWorkerSource::spawn(
            clock(),
            ENDPOINT,
            fake_factory(actions, |worker| {
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
            }),
        )
        .await
        .unwrap();
        let error = source.scrape(GpuScrapeMode::Boundary).await.unwrap_err();
        assert!(matches!(error, GpuTelemetryError::Protocol(_)));
        source.shutdown().await.unwrap();
    }
}
