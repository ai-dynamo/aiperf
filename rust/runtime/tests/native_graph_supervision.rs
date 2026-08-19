// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial contracts for supervised NativeGraph adapter sessions.

use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    time::Duration,
};

use aiperf_runtime::eval::{
    AdapterEnvelope, AdapterExit, AdapterLifecycleDeadlines, AdapterMessage, AdapterPool,
    AdapterPoolKey, AdapterProcess, AdapterProtocolConfig, AdapterRole, AdapterRuntimeFactory,
    AdapterSpawnRequest, AdapterSpawnTransaction, AdapterSpawner, AdapterSupervisionError,
    ArtifactDigest, CancelReason, HostEnvelope, HostMessage, LocalAdapterSpawner,
    ProtocolCapability, ProtocolError, ProtocolLimits, StrictAdapterProtocolFactory,
    SupervisedAdapter,
};
use async_trait::async_trait;

#[derive(Default)]
struct ChildObservations {
    events: Vec<String>,
    stdout: VecDeque<Vec<u8>>,
    stderr: Vec<u8>,
    read_deadlines: Vec<std::time::Duration>,
    read_max_bytes: Vec<usize>,
    write_deadlines: Vec<std::time::Duration>,
    write_delay: Option<std::time::Duration>,
}

struct RecordingChild {
    observations: Rc<RefCell<ChildObservations>>,
}

#[async_trait(?Send)]
impl AdapterProcess for RecordingChild {
    async fn write_frame(
        &mut self,
        _: &[u8],
        deadline: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let delay = {
            let mut observations = self.observations.borrow_mut();
            observations.events.push("write".to_owned());
            observations.write_deadlines.push(deadline);
            observations.write_delay
        };
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        Ok(())
    }

    async fn read_stdout_frame(
        &mut self,
        max_bytes: usize,
        deadline: std::time::Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        let mut observations = self.observations.borrow_mut();
        observations.read_deadlines.push(deadline);
        observations.read_max_bytes.push(max_bytes);
        observations
            .stdout
            .pop_front()
            .ok_or(AdapterSupervisionError::EndOfStream)
    }

    async fn drain_stderr(&mut self, max_bytes: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        let diagnostics = std::mem::take(&mut self.observations.borrow_mut().stderr);
        if diagnostics.len() > max_bytes {
            return Err(AdapterSupervisionError::bounded_diagnostic_output(
                diagnostics.len(),
                max_bytes,
            ));
        }
        Ok(diagnostics)
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.observations
            .borrow_mut()
            .events
            .push("cancel".to_owned());
        Ok(())
    }

    async fn reap(
        &mut self,
        _: std::time::Duration,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        self.observations
            .borrow_mut()
            .events
            .push("reap".to_owned());
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {
        self.observations
            .borrow_mut()
            .events
            .push("fence".to_owned());
    }
}

struct ResetFailsAdapter {
    observations: Rc<RefCell<Vec<String>>>,
}

#[async_trait(?Send)]
impl SupervisedAdapter for ResetFailsAdapter {
    async fn send(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn receive(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_heartbeat(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_idle(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn reset(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        self.observations.borrow_mut().push("reset".to_owned());
        Err(AdapterSupervisionError::ResetRejected("fixture".to_owned()))
    }

    fn release_download_handle(
        &mut self,
        _: &aiperf_runtime::eval::ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        _: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        self.observations.borrow_mut().push("reap".to_owned());
        Ok(AdapterExit::Reaped)
    }
}

struct FreshAdapter;

#[async_trait(?Send)]
impl SupervisedAdapter for FreshAdapter {
    async fn send(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn receive(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_heartbeat(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_idle(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn reset(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    fn release_download_handle(
        &mut self,
        _: &aiperf_runtime::eval::ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        _: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        Ok(AdapterExit::Reaped)
    }
}

struct FreshOnlyFactory {
    spawns: Rc<RefCell<usize>>,
}

struct RecordingSpawner {
    observations: Rc<RefCell<ChildObservations>>,
    launch_delay: Option<Duration>,
}

impl AdapterSpawner for RecordingSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        Ok(Box::new(RecordingSpawnTransaction {
            process: Some(Box::new(RecordingChild {
                observations: self.observations.clone(),
            })),
            delay: self.launch_delay,
        }))
    }
}

struct RecordingSpawnTransaction {
    process: Option<Box<dyn AdapterProcess>>,
    delay: Option<Duration>,
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for RecordingSpawnTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        self.process
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn abort(&mut self, deadline: Duration) -> Result<(), AdapterSupervisionError> {
        let Some(mut process) = self.process.take() else {
            return Ok(());
        };
        process.cancel(CancelReason::HostShutdown, deadline).await?;
        process.reap(deadline).await?;
        Ok(())
    }

    fn fence(&mut self) {
        if let Some(process) = self.process.as_deref_mut() {
            process.fence();
        }
    }
}

fn runtime_factory(observations: Rc<RefCell<ChildObservations>>) -> impl AdapterRuntimeFactory {
    runtime_factory_with_limits(observations, ProtocolLimits::default())
}

fn runtime_factory_with_limits(
    observations: Rc<RefCell<ChildObservations>>,
    limits: ProtocolLimits,
) -> impl AdapterRuntimeFactory {
    observations.borrow_mut().stdout.push_back(ready_frame());
    let config = AdapterProtocolConfig::new(
        AdapterRole::Tool,
        "episode",
        [ProtocolCapability::Tool].into_iter().collect(),
        Default::default(),
        limits,
    )
    .expect("fixture protocol config is valid");
    aiperf_runtime::eval::ProtocolAdapterRuntimeFactory::new(
        config,
        Rc::new(StrictAdapterProtocolFactory),
        Rc::new(RecordingSpawner {
            observations,
            launch_delay: None,
        }),
    )
}

fn delayed_runtime_factory(
    observations: Rc<RefCell<ChildObservations>>,
    launch_delay: Duration,
) -> impl AdapterRuntimeFactory {
    observations.borrow_mut().stdout.push_back(ready_frame());
    let config = AdapterProtocolConfig::new(
        AdapterRole::Tool,
        "episode",
        [ProtocolCapability::Tool].into_iter().collect(),
        Default::default(),
        ProtocolLimits::default(),
    )
    .expect("fixture protocol config is valid");
    aiperf_runtime::eval::ProtocolAdapterRuntimeFactory::new(
        config,
        Rc::new(StrictAdapterProtocolFactory),
        Rc::new(RecordingSpawner {
            observations,
            launch_delay: Some(launch_delay),
        }),
    )
}

fn driver_runtime_factory(
    observations: Rc<RefCell<ChildObservations>>,
) -> impl AdapterRuntimeFactory {
    observations
        .borrow_mut()
        .stdout
        .push_back(driver_ready_frame());
    let config = AdapterProtocolConfig::new(
        AdapterRole::Driver,
        "episode",
        [ProtocolCapability::Driver].into_iter().collect(),
        Default::default(),
        ProtocolLimits::default(),
    )
    .expect("fixture Driver protocol config is valid");
    aiperf_runtime::eval::ProtocolAdapterRuntimeFactory::new(
        config,
        Rc::new(StrictAdapterProtocolFactory),
        Rc::new(RecordingSpawner {
            observations,
            launch_delay: None,
        }),
    )
}

fn ready_frame() -> Vec<u8> {
    let mut frame = serde_json::to_vec(&AdapterEnvelope::new(
        "episode",
        "startup",
        0,
        "hello",
        AdapterMessage::Ready {
            protocol_version: 1,
            capabilities: vec![ProtocolCapability::Tool],
            implementation_digest: ArtifactDigest::from_bytes(b"fixture-adapter"),
        },
    ))
    .expect("fixture ready frame serializes");
    frame.push(b'\n');
    frame
}

fn driver_ready_frame() -> Vec<u8> {
    let mut frame = serde_json::to_vec(&AdapterEnvelope::new(
        "episode",
        "startup",
        0,
        "hello",
        AdapterMessage::Ready {
            protocol_version: 1,
            capabilities: vec![ProtocolCapability::Driver],
            implementation_digest: ArtifactDigest::from_bytes(b"fixture-driver"),
        },
    ))
    .expect("fixture Driver ready frame serializes");
    frame.push(b'\n');
    frame
}

fn terminal_candidate_frame(sequence: u64) -> Vec<u8> {
    let mut frame = serde_json::to_vec(&AdapterEnvelope::new(
        "episode",
        "external-driver-terminal",
        sequence,
        "external-driver-terminal",
        AdapterMessage::EpisodeTerminalCandidate {
            output: serde_json::json!({"terminal": "accepted"}),
        },
    ))
    .expect("fixture terminal candidate frame serializes");
    frame.push(b'\n');
    frame
}

fn reset_ack_frame() -> Vec<u8> {
    let mut frame = serde_json::to_vec(&AdapterEnvelope::new(
        "episode",
        "span",
        1,
        "reset",
        AdapterMessage::ResetAck {
            effective_seed: 7,
            implementation_digest: ArtifactDigest::from_bytes(b"fixture-adapter"),
        },
    ))
    .expect("fixture reset acknowledgement serializes");
    frame.push(b'\n');
    frame
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for FreshOnlyFactory {
    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        *self.spawns.borrow_mut() += 1;
        Ok(Box::new(FreshAdapter))
    }
}

fn reset() -> HostEnvelope {
    HostEnvelope::new(
        "episode",
        "span",
        1,
        "reset",
        HostMessage::Reset {
            seed: 7,
            identities: Vec::new(),
        },
    )
}

fn key(task: &[u8]) -> AdapterPoolKey {
    AdapterPoolKey::new(
        aiperf_runtime::eval::ArtifactDigest::from_bytes(task),
        aiperf_runtime::eval::ArtifactDigest::from_bytes(b"environment"),
        aiperf_runtime::eval::ArtifactDigest::from_bytes(b"adapter"),
        aiperf_runtime::eval::AdapterRole::Tool,
        aiperf_runtime::eval::ArtifactDigest::from_bytes(b"protocol"),
    )
}

fn spawn_request() -> AdapterSpawnRequest {
    spawn_request_with_deadlines(AdapterLifecycleDeadlines::default())
}

fn spawn_request_with_deadlines(deadlines: AdapterLifecycleDeadlines) -> AdapterSpawnRequest {
    AdapterSpawnRequest::for_non_model_adapter(["adapter".to_owned()], BTreeMap::new(), deadlines)
        .expect("fixture adapter request is valid")
}

#[test]
fn strict_output_limits_can_only_be_lowered() {
    let error = spawn_request()
        .with_output_limits(usize::MAX, usize::MAX)
        .expect_err("a peer cannot expand strict adapter output caps");
    assert!(matches!(
        error,
        AdapterSupervisionError::OutputLimitIncrease { .. }
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn protocol_factory_intersects_peer_stdout_cap_with_its_protocol_limit() {
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut limits = ProtocolLimits::default();
    limits.max_frame_bytes = 1024;
    let _adapter = runtime_factory_with_limits(observations.clone(), limits)
        .start(spawn_request())
        .await
        .expect("the Ready frame is below the protocol frame limit");

    assert_eq!(observations.borrow().read_max_bytes, [1024]);
}

#[tokio::test(flavor = "current_thread")]
async fn supervised_driver_refuses_a_candidate_after_terminal_settlement() {
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut driver = driver_runtime_factory(observations.clone())
        .start(spawn_request())
        .await
        .expect("the Driver session starts and acknowledges its exact capability");
    observations
        .borrow_mut()
        .stdout
        .push_back(terminal_candidate_frame(1));
    driver
        .send(HostEnvelope::new(
            "episode",
            "external-driver-terminal",
            1,
            "external-driver-terminal",
            HostMessage::RequestEpisodeTerminal {
                input: serde_json::json!({}),
            },
        ))
        .await
        .expect("the Driver terminal request is sent with its fixed correlation");
    assert!(matches!(
        driver.receive().await,
        Ok(AdapterEnvelope {
            message: AdapterMessage::EpisodeTerminalCandidate { .. },
            ..
        })
    ));

    observations
        .borrow_mut()
        .stdout
        .push_back(terminal_candidate_frame(2));
    assert!(matches!(
        driver.receive().await,
        Err(AdapterSupervisionError::Protocol(
            ProtocolError::OperationState { .. }
        ))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_start_fences_a_delayed_spawn_transaction() {
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let factory = delayed_runtime_factory(observations.clone(), Duration::from_secs(1));
    let mut startup = Box::pin(factory.start(spawn_request()));

    tokio::select! {
        _ = &mut startup => panic!("the delayed spawn must still be pending"),
        _ = tokio::time::sleep(Duration::from_millis(20)) => {}
    }
    drop(startup);

    assert_eq!(observations.borrow().events, ["fence"]);
}

#[tokio::test(flavor = "current_thread")]
async fn reset_failure_reaps_then_forces_a_fresh_adapter() {
    let observations = Rc::new(RefCell::new(Vec::new()));
    let spawns = Rc::new(RefCell::new(0));
    let factory = FreshOnlyFactory {
        spawns: spawns.clone(),
    };
    let mut pool = AdapterPool::default();
    pool.return_adapter(
        key(b"task-a"),
        Box::new(ResetFailsAdapter {
            observations: observations.clone(),
        }),
    )
    .await
    .expect("fixture worker enters pool");

    let checkout = pool
        .checkout_or_start(key(b"task-a"), reset(), spawn_request(), &factory)
        .await
        .expect("failed reset must be reaped and replaced with a fresh adapter");
    assert!(checkout.is_fresh());
    assert_eq!(*spawns.borrow(), 1);
    assert_eq!(
        observations
            .borrow()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["reset", "reap"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn pool_key_never_reuses_an_adapter_across_tasks() {
    let spawns = Rc::new(RefCell::new(0));
    let factory = FreshOnlyFactory {
        spawns: spawns.clone(),
    };
    let mut pool = AdapterPool::default();
    pool.return_adapter(key(b"task-a"), Box::new(FreshAdapter))
        .await
        .expect("fixture worker enters pool");

    let checkout = pool
        .checkout_or_start(key(b"task-b"), reset(), spawn_request(), &factory)
        .await
        .expect("a different task digest must never draw task-a's worker");
    assert!(checkout.is_fresh());
    assert_eq!(*spawns.borrow(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn bounded_pool_evicts_and_reaps_the_oldest_idle_adapter() {
    let observations = Rc::new(RefCell::new(Vec::new()));
    let mut pool = AdapterPool::with_capacity(1);
    pool.return_adapter(
        key(b"task-a"),
        Box::new(ResetFailsAdapter {
            observations: observations.clone(),
        }),
    )
    .await
    .expect("first idle adapter fits the bounded pool");
    pool.return_adapter(key(b"task-b"), Box::new(FreshAdapter))
        .await
        .expect("eviction reaps the old worker before retaining a new one");
    assert_eq!(
        observations
            .borrow()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["reap"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_reaps_a_child_and_drop_fences_an_unfinished_child() {
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut child = RecordingChild {
        observations: observations.clone(),
    };
    child
        .cancel(
            CancelReason::HostShutdown,
            AdapterLifecycleDeadlines::default().cancel(),
        )
        .await
        .expect("fixture child accepts cancellation");
    assert_eq!(
        child
            .reap(AdapterLifecycleDeadlines::default().reap())
            .await
            .expect("fixture child reaps"),
        AdapterExit::Reaped
    );
    child.fence();
    assert_eq!(&*observations.borrow().events, ["cancel", "reap", "fence"]);
}

#[tokio::test(flavor = "current_thread")]
async fn supervision_bounded_drain_rejects_diagnostic_overflow() {
    let observations = Rc::new(RefCell::new(ChildObservations {
        stderr: vec![b'x'; 9],
        ..ChildObservations::default()
    }));
    let mut child = RecordingChild { observations };
    let error = child
        .drain_stderr(8)
        .await
        .expect_err("adapter stderr must be bounded before it can deadlock a child");
    assert!(
        error
            .to_string()
            .contains("diagnostic output exceeds 8 bytes")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn local_process_enforces_the_stdout_cap_before_a_long_unterminated_frame() {
    let temporary = tempfile::tempdir().expect("temporary local adapter root");
    let request = AdapterSpawnRequest::for_non_model_adapter(
        [
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            "head -c 1024 /dev/zero".to_owned(),
        ],
        BTreeMap::new(),
        AdapterLifecycleDeadlines::default(),
    )
    .expect("fixture local spawn request is valid")
    .with_output_limits(8, 8)
    .expect("fixture output cap is valid");
    let mut transaction = LocalAdapterSpawner::new(temporary.path())
        .begin_spawn(request)
        .expect("local adapter starts");
    let process = transaction
        .await_process()
        .await
        .expect("local adapter launch completes");
    let mut process = process;
    let error = process
        .read_stdout_frame(8, std::time::Duration::from_secs(1))
        .await
        .expect_err("unterminated stdout must fail at max plus one byte");
    assert!(error.to_string().contains("stdout frame exceeds 8 bytes"));
    process
        .cancel(
            CancelReason::HostShutdown,
            std::time::Duration::from_secs(1),
        )
        .await
        .expect("local adapter cancellation succeeds");
    process
        .reap(std::time::Duration::from_secs(1))
        .await
        .expect("local adapter reaps after the bounded-frame failure");
}

#[tokio::test(flavor = "current_thread")]
async fn local_frame_timeout_is_one_absolute_budget_across_all_bytes() {
    let temporary = tempfile::tempdir().expect("temporary local adapter root");
    let request = AdapterSpawnRequest::for_non_model_adapter(
        [
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            "printf a; sleep 0.10; printf b; sleep 0.10; printf c; sleep 0.10; printf '\\n'"
                .to_owned(),
        ],
        BTreeMap::new(),
        AdapterLifecycleDeadlines::default(),
    )
    .expect("fixture local spawn request is valid");
    let mut transaction = LocalAdapterSpawner::new(temporary.path())
        .begin_spawn(request)
        .expect("local adapter starts");
    let mut process = transaction
        .await_process()
        .await
        .expect("local adapter launch completes");

    let error = process
        .read_stdout_frame(16, Duration::from_millis(180))
        .await
        .expect_err("a JSONL frame cannot renew its deadline per byte");
    assert!(error.to_string().contains("stdout deadline elapsed"));
    process
        .cancel(CancelReason::HostShutdown, Duration::from_secs(1))
        .await
        .expect("local adapter cancellation succeeds");
    process
        .reap(Duration::from_secs(1))
        .await
        .expect("local adapter reaps after the deadline failure");
}

#[tokio::test(flavor = "current_thread")]
async fn startup_negotiates_hello_and_ready_before_returning_a_session() {
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let _adapter = runtime_factory(observations.clone())
        .start(spawn_request())
        .await
        .expect("startup returns only after a ready acknowledgement");

    let observations = observations.borrow();
    assert_eq!(observations.events, ["write"]);
    assert_eq!(observations.read_deadlines.len(), 1);
    assert!(
        observations.read_deadlines[0] <= AdapterLifecycleDeadlines::default().startup()
            && observations.read_deadlines[0] > Duration::from_secs(29),
        "the Ready read receives the startup budget remaining after spawn and Hello"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn reset_send_and_ack_share_one_absolute_reset_deadline() {
    let deadlines = AdapterLifecycleDeadlines::new(
        Duration::from_secs(5),
        Duration::from_millis(300),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(5),
    )
    .expect("fixture deadlines are valid");
    let observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut adapter = runtime_factory(observations.clone())
        .start(spawn_request_with_deadlines(deadlines))
        .await
        .expect("fixture adapter negotiates startup");
    {
        let mut observations = observations.borrow_mut();
        observations.events.clear();
        observations.read_deadlines.clear();
        observations.write_deadlines.clear();
        observations.write_delay = Some(Duration::from_millis(100));
        observations.stdout.push_back(reset_ack_frame());
    }

    adapter
        .reset(reset())
        .await
        .expect("reset acknowledgement fits the shared deadline");
    let observations = observations.borrow();
    assert_eq!(observations.write_deadlines.len(), 1);
    assert!(
        observations.write_deadlines[0] <= deadlines.reset()
            && observations.write_deadlines[0] > Duration::from_millis(250)
    );
    assert_eq!(observations.read_deadlines.len(), 1);
    assert!(
        observations.read_deadlines[0] < Duration::from_millis(250),
        "the acknowledgement receives only the reset budget remaining after send"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn local_reap_waits_for_a_stubborn_process_group_descendant() {
    let temporary = tempfile::tempdir().expect("temporary local adapter root");
    let pid_path = temporary.path().join("stubborn-child.pid");
    let script = format!(
        "trap '' TERM; (trap '' TERM; while :; do sleep 1; done) & child=$!; echo $child > {}; wait $child",
        pid_path.display()
    );
    let request = AdapterSpawnRequest::for_non_model_adapter(
        ["/bin/sh".to_owned(), "-c".to_owned(), script],
        BTreeMap::new(),
        AdapterLifecycleDeadlines::default(),
    )
    .expect("fixture local spawn request is valid");
    let mut transaction = LocalAdapterSpawner::new(temporary.path())
        .begin_spawn(request)
        .expect("local adapter starts");
    let mut process = transaction
        .await_process()
        .await
        .expect("local adapter launch completes");
    tokio::time::timeout(Duration::from_secs(1), async {
        while !pid_path.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("stubborn descendant records its process identifier");
    let pid = std::fs::read_to_string(&pid_path)
        .expect("stubborn pid file exists")
        .trim()
        .parse::<i32>()
        .expect("stubborn pid is numeric");

    process
        .cancel(CancelReason::HostShutdown, Duration::from_secs(1))
        .await
        .expect("initial graceful group termination succeeds");
    let reaped = process.reap(Duration::from_secs(1)).await;
    let descendant_gone = process_is_absent(pid);
    process.fence();

    assert!(matches!(reaped, Ok(AdapterExit::Reaped)));
    assert!(
        descendant_gone,
        "reap must not return while a PGID descendant lives"
    );
}

#[cfg(unix)]
fn process_is_absent(pid: i32) -> bool {
    let result = unsafe { libc::kill(pid, 0) };
    result != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn process_is_absent(_: i32) -> bool {
    true
}

#[tokio::test(flavor = "current_thread")]
async fn supervised_receive_paths_consume_distinct_heartbeat_and_idle_deadlines() {
    let deadlines = AdapterLifecycleDeadlines::new(
        std::time::Duration::from_secs(1),
        std::time::Duration::from_secs(2),
        std::time::Duration::from_secs(3),
        std::time::Duration::from_secs(4),
        std::time::Duration::from_secs(5),
        std::time::Duration::from_secs(6),
        std::time::Duration::from_secs(7),
    )
    .expect("fixture deadlines are valid");
    let heartbeat_observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut heartbeat = runtime_factory(heartbeat_observations.clone())
        .start(spawn_request_with_deadlines(deadlines))
        .await
        .expect("fixture runtime starts");
    let _ = heartbeat.receive_heartbeat().await;
    let heartbeat_reads = &heartbeat_observations.borrow().read_deadlines;
    assert_eq!(heartbeat_reads.len(), 2);
    assert!(heartbeat_reads[0] <= deadlines.startup());
    assert_eq!(heartbeat_reads[1], deadlines.heartbeat());

    let idle_observations = Rc::new(RefCell::new(ChildObservations::default()));
    let mut idle = runtime_factory(idle_observations.clone())
        .start(spawn_request_with_deadlines(deadlines))
        .await
        .expect("fixture runtime starts");
    let _ = idle.receive_idle().await;
    let idle_reads = &idle_observations.borrow().read_deadlines;
    assert_eq!(idle_reads.len(), 2);
    assert!(idle_reads[0] <= deadlines.startup());
    assert_eq!(idle_reads[1], deadlines.idle());
}
