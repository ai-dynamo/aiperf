// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted structured Docker command contracts for Harbor execution.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    io::{self, Read},
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use aiperf_runtime::clock::SimClock;
use aiperf_runtime::eval::{
    AdapterEnvelope, AdapterExit, AdapterLifecycleDeadlines, AdapterMessage, AdapterProcess,
    AdapterSpawnRequest, AdapterSpawnTransaction, AdapterSpawner, AdapterSupervisionError,
    AgentVariantRef, ArtifactDigest, CancelReason, CompatibilityTerminalReceipt, ComposeProjectId,
    DockerAdapterLease, DockerAdapterProcess, DockerBuildRequest, DockerComposeArchiveRequest,
    DockerComposeBuildRequest, DockerComposeConfigRequest, DockerComposeCopyRequest,
    DockerComposeDownRequest, DockerComposeExecRequest, DockerComposeRuntime,
    DockerComposeStopRequest, DockerComposeUpRequest, DockerCopyRequest, DockerCreateRequest,
    DockerExecRequest, DockerProcessSandbox, DockerRemoveRequest, DockerRuntime,
    DockerStartRequest, EnvName, EvalExecutionError, ExternalDriverDockerSpawner,
    ExternalDriverError, ExternalDriverSession, ExternalDriverSpawnExecutor, HarborImporter,
    HarborSandboxRecipe, HarborSource, HostEnvelope, HostMessage, ModelEndpointIsolationProof,
    ModelIdentity, ModelSecretId, NativeGraphEpisodeCallback, NativeGraphEpisodeLease,
    NativeGraphExternalDriverFactory, NativeGraphPackagePlan, NativeGraphSuiteManifest,
    NativeSourceAcquirer, OwnedComposeResources, PolicyIdentity, PreparedExternalDriver,
    PreparedExternalDriverCapability, ProtocolCapability, ProviderCapabilities, ProviderCapability,
    ProviderProfile, ResourceLeaseRequest, RuntimeIdentity, SecretProvider, SecretValue,
    SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec, preflight_docker,
};
use async_trait::async_trait;
use std::rc::Rc;

static DOCKER_RUNTIME_TEST_LOCK: Mutex<()> = Mutex::new(());

struct LeaseFenceClient {
    events: Rc<RefCell<Vec<String>>>,
}

struct RecordingAdapterSpawnTransaction {
    process: Option<Box<dyn AdapterProcess>>,
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for RecordingAdapterSpawnTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
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

struct RecordingAdapterSpawner {
    events: Rc<RefCell<Vec<String>>>,
    requests: Rc<RefCell<Vec<(Vec<String>, BTreeMap<String, String>)>>>,
}

struct RecordingExternalDriverSpawnExecutor {
    events: Rc<RefCell<Vec<String>>>,
    requests: Rc<
        RefCell<
            Vec<(
                String,
                Vec<String>,
                BTreeMap<String, String>,
                AdapterLifecycleDeadlines,
            )>,
        >,
    >,
}

impl ExternalDriverSpawnExecutor for RecordingExternalDriverSpawnExecutor {
    fn begin_spawn(
        &self,
        request: aiperf_runtime::eval::AuthorizedExternalDriverSpawn,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        self.events.borrow_mut().push("adapter-spawn".to_owned());
        self.requests.borrow_mut().push((
            request.container().to_owned(),
            request.argv().to_vec(),
            request.environment().clone(),
            request.deadlines(),
        ));
        Ok(Box::new(RecordingAdapterSpawnTransaction {
            process: Some(Box::new(LeaseFenceClient {
                events: self.events.clone(),
            })),
        }))
    }
}

impl AdapterSpawner for RecordingAdapterSpawner {
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        self.events.borrow_mut().push("adapter-spawn".to_owned());
        self.requests
            .borrow_mut()
            .push((request.argv().to_vec(), request.environment().clone()));
        Ok(Box::new(RecordingAdapterSpawnTransaction {
            process: Some(Box::new(LeaseFenceClient {
                events: self.events.clone(),
            })),
        }))
    }
}

#[async_trait(?Send)]
impl AdapterProcess for LeaseFenceClient {
    async fn write_frame(&mut self, _: &[u8], _: Duration) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn read_stdout_frame(
        &mut self,
        _: usize,
        _: Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        Ok(Vec::new())
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.events.borrow_mut().push("client-cancel".to_owned());
        Ok(())
    }

    async fn reap(&mut self, _: Duration) -> Result<AdapterExit, AdapterSupervisionError> {
        self.events.borrow_mut().push("client-reap".to_owned());
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {
        self.events.borrow_mut().push("client-fence".to_owned());
    }
}

/// Strict JSONL child that is sufficient to prove Docker can start a selected rollout.
/// It deliberately never receives reset or action authority in this start-only test.
struct ReadyAdapterClient {
    events: Rc<RefCell<Vec<String>>>,
    stdout: VecDeque<Vec<u8>>,
}

#[async_trait(?Send)]
impl AdapterProcess for ReadyAdapterClient {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        _: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let host: HostEnvelope = serde_json::from_slice(frame).map_err(|error| {
            AdapterSupervisionError::Process(format!("fixture host frame is invalid: {error}"))
        })?;
        let episode = host.episode.clone();
        let operation = host.operation.clone();
        if matches!(host.message, HostMessage::Hello { .. }) {
            let ready = AdapterEnvelope::new(
                episode,
                "startup",
                0,
                operation,
                AdapterMessage::Ready {
                    protocol_version: 1,
                    capabilities: vec![
                        ProtocolCapability::Environment,
                        ProtocolCapability::Artifacts,
                    ],
                    implementation_digest: ArtifactDigest::from_bytes(b"docker-rollout-ready"),
                },
            );
            let mut frame = serde_json::to_vec(&ready).map_err(|error| {
                AdapterSupervisionError::Process(format!("fixture ready frame is invalid: {error}"))
            })?;
            frame.push(b'\n');
            self.stdout.push_back(frame);
        }
        Ok(())
    }

    async fn read_stdout_frame(
        &mut self,
        _: usize,
        _: Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.stdout
            .pop_front()
            .ok_or(AdapterSupervisionError::EndOfStream)
    }

    async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        Ok(Vec::new())
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.events
            .borrow_mut()
            .push("rollout-client-cancel".to_owned());
        Ok(())
    }

    async fn reap(&mut self, _: Duration) -> Result<AdapterExit, AdapterSupervisionError> {
        self.events
            .borrow_mut()
            .push("rollout-client-reap".to_owned());
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {
        self.events
            .borrow_mut()
            .push("rollout-client-fence".to_owned());
    }
}

struct ReadyAdapterSpawner {
    events: Rc<RefCell<Vec<String>>>,
    requests: Rc<RefCell<Vec<(Vec<String>, BTreeMap<String, String>)>>>,
}

impl AdapterSpawner for ReadyAdapterSpawner {
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        self.events
            .borrow_mut()
            .push("rollout-adapter-spawn".to_owned());
        self.requests
            .borrow_mut()
            .push((request.argv().to_vec(), request.environment().clone()));
        Ok(Box::new(RecordingAdapterSpawnTransaction {
            process: Some(Box::new(ReadyAdapterClient {
                events: Rc::clone(&self.events),
                stdout: VecDeque::new(),
            })),
        }))
    }
}

struct ComposeLeaseFake {
    project: ComposeProjectId,
    container: String,
    events: Rc<RefCell<Vec<String>>>,
}

#[async_trait(?Send)]
impl DockerAdapterLease for ComposeLeaseFake {
    async fn terminate(&self, _: Duration) -> Result<(), AdapterSupervisionError> {
        self.events.borrow_mut().push(format!(
            "terminate:{}:{}",
            self.project.as_str(),
            self.container
        ));
        Ok(())
    }

    fn fence(&self) {
        self.events.borrow_mut().push(format!(
            "fence:{}:{}",
            self.project.as_str(),
            self.container
        ));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn compose_lease_fences_its_exact_remote_container_before_client_reap() {
    let project = ComposeProjectId::new("aiperf-task-lease");
    let container = "aiperf-task-lease-main-1".to_owned();
    let events = Rc::new(RefCell::new(Vec::new()));
    let lease = Rc::new(ComposeLeaseFake {
        project: project.clone(),
        container: container.clone(),
        events: events.clone(),
    });
    let mut process = DockerAdapterProcess::new(
        Box::new(LeaseFenceClient {
            events: events.clone(),
        }),
        lease,
    );

    process.fence();
    assert_eq!(
        events.borrow().as_slice(),
        [
            "client-fence",
            &format!("fence:{}:{container}", project.as_str()),
        ]
    );

    process
        .cancel(CancelReason::HostShutdown, Duration::from_secs(1))
        .await
        .expect("fixture remote termination succeeds");
    assert_eq!(
        events.borrow().as_slice(),
        [
            "client-fence",
            &format!("fence:{}:{container}", project.as_str()),
            "client-cancel",
            &format!("terminate:{}:{container}", project.as_str()),
        ]
    );
    let exit = process
        .reap(Duration::from_secs(1))
        .await
        .expect("remote lease was terminated before client reaping");
    assert_eq!(exit, AdapterExit::Reaped);
    assert_eq!(
        events.borrow().as_slice(),
        [
            "client-fence",
            &format!("fence:{}:{container}", project.as_str()),
            "client-cancel",
            &format!("terminate:{}:{container}", project.as_str()),
            "client-reap",
        ]
    );
}

#[derive(Default)]
struct RecordingRuntime {
    build_calls: Cell<usize>,
    events: RefCell<Vec<String>>,
}

#[derive(Default)]
struct LegacyRuntime {
    events: Rc<RefCell<Vec<String>>>,
    images: RefCell<Vec<String>>,
    creates: RefCell<Vec<Vec<String>>>,
    adapter_spawner_labels: RefCell<Vec<BTreeMap<String, String>>>,
    external_driver_spawner_calls: Cell<usize>,
    native_graph_secret_provider_calls: Cell<usize>,
    native_graph_profile: Option<ProviderProfile>,
    image_workdir: Option<String>,
    adapter_spawner: Option<Rc<dyn AdapterSpawner>>,
    external_driver_spawn_executor: Option<Rc<dyn ExternalDriverSpawnExecutor>>,
}

impl DockerRuntime for LegacyRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        let capabilities = ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_public_network();
        let capabilities = if self.native_graph_profile.is_some() {
            capabilities.with_model_endpoint_isolation()
        } else {
            capabilities
        };
        if self.image_workdir.is_some() {
            capabilities.with_workdir()
        } else {
            capabilities
        }
    }

    fn native_graph_provider_profile(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<ProviderProfile, EvalExecutionError> {
        self.native_graph_profile
            .clone()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "model endpoint isolation",
            ))
    }

    fn native_graph_model_secret_environment(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<BTreeMap<ModelSecretId, EnvName>, EvalExecutionError> {
        self.native_graph_secret_provider_calls
            .set(self.native_graph_secret_provider_calls.get() + 1);
        if self.native_graph_profile.is_some() {
            Ok(BTreeMap::new())
        } else {
            Err(EvalExecutionError::UnsupportedEnforcement(
                "native graph model secret environment",
            ))
        }
    }

    fn adapter_spawner(
        &self,
        request: &aiperf_runtime::eval::DockerAdapterSpawnerRequest,
        _: &aiperf_runtime::eval::NativeGraphAdapterAuthorization,
    ) -> Result<Rc<dyn AdapterSpawner>, EvalExecutionError> {
        self.adapter_spawner_labels
            .borrow_mut()
            .push(request.project().ownership_labels());
        self.adapter_spawner
            .clone()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "streaming Docker adapter spawn",
            ))
    }

    fn external_driver_spawner(
        &self,
        request: &aiperf_runtime::eval::DockerAdapterSpawnerRequest,
    ) -> Result<ExternalDriverDockerSpawner, EvalExecutionError> {
        self.external_driver_spawner_calls
            .set(self.external_driver_spawner_calls.get() + 1);
        self.events
            .borrow_mut()
            .push("external-driver-spawner".to_owned());
        self.external_driver_spawn_executor
            .clone()
            .map(|executor| ExternalDriverDockerSpawner::new(request, executor))
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "external Driver Docker adapter spawn",
            ))
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.creates
            .borrow_mut()
            .push(request.public_arguments().to_vec());
        self.images.borrow_mut().push(
            request
                .public_arguments()
                .iter()
                .rev()
                .nth(2)
                .expect("container image")
                .clone(),
        );
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push(request.phase().to_string());
        Ok(())
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .last()
            .is_some_and(|destination| destination.ends_with("reward.txt"))
        {
            fs::write(request.public_arguments().last().unwrap(), "1\n")
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        }
        Ok(())
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "reward.txt",
            b"1\n",
        ))))
    }

    fn container_workdir(&self, _: &str) -> Result<String, EvalExecutionError> {
        self.image_workdir
            .clone()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "container workdir inspection",
            ))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

impl DockerRuntime for RecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("exec".to_owned());
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("copy".to_owned());
        Ok(())
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "reward.txt",
            b"1\n",
        ))))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct ComposePreflightRuntime {
    config_calls: Cell<usize>,
    config_deadline: Cell<Duration>,
    build_calls: Cell<usize>,
    up_calls: Cell<usize>,
}

impl DockerRuntime for ComposePreflightRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_public_network()
            .with_compose_project()
            .with_compose_config()
            .with_service_exec()
            .with_service_archive()
            .with_service_stop()
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "reward.txt",
            b"1\n",
        ))))
    }

    fn compose_runtime(&self) -> Option<&dyn DockerComposeRuntime> {
        Some(self)
    }
    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        Ok(())
    }
    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
}

impl DockerComposeRuntime for ComposePreflightRuntime {
    fn compose_config(
        &self,
        request: &DockerComposeConfigRequest,
    ) -> Result<Vec<u8>, EvalExecutionError> {
        assert!(request.interpolation_disabled());
        assert!(request.env_file_disabled());
        let deadline = request
            .deadline()
            .filter(|deadline| !deadline.is_zero())
            .expect("Compose config request must be deadline-bounded");
        self.config_deadline.set(deadline);
        self.config_calls.set(self.config_calls.get() + 1);
        Ok(b"{}".to_vec())
    }
    fn compose_build(&self, _: &DockerComposeBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        Ok(())
    }
    fn compose_up(&self, _: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
        self.up_calls.set(self.up_calls.get() + 1);
        Ok(())
    }
    fn compose_exec(&self, _: &DockerComposeExecRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn compose_copy_archive(
        &self,
        _: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        Err(EvalExecutionError::ArtifactCollection(
            "unexpected archive".to_owned(),
        ))
    }
    fn compose_copy_archive_bounded(
        &self,
        _: &DockerComposeArchiveRequest,
        _: aiperf_runtime::eval::EvalExecutionPhase,
        _: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        Err(EvalExecutionError::ArtifactCollection(
            "unexpected bounded archive".to_owned(),
        ))
    }
    fn compose_copy_into(&self, _: &DockerComposeCopyRequest) -> Result<(), EvalExecutionError> {
        Err(EvalExecutionError::ArtifactCollection(
            "unexpected Compose copy".to_owned(),
        ))
    }
    fn compose_stop_service(&self, _: &DockerComposeStopRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn compose_stop_service_bounded(
        &self,
        _: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError> {
        Err(EvalExecutionError::ArtifactCollection(
            "unexpected bounded stop".to_owned(),
        ))
    }
    fn compose_down(&self, _: &DockerComposeDownRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn compose_owned_resources(
        &self,
        _: &ComposeProjectId,
        _: Duration,
    ) -> Result<OwnedComposeResources, EvalExecutionError> {
        Ok(OwnedComposeResources::default())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ComposeSessionFailure {
    Healthcheck,
    Hook,
    Archive,
    ArchiveTimeout,
    Stop,
    VerifierTimeout,
    VerifierCreateTimeoutLate,
    VerifierCreateFailure,
}

struct ComposeSessionRecordingRuntime {
    events: RefCell<Vec<String>>,
    creates: RefCell<Vec<RecordedCreate>>,
    artifact_transfer_sources: RefCell<Vec<String>>,
    generated_main: RefCell<Option<serde_json::Value>>,
    phase_calls: RefCell<Vec<(String, Option<String>, Option<String>, Vec<String>)>>,
    docker_phase_calls: RefCell<Vec<(String, Option<String>, Option<String>, Vec<String>)>>,
    failure: Option<ComposeSessionFailure>,
    agent_calls: Cell<usize>,
    removal_deadlines: RefCell<Vec<Option<Duration>>>,
    compose_down_deadlines: RefCell<Vec<Option<Duration>>>,
    advance_verifier_create_to: Option<(Rc<SimClock>, i64)>,
    advance_verifier_workdir_to: Option<(Rc<SimClock>, i64)>,
    late_verifier_present: Cell<bool>,
    late_verifier_remove_calls: Cell<usize>,
    late_verifier_create_resolved: Cell<bool>,
    late_verifier_create_clock: Option<Rc<SimClock>>,
    create_timeout_compensation_calls: Cell<usize>,
}

impl ComposeSessionRecordingRuntime {
    fn new(failure: Option<ComposeSessionFailure>) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            creates: RefCell::new(Vec::new()),
            artifact_transfer_sources: RefCell::new(Vec::new()),
            generated_main: RefCell::new(None),
            phase_calls: RefCell::new(Vec::new()),
            docker_phase_calls: RefCell::new(Vec::new()),
            failure,
            agent_calls: Cell::new(0),
            removal_deadlines: RefCell::new(Vec::new()),
            compose_down_deadlines: RefCell::new(Vec::new()),
            advance_verifier_create_to: None,
            advance_verifier_workdir_to: None,
            late_verifier_present: Cell::new(false),
            late_verifier_remove_calls: Cell::new(0),
            late_verifier_create_resolved: Cell::new(false),
            late_verifier_create_clock: None,
            create_timeout_compensation_calls: Cell::new(0),
        }
    }

    fn advance_after_verifier_create(mut self, clock: Rc<SimClock>, time_ns: i64) -> Self {
        self.advance_verifier_create_to = Some((clock, time_ns));
        self
    }

    fn advance_after_verifier_workdir(mut self, clock: Rc<SimClock>, time_ns: i64) -> Self {
        self.advance_verifier_workdir_to = Some((clock, time_ns));
        self
    }

    fn complete_late_verifier_create_after_absent(mut self, clock: Rc<SimClock>) -> Self {
        self.late_verifier_create_clock = Some(clock);
        self
    }
}

impl DockerRuntime for ComposeSessionRecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_public_network()
            .with_users()
            .with_workdir()
            .with_healthchecks()
            .with_phase_timeouts()
            .with_separate_verifier()
            .with_compose_project()
            .with_compose_config()
            .with_service_exec()
            .with_service_archive()
            .with_service_stop()
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "reward.txt",
            b"1\n",
        ))))
    }

    fn compose_runtime(&self) -> Option<&dyn DockerComposeRuntime> {
        Some(self)
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("docker-build".to_owned());
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        let arguments = request.public_arguments();
        let container = argument_after(arguments, "--name").to_owned();
        if container.contains("-verifier-")
            && self.failure == Some(ComposeSessionFailure::VerifierCreateTimeoutLate)
        {
            self.events
                .borrow_mut()
                .push(format!("create-timeout:{container}"));
            return Err(EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
                timeout: request.deadline().unwrap_or(Duration::from_secs(1)),
            });
        }
        if container.contains("-verifier-")
            && self.failure == Some(ComposeSessionFailure::VerifierCreateFailure)
        {
            return Err(EvalExecutionError::ProcessFailure(
                "deterministic verifier create failure".to_owned(),
            ));
        }
        if container.contains("-verifier-")
            && let Some((clock, time_ns)) = &self.advance_verifier_create_to
        {
            clock.advance_to(*time_ns);
        }
        let workspace = arguments
            .windows(2)
            .find(|pair| pair[0] == "--volume")
            .map(|pair| {
                pair[1]
                    .split_once(':')
                    .map_or_else(|| pair[1].clone(), |(host, _)| host.to_owned())
            });
        self.events.borrow_mut().push(format!("create:{container}"));
        self.creates.borrow_mut().push(RecordedCreate {
            container,
            workspace,
            arguments: arguments.to_vec(),
        });
        Ok(())
    }

    fn compensate_create_timeout(
        &self,
        request: &DockerCreateRequest,
        cleanup_deadline: Duration,
    ) -> Result<(), EvalExecutionError> {
        self.create_timeout_compensation_calls
            .set(self.create_timeout_compensation_calls.get() + 1);
        assert_eq!(cleanup_deadline, Duration::from_secs(10));
        assert!(matches!(
            request.creation_phase(),
            Some(aiperf_runtime::eval::EvalExecutionPhase::Verifier)
        ));
        let target = request
            .creation_target()
            .expect("bounded verifier create target");
        let removal = DockerRemoveRequest::new(["rm", "--force", "--volumes", target]);
        self.remove(&removal)?;
        self.remove(&removal)
    }

    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events
            .borrow_mut()
            .push(format!("start:{}", request.container()));
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        self.docker_phase_calls.borrow_mut().push((
            request.phase().to_string(),
            request.user().map(ToOwned::to_owned),
            request.workdir().map(ToOwned::to_owned),
            request.public_arguments().to_vec(),
        ));
        if request.phase() == aiperf_runtime::eval::EvalExecutionPhase::Healthcheck
            && self.failure == Some(ComposeSessionFailure::Healthcheck)
            && request.public_arguments() == ["ready".to_owned()]
        {
            return Err(EvalExecutionError::ProcessFailure(
                "healthcheck failed".to_owned(),
            ));
        }
        if request.phase() == aiperf_runtime::eval::EvalExecutionPhase::Verifier
            && self.failure == Some(ComposeSessionFailure::VerifierTimeout)
        {
            self.events.borrow_mut().push("verifier-timeout".to_owned());
            return Err(EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
                timeout: request.deadline().unwrap_or(Duration::from_secs(1)),
            });
        }
        self.events
            .borrow_mut()
            .push(format!("docker-exec:{}", request.phase()));
        Ok(())
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        let source = &request.public_arguments()[1];
        let destination = &request.public_arguments()[2];
        if source.ends_with("/reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward json absent".to_owned(),
            ));
        }
        if source.ends_with("/reward.txt") {
            fs::write(destination, "1\n")
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            return Ok(());
        }
        if !source.contains(':') && destination.contains(':') && !destination.ends_with(":/tests") {
            self.artifact_transfer_sources
                .borrow_mut()
                .push(source.to_owned());
        }
        self.events.borrow_mut().push("docker-copy".to_owned());
        Ok(())
    }

    fn container_workdir(&self, container: &str) -> Result<String, EvalExecutionError> {
        if container.contains("-verifier-")
            && let Some((clock, time_ns)) = &self.advance_verifier_workdir_to
        {
            clock.advance_to(*time_ns);
        }
        Ok("/work".to_owned())
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.removal_deadlines.borrow_mut().push(request.deadline());
        self.events.borrow_mut().push(format!(
            "remove:{}",
            request.public_arguments().last().unwrap_or(&String::new())
        ));
        if self.failure == Some(ComposeSessionFailure::VerifierTimeout)
            && request
                .public_arguments()
                .last()
                .is_some_and(|container| container.contains("-verifier-"))
        {
            return Err(EvalExecutionError::ProcessFailure(
                "verifier removal failed".to_owned(),
            ));
        }
        if self.failure == Some(ComposeSessionFailure::VerifierCreateTimeoutLate)
            && request
                .public_arguments()
                .last()
                .is_some_and(|container| container.contains("-verifier-"))
        {
            self.late_verifier_remove_calls
                .set(self.late_verifier_remove_calls.get() + 1);
            if self.late_verifier_create_resolved.get() {
                return Ok(());
            }
            if !self.late_verifier_present.replace(true) {
                if let Some(clock) = &self.late_verifier_create_clock {
                    clock.advance_to(clock.now_ns() + 1);
                }
                self.events.borrow_mut().push("remove-absent".to_owned());
                return Ok(());
            }
            self.late_verifier_present.set(false);
            self.late_verifier_create_resolved.set(true);
            self.events.borrow_mut().push("remove-late".to_owned());
        }
        Ok(())
    }
}

impl DockerComposeRuntime for ComposeSessionRecordingRuntime {
    fn compose_config(
        &self,
        request: &DockerComposeConfigRequest,
    ) -> Result<Vec<u8>, EvalExecutionError> {
        self.events.borrow_mut().push("compose-config".to_owned());
        assert!(request.interpolation_disabled());
        assert!(request.env_file_disabled());
        let mut generated =
            serde_yaml::from_slice::<serde_yaml::Value>(request.generated_definition())
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let generated_json = serde_json::to_value(&generated)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        *self.generated_main.borrow_mut() = generated_json
            .get("services")
            .and_then(serde_json::Value::as_object)
            .and_then(|services| services.get("main"))
            .cloned();
        let overlay = fs::read(request.overlay_definition())
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let overlay = serde_yaml::from_slice::<serde_yaml::Value>(&overlay)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        merge_compose_yaml(&mut generated, overlay);
        let mut canonical = serde_json::to_value(generated)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let services = canonical
            .get_mut("services")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or(EvalExecutionError::Materialization(
                "generated Compose services are missing".to_owned(),
            ))?;
        for service in services.values_mut() {
            let service = service
                .as_object_mut()
                .ok_or(EvalExecutionError::Materialization(
                    "canonical Compose service is not an object".to_owned(),
                ))?;
            let networks = service
                .get("networks")
                .and_then(serde_json::Value::as_array)
                .map(|networks| {
                    networks
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(|name| (name.to_owned(), serde_json::Value::Null))
                        .collect()
                })
                .unwrap_or_else(|| {
                    serde_json::Map::from_iter([(String::from("default"), serde_json::Value::Null)])
                });
            service.insert("networks".to_owned(), serde_json::Value::Object(networks));
        }
        serde_json::to_vec(&canonical)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))
    }

    fn compose_build(&self, _: &DockerComposeBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("compose-build".to_owned());
        Ok(())
    }

    fn compose_up(&self, _: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("compose-up".to_owned());
        Ok(())
    }

    fn compose_exec(&self, request: &DockerComposeExecRequest) -> Result<(), EvalExecutionError> {
        self.phase_calls.borrow_mut().push((
            request.phase().to_string(),
            request.user().map(ToOwned::to_owned),
            request.workdir().map(ToOwned::to_owned),
            request.public_arguments().to_vec(),
        ));
        if request.phase() == aiperf_runtime::eval::EvalExecutionPhase::Healthcheck
            && self.failure == Some(ComposeSessionFailure::Healthcheck)
            && request.public_arguments() == ["ready".to_owned()]
        {
            return Err(EvalExecutionError::ProcessFailure(
                "healthcheck failed".to_owned(),
            ));
        }
        match request.phase() {
            aiperf_runtime::eval::EvalExecutionPhase::Agent => {
                let call = self.agent_calls.get() + 1;
                self.agent_calls.set(call);
                self.events.borrow_mut().push(format!("agent:{call}"));
            }
            aiperf_runtime::eval::EvalExecutionPhase::CollectionHook => {
                self.events
                    .borrow_mut()
                    .push(format!("hook:{}", request.service().as_str()));
                if self.failure == Some(ComposeSessionFailure::Hook) {
                    return Err(EvalExecutionError::ProcessFailure("hook failed".to_owned()));
                }
            }
            aiperf_runtime::eval::EvalExecutionPhase::Verifier => {
                self.events.borrow_mut().push("shared-verifier".to_owned());
            }
            phase => self
                .events
                .borrow_mut()
                .push(format!("compose-exec:{phase}")),
        }
        Ok(())
    }

    fn compose_copy_archive(
        &self,
        request: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.compose_copy_archive_bounded(
            request,
            aiperf_runtime::eval::EvalExecutionPhase::CollectionHook,
            Duration::from_secs(1),
        )
    }

    fn compose_copy_archive_bounded(
        &self,
        request: &DockerComposeArchiveRequest,
        _: aiperf_runtime::eval::EvalExecutionPhase,
        _: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.events
            .borrow_mut()
            .push(format!("archive:{}", request.service().as_str()));
        if self.failure == Some(ComposeSessionFailure::Archive) {
            return Err(EvalExecutionError::ProcessFailure(
                "archive failed".to_owned(),
            ));
        }
        if self.failure == Some(ComposeSessionFailure::ArchiveTimeout) {
            return Err(EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::CollectionHook,
                timeout: Duration::from_secs(1),
            });
        }
        let path = if request.source().starts_with("/work/") {
            "result.txt"
        } else {
            "result.txt/payload"
        };
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            path,
            b"compose snapshot",
        ))))
    }

    fn compose_copy_into(&self, _: &DockerComposeCopyRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("compose-copy".to_owned());
        Ok(())
    }

    fn compose_stop_service(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError> {
        self.compose_stop_service_bounded(request)
    }

    fn compose_stop_service_bounded(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError> {
        self.events
            .borrow_mut()
            .push(format!("stop:{}", request.service().as_str()));
        if self.failure == Some(ComposeSessionFailure::Stop) {
            return Err(EvalExecutionError::ProcessFailure("stop failed".to_owned()));
        }
        Ok(())
    }

    fn compose_down(&self, request: &DockerComposeDownRequest) -> Result<(), EvalExecutionError> {
        self.compose_down_deadlines
            .borrow_mut()
            .push(request.deadline());
        self.events.borrow_mut().push("compose-down".to_owned());
        Ok(())
    }

    fn compose_owned_resources(
        &self,
        _: &ComposeProjectId,
        _: Duration,
    ) -> Result<OwnedComposeResources, EvalExecutionError> {
        Ok(OwnedComposeResources::default())
    }
}

fn merge_compose_yaml(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    let (Some(base), Some(overlay)) = (base.as_mapping_mut(), overlay.as_mapping()) else {
        return;
    };
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(base_value), serde_yaml::Value::Mapping(overlay_mapping)) => {
                merge_compose_yaml(
                    base_value,
                    serde_yaml::Value::Mapping(overlay_mapping.clone()),
                );
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

#[test]
fn compose_multi_step_session_keeps_one_project_and_fresh_verifiers() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(None);

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect("Compose multi-step execution");

    assert_eq!(result.steps.len(), 2);
    assert_eq!(runtime.agent_calls.get(), 2);
    let events = runtime.events.borrow();
    assert_eq!(
        events
            .iter()
            .filter(|event| *event == "compose-build")
            .count(),
        1
    );
    assert_eq!(
        events.iter().filter(|event| *event == "compose-up").count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| *event == "compose-down")
            .count(),
        1
    );
    let first_verifier = events
        .iter()
        .position(|event| event.starts_with("create:") && event.contains("verifier-one"))
        .expect("first verifier create");
    let second_agent = events
        .iter()
        .position(|event| event == "agent:2")
        .expect("second agent");
    let project_down = events
        .iter()
        .position(|event| event == "compose-down")
        .expect("project teardown");
    assert!(first_verifier < second_agent);
    assert!(
        second_agent < project_down,
        "the project remains live for the next step"
    );
    let snapshots = runtime.artifact_transfer_sources.borrow();
    assert_eq!(snapshots.len(), 2);
    assert_ne!(snapshots[0], snapshots[1]);
}

#[test]
fn compose_terminal_sidecar_evidence_tears_down_before_separate_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_terminal_evidence_task_root(&temporary, false, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(None);

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect("terminal Compose evidence execution");

    let events = runtime.events.borrow();
    let down = events
        .iter()
        .position(|event| event == "compose-down")
        .expect("sidecar project teardown");
    let verifier = events
        .iter()
        .position(|event| event.starts_with("create:") && event.contains("verifier-"))
        .expect("separate verifier create");
    assert!(
        down < verifier,
        "sidecar evidence must be terminal before verifier creation"
    );
}

#[test]
fn compose_terminal_sidecar_teardown_uses_the_separate_verifier_deadline() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_terminal_evidence_task_root(&temporary, false, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(None);

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect("terminal Compose evidence execution");

    let deadlines = runtime.compose_down_deadlines.borrow();
    assert!(
        deadlines
            .first()
            .and_then(|deadline| *deadline)
            .is_some_and(|deadline| deadline <= Duration::from_secs(1)),
        "sidecar teardown must consume the verifier deadline: {deadlines:?}"
    );
}

#[test]
fn compose_recipe_workdir_override_prepares_nonroot_main_workdir_before_health_and_agent_without_mutating_plan()
 {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment]
workdir = "/task"
user = "bench"

[environment.healthcheck]
command = ["ready"]

[verifier]
environment_mode = "separate"
"#,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(None);
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/override".to_owned()),
    )
    .unwrap();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect("Compose execution with runtime workdir override");

    assert_eq!(
        imported.package.execution_plan().environment().workdir(),
        Some("/task"),
        "runtime overrides must not mutate normalized package identity"
    );
    let generated = runtime
        .generated_main
        .borrow()
        .clone()
        .expect("generated main definition");
    assert_eq!(
        generated
            .get("working_dir")
            .and_then(serde_json::Value::as_str),
        Some("/override")
    );
    assert_eq!(
        generated
            .get("volumes")
            .and_then(serde_json::Value::as_array)
            .and_then(|mounts| mounts.first())
            .and_then(serde_json::Value::as_object)
            .and_then(|mount| mount.get("target"))
            .and_then(serde_json::Value::as_str),
        Some("/override")
    );
    let calls = runtime.phase_calls.borrow();
    assert!(calls.iter().any(|(phase, user, workdir, command)| {
        phase == "healthcheck"
            && user.as_deref() == Some("bench")
            && workdir.as_deref() == Some("/override")
            && command.as_slice() == ["ready".to_owned()]
    }));
    assert!(calls.iter().any(|(phase, user, workdir, command)| {
        phase == "healthcheck"
            && user.as_deref() == Some("root")
            && workdir.as_deref() == Some("/override")
            && command
                .first()
                .is_some_and(|argument| argument == "/bin/sh")
    }));
    assert!(calls.iter().any(|(phase, user, workdir, command)| {
        phase == "agent"
            && user.as_deref() == Some("bench")
            && workdir.as_deref() == Some("/override")
            && command.as_slice() == ["agent".to_owned()]
    }));
    let preparation = calls
        .iter()
        .position(|(phase, user, workdir, command)| {
            phase == "healthcheck"
                && user.as_deref() == Some("root")
                && workdir.as_deref() == Some("/override")
                && command
                    .first()
                    .is_some_and(|argument| argument == "/bin/sh")
        })
        .expect("root workdir preparation");
    let healthcheck = calls
        .iter()
        .position(|(phase, user, workdir, command)| {
            phase == "healthcheck"
                && user.as_deref() == Some("bench")
                && workdir.as_deref() == Some("/override")
                && command.as_slice() == ["ready".to_owned()]
        })
        .expect("healthcheck execution");
    let agent = calls
        .iter()
        .position(|(phase, user, workdir, command)| {
            phase == "agent"
                && user.as_deref() == Some("bench")
                && workdir.as_deref() == Some("/override")
                && command.as_slice() == ["agent".to_owned()]
        })
        .expect("agent execution");
    assert!(
        preparation < healthcheck && healthcheck < agent,
        "a non-root healthcheck must execute in its prepared workdir before the agent"
    );
    let docker_calls = runtime.docker_phase_calls.borrow();
    let verifier_preparation = docker_calls
        .iter()
        .enumerate()
        .find_map(|(index, (phase, user, workdir, command))| {
            (phase == "verifier"
                && user.as_deref() == Some("root")
                && workdir.as_deref() == Some("/override")
                && command
                    .first()
                    .is_some_and(|argument| argument == "/bin/sh"))
            .then_some(index)
        })
        .expect("separate verifier root workdir preparation");
    let verifier_healthcheck = docker_calls
        .iter()
        .position(|(phase, user, workdir, command)| {
            phase == "healthcheck"
                && user.as_deref() == Some("bench")
                && workdir.as_deref() == Some("/override")
                && command.as_slice() == ["ready".to_owned()]
        })
        .expect("separate verifier inherited healthcheck");
    assert!(
        verifier_preparation < verifier_healthcheck,
        "a separate Compose verifier must prepare its non-root workdir before its inherited healthcheck"
    );
}

#[test]
fn compose_workdir_preparation_passes_metacharacters_as_literal_operands() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment]
user = "bench"

[verifier]
environment_mode = "separate"
"#,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(None);
    let workdir = "/workspace; touch /escape";
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some(workdir.to_owned()),
    )
    .unwrap();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect("Compose execution with a literal metacharacter workdir");

    let calls = runtime.phase_calls.borrow();
    let preparations = calls
        .iter()
        .filter(|(phase, user, _, _)| phase == "agent" && user.as_deref() == Some("root"))
        .map(|(_, _, _, command)| command.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        preparations,
        vec![vec![
            "/bin/sh".to_owned(),
            "-ec".to_owned(),
            "mkdir -p -- \"$1\"\nchown -- \"$2\" \"$1\"\nexec su -s /bin/sh -c 'test -w \"$0\"' -- \"$2\" \"$1\""
                .to_owned(),
            "--".to_owned(),
            workdir.to_owned(),
            "bench".to_owned(),
        ]],
        "the workdir must remain an argv operand rather than root shell source"
    );
}

#[test]
fn compose_healthcheck_failure_still_prevents_agent_after_nonroot_workdir_preparation() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment]
workdir = "/task"
user = "bench"

[environment.healthcheck]
command = ["ready"]
retries = 1
"#,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::Healthcheck));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("unhealthy Compose project must not run the agent");

    assert!(matches!(error, EvalExecutionError::Unhealthy(_)));
    let calls = runtime.phase_calls.borrow();
    assert!(calls.iter().any(|(phase, user, workdir, command)| {
        phase == "healthcheck"
            && user.as_deref() == Some("root")
            && workdir.as_deref() == Some("/work")
            && command
                .first()
                .is_some_and(|argument| argument == "/bin/sh")
    }));
    assert!(calls.iter().any(|(phase, user, workdir, command)| {
        phase == "healthcheck"
            && user.as_deref() == Some("bench")
            && workdir.as_deref() == Some("/work")
            && command.as_slice() == ["ready".to_owned()]
    }));
    assert!(!calls.iter().any(|(phase, _, _, command)| {
        phase == "agent" && command.as_slice() == ["agent".to_owned()]
    }));
}

#[test]
fn compose_separate_verifier_healthcheck_failure_prevents_verifier_command_and_cleans_up() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment]
workdir = "/task"
user = "bench"

[verifier]
environment_mode = "separate"

[verifier.environment]
workdir = "/task"
user = "bench"

[verifier.environment.healthcheck]
command = ["ready"]
retries = 1
"#,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert!(
        imported
            .package
            .execution_plan()
            .verifier()
            .environment()
            .healthcheck()
            .is_some(),
        "test task must give the separate verifier its own healthcheck"
    );
    let runtime = ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::Healthcheck));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("unhealthy separate Compose verifier must stop before its command");

    assert!(matches!(error, EvalExecutionError::Unhealthy(_)));
    let docker_calls = runtime.docker_phase_calls.borrow();
    assert!(docker_calls.iter().any(|(phase, user, workdir, command)| {
        phase == "verifier"
            && user.as_deref() == Some("root")
            && workdir.as_deref() == Some("/work")
            && command
                .first()
                .is_some_and(|argument| argument == "/bin/sh")
    }));
    assert!(!docker_calls.iter().any(|(phase, _, _, command)| {
        phase == "verifier"
            && command.as_slice() == ["/bin/sh".to_owned(), "/tests/test.sh".to_owned()]
    }));
    let events = runtime.events.borrow();
    assert!(events.iter().any(|event| event.contains("verifier-")));
    assert!(events.iter().any(|event| event == "compose-down"));
}

#[test]
fn compose_evidence_failures_never_create_a_verifier_and_always_teardown() {
    for (failure, expected) in [
        (ComposeSessionFailure::Hook, "hook:api"),
        (ComposeSessionFailure::Archive, "archive:api"),
        (ComposeSessionFailure::Stop, "stop:main"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = compose_terminal_evidence_task_root(&temporary, true, false);
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let runtime = ComposeSessionRecordingRuntime::new(Some(failure));

        let error = DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &compose_recipe(),
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret,
            )
            .expect_err("collection failure must stop before verifier provisioning");

        assert!(matches!(
            error,
            EvalExecutionError::CollectionHook { .. } | EvalExecutionError::ArtifactCollection(_)
        ));
        let events = runtime.events.borrow();
        assert!(events.iter().any(|event| event == expected));
        assert!(events.iter().any(|event| event == "compose-down"));
        assert!(events.iter().all(|event| !event.starts_with("create:")));
    }
}

#[test]
fn compose_archive_process_failure_is_an_artifact_collection_error_with_context() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_terminal_evidence_task_root(&temporary, false, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::Archive));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a missing Compose archive must prevent verifier provisioning");

    assert!(matches!(
        error,
        EvalExecutionError::ArtifactCollection(ref reason)
            if reason.contains("api")
                && reason.contains("/var/lib/api/result.txt")
                && reason.contains("archive failed")
    ));
    let events = runtime.events.borrow();
    assert!(events.iter().any(|event| event == "archive:api"));
    assert!(events.iter().any(|event| event == "compose-down"));
    assert!(events.iter().all(|event| !event.starts_with("create:")));
}

#[test]
fn compose_archive_timeout_keeps_the_collection_hook_phase() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_terminal_evidence_task_root(&temporary, false, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::ArchiveTimeout));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a Compose archive timeout must prevent verifier provisioning");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::CollectionHook,
            timeout,
        } if timeout == Duration::from_secs(1)
    ));
}

#[test]
fn compose_verifier_timeout_removes_verifier_before_project_teardown() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_main_evidence_timeout_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::VerifierTimeout));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("verifier timeout");

    assert!(
        matches!(error, EvalExecutionError::ContainerTeardown { .. }),
        "verifier cleanup failure must not be silently discarded: {error:?}"
    );
    let events = runtime.events.borrow();
    let remove = events
        .iter()
        .position(|event| event.starts_with("remove:") && event.contains("verifier-"))
        .expect("verifier removal");
    let down = events
        .iter()
        .position(|event| event == "compose-down")
        .expect("project teardown");
    assert!(remove < down);
}

#[test]
fn compose_late_verifier_create_is_compensated_before_project_teardown() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_main_evidence_timeout_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime =
        ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::VerifierCreateTimeoutLate))
            .complete_late_verifier_create_after_absent(clock.clone());

    let error = DockerProcessSandbox::with_clock(clock.clone())
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a verifier create timeout must remain a verifier failure");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    assert_eq!(runtime.late_verifier_remove_calls.get(), 3);
    assert_eq!(clock.now_ns(), 1);
    assert!(
        !runtime.late_verifier_present.get(),
        "the late verifier create must be removed before evaluation returns"
    );
    let events = runtime.events.borrow();
    let remove = events
        .iter()
        .position(|event| event == "remove-late")
        .expect("late verifier removal");
    let down = events
        .iter()
        .position(|event| event == "compose-down")
        .expect("project teardown");
    assert!(remove < down);
}

#[test]
fn compose_deterministic_verifier_create_failure_does_not_compensate() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_main_evidence_timeout_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime =
        ComposeSessionRecordingRuntime::new(Some(ComposeSessionFailure::VerifierCreateFailure));

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("deterministic verifier creation failure");

    assert!(matches!(
        error,
        EvalExecutionError::ProcessFailure(ref reason)
            if reason == "deterministic verifier create failure"
    ));
    assert_eq!(runtime.create_timeout_compensation_calls.get(), 0);
}

#[test]
fn compose_verifier_deadline_exhausted_during_isolation_skips_later_setup_and_reward() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_main_evidence_timeout_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime = ComposeSessionRecordingRuntime::new(None)
        .advance_after_verifier_create(clock.clone(), Duration::from_secs(1).as_nanos() as i64);

    let error = DockerProcessSandbox::with_clock(clock)
        .execute_with_runtime(
            &runtime,
            &compose_recipe(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("an exhausted verifier deadline must stop before verifier startup");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    let events = runtime.events.borrow();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("create:") && event.contains("verifier"))
    );
    assert!(
        events
            .iter()
            .all(|event| !event.starts_with("start:") || !event.contains("verifier"))
    );
    assert!(events.iter().all(|event| event != "docker-copy"));
}

#[test]
fn compose_verifier_deadline_exhausted_by_workdir_inspection_skips_transfer_and_test() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_main_evidence_timeout_task_root(&temporary);
    fs::write(
        task_root.join("task.toml"),
        format!(
            "{}\n[verifier.environment.healthcheck]\ncommand = [\"ready\"]\nstart_period_sec = 1\nretries = 1\n",
            fs::read_to_string(task_root.join("task.toml")).unwrap()
        ),
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime = ComposeSessionRecordingRuntime::new(None)
        .advance_after_verifier_workdir(clock.clone(), Duration::from_secs(1).as_nanos() as i64);

    let error = DockerProcessSandbox::with_clock(clock)
        .execute_with_runtime(
            &runtime,
            &HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .unwrap(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("workdir inspection must consume the verifier envelope");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    let events = runtime.events.borrow();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("start:") && event.contains("verifier"))
    );
    assert!(
        events.iter().all(|event| event != "docker-copy"),
        "unexpected transfer after inspection deadline: {events:?}"
    );
    assert!(
        runtime
            .docker_phase_calls
            .borrow()
            .iter()
            .all(|(phase, _, _, _)| phase != "verifier")
    );
    assert!(
        runtime
            .docker_phase_calls
            .borrow()
            .iter()
            .all(|(phase, _, _, _)| phase != "healthcheck")
    );
    assert!(
        runtime
            .removal_deadlines
            .borrow()
            .iter()
            .all(Option::is_some)
    );
}

#[test]
fn compose_preflight_runs_only_read_only_configuration_before_lifecycle() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
build_timeout_sec = 3
startup_timeout_sec = 2
"#,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ComposePreflightRuntime::default();
    let result = DockerProcessSandbox::new().execute_with_runtime(
        &runtime,
        &HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap(),
        &imported.package,
        imported.package.execution_plan(),
        &["true".to_owned()],
        &FixedSecret,
    );
    assert!(result.is_err(), "{result:?}");
    assert_eq!(runtime.config_calls.get(), 1);
    assert_eq!(runtime.config_deadline.get(), Duration::from_secs(2));
    assert_eq!(runtime.build_calls.get(), 0);
    assert_eq!(runtime.up_calls.get(), 0);
}

#[test]
fn planned_lifecycle_preflights_before_build_and_health_before_agent() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
workdir = "/task"
user = "bench"
network = "no-network"

[environment.env]
BASE = "baseline"

[environment.healthcheck]
command = ["true"]
start_period_sec = 0.05
start_interval_sec = 0.1
interval_sec = 0.2
timeout_sec = 0.3
retries = 1

[agent]
user = "agent"
network = "public"

[agent.env]
PHASE = "agent"

[verifier]
user = "verifier"
network = "no-network"

[verifier.env]
PHASE = "verifier"
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/ignored-by-plan",
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["true".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        runtime.events.borrow().clone(),
        vec![
            "preflight",
            "build:none",
            "create:none",
            "start",
            "healthcheck:bench:/task:none:BASE=baseline",
            "prepare:root:/task:none",
            "agent:agent:/task:aiperf-eval-public:BASE=baseline,PHASE=agent",
            "prepare-verifier-files",
            "copy-tests",
            "prepare:root:/task:none",
            "verifier:verifier:/task:none:BASE=baseline,PHASE=verifier",
            "archive-reward",
            "archive-reward",
            "remove",
        ]
    );
}

#[test]
fn cli_recipe_workdir_overrides_the_manifest_without_mutating_the_plan() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[environment]\nworkdir = \"/manifest-work\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/cli-work".to_owned()),
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["true".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        imported.package.execution_plan().environment().workdir(),
        Some("/manifest-work")
    );
    assert!(
        runtime
            .events
            .into_inner()
            .iter()
            .any(|event| event == "agent:root:/cli-work:aiperf-eval-public:"),
        "the explicit CLI workdir must be applied only at runtime"
    );
}

#[test]
fn shared_cli_workdir_is_rejected_before_implicit_step_build() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/tests".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a CLI workdir cannot occupy shared verifier paths");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("shared verifier workdir")
    ));
    assert_eq!(runtime.build_calls.get(), 0);
    assert!(runtime.creates.borrow().is_empty());
}

#[test]
fn shared_cli_workdir_is_rejected_before_mixed_plan_build() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
[task]
name = "example/mixed-workdir"
[[steps]]
name = "one"
[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/logs/verifier/nested".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("one shared step reserves the persistent agent workdir");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("shared verifier workdir")
    ));
    assert_eq!(runtime.build_calls.get(), 0);
    assert!(runtime.creates.borrow().is_empty());
}

#[test]
fn shared_image_workdir_is_rejected_after_implicit_step_start() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::with_image_workdir("/logs/verifier");

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the image workdir cannot occupy shared verifier paths");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("shared verifier workdir")
    ));
    assert_eq!(runtime.build_calls.get(), 1);
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert_eq!(runtime.removals.get(), 1);
    let events = runtime.events.borrow().clone();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("inspect-workdir:"))
    );
    assert!(events.iter().all(|event| {
        !event.starts_with("agent:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
fn shared_image_workdir_is_rejected_after_explicit_multi_step_start() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::with_image_workdir("/tests/nested");

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the image workdir cannot occupy shared verifier paths");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("shared verifier workdir")
    ));
    assert_eq!(runtime.build_calls.get(), 1);
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert_eq!(runtime.removals.get(), 1);
    let events = runtime.events.borrow().clone();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("inspect-workdir:"))
    );
    assert!(events.iter().all(|event| {
        !event.starts_with("agent:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
fn each_execution_uses_distinct_image_and_container_names() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    for _ in 0..2 {
        DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["true".to_owned()],
                &FixedSecret,
            )
            .unwrap();
    }

    let names = runtime.names.into_inner();
    assert_eq!(names.len(), 4);
    assert_ne!(names[0], names[2]);
    assert_ne!(names[1], names[3]);
    assert!(names[0].starts_with("aiperf-eval:"));
    assert!(names[1].starts_with("aiperf-eval-"));
}

#[test]
fn unhealthy_readiness_retries_then_prevents_agent_and_cleans_up() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
network = "no-network"

[environment.healthcheck]
command = ["false"]
start_period_sec = 0.05
start_interval_sec = 0.1
interval_sec = 0.2
timeout_sec = 0.3
retries = 3
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = UnhealthyRuntime::default();
    let clock = Rc::new(SimClock::new());

    let error = DockerProcessSandbox::with_clock(clock.clone())
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent-must-not-run".to_owned()],
            &FixedSecret,
        )
        .expect_err("exhausted readiness must stop the lifecycle");

    assert!(matches!(error, EvalExecutionError::Unhealthy(_)));
    assert_eq!(
        runtime.events.borrow().clone(),
        vec![
            "preflight",
            "build",
            "create",
            "start",
            "health:1:300",
            "health:2:300",
            "health:3:300",
            "remove",
        ]
    );
    assert_eq!(clock.now_ns(), 350_000_000);
}

#[derive(Default)]
struct UnhealthyRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for UnhealthyRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "healthcheck");
        let count = self
            .events
            .borrow()
            .iter()
            .filter(|event| event.starts_with("health:"))
            .count()
            + 1;
        self.events.borrow_mut().push(format!(
            "health:{count}:{}",
            request.deadline().expect("health deadline").as_millis()
        ));
        Err(EvalExecutionError::ProcessFailure("not ready".to_owned()))
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("unhealthy environment must not copy verifier files")
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(
            &request.public_arguments()[..3],
            ["rm", "--force", "--volumes"]
        );
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn agent_terminal_errors_prevent_verifier_and_remove_the_container() {
    for error in [
        EvalExecutionError::ProcessFailure("agent failed".to_owned()),
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
            timeout: std::time::Duration::from_secs(1),
        },
        EvalExecutionError::TerminalUncertainty {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
            container: "task-container".to_owned(),
            reason: "docker client lost".to_owned(),
        },
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, "");
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();
        let runtime = AgentTerminalRuntime::new(error.clone());

        assert_eq!(
            DockerProcessSandbox::new()
                .execute_with_runtime(
                    &runtime,
                    &recipe,
                    &imported.package,
                    imported.package.execution_plan(),
                    &["agent-must-fail".to_owned()],
                    &FixedSecret,
                )
                .expect_err("a terminal agent error must stop the lifecycle"),
            error
        );
        assert_eq!(
            runtime.events.borrow().clone(),
            vec!["preflight", "build", "create", "start", "agent", "remove"]
        );
    }
}

#[test]
fn artifact_collection_failure_prevents_separate_verifier_setup() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/missing-result.txt\"]",
        "[verifier]\nenvironment_mode = \"separate\"\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = ArtifactFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("artifact collection must terminate before verifier setup");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
    assert_eq!(
        runtime.events.borrow().clone(),
        vec!["build", "create", "start", "agent", "collect", "remove"]
    );
}

#[derive(Default)]
struct ArtifactFailureRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for ArtifactFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("artifact failure must prevent verifier files and reward copies")
    }

    fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        panic!("ordinary Docker artifact collection must use the bounded archive path")
    }

    fn copy_archive_bounded(
        &self,
        _: &str,
        _: &str,
        deadline: std::time::Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        assert!(!deadline.is_zero());
        self.events.borrow_mut().push("collect".to_owned());
        Err(EvalExecutionError::ArtifactCollection(
            "declared source is absent".to_owned(),
        ))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

struct AgentTerminalRuntime {
    events: RefCell<Vec<String>>,
    error: EvalExecutionError,
}

impl AgentTerminalRuntime {
    fn new(error: EvalExecutionError) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            error,
        }
    }
}

impl DockerRuntime for AgentTerminalRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Err(self.error.clone())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("terminal agent errors must not copy verifier files")
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn separate_verifier_failure_removes_both_container_leases() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[verifier]\nenvironment_mode = \"separate\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = VerifierFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a failed verifier must be terminal");

    assert!(matches!(error, EvalExecutionError::ProcessFailure(_)));
    assert_eq!(
        runtime.events.borrow().clone(),
        vec![
            "preflight",
            "build",
            "create",
            "start",
            "agent",
            "create",
            "start",
            "prepare-verifier-files",
            "copy-tests",
            "verifier",
            "remove",
            "remove",
        ]
    );
}

#[test]
fn separate_verifier_start_failure_removes_registered_verifier_lease() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[verifier]\nenvironment_mode = \"separate\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = SeparateStartFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("failed verifier start must be terminal");
    assert!(matches!(
        error,
        EvalExecutionError::ContainerTeardown { reason, .. }
            if reason.contains("verifier start failed")
                && reason.contains("verifier removal failed")
    ));
    assert_eq!(
        runtime.events.borrow().clone(),
        vec![
            "build", "create", "start", "agent", "create", "start", "remove", "remove"
        ]
    );
}

#[derive(Default)]
struct SeparateStartFailureRuntime {
    events: RefCell<Vec<String>>,
    starts: Cell<u8>,
    removals: Cell<u8>,
}

impl DockerRuntime for SeparateStartFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }
    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }
    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }
    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        self.starts.set(self.starts.get() + 1);
        if self.starts.get() == 2 {
            Err(EvalExecutionError::ProcessFailure(
                "verifier start failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Ok(())
    }
    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("verifier start failure must precede file copy")
    }
    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        self.removals.set(self.removals.get() + 1);
        if self.removals.get() == 1 {
            Err(EvalExecutionError::ProcessFailure(
                "verifier removal failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_verifier_health_failure_prevents_verifier_files_and_cleans_leases() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        "[verifier]\nenvironment_mode = \"separate\"\n[verifier.environment]\nworkdir = \"/verify-work\"\nuser = \"bench\"\n[verifier.environment.healthcheck]\ncommand = [\"false\"]\ntimeout_sec = 1\nretries = 1\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = SeparateHealthFailureRuntime::default();

    assert!(matches!(
        DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret
            )
            .expect_err("unhealthy separate verifier must stop before verifier files"),
        EvalExecutionError::Unhealthy(_)
    ));
    assert_eq!(
        runtime.events.borrow().clone(),
        vec![
            "build",
            "create",
            "start",
            "agent",
            "create",
            "start",
            "prepare:verifier",
            "healthcheck",
            "remove",
            "remove"
        ]
    );
}

#[derive(Default)]
struct SeparateHealthFailureRuntime {
    events: RefCell<Vec<String>>,
}
impl DockerRuntime for SeparateHealthFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_healthchecks()
            .with_users()
            .with_workdir()
            .with_no_network()
            .with_public_network()
    }
    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }
    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }
    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }
    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("mkdir -p"))
            && request
                .public_arguments()
                .first()
                .is_some_and(|argument| argument == "/bin/sh")
        {
            assert_eq!(request.phase().to_string(), "verifier");
            assert_eq!(request.user(), Some("root"), "{request:?}");
            assert_eq!(request.workdir(), Some("/verify-work"));
            self.events.borrow_mut().push("prepare:verifier".to_owned());
            return Ok(());
        }
        let phase = request.phase().to_string();
        self.events.borrow_mut().push(phase.clone());
        if phase == "healthcheck" {
            Err(EvalExecutionError::ProcessFailure("not ready".to_owned()))
        } else {
            Ok(())
        }
    }
    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("unhealthy verifier must not receive verifier files")
    }
    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct VerifierFailureRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for VerifierFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("rm -rf /tests /logs/verifier"))
        {
            assert_eq!(request.user(), Some("root"));
            self.events
                .borrow_mut()
                .push("prepare-verifier-files".to_owned());
            return Ok(());
        }
        match request.phase().to_string().as_str() {
            "agent" => {
                self.events.borrow_mut().push("agent".to_owned());
                Ok(())
            }
            "verifier" => {
                self.events.borrow_mut().push("verifier".to_owned());
                Err(EvalExecutionError::ProcessFailure(
                    "verifier failed".to_owned(),
                ))
            }
            phase => panic!("unexpected {phase} phase"),
        }
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        assert!(request.public_arguments()[1].contains("/tests"));
        self.events.borrow_mut().push("copy-tests".to_owned());
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct LifecycleRuntime {
    events: RefCell<Vec<String>>,
    names: RefCell<Vec<String>>,
}

impl DockerRuntime for LifecycleRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_users()
            .with_phase_env()
            .with_workdir()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
    }

    fn supports_phase_network_transitions(&self) -> bool {
        true
    }

    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        assert!(request.public_arguments().windows(2).any(|arguments| {
            arguments[0] == "--network" && Some(arguments[1].as_str()) == request.network_lease()
        }));
        self.events.borrow_mut().push(format!(
            "build:{}",
            request.network_lease().expect("build network")
        ));
        self.names.borrow_mut().push(
            request
                .public_arguments()
                .windows(2)
                .find(|arguments| arguments[0] == "--tag")
                .expect("image tag")[1]
                .clone(),
        );
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push(format!(
            "create:{}",
            request.network_lease().expect("container network")
        ));
        self.names.borrow_mut().push(
            request
                .public_arguments()
                .windows(2)
                .find(|arguments| arguments[0] == "--name")
                .expect("container name")[1]
                .clone(),
        );
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            self.events.borrow_mut().push("archive-reward".to_owned());
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        self.events.borrow_mut().push("archive-reward".to_owned());
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "reward.txt",
            b"1\n",
        ))))
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("rm -rf /tests /logs/verifier"))
        {
            assert_eq!(request.user(), Some("root"));
            assert!(request.public_arguments().iter().any(|argument| {
                argument.contains("mkdir -p /logs/verifier")
                    && argument.contains("chmod 0777 /logs/verifier")
            }));
            self.events
                .borrow_mut()
                .push("prepare-verifier-files".to_owned());
            return Ok(());
        }
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("mkdir -p"))
        {
            self.events.borrow_mut().push(format!(
                "prepare:root:{}:{}",
                request.workdir().unwrap_or("<image-workdir>"),
                request.network_lease(),
            ));
            return Ok(());
        }
        let environment = request
            .public_environment()
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        self.events.borrow_mut().push(format!(
            "{}:{}:{}:{}:{}",
            request.phase(),
            request.user().unwrap_or("root"),
            request.workdir().unwrap_or("<image-workdir>"),
            request.network_lease(),
            environment,
        ));
        Ok(())
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(
            request.public_arguments().first().map(String::as_str),
            Some("cp")
        );
        let event = if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("/tests"))
        {
            "copy-tests"
        } else {
            "copy-reward"
        };
        self.events.borrow_mut().push(event.to_owned());
        if let Some(destination) = request.public_arguments().last() {
            if destination.ends_with("reward.txt") {
                fs::write(destination, "1\n").unwrap();
            }
        }
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn unsupported_plan_is_rejected_before_a_docker_build_is_possible() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[environment]\ncpus = 1\nmemory_mb = 512\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = RecordingRuntime::default();

    assert_eq!(
        preflight_docker(&runtime, imported.package.execution_plan()),
        Err(EvalExecutionError::UnsupportedEnforcement("docker"))
    );
    assert_eq!(runtime.build_calls.get(), 0);
}

#[test]
fn legacy_json_package_runs_in_the_recipe_image_without_a_docker_build() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"legacy","instruction":"Fix it","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["agent"],"verifier_command":["verify"],"declared_artifacts":[]}"#,
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(package_path.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "/work",
    )
    .unwrap();
    let runtime = LegacyRuntime::default();

    let result = DockerProcessSandbox::new().execute_with_runtime(
        &runtime,
        &recipe,
        &imported.package,
        imported.package.execution_plan(),
        &["agent".to_owned()],
        &FixedSecret,
    );

    assert!(result.is_ok(), "legacy Docker execution: {result:?}");
    assert_eq!(
        runtime.images.into_inner(),
        vec![
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
        ]
    );
    let events = runtime.events.borrow().clone();
    assert!(!events.contains(&"build".to_owned()));
    assert!(events.iter().filter(|event| *event == "agent").count() == 1);
    assert!(events.iter().filter(|event| *event == "verifier").count() >= 2);
}

struct FixedSecret;

impl SecretProvider for FixedSecret {
    fn resolve(&self, _: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        Ok(SecretValue::new("unrenderable-secret".to_owned()))
    }
}

#[test]
fn docker_exec_request_redacts_secret_environment_values() {
    let request = DockerExecRequest::new(
        "task-container",
        ["/bin/sh", "-c", "true"],
        BTreeMap::from([("VISIBLE".to_owned(), "value".to_owned())]),
        BTreeMap::from([("TOKEN".to_owned(), SecretValue::new("unrenderable-secret"))]),
    );

    let rendering = format!("{request:?}");
    assert!(rendering.contains("VISIBLE"));
    assert!(rendering.contains("TOKEN"));
    assert!(!rendering.contains("unrenderable-secret"));
    assert_eq!(
        format!("{}", FixedSecret.resolve(&"TOKEN".to_owned()).unwrap()),
        "[REDACTED]"
    );
}

#[test]
fn multi_step_session_keeps_one_agent_and_injects_only_the_current_instruction() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(runtime.build_calls.get(), 1);
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert_eq!(runtime.verifier_execs.get(), 2);
    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        runtime.agent_environments.into_inner(),
        vec![
            BTreeMap::from([(
                "AIPERF_EVAL_INSTRUCTION".to_owned(),
                "First instruction.\n".to_owned(),
            )]),
            BTreeMap::from([(
                "AIPERF_EVAL_INSTRUCTION".to_owned(),
                "Second instruction.\n".to_owned(),
            )]),
        ]
    );
    assert!(
        runtime.creates.into_inner()[0]
            .arguments
            .iter()
            .all(|argument| !argument.contains("AIPERF_EVAL_INSTRUCTION")),
        "an instruction captured at container creation would become stale"
    );
}

#[test]
fn shared_verifier_resets_tests_before_each_selected_tree_copy() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    let events = runtime.events.borrow().clone();
    let reset_indices = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| event.starts_with("reset-tests:").then_some(index))
        .collect::<Vec<_>>();
    let copy_indices = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| event.starts_with("copy-tests:").then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(reset_indices.len(), 4);
    assert_eq!(copy_indices.len(), 2);
    assert!(reset_indices[0] < copy_indices[0]);
    assert!(reset_indices[1] > copy_indices[0]);
    assert!(reset_indices[2] < copy_indices[1]);
    assert!(reset_indices[3] > copy_indices[1]);
    let second_agent = events.iter().position(|event| event == "agent:2").unwrap();
    assert!(reset_indices[1] < second_agent);
    assert!(events[copy_indices[0]].contains("/tests/."));
    assert!(events[copy_indices[1]].contains("/steps/two/tests/."));
    assert_eq!(
        runtime.reset_users.into_inner(),
        vec![
            Some("root".to_owned()),
            Some("root".to_owned()),
            Some("root".to_owned()),
            Some("root".to_owned()),
        ]
    );
}

#[test]
fn shared_verifier_failure_reports_its_cleanup_error() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::failing_shared_verifier_cleanup();

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the verifier failure must stop before the second agent");

    assert!(matches!(
        error,
        EvalExecutionError::ContainerTeardown { reason, .. }
            if reason.contains("verifier 1 failed") && reason.contains("reset 2 failed")
    ));
    assert_eq!(runtime.agent_execs.get(), 1);
    assert_eq!(runtime.reset_calls.get(), 2);
}

#[test]
fn separate_verifiers_use_fresh_staging_and_artifact_snapshots() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        result.steps[0].artifacts,
        vec![(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"first snapshot"),
        )]
    );
    assert_eq!(
        result.steps[1].artifacts,
        vec![(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"second snapshot"),
        )]
    );
    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 3);
    assert!(creates[0].workspace.is_some());
    assert_eq!(creates[1].workspace, None);
    assert_eq!(creates[2].workspace, None);
    assert!(creates[1].container.contains("verifier-one"));
    assert!(creates[2].container.contains("verifier-two"));
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 2);
    assert_ne!(transfers[0].0, transfers[1].0);
    assert!(transfers[0].1.contains("verifier-one:/work"));
    assert!(transfers[1].1.contains("verifier-two:/work"));
}

#[test]
fn single_step_separate_verifier_stages_artifacts_at_implicit_image_workdir() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/result.txt\"]",
        "[verifier]\nenvironment_mode = \"separate\"\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(result.artifacts.len(), 1);
    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 2);
    assert_eq!(creates[0].workspace, None);
    assert_eq!(creates[1].workspace, None);
    assert_eq!(runtime.inspected_workdirs.borrow().len(), 1);
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 1);
    assert_eq!(
        transfers[0].1,
        format!("{}:/image-workdir", creates[1].container)
    );
    assert_eq!(runtime.verifier_workdirs.into_inner(), vec![None]);
}

#[test]
fn single_step_separate_verifier_uses_one_absolute_deadline_for_setup_and_reward() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/result.txt\"]",
        "[agent]\ntimeout_sec = 20\n\n[verifier]\nenvironment_mode = \"separate\"\ntimeout_sec = 20\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/work".to_owned()),
    )
    .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime = StepRecordingRuntime::recording_deadlines(clock.clone());

    DockerProcessSandbox::with_clock(clock)
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    let events = runtime.deadline_events.into_inner();
    let verifier_operations = &events[..7];
    assert_eq!(
        verifier_operations
            .iter()
            .map(|(event, _)| event.as_str())
            .collect::<Vec<_>>(),
        vec![
            "create",
            "start",
            "exec:verifier",
            "copy",
            "exec:verifier",
            "copy",
            "exec:verifier",
        ]
    );
    assert!(
        verifier_operations
            .windows(2)
            .all(|operations| operations[0].1 > operations[1].1),
        "verifier deadlines must consume one absolute phase budget: {events:?}"
    );
    assert!(events[7..].iter().all(|(event, deadline)| {
        event == "remove" && !deadline.is_zero() && *deadline <= Duration::from_secs(10)
    }));
}

#[test]
fn single_step_verifier_created_at_its_deadline_is_removed() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/result.txt\"]",
        "[agent]\ntimeout_sec = 1\n\n[verifier]\nenvironment_mode = \"separate\"\ntimeout_sec = 1\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/work".to_owned()),
    )
    .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime = StepRecordingRuntime::default()
        .timeout_after_verifier_create(clock.clone(), Duration::from_secs(1));

    let error = DockerProcessSandbox::with_clock(clock)
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the exhausted verifier deadline must stop startup");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    let verifier = runtime
        .creates
        .borrow()
        .iter()
        .find(|create| create.container.contains("-verifier"))
        .expect("verifier create side effect")
        .container
        .clone();
    assert!(runtime.removal_arguments.borrow().iter().any(|arguments| {
        arguments
            .last()
            .is_some_and(|container| container == &verifier)
    }));
}

#[test]
fn single_step_separate_verifier_copies_artifacts_to_explicit_workdir_without_mounting_it() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/result.txt\"]",
        "[verifier]\nenvironment_mode = \"separate\"\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/work".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 2);
    assert!(creates[0].workspace.is_some());
    assert_eq!(creates[1].workspace, None);
    assert!(runtime.inspected_workdirs.borrow().is_empty());
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 1);
    assert_eq!(transfers[0].1, format!("{}:/work", creates[1].container));
    assert_eq!(
        runtime.verifier_workdirs.into_inner(),
        vec![Some("/work".to_owned())]
    );
}

#[test]
fn single_step_separate_verifier_rejects_reserved_workdir_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/result.txt\"]",
        "[verifier]\nenvironment_mode = \"separate\"\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/tests".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("artifact staging cannot occupy the verifier test namespace");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("reserved verifier path")
    ));
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert!(runtime.events.borrow().iter().all(|event| {
        !event.starts_with("inspect-workdir:")
            && !event.starts_with("copy-artifacts:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
fn separate_verifier_stages_artifacts_without_overriding_image_workdir() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(result.steps.len(), 2);
    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 3);
    assert_eq!(creates[0].workspace, None);
    assert_eq!(creates[1].workspace, None);
    assert_eq!(creates[2].workspace, None);
    assert_eq!(runtime.inspected_workdirs.borrow().len(), 2);
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 2);
    assert_ne!(transfers[0].0, transfers[1].0);
    assert!(transfers[0].1.ends_with(":/image-workdir"));
    assert!(transfers[1].1.ends_with(":/image-workdir"));
    assert_eq!(runtime.verifier_workdirs.into_inner(), vec![None, None]);
}

#[test]
fn separate_verifier_rejects_reserved_image_workdir_before_artifact_transfer() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::with_image_workdir("/tests");

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the image workdir would stage an artifact below /tests");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("reserved verifier path")
    ));
    assert_eq!(runtime.creates.borrow().len(), 2);
    assert_eq!(runtime.starts.get(), 2);
    let events = runtime.events.borrow().clone();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("inspect-workdir:"))
    );
    assert!(events.iter().all(|event| {
        !event.starts_with("copy-artifacts:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
fn separate_verifier_rejects_reserved_cli_workdir_before_verifier_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/logs/verifier".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the CLI workdir would stage an artifact below evaluator reward paths");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("reserved verifier path")
    ));
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert!(runtime.events.borrow().iter().all(|event| {
        !event.starts_with("inspect-workdir:")
            && !event.starts_with("copy-artifacts:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_transfer_preserves_colliding_image_workdir_contents() {
    let _docker_lock = docker_runtime_test_lock();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/aiperf-eval-artifacts/result.txt"]

[task]
name = "example/multi-step-colliding-image-workdir"

[[steps]]
name = "one"
[steps.verifier]
environment_mode = "separate"

[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier /aiperf-eval-artifacts && printf image-sentinel > /aiperf-eval-artifacts/image.txt\nWORKDIR /aiperf-eval-artifacts\n",
    )
    .unwrap();
    let verifier = "test \"$(cat image.txt)\" = image-sentinel\ntest \"$(cat result.txt)\" = agent-artifact\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n";
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), verifier).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();

    let result = DockerProcessSandbox::new()
        .execute_multi_step(
            &recipe,
            &imported.package,
            &[
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "printf agent-artifact > result.txt".to_owned(),
            ],
        )
        .unwrap();

    assert_eq!(result.steps.len(), 2);
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.reward.metrics.get("reward") == Some(&1.0))
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_anonymous_volumes_are_removed_after_success() {
    let _docker_lock = docker_runtime_test_lock();
    let before = docker_resource_names();
    let temporary = tempfile::tempdir().unwrap();
    let result = run_multi_step_volume_task(&temporary, false).unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_eq!(docker_resource_names(), before);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_anonymous_volumes_are_removed_after_timeout() {
    let _docker_lock = docker_runtime_test_lock();
    let before = docker_resource_names();
    let temporary = tempfile::tempdir().unwrap();
    let error = run_multi_step_volume_task(&temporary, true)
        .expect_err("the first separate verifier must time out");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    assert_eq!(docker_resource_names(), before);
}

#[test]
fn multi_step_failures_stop_successors_and_cleanup_every_acquired_lease() {
    for (failure, expected_counts, expected_error, fail_first_removal) in [
        (
            StepFailure::Agent(2),
            (2, 1, 1, 2),
            EvalExecutionError::ProcessFailure("agent 2 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Collection(2),
            (2, 2, 1, 2),
            EvalExecutionError::ArtifactCollection("collection 2 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Verifier(1),
            (1, 1, 1, 2),
            EvalExecutionError::ProcessFailure("verifier 1 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Verifier(2),
            (2, 2, 2, 3),
            EvalExecutionError::ProcessFailure("verifier 2 failed".to_owned()),
            true,
        ),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = multi_step_task_root(&temporary, true);
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();
        let runtime = StepRecordingRuntime::failing(failure, fail_first_removal);

        let error = DockerProcessSandbox::new()
            .execute_multi_step_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret,
            )
            .expect_err("the injected phase failure must be terminal");

        if fail_first_removal {
            assert!(matches!(
                error,
                EvalExecutionError::ContainerTeardown { reason, .. }
                    if reason.contains(&expected_error.to_string())
                        && reason.contains("first removal failed")
            ));
        } else {
            assert_eq!(error, expected_error);
        }
        assert_eq!(runtime.agent_execs.get(), expected_counts.0);
        assert_eq!(runtime.collection_calls.get(), expected_counts.1);
        assert_eq!(runtime.verifier_execs.get(), expected_counts.2);
        assert_eq!(runtime.creates.borrow().len(), expected_counts.3);
        assert_eq!(runtime.removals.get(), expected_counts.3);
        assert!(runtime.removal_arguments.borrow().iter().all(|arguments| {
            arguments.len() == 4
                && arguments[0] == "rm"
                && arguments[1] == "--force"
                && arguments[2] == "--volumes"
        }));
    }
}

#[derive(Clone, Copy)]
enum StepFailure {
    Agent(usize),
    Collection(usize),
    Verifier(usize),
}

struct RecordedCreate {
    container: String,
    workspace: Option<String>,
    arguments: Vec<String>,
}

#[test]
fn single_step_agent_workdir_preparation_exhausts_its_deadline_before_agent_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
user = "bench"

[agent]
timeout_sec = 1

[verifier]
timeout_sec = 1
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let clock = Rc::new(SimClock::new());
    let runtime = StepRecordingRuntime::default().advance_after_agent_workdir_prepare(
        clock.clone(),
        Duration::from_secs(1).as_nanos() as i64,
    );

    let error = DockerProcessSandbox::with_clock(clock)
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent-must-not-run".to_owned()],
            &FixedSecret,
        )
        .expect_err("exhausted workdir preparation must stop the agent phase");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
            ..
        }
    ));
    assert_eq!(runtime.agent_execs.get(), 0);
    assert!(
        runtime
            .events
            .borrow()
            .iter()
            .any(|event| event.starts_with("prepare-agent-workdir:"))
    );
}

#[derive(Default)]
struct StepRecordingRuntime {
    build_calls: Cell<usize>,
    starts: Cell<usize>,
    agent_execs: Cell<usize>,
    collection_calls: Cell<usize>,
    verifier_execs: Cell<usize>,
    reset_calls: Cell<usize>,
    removals: Cell<usize>,
    events: RefCell<Vec<String>>,
    creates: RefCell<Vec<RecordedCreate>>,
    agent_environments: RefCell<Vec<BTreeMap<String, String>>>,
    reset_users: RefCell<Vec<Option<String>>>,
    inspected_workdirs: RefCell<Vec<String>>,
    artifact_transfers: RefCell<Vec<(String, String)>>,
    verifier_workdirs: RefCell<Vec<Option<String>>>,
    removal_arguments: RefCell<Vec<Vec<String>>>,
    failure: Option<StepFailure>,
    fail_reset_call: Option<usize>,
    fail_first_removal: bool,
    image_workdir: Option<String>,
    observe_source_snapshot: bool,
    observed_source_roots: RefCell<Vec<PathBuf>>,
    deadline_events: RefCell<Vec<(String, Duration)>>,
    deadline_clock: Option<Rc<SimClock>>,
    advance_agent_workdir_prepare_to: Option<(Rc<SimClock>, i64)>,
    verifier_create_timeout: Option<(Rc<SimClock>, Duration)>,
}

impl StepRecordingRuntime {
    fn with_image_workdir(workdir: &str) -> Self {
        Self {
            image_workdir: Some(workdir.to_owned()),
            ..Self::default()
        }
    }

    fn failing(failure: StepFailure, fail_first_removal: bool) -> Self {
        Self {
            failure: Some(failure),
            fail_first_removal,
            ..Self::default()
        }
    }

    fn failing_shared_verifier_cleanup() -> Self {
        Self {
            failure: Some(StepFailure::Verifier(1)),
            fail_reset_call: Some(2),
            ..Self::default()
        }
    }

    fn observing_source_snapshot(failure: Option<StepFailure>) -> Self {
        Self {
            failure,
            observe_source_snapshot: true,
            ..Self::default()
        }
    }

    fn recording_deadlines(clock: Rc<SimClock>) -> Self {
        Self {
            deadline_clock: Some(clock),
            ..Self::default()
        }
    }

    fn advance_after_agent_workdir_prepare(mut self, clock: Rc<SimClock>, time_ns: i64) -> Self {
        self.advance_agent_workdir_prepare_to = Some((clock, time_ns));
        self
    }

    fn timeout_after_verifier_create(mut self, clock: Rc<SimClock>, timeout: Duration) -> Self {
        self.verifier_create_timeout = Some((clock, timeout));
        self
    }

    fn record_deadline(&self, event: &str, deadline: Option<Duration>) {
        let Some(deadline) = deadline else {
            return;
        };
        self.deadline_events
            .borrow_mut()
            .push((event.to_owned(), deadline));
        if let Some(clock) = &self.deadline_clock {
            clock.advance_to(clock.now_ns() + 1_000_000_000);
        }
    }
}

impl DockerRuntime for StepRecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
            .with_users()
            .with_workdir()
            .with_phase_timeouts()
    }

    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        self.events.borrow_mut().push("build".to_owned());
        if self.observe_source_snapshot {
            let context = Path::new(request.public_arguments().last().ok_or_else(|| {
                EvalExecutionError::Materialization("Docker build context is absent".to_owned())
            })?);
            assert_eq!(
                fs::read(context.join("Dockerfile")).unwrap(),
                b"FROM scratch\n"
            );
            assert_eq!(
                fs::read(context.join("context.txt")).unwrap(),
                b"original context\n"
            );
            assert!(context.join("empty").is_dir());
            assert_eq!(
                fs::metadata(context.join("helper.sh"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o755
            );
            self.observed_source_roots
                .borrow_mut()
                .push(context.parent().unwrap().to_path_buf());
        }
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.record_deadline("create", request.deadline());
        let arguments = request.public_arguments();
        let container = argument_after(arguments, "--name").to_owned();
        let workspace = argument_after(arguments, "--volume")
            .split_once(':')
            .map(|(host, _)| host.to_owned());
        self.events.borrow_mut().push(format!("create:{container}"));
        self.creates.borrow_mut().push(RecordedCreate {
            container: container.clone(),
            workspace,
            arguments: arguments.to_vec(),
        });
        if container.contains("-verifier")
            && let Some((clock, timeout)) = &self.verifier_create_timeout
        {
            clock.advance_to(timeout.as_nanos() as i64);
            return Err(EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
                timeout: *timeout,
            });
        }
        Ok(())
    }

    fn compensate_create_timeout(
        &self,
        request: &DockerCreateRequest,
        cleanup_deadline: Duration,
    ) -> Result<(), EvalExecutionError> {
        let target = request
            .creation_target()
            .expect("bounded verifier create target");
        self.remove(
            &DockerRemoveRequest::new(["rm", "--force", "--volumes", target])
                .with_deadline(cleanup_deadline),
        )
    }

    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.record_deadline("start", request.deadline());
        self.starts.set(self.starts.get() + 1);
        self.events
            .borrow_mut()
            .push(format!("start:{}", request.container()));
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request.phase() == aiperf_runtime::eval::EvalExecutionPhase::Verifier {
            self.record_deadline("exec:verifier", request.deadline());
        }
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("rm -rf /tests"))
        {
            let call = self.reset_calls.get() + 1;
            self.reset_calls.set(call);
            self.reset_users
                .borrow_mut()
                .push(request.user().map(str::to_owned));
            self.events
                .borrow_mut()
                .push(format!("reset-tests:{}", request.container()));
            if self.fail_reset_call == Some(call) {
                return Err(EvalExecutionError::ProcessFailure(format!(
                    "reset {call} failed"
                )));
            }
            return Ok(());
        }
        if request.phase() == aiperf_runtime::eval::EvalExecutionPhase::Agent
            && request.public_arguments().first().map(String::as_str) == Some("/bin/sh")
        {
            if let Some((clock, time_ns)) = &self.advance_agent_workdir_prepare_to {
                clock.advance_to(*time_ns);
            }
            self.events
                .borrow_mut()
                .push(format!("prepare-agent-workdir:{}", request.container()));
            return Ok(());
        }
        if request.public_arguments().first().map(String::as_str) == Some("mkdir") {
            assert_eq!(request.user(), Some("root"));
            assert_eq!(request.workdir(), None);
            self.events
                .borrow_mut()
                .push(format!("prepare-artifacts:{}", request.container()));
            return Ok(());
        }
        match request.phase().to_string().as_str() {
            "agent" => {
                let call = self.agent_execs.get() + 1;
                self.agent_execs.set(call);
                self.agent_environments
                    .borrow_mut()
                    .push(request.public_environment().clone());
                self.events.borrow_mut().push(format!("agent:{call}"));
                if matches!(self.failure, Some(StepFailure::Agent(failed)) if failed == call) {
                    return Err(EvalExecutionError::ProcessFailure(format!(
                        "agent {call} failed"
                    )));
                }
                Ok(())
            }
            "verifier" => {
                let call = self.verifier_execs.get() + 1;
                self.verifier_execs.set(call);
                self.verifier_workdirs
                    .borrow_mut()
                    .push(request.workdir().map(str::to_owned));
                self.events.borrow_mut().push(format!("verifier:{call}"));
                if matches!(self.failure, Some(StepFailure::Verifier(failed)) if failed == call) {
                    return Err(EvalExecutionError::ProcessFailure(format!(
                        "verifier {call} failed"
                    )));
                }
                Ok(())
            }
            phase => panic!("unexpected {phase} phase"),
        }
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        self.record_deadline("copy", request.deadline());
        let arguments = request.public_arguments();
        let source = &arguments[1];
        let destination = &arguments[2];
        if destination.ends_with(":/tests") {
            if self.observe_source_snapshot {
                let tree = Path::new(source.trim_end_matches("/."));
                let expected = if source.contains("steps/two/tests") {
                    b"original step helper\n".as_slice()
                } else {
                    b"original root helper\n".as_slice()
                };
                assert_eq!(fs::read(tree.join("helper.sh")).unwrap(), expected);
                assert!(tree.join("empty").is_dir());
                assert_eq!(
                    fs::metadata(tree.join("helper.sh"))
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o755
                );
                let source_root = if source.contains("steps/two/tests") {
                    tree.parent().unwrap().parent().unwrap().parent().unwrap()
                } else {
                    tree.parent().unwrap()
                };
                self.observed_source_roots
                    .borrow_mut()
                    .push(source_root.to_path_buf());
            }
            self.events
                .borrow_mut()
                .push(format!("copy-tests:{source}"));
            return Ok(());
        }
        if source.ends_with("/reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        if source.ends_with("/reward.txt") {
            fs::write(destination, format!("{}\n", self.verifier_execs.get())).unwrap();
            return Ok(());
        }
        if !source.contains(':') && destination.contains(':') {
            self.artifact_transfers
                .borrow_mut()
                .push((source.to_owned(), destination.to_owned()));
            self.events
                .borrow_mut()
                .push(format!("copy-artifacts:{destination}"));
            return Ok(());
        }
        panic!("unexpected Docker copy: {arguments:?}")
    }

    fn container_workdir(&self, container: &str) -> Result<String, EvalExecutionError> {
        self.inspected_workdirs
            .borrow_mut()
            .push(container.to_owned());
        self.events
            .borrow_mut()
            .push(format!("inspect-workdir:{container}"));
        Ok(self
            .image_workdir
            .clone()
            .unwrap_or_else(|| "/image-workdir".to_owned()))
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        if source.ends_with("reward.txt") {
            return Ok(Box::new(io::Cursor::new(test_tar_archive(
                "reward.txt",
                b"1\n",
            ))));
        }
        let call = self.collection_calls.get() + 1;
        self.collection_calls.set(call);
        self.events.borrow_mut().push(format!("collect:{call}"));
        if matches!(self.failure, Some(StepFailure::Collection(failed)) if failed == call) {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "collection {call} failed"
            )));
        }
        let contents = if call == 1 {
            b"first snapshot".as_slice()
        } else {
            b"second snapshot".as_slice()
        };
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "result.txt",
            contents,
        ))))
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.record_deadline("remove", request.deadline());
        let call = self.removals.get() + 1;
        self.removals.set(call);
        self.removal_arguments
            .borrow_mut()
            .push(request.public_arguments().to_vec());
        self.events.borrow_mut().push(format!(
            "remove:{}",
            request.public_arguments().last().unwrap()
        ));
        if self.fail_first_removal && call == 1 {
            Err(EvalExecutionError::ProcessFailure(
                "first removal failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn docker_execution_uses_one_imported_source_materialization_and_releases_it() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    fs::create_dir(task_root.join("environment/empty")).unwrap();
    fs::write(
        task_root.join("environment/context.txt"),
        "original context\n",
    )
    .unwrap();
    fs::write(task_root.join("environment/helper.sh"), "#!/bin/sh\n").unwrap();
    fs::create_dir(task_root.join("tests/empty")).unwrap();
    fs::write(task_root.join("tests/helper.sh"), "original root helper\n").unwrap();
    fs::create_dir(task_root.join("steps/two/tests/empty")).unwrap();
    fs::write(
        task_root.join("steps/two/tests/helper.sh"),
        "original step helper\n",
    )
    .unwrap();
    for helper in [
        "environment/helper.sh",
        "tests/helper.sh",
        "steps/two/tests/helper.sh",
    ] {
        fs::set_permissions(task_root.join(helper), fs::Permissions::from_mode(0o755)).unwrap();
    }
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    fs::remove_dir_all(&task_root).unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    for failure in [None, Some(StepFailure::Agent(1))] {
        let runtime = StepRecordingRuntime::observing_source_snapshot(failure);
        let result = DockerProcessSandbox::new().execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        );
        if failure.is_some() {
            assert!(matches!(result, Err(EvalExecutionError::ProcessFailure(_))));
        } else {
            assert_eq!(result.unwrap().steps.len(), 2);
        }
        let observed = runtime.observed_source_roots.into_inner();
        assert!(!observed.is_empty());
        assert!(observed.iter().all(|root| root == &observed[0]));
        assert!(!observed[0].exists());
    }
}

fn argument_after<'a>(arguments: &'a [String], flag: &str) -> &'a str {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
        .unwrap_or("")
}

fn test_tar_archive(path: &str, contents: &[u8]) -> Vec<u8> {
    let mut header = [0_u8; 512];
    header[..path.len()].copy_from_slice(path.as_bytes());
    header[100..108].copy_from_slice(b"0000644\0");
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    let size = format!("{:011o}\0", contents.len());
    header[124..136].copy_from_slice(size.as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[148..156].fill(b' ');
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let checksum = header.iter().map(|byte| u32::from(*byte)).sum::<u32>();
    header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    let mut archive = header.to_vec();
    archive.extend_from_slice(contents);
    archive.resize(archive.len().next_multiple_of(512) + 1024, 0);
    archive
}

fn docker_runtime_test_lock() -> MutexGuard<'static, ()> {
    DOCKER_RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn docker_resource_names() -> (BTreeSet<String>, BTreeSet<String>) {
    let containers = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .expect("inspect Docker containers");
    assert!(
        containers.status.success(),
        "docker container listing failed: {}",
        String::from_utf8_lossy(&containers.stderr)
    );
    let volumes = Command::new("docker")
        .args(["volume", "ls", "--quiet"])
        .output()
        .expect("inspect Docker volumes");
    assert!(
        volumes.status.success(),
        "docker volume listing failed: {}",
        String::from_utf8_lossy(&volumes.stderr)
    );
    (
        String::from_utf8(containers.stdout)
            .unwrap()
            .lines()
            .filter(|name| name.starts_with("aiperf-eval-"))
            .map(str::to_owned)
            .collect(),
        String::from_utf8(volumes.stdout)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect(),
    )
}

fn multi_step_volume_task_root(
    temporary: &tempfile::TempDir,
    has_verifier_timeout: bool,
) -> std::path::PathBuf {
    let task_root = multi_step_task_root(temporary, true);
    let agent_timeout = has_verifier_timeout
        .then_some("[steps.agent]\ntimeout_sec = 5\n")
        .unwrap_or("");
    let verifier_timeout = has_verifier_timeout
        .then_some("timeout_sec = 1\n")
        .unwrap_or("");
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/aiperf-eval-artifacts/result.txt"]

[task]
name = "example/multi-step-volume-workdir"

[[steps]]
name = "one"
{agent_timeout}
[steps.verifier]
environment_mode = "separate"
{verifier_timeout}
[[steps]]
name = "two"
{agent_timeout}
[steps.verifier]
environment_mode = "separate"
{verifier_timeout}"#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier /aiperf-eval-artifacts && printf image-sentinel > /aiperf-eval-artifacts/image.txt\nWORKDIR /aiperf-eval-artifacts\nVOLUME /aiperf-eval-artifacts\n",
    )
    .unwrap();
    let verifier = if has_verifier_timeout {
        "sleep 2\n"
    } else {
        "test \"$(cat image.txt)\" = image-sentinel\ntest \"$(cat result.txt)\" = agent-artifact\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n"
    };
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), verifier).unwrap();
    task_root
}

fn run_multi_step_volume_task(
    temporary: &tempfile::TempDir,
    has_verifier_timeout: bool,
) -> Result<aiperf_runtime::eval::MultiStepExecutionResult, EvalExecutionError> {
    let task_root = multi_step_volume_task_root(temporary, has_verifier_timeout);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    DockerProcessSandbox::new().execute_multi_step(
        &recipe,
        &imported.package,
        &[
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            "printf agent-artifact > result.txt".to_owned(),
        ],
    )
}

fn multi_step_task_root(
    temporary: &tempfile::TempDir,
    has_separate_verifiers: bool,
) -> std::path::PathBuf {
    let task_root = temporary.path().join("multi-step-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/one")).unwrap();
    fs::create_dir_all(task_root.join("steps/two/tests")).unwrap();
    let verifier = has_separate_verifiers
        .then_some("\n[steps.verifier]\nenvironment_mode = \"separate\"\n")
        .unwrap_or("");
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/work/result.txt"]

[task]
name = "example/multi-step-runtime"

[[steps]]
name = "one"
{verifier}
[[steps]]
name = "two"
{verifier}"#,
        ),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Legacy instruction.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("steps/one/instruction.md"),
        "First instruction.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/instruction.md"),
        "Second instruction.\n",
    )
    .unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), "exit 0\n").unwrap();
    task_root
}

fn standard_task_root(temporary: &tempfile::TempDir, manifest_suffix: &str) -> std::path::PathBuf {
    let task_root = temporary.path().join("task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n\n[task]\nname = \"example/task\"\n\n{manifest_suffix}"),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    task_root
}

fn native_graph_task_root(temporary: &tempfile::TempDir) -> std::path::PathBuf {
    let task_root = temporary.path().join("native-graph-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("tools")).unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-docker-authority"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("agent_graph.json"),
        r#"{
  "schema_version": "1.0",
  "trace_id": "authority",
  "stage_bound": 1,
  "channels": { "output": { "type": "text", "reducer": "overwrite" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": ["output"]
}"#,
    )
    .unwrap();
    fs::write(
        task_root.join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.sh"]
executable = "tools/adapter.sh"
"#,
    )
    .unwrap();
    fs::write(task_root.join("tools/adapter.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    task_root
}

fn external_driver_task_root(
    temporary: &tempfile::TempDir,
    manifest_suffix: &str,
) -> std::path::PathBuf {
    let task_root = temporary.path().join("external-driver-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("tools")).unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("instruction.md"), "Do external work.\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.1"

[task]
name = "example/external-driver-docker-authority"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
external_driver_factory_id = "fixture"

{manifest_suffix}"#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("adapters.toml"),
        r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
    )
    .unwrap();
    fs::write(task_root.join("tools/driver.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    task_root
}

fn resolve_docker_driver_trial(
    imported: aiperf_runtime::eval::ImportedTask,
) -> aiperf_runtime::eval::ResolvedEpisodeTrial {
    resolve_docker_driver_trial_for_run(imported, b"external-driver-docker-run")
}

fn resolve_docker_driver_trial_for_run(
    imported: aiperf_runtime::eval::ImportedTask,
    run: &[u8],
) -> aiperf_runtime::eval::ResolvedEpisodeTrial {
    let trial = TrialSpec::new(
        imported.task.clone(),
        AgentVariantRef::new("external-driver").unwrap(),
        ModelIdentity::new("compatibility", "opaque-driver").unwrap(),
        7,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"external-policy")),
        TrialBudget::new(30.0, 30.0).unwrap(),
        ArtifactDigest::from_bytes(b"external-environment"),
        ArtifactDigest::from_bytes(b"external-verifier"),
        RuntimeIdentity::new("external-compatibility").unwrap(),
    )
    .unwrap();
    NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported,
            trial,
            NonZeroUsize::new(1).unwrap(),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap()
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(run)))
    .unwrap()
    .trials()
    .first()
    .unwrap()
    .clone()
}

fn external_driver_deadlines() -> AdapterLifecycleDeadlines {
    AdapterLifecycleDeadlines::new(
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .unwrap()
}

struct PreparedDriverFixture;

#[async_trait(?Send)]
impl PreparedExternalDriver for PreparedDriverFixture {
    async fn run(
        &mut self,
        session: &mut dyn ExternalDriverSession,
    ) -> Result<CompatibilityTerminalReceipt, ExternalDriverError> {
        session.request_terminal().await
    }
}

struct RecordingExternalDriverFactory {
    prepare_calls: Arc<AtomicUsize>,
}

impl NativeGraphExternalDriverFactory for RecordingExternalDriverFactory {
    fn id(&self) -> &str {
        "fixture"
    }

    fn prepare_driver(
        &self,
        _: &NativeGraphPackagePlan,
        _: &aiperf_runtime::eval::ResolvedEpisodeTrial,
    ) -> Result<Box<dyn PreparedExternalDriver>, ExternalDriverError> {
        self.prepare_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(PreparedDriverFixture))
    }
}

fn prepare_external_driver(
    imported: &aiperf_runtime::eval::ImportedTask,
    trial: &aiperf_runtime::eval::ResolvedEpisodeTrial,
) -> PreparedExternalDriverCapability {
    RecordingExternalDriverFactory {
        prepare_calls: Arc::new(AtomicUsize::new(0)),
    }
    .prepare(&imported.package, trial)
    .unwrap()
}

#[test]
fn external_driver_authorization_is_prepared_before_build_and_spawns_only_declared_secret_free_argv()
 {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let events = Rc::new(RefCell::new(Vec::new()));
    let requests = Rc::new(RefCell::new(Vec::new()));
    let runtime = LegacyRuntime {
        events: Rc::clone(&events),
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::clone(&events),
            requests: Rc::clone(&requests),
        })),
        ..LegacyRuntime::default()
    };
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let factory = RecordingExternalDriverFactory {
        prepare_calls: Arc::clone(&prepare_calls),
    };
    let prepared = factory
        .prepare(&imported.package, &trial)
        .expect("factory preparation seals the exact package, adapter, and trial");
    assert_eq!(prepare_calls.load(Ordering::Relaxed), 1);

    let launch = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            imported.package.execution_plan(),
            Some(prepared),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect("the exact external Driver launch is authorized before provisioning");

    assert_eq!(runtime.native_graph_secret_provider_calls.get(), 0);
    runtime
        .build(&DockerBuildRequest::new(["build"]))
        .expect("fixture build succeeds");
    let _started = launch.start().expect("authorized spawn begins once");
    assert_eq!(
        requests.borrow().as_slice(),
        &[(
            "external-driver-task-container".to_owned(),
            vec!["tools/driver.sh".to_owned()],
            BTreeMap::new(),
            external_driver_deadlines(),
        )]
    );
    assert_eq!(
        events.borrow().as_slice(),
        ["external-driver-spawner", "build", "adapter-spawn",]
    );
}

#[test]
fn dropping_started_external_driver_fences_its_live_transaction_once() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = LegacyRuntime {
        events: Rc::clone(&events),
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::clone(&events),
            requests: Rc::new(RefCell::new(Vec::new())),
        })),
        ..LegacyRuntime::default()
    };
    let launch = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            imported.package.execution_plan(),
            Some(prepare_external_driver(&imported, &trial)),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .unwrap();

    let started = launch.start().unwrap();
    drop(started);

    assert_eq!(
        events.borrow().as_slice(),
        ["external-driver-spawner", "adapter-spawn", "client-fence"]
    );
}

#[test]
fn external_driver_missing_spawner_refuses_before_docker_create() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let runtime = LegacyRuntime::default();
    let prepared = prepare_external_driver(&imported, &trial);

    let error = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            imported.package.execution_plan(),
            Some(prepared),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect_err("a runtime without the distinct external spawner must fail closed");

    assert!(matches!(
        error,
        EvalExecutionError::UnsupportedEnforcement("external Driver Docker adapter spawn")
    ));
    assert!(runtime.creates.borrow().is_empty());
    assert_eq!(runtime.native_graph_secret_provider_calls.get(), 0);
}

#[test]
fn external_driver_compose_plan_refuses_before_docker_create() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  helper:\n    image: helper:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let runtime = LegacyRuntime {
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::new(RefCell::new(Vec::new())),
            requests: Rc::new(RefCell::new(Vec::new())),
        })),
        ..LegacyRuntime::default()
    };
    let prepared = prepare_external_driver(&imported, &trial);

    let error = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            imported.package.execution_plan(),
            Some(prepared),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect_err("external compatibility does not authorize Compose");

    assert!(matches!(
        error,
        EvalExecutionError::UnsupportedEnforcement("external Driver Docker Compose")
    ));
    assert!(runtime.creates.borrow().is_empty());
    assert_eq!(runtime.external_driver_spawner_calls.get(), 0);
}

#[test]
fn external_driver_multi_step_plan_refuses_before_docker_create() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let multi_step_temporary = tempfile::tempdir().unwrap();
    let multi_step_root = multi_step_task_root(&multi_step_temporary, true);
    let multi_step_imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(multi_step_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = LegacyRuntime {
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::new(RefCell::new(Vec::new())),
            requests: Rc::new(RefCell::new(Vec::new())),
        })),
        ..LegacyRuntime::default()
    };
    let prepared = prepare_external_driver(&imported, &trial);

    let error = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            multi_step_imported.package.execution_plan(),
            Some(prepared),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect_err("external compatibility does not authorize explicit steps");

    assert!(matches!(error, EvalExecutionError::UnsupportedMultiStep));
    assert!(runtime.creates.borrow().is_empty());
    assert_eq!(runtime.external_driver_spawner_calls.get(), 0);
}

#[test]
fn external_driver_missing_prepared_driver_refuses_before_docker_create() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let runtime = LegacyRuntime {
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::new(RefCell::new(Vec::new())),
            requests: Rc::new(RefCell::new(Vec::new())),
        })),
        ..LegacyRuntime::default()
    };

    let error = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &trial,
            imported.package.execution_plan(),
            None,
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect_err("external compatibility requires completed driver preparation");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidRecipe("prepared external Driver")
    ));
    assert!(runtime.creates.borrow().is_empty());
    assert_eq!(runtime.external_driver_spawner_calls.get(), 0);
}

#[test]
fn external_driver_mismatched_resolved_trial_refuses_before_docker_create() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = external_driver_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let trial = resolve_docker_driver_trial(imported.clone());
    let foreign_trial =
        resolve_docker_driver_trial_for_run(imported.clone(), b"substituted-resolved-trial");
    let runtime = LegacyRuntime {
        external_driver_spawn_executor: Some(Rc::new(RecordingExternalDriverSpawnExecutor {
            events: Rc::new(RefCell::new(Vec::new())),
            requests: Rc::new(RefCell::new(Vec::new())),
        })),
        ..LegacyRuntime::default()
    };
    let prepared = prepare_external_driver(&imported, &trial);

    let error = DockerProcessSandbox::new()
        .prepare_external_driver_spawn_with_runtime(
            &runtime,
            &imported.package,
            &foreign_trial,
            imported.package.execution_plan(),
            Some(prepared),
            "external-driver-task-container",
            ComposeProjectId::new("aiperf-external-driver"),
            external_driver_deadlines(),
        )
        .expect_err("a prepared driver cannot substitute another resolved external trial");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidRecipe("prepared external Driver capability")
    ));
    assert!(runtime.creates.borrow().is_empty());
    assert_eq!(runtime.external_driver_spawner_calls.get(), 0);
}

fn native_graph_rollout_task_root(temporary: &tempfile::TempDir) -> std::path::PathBuf {
    let task_root = native_graph_task_root(temporary);
    fs::create_dir_all(task_root.join("rollout")).unwrap();
    fs::write(
        task_root.join("adapters.toml"),
        r#"[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["tools/environment.sh"]
executable = "tools/environment.sh"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("tools/environment.sh"),
        "#!/bin/sh\nexit 0\n",
    )
    .unwrap();
    fs::write(task_root.join("rollout/reset.json"), b"{}\n").unwrap();
    fs::write(task_root.join("rollout/policy.json"), b"{}\n").unwrap();
    fs::write(
        task_root.join("rollout.toml"),
        r#"[environment]
adapter_id = "environment-adapter"
protocol_factory_id = "strict_jsonl"
runtime_provider_id = "strict_supervised"
stepper_factory_id = "supervised_environment"
action_encoder_id = "move_v1"
operation_deadline_ms = 5000
reset_source = "rollout/reset.json"
max_frame_bytes = 4096
max_identifier_bytes = 128
max_json_bytes = 2048
max_json_depth = 4
max_json_array_entries = 8
max_json_object_entries = 8
max_operation_ledger_entries = 16
max_model_call_lineage_entries = 4
max_session_model_call_lineage_entries = 16
max_session_model_call_lineage_bytes = 2048
max_artifact_handles = 4
max_artifact_bytes = 4096

[artifacts]
max_artifacts = 8
max_total_bytes = 16384
max_artifact_bytes = 3072
max_download_handles = 4

[policy]
environment = "counter-v1"
model_binding_id = "primary"
prompt_source = "rollout/policy.json"
max_decision_bytes = 256
horizon = 4
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
max_prompt_bytes = 256

[workspace_patch]
mutable_paths = ["result.txt"]
max_patches = 4
max_patch_bytes = 4096
max_total_patch_bytes = 8192
"#,
    )
    .unwrap();
    task_root
}

#[test]
fn native_graph_docker_execution_requires_runtime_authorization_before_build() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = LegacyRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .unwrap(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("native exact profile must be authorized before Docker build");

    assert!(
        matches!(
            error,
            EvalExecutionError::UnsupportedEnforcement("model endpoint isolation")
        ),
        "unexpected preflight result: {error:?}"
    );
    assert!(runtime.events.borrow().is_empty());
}

#[test]
fn native_graph_docker_execution_accepts_the_runtime_resolved_no_egress_proof() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = LegacyRuntime {
        native_graph_profile: Some(
            ProviderProfile::new(
                "runtime-no-egress",
                vec![ProviderCapability::ModelEndpointIsolation],
            )
            .unwrap()
            .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
            .unwrap(),
        ),
        ..LegacyRuntime::default()
    };

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .unwrap(),
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("fixture reaches the later Docker workdir boundary");
    assert!(matches!(
        error,
        EvalExecutionError::UnsupportedEnforcement("container workdir inspection")
    ));
    assert!(runtime.events.borrow().contains(&"build".to_owned()));
}

struct OrderedNativeGraphCallback<'a> {
    events: &'a Rc<RefCell<Vec<String>>>,
    fail: bool,
}

struct AdapterStartingFailingNativeGraphCallback<'a> {
    events: &'a Rc<RefCell<Vec<String>>>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for AdapterStartingFailingNativeGraphCallback<'_> {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        lease.environment_adapter_start()?.start().await?;
        self.events
            .borrow_mut()
            .push("native-graph-callback-failed".to_owned());
        Err(EvalExecutionError::ProcessFailure(
            "native graph callback failed".to_owned(),
        ))
    }
}

struct RolloutStartingNativeGraphCallback {
    events: Rc<RefCell<Vec<String>>>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for RolloutStartingNativeGraphCallback {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        lease.environment_adapter_start()?.start_rollout().await?;
        self.events
            .borrow_mut()
            .push("native-graph-rollout-started".to_owned());
        Ok(())
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for OrderedNativeGraphCallback<'_> {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        assert!(lease.is_authorized());
        assert!(lease.is_environment_acquired());
        assert_eq!(lease.instruction(), "Do work.\n");
        self.events.borrow_mut().push("native-graph".to_owned());
        if self.fail {
            return Err(EvalExecutionError::ProcessFailure(
                "native graph callback failed".to_owned(),
            ));
        }
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_graph_callback_precedes_verification_and_failure_keeps_reverse_cleanup() {
    let _guard = DOCKER_RUNTIME_TEST_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let profile = ProviderProfile::new(
        "runtime-no-egress",
        vec![ProviderCapability::ModelEndpointIsolation],
    )
    .unwrap()
    .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
    .unwrap();

    for fail in [false, true] {
        let runtime = LegacyRuntime {
            native_graph_profile: Some(profile.clone()),
            image_workdir: Some("/work".to_owned()),
            ..LegacyRuntime::default()
        };
        let mut callback = OrderedNativeGraphCallback {
            events: &runtime.events,
            fail,
        };
        let result = DockerProcessSandbox::new()
            .execute_native_graph_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &FixedSecret,
                &mut callback,
            )
            .await;
        let events = runtime.events.borrow();
        let callback_index = events
            .iter()
            .position(|event| event == "native-graph")
            .expect("callback is invoked after environment acquisition");
        let removal_index = events
            .iter()
            .position(|event| event == "remove")
            .expect("ordinary reverse cleanup removes the task container");
        assert!(callback_index < removal_index);
        if fail {
            assert!(matches!(
                result,
                Err(EvalExecutionError::ProcessFailure(reason)) if reason == "native graph callback failed"
            ));
            assert!(events.iter().all(|event| event != "verifier"));
        } else {
            assert!(result.is_ok(), "native graph callback result: {result:?}");
            let verifier_index = events
                .iter()
                .position(|event| event == "verifier")
                .expect("verifier runs after graph callback");
            assert!(callback_index < verifier_index);
            assert!(verifier_index < removal_index);
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_graph_non_rollout_callback_failure_does_not_provision_an_environment_adapter() {
    let _guard = DOCKER_RUNTIME_TEST_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let profile = ProviderProfile::new(
        "runtime-no-egress",
        vec![ProviderCapability::ModelEndpointIsolation],
    )
    .unwrap()
    .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
    .unwrap();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let adapter_events = Rc::new(RefCell::new(Vec::new()));
    let runtime = LegacyRuntime {
        native_graph_profile: Some(profile),
        image_workdir: Some("/work".to_owned()),
        adapter_spawner: Some(Rc::new(RecordingAdapterSpawner {
            events: adapter_events.clone(),
            requests: requests.clone(),
        })),
        ..LegacyRuntime::default()
    };
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let mut callback = OrderedNativeGraphCallback {
        events: &runtime.events,
        fail: true,
    };

    let error = DockerProcessSandbox::new()
        .execute_native_graph_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &FixedSecret,
            &mut callback,
        )
        .await
        .expect_err("callback failure must retain its primary error after adapter reaping");

    assert!(matches!(
        error,
        EvalExecutionError::ProcessFailure(reason) if reason == "native graph callback failed"
    ));
    assert!(
        requests.borrow().is_empty(),
        "a non-rollout callback failure must not provision an environment adapter"
    );
    assert!(
        adapter_events.borrow().is_empty(),
        "a non-rollout callback has no adapter child to cancel or reap"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn docker_rollout_callback_cannot_supply_an_unleased_start_plan() {
    let _guard = DOCKER_RUNTIME_TEST_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_rollout_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let profile = ProviderProfile::new(
        "runtime-no-egress",
        vec![ProviderCapability::ModelEndpointIsolation],
    )
    .unwrap()
    .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
    .unwrap();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let adapter_events = Rc::new(RefCell::new(Vec::new()));
    let runtime = LegacyRuntime {
        native_graph_profile: Some(profile),
        image_workdir: Some("/work".to_owned()),
        adapter_spawner: Some(Rc::new(ReadyAdapterSpawner {
            events: adapter_events.clone(),
            requests: requests.clone(),
        })),
        ..LegacyRuntime::default()
    };
    let mut callback = RolloutStartingNativeGraphCallback {
        events: adapter_events.clone(),
    };

    let error = DockerProcessSandbox::new()
        .execute_native_graph_with_runtime(
            &runtime,
            &HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .unwrap(),
            &imported.package,
            imported.package.execution_plan(),
            &FixedSecret,
            &mut callback,
        )
        .await
        .expect_err("a callback cannot mint or inject a rollout start plan");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidRecipe("NativeGraph sealed rollout start")
    ));
    assert!(
        requests.borrow().is_empty(),
        "the unleased callback cannot reach adapter provisioning"
    );
    assert!(
        adapter_events.borrow().is_empty(),
        "the callback cannot make a child exist before lease-owned plan admission"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn native_graph_adapter_start_refuses_an_absent_rollout_selector_before_spawning() {
    let _guard = DOCKER_RUNTIME_TEST_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = native_graph_task_root(&temporary);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let profile = ProviderProfile::new(
        "runtime-no-egress",
        vec![ProviderCapability::ModelEndpointIsolation],
    )
    .unwrap()
    .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
    .unwrap();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let adapter_events = Rc::new(RefCell::new(Vec::new()));
    let runtime = LegacyRuntime {
        native_graph_profile: Some(profile),
        image_workdir: Some("/work".to_owned()),
        adapter_spawner: Some(Rc::new(RecordingAdapterSpawner {
            events: adapter_events.clone(),
            requests: requests.clone(),
        })),
        ..LegacyRuntime::default()
    };
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let mut callback = AdapterStartingFailingNativeGraphCallback {
        events: &runtime.events,
    };

    let error = DockerProcessSandbox::new()
        .execute_native_graph_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &FixedSecret,
            &mut callback,
        )
        .await
        .expect_err("a graph without a rollout selector has no adapter operation");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidRecipe("NativeGraph rollout environment adapter")
    ));
    assert!(requests.borrow().is_empty());
    assert!(adapter_events.borrow().is_empty());
}

fn standard_task_with_artifacts(
    temporary: &tempfile::TempDir,
    artifacts: &str,
    manifest_suffix: &str,
) -> std::path::PathBuf {
    let task_root = standard_task_root(temporary, "");
    fs::write(
        task_root.join("task.toml"),
        format!(
            "schema_version = \"1.0\"\nartifacts = {artifacts}\n\n[task]\nname = \"example/artifacts\"\n{manifest_suffix}"
        ),
    )
    .unwrap();
    task_root
}

fn compose_recipe() -> HarborSandboxRecipe {
    HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/work".to_owned()),
    )
    .unwrap()
}

fn compose_multi_step_task_root(
    temporary: &tempfile::TempDir,
    terminal_sidecar_evidence: bool,
) -> PathBuf {
    let task_root = multi_step_task_root(temporary, true);
    let artifact = if terminal_sidecar_evidence {
        "{ source = \"/var/lib/api/result.txt\", destination = \"result.txt\", service = \"api\" }"
    } else {
        "\"/work/result.txt\""
    };
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = [{artifact}]

[task]
name = "example/compose-multi-step"

[[steps]]
name = "one"
[steps.verifier]
environment_mode = "separate"

[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate""#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    task_root
}

fn compose_terminal_evidence_task_root(
    temporary: &tempfile::TempDir,
    has_hook: bool,
    has_verifier_timeout: bool,
) -> PathBuf {
    let task_root = standard_task_root(temporary, "");
    let hook = has_hook
        .then_some("\n[[verifier.collect]]\nservice = \"api\"\ncommand = [\"flush-api\"]\n")
        .unwrap_or("");
    let verifier_timeout = has_verifier_timeout
        .then_some("timeout_sec = 1\n[agent]\ntimeout_sec = 1\n")
        .unwrap_or("");
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
artifacts = [{{ source = "/var/lib/api/result.txt", destination = "result.txt", service = "api" }}]

[task]
name = "example/compose-terminal-evidence"

[verifier]
environment_mode = "separate"
{verifier_timeout}{hook}"#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    task_root
}

fn compose_main_evidence_timeout_task_root(temporary: &tempfile::TempDir) -> PathBuf {
    let task_root = standard_task_root(temporary, "");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = ["/work/result.txt"]

[task]
name = "example/compose-main-timeout"

[agent]
timeout_sec = 1

[verifier]
environment_mode = "separate"
timeout_sec = 1
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    task_root
}
