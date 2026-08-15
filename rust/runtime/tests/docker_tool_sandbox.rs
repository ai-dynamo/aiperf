// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for the Docker recorded-agent tool sandbox.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::future::pending;
use std::path::PathBuf;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::graph::driver::TraceIdentity;
use aiperf_runtime::graph::replay::ReplayRunIdentity;
use aiperf_runtime::graph::replay::{
    ReplayArtifactPaths, ReplayTraceSupplement, ToolCallMeasurement, write_replay_artifacts,
};
use aiperf_runtime::graph::tools::{
    CONTAINER_RUN_LABEL_KEY, ContainerCreateSpec, ContainerId, ContainerRuntime, DockerCliRuntime,
    DockerSessionSandbox, EnvironmentRecipe, EnvironmentToolDispatcher, FramedCommandIo,
    GuardedToolCommandPolicy, ResolvedTraceEnvironment, ToolCommandResult, ToolDispatchContext,
    ToolDispatchRequest, ToolDispatcher, ToolSandbox, ToolSandboxError, WorkspaceSpec,
    cleanup_recorded_agent_containers, preflight_docker_sandbox,
};
use aiperf_runtime::rng::RngRoot;

#[cfg(feature = "engine")]
use aiperf_runtime::engine::application::preflight_recorded_agent_docker_environments;
#[cfg(feature = "engine")]
use aiperf_runtime::engine::control_hooks::cleanup_recorded_agent_docker_on_shutdown;

struct FakeIo {
    output: VecDeque<Bytes>,
}

#[async_trait(?Send)]
impl FramedCommandIo for FakeIo {
    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        let chunk = match self.output.pop_front() {
            Some(chunk) => chunk,
            None => pending().await,
        };
        let count = chunk.len();
        output.extend_from_slice(&chunk);
        Ok(count)
    }

    async fn close(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }
}

struct FakeRuntime {
    created: RefCell<Vec<ContainerCreateSpec>>,
    exec_argv: RefCell<Vec<Vec<OsString>>>,
    remove_count: Cell<u8>,
    next_id: Cell<u8>,
    has_start_failure: Cell<bool>,
    outputs: RefCell<VecDeque<Vec<Bytes>>>,
    inspected_images: RefCell<Vec<String>>,
    label_queries: RefCell<Vec<(String, String)>>,
    listed: RefCell<Vec<ContainerId>>,
}

struct LocalCaptureSandbox;

#[async_trait(?Send)]
impl ToolSandbox for LocalCaptureSandbox {
    async fn open(&self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn run(
        &self,
        _command: &str,
        _timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        Ok(ToolCommandResult::completed(0, Bytes::new()))
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        Ok(())
    }
}

impl FakeRuntime {
    fn new(outputs: impl IntoIterator<Item = Vec<Bytes>>) -> Self {
        Self {
            created: RefCell::new(Vec::new()),
            exec_argv: RefCell::new(Vec::new()),
            remove_count: Cell::new(0),
            next_id: Cell::new(0),
            has_start_failure: Cell::new(false),
            outputs: RefCell::new(outputs.into_iter().collect()),
            inspected_images: RefCell::new(Vec::new()),
            label_queries: RefCell::new(Vec::new()),
            listed: RefCell::new(Vec::new()),
        }
    }

    fn with_listed(self, containers: impl IntoIterator<Item = ContainerId>) -> Self {
        self.listed.replace(containers.into_iter().collect());
        self
    }

    fn with_start_failure(self) -> Self {
        self.has_start_failure.set(true);
        self
    }
}

#[async_trait(?Send)]
impl ContainerRuntime for FakeRuntime {
    async fn inspect_image(&self, image: &str) -> Result<(), ToolSandboxError> {
        self.inspected_images.borrow_mut().push(image.to_string());
        Ok(())
    }

    async fn create(&self, spec: &ContainerCreateSpec) -> Result<ContainerId, ToolSandboxError> {
        self.created.borrow_mut().push(spec.clone());
        let sequence = self.next_id.get();
        self.next_id.set(sequence.saturating_add(1));
        Ok(ContainerId::new(format!("container-{sequence}")))
    }

    async fn start(&self, _id: &ContainerId) -> Result<(), ToolSandboxError> {
        if self.has_start_failure.get() {
            return Err(ToolSandboxError::new("fake Docker start failed"));
        }
        Ok(())
    }

    async fn open_exec(
        &self,
        _id: &ContainerId,
        argv: &[OsString],
    ) -> Result<Box<dyn FramedCommandIo>, ToolSandboxError> {
        self.exec_argv.borrow_mut().push(argv.to_vec());
        let mut output = self
            .outputs
            .borrow_mut()
            .pop_front()
            .unwrap_or_default()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !output.is_empty() {
            let script = argv
                .last()
                .expect("Docker exec command includes the framed script")
                .to_string_lossy();
            let marker = script
                .split("aiperf-terminal:")
                .nth(1)
                .and_then(|tail| tail.split(":%d\\0").next())
                .expect("framed script includes one terminal marker");
            output.extend_from_slice(b"\0aiperf-terminal:");
            output.extend_from_slice(marker.as_bytes());
            output.extend_from_slice(b":0\0");
        }
        Ok(Box::new(FakeIo {
            output: (!output.is_empty())
                .then(|| VecDeque::from([Bytes::from(output)]))
                .unwrap_or_default(),
        }))
    }

    async fn force_remove(
        &self,
        _id: &ContainerId,
        _timeout_ns: u64,
    ) -> Result<(), ToolSandboxError> {
        self.remove_count
            .set(self.remove_count.get().saturating_add(1));
        Ok(())
    }

    async fn list_by_label(
        &self,
        key: &str,
        value: &str,
    ) -> Result<Vec<ContainerId>, ToolSandboxError> {
        self.label_queries
            .borrow_mut()
            .push((key.to_string(), value.to_string()));
        Ok(self.listed.borrow().clone())
    }
}

fn pinch_environment() -> ResolvedTraceEnvironment {
    ResolvedTraceEnvironment {
        kind: EnvironmentRecipe::PinchBench,
        image: "aiperf-recorded-agent-pinchbench:v1".into(),
        workspace: WorkspaceSpec {
            files: Vec::new(),
            workdir: "/workspace".into(),
            interpreter: vec!["bash".into(), "-lc".into()],
            mount_workspace: true,
            command_timeout_ns: 30_000_000_000,
        },
    }
}

fn trace() -> TraceIdentity {
    TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "trace / untrusted".into(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn dispatch_capture_uses_local_docker_and_mixed_backend_labels() {
    let docker_runtime = Rc::new(FakeRuntime::new([vec![Bytes::from_static(b"docker")]]));
    let docker = Rc::new(
        DockerSessionSandbox::new(
            pinch_environment(),
            Some(PathBuf::from("/tmp/pinch-workspace")),
            trace(),
            ReplayRunIdentity::mint(RngRoot::new(Some(29)), "replay-run-29"),
            RealClock::new(),
            docker_runtime,
            4096,
        )
        .expect("valid Docker sandbox"),
    );
    let docker_dispatcher =
        EnvironmentToolDispatcher::new(docker, Rc::new(GuardedToolCommandPolicy));
    let docker_result = docker_dispatcher
        .dispatch(
            ToolDispatchRequest::new("docker-call", "printf docker"),
            &ToolDispatchContext::default(),
        )
        .await
        .expect("Docker tool dispatch completes");
    let local_dispatcher = EnvironmentToolDispatcher::new(
        Rc::new(LocalCaptureSandbox),
        Rc::new(GuardedToolCommandPolicy),
    );
    let local_result = local_dispatcher
        .dispatch(
            ToolDispatchRequest::new("local-call", "printf local"),
            &ToolDispatchContext::default(),
        )
        .await
        .expect("local tool dispatch completes");
    let artifact_dir = tempfile::tempdir().expect("artifact directory");
    let tool_path = artifact_dir.path().join("tool-time.json");
    let tools = vec![
        ToolCallMeasurement::new(
            local_result.duration_ns as f64 / 1_000_000_000.0,
            local_dispatcher.backend_identity().artifact_label(),
        )
        .with_call_index(0),
        ToolCallMeasurement::new(
            docker_result.duration_ns as f64 / 1_000_000_000.0,
            docker_dispatcher.backend_identity().artifact_label(),
        )
        .with_call_index(1),
    ];
    write_replay_artifacts(
        &ReplayArtifactPaths {
            tool_time_path: Some(tool_path.clone()),
            ..ReplayArtifactPaths::default()
        },
        &[ReplayTraceSupplement {
            trace_id: "trace".into(),
            trajectory_id: "trajectory".into(),
            worker_id: 0,
            completed: true,
            calls: Vec::new(),
            tools,
            trace_wall_ms: 1.0,
        }],
    )
    .expect("capture writes strict tool-time artifact");
    let tool_time: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tool_path).expect("tool-time artifact"))
            .expect("strict tool-time JSON");
    assert_eq!(tool_time["backend"], "mixed");
    assert_eq!(
        docker_dispatcher.backend_identity().artifact_label(),
        "docker:aiperf-recorded-agent-pinchbench:v1"
    );
    assert_eq!(
        local_dispatcher.backend_identity().artifact_label(),
        "local"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn docker_uses_recipe_argv_network_none_and_exact_run_label() {
    // This catches a Docker backend that weakens isolation, mounts a Pinch
    // workspace somewhere other than its recipe path, or derives cleanup scope
    // from an untrusted trace id rather than the controller-minted run identity.
    let runtime = Rc::new(FakeRuntime::new([vec![Bytes::from_static(b"result")]]));
    let identity = ReplayRunIdentity::mint(RngRoot::new(Some(17)), "  replay-run-17  ");
    let sandbox = DockerSessionSandbox::new(
        pinch_environment(),
        Some(PathBuf::from("/tmp/pinch-workspace")),
        trace(),
        identity,
        RealClock::new(),
        runtime.clone(),
        4096,
    )
    .expect("valid Pinch recipe creates a Docker sandbox");

    sandbox
        .open()
        .await
        .expect("container starts before timing");
    let result = sandbox
        .run("printf result", None)
        .await
        .expect("framed docker exec completes");

    assert_eq!(result.output, Bytes::from_static(b"result"));
    let created = runtime.created.borrow();
    assert_eq!(created.len(), 1);
    assert!(created[0].has_network_disabled);
    assert_eq!(created[0].workdir, "/workspace");
    assert_eq!(created[0].image, "aiperf-recorded-agent-pinchbench:v1");
    assert_eq!(created[0].mounts.len(), 1);
    assert_eq!(
        created[0].mounts[0].host_path,
        PathBuf::from("/tmp/pinch-workspace")
    );
    assert_eq!(created[0].mounts[0].container_path, "/workspace");
    assert_eq!(
        created[0].labels,
        BTreeMap::from([(
            "aiperf.recorded-agent.run-label".to_string(),
            "replay-run-17".to_string(),
        )])
    );
    assert!(
        created[0]
            .name
            .starts_with("aiperf-recorded-agent-trace-untrusted-")
    );
    let argv = runtime.exec_argv.borrow();
    assert_eq!(argv.len(), 1);
    assert_eq!(argv[0][0], OsString::from("bash"));
    assert_eq!(argv[0][1], OsString::from("-lc"));
    let script = argv[0][2].to_string_lossy();
    assert!(script.contains("\nprintf result\n"));
    assert!(!script.contains("'printf result'"));
    let recipe_argv = argv[0].clone();
    drop(argv);
    let output = tokio::process::Command::new(&recipe_argv[0])
        .args(&recipe_argv[1..])
        .output()
        .await
        .expect("recipe argv runs without Docker");
    assert!(output.status.success());
    assert!(output.stdout.starts_with(b"result\0aiperf-terminal:"));
    assert!(output.stdout.ends_with(b":0\0"));
}

#[tokio::test(flavor = "current_thread")]
async fn failed_container_start_force_removes_the_created_orphan() {
    // This catches a split Docker create/start path that returns the start
    // error while leaving the just-created detached container behind.
    let runtime = Rc::new(FakeRuntime::new([]).with_start_failure());
    let sandbox = DockerSessionSandbox::new(
        pinch_environment(),
        Some(PathBuf::from("/tmp/pinch-workspace")),
        trace(),
        ReplayRunIdentity::mint(RngRoot::new(Some(20)), "replay-run-20"),
        RealClock::new(),
        runtime.clone(),
        4096,
    )
    .expect("valid Pinch recipe creates a Docker sandbox");

    let error = sandbox
        .open()
        .await
        .expect_err("failed Docker start rejects the sandbox");

    assert!(error.to_string().contains("start failed"));
    assert_eq!(runtime.created.borrow().len(), 1);
    assert_eq!(runtime.remove_count.get(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn docker_recipe_wrapper_emits_a_terminal_frame_after_raw_shell_exit() {
    // This catches appending the terminal frame after a raw command that calls
    // `exit`, which leaves Docker exec at EOF and misclassifies a normal
    // nonzero tool result as sandbox infrastructure failure.
    let runtime = Rc::new(FakeRuntime::new([vec![Bytes::from_static(b"begin")]]));
    let sandbox = DockerSessionSandbox::new(
        pinch_environment(),
        Some(PathBuf::from("/tmp/pinch-workspace")),
        trace(),
        ReplayRunIdentity::mint(RngRoot::new(Some(22)), "replay-run-22"),
        RealClock::new(),
        runtime.clone(),
        4096,
    )
    .expect("valid Pinch recipe creates a Docker sandbox");

    sandbox
        .open()
        .await
        .expect("container starts before timing");
    let result = sandbox
        .run("printf begin; exit 7", None)
        .await
        .expect("fake framed command completes");
    assert_eq!(result.exit_code, 0);
    let argv = runtime.exec_argv.borrow()[0].clone();
    let output = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .await
        .expect("recipe argv executes without Docker");
    assert!(output.status.success());
    assert!(output.stdout.starts_with(b"begin\0aiperf-terminal:"));
    assert!(output.stdout.ends_with(b":7\0"));
}

#[tokio::test(flavor = "current_thread")]
async fn timeout_recycles_container_and_close_force_removes_once() {
    // This catches a timeout that leaves descendants in the old container or a
    // non-idempotent close that widens cleanup beyond the active container.
    let runtime = Rc::new(FakeRuntime::new([
        Vec::new(),
        vec![Bytes::from_static(b"next")],
    ]));
    let sandbox = DockerSessionSandbox::new(
        pinch_environment(),
        Some(PathBuf::from("/tmp/pinch-workspace")),
        trace(),
        ReplayRunIdentity::mint(RngRoot::new(Some(18)), "replay-run-18"),
        RealClock::new(),
        runtime.clone(),
        4096,
    )
    .expect("valid Pinch recipe creates a Docker sandbox");

    sandbox.open().await.expect("first container starts");
    let timed_out = sandbox
        .run("sleep forever", Some(1))
        .await
        .expect("timeout recovers the sandbox");
    assert!(timed_out.is_timed_out);
    let next = sandbox
        .run("printf next", None)
        .await
        .expect("recycled container accepts next command");
    assert_eq!(next.output, Bytes::from_static(b"next"));
    sandbox
        .close()
        .await
        .expect("close removes active container");
    sandbox.close().await.expect("second close is idempotent");

    assert_eq!(runtime.created.borrow().len(), 2);
    assert_eq!(runtime.remove_count.get(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn preflight_and_restart_cleanup_scope_docker_to_the_exact_run_label() {
    // This catches startup that skips image validation or restart cleanup that
    // filters a broad prefix and could remove another replay run's container.
    let runtime =
        Rc::new(FakeRuntime::new([]).with_listed([ContainerId::new("orphaned-recorded-agent")]));
    let identity = ReplayRunIdentity::mint(RngRoot::new(Some(19)), "  replay-run-19  ");

    preflight_docker_sandbox(runtime.as_ref(), &pinch_environment())
        .await
        .expect("Docker preflight inspects the resolved recipe image");
    cleanup_recorded_agent_containers(runtime.as_ref(), &identity)
        .await
        .expect("label-scoped restart cleanup removes matching containers");

    assert_eq!(
        runtime.inspected_images.borrow().as_slice(),
        ["aiperf-recorded-agent-pinchbench:v1"]
    );
    assert_eq!(
        runtime.label_queries.borrow().as_slice(),
        [(
            CONTAINER_RUN_LABEL_KEY.to_string(),
            "replay-run-19".to_string()
        )]
    );
    assert_eq!(runtime.remove_count.get(), 1);
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn docker_cli_calls_are_clock_bounded() {
    // This catches a Docker CLI invocation whose `output().await` can hold a
    // preflight or recovery path forever when the daemon command stops making
    // progress.
    use std::os::unix::fs::PermissionsExt as _;

    let script_root = tempfile::tempdir().expect("temporary fake Docker binary root");
    let script = script_root.path().join("docker");
    std::fs::write(&script, b"#!/bin/sh\nwhile :; do :; done\n")
        .expect("fake Docker binary is written");
    let mut permissions = std::fs::metadata(&script)
        .expect("fake Docker binary metadata exists")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&script, permissions).expect("fake Docker binary is executable");
    let runtime = DockerCliRuntime::with_binary_and_timeout(
        PathBuf::from(&script),
        RealClock::new(),
        1_000_000,
    );

    let error = runtime
        .inspect_image("unreachable-image")
        .await
        .expect_err("hung Docker CLI invocation reaches its clock deadline");
    assert!(error.to_string().contains("timed out"));
}

#[cfg(feature = "engine")]
#[tokio::test(flavor = "current_thread")]
async fn application_preflight_and_shutdown_cleanup_use_the_docker_run_identity_scope() {
    // This catches product composition that exposes a Docker seam but never
    // performs preflight or uses an unscoped cleanup path on shutdown.
    let runtime = Rc::new(FakeRuntime::new([]).with_listed([ContainerId::new("shutdown-orphan")]));
    let identity = ReplayRunIdentity::mint(RngRoot::new(Some(21)), "replay-run-21");

    preflight_recorded_agent_docker_environments(runtime.as_ref(), [&pinch_environment()])
        .await
        .expect("application preflight validates the selected recipe");
    cleanup_recorded_agent_docker_on_shutdown(runtime.as_ref(), &identity)
        .await
        .expect("shutdown cleanup selects only the exact run label");

    assert_eq!(
        runtime.inspected_images.borrow().as_slice(),
        ["aiperf-recorded-agent-pinchbench:v1"]
    );
    assert_eq!(
        runtime.label_queries.borrow().as_slice(),
        [(
            CONTAINER_RUN_LABEL_KEY.to_string(),
            "replay-run-21".to_string(),
        )]
    );
    assert_eq!(runtime.remove_count.get(), 1);
}
