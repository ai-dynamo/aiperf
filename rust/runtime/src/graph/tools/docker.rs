// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed, trace-local recorded-agent tool sandboxes.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::clock::Clock;
use crate::graph::driver::{TraceAgentInvocationContext, TraceIdentity};
use crate::graph::replay::ReplayRunIdentity;

use super::{
    EnvironmentToolDispatcher, GuardedToolCommandPolicy, LocalSessionSandbox, ProvisionedWorkspace,
    ResolvedTraceEnvironment, SegmentWorkspaceProvisioner, ToolBackendIdentity, ToolCommandResult,
    ToolDispatchContext, ToolDispatchError, ToolDispatchRequest, ToolDispatchResult,
    ToolDispatcher, ToolDispatcherFactory, ToolExecutionBackend, ToolSandbox,
    ToolSandboxCapabilities, ToolSandboxError, TraceEnvironmentError, TraceOpenContext,
    WorkspaceProvisioner, policy::contains_detaching_command,
};

/// Docker label key whose value is the exact controller-minted replay run label.
pub const CONTAINER_RUN_LABEL_KEY: &str = "aiperf.recorded-agent.run-label";

const TERMINAL_PREFIX: &[u8] = b"\0aiperf-terminal:";
const CONTAINER_REMOVE_TIMEOUT_NS: u64 = 10_000_000_000;
// `run_docker` may spend one command timeout terminating and one more reaping.
const CONTAINER_REMOVE_FENCE_TIMEOUT_NS: i64 = 20_000_000_000;
static CONTAINER_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const DEFAULT_TOOL_OUTPUT_LIMIT: usize = 1 << 20;
type DockerRuntimeFactory = dyn Fn(Rc<dyn Clock>) -> Rc<dyn ContainerRuntime> + Send + Sync;

/// Stock dispatcher factory selecting the resolved local or Docker backend at trace open.
#[derive(Clone)]
pub struct NativeToolDispatcherFactory {
    output_limit: usize,
    runtime: Arc<DockerRuntimeFactory>,
}

impl Default for NativeToolDispatcherFactory {
    fn default() -> Self {
        Self {
            output_limit: DEFAULT_TOOL_OUTPUT_LIMIT,
            runtime: Arc::new(|clock| Rc::new(DockerCliRuntime::new(clock))),
        }
    }
}

impl std::fmt::Debug for NativeToolDispatcherFactory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeToolDispatcherFactory")
            .field("output_limit", &self.output_limit)
            .finish_non_exhaustive()
    }
}

impl NativeToolDispatcherFactory {
    /// Build the stock factory with an explicit per-command captured-output bound.
    #[must_use]
    pub fn with_output_limit(output_limit: usize) -> Self {
        Self {
            output_limit: output_limit.max(1),
            ..Self::default()
        }
    }

    /// Build a factory around an injected worker-local Docker process boundary.
    pub fn with_runtime_factory(
        output_limit: usize,
        runtime: impl Fn(Rc<dyn Clock>) -> Rc<dyn ContainerRuntime> + Send + Sync + 'static,
    ) -> Self {
        Self {
            output_limit: output_limit.max(1),
            runtime: Arc::new(runtime),
        }
    }
}

impl ToolDispatcherFactory for NativeToolDispatcherFactory {
    fn create(&self, _trace_id: &str) -> Result<Rc<dyn ToolDispatcher>, ToolDispatchError> {
        Ok(Rc::new(DeferredNativeToolDispatcher {
            output_limit: self.output_limit,
            runtime: self.runtime.clone(),
            inner: RefCell::new(None),
            provisioned_workspace: RefCell::new(None),
        }))
    }
}

/// Backward-compatible name for the stock native dispatcher factory.
pub type DockerToolDispatcherFactory = NativeToolDispatcherFactory;

/// Trace-local dispatcher whose sandbox cannot be selected before `TraceOpenContext`.
struct DeferredNativeToolDispatcher {
    output_limit: usize,
    runtime: Arc<DockerRuntimeFactory>,
    inner: RefCell<Option<Rc<dyn ToolDispatcher>>>,
    provisioned_workspace: RefCell<Option<ProvisionedWorkspace>>,
}

/// Local-backend adapter that rewrites the authored mount path to the staged temp root.
///
/// Docker can mount the staged Pinch workspace at `/workspace` directly. The
/// host-local backend cannot bind an arbitrary absolute path into the process
/// namespace, so replay commands authored against `/workspace/...` must be
/// rewritten to the worker-local provisioned root before they hit the shell.
struct RebasedLocalToolSandbox {
    inner: LocalSessionSandbox,
    authored_workdir: String,
    provisioned_root: PathBuf,
}

impl RebasedLocalToolSandbox {
    fn new(
        inner: LocalSessionSandbox,
        authored_workdir: String,
        provisioned_root: PathBuf,
    ) -> Self {
        Self {
            inner,
            authored_workdir,
            provisioned_root,
        }
    }

    fn rewrite_command(&self, command: &str) -> String {
        let mut delimiters = VecDeque::new();
        let mut rewritten = String::with_capacity(command.len());
        for line in command.split_inclusive('\n') {
            if let Some((delimiter, strips_tabs)) = delimiters.front() {
                rewritten.push_str(line);
                let body_line = line.strip_suffix('\n').unwrap_or(line);
                let body_line = body_line.strip_suffix('\r').unwrap_or(body_line);
                let candidate = if *strips_tabs {
                    body_line.trim_start_matches('\t')
                } else {
                    body_line
                };
                if candidate == delimiter {
                    delimiters.pop_front();
                }
                continue;
            }
            delimiters.extend(heredoc_delimiters(line));
            self.rewrite_path_tokens(line, &mut rewritten);
        }
        rewritten
    }

    fn rewrite_path_tokens(&self, command: &str, rewritten: &mut String) {
        let root = self.authored_workdir.as_str();
        let replacement = self.provisioned_root.to_string_lossy();
        let mut remainder = command;
        while let Some(offset) = remainder.find(root) {
            let (prefix, candidate) = remainder.split_at(offset);
            let suffix = &candidate[root.len()..];
            let preceding = prefix.chars().next_back();
            let following = suffix.chars().next();
            let has_path_boundary_before = preceding.is_none_or(|character| {
                !character.is_ascii_alphanumeric() && !matches!(character, '_' | '-' | '.' | '/')
            });
            let has_path_boundary_after = following.is_none_or(|character| {
                matches!(
                    character,
                    '/' | ':' | ';' | ',' | ')' | '(' | '|' | '&' | '>' | '<'
                ) || character.is_whitespace()
                    || matches!(character, '\'' | '"')
            });
            rewritten.push_str(prefix);
            if has_path_boundary_before && has_path_boundary_after {
                rewritten.push_str(&replacement);
            } else {
                rewritten.push_str(root);
            }
            remainder = suffix;
        }
        rewritten.push_str(remainder);
    }
}

fn heredoc_delimiters(command_line: &str) -> Vec<(String, bool)> {
    let bytes = command_line.as_bytes();
    let mut delimiters = Vec::new();
    let mut index = 0;
    let mut quote = None;
    while index < bytes.len() {
        match (quote, bytes[index]) {
            (Some(active), character) if character == active => {
                quote = None;
                index += 1;
            }
            (Some(b'"'), b'\\') | (None, b'\\') => {
                index = (index + 2).min(bytes.len());
            }
            (Some(_), _) => index += 1,
            (None, character @ (b'\'' | b'"')) => {
                quote = Some(character);
                index += 1;
            }
            (None, b'<') if bytes.get(index + 1) == Some(&b'<') => {
                index += 2;
                if bytes.get(index) == Some(&b'<') {
                    index += 1;
                    continue;
                }
                let strips_tabs = bytes.get(index) == Some(&b'-');
                index += usize::from(strips_tabs);
                while bytes
                    .get(index)
                    .is_some_and(|character| matches!(character, b' ' | b'\t'))
                {
                    index += 1;
                }
                let delimiter = match bytes.get(index) {
                    Some(quote @ (b'\'' | b'"')) => {
                        index += 1;
                        let start = index;
                        while bytes.get(index).is_some_and(|character| character != quote) {
                            index += 1;
                        }
                        let delimiter = command_line[start..index].to_string();
                        index += usize::from(index < bytes.len());
                        delimiter
                    }
                    Some(_) => {
                        let start = index;
                        while bytes.get(index).is_some_and(|character| {
                            !character.is_ascii_whitespace()
                                && !matches!(
                                    character,
                                    b';' | b'|' | b'&' | b'(' | b')' | b'<' | b'>'
                                )
                        }) {
                            index += 1;
                        }
                        command_line[start..index].to_string()
                    }
                    None => String::new(),
                };
                if !delimiter.is_empty() {
                    delimiters.push((delimiter, strips_tabs));
                }
            }
            (None, _) => index += 1,
        }
    }
    delimiters
}

#[async_trait(?Send)]
impl ToolSandbox for RebasedLocalToolSandbox {
    fn backend_identity(&self) -> ToolBackendIdentity {
        self.inner.backend_identity()
    }

    async fn open(&self) -> Result<(), ToolSandboxError> {
        self.inner.open().await
    }

    async fn run(
        &self,
        command: &str,
        timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        let rewritten = self.rewrite_command(command);
        self.inner.run(&rewritten, timeout_ns).await
    }

    fn recovers_timed_out_commands(&self) -> bool {
        self.inner.recovers_timed_out_commands()
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        self.inner.recycle().await
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        self.inner.close().await
    }
}

#[async_trait(?Send)]
impl ToolDispatcher for DeferredNativeToolDispatcher {
    fn backend_identity(&self) -> ToolBackendIdentity {
        self.inner
            .borrow()
            .as_ref()
            .map_or(ToolBackendIdentity::Local, |dispatcher| {
                dispatcher.backend_identity()
            })
    }

    async fn open_trace(&self, context: TraceOpenContext<'_>) -> Result<(), ToolDispatchError> {
        if self.inner.borrow().is_some() {
            return Err(ToolDispatchError::new(
                "native tool dispatcher trace is already open",
            ));
        }
        let Some(environment) = context.environment else {
            if context.workspace.is_some() {
                return Err(ToolDispatchError::new(
                    "trace workspace was provided without a resolved environment",
                ));
            }
            return Ok(());
        };
        if context.workspace != Some(&environment.workspace) {
            return Err(ToolDispatchError::new(
                "trace-open workspace does not match its resolved environment recipe",
            ));
        }
        let capabilities = match environment.backend {
            ToolExecutionBackend::Local => ToolSandboxCapabilities {
                has_persistent_workspace: true,
                has_workspace_materialization: true,
                has_network_disabled: false,
                has_timeout_recycle: true,
            },
            ToolExecutionBackend::Docker => ToolSandboxCapabilities {
                has_persistent_workspace: true,
                has_workspace_materialization: true,
                has_network_disabled: true,
                has_timeout_recycle: true,
            },
        };
        capabilities.validate(environment)?;
        let provisioned_workspace = if environment.workspace.mount_workspace {
            Some(
                SegmentWorkspaceProvisioner::new(context.segments)
                    .provision(&environment.workspace)
                    .await?,
            )
        } else {
            None
        };
        let sandbox: Rc<dyn ToolSandbox> = match environment.backend {
            ToolExecutionBackend::Local => {
                let mut workspace = environment.workspace.clone();
                if let Some(provisioned) = provisioned_workspace.as_ref() {
                    let authored_workdir = workspace.workdir.clone();
                    workspace.workdir = provisioned.root.to_string_lossy().into_owned();
                    workspace.mount_workspace = false;
                    Rc::new(RebasedLocalToolSandbox::new(
                        LocalSessionSandbox::with_tokio_processes(
                            workspace,
                            context.clock.clone(),
                            self.output_limit,
                        ),
                        authored_workdir,
                        provisioned.root.clone(),
                    ))
                } else {
                    Rc::new(LocalSessionSandbox::with_tokio_processes(
                        workspace,
                        context.clock.clone(),
                        self.output_limit,
                    ))
                }
            }
            ToolExecutionBackend::Docker => Rc::new(DockerSessionSandbox::new_for_invocation(
                environment.clone(),
                provisioned_workspace
                    .as_ref()
                    .map(|workspace| workspace.root.clone()),
                context.trace.clone(),
                context.invocation.clone(),
                context.clock.clone(),
                (self.runtime)(context.clock.clone()),
                self.output_limit,
            )?),
        };
        let dispatcher: Rc<dyn ToolDispatcher> = Rc::new(EnvironmentToolDispatcher::new(
            sandbox,
            Rc::new(GuardedToolCommandPolicy),
        ));
        dispatcher.open_trace(context).await?;
        self.provisioned_workspace.replace(provisioned_workspace);
        self.inner.replace(Some(dispatcher));
        Ok(())
    }

    async fn dispatch(
        &self,
        request: ToolDispatchRequest,
        context: &ToolDispatchContext,
    ) -> Result<ToolDispatchResult, ToolDispatchError> {
        let dispatcher = self.inner.borrow().clone().ok_or_else(|| {
            ToolDispatchError::new("recorded-agent tool dispatch has no open environment")
        })?;
        dispatcher.dispatch(request, context).await
    }

    async fn close_trace(&self, trace: &TraceIdentity) -> Result<(), ToolDispatchError> {
        let Some(dispatcher) = self.inner.take() else {
            self.provisioned_workspace.take();
            return Ok(());
        };
        let result = dispatcher.close_trace(trace).await;
        self.provisioned_workspace.take();
        result
    }
}

/// Opaque Docker container identifier returned after successful start.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContainerId(String);

impl ContainerId {
    /// Construct an identifier from the Docker runtime's opaque response.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the opaque Docker identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One bind mount visible to a trace-owned container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerMount {
    /// Worker-local workspace source path.
    pub host_path: PathBuf,
    /// Recipe-authored container destination.
    pub container_path: String,
}

/// Fully argv-safe detached-container creation inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerCreateSpec {
    /// Recipe-selected image, inspected before creation.
    pub image: String,
    /// Sanitized unique name used only for operator diagnosis.
    pub name: String,
    /// Persisted labels used for recovery cleanup.
    pub labels: BTreeMap<String, String>,
    /// Whether Docker must disable all container networking.
    pub has_network_disabled: bool,
    /// Explicit recipe mounts; SWE-Bench intentionally has none.
    pub mounts: Vec<ContainerMount>,
    /// Recipe-authored current working directory.
    pub workdir: String,
}

impl ContainerCreateSpec {
    /// Compose an exact Docker creation contract from resolved worker-local inputs.
    pub fn from_environment(
        environment: &ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: &TraceIdentity,
        run_identity: &ReplayRunIdentity,
    ) -> Result<Self, ToolSandboxError> {
        Self::from_cleanup_label(
            environment,
            workspace_root,
            trace,
            run_identity.label(),
            "aiperf-recorded-agent",
        )
    }

    /// Compose a Docker creation contract from a trace-local invocation scope.
    pub fn from_invocation(
        environment: &ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: &TraceIdentity,
        invocation: &TraceAgentInvocationContext,
    ) -> Result<Self, ToolSandboxError> {
        Self::from_cleanup_label(
            environment,
            workspace_root,
            trace,
            invocation.cleanup_label(),
            "aiperf-native-graph-agent",
        )
    }

    fn from_cleanup_label(
        environment: &ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: &TraceIdentity,
        cleanup_label: &str,
        container_name_prefix: &str,
    ) -> Result<Self, ToolSandboxError> {
        let mut mounts = Vec::new();
        if environment.workspace.mount_workspace {
            let root = workspace_root.ok_or_else(|| {
                ToolSandboxError::new(
                    "Docker sandbox recipe requires a provisioned workspace mount",
                )
            })?;
            mounts.push(ContainerMount {
                host_path: root,
                container_path: environment.workspace.workdir.clone(),
            });
        }
        let label = cleanup_label.trim();
        if label.is_empty() {
            return Err(ToolSandboxError::new(
                "Docker cleanup requires a nonempty invocation label",
            ));
        }
        let sequence = CONTAINER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let trace_slug = sanitize_container_component(&trace.trace_id);
        let name = format!("{container_name_prefix}-{trace_slug}-{sequence:x}");
        Ok(Self {
            image: environment.image.clone(),
            name,
            labels: BTreeMap::from([(CONTAINER_RUN_LABEL_KEY.to_string(), label.to_string())]),
            has_network_disabled: true,
            mounts,
            workdir: environment.workspace.workdir.clone(),
        })
    }
}

/// Async output handle for one argv-only `docker exec -i` invocation.
#[async_trait(?Send)]
pub trait FramedCommandIo {
    /// Append the next stdout bytes and return their count.
    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError>;
    /// Reap or abandon this exec invocation after its terminal frame or recycle.
    async fn close(&mut self) -> Result<(), ToolSandboxError>;
}

/// Injectable Docker process boundary used by the sandbox and cleanup hooks.
#[async_trait(?Send)]
pub trait ContainerRuntime {
    /// Refuse a missing or otherwise unusable recipe image before measurement.
    async fn inspect_image(&self, image: &str) -> Result<(), ToolSandboxError>;
    /// Create one detached recipe container without starting it.
    async fn create(&self, spec: &ContainerCreateSpec) -> Result<ContainerId, ToolSandboxError>;
    /// Start one previously created detached container.
    async fn start(&self, id: &ContainerId) -> Result<(), ToolSandboxError>;
    /// Open one stdin-attached exec process using the supplied argv without a shell.
    async fn open_exec(
        &self,
        id: &ContainerId,
        argv: &[OsString],
    ) -> Result<Box<dyn FramedCommandIo>, ToolSandboxError>;
    /// Force-remove one known container with an explicit cleanup bound.
    async fn force_remove(&self, id: &ContainerId, timeout_ns: u64)
    -> Result<(), ToolSandboxError>;
    /// Start best-effort removal when an opening future is dropped mid-command.
    fn force_remove_on_drop(&self, _id: &ContainerId) {}
    /// Return only containers carrying this exact label key and value.
    async fn list_by_label(
        &self,
        key: &str,
        value: &str,
    ) -> Result<Vec<ContainerId>, ToolSandboxError>;
}

/// Stock argv-only Docker CLI runtime with Clock-bounded daemon commands.
#[derive(Clone)]
pub struct DockerCliRuntime {
    binary: PathBuf,
    clock: Rc<dyn Clock>,
    command_timeout_ns: u64,
}

impl DockerCliRuntime {
    /// Construct the production Docker CLI runtime on this trace's clock.
    #[must_use]
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Self::with_binary_and_timeout("docker", clock, CONTAINER_REMOVE_TIMEOUT_NS)
    }

    /// Construct a CLI runtime with a testable binary path and command bound.
    #[must_use]
    pub fn with_binary_and_timeout(
        binary: impl Into<PathBuf>,
        clock: Rc<dyn Clock>,
        command_timeout_ns: u64,
    ) -> Self {
        Self {
            binary: binary.into(),
            clock,
            command_timeout_ns: command_timeout_ns.max(1),
        }
    }

    async fn run(
        &self,
        args: impl IntoIterator<Item = OsString>,
    ) -> Result<Vec<u8>, ToolSandboxError> {
        run_docker(
            &self.binary,
            self.clock.clone(),
            self.command_timeout_ns,
            args,
        )
        .await
    }
}

#[async_trait(?Send)]
impl ContainerRuntime for DockerCliRuntime {
    async fn inspect_image(&self, image: &str) -> Result<(), ToolSandboxError> {
        self.run([
            OsString::from("image"),
            OsString::from("inspect"),
            OsString::from(image),
        ])
        .await
        .map(|_| ())
    }

    async fn create(&self, spec: &ContainerCreateSpec) -> Result<ContainerId, ToolSandboxError> {
        if !spec.has_network_disabled {
            return Err(ToolSandboxError::new(
                "Docker recorded-agent containers must use network=none",
            ));
        }
        let mut create = vec![
            OsString::from("create"),
            OsString::from("--name"),
            OsString::from(&spec.name),
            OsString::from("--network"),
            OsString::from("none"),
            OsString::from("--workdir"),
            OsString::from(&spec.workdir),
        ];
        for (key, value) in &spec.labels {
            create.push(OsString::from("--label"));
            create.push(OsString::from(format!("{key}={value}")));
        }
        for mount in &spec.mounts {
            create.push(OsString::from("--mount"));
            create.push(OsString::from(format!(
                "type=bind,src={},dst={}",
                mount.host_path.display(),
                mount.container_path
            )));
        }
        create.push(OsString::from(&spec.image));
        create.push(OsString::from("sleep"));
        create.push(OsString::from("infinity"));
        let id = String::from_utf8(self.run(create).await?).map_err(|error| {
            ToolSandboxError::new(format!("Docker returned a non-UTF-8 container id: {error}"))
        })?;
        let id = id.trim();
        if id.is_empty() {
            return Err(ToolSandboxError::new(
                "Docker create returned an empty container id",
            ));
        }
        Ok(ContainerId::new(id))
    }

    async fn start(&self, id: &ContainerId) -> Result<(), ToolSandboxError> {
        self.run([OsString::from("start"), OsString::from(id.as_str())])
            .await
            .map(|_| ())
    }

    async fn open_exec(
        &self,
        id: &ContainerId,
        argv: &[OsString],
    ) -> Result<Box<dyn FramedCommandIo>, ToolSandboxError> {
        let mut command = Command::new("docker");
        command.arg("exec").arg("-i").arg(id.as_str()).args(argv);
        command.stdin(std::process::Stdio::null());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::null());
        let mut child = command.spawn().map_err(|error| {
            ToolSandboxError::new(format!(
                "cannot start Docker exec for {}: {error}",
                id.as_str()
            ))
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolSandboxError::new("Docker exec did not expose stdout"))?;
        Ok(Box::new(DockerExecIo {
            child,
            stdout,
            clock: self.clock.clone(),
            command_timeout_ns: self.command_timeout_ns,
        }))
    }

    async fn force_remove(
        &self,
        id: &ContainerId,
        timeout_ns: u64,
    ) -> Result<(), ToolSandboxError> {
        let timeout_seconds = timeout_ns.saturating_add(999_999_999) / 1_000_000_000;
        run_docker(
            &self.binary,
            self.clock.clone(),
            timeout_ns.max(1),
            [
                OsString::from("rm"),
                OsString::from("--force"),
                OsString::from("--time"),
                OsString::from(timeout_seconds.max(1).to_string()),
                OsString::from(id.as_str()),
            ],
        )
        .await
        .map(|_| ())
    }

    fn force_remove_on_drop(&self, id: &ContainerId) {
        let mut command = std::process::Command::new(&self.binary);
        command
            .arg("rm")
            .arg("--force")
            .arg(id.as_str())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if let Err(error) = command.spawn() {
            tracing::warn!(
                error = %error,
                container_id = id.as_str(),
                "could not launch Docker cleanup after trace-open cancellation"
            );
        }
    }

    async fn list_by_label(
        &self,
        key: &str,
        value: &str,
    ) -> Result<Vec<ContainerId>, ToolSandboxError> {
        let output = self
            .run([
                OsString::from("ps"),
                OsString::from("--all"),
                OsString::from("--quiet"),
                OsString::from("--filter"),
                OsString::from(format!("label={key}={value}")),
            ])
            .await?;
        let output = String::from_utf8(output).map_err(|error| {
            ToolSandboxError::new(format!("Docker listed non-UTF-8 container ids: {error}"))
        })?;
        Ok(output
            .lines()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(ContainerId::new)
            .collect())
    }
}

struct DockerExecIo {
    child: Child,
    stdout: ChildStdout,
    clock: Rc<dyn Clock>,
    command_timeout_ns: u64,
}

#[async_trait(?Send)]
impl FramedCommandIo for DockerExecIo {
    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        self.stdout.read_buf(output).await.map_err(|error| {
            ToolSandboxError::new(format!("cannot read Docker exec output: {error}"))
        })
    }

    async fn close(&mut self) -> Result<(), ToolSandboxError> {
        if let Some(status) = self.child.try_wait().map_err(|error| {
            ToolSandboxError::new(format!("cannot inspect Docker exec status: {error}"))
        })? {
            if status.success() {
                return Ok(());
            }
            return Err(ToolSandboxError::new(format!(
                "Docker exec exited unsuccessfully: {status}"
            )));
        }
        self.child.start_kill().map_err(|error| {
            ToolSandboxError::new(format!("cannot terminate Docker exec: {error}"))
        })?;
        wait_for_child(
            &mut self.child,
            self.clock.clone(),
            self.command_timeout_ns,
            "Docker exec",
        )
        .await
    }
}

/// Validate Docker's required isolation capabilities and resolve its image now.
pub async fn preflight_docker_sandbox(
    runtime: &dyn ContainerRuntime,
    environment: &ResolvedTraceEnvironment,
) -> Result<(), TraceEnvironmentError> {
    ToolSandboxCapabilities {
        has_persistent_workspace: true,
        has_workspace_materialization: true,
        has_network_disabled: true,
        has_timeout_recycle: true,
    }
    .validate(environment)?;
    runtime
        .inspect_image(&environment.image)
        .await
        .map_err(|error| {
            TraceEnvironmentError::new(format!("Docker image preflight failed: {error}"))
        })
}

/// Force-remove only containers belonging to this exact persisted replay run.
pub async fn cleanup_recorded_agent_containers(
    runtime: &dyn ContainerRuntime,
    run_identity: &ReplayRunIdentity,
) -> Result<(), ToolSandboxError> {
    let label = run_identity.label().trim();
    if label.is_empty() {
        return Err(ToolSandboxError::new(
            "refusing Docker cleanup for an empty replay run label",
        ));
    }
    for id in runtime
        .list_by_label(CONTAINER_RUN_LABEL_KEY, label)
        .await?
    {
        runtime
            .force_remove(&id, CONTAINER_REMOVE_TIMEOUT_NS)
            .await?;
    }
    Ok(())
}

enum ContainerState {
    Idle,
    Live(ContainerId),
}

struct ContainerOpeningGuard<'a> {
    runtime: &'a dyn ContainerRuntime,
    cleanup_target: Option<ContainerId>,
}

impl<'a> ContainerOpeningGuard<'a> {
    fn new(runtime: &'a dyn ContainerRuntime, container_name: &str) -> Self {
        Self {
            runtime,
            cleanup_target: Some(ContainerId::new(container_name)),
        }
    }

    fn created(&mut self, id: ContainerId) {
        self.cleanup_target = Some(id);
    }

    fn disarm(&mut self) -> Result<ContainerId, ToolSandboxError> {
        self.cleanup_target.take().ok_or_else(|| {
            ToolSandboxError::new("Docker opening guard lost its container identity")
        })
    }

    async fn cleanup(&mut self) -> Result<(), ToolSandboxError> {
        let id = self.disarm()?;
        self.runtime
            .force_remove(&id, CONTAINER_REMOVE_TIMEOUT_NS)
            .await
    }
}

impl Drop for ContainerOpeningGuard<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.cleanup_target.take() {
            self.runtime.force_remove_on_drop(&id);
        }
    }
}

struct ContainerRemovalGuard<'a> {
    runtime: &'a dyn ContainerRuntime,
    cleanup_target: Option<ContainerId>,
}

impl<'a> ContainerRemovalGuard<'a> {
    fn new(runtime: &'a dyn ContainerRuntime, id: ContainerId) -> Self {
        Self {
            runtime,
            cleanup_target: Some(id),
        }
    }

    fn target(&self) -> Result<&ContainerId, ToolSandboxError> {
        self.cleanup_target.as_ref().ok_or_else(|| {
            ToolSandboxError::new("Docker removal guard lost its container identity")
        })
    }

    fn disarm(&mut self) -> Result<ContainerId, ToolSandboxError> {
        self.cleanup_target.take().ok_or_else(|| {
            ToolSandboxError::new("Docker removal guard lost its container identity")
        })
    }
}

impl Drop for ContainerRemovalGuard<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.cleanup_target.take() {
            self.runtime.force_remove_on_drop(&id);
        }
    }
}

/// One trace-owned Docker container and its independently framed exec commands.
pub struct DockerSessionSandbox {
    environment: ResolvedTraceEnvironment,
    create_spec: ContainerCreateSpec,
    clock: Rc<dyn Clock>,
    runtime: Rc<dyn ContainerRuntime>,
    output_limit: usize,
    container: RefCell<ContainerState>,
    command_gate: Mutex<()>,
}

/// Worker-local factory retaining the Docker owners for one trace composition.
#[derive(Clone)]
pub struct DockerSandboxFactory {
    runtime: Rc<dyn ContainerRuntime>,
    clock: Rc<dyn Clock>,
    output_limit: usize,
}

impl DockerSandboxFactory {
    /// Freeze worker-local Docker dependencies before trace-specific creation.
    pub fn new(
        runtime: Rc<dyn ContainerRuntime>,
        clock: Rc<dyn Clock>,
        output_limit: usize,
    ) -> Self {
        Self {
            runtime,
            clock,
            output_limit,
        }
    }

    /// Create one Docker sandbox from already resolved and provisioned inputs.
    pub fn create(
        &self,
        environment: ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: TraceIdentity,
        run_identity: ReplayRunIdentity,
    ) -> Result<Rc<DockerSessionSandbox>, ToolSandboxError> {
        Ok(Rc::new(DockerSessionSandbox::new(
            environment,
            workspace_root,
            trace,
            run_identity,
            self.clock.clone(),
            self.runtime.clone(),
            self.output_limit,
        )?))
    }
}

impl DockerSessionSandbox {
    /// Construct one sandbox without probing Docker; [`ToolSandbox::open`] preflights it.
    pub fn new(
        environment: ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: TraceIdentity,
        run_identity: ReplayRunIdentity,
        clock: Rc<dyn Clock>,
        runtime: Rc<dyn ContainerRuntime>,
        output_limit: usize,
    ) -> Result<Self, ToolSandboxError> {
        let create_spec = ContainerCreateSpec::from_environment(
            &environment,
            workspace_root,
            &trace,
            &run_identity,
        )?;
        Ok(Self {
            environment,
            create_spec,
            clock,
            runtime,
            output_limit,
            container: RefCell::new(ContainerState::Idle),
            command_gate: Mutex::new(()),
        })
    }

    /// Construct one sandbox from a non-replay trace invocation scope.
    pub fn new_for_invocation(
        environment: ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: TraceIdentity,
        invocation: TraceAgentInvocationContext,
        clock: Rc<dyn Clock>,
        runtime: Rc<dyn ContainerRuntime>,
        output_limit: usize,
    ) -> Result<Self, ToolSandboxError> {
        let create_spec = ContainerCreateSpec::from_invocation(
            &environment,
            workspace_root,
            &trace,
            &invocation,
        )?;
        Ok(Self {
            environment,
            create_spec,
            clock,
            runtime,
            output_limit,
            container: RefCell::new(ContainerState::Idle),
            command_gate: Mutex::new(()),
        })
    }

    /// Construct the stock Docker CLI-backed sandbox.
    pub fn with_docker_cli(
        environment: ResolvedTraceEnvironment,
        workspace_root: Option<PathBuf>,
        trace: TraceIdentity,
        run_identity: ReplayRunIdentity,
        clock: Rc<dyn Clock>,
        output_limit: usize,
    ) -> Result<Self, ToolSandboxError> {
        Self::new(
            environment,
            workspace_root,
            trace,
            run_identity,
            clock.clone(),
            Rc::new(DockerCliRuntime::new(clock.clone())),
            output_limit,
        )
    }

    async fn open_unlocked(&self) -> Result<(), ToolSandboxError> {
        if matches!(*self.container.borrow(), ContainerState::Live(_)) {
            return Ok(());
        }
        preflight_docker_sandbox(self.runtime.as_ref(), &self.environment)
            .await
            .map_err(|error| ToolSandboxError::new(error.to_string()))?;
        let mut opening = ContainerOpeningGuard::new(self.runtime.as_ref(), &self.create_spec.name);
        let id = self.runtime.create(&self.create_spec).await?;
        opening.created(id.clone());
        if let Err(start_error) = self.runtime.start(&id).await {
            let cleanup = opening.cleanup().await;
            return match cleanup {
                Ok(()) => Err(start_error),
                Err(cleanup_error) => Err(ToolSandboxError::new(format!(
                    "{start_error}; Docker cleanup after failed container start also failed: {cleanup_error}"
                ))),
            };
        }
        self.container
            .replace(ContainerState::Live(opening.disarm()?));
        Ok(())
    }

    fn live_container(&self) -> Result<ContainerId, ToolSandboxError> {
        match &*self.container.borrow() {
            ContainerState::Live(id) => Ok(id.clone()),
            ContainerState::Idle => Err(ToolSandboxError::new(
                "Docker sandbox has no live container after open",
            )),
        }
    }

    async fn remove_unlocked(&self) -> Result<(), ToolSandboxError> {
        let previous = std::mem::replace(&mut *self.container.borrow_mut(), ContainerState::Idle);
        let ContainerState::Live(id) = previous else {
            return Ok(());
        };
        let mut removal = ContainerRemovalGuard::new(self.runtime.as_ref(), id);
        let result = {
            let remove = self
                .runtime
                .force_remove(removal.target()?, CONTAINER_REMOVE_TIMEOUT_NS);
            tokio::pin!(remove);
            let timeout = self.clock.clone().sleep(CONTAINER_REMOVE_FENCE_TIMEOUT_NS);
            tokio::pin!(timeout);
            tokio::select! {
                biased;
                result = &mut remove => Some(result),
                () = &mut timeout => None,
            }
        };
        match result {
            Some(Ok(())) => {
                removal.disarm()?;
                Ok(())
            }
            Some(Err(error)) => Err(error),
            None => Err(ToolSandboxError::new(
                "Docker container cleanup timed out after 20 seconds",
            )),
        }
    }

    async fn recycle_unlocked(&self) -> Result<(), ToolSandboxError> {
        self.remove_unlocked().await?;
        self.open_unlocked().await
    }
}

#[async_trait(?Send)]
impl ToolSandbox for DockerSessionSandbox {
    fn backend_identity(&self) -> crate::graph::tools::ToolBackendIdentity {
        crate::graph::tools::ToolBackendIdentity::Docker(self.environment.image.clone())
    }

    async fn open(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.open_unlocked().await
    }

    async fn run(
        &self,
        command: &str,
        timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        if contains_detaching_command(command) {
            return Err(ToolSandboxError::new(
                "recorded-agent replay blocked a detaching command to preserve sandbox containment",
            ));
        }
        self.open_unlocked().await?;
        let id = self.live_container()?;
        let sentinel = uuid::Uuid::new_v4().simple().to_string();
        let argv = command_argv(&self.environment.workspace.interpreter, command, &sentinel)?;
        let started_ns = self.clock.now_ns();
        let mut io = self.runtime.open_exec(&id, &argv).await?;
        let timeout_ns = timeout_ns.or(Some(self.environment.workspace.command_timeout_ns));
        match terminal_result(
            &mut *io,
            &sentinel,
            timeout_ns,
            self.clock.clone(),
            self.output_limit,
        )
        .await
        {
            Ok(CommandEnd::Completed {
                output,
                exit_code,
                is_output_truncated,
            }) => {
                io.close().await?;
                Ok(ToolCommandResult {
                    output: Bytes::from(output),
                    exit_code,
                    duration_ns: elapsed_ns(started_ns, self.clock.now_ns()),
                    is_timed_out: false,
                    is_output_truncated,
                })
            }
            Ok(CommandEnd::TimedOut {
                output,
                is_output_truncated,
            }) => {
                let duration_ns = elapsed_ns(started_ns, self.clock.now_ns());
                self.recycle_unlocked().await?;
                let _ = io.close().await;
                Ok(ToolCommandResult {
                    output: Bytes::from(output),
                    exit_code: 124,
                    duration_ns,
                    is_timed_out: true,
                    is_output_truncated,
                })
            }
            Err(error) => {
                let _ = io.close().await;
                let _ = self.remove_unlocked().await;
                Err(error)
            }
        }
    }

    fn recovers_timed_out_commands(&self) -> bool {
        true
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.recycle_unlocked().await
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.remove_unlocked().await
    }
}

enum CommandEnd {
    Completed {
        output: Vec<u8>,
        exit_code: i32,
        is_output_truncated: bool,
    },
    TimedOut {
        output: Vec<u8>,
        is_output_truncated: bool,
    },
}

async fn terminal_result(
    io: &mut dyn FramedCommandIo,
    sentinel: &str,
    timeout_ns: Option<u64>,
    clock: Rc<dyn Clock>,
    output_limit: usize,
) -> Result<CommandEnd, ToolSandboxError> {
    let timeout_ns = timeout_ns.filter(|timeout| *timeout > 0);
    let timer = timeout_ns.map(|timeout| clock.clone().sleep(timeout.min(i64::MAX as u64) as i64));
    tokio::pin!(timer);
    let frame_prefix = [TERMINAL_PREFIX, sentinel.as_bytes(), b":"].concat();
    let mut wire = BytesMut::new();
    let mut output = Vec::new();
    let mut is_output_truncated = false;
    loop {
        let mut chunk = BytesMut::with_capacity(4096);
        let read = async {
            let count = io.read(&mut chunk).await?;
            Ok::<_, ToolSandboxError>((count, chunk))
        };
        let (count, chunk) = match timer.as_mut().as_pin_mut() {
            Some(timer) => tokio::select! {
                result = read => result?,
                () = timer => {
                    capture_output(&wire, &mut output, &mut is_output_truncated, output_limit);
                    return Ok(CommandEnd::TimedOut { output, is_output_truncated });
                }
            },
            None => read.await?,
        };
        if count == 0 {
            return Err(ToolSandboxError::new(
                "Docker exec reached EOF before its terminal frame",
            ));
        }
        wire.extend_from_slice(&chunk);
        if let Some(exit_code) = consume_terminal_frame(
            &mut wire,
            &frame_prefix,
            &mut output,
            &mut is_output_truncated,
            output_limit,
        )? {
            return Ok(CommandEnd::Completed {
                output,
                exit_code,
                is_output_truncated,
            });
        }
    }
}

fn command_argv(
    interpreter: &[String],
    command: &str,
    sentinel: &str,
) -> Result<Vec<OsString>, ToolSandboxError> {
    let Some((program, arguments)) = interpreter.split_first() else {
        return Err(ToolSandboxError::new(
            "Docker sandbox recipe has no command interpreter",
        ));
    };
    let mut argv = Vec::with_capacity(interpreter.len() + 1);
    argv.push(OsString::from(program));
    argv.extend(arguments.iter().map(OsString::from));
    argv.push(OsString::from(format!(
        "exec 2>&1\n(\n{command}\n)\nstatus=$?\nprintf '\\0aiperf-terminal:{sentinel}:%d\\0' \"$status\"\nexit 0"
    )));
    Ok(argv)
}

fn sanitize_container_component(raw: &str) -> String {
    let mut value = String::new();
    let mut previous_was_separator = false;
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() {
            value.push(character.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !previous_was_separator {
            value.push('-');
            previous_was_separator = true;
        }
    }
    let value = value.trim_matches('-');
    if value.is_empty() {
        "trace".to_string()
    } else {
        value.chars().take(48).collect()
    }
}

fn elapsed_ns(started_ns: i64, ended_ns: i64) -> u64 {
    ended_ns.saturating_sub(started_ns) as u64
}

fn consume_terminal_frame(
    wire: &mut BytesMut,
    frame_prefix: &[u8],
    output: &mut Vec<u8>,
    is_output_truncated: &mut bool,
    output_limit: usize,
) -> Result<Option<i32>, ToolSandboxError> {
    if let Some(index) = find_subsequence(wire, frame_prefix) {
        capture_output(&wire[..index], output, is_output_truncated, output_limit);
        wire.advance(index);
        let frame = &wire[frame_prefix.len()..];
        let Some(status_end) = frame.iter().position(|byte| *byte == b'\0') else {
            return Ok(None);
        };
        let exit_code = std::str::from_utf8(&frame[..status_end])
            .ok()
            .and_then(|status| status.parse::<i32>().ok())
            .ok_or_else(|| {
                ToolSandboxError::new("Docker exec emitted a malformed terminal frame")
            })?;
        if frame.len() != status_end + 1 {
            return Err(ToolSandboxError::new(
                "Docker exec emitted bytes after its terminal frame",
            ));
        }
        return Ok(Some(exit_code));
    }
    let retained = frame_prefix.len().saturating_sub(1);
    let capture_end = wire.len().saturating_sub(retained);
    if capture_end > 0 {
        capture_output(
            &wire[..capture_end],
            output,
            is_output_truncated,
            output_limit,
        );
        wire.advance(capture_end);
    }
    Ok(None)
}

fn capture_output(
    bytes: &[u8],
    output: &mut Vec<u8>,
    is_output_truncated: &mut bool,
    output_limit: usize,
) {
    let remaining = output_limit.saturating_sub(output.len());
    let captured = bytes.len().min(remaining);
    output.extend_from_slice(&bytes[..captured]);
    *is_output_truncated |= captured != bytes.len();
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}

async fn run_docker(
    binary: &Path,
    clock: Rc<dyn Clock>,
    timeout_ns: u64,
    args: impl IntoIterator<Item = OsString>,
) -> Result<Vec<u8>, ToolSandboxError> {
    let mut command = Command::new(binary);
    command.args(args);
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    command.kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| ToolSandboxError::new(format!("cannot run Docker CLI: {error}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ToolSandboxError::new("Docker CLI did not expose stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ToolSandboxError::new("Docker CLI did not expose stderr"))?;
    let completed = {
        let wait = async {
            let mut stdout = stdout;
            let mut stderr = stderr;
            let mut output = Vec::new();
            let mut error_output = Vec::new();
            let (status, output_result, error_result) = tokio::join!(
                child.wait(),
                stdout.read_to_end(&mut output),
                stderr.read_to_end(&mut error_output),
            );
            let status = status.map_err(|error| {
                ToolSandboxError::new(format!("cannot wait for Docker CLI: {error}"))
            })?;
            output_result.map_err(|error| {
                ToolSandboxError::new(format!("cannot read Docker CLI stdout: {error}"))
            })?;
            error_result.map_err(|error| {
                ToolSandboxError::new(format!("cannot read Docker CLI stderr: {error}"))
            })?;
            Ok::<_, ToolSandboxError>((status, output, error_output))
        };
        tokio::pin!(wait);
        tokio::select! {
            result = &mut wait => Some(result?),
            () = clock.clone().sleep(timeout_ns.min(i64::MAX as u64) as i64) => None,
        }
    };
    let Some((status, stdout, stderr)) = completed else {
        let _ = child.start_kill();
        let _ = wait_for_child(&mut child, clock, timeout_ns, "Docker CLI").await;
        return Err(ToolSandboxError::new("Docker CLI timed out"));
    };
    if status.success() {
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&stderr);
        Err(ToolSandboxError::new(format!(
            "Docker CLI exited with {}: {}",
            status,
            stderr.trim()
        )))
    }
}

async fn wait_for_child(
    child: &mut Child,
    clock: Rc<dyn Clock>,
    timeout_ns: u64,
    operation: &str,
) -> Result<(), ToolSandboxError> {
    tokio::select! {
        result = child.wait() => result
            .map(|_| ())
            .map_err(|error| ToolSandboxError::new(format!("cannot reap {operation}: {error}"))),
        () = clock.sleep(timeout_ns.min(i64::MAX as u64) as i64) => Err(ToolSandboxError::new(format!("{operation} timed out while reaping"))),
    }
}
