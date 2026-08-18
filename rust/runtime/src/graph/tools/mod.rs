// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-dispatch contracts used by recorded-agent trace drivers.

pub mod dispatch;
pub mod docker;
pub mod environment;
pub mod environment_stepper;
pub mod local;
pub mod policy;
pub mod workspace;

pub use crate::graph::recorded::agent_recording::resolve_recorded_environment;
pub use dispatch::{
    AgentObservationFormatter, AgentObservationFormatterFactory, AgentToolCall,
    AgentToolCallDecoder, AgentToolCallDecoderFactory, EnvironmentToolDispatcher,
    InMemoryAgentObservationFormatter, InMemoryAgentObservationFormatterFactory,
    InMemoryAgentToolCallDecoder, InMemoryAgentToolCallDecoderFactory, InMemoryToolDispatcher,
    InMemoryToolDispatcherFactory, SandboxCreateContext, ToolBackendIdentity, ToolDispatchContext,
    ToolDispatchError, ToolDispatchRequest, ToolDispatchResult, ToolDispatcher,
    ToolDispatcherFactory, ToolSandbox, ToolSandboxError, ToolSandboxFactory, TraceOpenContext,
    close_trace_preserving_primary,
};
pub use docker::{
    CONTAINER_RUN_LABEL_KEY, ContainerCreateSpec, ContainerId, ContainerMount, ContainerRuntime,
    DockerCliRuntime, DockerSandboxFactory, DockerSessionSandbox, DockerToolDispatcherFactory,
    FramedCommandIo, NativeToolDispatcherFactory, cleanup_recorded_agent_containers,
    preflight_docker_sandbox,
};
pub use environment::{
    EnvironmentRecipe, ResolvedTraceEnvironment, ToolExecutionBackend, ToolSandboxCapabilities,
    TraceEnvironmentError, TraceEnvironmentResolver,
};
pub use environment_stepper::{
    EnvironmentArtifactBindings, EnvironmentEpisodeIdentity, EnvironmentResetRecord,
    EnvironmentResetRequest, EnvironmentSessionAuthority, EnvironmentStepRequest,
    EnvironmentStepper, EnvironmentStepperBinding, EnvironmentStepperError,
    EnvironmentStepperFactory, SupervisedEnvironmentStepperFactory,
};
pub use local::{
    LocalProcessRequest, LocalSessionSandbox, ProcessSession, ProcessSpawner, TokioProcessSpawner,
};
pub use policy::{
    CommandDisposition, GuardedToolCommandPolicy, ToolCommandPolicy, ToolCommandResult,
};
pub use workspace::{
    PinchWorkspaceStager, ProvisionedWorkspace, SegmentWorkspaceProvisioner, WorkspaceEntrySource,
    WorkspaceFile, WorkspaceProvisioner, WorkspaceSpec, WorkspaceTreeStager,
};
