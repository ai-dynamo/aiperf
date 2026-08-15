// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-dispatch contracts used by recorded-agent trace drivers.

pub mod dispatch;
pub mod environment;
pub mod local;
pub mod policy;
pub mod workspace;

pub use crate::graph::recorded::agent_recording::resolve_recorded_environment;
pub use dispatch::{
    AgentObservationFormatter, AgentObservationFormatterFactory, AgentToolCall,
    AgentToolCallDecoder, AgentToolCallDecoderFactory, EnvironmentToolDispatcher,
    InMemoryAgentObservationFormatter, InMemoryAgentObservationFormatterFactory,
    InMemoryAgentToolCallDecoder, InMemoryAgentToolCallDecoderFactory, InMemoryToolDispatcher,
    InMemoryToolDispatcherFactory, SandboxCreateContext, ToolDispatchContext, ToolDispatchError,
    ToolDispatchRequest, ToolDispatchResult, ToolDispatcher, ToolDispatcherFactory, ToolSandbox,
    ToolSandboxError, ToolSandboxFactory, TraceOpenContext, close_trace_preserving_primary,
};
pub use environment::{
    EnvironmentRecipe, ResolvedTraceEnvironment, ToolSandboxCapabilities, TraceEnvironmentError,
    TraceEnvironmentResolver,
};
pub use local::{
    LocalProcessRequest, LocalSessionSandbox, ProcessSession, ProcessSpawner, TokioProcessSpawner,
};
pub use policy::{
    CommandDisposition, GuardedToolCommandPolicy, ToolCommandPolicy, ToolCommandResult,
};
pub use workspace::{
    PinchWorkspaceStager, ProvisionedWorkspace, WorkspaceEntrySource, WorkspaceFile,
    WorkspaceProvisioner, WorkspaceSpec,
};
