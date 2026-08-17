// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local response, trajectory, and invocation seams for agent drivers.

mod lease;
mod response_store;
mod trajectory;
mod turn;

pub use lease::{
    AgentInvocationEnvironment, AgentInvocationIdentity, AgentInvocationLease,
    AgentInvocationLeaseFactory, AgentInvocationLeaseFactoryFactory, AgentInvocationLeaseOpening,
    AgentInvocationRequest, DelegatedInvocationTerminal, InMemoryAgentInvocationLeaseFactory,
    InMemoryAgentInvocationLeaseFactoryFactory, InMemoryInvocationLeaseFactory,
    InMemoryInvocationLeaseFactoryFactory, InvocationLease, InvocationLeaseFactory,
    InvocationLeaseFactoryFactory, deterministic_delegated_join_order,
};
pub use response_store::{
    AgentResponseHandle, AgentResponseSource, AgentResponseStore, AgentResponseStoreError,
    AgentResponseStoreFactory, InMemoryAgentResponseStore, InMemoryAgentResponseStoreFactory,
};
pub use trajectory::{
    AgentTrajectory, AgentTrajectoryCodec, AgentTrajectoryError, AgentTrajectoryResponse,
    AgentTrajectorySink, AgentTrajectorySinkFactory, InMemoryAgentTrajectoryCodec,
    InMemoryAgentTrajectorySink, InMemoryAgentTrajectorySinkFactory,
};
pub use turn::{
    AgentLoopError, AgentTurn, AgentTurnCoordinator, AgentTurnCoordinatorFactory,
    AgentTurnCoordinatorSpec, LiveAgentTurnDirective, ResponseSelection,
    StaticAgentTurnCoordinator, StaticAgentTurnCoordinatorFactory,
};
