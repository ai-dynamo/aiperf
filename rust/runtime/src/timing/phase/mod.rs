// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native phase lifecycle and progress policy.
//!
//! The Python message bus is deliberately absent: typed snapshots flow directly
//! through [`PhaseObserver`]. The async runner and orchestrator build on these
//! leaf types without changing their single-threaded `Rc`/`RefCell` contract.

mod config;
mod lifecycle;
mod observer;
mod orchestrator;
mod progress;
mod runner;
mod stats;

pub use config::{
    DISABLED_PROGRESS_INTERVAL_NS, GracePeriod, PhaseConfig, PhaseConfigError, PhaseKind,
};
pub use lifecycle::{
    PhaseCompletionReason, PhaseLifecycle, PhaseLifecycleError, PhaseLifecycleSnapshot, PhaseState,
};
pub use observer::{
    ConsolePhaseObserver, NoopPhaseObserver, PhaseBranchStats, PhaseEvent, PhaseEventKind,
    PhaseObserver, RecordingPhaseObserver,
};
pub(crate) use orchestrator::drive_phases;
pub use orchestrator::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, PhaseOrchestrator, PhaseOrchestratorError,
    PhaseRunnerFactory,
};
pub use progress::{
    PhaseProgress, PhaseProgressCounters, PhaseProgressError, PhaseReturn, PhaseReturnOutcome,
    PhaseSend, PhaseSendOutcome,
};
pub use runner::{
    ClockPhaseRunner, LocalPhaseFuture, NoopPhaseExecution, NoopPhaseExecutionFactory,
    PhaseContext, PhaseExecution, PhaseExecutionError, PhaseExecutionFactory, PhaseRunError,
    PhaseRunner, ReleasedStuckSlots,
};
pub use stats::PhaseStats;
