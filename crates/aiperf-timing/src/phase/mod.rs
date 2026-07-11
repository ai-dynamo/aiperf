// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native phase lifecycle and progress policy.
//!
//! This module ports the process-independent policy from Python's
//! `src/aiperf/timing/phase/lifecycle.py:35-175`,
//! `src/aiperf/timing/phase/credit_counter.py:14-297`, and
//! `src/aiperf/timing/phase/progress_tracker.py:23-220`. The Python message bus
//! is deliberately absent: typed snapshots flow directly through
//! [`PhaseObserver`]. The async runner and orchestrator build on these leaf
//! types without changing their single-threaded `Rc`/`RefCell` contract.

mod config;
mod lifecycle;
mod observer;
mod progress;
mod stats;

pub use config::{GracePeriod, PhaseConfig, PhaseConfigError, PhaseKind};
pub use lifecycle::{
    PhaseCompletionReason, PhaseLifecycle, PhaseLifecycleError, PhaseLifecycleSnapshot, PhaseState,
};
pub use observer::{
    ConsolePhaseObserver, NoopPhaseObserver, PhaseBranchStats, PhaseEvent, PhaseEventKind,
    PhaseObserver, RecordingPhaseObserver,
};
pub use progress::{
    PhaseProgress, PhaseProgressCounters, PhaseProgressError, PhaseReturn, PhaseReturnOutcome,
    PhaseSend, PhaseSendOutcome,
};
pub use stats::PhaseStats;
