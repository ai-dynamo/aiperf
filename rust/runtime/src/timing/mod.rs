// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native load-generation timing plane.
//!
//! The Rust home for the Python `src/aiperf/timing/` subsystem, built clock-first
//! rather than retrofitted. Time is either injected as integer nanoseconds or read
//! through an injected [`crate::clock::Clock`] by an async policy driver. Nothing
//! here reads a wall clock directly, so the identical policy drives a `RealClock`
//! (online) or a `SimClock` (offline) run.
//!
//! The seam is eight trait families, each with at least one concrete impl:
//! - [`intervals`] — inter-arrival distribution ([`IntervalGenerator`]),
//! - [`slots`] — concurrency admission ([`SlotPool`], debt-drain-capable),
//! - [`stop`] — run-termination bounds ([`StopCondition`] / [`StopChecker`]),
//! - [`user_centric`] — per-user session schedule math ([`plan_user_centric`]),
//! - [`ramping`] — clock-driven value ramps ([`RampStrategy`] / [`RampDriver`]),
//! - [`cancellation`] — per-request disconnect decisions ([`CancellationPolicy`]),
//! - [`url_selection`] — endpoint selection ([`UrlSelector`]).
//! - [`phase`] — lifecycle, progress, execution, and multi-phase orchestration.

pub mod arrival;
pub mod cancellation;
pub mod intervals;
pub mod phase;
pub mod ramping;
pub mod slots;
pub mod stop;
pub mod url_selection;
pub mod user_centric;

pub use arrival::{FirstArrival, WhenBehind, next_arrival_target};
pub use cancellation::{BernoulliFixedDelay, CancellationPolicy, CancellationPolicyError, Phase};
pub use intervals::{ArrivalPattern, IntervalGenerator, make_interval_generator};
pub use phase::{
    ClockPhaseOrchestrator, ClockPhaseRunner, ClockPhaseRunnerFactory, ConsolePhaseObserver,
    DISABLED_PROGRESS_INTERVAL_NS, GracePeriod, LocalPhaseFuture, NoopPhaseExecution,
    NoopPhaseExecutionFactory, NoopPhaseObserver, PhaseBranchStats, PhaseCompletionReason,
    PhaseConfig, PhaseConfigError, PhaseContext, PhaseEvent, PhaseEventKind, PhaseExecution,
    PhaseExecutionError, PhaseExecutionFactory, PhaseKind, PhaseLifecycle, PhaseLifecycleError,
    PhaseLifecycleSnapshot, PhaseObserver, PhaseOrchestrator, PhaseOrchestratorError,
    PhaseProgress, PhaseProgressCounters, PhaseProgressError, PhaseReturn, PhaseReturnOutcome,
    PhaseRunError, PhaseRunner, PhaseRunnerFactory, PhaseSend, PhaseSendOutcome, PhaseState,
    PhaseStats, RecordingPhaseObserver, ReleasedStuckSlots,
};
pub use ramping::{
    ExponentialRamp, LinearRamp, PoissonRamp, RampConfigError, RampDriver, RampHandle,
    RampStrategy, RampTaskError, RamperConfig,
};
pub use slots::{ConcurrencyManager, ConcurrencyStats, SlotGuard, SlotPool};
pub use stop::{RunState, StopChecker, StopCondition, StopConfig};
pub use url_selection::{RoundRobinUrlSelector, UrlSelectionError, UrlSelector};
pub use user_centric::{InitialUser, UserCentricPlan, plan_user_centric};
