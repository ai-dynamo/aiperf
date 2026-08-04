// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native load-generation timing plane.
//!
//! Time is injected as integer nanoseconds or read through
//! [`crate::clock::Clock`]. No policy reads wall time directly, so the same code
//! drives online and virtual execution.
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
pub mod rate_gate;
pub mod rate_series;
pub mod slots;
pub mod stop;
pub mod url_selection;
pub mod user_centric;

pub use arrival::{FirstArrival, WhenBehind, next_arrival_target};
pub use cancellation::{BernoulliFixedDelay, CancellationPolicy, CancellationPolicyError, Phase};
pub use intervals::{ArrivalPattern, IntervalGenerator, make_interval_generator};
pub(crate) use phase::drive_phases;
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
pub use rate_gate::{ClaimedSlot, GlobalRateGate};
pub use slots::{
    ConcurrencyManager, ConcurrencyStats, GlobalSlotGuard, GlobalSlotPool, SlotGuard, SlotPool,
};
pub use stop::{RunState, StopChecker, StopCondition, StopConfig};
pub use url_selection::{RoundRobinUrlSelector, UrlSelectionError, UrlSelector};
pub use user_centric::{InitialUser, UserCentricPlan, plan_user_centric};

/// Nanoseconds per second, as `f64`, for seconds↔nanoseconds conversions across
/// the timing plane. Single definition so every module rounds against the same
/// constant.
pub(crate) const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Convert a non-negative interval in seconds to integer nanoseconds with
/// ties-away-from-zero rounding. Non-finite or negative inputs clamp to 0.
pub(crate) fn secs_to_ns(secs: f64) -> i64 {
    if !secs.is_finite() || secs <= 0.0 {
        return 0;
    }
    (secs * NANOS_PER_SECOND).round() as i64
}
