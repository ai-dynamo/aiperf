// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapters over the narrow host services a direct transport receives.
//!
//! A direct transport does not run inside the shared worker kernel: it drives
//! its own execution and reaches back through `aiperf_core::services` for time,
//! replay structure, measurement, artifacts, and shutdown. The runtime's
//! `RunContext` is deliberately not representable there — it owns the process
//! registry, execution factories, dataset resolvers, and validated endpoint
//! profiles, none of which a plugin may name.
//!
//! What is missing from those narrow traits is the *usage pattern*: polling
//! cancellation against a deadline, recording a metric in the unit the metrics
//! plane expects, resolving an artifact path through the capability. Each
//! transport would otherwise reimplement those, and each reimplementation is a
//! chance to read the wrong clock or emit the wrong unit. They live here once.

use std::rc::Rc;

use aiperf_core::artifact::{ArtifactAccess, ArtifactError};
use aiperf_core::clock::Clock;
use aiperf_core::histogram::seconds_scale;
use aiperf_core::services::{
    CancellationService, DirectTransportServices, GraphService, MetricsService,
};

/// Why a direct transport stopped admitting work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The host asked the transport to stop admitting.
    Cancelled,
    /// The phase's force deadline elapsed; in-flight work is terminated.
    DeadlineElapsed,
}

/// Whether the transport should stop admitting, and why.
///
/// Both conditions are checked because they mean different things: cancellation
/// is a cooperative drain that lets in-flight work finish, while an elapsed
/// deadline is a force escalation. Collapsing them into one boolean loses the
/// distinction the phase policy depends on.
pub fn stop_reason<C: CancellationService + ?Sized>(
    cancellation: &C,
    clock: &dyn Clock,
) -> Option<StopReason> {
    if cancellation.is_cancelled() {
        return Some(StopReason::Cancelled);
    }
    // The deadline is an instant on the same clock, not a duration: comparing it
    // against `Instant::now` would leave virtual time under `SimClock`.
    match cancellation.deadline_ns() {
        Some(deadline_ns) if clock.now_ns() >= deadline_ns => Some(StopReason::DeadlineElapsed),
        _ => None,
    }
}

/// Nanoseconds remaining before the force deadline, saturating at zero.
///
/// `None` means the phase set no deadline, which is not the same as zero
/// remaining: a transport must not treat an unbounded phase as already expired.
pub fn remaining_ns<C: CancellationService + ?Sized>(
    cancellation: &C,
    clock: &dyn Clock,
) -> Option<i64> {
    cancellation
        .deadline_ns()
        .map(|deadline_ns| deadline_ns.saturating_sub(clock.now_ns()).max(0))
}

/// Record one duration in seconds, converting from nanoseconds.
///
/// Routed through [`seconds_scale`] so a direct transport's latency lands in the
/// same unit as a request transport's, rather than each transport inventing its
/// own divisor.
pub fn record_duration_ns(metrics: &dyn MetricsService, name: &str, duration_ns: i64) {
    let seconds = duration_ns as f64 * seconds_scale("ns");
    if seconds.is_finite() {
        metrics.record_metric(name, seconds, "s");
    }
}

/// Record one already-scaled value, dropping non-finite readings.
///
/// A `NaN` or infinity is not representable at the serialization boundary, so it
/// is dropped here rather than propagated into the report as a corrupt number.
pub fn record_finite(metrics: &dyn MetricsService, name: &str, value: f64, unit: &str) {
    if value.is_finite() {
        metrics.record_metric(name, value, unit);
    }
}

/// Total dispatching nodes across every trace in the resolved graph.
///
/// A trace whose node count the host cannot answer contributes nothing rather
/// than aborting the walk: an unknown trace is a gap in the read view, not a
/// reason to refuse the whole program.
pub fn total_graph_nodes(graph: &dyn GraphService) -> usize {
    (0..graph.trace_count())
        .filter_map(|trace| graph.node_count(trace))
        .sum()
}

/// Write one artifact through the capability, never through a raw path.
///
/// The relative path is validated by [`ArtifactAccess`]; a plugin has no way to
/// name a directory outside the run's artifact scope.
pub fn write_artifact(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    bytes: &[u8],
) -> Result<(), ArtifactError> {
    artifacts.create(relative_path, bytes)
}

/// Convenience accessors over the full direct-transport service set.
pub trait DirectServicesExt {
    /// The run's clock.
    fn run_clock(&self) -> Rc<dyn Clock>;

    /// Whether and why the transport should stop admitting work.
    fn stop_reason(&self) -> Option<StopReason>;

    /// Nanoseconds remaining before the force deadline, if the phase set one.
    fn remaining_ns(&self) -> Option<i64>;

    /// Total dispatching nodes, or zero when the run resolved no graph program.
    fn total_graph_nodes(&self) -> usize;
}

impl<S: DirectTransportServices + ?Sized> DirectServicesExt for S {
    fn run_clock(&self) -> Rc<dyn Clock> {
        self.clock()
    }

    fn stop_reason(&self) -> Option<StopReason> {
        let clock = self.clock();
        stop_reason(self, clock.as_ref())
    }

    fn remaining_ns(&self) -> Option<i64> {
        let clock = self.clock();
        remaining_ns(self, clock.as_ref())
    }

    fn total_graph_nodes(&self) -> usize {
        self.graph().map_or(0, total_graph_nodes)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::future::Future;
    use std::pin::Pin;

    use super::*;

    struct StepClock {
        now_ns: Cell<i64>,
    }

    impl Clock for StepClock {
        fn now_ns(&self) -> i64 {
            self.now_ns.get()
        }
        fn sleep(self: Rc<Self>, _duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async {})
        }
    }

    struct Signal {
        cancelled: Cell<bool>,
        deadline_ns: Option<i64>,
    }

    impl CancellationService for Signal {
        fn is_cancelled(&self) -> bool {
            self.cancelled.get()
        }
        fn deadline_ns(&self) -> Option<i64> {
            self.deadline_ns
        }
    }

    #[derive(Default)]
    struct MetricLog {
        recorded: RefCell<Vec<(String, f64, String)>>,
    }

    impl MetricsService for MetricLog {
        fn record_metric(&self, name: &str, value: f64, unit: &str) {
            self.recorded
                .borrow_mut()
                .push((name.to_owned(), value, unit.to_owned()));
        }
        fn increment_counter(&self, _name: &str, _delta: u64) {}
    }

    #[test]
    fn cancellation_outranks_an_unelapsed_deadline() {
        let clock = StepClock {
            now_ns: Cell::new(0),
        };
        let signal = Signal {
            cancelled: Cell::new(true),
            deadline_ns: Some(1_000),
        };
        assert_eq!(stop_reason(&signal, &clock), Some(StopReason::Cancelled));
    }

    #[test]
    fn an_unbounded_phase_is_not_treated_as_expired() {
        let clock = StepClock {
            now_ns: Cell::new(i64::MAX),
        };
        let signal = Signal {
            cancelled: Cell::new(false),
            deadline_ns: None,
        };
        assert_eq!(stop_reason(&signal, &clock), None);
        assert_eq!(remaining_ns(&signal, &clock), None);
    }

    #[test]
    fn an_elapsed_deadline_stops_and_leaves_zero_remaining() {
        let clock = StepClock {
            now_ns: Cell::new(2_000),
        };
        let signal = Signal {
            cancelled: Cell::new(false),
            deadline_ns: Some(1_000),
        };
        assert_eq!(
            stop_reason(&signal, &clock),
            Some(StopReason::DeadlineElapsed)
        );
        assert_eq!(remaining_ns(&signal, &clock), Some(0));
    }

    #[test]
    fn durations_are_recorded_in_seconds_and_non_finite_values_dropped() {
        let metrics = MetricLog::default();
        record_duration_ns(&metrics, "request_latency", 1_500_000_000);
        record_finite(&metrics, "ratio", f64::NAN, "1");
        let recorded = metrics.recorded.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "request_latency");
        assert!((recorded[0].1 - 1.5).abs() < 1e-9);
        assert_eq!(recorded[0].2, "s");
    }
}
