// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The narrow host services a direct transport is given.
//!
//! A direct transport does not issue requests through the shared worker kernel:
//! it drives its own execution and reaches back to the host for time, replay
//! structure, measurement, artifacts, and shutdown. The runtime's `RunContext`
//! is the value that today aggregates those capabilities, and it is not
//! representable here — it owns the process registry, execution factories,
//! graph/dataset input resolvers, sidecar inputs, and validated endpoint
//! profiles, none of which a plugin may name. Each capability is instead a
//! separate narrow trait, and [`DirectTransportServices`] is the only aggregate.
//!
//! Every trait is `!Send`-friendly: a direct transport runs on a current-thread
//! runtime, so worker-local `Rc<RefCell<_>>` state satisfies these without
//! synchronization.

use std::rc::Rc;

use crate::artifact::ArtifactAccess;
use crate::clock::Clock;

/// Access to the run's single time source.
///
/// A direct transport must route every measurement and firing gate through this
/// clock: under `SimClock` an `Instant::now` call silently leaves virtual time.
pub trait ClockService {
    /// The clock the run was constructed with.
    fn clock(&self) -> Rc<dyn Clock>;
}

/// The narrow replay-structure facts a direct transport reads.
///
/// This is deliberately a read view over resolved Graph-IR shape and recorded
/// timing. It exposes no node payloads, no segment store, no scheduler, and no
/// way to mutate the program: a direct transport replays a structure the host
/// already resolved.
pub trait GraphService {
    /// Number of traces in the resolved graph program.
    fn trace_count(&self) -> usize;

    /// Number of dispatching nodes in one trace, or `None` for an unknown trace.
    fn node_count(&self, trace_index: usize) -> Option<usize>;

    /// Recorded completion delay in nanoseconds for one node, or `None` when the
    /// node is unknown or carries no recorded anchor.
    fn recorded_completion_delay_ns(&self, trace_index: usize, node_index: usize) -> Option<i64>;
}

/// The narrow measurement sink a direct transport writes through.
///
/// Values are recorded in their display unit so the metrics plane applies the
/// same unit handling it applies to a request transport's records; see
/// [`crate::histogram::seconds_scale`].
pub trait MetricsService {
    /// Record one projected metric value for the current record.
    fn record_metric(&self, name: &str, value: f64, unit: &str);

    /// Add to one monotonic counter.
    fn increment_counter(&self, name: &str, delta: u64);
}

/// Access to the run's capability-limited artifact scope.
pub trait ArtifactService {
    /// The artifact capability, which exposes no raw directory path.
    fn artifacts(&self) -> &dyn ArtifactAccess;
}

/// The run's cooperative shutdown signal.
///
/// A direct transport polls this at its own admission and drain points; the host
/// never interrupts it. `deadline_ns` is a clock-ns instant on the same clock
/// [`ClockService`] returns, not a duration.
pub trait CancellationService {
    /// Whether the host has asked the transport to stop admitting work.
    fn is_cancelled(&self) -> bool;

    /// Clock-ns instant after which in-flight work is force-terminated, if the
    /// phase set one.
    fn deadline_ns(&self) -> Option<i64>;
}

/// The complete service set a direct transport receives.
///
/// The graph accessor is optional because a direct transport may run a scheduled
/// workload with no resolved graph program; every other capability is always
/// present.
pub trait DirectTransportServices:
    ClockService + MetricsService + ArtifactService + CancellationService
{
    /// The replay-structure view, when the run resolved a graph program.
    fn graph(&self) -> Option<&dyn GraphService>;
}
