// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared absolute-arrival scheduling policy for the online loops.
//!
//! Three live loops pace arrivals off an
//! [`IntervalGenerator`](crate::timing::IntervalGenerator): the scheduled
//! `RequestRateWorkload` (`crate::request_rate`), the dynosim paced online/offline
//! driver (`crate::run`), and the graph `IntervalGraphArrival`
//! (`crate::graph::workload`). All three compute the same quantity — *the next
//! arrival's absolute target time on the clock timeline* — and differ in exactly
//! two policy axes this module names rather than leaving implicit in each loop's
//! shape:
//!
//! - [`FirstArrival`] — whether arrival 0 is due at `start` exactly
//!   ([`FirstArrival::AtStart`], the graph policy: a trace's first turn arrives at
//!   `run_start`) or one drawn interval in ([`FirstArrival::AfterInterval`], the
//!   scheduled / dynosim policy: the first request is paced like every other).
//! - [`WhenBehind`] — when the computed target is already in the past, whether to
//!   re-anchor to `now` ([`WhenBehind::Reanchor`], the scheduled / dynosim policy:
//!   a slow tick never fires a catch-up burst) or keep the absolute target
//!   ([`WhenBehind::KeepAbsolute`], the graph policy: the schedule is honored and
//!   the loop bursts to catch up).
//!
//! Draw timing is observable when a generator is shared with live ramp actuators;
//! callers must preserve whether the draw occurs at the start or tail of an
//! iteration and whether closed-loop backpressure peeks the next target.

/// Where the first arrival (the one with no prior target) is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstArrival {
    /// Due at `start_ns` exactly — no leading interval is drawn. The graph policy.
    AtStart,
    /// Due one drawn interval after `start_ns`. The scheduled / dynosim policy.
    AfterInterval,
}

/// What to do when the freshly computed target is already in the past.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhenBehind {
    /// Re-anchor the target to `now` — fire now, draw the following interval from
    /// `now`, never a catch-up burst. The scheduled / dynosim policy.
    Reanchor,
    /// Keep the absolute target — honor the schedule and let the caller burst to
    /// catch up. The graph policy.
    KeepAbsolute,
}

/// Compute the next arrival's absolute target time on the clock timeline.
///
/// `prev_target_ns` is the previously returned target, or `None` for the first
/// arrival. `draw_interval_ns` is called to draw the next inter-arrival interval —
/// exactly once, and only when an interval is needed (never for
/// [`FirstArrival::AtStart`]'s first arrival).
///
/// See the module docs for the policy axes. The result is pure in its inputs; the
/// caller stores it as the next `prev_target_ns` and performs its own wait.
pub fn next_arrival_target(
    prev_target_ns: Option<i64>,
    start_ns: i64,
    now_ns: i64,
    first: FirstArrival,
    when_behind: WhenBehind,
    draw_interval_ns: impl FnOnce() -> i64,
) -> i64 {
    let raw = match prev_target_ns {
        None => match first {
            FirstArrival::AtStart => start_ns,
            FirstArrival::AfterInterval => start_ns.saturating_add(draw_interval_ns()),
        },
        Some(prev) => prev.saturating_add(draw_interval_ns()),
    };
    match when_behind {
        WhenBehind::Reanchor => raw.max(now_ns),
        WhenBehind::KeepAbsolute => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The scheduled / dynosim policy: first arrival one interval in, and a target
    // that has fallen behind re-anchors to `now` with no catch-up burst. Uses
    // the `crate::run` / `crate::request_rate` loops' `next_target_ns` arithmetic.
    #[test]
    fn after_interval_reanchor_matches_scheduled_loop() {
        let start = 1_000;
        // A generator emitting a fixed 100ns interval.
        let draw = || 100i64;
        let t0 = next_arrival_target(
            None,
            start,
            900,
            FirstArrival::AfterInterval,
            WhenBehind::Reanchor,
            draw,
        );
        assert_eq!(t0, 1_100);
        // On schedule: next = prev + interval, still ahead of the clock.
        let t1 = next_arrival_target(
            Some(t0),
            start,
            1_150,
            FirstArrival::AfterInterval,
            WhenBehind::Reanchor,
            draw,
        );
        assert_eq!(t1, 1_200);
        // Fallen behind (now=5_000 > prev+interval): re-anchor to now, no burst.
        let t2 = next_arrival_target(
            Some(t1),
            start,
            5_000,
            FirstArrival::AfterInterval,
            WhenBehind::Reanchor,
            draw,
        );
        assert_eq!(t2, 5_000);
        // The interval following a re-anchor is drawn from the re-anchored target.
        let t3 = next_arrival_target(
            Some(t2),
            start,
            5_000,
            FirstArrival::AfterInterval,
            WhenBehind::Reanchor,
            draw,
        );
        assert_eq!(t3, 5_100);
    }

    // The graph policy: first arrival at `start` exactly (no interval drawn), and
    // an absolute schedule that never re-anchors — the caller bursts to catch up.
    // Uses the same arrival policy as `IntervalGraphArrival::wait_for_arrival`.
    #[test]
    fn at_start_keep_absolute_matches_graph_loop() {
        let start = 1_000;
        let draw = || 100i64;
        // Ordinal 0 fires at start exactly; the closure must not be consulted.
        let t0 = next_arrival_target(
            None,
            start,
            10_000,
            FirstArrival::AtStart,
            WhenBehind::KeepAbsolute,
            || panic!("first AtStart arrival must not draw an interval"),
        );
        assert_eq!(t0, 1_000);
        // Subsequent arrivals accumulate absolutely, ignoring how far behind the
        // clock is (no re-anchor).
        let t1 = next_arrival_target(
            Some(t0),
            start,
            10_000,
            FirstArrival::AtStart,
            WhenBehind::KeepAbsolute,
            draw,
        );
        assert_eq!(t1, 1_100);
        let t2 = next_arrival_target(
            Some(t1),
            start,
            10_000,
            FirstArrival::AtStart,
            WhenBehind::KeepAbsolute,
            draw,
        );
        assert_eq!(t2, 1_200);
    }

    // Saturating arithmetic guards against a pathological interval overflowing the
    // timeline rather than panicking.
    #[test]
    fn saturates_on_overflow() {
        let t = next_arrival_target(
            Some(i64::MAX - 1),
            0,
            0,
            FirstArrival::AfterInterval,
            WhenBehind::KeepAbsolute,
            || 1_000,
        );
        assert_eq!(t, i64::MAX);
    }
}
