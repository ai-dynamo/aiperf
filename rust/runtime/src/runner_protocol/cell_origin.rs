// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Barrier-synchronized cross-cell timing origin.
//!
//! In cellular mode every cell independently captures its run origin
//! (`start_ns = clock.now_ns()`) inside [`execute`](super::execute) — but that
//! reading happens AFTER the velo START barrier releases AND after each cell's
//! own dataset download + run setup. Cells with a larger shard (slower download)
//! or slower setup therefore zero their record timeline at a LATER instant than
//! their peers, so cross-cell timing (per-request timestamps, benchmark
//! duration, the realtime block's elapsed) references a different `t0` per cell.
//!
//! This module makes the origin shareable via the barrier itself (opt-in behind
//! [`CELL_SHARED_ORIGIN_ENV`]): the moment the cell's velo START barrier releases
//! (inside [`fetch_cell_envelope`](super::cellular_cell::fetch_cell_envelope)) it
//! captures a [`RealClockAnchor`] — the shared logical instant every cell reaches
//! together. Later, [`run_origin_now_ns`] returns the barrier's reading on the
//! execute clock's own timeline instead of the post-setup `now`, so every cell
//! zeroes its record timeline at the SAME barrier event regardless of per-cell
//! setup time.
//!
//! **Cross-host semantics (deliberate).** Cells may run on different hosts with
//! unsynchronized wall clocks, so this does NOT adopt an absolute controller `t0`
//! (which would import clock skew). Each cell zeroes at its OWN clock reading of
//! the barrier-release instant; the barrier guarantees those instants coincide
//! within network latency, so "elapsed since START" stays coherent across hosts
//! without any clock-sync assumption. See the design decision recorded in
//! `specs/2026-07-15-ultimate-cellular-velo-runtime-design.md` §4.
//!
//! Default off: a single-process run never calls
//! [`capture_cell_shared_origin`], so [`run_origin_now_ns`] returns
//! `clock.now_ns()` unchanged and the byte-parity / timing tests are untouched.

use std::rc::Rc;
use std::sync::OnceLock;

use crate::clock::{Clock, RealClockAnchor};

/// Env flag opting a cellular run into the barrier-synchronized timing origin.
/// Truthy values: `1`/`true`/`on`/`yes` (case-insensitive). Default off.
pub const CELL_SHARED_ORIGIN_ENV: &str = "AIPERF_CELL_SHARED_ORIGIN";

/// The barrier-release anchor for this cell process, set once when the velo START
/// barrier releases. `Some(None)` means "capture ran but the feature was off"
/// (so a late reader does not re-check the env); `Some(Some(anchor))` is an active
/// shared origin. Process-global because each cell is its own `aiperf --cell`
/// process, so a single `OnceLock` cannot collide across cells.
static CELL_SHARED_ORIGIN: OnceLock<Option<RealClockAnchor>> = OnceLock::new();

/// Whether [`CELL_SHARED_ORIGIN_ENV`] is set to a truthy value.
fn shared_origin_enabled() -> bool {
    matches!(
        std::env::var(CELL_SHARED_ORIGIN_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// Capture the barrier-release instant as this cell's shared timing origin.
///
/// Call ONCE, immediately after the cell's velo START barrier releases and BEFORE
/// dataset download + run setup, so the captured instant is the shared START
/// moment rather than a post-setup local time. Idempotent (first call wins) and a
/// no-op unless [`CELL_SHARED_ORIGIN_ENV`] is truthy.
pub fn capture_cell_shared_origin() {
    let anchor = shared_origin_enabled().then(RealClockAnchor::now);
    // First set wins; a redundant call (e.g. a retry) keeps the earliest instant.
    let _ = CELL_SHARED_ORIGIN.set(anchor);
}

/// The run origin on `clock`'s timeline.
///
/// Normally `clock.now_ns()` (the local reading at run start). When a barrier
/// shared origin was captured ([`capture_cell_shared_origin`]), this instead
/// returns the barrier's reading on `clock`'s timeline — computed as
/// `clock.now_ns() - barrier.now_ns()` at one instant, so the shared wall-now
/// cancels and only the (execute-anchor − barrier) offset remains. That offset is
/// negative when the barrier preceded `clock`'s anchor (the common case: the
/// execute clock's anchor is created during run setup, after the barrier), which
/// correctly shifts every record's timestamp forward to be measured from the
/// barrier rather than from the cell's local run start.
pub fn run_origin_now_ns(clock: &Rc<dyn Clock>) -> i64 {
    let now = clock.now_ns();
    match CELL_SHARED_ORIGIN.get().copied().flatten() {
        // Read the barrier elapsed right after `now`; the sub-microsecond skew
        // between the two reads is immaterial to a run-length timeline.
        Some(barrier) => shifted_origin(now, barrier.now_ns()),
        None => now,
    }
}

/// The run origin on a timeline whose current reading is `now_ns` and on which the
/// barrier was `barrier_elapsed_ns` ago: `now_ns - barrier_elapsed_ns`. Negative
/// when the barrier preceded this timeline's anchor (the common case — the execute
/// clock's anchor is created during run setup, after the barrier), which shifts
/// every record's timestamp forward so it is measured from the barrier.
fn shifted_origin(now_ns: i64, barrier_elapsed_ns: i64) -> i64 {
    now_ns - barrier_elapsed_ns
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::clock::{Clock, SimClock};

    #[test]
    fn shifted_origin_moves_t0_back_to_the_barrier() {
        // Barrier 30ns ago on a timeline now reading 100ns => origin at 70ns.
        assert_eq!(shifted_origin(100, 30), 70);
        // Barrier BEFORE this timeline's anchor (elapsed since barrier exceeds the
        // local now) => a negative origin, which shifts record timestamps forward.
        assert_eq!(shifted_origin(2_000, 9_000), -7_000);
        // A barrier captured exactly at the anchor is a no-op.
        assert_eq!(shifted_origin(42, 0), 42);
    }

    #[test]
    fn run_origin_is_the_local_now_when_no_barrier_was_captured() {
        // Without a captured shared origin (the single-process / default path),
        // the run origin is exactly the clock's current reading — byte-unchanged.
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let now = clock.now_ns();
        assert_eq!(run_origin_now_ns(&clock), now);
    }
}
