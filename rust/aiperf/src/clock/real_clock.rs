// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Real wall-clock time with ns-precision timers.
//!
//! The **live** [`Clock`] backend. `now_ns` is a monotonic reading; `sleep` on
//! Linux arms a `timerfd` (CLOCK_MONOTONIC) and awaits it via tokio's IO reactor
//! ([`tokio::io::unix::AsyncFd`]) — real nanosecond-resolution async sleeps, not
//! `tokio::time`'s 1 ms wheel. Non-Linux falls back to `tokio::time` (coarser).
//!
//! Real timers jitter, so this backend is **not** deterministic — it is for
//! high-throughput live execution; use [`SimClock`](crate::clock::sim_clock) for
//! reproducible runs.

use crate::clock::clock::Clock;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::time::Instant;

/// Copyable monotonic origin shared by cooperating real-clock runtimes.
///
/// Thread-per-core and remote-executor adapters construct one [`RealClock`] per
/// reactor. Giving every instance the same anchor keeps scheduler, transport,
/// and observer timestamps on one nanosecond timeline without sharing a clock
/// object or placing synchronization on the hot path.
#[derive(Clone, Copy, Debug)]
pub struct RealClockAnchor {
    start: Instant,
}

impl RealClockAnchor {
    /// Capture a fresh monotonic origin.
    pub fn now() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Nanoseconds elapsed on this timeline since the anchor was captured.
    ///
    /// The allocation-free `now_ns` reader for callers that hold a copyable
    /// [`RealClockAnchor`] (it is `Copy`/`Send`) rather than an `Rc<RealClock>`.
    /// Same monotonic reading as [`RealClock::now_ns`], so both sit on one
    /// timeline — the entry point for `Send` contexts (e.g. a multi-threaded
    /// server) that cannot hold the `!Send` `Rc<RealClock>` across an await.
    pub fn now_ns(&self) -> i64 {
        self.start.elapsed().as_nanos() as i64
    }
}

/// Monotonic wall clock with ns-precision `timerfd` sleeps.
pub struct RealClock {
    start: Instant,
}

impl RealClock {
    /// Construct a real clock with a fresh monotonic origin.
    pub fn new() -> Rc<Self> {
        Self::from_anchor(RealClockAnchor::now())
    }

    /// Construct a reactor-local clock on an existing shared timeline.
    pub fn from_anchor(anchor: RealClockAnchor) -> Rc<Self> {
        Rc::new(RealClock {
            start: anchor.start,
        })
    }
}

impl Clock for RealClock {
    fn now_ns(&self) -> i64 {
        self.start.elapsed().as_nanos() as i64
    }

    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
        Box::pin(sleep_ns(duration_ns))
    }
}

/// Sleep for `duration_ns` on this platform's most precise timer — a
/// `CLOCK_MONOTONIC` `timerfd` awaited through tokio's IO reactor on Linux, or
/// `tokio::time` elsewhere. This is the exact primitive backing
/// [`RealClock::sleep`], exposed as a standalone `Send` future so `Send`-bound
/// callers (a multi-threaded server whose tasks cannot hold the `!Send`
/// `Rc<RealClock>`) get the same nanosecond-resolution sleep without the coarse
/// `tokio::time` 1 ms wheel. Non-positive durations resolve after a single yield.
#[cfg(target_os = "linux")]
pub async fn sleep_ns(duration_ns: i64) {
    if duration_ns <= 0 {
        tokio::task::yield_now().await;
        return;
    }

    // Fast path: ns-precision `timerfd`. On any syscall failure (e.g. fd
    // pressure — EMFILE/ENFILE from `timerfd_create`, or reactor registration
    // failure) degrade gracefully to `tokio::time` instead of aborting the run.
    let started = Instant::now();
    if timerfd_sleep_ns(duration_ns).await.is_ok() {
        return;
    }

    // Fallback: sleep for whatever time remains of the requested duration.
    let elapsed_ns = started.elapsed().as_nanos();
    let remaining_ns = (duration_ns as u128).saturating_sub(elapsed_ns);
    if remaining_ns > 0 {
        tokio::time::sleep(std::time::Duration::from_nanos(remaining_ns as u64)).await;
    } else {
        tokio::task::yield_now().await;
    }
}

/// Arm a one-shot monotonic `timerfd` for `duration_ns` and await its
/// expiration via tokio's IO reactor. Returns `Err` (without panicking) on any
/// syscall failure so the caller can fall back to a coarser sleep.
#[cfg(target_os = "linux")]
async fn timerfd_sleep_ns(duration_ns: i64) -> std::io::Result<()> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use tokio::io::unix::AsyncFd;

    const NANOS_PER_SEC: i64 = 1_000_000_000;

    // Arm a one-shot monotonic timerfd for the requested duration.
    let owned = unsafe {
        let fd = libc::timerfd_create(
            libc::CLOCK_MONOTONIC,
            libc::TFD_NONBLOCK | libc::TFD_CLOEXEC,
        );
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // Take ownership immediately so the fd is closed on any early return.
        let owned = OwnedFd::from_raw_fd(fd);
        let spec = libc::itimerspec {
            it_interval: libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
            it_value: libc::timespec {
                tv_sec: (duration_ns / NANOS_PER_SEC) as libc::time_t,
                tv_nsec: (duration_ns % NANOS_PER_SEC) as libc::c_long,
            },
        };
        let rc = libc::timerfd_settime(fd, 0, &spec, std::ptr::null_mut());
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        owned
    };

    let afd = AsyncFd::new(owned)?;
    loop {
        let mut guard = afd.readable().await?;
        let raw = afd.get_ref().as_raw_fd();
        match guard.try_io(|_| {
            let mut buf = [0u8; 8];
            let n = unsafe { libc::read(raw, buf.as_mut_ptr() as *mut libc::c_void, 8) };
            if n < 0 {
                // Surface the real errno; `try_io` maps EWOULDBLOCK/EAGAIN back
                // to `WouldBlock` (a spurious wakeup → retry) and any other
                // error to `Err` below (→ fallback), so a genuine read failure
                // is never mistaken for timer expiration.
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }) {
            Ok(Ok(())) => return Ok(()),   // timer expired: the sleep is done
            Ok(Err(e)) => return Err(e),   // genuine read error → fall back
            Err(_would_block) => continue, // not ready yet: re-arm the readiness wait
        }
    }
}

/// Non-Linux fallback for [`sleep_ns`]: `tokio::time` (coarser than a `timerfd`).
#[cfg(not(target_os = "linux"))]
pub async fn sleep_ns(duration_ns: i64) {
    if duration_ns <= 0 {
        tokio::task::yield_now().await;
    } else {
        tokio::time::sleep(std::time::Duration::from_nanos(duration_ns as u64)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-trivial sleep waits at least the requested duration (timing
    /// semantics preserved by both the timerfd fast path and any fallback).
    #[tokio::test]
    async fn sleep_ns_waits_at_least_requested() {
        let want_ns = 5_000_000; // 5 ms
        let start = Instant::now();
        sleep_ns(want_ns).await;
        assert!(
            start.elapsed().as_nanos() as i64 >= want_ns,
            "slept less than requested"
        );
    }

    /// Zero / negative durations return promptly without arming a timer.
    #[tokio::test]
    async fn sleep_ns_zero_returns_fast() {
        let start = Instant::now();
        sleep_ns(0).await;
        sleep_ns(-1).await;
        assert!(start.elapsed() < std::time::Duration::from_millis(50));
    }

    #[test]
    fn clocks_from_one_anchor_share_a_timeline() {
        let anchor = RealClockAnchor::now();
        let first = RealClock::from_anchor(anchor);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let second = RealClock::from_anchor(anchor);

        assert!(first.now_ns() >= 1_000_000);
        assert!(second.now_ns() >= 1_000_000);
        assert!(first.now_ns().abs_diff(second.now_ns()) < 5_000_000);
    }
}
