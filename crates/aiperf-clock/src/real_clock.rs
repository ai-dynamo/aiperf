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
//! high-throughput live execution; use [`SimClock`](crate::sim_clock) for
//! reproducible runs.

use crate::clock::Clock;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::time::Instant;

/// Monotonic wall clock with ns-precision `timerfd` sleeps.
pub struct RealClock {
    start: Instant,
}

impl RealClock {
    pub fn new() -> Rc<Self> {
        Rc::new(RealClock {
            start: Instant::now(),
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

#[cfg(target_os = "linux")]
async fn sleep_ns(duration_ns: i64) {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use tokio::io::unix::AsyncFd;

    if duration_ns <= 0 {
        tokio::task::yield_now().await;
        return;
    }

    const NANOS_PER_SEC: i64 = 1_000_000_000;

    // Arm a one-shot monotonic timerfd for the requested duration.
    let owned = unsafe {
        let fd = libc::timerfd_create(
            libc::CLOCK_MONOTONIC,
            libc::TFD_NONBLOCK | libc::TFD_CLOEXEC,
        );
        assert!(fd >= 0, "timerfd_create failed");
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
        assert_eq!(rc, 0, "timerfd_settime failed");
        OwnedFd::from_raw_fd(fd)
    };

    let afd = AsyncFd::new(owned).expect("register timerfd with reactor");
    loop {
        let mut guard = match afd.readable().await {
            Ok(g) => g,
            Err(_) => return,
        };
        let raw = afd.get_ref().as_raw_fd();
        match guard.try_io(|_| {
            let mut buf = [0u8; 8];
            let n = unsafe { libc::read(raw, buf.as_mut_ptr() as *mut libc::c_void, 8) };
            if n < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }) {
            Ok(_) => return, // timer expiration read (or a real error): the sleep is done
            Err(_would_block) => continue,
        }
    }
}

#[cfg(not(target_os = "linux"))]
async fn sleep_ns(duration_ns: i64) {
    if duration_ns <= 0 {
        tokio::task::yield_now().await;
    } else {
        tokio::time::sleep(std::time::Duration::from_nanos(duration_ns as u64)).await;
    }
}
