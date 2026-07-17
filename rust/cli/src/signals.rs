// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graceful SIGINT/SIGTERM forwarding to the active `aiperf` child.
//!
//! Terminating signals are blocked in the parent and forwarded as SIGINT to the
//! running child so it can drain and write a
//! partial `was_cancelled=true` report instead of dying abruptly. Between cells
//! (no child), the first signal exits the process (code 130) so Ctrl-C still
//! stops a sweep.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// Shared current-child PID, or zero when idle.
#[derive(Clone, Default)]
pub struct ChildPid(Arc<AtomicI32>);

impl ChildPid {
    /// Publish the running child's PID.
    pub fn set(&self, pid: u32) {
        self.0.store(pid as i32, Ordering::SeqCst);
    }
    /// Clear the child PID.
    pub fn clear(&self) {
        self.0.store(0, Ordering::SeqCst);
    }
    fn get(&self) -> i32 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Install the Unix signal-forwarding thread.
#[cfg(unix)]
pub fn install() -> ChildPid {
    use nix::sys::signal::{SigSet, Signal, kill};
    use nix::unistd::Pid;

    let child = ChildPid::default();
    let mut set = SigSet::empty();
    set.add(Signal::SIGINT);
    set.add(Signal::SIGTERM);
    // Spawned threads inherit this mask, leaving the forwarder as the only
    // handler for these signals.
    if set.thread_block().is_err() {
        return child;
    }
    let child_for_thread = child.clone();
    std::thread::spawn(move || {
        loop {
            let _sig = set.wait();
            let pid = child_for_thread.get();
            if pid > 0 {
                let _ = kill(Pid::from_raw(pid), Signal::SIGINT);
            } else {
                std::process::exit(130);
            }
        }
    });
    child
}

#[cfg(not(unix))]
pub fn install() -> ChildPid {
    ChildPid::default()
}
