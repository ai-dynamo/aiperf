// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graceful SIGINT/SIGTERM forwarding to the active `aiperf` child.
//!
//! Ports `orchestrator/runner_installation.py::_communicate_forwarding_signals`:
//! block the terminating signals in the process and, on the first delivery,
//! forward ONE SIGINT to the running child so the runner drains and writes a
//! partial `was_cancelled=true` report instead of dying abruptly. Between cells
//! (no child), the first signal exits the process (code 130) so Ctrl-C still
//! stops a sweep.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// Shared handle: the current child PID (`0` when no child is running).
#[derive(Clone, Default)]
pub struct ChildPid(Arc<AtomicI32>);

impl ChildPid {
    /// Publish the running child's PID (call right after spawn).
    pub fn set(&self, pid: u32) {
        self.0.store(pid as i32, Ordering::SeqCst);
    }
    /// Clear the PID (call after the child is reaped).
    pub fn clear(&self) {
        self.0.store(0, Ordering::SeqCst);
    }
    fn get(&self) -> i32 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Install the forwarder once. On unix, blocks SIGINT/SIGTERM and spawns a
/// daemon thread that `sigwait`s for them; each delivery forwards SIGINT to the
/// current child (or exits 130 when none is running). No-op on non-unix.
#[cfg(unix)]
pub fn install() -> ChildPid {
    use nix::sys::signal::{SigSet, Signal, kill};
    use nix::unistd::Pid;

    let child = ChildPid::default();
    let mut set = SigSet::empty();
    set.add(Signal::SIGINT);
    set.add(Signal::SIGTERM);
    // Block in this (main) thread; spawned threads inherit the mask, so the
    // forwarder is the only place these signals are handled.
    if set.thread_block().is_err() {
        return child;
    }
    let child_for_thread = child.clone();
    std::thread::spawn(move || {
        loop {
            // `sigwait` blocks until one of the masked signals is delivered.
            let _sig = set.wait();
            let pid = child_for_thread.get();
            if pid > 0 {
                // Forward one SIGINT; the runner drains + writes a partial report.
                let _ = kill(Pid::from_raw(pid), Signal::SIGINT);
            } else {
                // No child in flight: honor the interrupt and stop.
                std::process::exit(130);
            }
        }
    });
    child
}

/// No-op forwarder for non-unix targets (the product target is Linux).
#[cfg(not(unix))]
pub fn install() -> ChildPid {
    ChildPid::default()
}
