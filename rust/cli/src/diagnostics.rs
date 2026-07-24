// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native process diagnostics hooks.

/// Register a best-effort SIGUSR1 stack-dump handler on Linux.
///
/// `kill -USR1 <pid>` writes a backtrace for the current process to stderr.
pub fn register_sigusr1_faulthandler() {
    #[cfg(unix)]
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        use nix::sys::signal::{SaFlags, SigAction, SigHandler, Signal};

        static REGISTERED: AtomicBool = AtomicBool::new(false);
        if REGISTERED.swap(true, Ordering::SeqCst) {
            return;
        }
        extern "C" fn on_sigusr1(_: i32) {
            eprintln!("SIGUSR1 backtrace:");
            eprintln!("{:?}", std::backtrace::Backtrace::force_capture());
        }
        let action = SigAction::new(
            SigHandler::Handler(on_sigusr1),
            SaFlags::empty(),
            nix::sys::signal::SigSet::empty(),
        );
        unsafe {
            let _ = nix::sys::signal::sigaction(Signal::SIGUSR1, &action);
        }
    }
}
