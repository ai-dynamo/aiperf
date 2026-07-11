// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Process resource-limit helpers shared by the benchmark drivers.

/// Raise this process's open-file soft limit to its hard limit, so a
/// high-concurrency run can open tens of thousands of sockets without a shell
/// `ulimit`. No root needed (soft up to hard).
#[cfg(target_os = "linux")]
pub fn raise_fd_limit() {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) == 0 {
            lim.rlim_cur = lim.rlim_max;
            let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &lim);
        }
    }
}
#[cfg(not(target_os = "linux"))]
pub fn raise_fd_limit() {}
