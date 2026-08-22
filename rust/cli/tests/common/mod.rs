// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared integration-test support.

/// Run a test body with enough stack for clap's derived `ProfileFlags` parser.
pub(crate) fn on_profile_flags_stack(body: impl FnOnce() + Send + 'static) {
    let worker = std::thread::Builder::new()
        .name("profile-flags-test".to_owned())
        .stack_size(4 * 1024 * 1024)
        .spawn(body)
        .expect("spawn ProfileFlags test worker");
    if let Err(payload) = worker.join() {
        std::panic::resume_unwind(payload);
    }
}
