// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf` process entry point.
//!
//! Internal re-exec modes are intercepted before CLI parsing so their stdio
//! protocol and hidden command surface remain isolated from public commands.

use aiperf_cli::{dispatch, execute_mode};

// The shared allocator provider (libaiperf_alloc_v1.so) is loaded as a
// mandatory non-delay dependency before any Rust code runs.  Set
// MIMALLOC_ARENA_EAGER_COMMIT=0 in the environment to suppress eager arena
// commit; no per-binary preinit hook is needed.
#[global_allocator]
static GLOBAL: aiperf_allocator_shim::MiMallocShim = aiperf_allocator_shim::MiMallocShim;

fn main() {
    aiperf_cli::diagnostics::register_sigusr1_faulthandler();
    let argv: Vec<String> = std::env::args().skip(1).collect();

    aiperf_cli::logging::init(&argv);

    if execute_mode::is_execution_mode(&argv) {
        execute_mode::dispatch(&argv);
    }

    let code = match dispatch::run(&argv) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("aiperf: {error:#}");
            1
        }
    };
    std::process::exit(code);
}
