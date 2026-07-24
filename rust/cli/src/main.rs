// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf` process entry point.
//!
//! Internal re-exec modes are intercepted before CLI parsing so their stdio
//! protocol and hidden command surface remain isolated from public commands.

use aiperf_cli::{dispatch, execute_mode};

// Per-request allocation dominates the execution hot path.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(target_os = "linux")]
#[used]
#[unsafe(link_section = ".init_array.00100")]
static AIPERF_MIMALLOC_PREINIT: unsafe extern "C" fn() = configure_mimalloc_before_process_init;

#[cfg(target_os = "linux")]
unsafe extern "C" fn configure_mimalloc_before_process_init() {
    // mimalloc's Linux constructor has priority 101. This priority-100 hook
    // changes its default before that constructor commits the initial arena.
    // Leaving the option uninitialized lets mimalloc parse supported environment
    // spellings.
    // The C shim (build.rs) resolves the experimental enum from the exact header
    // compiled by libmimalloc-sys instead of duplicating its unstable numeric value.
    // SAFETY: mimalloc has not run process initialization and no Rust heap
    // allocation can precede an ELF init-array constructor.
    unsafe { libmimalloc_sys::mi_option_set_default(aiperf_mi_option_arena_eager_commit(), 0) };
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn aiperf_mi_option_arena_eager_commit() -> libmimalloc_sys::mi_option_t;
}

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
