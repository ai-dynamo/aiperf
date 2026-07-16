// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf` front door AND execution engine.
//!
//! The single `aiperf` binary owns both roles: `profile`/`config` (and the other
//! subcommands) as the human-facing front door, and — behind an INTERNAL protocol
//! the front door re-execs itself over — one benchmark run's execution
//! ([`aiperf_cli::execute_mode`]). The re-exec modes (`--execute`, `--cell`,
//! `--aggregator`) are intercepted here BEFORE clap parses, so they never appear
//! in `--help` and are not part of the public CLI surface; they exist only for
//! the front door → child re-exec seam. (Capabilities is an in-process function,
//! not a subprocess mode.)

use aiperf_cli::{dispatch, execute_mode};

// mimalloc as the global allocator (moved from the deleted aiperf-runner binary):
// per-request allocation churn on the execution hot path was the top profiled
// hotspot.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(target_os = "linux")]
#[used]
#[unsafe(link_section = ".init_array.00100")]
static AIPERF_MIMALLOC_PREINIT: unsafe extern "C" fn() = configure_mimalloc_before_process_init;

#[cfg(target_os = "linux")]
unsafe extern "C" fn configure_mimalloc_before_process_init() {
    // mimalloc's own Linux constructor has priority 101. This priority-100 hook
    // changes only its uninitialized default before that constructor commits the
    // initial arena. Leaving the option uninitialized lets mimalloc's own parser
    // honor canonical, case-insensitive, and legacy environment spellings.
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
    let argv: Vec<String> = std::env::args().skip(1).collect();

    // Install the tracing subscriber (stderr console + deferred `logs/aiperf.log`
    // file layer) at INFO by default, honoring `--verbose`/`--extra-verbose`/
    // `--log-level` and the `AIPERF_LOG` env. Done before the `--execute`
    // interception below so the re-exec child inherits the same subscriber.
    aiperf_cli::logging::init(&argv);

    // Internal re-exec protocol: the front door spawns `aiperf --execute` (and the
    // cellular launcher `aiperf --cell` / `--aggregator`) for one run's execution.
    // Intercept before clap so these stay off the public CLI surface entirely and
    // the stdin protocol channel is never touched by argument parsing.
    if execute_mode::is_execution_mode(&argv) {
        execute_mode::dispatch(&argv);
    }

    let code = match dispatch::run(&argv) {
        Ok(code) => code,
        Err(error) => {
            // App-layer errors are reported with their full anyhow context chain.
            eprintln!("aiperf: {error:#}");
            1
        }
    };
    std::process::exit(code);
}
