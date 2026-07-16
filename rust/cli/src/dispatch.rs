// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing: `profile`/`config` and the Kubernetes cellular
//! roles (`controller`/`cell`/`aggregator`) are native; everything else delegates.

use crate::{
    analyze_trace, cellular_role, chat, config, delegate, profile, speed_bench, synthesize,
    validate,
};

/// Route one invocation (argv with the program name already stripped). Returns
/// the process exit code.
///
/// `profile` and `config` are owned natively. Every other subcommand — and a
/// bare `aiperf` — goes through [`delegate::exec_python`]: run in-process via the
/// embedded CPython in the `pyo3-embed`/`search-pyo3` build (zero subprocess
/// shell-out for the whole CLI), or spawned as `python -m aiperf` in the lean
/// Python-free build.
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    match argv.first().map(String::as_str) {
        Some("profile") => profile::run(&argv[1..]),
        Some("config") => config::run(&argv[1..]),
        // Native Kubernetes cellular roles (operator JobSet pod commands). The
        // native binary owns cellular execution; these no longer delegate to the
        // Python `_cellular_role` adapter.
        Some("controller") => cellular_role::run_controller(&argv[1..]),
        Some("cell") => cellular_role::run_cell(&argv[1..]),
        Some("aggregator") => cellular_role::run_aggregator(&argv[1..]),
        Some("analyze-trace") => analyze_trace::run(&argv[1..]),
        Some("chat") => chat::run(&argv[1..]),
        Some("validate") => validate::run(&argv[1..]),
        Some("speed-bench-report") => speed_bench::run(&argv[1..]),
        Some("synthesize") => synthesize::run(&argv[1..]),
        _ => delegate::exec_python(argv),
    }
}
