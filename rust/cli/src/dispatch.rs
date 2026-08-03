// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing.

use crate::{
    analyze_trace, cellular_role, chat, compare, config, delegate, metrics_list, profile,
    results_sidecar, serve, slurm, speed_bench, synthesize, validate,
};

/// Route arguments with the program name removed and return the process exit code.
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    match argv.first().map(String::as_str) {
        Some("profile") => profile::run(&argv[1..]),
        Some("config") => config::run(&argv[1..]),
        Some("serve") => serve::run(&argv[1..]),
        Some("controller") => cellular_role::run_controller(&argv[1..]),
        Some("cell") => cellular_role::run_cell(&argv[1..]),
        Some("aggregator") => cellular_role::run_aggregator(&argv[1..]),
        // `slurm run` is the native per-task rank dispatch; every other `slurm`
        // subcommand (`generate`) is delegated to the Python CLI.
        Some("slurm") if argv.get(1).map(String::as_str) == Some(slurm::RUN_SUBCOMMAND) => {
            slurm::run(&argv[2..])
        }
        Some("slurm") => delegate::exec_python(argv),
        Some("results-sidecar") => results_sidecar::run(&argv[1..]),
        Some("analyze-trace") => analyze_trace::run(&argv[1..]),
        Some("compare") => compare::run(&argv[1..]),
        Some("chat") => chat::run(&argv[1..]),
        Some("validate") => validate::run(&argv[1..]),
        Some("speed-bench-report") => speed_bench::run(&argv[1..]),
        Some("synthesize") => synthesize::run(&argv[1..]),
        Some("metrics") => metrics_list::run(&argv[1..]),
        _ => delegate::exec_python(argv),
    }
}
