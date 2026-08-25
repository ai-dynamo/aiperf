// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing.

use crate::{
    analyze_trace, cellular_role, chat, compare, config, delegate, eval, graph, kube, metrics_list,
    profile, results_sidecar, serve, slurm, speed_bench, synthesize, validate,
};

/// Route arguments with the program name removed and return the process exit code.
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    match argv.first().map(String::as_str) {
        Some("profile") => profile::run(&argv[1..]),
        Some("config") => config::run(&argv[1..]),
        Some("graph") => graph::run(&argv[1..]),
        Some("kube") => kube::command::run(&argv[1..]),
        Some("eval") => eval::run(&argv[1..]),
        Some("serve") => serve::run(&argv[1..]),
        Some("controller") => cellular_role::run_controller(&argv[1..]),
        Some("cell") => cellular_role::run_cell(&argv[1..]),
        Some("aggregator") => cellular_role::run_aggregator(&argv[1..]),
        Some("slurm") if argv.get(1).map(String::as_str) == Some(slurm::RUN_SUBCOMMAND) => {
            slurm::run(&argv[2..])
        }
        Some("slurm") if argv.get(1).map(String::as_str) == Some("generate") => {
            delegate::exec_rust_shim(&delegate::python_executable()?, "slurm-generate", &argv[2..])
        }
        Some("results-sidecar") => results_sidecar::run(&argv[1..]),
        Some("analyze-trace") => analyze_trace::run(&argv[1..]),
        Some("compare") => compare::run(&argv[1..]),
        Some("chat") => chat::run(&argv[1..]),
        Some("validate") => validate::run(&argv[1..]),
        Some("speed-bench-report") => speed_bench::run(&argv[1..]),
        Some("synthesize") => synthesize::run(&argv[1..]),
        Some("metrics") => metrics_list::run(&argv[1..]),
        Some("analyze" | "plot" | "plugins")
        | None
        | Some("-h" | "--help" | "--install-completion") => {
            delegate::exec_python_utility(argv)
        }
        Some("service") => anyhow::bail!(
            "aiperf service is unavailable from the native binary; use `aiperf-python service`"
        ),
        _ => anyhow::bail!(
            "unsupported native aiperf command; supported delegated utilities are analyze, plot, plugins, --help, and --install-completion"
        ),
    }
}
