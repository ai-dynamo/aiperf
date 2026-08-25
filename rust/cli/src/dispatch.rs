// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing.

use crate::{
    analyze_trace, cellular_role, chat, compare, config, delegate, eval, graph, kube, metrics_list,
    profile, results_sidecar, serve, slurm, speed_bench, synthesize, validate,
};

const PUBLIC_COMMANDS: &[&str] = &[
    "profile",
    "config",
    "graph",
    "kube",
    "eval",
    "serve",
    "controller",
    "cell",
    "aggregator",
    "slurm",
    "results-sidecar",
    "analyze-trace",
    "compare",
    "chat",
    "validate",
    "speed-bench-report",
    "synthesize",
    "metrics",
    "analyze",
    "plot",
];

fn print_help() {
    println!(
        "AIPerf {}\n\nUsage: aiperf <COMMAND> [ARGS...]\n\nCommands:\n  {}\n\nPython utilities: analyze, plot\n\nOptions:\n  -h, --help                       Print help\n  -V, --version                    Print version\n      --install-completion <SHELL> Print completion script for bash, zsh, or fish",
        env!("CARGO_PKG_VERSION"),
        PUBLIC_COMMANDS.join("\n  "),
    );
}

fn print_completion(shell: &str) -> anyhow::Result<()> {
    let commands = PUBLIC_COMMANDS.join(" ");
    match shell {
        "bash" => println!(
            "_aiperf() {{\n    local commands=\"{commands}\"\n    COMPREPLY=( $(compgen -W \"$commands\" -- \"${{COMP_WORDS[COMP_CWORD]}}\") )\n}}\ncomplete -F _aiperf aiperf"
        ),
        "zsh" => println!("#compdef aiperf\n\n_arguments '1:command:({commands})'"),
        "fish" => println!(
            "complete -c aiperf -f\ncomplete -c aiperf -n '__fish_use_subcommand' -a '{commands}'"
        ),
        _ => anyhow::bail!(
            "unsupported completion shell `{shell}`; supported shells are bash, zsh, and fish"
        ),
    }
    Ok(())
}

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
        Some("slurm")
            if argv.get(1).map(String::as_str) == Some(slurm::generate::GENERATE_SUBCOMMAND) =>
        {
            slurm::generate::run(&argv[2..])
        }
        Some("results-sidecar") => results_sidecar::run(&argv[1..]),
        Some("analyze-trace") => analyze_trace::run(&argv[1..]),
        Some("compare") => compare::run(&argv[1..]),
        Some("chat") => chat::run(&argv[1..]),
        Some("validate") => validate::run(&argv[1..]),
        Some("speed-bench-report") => speed_bench::run(&argv[1..]),
        Some("synthesize") => synthesize::run(&argv[1..]),
        Some("metrics") => metrics_list::run(&argv[1..]),
        Some("analyze" | "plot") => delegate::exec_python_utility(argv),
        None => {
            print_help();
            Ok(0)
        }
        Some("-h" | "--help") if argv.len() == 1 => {
            print_help();
            Ok(0)
        }
        Some("--install-completion") => {
            let shell = argv
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("--install-completion requires a shell"))?;
            if argv.len() != 2 {
                anyhow::bail!("--install-completion accepts exactly one shell");
            }
            print_completion(shell)?;
            Ok(0)
        }
        Some("-V" | "--version") => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        Some("service") => anyhow::bail!(
            "aiperf service is unavailable from the native binary; use `aiperf-python service`"
        ),
        _ => anyhow::bail!(
            "unsupported native aiperf command; supported Python utilities are analyze and plot"
        ),
    }
}
