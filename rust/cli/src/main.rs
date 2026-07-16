// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf` front door. Owns `profile` for a single run; delegates the rest.

use aiperf_cli::dispatch;

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("AIPERF_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let argv: Vec<String> = std::env::args().skip(1).collect();
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
