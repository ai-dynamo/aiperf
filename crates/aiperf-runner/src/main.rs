// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stdio entry point for one orchestrator-authored benchmark run.

use std::io::{self, BufReader, Write};

use aiperf_runner::{
    RUNNER_PROTOCOL_VERSION, RunRequest, RunTerminal, RunnerCapabilities, execute_run,
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && arguments[0] == "--capabilities" {
        write_json_line(&RunnerCapabilities::current(), 0);
    }
    if !arguments.is_empty() {
        eprintln!("usage: aiperf-runner [--capabilities]");
        std::process::exit(2);
    }
    let terminal = match serde_json::from_reader::<_, RunRequest>(BufReader::new(io::stdin())) {
        Ok(request) if request.protocol_version == RUNNER_PROTOCOL_VERSION => {
            let run_id = request.run.benchmark_id.clone();
            match execute_run(request) {
                Ok(result) => result,
                Err(error) => RunTerminal::failed(Some(run_id), "execution", format!("{error:#}")),
            }
        }
        Ok(request) => RunTerminal::failed(
            Some(request.run.benchmark_id),
            "protocol",
            format!(
                "runner protocol {} is unsupported; expected {}",
                request.protocol_version, RUNNER_PROTOCOL_VERSION
            ),
        ),
        Err(error) => {
            RunTerminal::failed(None, "protocol", format!("invalid run request: {error}"))
        }
    };
    write_json_line(&terminal, if terminal.success { 0 } else { 1 });
}

fn write_json_line(value: &impl serde::Serialize, exit_code: i32) -> ! {
    let mut stdout = io::stdout().lock();
    if serde_json::to_writer(&mut stdout, value).is_err()
        || stdout.write_all(b"\n").is_err()
        || stdout.flush().is_err()
    {
        std::process::exit(2);
    }
    std::process::exit(exit_code);
}
