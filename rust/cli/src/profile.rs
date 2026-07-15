// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The native `aiperf profile` command (single run).
//!
//! Flow: parse flags → load the native [`BenchmarkRun`] → serialize the
//! protocol-v2 execute envelope → spawn the unchanged `aiperf-runner` once →
//! map its terminal outcome to a process exit code. A YAML `--config` is not yet
//! ported, so it is delegated to the Python frontend; a multi-run sweep is
//! rejected with a clear error by the loader.

use crate::model::{Operation, RunnerRequest};
use crate::{delegate, execute, flags::ProfileFlags, load, runner_install};

/// Run `aiperf profile <args>` natively. Returns the process exit code.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let flags = match ProfileFlags::parse_from_args(args) {
        Ok(flags) => flags,
        // clap already rendered help/usage to stderr; propagate its exit code.
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    // YAML config parsing is not yet native; hand the whole invocation to Python
    // so `-f config.yaml` keeps working through the front door.
    if flags.config_file.is_some() {
        tracing::info!("delegating YAML-config profile run to the Python frontend");
        let mut argv = vec!["profile".to_string()];
        argv.extend_from_slice(args);
        return delegate::exec_python(&argv);
    }

    let run = load::resolve(&flags)?;
    let request = RunnerRequest::new(Operation::Execute, run);
    let payload = serde_json::to_vec(&request)
        .map_err(|e| anyhow::anyhow!("failed to serialize the runner request: {e}"))?;

    let runner = runner_install::resolve()?;
    let terminal = execute::run_once(&runner, &payload)?;

    if terminal.success {
        if let Some(path) = &terminal.report_path {
            tracing::info!(report = %path, "run complete");
        }
        Ok(0)
    } else {
        let detail = terminal
            .error
            .as_deref()
            .unwrap_or("native benchmark failed");
        eprintln!("aiperf: {detail}");
        Ok(if terminal.returncode == 0 {
            1
        } else {
            terminal.returncode
        })
    }
}
