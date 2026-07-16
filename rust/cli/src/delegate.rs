// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Delegation of not-natively-owned subcommands to the Python `aiperf` app.
//!
//! `profile` and `config` are native (see [`crate::profile`] / [`crate::config`]).
//! Every other subcommand (chat/plot/plugins/service/synthesize/analyze-trace/
//! speed-bench-report/validate and a bare `aiperf`) wraps an inherently-Python
//! subsystem (matplotlib, the async chat client, the service mesh, the dataset
//! generators). Two dispatch modes:
//!
//! * **`pyo3-embed` build (default for the shipped native binary):** run the
//!   Python `aiperf.entrypoint.main(argv)` **in-process** via the embedded
//!   CPython — ZERO subprocess, so the whole CLI is shell-out-free.
//! * **lean build (no `pyo3-embed`):** spawn `python -m aiperf <argv>` as a
//!   subprocess so the binary can stay Python-free.

/// Dispatch `argv` (subcommand + args, program name already stripped) to the
/// Python `aiperf` app, returning its exit code.
#[cfg(feature = "pyo3-embed")]
pub fn exec_python(argv: &[String]) -> anyhow::Result<i32> {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    Python::with_gil(|py| -> PyResult<i32> {
        // Present the full command line to Python so commands that read
        // `sys.argv` (e.g. the `cli_command` echo) see `aiperf <argv>`.
        let full: Vec<String> = std::iter::once("aiperf".to_string())
            .chain(argv.iter().cloned())
            .collect();
        py.import("sys")?.setattr("argv", PyList::new(py, &full)?)?;

        // `aiperf.entrypoint.main(list)` runs the cyclopts app over the token
        // list and returns the command's `int | None` result.
        let main = py.import("aiperf.entrypoint")?.getattr("main")?;
        match main.call1((PyList::new(py, argv)?,)) {
            Ok(ret) => {
                if ret.is_none() {
                    Ok(0)
                } else {
                    Ok(ret.extract::<i32>().unwrap_or(0))
                }
            }
            Err(err) if err.is_instance_of::<pyo3::exceptions::PySystemExit>(py) => {
                // cyclopts raises SystemExit on --help / parse errors. Match
                // CPython's `SystemExit` semantics exactly: `None` → 0, an int →
                // that int, anything else (e.g. a string message, already
                // printed) → 1.
                let code_obj = err.value(py).getattr("code").ok();
                let code = match code_obj {
                    None => 0,
                    Some(c) if c.is_none() => 0,
                    Some(c) => c.extract::<i32>().unwrap_or(1),
                };
                Ok(code)
            }
            Err(err) => {
                err.print(py);
                Ok(1)
            }
        }
    })
    .map_err(|e| anyhow::anyhow!("in-process aiperf delegation failed: {e}"))
}

/// Subprocess fallback for the lean (Python-free) build: spawn
/// `python -m aiperf <argv>` inheriting stdio. The interpreter is
/// `$AIPERF_PYTHON` when set, else `python`.
#[cfg(not(feature = "pyo3-embed"))]
pub fn exec_python(argv: &[String]) -> anyhow::Result<i32> {
    use std::process::Command;
    let python = std::env::var("AIPERF_PYTHON").unwrap_or_else(|_| "python".to_string());
    let status = Command::new(&python)
        .arg("-m")
        .arg("aiperf")
        .args(argv)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to delegate to `{python} -m aiperf`: {e}"))?;
    Ok(status.code().unwrap_or(1))
}
