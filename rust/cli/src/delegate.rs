// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Delegation of Python-backed subcommands to the `aiperf` Python app.
//!
//! Builds with `pyo3-embed` invoke `aiperf.entrypoint.main(argv)` in-process.
//! Other builds spawn `python -m aiperf <argv>`.

/// Dispatch arguments to the Python `aiperf` app and return its exit code.
#[cfg(feature = "pyo3-embed")]
pub fn exec_python(argv: &[String]) -> anyhow::Result<i32> {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    Python::with_gil(|py| -> PyResult<i32> {
        // Commands that inspect `sys.argv` require the executable name.
        let full: Vec<String> = std::iter::once("aiperf".to_string())
            .chain(argv.iter().cloned())
            .collect();
        py.import("sys")?.setattr("argv", PyList::new(py, &full)?)?;

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
                // Preserve CPython's `SystemExit` semantics: `None` is zero, an
                // integer is its exit code, and any other value is one.
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

/// Spawn `python -m aiperf <argv>` with inherited stdio.
///
/// `$AIPERF_PYTHON` selects the interpreter and defaults to `python`.
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
