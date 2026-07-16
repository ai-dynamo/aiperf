// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing: `profile` is native, everything else delegates.

use crate::{config, delegate, profile};

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
        _ => delegate::exec_python(argv),
    }
}
