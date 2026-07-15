// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Top-level command routing: `profile` is native, everything else delegates.

use crate::{delegate, profile};

/// Route one invocation (argv with the program name already stripped). Returns
/// the process exit code.
///
/// Only `profile` is owned natively. Every other subcommand — and a bare
/// `aiperf` with no subcommand — is handed to the Python frontend unchanged, so
/// the native binary is a drop-in front door while the port is incremental.
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    match argv.first().map(String::as_str) {
        Some("profile") => profile::run(&argv[1..]),
        _ => delegate::exec_python(argv),
    }
}
