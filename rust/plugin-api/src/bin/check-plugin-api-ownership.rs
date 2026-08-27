// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Documentation guard for the plugin source API boundary.
//!
//! Prints the generation-1 ownership table and verifies that every row in
//! `aiperf_plugin_api::GENERATION_1_SURFACE` also appears in
//! `docs/specs/plugin-api-ownership.md`. A boundary item entered into the
//! constant without its ownership facts therefore fails this check rather than
//! landing silently.
//!
//! Both sides of the comparison are hand-maintained. This guard reads no Rust
//! source and consumes no rustdoc output, so it does not detect an item added
//! to the crate's source and never entered into the constant, and it does not
//! compare the argument, return, ownership, or phase columns against anything.
//! The source-derived direction is Task 6 work; see the Task-5 amendment in
//! `docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-implementation.md`.
//!
//! Exit codes: `0` when the table and the spec agree, `1` when a row is missing
//! from the spec, `2` when the spec could not be read.

use std::{path::PathBuf, process::ExitCode};

use aiperf_plugin_api::{GENERATION_1_SURFACE, ownership::render_surface_table};

/// Path to the spec, relative to this crate's manifest directory.
const SPEC_RELATIVE_PATH: &str = "../../docs/specs/plugin-api-ownership.md";

fn main() -> ExitCode {
    print!("{}", render_surface_table(GENERATION_1_SURFACE));

    let spec_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SPEC_RELATIVE_PATH);
    let spec = match std::fs::read_to_string(&spec_path) {
        Ok(spec) => spec,
        Err(error) => {
            eprintln!(
                "error: cannot read ownership spec at {}: {error}",
                spec_path.display()
            );
            return ExitCode::from(2);
        }
    };

    let missing: Vec<&str> = GENERATION_1_SURFACE
        .iter()
        .map(|row| row.item)
        .filter(|item| !spec.contains(*item))
        .collect();

    if missing.is_empty() {
        println!(
            "\nok: all {} boundary items documented in {}",
            GENERATION_1_SURFACE.len(),
            SPEC_RELATIVE_PATH
        );
        return ExitCode::SUCCESS;
    }

    eprintln!(
        "\nerror: {} boundary item(s) missing from {}:",
        missing.len(),
        spec_path.display()
    );
    for item in missing {
        eprintln!("  - {item}");
    }
    ExitCode::FAILURE
}
