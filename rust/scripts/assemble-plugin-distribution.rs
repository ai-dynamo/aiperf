// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assemble a plugin distribution generation from built artifacts.
//!
//! Run with:
//!
//! ```bash
//! # From the workspace rust/ dir:
//! cargo run -p aiperf-plugin-packaging-tests --bin assemble-plugin-distribution -- \
//!     --fixture plugin-packaging-tests/fixtures/candidate-generation/fixture.toml \
//!     --artifacts-dir target/release \
//!     --output-dir target/plugin-release-candidate
//! ```
//!
//! The file lives under `rust/scripts/` because it is release tooling rather
//! than a product surface, but it is a real Cargo binary target of
//! `aiperf-plugin-packaging-tests` (declared with an out-of-package `path`), so
//! it is compiled, linted, and gated with everything else and shares exactly
//! one implementation with the distribution census tests.
//!
//! ## What this binary does
//!
//! 1. Loads the candidate generation fixture, refusing an unknown schema, a
//!    duplicate package id, or a file name that is not a plain relative name.
//! 2. Optionally materializes each package's declared synthetic bytes, so the
//!    pipeline can be rehearsed without a build. Synthetic bytes never replace
//!    a staged build product: a path that already holds bytes is refused.
//! 3. Hashes every declared artifact and manifest under `--artifacts-dir` into
//!    canonical `blake3:<hex>` digests, reading each through a no-follow
//!    descriptor so a planted symlink cannot redirect the hash.
//! 4. Publishes `plugin-inventory.json` into `--output-dir`, atomically and
//!    authenticated by its own inventory digest.
//!
//! `--auto-generation` advances past a prior inventory that verifies, and
//! refuses outright when one is present but unverifiable, so a tampered
//! document can never be answered with a lower generation.
//!
//! It exits 0 on success and 1 on any refusal, printing the reason to stderr.

use std::path::PathBuf;
use std::process::ExitCode;

use aiperf_plugin_packaging_tests::assemble::{
    CandidateFixture, INVENTORY_FILE_NAME, assemble_distribution, next_generation,
};

/// Resolved command line for one assembly run.
struct Args {
    /// Candidate generation fixture to assemble.
    fixture: PathBuf,
    /// Directory holding the built artifacts and manifests.
    artifacts_dir: PathBuf,
    /// Directory the inventory document is published into.
    output_dir: PathBuf,
    /// Advance past any inventory already published in the output directory.
    is_auto_generation: bool,
    /// Write each package's declared synthetic bytes before hashing.
    needs_synthetic_artifacts: bool,
}

const USAGE: &str = "usage: assemble-plugin-distribution \
--fixture FIXTURE.toml --artifacts-dir DIR --output-dir DIR \
[--auto-generation] [--materialize-synthetic]";

/// Parse the command line, or return the reason it is unusable.
fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut fixture = None;
    let mut artifacts_dir = None;
    let mut output_dir = None;
    let mut is_auto_generation = false;
    let mut needs_synthetic_artifacts = false;

    let mut rest = argv.into_iter();
    while let Some(flag) = rest.next() {
        let mut value = || {
            rest.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--fixture" => fixture = Some(PathBuf::from(value()?)),
            "--artifacts-dir" => artifacts_dir = Some(PathBuf::from(value()?)),
            "--output-dir" => output_dir = Some(PathBuf::from(value()?)),
            "--auto-generation" => is_auto_generation = true,
            "--materialize-synthetic" => needs_synthetic_artifacts = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    Ok(Args {
        fixture: fixture.ok_or("--fixture is required")?,
        artifacts_dir: artifacts_dir.ok_or("--artifacts-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        is_auto_generation,
        needs_synthetic_artifacts,
    })
}

/// Assemble one generation, returning the human-readable receipt.
fn run(args: &Args) -> Result<String, String> {
    let mut fixture = CandidateFixture::load(&args.fixture).map_err(|e| e.to_string())?;
    if args.is_auto_generation {
        let generation =
            next_generation(&args.output_dir, fixture.generation).map_err(|e| e.to_string())?;
        fixture = fixture.with_generation(generation);
    }
    if args.needs_synthetic_artifacts {
        fixture
            .materialize_synthetic_artifacts(&args.artifacts_dir)
            .map_err(|e| e.to_string())?;
    }

    let published = assemble_distribution(&fixture, &args.artifacts_dir, &args.output_dir)
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "published {INVENTORY_FILE_NAME} generation {} with {} package(s) at {}",
        fixture.generation,
        fixture.packages.len(),
        published.display()
    ))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(reason) => {
            eprintln!("{reason}\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(reason) => {
            eprintln!("assemble-plugin-distribution: {reason}");
            ExitCode::FAILURE
        }
    }
}
