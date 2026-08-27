// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line entry point for repository maintenance measurements.

use std::path::PathBuf;

use aiperf_xtask::abi_closure::{Baseline, Seeds, compute, compute_in, workspace_root};
use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        bail!("usage: cargo xtask <abi-closure|abi-gate> [options]");
    };

    match command.as_str() {
        "abi-closure" => {
            let (seeds_path, workspace, is_json) = closure_options(arguments)?;
            let seeds = Seeds::load(seeds_path)?;
            let closure = match workspace {
                Some(workspace) => compute_in(&workspace, &seeds)?,
                None => compute(&seeds)?,
            };
            let baseline = Baseline::from_closure(&closure);
            if is_json {
                println!("{}", serde_json::to_string(&baseline)?);
            } else {
                println!(
                    "{} types / {} files / {} type lines / {} file lines",
                    baseline.types, baseline.files, baseline.type_lines, baseline.file_lines
                );
            }
        }
        "abi-gate" => {
            if arguments.next().is_some() {
                bail!("abi-gate accepts no options");
            }
            let crate_root = workspace_root().join("xtask");
            let seeds = Seeds::load(crate_root.join("abi-seeds.toml"))?;
            let measured = Baseline::from_closure(&compute(&seeds)?);
            let baseline_path = crate_root.join("abi-baseline.json");
            let baseline: Baseline = serde_json::from_str(
                &std::fs::read_to_string(&baseline_path)
                    .with_context(|| format!("reading {}", baseline_path.display()))?,
            )
            .with_context(|| format!("parsing {}", baseline_path.display()))?;
            if measured.types > baseline.types {
                bail!(
                    "ABI closure grew from {} to {} types; update the boundary or review the baseline",
                    baseline.types,
                    measured.types
                );
            }
        }
        other => bail!("unknown xtask subcommand {other:?}"),
    }

    Ok(())
}

fn closure_options(
    arguments: impl Iterator<Item = String>,
) -> Result<(PathBuf, Option<PathBuf>, bool)> {
    let mut seeds = PathBuf::from("xtask/abi-seeds.toml");
    let mut workspace = None;
    let mut is_json = false;
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--seeds" => {
                seeds = PathBuf::from(
                    arguments
                        .next()
                        .context("--seeds requires a path argument")?,
                );
            }
            "--workspace" => {
                workspace = Some(PathBuf::from(
                    arguments
                        .next()
                        .context("--workspace requires a path argument")?,
                ));
            }
            "--json" => is_json = true,
            other => bail!("unknown abi-closure option {other:?}"),
        }
    }
    Ok((seeds, workspace, is_json))
}
