// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static comparator entry point for the plugin parity experiment.
//!
//! This binary is the statically-linked baseline the dynamic plugin
//! distribution is measured against. Before it is worth measuring anything, it
//! proves that the components it links are exactly the components the dynamic
//! distribution publishes: a parity number taken across two different censuses
//! measures the census difference, not the linkage difference.
//!
//! Run with:
//!
//! ```bash
//! # From the workspace rust/ dir:
//! cargo run -p aiperf-plugin-static-comparator            # default census
//! cargo run -p aiperf-plugin-static-comparator -- --full  # full census
//! ```
//!
//! Exits 0 when the census matches and 1 on any drift, printing the differing
//! components to stderr.

use std::process::ExitCode;

use aiperf_plugin_static_comparator::static_inventory::{
    DEFAULT_DISTRIBUTION_CENSUS, StaticComparatorRegistry, default_distribution_registry,
    full_distribution_census, full_distribution_registry,
};

const USAGE: &str = "usage: aiperf-plugin-static-comparator [--full]";

/// A census as `(id, version)` pairs in canonical id order.
type Census = Vec<(&'static str, &'static str)>;

/// The registry a distribution links, paired with the census it must match.
type SelectedDistribution = (StaticComparatorRegistry, Census);

/// Build the registry and expected census the requested distribution demands.
fn selected_distribution(is_full: bool) -> Result<SelectedDistribution, String> {
    if is_full {
        let registry = full_distribution_registry().map_err(|e| e.to_string())?;
        Ok((registry, full_distribution_census()))
    } else {
        let registry = default_distribution_registry().map_err(|e| e.to_string())?;
        Ok((registry, DEFAULT_DISTRIBUTION_CENSUS.to_vec()))
    }
}

fn main() -> ExitCode {
    let mut is_full = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--full" => is_full = true,
            other => {
                eprintln!("unknown argument `{other}`\n{USAGE}");
                return ExitCode::FAILURE;
            }
        }
    }

    let (registry, expected) = match selected_distribution(is_full) {
        Ok(selected) => selected,
        Err(reason) => {
            eprintln!("aiperf-plugin-static-comparator: {reason}");
            return ExitCode::FAILURE;
        }
    };

    match registry.assert_census(&expected) {
        Ok(()) => {
            let label = if is_full { "full" } else { "default" };
            println!(
                "static comparator census matches the {label} distribution ({} components)",
                registry.len()
            );
            for (id, version) in registry.census() {
                println!("  {id} {version}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("aiperf-plugin-static-comparator: {error}");
            ExitCode::FAILURE
        }
    }
}
