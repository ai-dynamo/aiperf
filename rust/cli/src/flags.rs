// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The `aiperf profile` CLI flag surface (clap), byte-exact to the Python names.
//!
//! This is the CLI half of the pre-translation input contract. Flag long-names
//! and aliases mirror `src/aiperf/config/flags/cli_config.py` (which in turn
//! keeps GenAI-Perf aliases). The loader ([`crate::load`]) maps these into the
//! native [`crate::model::BenchmarkRun`].
//!
//! Only the flags needed for the single-run synthetic path are modeled today;
//! the surface grows as sections are ported. Sweep-capable numeric flags are
//! taken as `String` so a comma-list (`--concurrency 10,20,30`) can be detected
//! and rejected as multi-run rather than mis-parsed.

use std::path::PathBuf;

use clap::Parser;

/// Parsed `aiperf profile` flags.
#[derive(Debug, Parser)]
#[command(name = "profile", disable_help_subcommand = true)]
pub struct ProfileFlags {
    /// Model name(s) to benchmark (`--model-names` / `--model` / `-m`).
    #[arg(
        long = "model-names",
        visible_alias = "model",
        short = 'm',
        value_delimiter = ',',
        num_args = 1..,
    )]
    pub model_names: Vec<String>,

    /// Base URL(s) of the server(s) (`--url` / `-u`).
    #[arg(long = "url", short = 'u', num_args = 1..)]
    pub urls: Vec<String>,

    /// Endpoint dialect id (`--endpoint-type`).
    #[arg(long = "endpoint-type")]
    pub endpoint_type: Option<String>,

    /// Enable streaming responses (`--streaming`).
    #[arg(long = "streaming", default_value_t = false)]
    pub streaming: bool,

    /// Concurrent requests to maintain (`--concurrency`); comma-list ⇒ sweep.
    #[arg(long = "concurrency")]
    pub concurrency: Option<String>,

    /// Target request rate, requests/second (`--request-rate`); comma ⇒ sweep.
    #[arg(long = "request-rate")]
    pub request_rate: Option<String>,

    /// Maximum requests to send (`--request-count`); comma-list ⇒ sweep.
    #[arg(long = "request-count")]
    pub request_count: Option<String>,

    /// Artifact output directory (`--artifact-dir`).
    #[arg(long = "artifact-dir")]
    pub artifact_dir: Option<PathBuf>,

    /// YAML configuration file (`--config` / `-f`).
    #[arg(long = "config", short = 'f')]
    pub config_file: Option<PathBuf>,
}

impl ProfileFlags {
    /// Parse `profile` flags from an argv slice (program name already stripped,
    /// i.e. the tokens after `aiperf profile`).
    pub fn parse_from_args(args: &[String]) -> Result<Self, clap::Error> {
        // clap expects argv[0] to be the binary name.
        let mut argv = vec!["profile".to_string()];
        argv.extend_from_slice(args);
        Self::try_parse_from(argv)
    }
}
