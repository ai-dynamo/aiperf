// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible build measurement with explicit target and artifact paths.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::{self, Write},
    process::Command,
    time::Instant,
};

use aiperf_bench_tools::plugin_stats::{PairedSample, Variant};
use serde::Serialize;

#[derive(Serialize)]
struct BuildMeasurement {
    sample: PairedSample,
    command: Vec<String>,
    target_dir: String,
    cargo_incremental: String,
    inherited_build_environment: BTreeMap<String, Option<String>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let separator = arguments
        .iter()
        .position(|argument| argument == "--")
        .ok_or("missing -- command separator")?;
    let options = parse_options(&arguments[..separator])?;
    let command = &arguments[separator + 1..];
    let executable = command.first().ok_or("missing measured build command")?;
    let scenario = required_nonempty(&options, "--scenario")?;
    let pair_id = required_nonempty(&options, "--pair-id")?;
    let variant = match required(&options, "--variant")? {
        "static" => Variant::Static,
        "dynamic" => Variant::Dynamic,
        value => return Err(format!("invalid variant {value}").into()),
    };
    let target_dir = required_nonempty(&options, "--target-dir")?;
    let artifact_path = required_nonempty(&options, "--artifact")?;
    let commit = required(&options, "--commit")?;
    if !is_lower_hex(commit, 40) {
        return Err("commit is not canonical 40-character lowercase hex".into());
    }
    let experiment_identity_digest = required(&options, "--experiment-identity-digest")?;
    if !experiment_identity_digest
        .strip_prefix("blake3:")
        .is_some_and(|digest| is_lower_hex(digest, 64))
    {
        return Err("experiment identity digest is not canonical BLAKE3".into());
    }
    if command
        .iter()
        .any(|argument| argument == "--target-dir" || argument.starts_with("--target-dir="))
    {
        return Err("measured command cannot override the explicit target path".into());
    }
    let mut child = Command::new(executable);
    child.args(&command[1..]);
    child.env("CARGO_INCREMENTAL", "1");
    child.env("CARGO_TARGET_DIR", target_dir);
    let started = Instant::now();
    let status = child.status()?;
    let elapsed = started.elapsed();
    if !status.success() {
        return Err(format!("measured build command failed with {status}").into());
    }

    let artifact = fs::read(artifact_path)?;
    let value = elapsed.as_secs_f64() * 1_000_000_000.0;
    if !value.is_finite() || value <= 0.0 {
        return Err("measured build duration is not finite and positive".into());
    }
    let inherited_build_environment = [
        "RUSTC_WRAPPER",
        "SCCACHE_DIR",
        "SCCACHE_CACHE_SIZE",
        "SCCACHE_IDLE_TIMEOUT",
        "CARGO_BUILD_JOBS",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), std::env::var(name).ok()))
    .collect();
    let measurement = BuildMeasurement {
        sample: PairedSample {
            scenario: scenario.to_owned(),
            pair_id: pair_id.to_owned(),
            variant,
            metric: "build_nanoseconds".to_owned(),
            value,
            unit: "nanoseconds".to_owned(),
            commit: commit.to_owned(),
            artifact_digest: format!("blake3:{}", blake3::hash(&artifact).to_hex()),
            experiment_identity_digest: experiment_identity_digest.to_owned(),
        },
        command: command.to_vec(),
        target_dir: target_dir.to_owned(),
        cargo_incremental: "1".to_owned(),
        inherited_build_environment,
    };
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &measurement)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn parse_options(arguments: &[String]) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    if !arguments.len().is_multiple_of(2) {
        return Err("build-bench options must be name/value pairs".into());
    }
    let mut options = BTreeMap::new();
    const ALLOWED: &[&str] = &[
        "--scenario",
        "--pair-id",
        "--variant",
        "--target-dir",
        "--artifact",
        "--commit",
        "--experiment-identity-digest",
    ];
    for pair in arguments.as_chunks::<2>().0 {
        if !ALLOWED.contains(&pair[0].as_str())
            || options.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(format!("invalid or duplicate option {}", pair[0]).into());
        }
    }
    Ok(options)
}

fn required_nonempty<'a>(
    options: &'a BTreeMap<String, String>,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    let value = required(options, name)?;
    if value.is_empty() {
        return Err(format!("{name} must be non-empty").into());
    }
    Ok(value)
}

fn is_lower_hex(value: &str, expected_length: usize) -> bool {
    value.len() == expected_length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn required<'a>(
    options: &'a BTreeMap<String, String>,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    options
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required option {name}").into())
}
