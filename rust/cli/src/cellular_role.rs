// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes cellular roles: `aiperf controller`, `cell`, and `aggregator`.
//!
//! The operator's JobSet invokes these subcommands in its controller, cell, and
//! aggregator pods. Cross-pod execution uses the velo transport.
//!
//! - `controller` projects the mounted Config v2. `runtime.cells` and the
//!   `AIPERF_CELL_*` environment select the pre-created pod topology.
//! - `cell` fetches its sliced envelope over velo from the controller using the
//!   `AIPERF_CELL_*` environment; it does not read the mounted `--config`.
//! - `aggregator` projects Config v2 and sends the execute envelope to
//!   `--aggregator` over stdin.

use std::path::PathBuf;

/// Run the Kubernetes controller pod.
pub fn run_controller(args: &[String]) -> anyhow::Result<i32> {
    let reporter = crate::k8s::CrReporter::from_env();

    let exit = crate::profile::run(args)?;

    // Publish the ready marker before the completion annotation so the
    // annotation-triggered fetch cannot race ahead of completed artifacts.
    if reporter.active() {
        if exit == 0 {
            if let Some(artifact_dir) = resolve_artifact_dir(args) {
                report_completion(&reporter, &artifact_dir);
            } else {
                tracing::warn!(
                    "could not resolve the artifact directory; skipping k8s completion reporting"
                );
                reporter.signal_complete();
            }
        } else {
            reporter.signal_complete();
        }
    }
    Ok(exit)
}

/// Report the final snapshot and completion without masking the run's exit code.
fn report_completion(reporter: &crate::k8s::CrReporter, artifact_dir: &std::path::Path) {
    let native_v2 = artifact_dir.join("native-v2.json");
    match std::fs::read(&native_v2) {
        Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(snapshot) => reporter.patch_status(&crate::k8s::snapshot_body(snapshot)),
            Err(e) => {
                tracing::warn!(error = %e, "native-v2.json is not valid JSON; skipping snapshot")
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, path = %native_v2.display(), "native-v2.json unreadable; skipping snapshot")
        }
    }
    if let Err(e) = crate::k8s::write_ready_marker(artifact_dir, false) {
        tracing::warn!(error = %e, "failed to write the results-ready marker");
    }
    reporter.signal_complete();
}

/// Resolve the artifact directory from `--artifact-dir` or the mounted config.
fn resolve_artifact_dir(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(rest) = a.strip_prefix("--artifact-dir=") {
            return Some(PathBuf::from(rest));
        }
        if a == "--artifact-dir" {
            return it.next().map(PathBuf::from);
        }
    }
    let config = config_flag(args)?;
    let value = crate::yaml::read_env_substituted(&config).ok()?;
    for path in [
        value.pointer("/benchmark/artifacts/dir"),
        value.pointer("/artifacts/dir"),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(dir) = path.as_str() {
            return Some(PathBuf::from(dir));
        }
    }
    None
}

/// Run one cellular cell pod.
pub fn run_cell(_args: &[String]) -> anyhow::Result<i32> {
    crate::execute_mode::dispatch(&[crate::execute_mode::CELL_FLAG.to_string()])
}

/// Run a tier-T2 merge aggregator pod.
pub fn run_aggregator(args: &[String]) -> anyhow::Result<i32> {
    let config_path = config_flag(args).ok_or_else(|| {
        anyhow::anyhow!("`aiperf aggregator` requires `--config <file>` (the mounted Config v2)")
    })?;
    let envelope = project_execute_envelope(&config_path, None)?;

    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("failed to resolve the aiperf executable: {e}"))?;
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(exe)
        .arg(crate::execute_mode::AGGREGATOR_FLAG)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn `aiperf --aggregator`: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(&envelope)
        .map_err(|e| anyhow::anyhow!("failed to pipe the aggregator envelope: {e}"))?;
    let status = child
        .wait()
        .map_err(|e| anyhow::anyhow!("failed to wait for `aiperf --aggregator`: {e}"))?;
    Ok(status.code().unwrap_or(1))
}

fn config_flag(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(rest) = a.strip_prefix("--config=") {
            return Some(PathBuf::from(rest));
        }
        if a == "--config" {
            return it.next().map(PathBuf::from);
        }
    }
    None
}

/// Project Config v2 to a serialized protocol-v2 execute envelope.
fn project_execute_envelope(
    config_path: &std::path::Path,
    artifact_dir: Option<PathBuf>,
) -> anyhow::Result<Vec<u8>> {
    let base = crate::yaml::read_env_substituted(config_path)?;
    let expanded = crate::expand::render_with_context(base)?;
    // Kubernetes roles project the mounted config without profile flag overrides.
    let run = crate::yaml::resolve_expanded_value(expanded, artifact_dir, None)?;
    // Cellular merge helpers require the `{"run": …}` wrapper for
    // `/run/cfg/...` JSON pointers.
    serde_json::to_vec(&serde_json::json!({ "run": run }))
        .map_err(|e| anyhow::anyhow!("failed to serialize the cellular execute envelope: {e}"))
}

#[cfg(test)]
mod tests {
    use super::config_flag;
    use std::path::PathBuf;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn config_flag_reads_separate_value() {
        assert_eq!(
            config_flag(&args(&["--config", "/etc/aiperf/config.yaml"])),
            Some(PathBuf::from("/etc/aiperf/config.yaml"))
        );
    }

    #[test]
    fn config_flag_reads_equals_form() {
        assert_eq!(
            config_flag(&args(&["--config=/etc/aiperf/c.yaml"])),
            Some(PathBuf::from("/etc/aiperf/c.yaml"))
        );
    }

    #[test]
    fn config_flag_absent_is_none() {
        assert_eq!(config_flag(&args(&["--ui", "simple"])), None);
        assert_eq!(config_flag(&args(&[])), None);
    }

    #[test]
    fn config_flag_dangling_is_none() {
        assert_eq!(config_flag(&args(&["--config"])), None);
    }
}
