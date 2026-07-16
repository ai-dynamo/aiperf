// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Kubernetes cellular roles: `aiperf controller` / `cell` / `aggregator`.
//!
//! These subcommands are the entry points the operator's JobSet sets as each
//! pod's `command` (`command=["aiperf"]`, `args=["controller"|"cell"|"aggregator", …]`).
//! They replace the Python `aiperf.cli_commands._cellular_role` adapter: the
//! native binary already owns the full cross-pod cellular execution (velo
//! transport, budget partition, cross-host artifact upload, skew-free timing
//! origin — see `aiperf_runtime::engine::cellular_controller`/`cellular_cell`),
//! so these roles reuse that machinery directly instead of the Python wrapper
//! projecting an envelope and re-spawning the binary.
//!
//! - `controller` projects the mounted Config v2 and runs the cellular
//!   controller. Execution is identical to `aiperf profile --config` — the
//!   config's `runtime.cells` and the operator's `AIPERF_CELL_*` env (with
//!   `AIPERF_CELL_LAUNCHER=k8s`) select the pre-created-pod topology. The
//!   in-cluster Kubernetes CR reporting (progress/snapshot/completion) is layered
//!   on top; off-cluster (no `AIPERF_JOB_ID`) it is a no-op.
//! - `cell` fetches its sliced envelope over velo from the controller using the
//!   `AIPERF_CELL_*` env; the mounted `--config` is not consulted on this path.
//! - `aggregator` (tier-T2) projects the mounted Config v2 to the execute
//!   envelope and enters the native `--aggregator` merge mode with it on stdin.

use std::path::PathBuf;

/// `aiperf controller` — the Kubernetes controller pod frontend.
///
/// Delegates execution to the native profile path (`crate::profile::run`), which
/// projects the mounted Config v2, spawns `aiperf --execute`, and self-promotes
/// to the cellular controller when `runtime.cells > 1`. The operator sets
/// `AIPERF_CELL_LAUNCHER=k8s` so the launcher expects the operator-created cell
/// pods rather than spawning children.
pub fn run_controller(args: &[String]) -> anyhow::Result<i32> {
    let reporter = crate::k8s::CrReporter::from_env();

    // Run the benchmark (native cellular controller when `runtime.cells > 1`).
    let exit = crate::profile::run(args)?;

    // On-cluster, after a successful run, push the final metric snapshot, write
    // the results-ready marker, and set the completion annotation the operator
    // watches — in that order, so the operator's fetch (triggered by the
    // annotation) never races ahead of the marker. Off-cluster this is a no-op.
    // (Live in-run progress/snapshot streaming is a follow-up: SP-A slice 2b.)
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
            // A failed run still signals completion so the operator stops waiting.
            reporter.signal_complete();
        }
    }
    Ok(exit)
}

/// Push the final `native-v2.json` snapshot into `.status.snapshot`, write the
/// results-ready marker, then set the completion annotation. Every step is
/// best-effort (the reporter swallows API errors) so reporting never masks the
/// run's own exit code.
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

/// Resolve the run's artifact directory: the `--artifact-dir` flag if present,
/// else the mounted config's `benchmark.artifacts.dir` (`artifacts.dir`). Used
/// to locate `native-v2.json` and place the results-ready marker.
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

/// `aiperf cell` — one cellular cell pod. Enters the native `--cell` mode, which
/// fetches this cell's sliced envelope over velo from the controller and runs it
/// through the cell-aware single-process path. The mounted `--config` is not
/// needed here (the controller ships the resolved envelope over velo).
pub fn run_cell(_args: &[String]) -> anyhow::Result<i32> {
    // `dispatch` is `-> !` (it always terminates the process); the never type
    // coerces to the declared return type.
    crate::execute_mode::dispatch(&[crate::execute_mode::CELL_FLAG.to_string()])
}

/// `aiperf aggregator` — a tier-T2 merge aggregator pod. Projects the mounted
/// Config v2 to the execute envelope and re-execs `aiperf --aggregator` with it
/// on stdin (the aggregator reads the envelope only for the merge `MetricsConfig`).
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

/// Scan `args` for `--config <path>` (or `--config=<path>`) and return the path.
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

/// Project a mounted Config v2 file to the serialized protocol-v2 execute
/// envelope, reusing the same native YAML → `BenchmarkRun` resolution the
/// `aiperf profile --config` path uses (env substitution + Jinja expansion +
/// the typed resolver). `artifact_dir` overrides the config's when `Some`.
fn project_execute_envelope(
    config_path: &std::path::Path,
    artifact_dir: Option<PathBuf>,
) -> anyhow::Result<Vec<u8>> {
    let base = crate::yaml::read_env_substituted(config_path)?;
    let expanded = crate::expand::render_with_context(base)?;
    let run = crate::yaml::resolve_expanded_value(expanded, artifact_dir)?;
    // The tier-T2 aggregator (`aiperf --aggregator`) reads this envelope only for
    // its merge `MetricsConfig`, and the engine's cellular merge helpers address it
    // through `/run/cfg/...` JSON pointers. Unlike the bare-run stdin execute wire,
    // this cellular-engine envelope keeps the `{"run": …}` wrapper so those pointer
    // reads resolve unchanged.
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
        // `--config` with no following token yields None rather than panicking.
        assert_eq!(config_flag(&args(&["--config"])), None);
    }
}
