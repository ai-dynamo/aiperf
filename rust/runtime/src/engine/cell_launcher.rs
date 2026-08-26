// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a cellular run's cells are started.
//!
//! The velo transport is uniform across deployments; only *how the cell processes
//! come to exist* differs, so that is the one seam here:
//!
//! - [`LocalLauncher`] (default, dev/test) spawns `aiperf --cell`
//!   subprocesses on the same host. Each child learns only its `cell_id`, the
//!   `cell_count`, and the controller's bootstrap coordinate — all via env — and
//!   fetches its full execute envelope over velo (no stdin pipe).
//! - [`K8sLauncher`] does **not** spawn: the operator/JobSet already created the
//!   cell pods. It only reports how many cells to expect; the pods find the
//!   controller from the same env the operator injects.
//!
//! A cell that never comes up is caught by [`CellHandle::wait_failure`] (a local
//! child exit) or, for k8s where there is no child to watch, by the controller's
//! registration timeout.

use std::collections::BTreeMap;
#[cfg(unix)]
use std::io::Write;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream;

#[cfg(not(unix))]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
use tokio::process::Child;

use crate::cellular::partition::{CELL_COUNT_ENV, CELL_ID_ENV};

use crate::engine::cellular_bootstrap::{
    CELL_SECURITY_FD, CELL_SECURITY_FD_ENV, CellularRole, LocalRoleProvisioner,
};
use crate::engine::cellular_cell::{
    CELL_ARTIFACT_ADDR_ENV, CELL_CONTROLLER_ADDR_ENV, CELL_PHASE_ORDINAL_BASES_ENV,
};

/// Env var selecting the launcher: `local` (default), `k8s`, or `slurm`.
pub const CELL_LAUNCHER_ENV: &str = "AIPERF_CELL_LAUNCHER";

/// Everything a launcher needs to start (or expect) a run's cells. The controller
/// builds this after it has bound its velo transport and published its bootstrap
/// coordinate.
pub struct CellLaunchContext {
    /// Number of cells the run is partitioned across.
    pub cell_count: u32,
    /// The controller's bootstrap coordinate cells fetch its `PeerInfo` from
    /// (`file:PATH` locally, `tcp://HOST:PORT` in k8s), injected as
    /// [`CELL_CONTROLLER_ADDR_ENV`].
    pub controller_coordinate: String,
    /// Each phase's global dispatch-ordinal base, injected as
    /// [`CELL_PHASE_ORDINAL_BASES_ENV`] so a cell's issuer stamps
    /// single-cell-equivalent absolute slots.
    pub phase_ordinal_bases: BTreeMap<String, u64>,
    /// The controller's artifact upload `host:port`, injected as
    /// [`CELL_ARTIFACT_ADDR_ENV`] so a local-launched cell POSTs its per-record
    /// artifact files there. `None` when HTTP artifact shipping is off or on the
    /// same-host path, which concatenates local writes instead of shipping.
    pub artifact_authority: Option<String>,
    /// One-shot local role material. Cross-host launchers receive `None`.
    pub(crate) local_roles: Option<LocalRoleProvisioner>,
}

/// A started cell the controller watches for hard failure. For a local subprocess
/// this wraps the child; for a k8s pod there is nothing to wait on (pod liveness
/// is the operator's concern; the controller uses a registration timeout).
pub struct CellHandle {
    child: Option<Child>,
    cell_id: u32,
}

impl CellHandle {
    /// Await this cell's failure, returning a diagnostic if it exits non-zero (or
    /// cannot be waited on). For a k8s cell (no child) this never resolves.
    pub async fn wait_failure(&mut self) -> String {
        match self.child.as_mut() {
            Some(child) => match child.wait().await {
                Ok(status) if status.success() => {
                    // A cell that exits cleanly is not a failure; park so the
                    // controller's select! keeps waiting on the transport instead.
                    std::future::pending::<()>().await;
                    unreachable!()
                }
                Ok(status) => format!("cell {} exited with {status}", self.cell_id),
                Err(error) => format!("cell {} could not be waited on: {error}", self.cell_id),
            },
            None => {
                std::future::pending::<()>().await;
                unreachable!()
            }
        }
    }
}

/// Starts (or expects) a run's cells; the transport is always velo.
pub trait CellLauncher {
    /// Start the cells and return handles the controller watches for hard failure.
    fn launch(&self, ctx: CellLaunchContext) -> Result<Vec<CellHandle>>;
}

/// Spawns `aiperf --cell` subprocesses on this host.
pub struct LocalLauncher;

impl LocalLauncher {
    /// Build (but do not spawn) the `Command` for one cell — its own function so
    /// the env wiring is unit-testable without spawning a process.
    pub fn cell_command(&self, ctx: &CellLaunchContext, cell_id: u32) -> tokio::process::Command {
        use std::process::Stdio;
        // `current_exe` can fail; a failure surfaces at launch, so build eagerly.
        let exe = std::env::current_exe().unwrap_or_else(|_| "aiperf runner".into());
        let mut command = tokio::process::Command::new(exe);
        command
            .arg("--cell")
            .env(CELL_ID_ENV, cell_id.to_string())
            .env(CELL_COUNT_ENV, ctx.cell_count.to_string())
            .env(CELL_CONTROLLER_ADDR_ENV, &ctx.controller_coordinate)
            // A cell fetches its spec over velo and ships records over velo; keep
            // stderr for diagnostics and drop stdout (its would-be terminal
            // envelope is unused by the controller).
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            // On any controller abort the runtime drops the watcher tasks owning
            // these children; kill_on_drop then SIGKILLs each cell so a failed run
            // never leaves cells generating load.
            .kill_on_drop(true);
        // kill_on_drop only fires on the controller's graceful Drop. A SIGKILLed
        // or OOM-killed controller skips Drop entirely, so also arm a kernel-backed
        // parent-death signal: the cell is SIGKILLed the moment the controller dies.
        set_parent_death_signal(&mut command);
        if let Ok(bases) = serde_json::to_string(&ctx.phase_ordinal_bases) {
            command.env(CELL_PHASE_ORDINAL_BASES_ENV, bases);
        }
        if let Some(authority) = &ctx.artifact_authority {
            command.env(CELL_ARTIFACT_ADDR_ENV, authority);
        }
        command
    }
}

#[cfg(unix)]
fn inherit_security_fd(command: &mut tokio::process::Command, source_fd: i32) {
    // SAFETY: this hook runs after fork and before exec and invokes only async-signal-safe
    // descriptor operations. The parent retains ownership and closes its copy after spawn.
    unsafe {
        command.pre_exec(move || {
            if source_fd == CELL_SECURITY_FD {
                if libc::fcntl(CELL_SECURITY_FD, libc::F_SETFD, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            } else {
                if libc::dup2(source_fd, CELL_SECURITY_FD) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                libc::close(source_fd);
            }
            Ok(())
        });
    }
    command.env(CELL_SECURITY_FD_ENV, CELL_SECURITY_FD.to_string());
}

/// Arm a kernel-backed parent-death signal so a cell is SIGKILLed the instant the
/// controller dies, even on a hard controller kill (SIGKILL/OOM) that skips the
/// `kill_on_drop` Drop path. Mirrors the mock-server balancer's guard. Linux-only.
#[cfg(target_os = "linux")]
fn set_parent_death_signal(command: &mut tokio::process::Command) {
    // SAFETY: `pre_exec` runs in the forked child before `exec`; `prctl` and
    // `getppid`/`raise` are async-signal-safe and touch no shared state.
    unsafe {
        command.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() == 1 {
                libc::raise(libc::SIGKILL);
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn set_parent_death_signal(_command: &mut tokio::process::Command) {}

impl CellLauncher for LocalLauncher {
    fn launch(&self, mut ctx: CellLaunchContext) -> Result<Vec<CellHandle>> {
        let mut local_roles = ctx
            .local_roles
            .take()
            .context("local launcher has no role provisioner")?;
        let mut handles = Vec::with_capacity(ctx.cell_count as usize);
        for cell_id in 0..ctx.cell_count {
            let material = local_roles.take(CellularRole::Cell(cell_id))?;
            #[cfg(unix)]
            let (child_read, mut parent_write) =
                UnixStream::pair().context("creating cell security pipe")?;
            let mut command = self.cell_command(&ctx, cell_id);
            #[cfg(unix)]
            inherit_security_fd(&mut command, child_read.as_raw_fd());
            #[cfg(not(unix))]
            bail!("local cellular security delivery requires unix inherited descriptors");
            let child = command
                .spawn()
                .with_context(|| format!("spawning cell {cell_id}"))?;
            #[cfg(unix)]
            {
                drop(child_read);
                parent_write
                    .write_all(&material)
                    .with_context(|| format!("delivering security material to cell {cell_id}"))?;
                drop(parent_write);
            }
            handles.push(CellHandle {
                child: Some(child),
                cell_id,
            });
        }
        Ok(handles)
    }
}

/// Expects cells that a Kubernetes JobSet/operator already created. Spawns nothing;
/// the pods discover the controller from the operator-injected env.
pub struct K8sLauncher;

impl CellLauncher for K8sLauncher {
    fn launch(&self, ctx: CellLaunchContext) -> Result<Vec<CellHandle>> {
        ensure!(
            ctx.local_roles.is_none(),
            "k8s launcher received local role material"
        );
        tracing::info!(
            cell_count = ctx.cell_count,
            "cellular k8s launcher: expecting cell pods to register (no local spawn)"
        );
        Ok((0..ctx.cell_count)
            .map(|cell_id| CellHandle {
                child: None,
                cell_id,
            })
            .collect())
    }
}

/// Expects cells that `srun` already launched as sibling tasks of this allocation.
///
/// Like [`K8sLauncher`], it spawns nothing: under SLURM every task of the run is
/// launched by `srun` at once, so the rank-0 (controller) task only reports how many
/// cell tasks to expect. Each cell task discovers the controller from the
/// allocation-derived coordinate (rank-0 node host + velo port), injected into its
/// environment by the `aiperf slurm run` rank dispatch — not by an operator. Cell
/// liveness is SLURM's concern (a failed task fails the step); the controller uses
/// its registration/collect timeout as the backstop, exactly as for k8s.
pub struct SlurmLauncher;

impl CellLauncher for SlurmLauncher {
    fn launch(&self, ctx: CellLaunchContext) -> Result<Vec<CellHandle>> {
        ensure!(
            ctx.local_roles.is_none(),
            "slurm launcher received local role material"
        );
        tracing::info!(
            cell_count = ctx.cell_count,
            "cellular slurm launcher: expecting srun-launched cell tasks to register (no local spawn)"
        );
        Ok((0..ctx.cell_count)
            .map(|cell_id| CellHandle {
                child: None,
                cell_id,
            })
            .collect())
    }
}

/// Select the launcher from [`CELL_LAUNCHER_ENV`] (`local` default, `k8s`, `slurm`).
pub fn select_launcher() -> Box<dyn CellLauncher> {
    match std::env::var(CELL_LAUNCHER_ENV).as_deref() {
        Ok("k8s") => Box::new(K8sLauncher),
        Ok("slurm") => Box::new(SlurmLauncher),
        _ => Box::new(LocalLauncher),
    }
}

/// Whether [`CELL_LAUNCHER_ENV`] selects a **cross-host** launcher (`k8s` or
/// `slurm`), where the cell processes already exist on separate nodes and dial the
/// controller over velo. This governs whether the runner promotes itself to the
/// cellular controller: a cross-host launcher must engage the controller even for a
/// single cell (`cells == 1`, e.g. a 2-task SLURM allocation), because a separate
/// cell task is already waiting to register, whereas the same-host default treats
/// `cells == 1` as a plain single-process run with no cell to coordinate.
pub fn is_cross_host_launcher() -> bool {
    matches!(
        std::env::var(CELL_LAUNCHER_ENV).as_deref(),
        Ok("k8s") | Ok("slurm")
    )
}

/// The number of dispatch-stream positions in `[0, total)` that cell `k` owns under
/// round-robin ownership (`position % cell_count == cell_id`) — `ceil((total-k)/C)`.
/// A phase's per-cell slice is the difference of this over the phase's `[base,
/// base+len)` window; over a single phase (`base=0`) it is just each cell's share,
/// summing to `total`.
///
/// Lives here (not the velo-gated controller) so the thread-per-core sharded
/// scheduled runtime ([`crate::engine::sharded_scheduled`]) reuses the identical round-robin
/// share **without** the `cellular` feature — the two-level `(cell × thread)` partition
/// tiles only if both levels use this exact `ceil((total-k)/C)` share.
pub(crate) fn owned_positions(total: u64, cell_id: u32, cell_count: u32) -> u64 {
    let count = cell_count as u64;
    let k = cell_id as u64;
    if k >= total {
        return 0;
    }
    (total - k).div_ceil(count)
}

/// Reads `cfg.runtime.cells` from a v2 envelope, defaulting to 1 (single process).
/// Lives here (not the velo-gated controller) so the runner's mode dispatch can
/// read it without the `cellular` feature.
pub fn cell_count_from_envelope(envelope: &serde_json::Value) -> u32 {
    envelope
        .pointer("/run/cfg/runtime/cells")
        .and_then(serde_json::Value::as_u64)
        .map(|cells| cells.clamp(1, 1024) as u32)
        .unwrap_or(1)
}

/// Wrap resolved `--execute` bytes in the envelope the cellular helpers require,
/// alongside the cell count read from it.
///
/// Mode dispatch resolves authoring input once before it selects process defaults
/// or cellular execution. This helper deliberately accepts only those resolved
/// bytes so cellular inspection cannot repeat adaptation.
///
/// Returns `None` when the resolved bytes do not decode to a JSON object. Mode
/// dispatch reports that typed protocol failure through its ordinary runner path.
pub fn envelope_from_resolved_run_bytes(input: &[u8]) -> Option<(serde_json::Value, u32)> {
    let run: serde_json::Value = serde_json::from_slice(input).ok()?;
    let envelope = serde_json::json!({ "run": run });
    let cells = cell_count_from_envelope(&envelope);
    Some((envelope, cells))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn local_child_receives_secret_only_on_inherited_pipe() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let output = directory.path().join("role-material.bin");
        let secret = b"opaque-role-material";
        let (child_read, mut parent_write) = UnixStream::pair().expect("security pipe");
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("cat <&3 >\"$1\"")
            .arg("aiperf-security-test")
            .arg(&output);
        inherit_security_fd(&mut command, child_read.as_raw_fd());

        let env_values = command
            .as_std()
            .get_envs()
            .filter_map(|(_, value)| value.and_then(std::ffi::OsStr::to_str))
            .collect::<Vec<_>>();
        assert_eq!(
            env_values
                .iter()
                .find(|value| **value == CELL_SECURITY_FD.to_string()),
            Some(&"3")
        );
        assert!(!env_values.iter().any(|value| value.as_bytes() == secret));
        assert!(
            !command
                .as_std()
                .get_args()
                .any(|value| value.as_encoded_bytes() == secret)
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let status = runtime.block_on(async move {
            let mut child = command.spawn().expect("spawn child");
            drop(child_read);
            parent_write.write_all(secret).expect("write material");
            drop(parent_write);
            child.wait().await.expect("wait child")
        });
        assert!(status.success());
        assert_eq!(std::fs::read(output).expect("read material"), secret);
    }

    fn context() -> CellLaunchContext {
        let mut bases = BTreeMap::new();
        bases.insert("profiling".to_owned(), 0);
        CellLaunchContext {
            cell_count: 2,
            controller_coordinate: "file:/tmp/controller-peer.rmp".to_owned(),
            phase_ordinal_bases: bases,
            artifact_authority: Some("controller.local:9600".to_owned()),
            local_roles: None,
        }
    }

    #[test]
    fn local_launcher_sets_cell_env() {
        let cmd = LocalLauncher.cell_command(&context(), 1);
        let envs: std::collections::HashMap<String, String> = cmd
            .as_std()
            .get_envs()
            .filter_map(|(key, value)| {
                Some((key.to_str()?.to_owned(), value?.to_str()?.to_owned()))
            })
            .collect();
        assert_eq!(envs.get(CELL_ID_ENV).map(String::as_str), Some("1"));
        assert_eq!(envs.get(CELL_COUNT_ENV).map(String::as_str), Some("2"));
        assert_eq!(
            envs.get(CELL_CONTROLLER_ADDR_ENV).map(String::as_str),
            Some("file:/tmp/controller-peer.rmp")
        );
        assert!(envs.contains_key(CELL_PHASE_ORDINAL_BASES_ENV));
        assert_eq!(
            envs.get(CELL_ARTIFACT_ADDR_ENV).map(String::as_str),
            Some("controller.local:9600")
        );
        assert!(!envs.contains_key("AIPERF_CELL_SHIP_ADDR"));
    }

    #[test]
    fn k8s_launcher_spawns_nothing_but_expects_all_cells() {
        let handles = K8sLauncher.launch(context()).expect("k8s launch");
        assert_eq!(handles.len(), 2);
        assert!(handles.iter().all(|handle| handle.child.is_none()));
    }

    #[test]
    fn slurm_launcher_spawns_nothing_but_expects_all_cells() {
        let handles = SlurmLauncher.launch(context()).expect("slurm launch");
        assert_eq!(handles.len(), 2);
        assert!(handles.iter().all(|handle| handle.child.is_none()));
    }

    #[test]
    fn owned_positions_sum_to_total_and_tile() {
        for total in [1_u64, 7, 100, 500, 501] {
            for count in 1..=8u32 {
                let sum: u64 = (0..count).map(|k| owned_positions(total, k, count)).sum();
                assert_eq!(sum, total, "total {total} count {count}");
            }
        }
    }

    #[test]
    fn cross_host_launcher_detection() {
        // SAFETY: single-threaded test body; the variable is restored before return.
        unsafe {
            let prior = std::env::var(CELL_LAUNCHER_ENV).ok();
            std::env::set_var(CELL_LAUNCHER_ENV, "slurm");
            assert!(is_cross_host_launcher());
            std::env::set_var(CELL_LAUNCHER_ENV, "k8s");
            assert!(is_cross_host_launcher());
            std::env::set_var(CELL_LAUNCHER_ENV, "local");
            assert!(!is_cross_host_launcher());
            std::env::remove_var(CELL_LAUNCHER_ENV);
            assert!(!is_cross_host_launcher());
            match prior {
                Some(value) => std::env::set_var(CELL_LAUNCHER_ENV, value),
                None => std::env::remove_var(CELL_LAUNCHER_ENV),
            }
        }
    }

    #[test]
    fn cell_count_reads_runtime_cells() {
        let envelope = serde_json::json!({"run": {"cfg": {"runtime": {"cells": 4}}}});
        assert_eq!(cell_count_from_envelope(&envelope), 4);
        let single = serde_json::json!({"run": {"cfg": {}}});
        assert_eq!(cell_count_from_envelope(&single), 1);
    }
}
