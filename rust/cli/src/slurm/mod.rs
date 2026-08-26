// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SLURM cellular role dispatch: `aiperf slurm run`.
//!
//! Under an `srun`/`sbatch` allocation every task runs the identical
//! `aiperf slurm run --config <file>` command; this module reads the `SLURM_*`
//! environment once and dispatches each task to its role, mirroring the k8s
//! [`cellular_role`](crate::cellular_role) controller/cell split but without an
//! operator:
//!
//! - rank 0 (`SLURM_PROCID == 0`) becomes the cellular **controller**. It binds the
//!   velo transport on all interfaces at `AIPERF_CONTROLLER_PORT`, expects the
//!   `srun`-launched cell tasks to register (the [`SlurmLauncher`] "expects, doesn't
//!   spawn"), and projects the mounted Config v2 with `--cells <cell_count>` so the
//!   controller path engages.
//! - ranks `1..ntasks` become **cells**. Each dials the controller at the
//!   allocation-derived coordinate (rank-0 node host + port) and fetches its sliced
//!   envelope over velo — it never reads `--config`.
//!
//! The `AIPERF_CELL_*` environment every downstream stage reads is set here from the
//! allocation, exactly where the k8s operator would inject it.
//!
//! Private role material is per-rank, but `srun` exports one environment to every
//! task: rank 0 reads `AIPERF_CONTROLLER_BOOTSTRAP_FILE`, while each cell resolves
//! `<AIPERF_ROLE_BOOTSTRAP_DIR>/cell-<id>.bin` from its own rank. An
//! operator-provisioned `AIPERF_ROLE_BOOTSTRAP_FILE` still takes precedence.
//!
//! `aiperf slurm generate` ([`generate`]) emits the sbatch script that submits
//! such an allocation.
//!
//! [`SlurmLauncher`]: aiperf_runtime::engine::cell_launcher::SlurmLauncher

pub mod generate;

use aiperf_runtime::cellular::partition::{CELL_COUNT_ENV, CELL_ID_ENV};
use aiperf_runtime::engine::cell_launcher::CELL_LAUNCHER_ENV;
use aiperf_runtime::engine::cellular_cell::CELL_CONTROLLER_ADDR_ENV;
use aiperf_runtime::engine::slurm_topology::{
    CONTROLLER_PORT_ENV, SlurmTopology, controller_port_from_env,
};
use anyhow::{Context, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The `slurm run` subcommand token: the per-task rank dispatch.
///
/// `slurm generate` is the other native subcommand ([`generate`]).
pub const RUN_SUBCOMMAND: &str = "run";

const CONTROLLER_BOOTSTRAP_FILE_ENV: &str = "AIPERF_CONTROLLER_BOOTSTRAP_FILE";
const ROLE_BOOTSTRAP_FILE_ENV: &str = "AIPERF_ROLE_BOOTSTRAP_FILE";
const ROLE_BOOTSTRAP_DIR_ENV: &str = "AIPERF_ROLE_BOOTSTRAP_DIR";

/// Require the deployment-owned bootstrap mount for this SLURM role.
///
/// This only establishes that the launcher mounted a path. The runtime opens it
/// with no-follow/regular-file checks and validates its signed-role material
/// before binding a controller or cell listener.
fn require_bootstrap_mount(env: &str, role: &str) -> Result<()> {
    let path = std::env::var_os(env)
        .filter(|path| !path.is_empty())
        .with_context(|| format!("SLURM {role} has no deployment-provisioned {env}"))?;
    validate_bootstrap_mount(std::path::Path::new(&path), role)
}

/// Resolve this cell task's bootstrap bundle.
///
/// `srun` exports one environment to every task, so a per-rank path cannot come from
/// the generated script. An operator-provisioned `AIPERF_ROLE_BOOTSTRAP_FILE` still
/// wins when present; otherwise the bundle is `<AIPERF_ROLE_BOOTSTRAP_DIR>/cell-<id>.bin`,
/// with `cell_id` derived from `SLURM_PROCID` by the same rank dispatch that selects
/// this branch.
fn resolve_role_bootstrap_path(
    file: Option<OsString>,
    directory: Option<OsString>,
    cell_id: u32,
) -> Result<PathBuf> {
    if let Some(file) = file.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(file));
    }
    let directory = directory.filter(|value| !value.is_empty()).with_context(|| {
        format!(
            "SLURM cell {cell_id} has no deployment-provisioned {ROLE_BOOTSTRAP_FILE_ENV} \
             or {ROLE_BOOTSTRAP_DIR_ENV} (generate the job script with `aiperf slurm generate`)"
        )
    })?;
    Ok(Path::new(&directory).join(format!("cell-{cell_id}.bin")))
}

fn validate_bootstrap_mount(path: &std::path::Path, role: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("SLURM {role} bootstrap mount is unavailable"))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "SLURM {role} bootstrap mount is not a regular file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            metadata.permissions().mode() & 0o7777 == 0o600,
            "SLURM {role} bootstrap mount is not private"
        );
    }
    Ok(())
}

/// Dispatch `aiperf slurm run` for one SLURM task.
///
/// `args` are the arguments after `run` (typically `--config <file>` plus any
/// profile overrides). Resolves the task's role from the `SLURM_*` environment and
/// either runs the controller (rank 0) or this cell (rank > 0). A cell call diverges
/// into the execute path and does not return.
pub fn run(args: &[String]) -> Result<i32> {
    let topology = SlurmTopology::from_env().context(
        "resolving the SLURM allocation topology (run `aiperf slurm run` under srun/sbatch, \
         e.g. via the script from `aiperf slurm generate`)",
    )?;
    let port = controller_port_from_env();
    let coordinate = topology.controller_coordinate(port);

    // The cellular stages downstream read these exactly as the k8s operator sets
    // them. Set on EVERY task (controller included) so a controller's
    // artifact-authority derivation sees the same coordinate the cells dial.
    // SAFETY: `slurm run` is the process entry for a SLURM task and runs on the sole
    // thread before any Tokio runtime or benchmark thread is constructed.
    unsafe {
        std::env::set_var(CELL_LAUNCHER_ENV, "slurm");
        std::env::set_var(CONTROLLER_PORT_ENV, port.to_string());
        std::env::set_var(CELL_COUNT_ENV, topology.cell_count().to_string());
        std::env::set_var(CELL_CONTROLLER_ADDR_ENV, &coordinate);
    }

    if topology.is_controller() {
        require_bootstrap_mount(CONTROLLER_BOOTSTRAP_FILE_ENV, "controller")?;
        run_controller(args, &topology, &coordinate)
    } else {
        let cell_id = topology
            .cell_id()
            .context("a non-controller SLURM rank must map to a cell id")?;
        let bootstrap = resolve_role_bootstrap_path(
            std::env::var_os(ROLE_BOOTSTRAP_FILE_ENV),
            std::env::var_os(ROLE_BOOTSTRAP_DIR_ENV),
            cell_id,
        )?;
        validate_bootstrap_mount(&bootstrap, "cell")?;
        // The runtime's one-shot role acquisition reads only the file variable, so a
        // directory-derived path is published here before any cellular stage runs.
        // SAFETY: same single-threaded process-entry invariant as above.
        unsafe {
            std::env::set_var(ROLE_BOOTSTRAP_FILE_ENV, &bootstrap);
        }
        run_cell(&topology, &coordinate, cell_id)
    }
}

/// Run the rank-0 controller task: project the mounted config with the
/// allocation's cell count and drive the cellular controller.
fn run_controller(args: &[String], topology: &SlurmTopology, coordinate: &str) -> Result<i32> {
    let cell_count = topology.cell_count();
    // A 2-task allocation (`cell_count == 1`) is a valid degenerate cellular run: rank
    // 0 controls and rank 1 is the sole cell. The cross-host launcher promotion (keyed
    // off `AIPERF_CELL_LAUNCHER=slurm`, set in `run`) engages the controller for a
    // single cell too, so no minimum-of-two guard is needed. `cell_count` is always
    // `>= 1` here because `SlurmTopology` rejects `ntasks < 2` (a 1-task allocation)
    // up front with its own actionable guidance.
    debug_assert!(cell_count >= 1, "topology guarantees at least one cell");
    tracing::info!(
        cell_count,
        %coordinate,
        controller_host = topology.controller_host(),
        "SLURM controller (rank 0): expecting srun-launched cell tasks"
    );
    // Append `--cells <cell_count>` so the profile projection marks the run cellular
    // regardless of the mounted config's `runtime.cells`. A caller-supplied `--cells`
    // earlier in `args` is overridden by this later occurrence (last wins).
    let mut controller_args = args.to_vec();
    controller_args.push("--cells".to_owned());
    controller_args.push(cell_count.to_string());
    crate::profile::run(&controller_args)
}

/// Run a cell task: inject this cell's identity and fetch/execute its slice over
/// velo. Diverges into the execute path.
fn run_cell(topology: &SlurmTopology, coordinate: &str, cell_id: u32) -> Result<i32> {
    // SAFETY: same single-threaded process-entry invariant as `run`.
    unsafe {
        std::env::set_var(CELL_ID_ENV, cell_id.to_string());
    }
    tracing::info!(
        cell_id,
        cell_count = topology.cell_count(),
        %coordinate,
        "SLURM cell (rank {}): fetching envelope over velo",
        topology.proc_id()
    );
    // Diverges (`-> !`): drives the cell to its terminal and exits the process.
    crate::execute_mode::dispatch(&[crate::execute_mode::CELL_FLAG.to_string()])
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn bootstrap_mount_requires_exact_private_permissions() {
        let file = tempfile::NamedTempFile::new().expect("temporary bootstrap file");

        for mode in [0o400, 0o700] {
            std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(mode))
                .expect("permissions");
            assert!(
                validate_bootstrap_mount(file.path(), "test role").is_err(),
                "mode {mode:o} must be refused"
            );
        }
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(0o600))
            .expect("permissions");
        assert!(validate_bootstrap_mount(file.path(), "test role").is_ok());
    }

    #[test]
    fn slurm_run_derives_its_role_file_from_procid() {
        let directory = std::ffi::OsString::from("/run/aiperf/bootstrap");
        let topology = SlurmTopology::new(2, 4, "node0").expect("rank-2 topology");
        let cell_id = topology.cell_id().expect("rank 2 is a cell");

        assert_eq!(
            resolve_role_bootstrap_path(None, Some(directory.clone()), cell_id)
                .expect("directory-derived path"),
            std::path::Path::new("/run/aiperf/bootstrap/cell-1.bin")
        );
        assert_eq!(
            resolve_role_bootstrap_path(
                Some(std::ffi::OsString::from("/mnt/operator/role.bin")),
                Some(directory),
                cell_id,
            )
            .expect("operator-mounted path"),
            std::path::Path::new("/mnt/operator/role.bin"),
            "an explicit AIPERF_ROLE_BOOTSTRAP_FILE outranks the directory"
        );
        assert!(resolve_role_bootstrap_path(None, None, cell_id).is_err());
    }
}
