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
//! [`SlurmLauncher`]: aiperf_runtime::engine::cell_launcher::SlurmLauncher

use aiperf_runtime::cellular::partition::{CELL_COUNT_ENV, CELL_ID_ENV};
use aiperf_runtime::engine::cell_launcher::CELL_LAUNCHER_ENV;
use aiperf_runtime::engine::cellular_cell::CELL_CONTROLLER_ADDR_ENV;
use aiperf_runtime::engine::slurm_topology::{
    CONTROLLER_PORT_ENV, SlurmTopology, controller_port_from_env,
};
use anyhow::{Context, Result};

/// The `slurm run` subcommand token intercepted natively; every other `slurm`
/// subcommand (`generate`) is delegated to the Python CLI.
pub const RUN_SUBCOMMAND: &str = "run";

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
        run_controller(args, &topology, &coordinate)
    } else {
        run_cell(&topology, &coordinate)
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
fn run_cell(topology: &SlurmTopology, coordinate: &str) -> Result<i32> {
    let cell_id = topology
        .cell_id()
        .expect("a non-controller task always has a cell id");
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
