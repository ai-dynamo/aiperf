<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLURM-native cellular execution

## Purpose

Run a cellular AIPerf benchmark natively under a SLURM allocation — the SLURM
analog of the Kubernetes path in [cellular.md](cellular.md), without the k8s
operator. Under `srun`/`sbatch` the allocation's tasks become the controller (rank
0) and the cells (other ranks); they discover each other from the allocation itself
(the rank-0 node's hostname plus the velo bootstrap port) instead of
operator-injected DNS. This is the connect-by-endpoint model the velo transport was
built for, sourced from `SLURM_*` environment rather than a Kubernetes JobSet.

## Built

### One command, rank-dispatched

Every task of the allocation runs the identical command:

```bash
srun aiperf slurm run --config benchmark.yaml
```

`aiperf slurm run` (native, `rust/cli/src/slurm.rs`) resolves the task's role from
the `SLURM_*` environment once and dispatches:

- **rank 0** (`SLURM_PROCID == 0`) becomes the **controller**. It projects the
  mounted Config v2 with `--cells <cell_count>` appended (so the controller path
  engages regardless of the config's `runtime.cells`), binds the velo transport on
  all interfaces at `AIPERF_CONTROLLER_PORT`, and — via the `SlurmLauncher` — expects
  the `srun`-launched cell tasks to register rather than spawning them.
- **ranks `1..ntasks`** become **cells**. Each dials the controller at the
  allocation-derived coordinate and fetches its sliced envelope over velo; it never
  reads `--config`.

The command sets the `AIPERF_CELL_*` environment every downstream stage reads
(`AIPERF_CELL_LAUNCHER=slurm`, `AIPERF_CELL_COUNT`, `AIPERF_CELL_ID`,
`AIPERF_CELL_CONTROLLER_ADDR`, `AIPERF_CONTROLLER_PORT`) — exactly where the k8s
operator would inject it. Any `slurm` subcommand other than `run` (i.e. `generate`)
delegates to the Python CLI.

### Session-security inputs

SLURM supplies placement, a public controller coordinate, and role-specific
private material; the coordinate is not identity. Rank 0 requires
`AIPERF_CONTROLLER_BOOTSTRAP_FILE`, and every cell rank requires its own
`AIPERF_ROLE_BOOTSTRAP_FILE`. Each mount contains fixed binary role material,
must be a regular no-follow file with exact `0600` permissions, and is acquired
once into the process-owned security context. Key bytes are never carried in
JSON, environment values, argv, or the controller coordinate.

The rank-0 hostname and Velo `_hello` are routing inputs only. Application
admission begins with the signed registration and controller reply attestation,
then each cell-originated Velo preflight, heartbeat, phase signal,
dataset/phaser subscription, partition, and artifact-control request is a
payload-bound, sequenced authenticated frame under the registered session.
Controller-to-cell dataset/phaser pushes are not per-push authenticated frames;
adding that direction is a separate follow-up. Cross-host HTTP artifact bytes
instead use the transactionally registered exact bearer over pinned TLS; those
uploads are not `AuthenticatedFrame`-sequenced. The transport handshake itself is not
authenticated by AIPerf and may insert a Velo peer before application rejection;
no confidentiality claim follows from this admission layer. A hierarchy request
is rejected before acquisition, bind, or launch because SLURM does not yet
provision controller-authorized material for aggregator tree edges.

### Topology mapping

`rust/runtime/src/engine/slurm_topology.rs` holds the pure `SLURM_*` → topology
functions (no velo dependency, exhaustively unit-tested):

- `SlurmTopology::from_env` reads `SLURM_PROCID` and `SLURM_NTASKS`; rank 0 is the
  controller, ranks `1..ntasks` map to dense cell ids `0..cell_count` with
  `cell_count = ntasks - 1` (tiling exactly, as `ModuloCellPartition` requires). A
  2-task allocation (`cell_count == 1`) is a valid single-cell run — rank 0 controls,
  rank 1 is the sole cell; only a 1-task allocation is rejected (with actionable
  guidance to increase `--ntasks`). The controller engages the cellular path for a
  single cell because `AIPERF_CELL_LAUNCHER=slurm` marks a cross-host launcher
  (`is_cross_host_launcher`), whose promotion gate fires at `cells >= 1` — a separate
  cell task is already dialing — whereas the same-host default treats `cells == 1` as
  a plain single-process run.
- The controller coordinate is `tcp://<rank0-host>:<port>`. The rank-0 host is the
  first host of the highest-precedence non-empty nodelist — `SLURM_STEP_NODELIST`,
  then `SLURM_NODELIST`, then `SLURM_JOB_NODELIST` — overridable with
  `AIPERF_SLURM_CONTROLLER_HOST` (honored first). The step-scoped lists outrank the
  job-wide list so a nested `srun` **step** scoped to a node subset (e.g. an
  orchestrator such as srt-slurm launching aiperf against part of the allocation)
  resolves rank 0 within the step's nodes, not the job's first host; a plain
  `srun`/`sbatch` allocation sets all three identically, so the order is a no-op there.
  SLURM's default block task distribution places task 0 on the first node of the
  chosen list. `expand_nodelist` handles SLURM's compressed hostlist syntax —
  `node[01-04]`, `node[01-02,05]`, top-level comma lists, and bracket suffixes —
  preserving the lower-bound zero-pad width.

### Cross-host placement

`run_cellular` (`rust/runtime/src/engine/cellular_controller.rs`) recognizes
`AIPERF_CELL_LAUNCHER=slurm` as a **cross-host** deployment alongside k8s: cell tasks
live on separate allocation nodes, so per-record artifacts and `file` datasets ship
over the controller's HTTP+zstd plane, the controller binds velo on all interfaces at
`AIPERF_CONTROLLER_PORT`, and the `SlurmLauncher` "expects, doesn't spawn" (`srun`
already launched the cell tasks; a failed task fails the step, and the controller's
registration/collect timeout is the backstop). The controller coordinate is read
from `AIPERF_CELL_CONTROLLER_ADDR` (SLURM sets it from the rank-0 host; k8s leaves it
operator-injected). All of cellular.md's fidelity guards, budget slicing, and merge
apply unchanged — SLURM changes only *where cells come from and how they find the
controller*, not the partition/merge math.

### Job-script generation

`aiperf slurm generate` (Python, `src/aiperf/cli_commands/slurm/`) emits an sbatch
script mirroring `aiperf kube generate`'s ergonomics. `--cells N` requests
`#SBATCH --ntasks=N+1` (one controller task plus N cell tasks), exports
`AIPERF_CELL_LAUNCHER=slurm` and `AIPERF_CONTROLLER_PORT`, and ends with the single
`srun aiperf slurm run --config <abs-path>` line. Optional `--partition`,
`--account`, `--time`, `--nodes`, `--ntasks-per-node`, `--gpus-per-node`, and
`--output` map to the corresponding `#SBATCH` directives.

## Future requirements

- **Aggregator tier**: Hierarchical aggregation is unavailable in every launcher.
  A fanout request is refused before cellular startup; SLURM runs the flat star
  when no hierarchy is requested.
- **Live in-sandbox multi-cell proof**: the topology, launcher selection, coordinate
  derivation, velo discovery/envelope-fetch, AND the controller's merge are proven
  end-to-end by the simulation in `rust/e2e-tests/scripts/slurm_sim.sh` — a 3-task loopback
  allocation (rank-0 controller + two cells) runs to completion (`controller exit=0`),
  and the controller merges both cells' partitions into one report (`request_count`
  total 40 over the two cells, ISL/OSL count 40, merged `cellular-heartbeat.json`
  issued/completed 40). Two bugs the sim surfaced were fixed to reach green: the
  worker-cap ignoring the `cells` factor (empty per-thread conversation source ->
  `request-rate conversation dataset cannot be empty`), and the controller's HTTP
  artifact-shipping gate not mirroring the cell's loopback carve-out (the controller
  waited forever in `wait_for_cells` for uploads a loopback-coordinate cell never
  sends). A real cross-node SLURM run keeps HTTP shipping on (routable coordinate);
  the loopback sim (and any same-host allocation) co-locates on both sides. The
  single-cell (2-task) path has its own loopback sim, `rust/e2e-tests/scripts/slurm_sim_single_cell.sh`
  (rank-0 controller + one cell): the controller promotes to the cellular path at
  `cell_count == 1` via the cross-host launcher gate, the sole cell registers and
  ships its slice, and the run completes with `controller exit=0` and a merged report.

## Source anchors

- `rust/runtime/src/engine/slurm_topology.rs` — the pure `SLURM_*` → topology
  mapping and nodelist expansion, with the unit tests.
- `rust/runtime/src/engine/cell_launcher.rs` — `SlurmLauncher` and
  `select_launcher`'s `slurm` branch.
- `rust/runtime/src/engine/cellular_controller.rs` — the cross-host placement
  (`cross_host = is_k8s || is_slurm`) and `controller_bind_and_endpoint`.
- `rust/cli/src/slurm.rs` and `rust/cli/src/dispatch.rs` — the `aiperf slurm run`
  rank dispatch and routing.
- `src/aiperf/cli_commands/slurm/` — the `aiperf slurm generate` sbatch generator
  and `tests/unit/cli_commands/test_slurm_generate.py`.
- `rust/e2e-tests/scripts/slurm_sim.sh` — the loopback multi-cell (3-task) SLURM-allocation
  simulation, and `rust/e2e-tests/scripts/slurm_sim_single_cell.sh` — the single-cell
  (2-task) variant.
- `rust/runtime/src/engine/cell_launcher.rs` — `is_cross_host_launcher`, the gate the
  runner uses to promote the controller for a single cell under `slurm`/`k8s`.
