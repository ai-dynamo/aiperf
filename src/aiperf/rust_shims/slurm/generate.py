# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SLURM generate command: emit an sbatch job script for a native cellular run.

Under SLURM, ``srun`` launches ``cells + 1`` identical tasks, each running
``aiperf slurm run --config <path>``. Rank ``SLURM_PROCID == 0`` becomes the
cellular controller; ranks ``1..N-1`` become load-generating cells. There is no
Kubernetes operator involved: the controller advertises itself from the
allocation's rank-0 node hostname plus the velo bootstrap port
(``AIPERF_CONTROLLER_PORT``), and tasks select the SLURM launcher via
``AIPERF_CELL_LAUNCHER=slurm``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="generate")


def build_sbatch_script(
    *,
    config: Path,
    cells: int,
    job_name: str = "aiperf",
    partition: str | None = None,
    account: str | None = None,
    time: str = "01:00:00",
    nodes: int | None = None,
    ntasks_per_node: int = 1,
    gpus_per_node: int | None = None,
    controller_port: int = 9500,
) -> str:
    """Build the sbatch job-script text for a native cellular AIPerf run.

    ``cells`` is the number of load-generating cells; the controller is an
    additional task, so total ``ntasks`` is ``cells + 1``. ``nodes`` defaults to
    ``cells + 1`` (one task per node) when not overridden.

    Raises:
        ValueError: if ``cells < 1`` or ``config`` does not exist.
    """
    if cells < 1:
        raise ValueError(f"--cells must be >= 1 (got {cells})")
    if not config.exists():
        raise ValueError(f"config file does not exist: {config}")

    abs_config = config.resolve()
    ntasks = cells + 1
    resolved_nodes = nodes if nodes is not None else cells + 1

    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --nodes={resolved_nodes}")
    lines.append(f"#SBATCH --ntasks={ntasks}")
    lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    lines.append(f"#SBATCH --time={time}")
    if partition is not None:
        lines.append(f"#SBATCH --partition={partition}")
    if account is not None:
        lines.append(f"#SBATCH --account={account}")
    if gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={gpus_per_node}")

    lines.append("")
    lines.append("export AIPERF_CELL_LAUNCHER=slurm")
    lines.append(f"export AIPERF_CONTROLLER_PORT={controller_port}")
    lines.append("")
    lines.append(f"srun aiperf slurm run --config {abs_config}")
    lines.append("")

    return "\n".join(lines)


@app.default
def generate(
    *,
    config: Annotated[
        Path,
        Parameter(name="--config", help="Path to the AIPerf Config v2 YAML file."),
    ],
    cells: Annotated[
        int,
        Parameter(
            name="--cells",
            help="Number of load-generating cells (controller is an extra task).",
        ),
    ],
    job_name: Annotated[
        str, Parameter(name="--job-name", help="SLURM job name.")
    ] = "aiperf",
    partition: Annotated[
        str | None, Parameter(name="--partition", help="SLURM partition.")
    ] = None,
    account: Annotated[
        str | None, Parameter(name="--account", help="SLURM account.")
    ] = None,
    time: Annotated[
        str, Parameter(name="--time", help="Job time limit (HH:MM:SS).")
    ] = "01:00:00",
    nodes: Annotated[
        int | None,
        Parameter(name="--nodes", help="Node count (default: cells + 1)."),
    ] = None,
    ntasks_per_node: Annotated[
        int, Parameter(name="--ntasks-per-node", help="Tasks per node.")
    ] = 1,
    gpus_per_node: Annotated[
        int | None,
        Parameter(name="--gpus-per-node", help="GPUs per node (optional)."),
    ] = None,
    controller_port: Annotated[
        int,
        Parameter(
            name="--controller-port",
            help="Velo bootstrap port for the controller (AIPERF_CONTROLLER_PORT).",
        ),
    ] = 9500,
    output: Annotated[
        Path | None,
        Parameter(
            name="--output",
            help="Write the script to this file instead of stdout.",
        ),
    ] = None,
) -> None:
    """Generate an sbatch job script for a native cellular AIPerf benchmark.

    The emitted script requests ``cells + 1`` tasks (one controller + ``cells``
    load-generating cells) and launches them with a single ``srun`` invocation
    of ``aiperf slurm run``.

    Examples:
        # Print a script for 4 load-generating cells to stdout
        aiperf slurm generate --config benchmark.yaml --cells 4

        # Write to a file, then submit it
        aiperf slurm generate --config benchmark.yaml --cells 8 --output job.sbatch
        sbatch job.sbatch
    """
    from aiperf import cli_utils

    try:
        script = build_sbatch_script(
            config=config,
            cells=cells,
            job_name=job_name,
            partition=partition,
            account=account,
            time=time,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            gpus_per_node=gpus_per_node,
            controller_port=controller_port,
        )
    except ValueError as exc:
        cli_utils.raise_startup_error_and_exit(
            str(exc),
            title="Error Generating SLURM Job Script",
        )

    if output is not None:
        output.write_text(script)
    else:
        sys.stdout.write(script)


def main(arguments: list[str] | None = None) -> int:
    """Run the SLURM-generation shim with explicitly supplied arguments."""
    argv = sys.argv[1:] if arguments is None else list(arguments)
    result = app(argv)
    return 0 if result is None else result
