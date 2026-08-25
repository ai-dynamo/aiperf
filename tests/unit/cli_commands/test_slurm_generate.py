# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `aiperf slurm generate` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.cli import app as cli_app
from aiperf.rust_shims.__main__ import main as shim_main
from aiperf.rust_shims.slurm.generate import build_sbatch_script, generate


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write a throwaway config YAML and return its path."""
    cfg = tmp_path / "benchmark.yaml"
    cfg.write_text("benchmark: {}\n")
    return cfg


def test_shebang_first_line(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=4)
    assert script.startswith("#!/bin/bash\n")


def test_ntasks_is_cells_plus_one(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=4)
    assert "#SBATCH --ntasks=5" in script


def test_nodes_defaults_to_cells_plus_one(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=4)
    assert "#SBATCH --nodes=5" in script


def test_nodes_override(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=4, nodes=2)
    assert "#SBATCH --nodes=2" in script
    # ntasks still cells + 1
    assert "#SBATCH --ntasks=5" in script


def test_srun_line_absolute_config(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=1)
    abs_cfg = config_path.resolve()
    assert f"srun aiperf slurm run --config {abs_cfg}" in script
    assert abs_cfg.is_absolute()


def test_selection_env_exports(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=1, controller_port=9700)
    assert "export AIPERF_CELL_LAUNCHER=slurm" in script
    assert "export AIPERF_CONTROLLER_PORT=9700" in script


def test_controller_port_default(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=1)
    assert "export AIPERF_CONTROLLER_PORT=9500" in script


def test_optional_directives_absent_by_default(config_path: Path) -> None:
    script = build_sbatch_script(config=config_path, cells=1)
    assert "--partition" not in script
    assert "--account" not in script
    assert "--gpus-per-node" not in script


def test_optional_directives_present_when_passed(config_path: Path) -> None:
    script = build_sbatch_script(
        config=config_path,
        cells=1,
        partition="batch",
        account="proj123",
        gpus_per_node=8,
    )
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --account=proj123" in script
    assert "#SBATCH --gpus-per-node=8" in script


def test_job_name_and_time(config_path: Path) -> None:
    script = build_sbatch_script(
        config=config_path, cells=1, job_name="myrun", time="02:30:00"
    )
    assert "#SBATCH --job-name=myrun" in script
    assert "#SBATCH --time=02:30:00" in script


def test_cells_less_than_one_errors(config_path: Path) -> None:
    with pytest.raises(ValueError, match="cells"):
        build_sbatch_script(config=config_path, cells=0)


def test_missing_config_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ValueError, match="does not exist"):
        build_sbatch_script(config=missing, cells=1)


def test_generate_writes_output_file(config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "job.sbatch"
    generate(config=config_path, cells=3, output=out)
    text = out.read_text()
    assert text.startswith("#!/bin/bash\n")
    assert "#SBATCH --ntasks=4" in text


def test_generate_stdout(config_path: Path, capsys: pytest.CaptureFixture) -> None:
    generate(config=config_path, cells=2)
    captured = capsys.readouterr()
    assert captured.out.startswith("#!/bin/bash\n")
    assert "#SBATCH --ntasks=3" in captured.out


def test_shim_launcher_generates_script(config_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert shim_main(["slurm-generate", "--config", str(config_path), "--cells", "2"]) == 0

    captured = capsys.readouterr()
    assert captured.out == build_sbatch_script(config=config_path, cells=2)


def test_python_slurm_command_routes_to_shim(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "job.sbatch"

    assert (
        cli_app(
            [
                "slurm",
                "generate",
                "--config",
                str(config_path),
                "--cells",
                "2",
                "--output",
                str(output),
            ],
            result_action="return_value",
        )
        is None
    )
    assert output.read_text() == build_sbatch_script(config=config_path, cells=2)


def test_generate_missing_config_exits(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit):
        generate(config=missing, cells=1)
