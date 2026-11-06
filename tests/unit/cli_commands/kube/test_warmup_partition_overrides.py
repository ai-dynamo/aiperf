# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes CLI overrides must precede final partition validation."""

import os
import subprocess
import sys
from pathlib import Path

import orjson
import pytest
import yaml
from pytest import param

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("command", ["sweep", "profile", "generate"])
@pytest.mark.parametrize(
    ("entries", "flags", "reserved", "warmup", "error"),
    [
        param(12, ["--warmup-request-count", "4"], 4, 4, None, id="cli-adds-warmup"),
        param(12, ["--warmup-dataset-entries", "0"], 0, None, None, id="cli-zero"),
        param(4, ["--warmup-dataset-entries", "0"], 0, None, None, id="zero-at-boundary"),
        param(12, [], None, None, "requires a warmup phase", id="missing-warmup"),
        param(4, ["--warmup-request-count", "4"], None, None, "must be smaller", id="insufficient-pool"),
    ],
)  # fmt: skip
def test_kube_partition_overrides_validate_final_cli_workload(
    tmp_path: Path,
    command: str,
    entries: int,
    flags: list[str],
    reserved: int | None,
    warmup: int | None,
    error: str | None,
) -> None:
    """Run the public offline CLI through parsing, merging and emitted workload."""
    benchmark = {
        "models": ["test"],
        "endpoint": {"urls": ["http://localhost:8000"]},
        "datasets": [
            {
                "name": "main",
                "type": "synthetic",
                "entries": entries,
                "warmup_entries": 4,
                "prompts": {"isl": 64, "osl": 32},
            }
        ],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 8,
                "concurrency": 1,
            }
        ],
    }
    # Sweep exercises the Config-v2 loader. Both other commands must take the
    # AIPerfJob CR input branch, which uses a different preliminary loader.
    document = {"benchmark": benchmark}
    if command == "sweep":
        document["sweep"] = {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1]},
        }
    else:
        document = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {"name": "partition-override"},
            "spec": {"image": "aiperf:test", "benchmark": benchmark},
        }
    source = tmp_path / "partition.yaml"
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    render_flags = ["--operator"] if command == "generate" else ["--dry-run"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aiperf",
            "kube",
            command,
            "-f",
            str(source),
            "--image",
            "aiperf:test",
            "--namespace",
            "bench-ns",
            *render_flags,
            *flags,
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if error is not None:
        assert result.returncode != 0
        assert error in " ".join((result.stderr + result.stdout).split())
        return
    assert result.returncode == 0, result.stderr or result.stdout
    output = (
        yaml.safe_load(result.stdout)
        if command == "generate"
        else orjson.loads(result.stdout)
    )
    assert output["kind"] == ("AIPerfSweep" if command == "sweep" else "AIPerfJob")
    emitted = output["spec"]["benchmark"]
    assert emitted["datasets"][0]["warmupEntries"] == reserved
    warmup_phases = [p for p in emitted["phases"] if p.get("excludeFromResults")]
    assert [p["requests"] for p in warmup_phases] == (
        [] if warmup is None else [warmup]
    )
