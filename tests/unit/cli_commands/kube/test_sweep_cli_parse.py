# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess CLI parsing regressions for ``aiperf kube sweep`` placement flags."""

from __future__ import annotations

import subprocess
from pathlib import Path

import orjson

_REPO_ROOT = Path(__file__).resolve().parents[4]


_MIN_SWEEP_CONFIG = """\
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [1]
"""


def _run_sweep_dry_run(tmp_path: Path, *placement_args: str) -> dict:
    """Run the public CLI and decode the dry-run AIPerfSweep JSON."""
    config_file = tmp_path / "sweep-config.yaml"
    config_file.write_text(_MIN_SWEEP_CONFIG)

    result = subprocess.run(
        [
            "uv",
            "run",
            "aiperf",
            "kube",
            "sweep",
            "-f",
            str(config_file),
            "--image",
            "aiperf:test",
            *placement_args,
            "--dry-run",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return orjson.loads(result.stdout)


def test_sweep_cli_dry_run_accepts_json_node_selector_and_tolerations(
    tmp_path: Path,
) -> None:
    """JSON object/array placement flags survive public CLI parsing."""
    cr = _run_sweep_dry_run(
        tmp_path,
        "--namespace",
        "tenant-a",
        "--node-selector",
        '{"kubernetes.io/arch":"amd64","nodeGroup":"customer-cpu"}',
        "--tolerations",
        '[{"key":"dedicated","operator":"Equal","value":"user-workload","effect":"NoSchedule"}]',
    )

    pod_template = cr["spec"]["podTemplate"]
    assert pod_template["nodeSelector"] == {
        "kubernetes.io/arch": "amd64",
        "nodeGroup": "customer-cpu",
    }
    assert pod_template["tolerations"] == [
        {
            "key": "dedicated",
            "operator": "Equal",
            "value": "user-workload",
            "effect": "NoSchedule",
        }
    ]
    assert cr["metadata"]["namespace"] == "tenant-a"


def test_sweep_cli_dry_run_accepts_repeated_key_value_node_selectors(
    tmp_path: Path,
) -> None:
    """Repeated key=value flags merge into the nodeSelector map."""
    cr = _run_sweep_dry_run(
        tmp_path,
        "--node-selector",
        "kubernetes.io/arch=amd64",
        "--node-selector",
        "nodeGroup=customer-cpu",
    )

    assert cr["spec"]["podTemplate"]["nodeSelector"] == {
        "kubernetes.io/arch": "amd64",
        "nodeGroup": "customer-cpu",
    }
