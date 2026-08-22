# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "aiperf-k8s-operator"
FIXTURE = ROOT / "contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json"


def test_wheel_declares_and_carries_its_runtime_contract(tmp_path: Path) -> None:
    """Removing the checkout must not remove schema validation from the wheel."""
    wheelhouse = tmp_path / "wheelhouse"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
            str(PROJECT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheelhouse.glob("*.whl"))
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert (
            "aiperf_k8s_operator/contracts/v1/controller-envelope.schema.json" in names
        )
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        assert "Requires-Dist: jsonschema" in metadata
        archive.extractall(installed)

    command = (
        "import json,sys; "
        "from aiperf_k8s_operator.contract import validate_envelope; "
        "payload=json.load(open(sys.argv[1], encoding='utf-8')); "
        "assert validate_envelope(payload).run_id == 'run-1'"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    subprocess.run(
        [sys.executable, "-c", command, str(FIXTURE)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
