# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static release-contract checks for the native runner companion."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _toml(relative: str) -> dict:
    return tomllib.loads((_ROOT / relative).read_text())


def test_companion_and_frontend_versions_move_together() -> None:
    frontend = _toml("pyproject.toml")["project"]
    companion = _toml("packaging/aiperf-runner/pyproject.toml")["project"]

    assert companion["name"] == "aiperf-runner"
    assert companion["version"] == frontend["version"]
    assert companion["requires-python"] == frontend["requires-python"]


def test_companion_has_no_python_runtime_entrypoint() -> None:
    project = _toml("packaging/aiperf-runner/pyproject.toml")
    wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "scripts" not in project["project"]
    assert "entry-points" not in project["project"]
    assert wheel["bypass-selection"] is True
    assert wheel["hooks"]["custom"]["path"] == "hatch_build.py"


def test_container_and_release_require_verified_companion_inputs() -> None:
    dockerfile = (_ROOT / "Dockerfile").read_text()
    nightly = (_ROOT / ".github/workflows/nightly.yml").read_text()

    assert "/dist/aiperf_runner-*.whl" in dockerfile
    assert "verify_runner_companion.py" in dockerfile
    assert "runner-build.json" in nightly
    assert "refuses to synthesize Cargo metadata" in nightly
    assert "aiperf/runner-inputs/${RESOLVED_SHA}/linux-${{ matrix.arch }}" in nightly
