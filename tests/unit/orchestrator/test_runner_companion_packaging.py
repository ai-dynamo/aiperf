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
    assert '--profile "${AIPERF_RUNNER_PROFILE}"' in dockerfile
    assert "AIPERF_RUNNER_PROFILE=offline" in dockerfile
    assert "runner-build.json" in nightly
    assert "refuses to synthesize Cargo metadata or substitute" in nightly
    assert (
        "aiperf/runner-inputs/${RESOLVED_SHA}/linux-${{ matrix.arch }}/${PROFILE}"
        in nightly
    )
    assert "for PROFILE in online offline" in nightly
    assert 'verify_runner_companion.py --profile "${PROFILE}"' in nightly
    assert "--build-arg AIPERF_RUNNER_PROFILE=offline" in nightly


def test_platform_ci_builds_and_executes_both_native_profiles() -> None:
    workflow = (_ROOT / ".github/workflows/native-runner.yml").read_text()

    assert "prod-aiperf-builder-amd-v1" in workflow
    assert "prod-aiperf-builder-arm-v1" in workflow
    assert "cargo test --locked --workspace" in workflow
    assert "--test stdio_e2e" in workflow
    assert "--test offline_stdio" in workflow
    assert "--test offline_scheduled_stdio" in workflow
    assert "cargo build --locked --release -p aiperf-runner" in workflow
    assert "--features dynamo-offline" in workflow
    assert "tools/runner_release_input.py create" in workflow
    assert "tools/runner_release_input.py verify" in workflow
    assert "dynamo-aiperf-native=${DYNAMO_REVISION}" in workflow
    assert "aiperf/runner-inputs/${GITHUB_SHA}/linux-${{ matrix.arch }}" in workflow
