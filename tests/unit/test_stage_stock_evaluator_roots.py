# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact stock-evaluator deployment-root staging."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aiperf.orchestrator import runner_installation
from tools.generate_stock_evaluator_manifest import (
    DeploymentSourceFile,
    DeploymentSourceRoot,
)
from tools.stage_stock_evaluator_roots import (
    PROVIDER_ROOT_SPECS,
    PROVIDER_ROOTS_REGISTRY,
    PROVIDER_ROOTS_SCHEMA,
    _stage_verified_roots,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_staging_and_product_discovery_share_one_strict_registry_contract() -> None:
    assert PROVIDER_ROOT_SPECS == runner_installation._PROVIDER_ROOT_SPECS
    assert PROVIDER_ROOTS_SCHEMA == runner_installation._PROVIDER_ROOTS_SCHEMA
    assert PROVIDER_ROOTS_REGISTRY == runner_installation._PROVIDER_ROOTS_REGISTRY


def _inventory(root: Path) -> tuple[DeploymentSourceRoot, ...]:
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (root_id, kind, relative) in enumerate(PROVIDER_ROOT_SPECS):
        source = root / f"source-{index}.bin"
        content = f"verified-{root_id}\n".encode()
        source.write_bytes(content)
        executable = index == 0
        source.chmod(0o755 if executable else 0o644)
        result.append(
            DeploymentSourceRoot(
                id=root_id,
                kind=kind,
                relative_path=relative,
                files=(
                    DeploymentSourceFile(
                        relative_path=f"nested/file-{index}.bin",
                        source=source,
                        artifact_content_sha256=hashlib.sha256(content).hexdigest(),
                        executable=executable,
                    ),
                ),
            )
        )
    return tuple(result)


def test_staging_writes_one_atomic_registry_consumed_by_product_discovery(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "aiperf-runner.evaluator-roots"

    _stage_verified_roots(output, inventory)

    expected_roots = tuple(
        (output / root.relative_path).resolve() for root in inventory
    )
    assert runner_installation._provider_roots_from_registry(output) == expected_roots
    for index, root in enumerate(inventory):
        staged = output / root.relative_path / f"nested/file-{index}.bin"
        assert staged.read_bytes() == root.files[0].source.read_bytes()
        assert bool(staged.stat().st_mode & 0o111) == root.files[0].executable
    with pytest.raises(FileExistsError):
        _stage_verified_roots(output, inventory)


def test_documented_module_invocation_works_from_source_checkout() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "tools.stage_stock_evaluator_roots", "--help"],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--nemo-root" in completed.stdout
    assert "--openbench-root" in completed.stdout
    assert "--output" in completed.stdout


def test_staging_digest_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    inventory = list(_inventory(tmp_path))
    source = inventory[1].files[0]
    inventory[1] = DeploymentSourceRoot(
        id=inventory[1].id,
        kind=inventory[1].kind,
        relative_path=inventory[1].relative_path,
        files=(
            DeploymentSourceFile(
                relative_path=source.relative_path,
                source=source.source,
                artifact_content_sha256="0" * 64,
                executable=source.executable,
            ),
        ),
    )
    output = tmp_path / "failed-output"

    with pytest.raises(RuntimeError, match="digest drifted"):
        _stage_verified_roots(output, tuple(inventory))

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-output.staging-*"))


def test_staging_rejects_source_symlinks_and_relative_path_escape(
    tmp_path: Path,
) -> None:
    inventory = list(_inventory(tmp_path))
    source = inventory[0].files[0]
    link = tmp_path / "source-link"
    link.symlink_to(source.source)
    inventory[0] = DeploymentSourceRoot(
        id=inventory[0].id,
        kind=inventory[0].kind,
        relative_path=inventory[0].relative_path,
        files=(
            DeploymentSourceFile(
                relative_path=source.relative_path,
                source=link,
                artifact_content_sha256=source.artifact_content_sha256,
                executable=source.executable,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="cannot open verified evaluator source"):
        _stage_verified_roots(tmp_path / "symlink-output", tuple(inventory))

    inventory = list(_inventory(tmp_path / "escape-sources"))
    source = inventory[0].files[0]
    inventory[0] = DeploymentSourceRoot(
        id=inventory[0].id,
        kind=inventory[0].kind,
        relative_path=inventory[0].relative_path,
        files=(
            DeploymentSourceFile(
                relative_path="../escape",
                source=source.source,
                artifact_content_sha256=source.artifact_content_sha256,
                executable=source.executable,
            ),
        ),
    )
    with pytest.raises(ValueError, match="invalid evaluator source relative path"):
        _stage_verified_roots(tmp_path / "escape-output", tuple(inventory))
