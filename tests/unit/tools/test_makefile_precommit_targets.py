# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make targets used by isolated pre-commit environments."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("generate-crd", id="generate-crd"),
        pytest.param("check-chart-consistency", id="check-chart-consistency"),
    ],
)
def test_precommit_target_runs_without_repository_virtualenv(
    target: str, tmp_path: Path
) -> None:
    """pre-commit.ci has its own environment and no repository ``.venv``."""
    result = subprocess.run(
        ["make", target, "VENV_PATH=.missing-pre-commit-venv"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        # The targets are ``uv run python ...``, which reconciles the project
        # environment with uv.lock. Point that at a throwaway path so the sync
        # cannot rewrite the dependencies of the suite running this test.
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv")},
    )

    assert result.returncode == 0, result.stdout + result.stderr
