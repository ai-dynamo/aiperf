# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for FastAPI route registration in `results_server.create_app`."""

from pathlib import Path

import pytest


def test_create_app_includes_sweeps_router(tmp_path: Path) -> None:
    """`/api/v1/sweeps` endpoints must be registered alongside jobs."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/v1/sweeps" in routes
    assert "/api/v1/sweeps/{namespace}/{name}" in routes
    assert "/api/v1/sweeps/{namespace}/{name}/cells" in routes


def test_create_app_mounts_packaged_ui_when_override_env_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The results server ignores runtime UI override env and serves bundled UI."""
    from aiperf.operator.results_server import create_app

    monkeypatch.setenv("AIPERF_DEV_UI_OVERRIDE_DIR", str(tmp_path / "override"))

    app = create_app(results_dir=tmp_path)
    ui_route = next(r for r in app.routes if getattr(r, "name", None) == "ui")

    assert Path(ui_route.app.directory).name == "ui"
