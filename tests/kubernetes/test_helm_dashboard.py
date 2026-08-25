# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chart-render assertions for the optional Plotly Dashboard sidecar."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[2] / "deploy" / "helm" / "aiperf-operator"


def _render(*set_args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.fail("helm CLI is required for Kubernetes chart-render tests")
    cmd = ["helm", "template", "test", str(CHART)]
    for s in set_args:
        cmd.extend(["--set", s])
    out = subprocess.check_output(cmd, text=True)
    return [d for d in yaml.safe_load_all(out) if d]


def _operator_deployment(docs: list[dict]) -> dict:
    for d in docs:
        if d.get("kind") == "Deployment" and "operator" in d["metadata"]["name"]:
            return d
    raise AssertionError("operator Deployment not in render")


def _container(deploy: dict, name: str) -> dict | None:
    return next(
        (
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == name
        ),
        None,
    )


def _env(container: dict, name: str) -> str | None:
    for e in container.get("env", []):
        if e["name"] == name:
            return e.get("value")
    return None


def test_dashboard_container_absent_by_default() -> None:
    docs = _render()
    deploy = _operator_deployment(docs)
    assert _container(deploy, "dashboard") is None
    operator = _container(deploy, "operator")
    results = _container(deploy, "results-server")
    assert _env(operator, "AIPERF_DASHBOARD_PORT") == "0"
    assert _env(results, "AIPERF_DASHBOARD_PROXY_ENABLED") == "0"


def test_dashboard_container_present_when_enabled() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash is not None
    assert dash["command"] == [
        "python",
        "-m",
        "aiperf.operator.dashboard_server",
    ]
    port = next(p["containerPort"] for p in dash["ports"] if p["name"] == "dashboard")
    assert port == 8082
    operator = _container(deploy, "operator")
    results = _container(deploy, "results-server")
    assert _env(operator, "AIPERF_DASHBOARD_PORT") == "8082"
    assert _env(results, "AIPERF_DASHBOARD_PROXY_ENABLED") == "1"
    assert _env(results, "AIPERF_DASHBOARD_PORT") == "8082"


def test_dashboard_limits_omitted_when_empty() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash["resources"]["requests"]["memory"] == "1Gi"
    assert dash["resources"].get("limits") in (None, {})


def test_dashboard_limits_respected_when_set() -> None:
    docs = _render(
        "dashboard.enabled=true",
        "dashboard.resources.limits.memory=4Gi",
    )
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash["resources"]["limits"]["memory"] == "4Gi"


def test_dashboard_pvc_mount_is_readonly() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    results_mount = next(m for m in dash["volumeMounts"] if m["name"] == "results")
    assert results_mount.get("readOnly") is True
