# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chart-render invariants for operator least-privilege and index mount access.

Locks in two audit findings:
- The operator ClusterRole must not hold cluster-wide ``list``/``watch`` on
  secrets. Preflight only reads named secrets (``read_namespaced_secret``), so
  ``get`` is sufficient; ``resourcequotas`` genuinely needs ``list``
  (``list_namespaced_resource_quota``).
- The results-server results mount must NOT be read-only: it opens the
  operator-written WAL-mode SQLite runs index, and a WAL reader has to be able
  to create the ``-shm`` sidecar.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

CHART_PATH = Path(__file__).parents[3] / "deploy" / "helm" / "aiperf-operator"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None or not CHART_PATH.exists(),
    reason="helm CLI and the aiperf-operator chart are required",
)


def _render(*extra: str) -> list[dict[str, Any]]:
    """Render the chart and return its YAML documents."""
    import yaml

    out = subprocess.check_output(
        [
            "helm",
            "template",
            "aiperf-operator",
            str(CHART_PATH),
            "-n",
            DEFAULT_OPERATOR_NAMESPACE,
            *extra,
        ],
        text=True,
    )
    return [doc for doc in yaml.safe_load_all(out) if isinstance(doc, dict)]


def _operator_cluster_role(docs: list[dict[str, Any]]) -> dict[str, Any]:
    for doc in docs:
        if (
            doc.get("kind") == "ClusterRole"
            and doc["metadata"]["name"] == "aiperf-operator"
        ):
            return doc
    raise AssertionError("operator ClusterRole not rendered")


def _rules_for(cluster_role: dict[str, Any], resource: str) -> list[dict[str, Any]]:
    return [
        rule
        for rule in cluster_role.get("rules", [])
        if resource in rule.get("resources", []) and "" in rule.get("apiGroups", [])
    ]


def _container(docs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            if container["name"] == name:
                return container
    raise AssertionError(f"container {name!r} not rendered")


def test_cluster_role_secrets_verbs_are_get_only() -> None:
    rules = _rules_for(_operator_cluster_role(_render()), "secrets")
    assert rules, "no core/secrets rule rendered"
    for rule in rules:
        assert set(rule["verbs"]) == {"get"}, (
            "cluster-wide list/watch on secrets lets the operator enumerate every "
            f"secret in the cluster; got {rule['verbs']}"
        )


@pytest.mark.parametrize(
    "verb",
    [
        param("get", id="get"),
        param("list", id="list"),
    ],
)  # fmt: skip
def test_cluster_role_resourcequotas_keeps_enumeration_verbs(verb: str) -> None:
    rules = _rules_for(_operator_cluster_role(_render()), "resourcequotas")
    assert rules, "no core/resourcequotas rule rendered"
    assert any(verb in rule["verbs"] for rule in rules)


def test_results_server_results_mount_is_writable() -> None:
    mounts = _container(_render(), "results-server")["volumeMounts"]
    results = next(m for m in mounts if m["name"] == "results")
    assert not results.get("readOnly"), (
        "the results-server opens the WAL-mode runs index, which needs to create "
        "the -shm sidecar; a read-only mount silently disables the index"
    )


def test_dashboard_results_mount_stays_read_only() -> None:
    mounts = _container(_render("--set", "dashboard.enabled=true"), "dashboard")[
        "volumeMounts"
    ]
    results = next(m for m in mounts if m["name"] == "results")
    assert results.get("readOnly") is True
