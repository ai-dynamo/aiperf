# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm install must adopt pre-existing chart CRDs into the helm field manager.

The package-scoped operator fixture installs the chart's CRDs with
``kubectl apply`` (field manager ``kubectl-client-side-apply``). Helm 4
applies server-side by default, so ``helm install`` of the same chart then
fails with field conflicts on the CRD objects
(``.metadata.labels.app.kubernetes.io/instance``, ``.spec.versions``) — the
h1/h2/h3 chaos scenarios failed every run this way. ``install_chart``
re-applies the chart's own rendered CRDs with SSA + ``--force-conflicts`` +
``--field-manager=helm`` before installing, scoped to CRD objects only so a
stale value anywhere else still fails loudly instead of being force-adopted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from tests.kubernetes.helpers import helm as helm_mod
from tests.kubernetes.helpers.helm import HelmDeployer, HelmValues

_RENDERED = """\
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: aiperfjobs.aiperf.nvidia.com
  labels:
    app.kubernetes.io/instance: aiperf-operator-helm
spec:
  group: aiperf.nvidia.com
  versions:
    - name: v1alpha1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiperf-operator
"""


def _ok(*args: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_install_chart_adopts_crds_into_helm_field_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = SimpleNamespace()
    kubectl.run = AsyncMock(side_effect=_ok)
    kubectl.apply_server_side_manifest = AsyncMock(return_value="")

    helm = SimpleNamespace()
    helm.get_release_status = AsyncMock(return_value="not-found")
    helm.template = AsyncMock(return_value=_RENDERED)
    helm.install = AsyncMock()

    monkeypatch.setattr(
        helm_mod, "wait_for_aiperf_crds_established", AsyncMock()
    )

    deployer = HelmDeployer(
        kubectl=kubectl,  # type: ignore[arg-type]
        helm=helm,  # type: ignore[arg-type]
        project_root=Path("/nonexistent"),
        values=HelmValues(),
        operator_namespace="aiperf-system-test",
        default_job_namespace="jobs-test",
    )

    await deployer.install_chart(wait=False)

    adopt = kubectl.apply_server_side_manifest
    assert adopt.await_count == 1, "CRDs must be pre-adopted exactly once"
    _, kwargs = adopt.await_args
    assert kwargs["field_manager"] == "helm"
    assert kwargs["force_conflicts"] is True

    adopted_docs = [
        doc
        for doc in yaml.safe_load_all(adopt.await_args.args[0])
        if doc
    ]
    assert [doc["kind"] for doc in adopted_docs] == ["CustomResourceDefinition"], (
        "SSA force-conflicts adoption must be scoped to CRD objects only"
    )
    assert adopted_docs[0]["metadata"]["name"] == "aiperfjobs.aiperf.nvidia.com"

    # Adoption must happen before helm install so the install's SSA finds no
    # foreign-owned CRD fields.
    assert helm.install.await_count == 1


@pytest.mark.asyncio
async def test_install_chart_skips_adoption_when_render_has_no_crds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = SimpleNamespace()
    kubectl.run = AsyncMock(side_effect=_ok)
    kubectl.apply_server_side_manifest = AsyncMock(return_value="")

    helm = SimpleNamespace()
    helm.get_release_status = AsyncMock(return_value="not-found")
    helm.template = AsyncMock(return_value="apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\n")
    helm.install = AsyncMock()

    monkeypatch.setattr(
        helm_mod, "wait_for_aiperf_crds_established", AsyncMock()
    )

    deployer = HelmDeployer(
        kubectl=kubectl,  # type: ignore[arg-type]
        helm=helm,  # type: ignore[arg-type]
        project_root=Path("/nonexistent"),
        values=HelmValues(),
        operator_namespace="aiperf-system-test",
        default_job_namespace="jobs-test",
    )

    await deployer.install_chart(wait=False)

    kubectl.apply_server_side_manifest.assert_not_awaited()
    assert helm.install.await_count == 1
