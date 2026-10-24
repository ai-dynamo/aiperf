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
``--field-manager=helm`` before installing, scoped to CRD objects only, and
refuses the takeover when the live CRD spec has drifted from the chart's.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
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
  names:
    kind: AIPerfJob
    plural: aiperfjobs
  scope: Namespaced
  versions:
    - name: v1alpha1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiperf-operator
"""


def _rendered_crd_spec() -> dict:
    return next(
        doc["spec"]
        for doc in yaml.safe_load_all(_RENDERED)
        if doc and doc.get("kind") == "CustomResourceDefinition"
    )


def _make_deployer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_crd_spec: dict | None = None,
    call_order: list[str] | None = None,
) -> tuple[HelmDeployer, SimpleNamespace, SimpleNamespace]:
    """HelmDeployer over fully mocked kubectl/helm.

    ``live_crd_spec=None`` simulates a fresh cluster (the CRD does not exist
    yet); a dict simulates a pre-existing CRD with that live spec.
    ``call_order`` records "adopt" / "install" invocation order.
    """
    kubectl = SimpleNamespace()

    async def _run(*args: str, **kwargs: object) -> SimpleNamespace:
        if args[:2] == ("get", "crd"):
            if live_crd_spec is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            return SimpleNamespace(
                returncode=0,
                stdout=orjson.dumps({"spec": live_crd_spec}).decode(),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def _adopt(*args: object, **kwargs: object) -> str:
        if call_order is not None:
            call_order.append("adopt")
        return ""

    async def _install(*args: object, **kwargs: object) -> None:
        if call_order is not None:
            call_order.append("install")

    kubectl.run = AsyncMock(side_effect=_run)
    kubectl.apply_server_side_manifest = AsyncMock(side_effect=_adopt)

    helm = SimpleNamespace()
    helm.get_release_status = AsyncMock(return_value="not-found")
    helm.template = AsyncMock(return_value=_RENDERED)
    helm.install = AsyncMock(side_effect=_install)

    monkeypatch.setattr(helm_mod, "wait_for_aiperf_crds_established", AsyncMock())

    deployer = HelmDeployer(
        kubectl=kubectl,  # type: ignore[arg-type]
        helm=helm,  # type: ignore[arg-type]
        project_root=Path("/nonexistent"),
        values=HelmValues(),
        operator_namespace="aiperf-system-test",
        default_job_namespace="jobs-test",
    )
    return deployer, kubectl, helm


@pytest.mark.asyncio
async def test_install_chart_adopts_crds_into_helm_field_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    deployer, kubectl, helm = _make_deployer(monkeypatch, call_order=call_order)

    await deployer.install_chart(wait=False)

    adopt = kubectl.apply_server_side_manifest
    assert adopt.await_count == 1, "CRDs must be pre-adopted exactly once"
    _, kwargs = adopt.await_args
    assert kwargs["field_manager"] == "helm"
    assert kwargs["force_conflicts"] is True

    adopted_docs = [doc for doc in yaml.safe_load_all(adopt.await_args.args[0]) if doc]
    assert [doc["kind"] for doc in adopted_docs] == ["CustomResourceDefinition"], (
        "SSA force-conflicts adoption must be scoped to CRD objects only"
    )
    assert adopted_docs[0]["metadata"]["name"] == "aiperfjobs.aiperf.nvidia.com"

    assert call_order == ["adopt", "install"], (
        "adoption must precede helm install so the install's SSA finds no "
        "foreign-owned CRD fields"
    )


@pytest.mark.asyncio
async def test_install_chart_adopts_when_live_spec_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live CRD the fixture installed from this same chart adopts cleanly."""
    deployer, kubectl, helm = _make_deployer(
        monkeypatch, live_crd_spec=_rendered_crd_spec()
    )

    await deployer.install_chart(wait=False)

    assert kubectl.apply_server_side_manifest.await_count == 1
    assert helm.install.await_count == 1


@pytest.mark.asyncio
async def test_install_chart_refuses_drifted_live_crd_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live CRD whose spec differs from the chart's was not produced by this
    chart's CRD install; the takeover must fail with a diff, not overwrite."""
    drifted = {**_rendered_crd_spec(), "group": "other.example.com"}
    deployer, kubectl, helm = _make_deployer(monkeypatch, live_crd_spec=drifted)

    with pytest.raises(RuntimeError, match="Refusing to force-adopt CRD") as exc:
        await deployer.install_chart(wait=False)

    assert "other.example.com" in str(exc.value), "the failure must show the drift"
    kubectl.apply_server_side_manifest.assert_not_awaited()
    helm.install.assert_not_awaited()


@pytest.mark.asyncio
async def test_install_chart_skips_adoption_when_render_has_no_crds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer, kubectl, helm = _make_deployer(monkeypatch)
    helm.template = AsyncMock(
        return_value="apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\n"
    )

    await deployer.install_chart(wait=False)

    kubectl.apply_server_side_manifest.assert_not_awaited()
    assert helm.install.await_count == 1
