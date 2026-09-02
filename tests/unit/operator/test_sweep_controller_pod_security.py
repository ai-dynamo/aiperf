# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sweep-controller pod must carry the same pod-level security baseline as
the JobSet path, with user-supplied keys merged over it rather than replacing it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.kubernetes.jobset_specs import POD_SECURITY_CONTEXT
from aiperf.operator.handlers.sweep import create as sweep_create


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a mock ApiClient without opening a real Kubernetes connection."""
    yield MagicMock(name="ApiClient")


async def _sweep_pod_spec(
    monkeypatch: pytest.MonkeyPatch, template_spec: dict[str, Any]
) -> dict[str, Any]:
    """Build the sweep-controller JobSet and return its pod spec."""
    captured: dict[str, Any] = {}

    async def _capture_create(**kwargs: Any) -> None:
        captured["body"] = kwargs["body"]

    custom = MagicMock(
        create_namespaced_custom_object=AsyncMock(side_effect=_capture_create)
    )
    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client",
        lambda **_kwargs: _fake_k8s_client(),
        raising=True,
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: custom
    )

    await sweep_create._create_sweep_controller_jobset(
        name="grid",
        namespace="production",
        sweep_uid="uid-grid",
        epoch="1714000000",
        template_spec=template_spec,
    )
    body = captured["body"]
    return body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]


def _template_spec(pod_template: dict[str, Any] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"image": "registry.example.com/aiperf:sweep"}
    if pod_template is not None:
        spec["podTemplate"] = pod_template
    return spec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pod_template",
    [
        param(None, id="no-pod-template"),
        param({}, id="empty-pod-template"),
        param({"nodeSelector": {"gpu": "h100"}}, id="pod-template-without-security"),
    ],
)  # fmt: skip
async def test_create_sweep_controller_jobset_omitted_security_context_applies_baseline(
    monkeypatch: pytest.MonkeyPatch, pod_template: dict[str, Any] | None
) -> None:
    pod_spec = await _sweep_pod_spec(monkeypatch, _template_spec(pod_template))

    assert pod_spec["securityContext"] == POD_SECURITY_CONTEXT


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_partial_security_context_merges_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pod_spec = await _sweep_pod_spec(
        monkeypatch,
        _template_spec({"podSecurityContext": {"fsGroup": 2000}}),
    )

    assert pod_spec["securityContext"] == {**POD_SECURITY_CONTEXT, "fsGroup": 2000}
    # Baseline keys the user did not mention survive the merge.
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert pod_spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_user_security_context_overrides_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Benchmarking tool: the baseline is a default, never a lock.
    override = {"runAsNonRoot": False, "runAsUser": 0, "runAsGroup": 0}
    pod_spec = await _sweep_pod_spec(
        monkeypatch, _template_spec({"podSecurityContext": override})
    )

    assert pod_spec["securityContext"]["runAsNonRoot"] is False
    assert pod_spec["securityContext"]["runAsUser"] == 0
    assert pod_spec["securityContext"]["runAsGroup"] == 0
    assert pod_spec["securityContext"]["fsGroup"] == POD_SECURITY_CONTEXT["fsGroup"]


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_baseline_constant_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_before = dict(POD_SECURITY_CONTEXT)

    await _sweep_pod_spec(
        monkeypatch, _template_spec({"podSecurityContext": {"fsGroup": 3000}})
    )
    await _sweep_pod_spec(monkeypatch, _template_spec(None))

    assert baseline_before == POD_SECURITY_CONTEXT


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_scheduling_keys_still_lifted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pod_spec = await _sweep_pod_spec(
        monkeypatch,
        _template_spec(
            {
                "nodeSelector": {"gpu": "h100"},
                "tolerations": [{"key": "gpu", "operator": "Exists"}],
                "priorityClassName": "high",
                "imagePullSecrets": [{"name": "regcred"}],
            }
        ),
    )

    assert pod_spec["nodeSelector"] == {"gpu": "h100"}
    assert pod_spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
    assert pod_spec["priorityClassName"] == "high"
    assert pod_spec["imagePullSecrets"] == [{"name": "regcred"}]
