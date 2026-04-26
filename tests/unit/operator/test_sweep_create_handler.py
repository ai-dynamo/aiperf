# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import kopf
import pytest

from aiperf.operator.handlers.sweep import create as sweep_create


def _valid_body() -> dict:
    return {
        "metadata": {
            "name": "s",
            "namespace": "ns",
            "uid": "u",
            "creationTimestamp": "2024-04-25T18:22:03Z",
        },
        "spec": {
            "multiRun": {"trials": 3},
            "template": {
                "spec": {
                    "image": "x:latest",
                    "benchmark": {
                        "models": ["m"],
                        "endpoint": {"urls": ["http://x"], "type": "chat"},
                        "datasets": [{"name": "main", "type": "synthetic"}],
                        "phases": [
                            {
                                "name": "profiling",
                                "type": "concurrency",
                                "duration": 1,
                                "concurrency": 1,
                            }
                        ],
                    },
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_handle_validates_spec_and_creates_jobset(monkeypatch):
    body = _valid_body()
    patch = kopf.Patch()
    provision_rbac = AsyncMock()
    create_jobset = AsyncMock()
    monkeypatch.setattr(sweep_create, "_provision_rbac", provision_rbac)
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", create_jobset)

    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )

    provision_rbac.assert_awaited_once()
    create_jobset.assert_awaited_once()
    assert patch.status["phase"] == "Pending"
    assert patch.status["totalVariations"] == 1
    assert patch.status["maxTotalRuns"] == 3
    assert "runtimeRef" in patch.status


@pytest.mark.asyncio
async def test_handle_rejects_invalid_spec(monkeypatch):
    body = {
        "metadata": {"name": "s", "namespace": "ns", "uid": "u"},
        "spec": {"template": {"spec": {"benchmark": {}}}},  # no axes
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    with pytest.raises(kopf.PermanentError, match="at least one of"):
        await sweep_create.handle(
            body=body,
            spec=body["spec"],
            name="s",
            namespace="ns",
            patch=patch,
        )


@pytest.mark.asyncio
async def test_epoch_from_creation_timestamp():
    """`metadata.creationTimestamp` parses to a decimal epoch in status.runEpoch."""
    from datetime import datetime, timezone

    expected = int(datetime(2024, 4, 25, 18, 22, 3, tzinfo=timezone.utc).timestamp())
    assert sweep_create._epoch_from_creation_ts("2024-04-25T18:22:03Z") == str(expected)


@pytest.mark.asyncio
async def test_handle_computes_max_total_runs_grid_x_trials(monkeypatch):
    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "grid",
        "variables": {"random_seed": [1, 2, 3, 4]},
    }
    body["spec"]["multiRun"]["trials"] = 5
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert patch.status["totalVariations"] == 4
    assert patch.status["maxTotalRuns"] == 20


# ===========================================================================
# Adversarial regression-locks for second-pass fixes (commit 793260d7b):
# `_create_or_skip_409` and `_create_or_skip_409_custom` must wrap non-409
# ApiException + connection errors in `kopf.TemporaryError(delay=30)`.
# ===========================================================================


@pytest.mark.asyncio
async def test_create_or_skip_409_non_409_apiexception_wraps_in_temporary_error():
    """ApiException(status=503) must surface as kopf.TemporaryError so kopf
    backs off rather than retrying unboundedly."""
    from kubernetes_asyncio.client import ApiException

    create_fn = AsyncMock(side_effect=ApiException(status=503, reason="Unavailable"))
    with pytest.raises(kopf.TemporaryError):
        await sweep_create._create_or_skip_409(create_fn, "ns", object())


@pytest.mark.asyncio
async def test_create_or_skip_409_409_apiexception_swallowed():
    """409 (AlreadyExists) is the idempotent-reconcile path — must not raise."""
    from kubernetes_asyncio.client import ApiException

    create_fn = AsyncMock(side_effect=ApiException(status=409, reason="AlreadyExists"))
    # Must not raise.
    await sweep_create._create_or_skip_409(create_fn, "ns", object())


@pytest.mark.asyncio
async def test_create_or_skip_409_aiohttp_connection_error_wraps_temporary_error():
    """aiohttp connection error must surface as kopf.TemporaryError too —
    transient network blips retry with backoff."""
    import aiohttp

    create_fn = AsyncMock(
        side_effect=aiohttp.ClientConnectionError("connection refused")
    )
    with pytest.raises(kopf.TemporaryError):
        await sweep_create._create_or_skip_409(create_fn, "ns", object())


# ===========================================================================
# Adversarial regression-locks for pod-spec lifting (second-pass fix).
# `_create_sweep_controller_jobset` must lift container-level resources,
# containerSecurityContext, pod-level securityContext, and merge user env
# (with reserved AIPERF_SWEEP_* names taking precedence).
# ===========================================================================


def _valid_template_spec(**overrides):
    """Minimal `template.spec` accepted by the Rule-5 validator.

    `overrides` are merged into `podTemplate`.
    """
    pod_template = dict(overrides)
    return {
        "image": "x:latest",
        "podTemplate": pod_template,
        "benchmark": {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ],
        },
    }


async def _capture_jobset_body(monkeypatch, template_spec) -> dict:
    """Drive `_create_sweep_controller_jobset` and capture the JobSet body
    passed to `create_namespaced_custom_object` (plural=jobsets)."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    captured: dict = {}

    create_mock = AsyncMock()

    async def _capture(**kwargs):
        # Only capture the jobsets call.
        if kwargs.get("plural") == "jobsets":
            captured["body"] = kwargs.get("body")
            captured["kwargs"] = kwargs

    create_mock.side_effect = _capture

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kw):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(create_namespaced_custom_object=create_mock),
    )

    await sweep_create._create_sweep_controller_jobset(
        name="s",
        namespace="ns",
        sweep_uid="uid",
        epoch="1714000000",
        template_spec=template_spec,
    )

    return captured["body"]


def _container_from_jobset(body: dict) -> dict:
    return body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]


def _pod_spec_from_jobset(body: dict) -> dict:
    return body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_container_resources(monkeypatch):
    """`podTemplate.resources` must land on container.resources."""
    template_spec = _valid_template_spec(
        resources={"requests": {"cpu": "500m", "memory": "1Gi"}}
    )
    body = await _capture_jobset_body(monkeypatch, template_spec)
    container = _container_from_jobset(body)
    assert container["resources"] == {"requests": {"cpu": "500m", "memory": "1Gi"}}


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_container_security_context(
    monkeypatch,
):
    """`podTemplate.containerSecurityContext` must land on container.securityContext."""
    template_spec = _valid_template_spec(
        containerSecurityContext={
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "allowPrivilegeEscalation": False,
        }
    )
    body = await _capture_jobset_body(monkeypatch, template_spec)
    container = _container_from_jobset(body)
    assert container["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "allowPrivilegeEscalation": False,
    }


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_pod_security_context(monkeypatch):
    """Pod-level `podTemplate.securityContext` must land on pod_spec.securityContext."""
    template_spec = _valid_template_spec(
        securityContext={"fsGroup": 2000, "runAsNonRoot": True}
    )
    body = await _capture_jobset_body(monkeypatch, template_spec)
    pod_spec = _pod_spec_from_jobset(body)
    assert pod_spec["securityContext"] == {"fsGroup": 2000, "runAsNonRoot": True}


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_merges_user_env_reserved_wins(
    monkeypatch,
):
    """User env (HTTP_PROXY) is merged in; reserved AIPERF_SWEEP_NAME from user
    is overridden by the controller's value, not vice versa."""
    template_spec = _valid_template_spec(
        env=[
            {"name": "HTTP_PROXY", "value": "http://proxy"},
            {"name": "AIPERF_SWEEP_NAME", "value": "hijack"},
        ]
    )
    body = await _capture_jobset_body(monkeypatch, template_spec)
    container = _container_from_jobset(body)
    env_by_name = {e["name"]: e["value"] for e in container["env"]}
    assert env_by_name.get("HTTP_PROXY") == "http://proxy", "user env must merge in"
    assert env_by_name.get("AIPERF_SWEEP_NAME") == "s", (
        "reserved AIPERF_SWEEP_NAME must keep controller's value, not user's 'hijack'"
    )
    # And the reserved var only appears once (no duplicate from user merge).
    sweep_name_entries = [
        e for e in container["env"] if e["name"] == "AIPERF_SWEEP_NAME"
    ]
    assert len(sweep_name_entries) == 1


# ===========================================================================
# Adversarial regression-lock: Role grants events create/patch.
# ===========================================================================


@pytest.mark.asyncio
async def test_provision_rbac_role_grants_events_create_patch(monkeypatch):
    """`_provision_rbac` Role must include a PolicyRule for events.create/patch
    (so the sweep-controller can emit kubectl-visible events)."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    captured: dict = {}

    async def _capture_role(_namespace, body):
        # The Role body has a `rules` attribute (V1Role).
        captured["role_rules"] = body.rules

    async def _noop_sa(_ns, _body):
        return None

    async def _noop_binding(_ns, _body):
        return None

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kw):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )

    core = MagicMock()
    core.create_namespaced_service_account = AsyncMock(side_effect=_noop_sa)
    rbac = MagicMock()
    rbac.create_namespaced_role = AsyncMock(side_effect=_capture_role)
    rbac.create_namespaced_role_binding = AsyncMock(side_effect=_noop_binding)

    import kubernetes_asyncio.client as k8s_client_mod

    monkeypatch.setattr(k8s_client_mod, "CoreV1Api", lambda _api: core)
    monkeypatch.setattr(k8s_client_mod, "RbacAuthorizationV1Api", lambda _api: rbac)

    await sweep_create._provision_rbac(name="s", namespace="ns", sweep_uid="uid")

    rules = captured["role_rules"]
    # Find the events rule.
    events_rules = [
        r for r in rules if r.api_groups == [""] and r.resources == ["events"]
    ]
    assert len(events_rules) == 1, "events PolicyRule missing"
    verbs = set(events_rules[0].verbs)
    assert "create" in verbs and "patch" in verbs
