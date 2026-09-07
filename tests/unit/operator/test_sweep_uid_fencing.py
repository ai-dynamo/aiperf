# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial identity-fencing tests for delayed AIPerfSweep callbacks."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiException
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_SWEEP_API_VERSION
from aiperf.operator import main as operator_main


@pytest.fixture
def install_custom_api(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[MagicMock], None]:
    """Route the production client helper to a focused CustomObjectsApi fake."""

    def _install(custom: MagicMock) -> None:
        @asynccontextmanager
        async def fake_k8s_client():
            yield MagicMock(name="ApiClient")

        import aiperf.kubernetes.client as k8s_client_module

        monkeypatch.setattr(k8s_client_module, "k8s_client", fake_k8s_client)
        monkeypatch.setattr(client, "CustomObjectsApi", lambda _api: custom)

    return _install


def _owned_jobset(*, sweep_uid: str = "sweep-uid") -> dict[str, Any]:
    return {
        "metadata": {
            "uid": "jobset-uid",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_SWEEP_API_VERSION,
                    "kind": "AIPerfSweep",
                    "name": "latency-sweep",
                    "uid": sweep_uid,
                    "controller": True,
                }
            ],
        }
    }


@pytest.mark.asyncio
async def test_sweep_parent_identity_same_name_replacement_is_stale(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A delayed callback must not treat name equality as parent identity."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={"metadata": {"uid": "replacement-uid"}}
    )
    install_custom_api(custom)

    current = await operator_main._sweep_parent_is_current(
        "benchmarks",
        "latency-sweep",
        "original-uid",
    )

    assert current is False


@pytest.mark.asyncio
async def test_sweep_parent_identity_transient_read_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """An apiserver outage is not evidence that the current parent is stale."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=503, reason="Unavailable")
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="identity read failed"):
        await operator_main._sweep_parent_is_current(
            "benchmarks",
            "latency-sweep",
            "sweep-uid",
        )


@pytest.mark.asyncio
async def test_owned_sweep_jobset_requires_exact_owner_and_returns_resource_uid(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """The JobSet owner triple and the JobSet's own UID are both preserved."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())
    install_custom_api(custom)

    jobset_uid = await operator_main._owned_sweep_jobset_uid(
        "benchmarks",
        "aiperf-latency-sweep",
        sweep_name="latency-sweep",
        sweep_uid="sweep-uid",
    )

    assert jobset_uid == "jobset-uid"


@pytest.mark.asyncio
async def test_owned_sweep_jobset_transient_read_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """An uncertain JobSet read cannot be treated as absence or replacement."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=503, reason="Unavailable")
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="identity read failed"):
        await operator_main._owned_sweep_jobset_uid(
            "benchmarks",
            "aiperf-latency-sweep",
            sweep_name="latency-sweep",
            sweep_uid="sweep-uid",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_field", "owner_value"),
    [
        param("apiVersion", "aiperf.nvidia.com/v0", id="wrong-api-version"),
        param("controller", False, id="not-controller"),
    ],
)  # fmt: skip
async def test_owned_sweep_jobset_rejects_incomplete_controller_identity(
    install_custom_api: Callable[[MagicMock], None],
    owner_field: str,
    owner_value: object,
) -> None:
    """Kind/name/UID alone cannot authorize destructive JobSet cleanup."""
    jobset = _owned_jobset()
    jobset["metadata"]["ownerReferences"][0][owner_field] = owner_value
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=jobset)
    install_custom_api(custom)

    with pytest.raises(operator_main._StaleSweepCallback, match="not owned"):
        await operator_main._owned_sweep_jobset_uid(
            "benchmarks",
            "aiperf-latency-sweep",
            sweep_name="latency-sweep",
            sweep_uid="sweep-uid",
        )


@pytest.mark.asyncio
async def test_delete_sweep_jobset_foreign_owner_is_safe_noop(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A same-name JobSet owned by a replacement sweep is never deleted."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value=_owned_jobset(sweep_uid="replacement-uid")
    )
    custom.delete_namespaced_custom_object = AsyncMock()
    install_custom_api(custom)

    await operator_main._delete_sweep_jobset(
        "benchmarks",
        "aiperf-latency-sweep",
        sweep_name="latency-sweep",
        sweep_uid="original-uid",
    )

    custom.delete_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_sweep_jobset_uses_resource_uid_precondition(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """Deletion is fenced again against replacement after the ownership read."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())
    custom.delete_namespaced_custom_object = AsyncMock()
    install_custom_api(custom)

    await operator_main._delete_sweep_jobset(
        "benchmarks",
        "aiperf-latency-sweep",
        sweep_name="latency-sweep",
        sweep_uid="sweep-uid",
    )

    kwargs = custom.delete_namespaced_custom_object.await_args.kwargs
    assert kwargs["body"].preconditions.uid == "jobset-uid"


@pytest.mark.asyncio
async def test_delete_sweep_jobset_uid_conflict_is_safe_noop(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A 409 from the UID precondition proves the read resource was replaced."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())
    custom.delete_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=409, reason="Conflict")
    )
    install_custom_api(custom)

    await operator_main._delete_sweep_jobset(
        "benchmarks",
        "aiperf-latency-sweep",
        sweep_name="latency-sweep",
        sweep_uid="sweep-uid",
    )


@pytest.mark.asyncio
async def test_delete_sweep_jobset_transient_error_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A failed delete against the still-current resource remains retryable."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())
    custom.delete_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=503, reason="Unavailable")
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="delete failed"):
        await operator_main._delete_sweep_jobset(
            "benchmarks",
            "aiperf-latency-sweep",
            sweep_name="latency-sweep",
            sweep_uid="sweep-uid",
        )


@pytest.mark.asyncio
async def test_publish_sweep_aggregate_status_starts_with_parent_uid_test(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """Status and discovery pointers cannot be published to a replacement CR."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    install_custom_api(custom)
    aggregate_ref = {"url": "http://operator/aggregate.json"}

    published = await operator_main._publish_durable_sweep_aggregate_ref(
        "benchmarks",
        "latency-sweep",
        aggregate_ref,
        "sweep-uid",
    )

    assert published is True
    kwargs = custom.patch_namespaced_custom_object_status.await_args.kwargs
    assert kwargs["_content_type"] == "application/json-patch+json"
    assert kwargs["body"] == [
        {
            "op": "test",
            "path": "/metadata/uid",
            "value": "sweep-uid",
        },
        {
            "op": "add",
            "path": "/status/aggregateRef",
            "value": aggregate_ref,
        },
        {
            "op": "add",
            "path": "/status/resultsAvailable",
            "value": True,
        },
    ]


@pytest.mark.asyncio
async def test_publish_sweep_aggregate_uid_test_failure_is_stale_noop(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A failed JSON Patch test stops the callback without retrying forever."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=422, reason="UID test failed")
    )
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={"metadata": {"uid": "replacement-uid"}}
    )
    install_custom_api(custom)

    published = await operator_main._publish_durable_sweep_aggregate_ref(
        "benchmarks",
        "latency-sweep",
        {"url": "http://operator/aggregate.json"},
        "old-uid",
    )

    assert published is False


@pytest.mark.asyncio
async def test_publish_sweep_aggregate_422_on_current_parent_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A schema or malformed-patch rejection must not strand current-owner harvest."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=422, reason="status schema rejected")
    )
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={"metadata": {"uid": "sweep-uid"}}
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="publication failed"):
        await operator_main._publish_durable_sweep_aggregate_ref(
            "benchmarks",
            "latency-sweep",
            {"url": "http://operator/aggregate.json"},
            "sweep-uid",
        )


@pytest.mark.asyncio
async def test_publish_sweep_aggregate_transient_error_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A current-owner status write remains retryable on apiserver failure."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=503, reason="Unavailable")
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="publication failed"):
        await operator_main._publish_durable_sweep_aggregate_ref(
            "benchmarks",
            "latency-sweep",
            {"url": "http://operator/aggregate.json"},
            "sweep-uid",
        )


@pytest.mark.asyncio
async def test_publish_sweep_aggregate_write_conflict_retries(
    install_custom_api: Callable[[MagicMock], None],
) -> None:
    """A generic 409 is not sufficient proof that the UID test failed."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=409, reason="Conflict")
    )
    install_custom_api(custom)

    with pytest.raises(kopf.TemporaryError, match="publication failed"):
        await operator_main._publish_durable_sweep_aggregate_ref(
            "benchmarks",
            "latency-sweep",
            {"url": "http://operator/aggregate.json"},
            "sweep-uid",
        )


@pytest.mark.asyncio
async def test_stale_status_publication_does_not_advance_discovery_or_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed UID fence stops discovery and delete after durable lineage."""
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    publish = AsyncMock(return_value=False)
    lineage = MagicMock(return_value=1)
    latest = MagicMock()
    index = AsyncMock()
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_publish_durable_sweep_aggregate_ref", publish)
    monkeypatch.setattr(operator_main, "_materialize_sweep_child_lineage", lineage)
    monkeypatch.setattr(_aggregate_fetch, "_write_sweep_latest_pointer", latest)
    monkeypatch.setattr(operator_main.runs_index, "_index_sweep_from_disk", index)
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)

    committed = await operator_main._commit_existing_sweep_archive(
        base_dir=tmp_path,
        namespace="benchmarks",
        name="latency-sweep",
        epoch="123",
        durable_ref={"url": "http://operator/aggregate.json"},
        jobset_name="aiperf-latency-sweep",
        sweep_uid="old-uid",
        delete_jobset=True,
    )

    assert committed is False
    lineage.assert_called_once()
    publish.assert_awaited_once()
    latest.assert_not_called()
    index.assert_not_awaited()
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_harvest_callback_does_not_fetch_or_inspect_jobset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted-and-recreated parent makes the whole delayed callback a no-op."""
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    fetch = AsyncMock()
    inspect_jobset = AsyncMock()
    monkeypatch.setattr(
        operator_main,
        "_sweep_parent_is_current",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(operator_main, "_owned_sweep_jobset_uid", inspect_jobset)
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)

    await operator_main.on_aiperfsweep_aggregation_complete(
        body={"metadata": {"uid": "old-uid"}},
        status={"runEpoch": "123", "aggregation": {"phase": "Complete"}},
        name="latency-sweep",
        namespace="benchmarks",
    )

    inspect_jobset.assert_not_awaited()
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreign_jobset_makes_harvest_callback_safe_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused JobSet name cannot redirect an old callback to new artifacts."""
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    fetch = AsyncMock()
    monkeypatch.setattr(
        operator_main,
        "_sweep_parent_is_current",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        operator_main,
        "_owned_sweep_jobset_uid",
        AsyncMock(side_effect=operator_main._StaleSweepCallback("foreign owner")),
    )
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)

    await operator_main.on_aiperfsweep_aggregation_complete(
        body={"metadata": {"uid": "old-uid"}},
        status={"runEpoch": "123", "aggregation": {"phase": "Complete"}},
        name="latency-sweep",
        namespace="benchmarks",
    )

    fetch.assert_not_awaited()
