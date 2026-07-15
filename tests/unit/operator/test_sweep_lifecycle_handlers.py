# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC
from unittest.mock import AsyncMock

import kopf
import pytest

from aiperf.operator.handlers.sweep import child_rollup, lifecycle


@pytest.fixture(autouse=True)
def _hermetic_k8s_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to a stub ``k8s_client``.

    ``on_child_phase_transition`` opens ``k8s_client()`` itself (it holds one
    client for the whole rollup tick), so tests that monkeypatch only the inner
    ``_count_owned_children`` / ``_patch_parent_status`` / ``_read_parent_status``
    helpers still reach a live client-open, which falls through to
    ``load_kube_config()`` and depends on the developer's ``~/.kube/config``.
    Tests that assert on apiserver calls install their own fake via
    ``_install_fake_k8s_for_rollup`` / ``_install_fake_k8s_for_lifecycle`` — those
    ``monkeypatch.setattr`` calls re-patch over this default within the test body.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    import aiperf.kubernetes.client as kclient

    @asynccontextmanager
    async def _stub(*, kubeconfig: str | None = None, context: str | None = None):
        yield MagicMock(name="ApiClient")

    monkeypatch.setattr(kclient, "k8s_client", _stub)


@pytest.mark.asyncio
async def test_cancel_handler_writes_cancelling_condition():
    patch = kopf.Patch()
    await lifecycle.cancel(
        body={"metadata": {"name": "s"}, "spec": {"cancel": True}},
        spec={"cancel": True},
        name="s",
        namespace="ns",
        patch=patch,
    )
    conditions = patch.status.get("conditions") or []
    assert any(
        c.get("type") == "Cancelling" and c.get("status") == "True" for c in conditions
    )


@pytest.mark.asyncio
async def test_cancel_noop_when_cancel_false():
    """If spec.cancel is false, the handler does nothing."""
    patch = kopf.Patch()
    await lifecycle.cancel(
        body={"metadata": {"name": "s"}, "spec": {"cancel": False}},
        spec={"cancel": False},
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert "conditions" not in patch.status or not patch.status["conditions"]


@pytest.mark.asyncio
async def test_child_rollup_skips_unowned_aiperfjob(monkeypatch):
    """A standalone AIPerfJob (no AIPerfSweep ownerRef) is a no-op."""
    patch_parent = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 0,
                "failed": 0,
                "in_flight": 0,
                "total_terminal_phase": None,
            }
        ),
    )

    body = {"metadata": {"name": "child", "namespace": "ns", "ownerReferences": []}}
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    patch_parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_rollup_increments_counts_and_transitions_phase(monkeypatch):
    """When all children terminal AND completed+failed >= maxTotalRuns, parent
    transitions to Aggregating.

    Counts/lastChildEvent ride a regular merge-patch via
    ``_patch_parent_status``; ``status.phase`` rides
    ``_conditional_phase_set`` (JSON-patch with a ``test`` op so a
    concurrent terminal write from the sweep-controller can't be
    clobbered).
    """
    parent_patches: list[dict] = []
    phase_calls: list[dict] = []

    async def fake_patch(*, group, version, plural, name, namespace, body, api=None):
        parent_patches.append(body)

    async def fake_phase_set(*, namespace, name, expect_phase, new_phase, api=None):
        phase_calls.append({"expect": expect_phase, "new": new_phase})

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
    monkeypatch.setattr(
        child_rollup,
        "_read_parent_status",
        AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 6}),
    )
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 5,
                "failed": 1,
                "in_flight": 0,
                "total_terminal_phase": "Aggregating",
            }
        ),
    )

    body = {
        "metadata": {
            "name": "child",
            "namespace": "ns",
            "ownerReferences": [{"kind": "AIPerfSweep", "name": "s", "uid": "u"}],
        },
    }
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    assert len(parent_patches) == 1
    body_patch = parent_patches[0]
    assert body_patch["status"]["completedRuns"] == 5
    assert body_patch["status"]["failedRuns"] == 1
    assert "phase" not in body_patch["status"]
    assert phase_calls == [{"expect": "Running", "new": "Aggregating"}]


@pytest.mark.asyncio
async def test_child_rollup_does_not_aggregate_mid_run(monkeypatch):
    """Mid-sweep: only 3 of 15 children created so far (all terminal); the
    sweep-controller is still walking variations × trials and will lazily
    create the next child. The currently-listed children are 100% terminal,
    but ``completedRuns + failedRuns < maxTotalRuns`` — so the rollup must
    NOT flip the parent phase to ``Aggregating``. Counts/lastChildEvent
    still ride the merge-patch.

    Regression for the live-DGX bug: a 5-concurrency × 3-trial sweep flipped
    to ``Aggregating`` after the very first cell terminated and stayed there
    for the remaining 14 children.
    """
    parent_patches: list[dict] = []
    phase_calls: list[dict] = []

    async def fake_patch(*, group, version, plural, name, namespace, body, api=None):
        parent_patches.append(body)

    async def fake_phase_set(*, namespace, name, expect_phase, new_phase, api=None):
        phase_calls.append({"expect": expect_phase, "new": new_phase})

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
    monkeypatch.setattr(
        child_rollup,
        "_read_parent_status",
        AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 15}),
    )
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 3,
                "failed": 0,
                "in_flight": 0,
                "total_terminal_phase": "Aggregating",
            }
        ),
    )

    body = {
        "metadata": {
            "name": "child",
            "namespace": "ns",
            "ownerReferences": [{"kind": "AIPerfSweep", "name": "s", "uid": "u"}],
        },
    }
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    # Counts merge-patch must still happen.
    assert len(parent_patches) == 1
    assert parent_patches[0]["status"]["completedRuns"] == 3
    # Phase write must NOT happen — sweep is mid-run.
    assert phase_calls == []


@pytest.mark.asyncio
async def test_child_rollup_aggregates_when_all_runs_accounted_for(monkeypatch):
    """Final child of a 15-cell sweep terminates: completedRuns+failedRuns ==
    maxTotalRuns, so the rollup advances ``phase=Aggregating``."""
    parent_patches: list[dict] = []
    phase_calls: list[dict] = []

    async def fake_patch(*, group, version, plural, name, namespace, body, api=None):
        parent_patches.append(body)

    async def fake_phase_set(*, namespace, name, expect_phase, new_phase, api=None):
        phase_calls.append({"expect": expect_phase, "new": new_phase})

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
    monkeypatch.setattr(
        child_rollup,
        "_read_parent_status",
        AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 15}),
    )
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 14,
                "failed": 1,
                "in_flight": 0,
                "total_terminal_phase": "Aggregating",
            }
        ),
    )

    body = {
        "metadata": {
            "name": "child",
            "namespace": "ns",
            "ownerReferences": [{"kind": "AIPerfSweep", "name": "s", "uid": "u"}],
        },
    }
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    assert phase_calls == [{"expect": "Running", "new": "Aggregating"}]


# ===========================================================================
# Adversarial regression-locks for second-pass fixes (commit 793260d7b).
# ===========================================================================


# ---------------------------------------------------------------------------
# child_rollup — apiserver errors wrap in kopf.TemporaryError(delay=15)
# ---------------------------------------------------------------------------


def _install_fake_k8s_for_rollup(
    monkeypatch,
    *,
    list_side_effect=None,
    list_return=None,
    patch_status_side_effect=None,
):
    """Install fake k8s_client + CustomObjectsApi for child_rollup helpers."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    list_mock = AsyncMock()
    if list_side_effect is not None:
        list_mock.side_effect = list_side_effect
    else:
        list_mock.return_value = list_return or {"items": []}
    patch_mock = AsyncMock()
    if patch_status_side_effect is not None:
        patch_mock.side_effect = patch_status_side_effect

    custom = MagicMock()
    custom.list_namespaced_custom_object = list_mock
    custom.patch_namespaced_custom_object_status = patch_mock

    fake_k8s_module = SimpleNamespace(CustomObjectsApi=lambda _api: custom)

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    import kubernetes_asyncio

    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module, raising=False)
    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    return list_mock, patch_mock


@pytest.mark.asyncio
async def test_count_owned_children_apiexception_500_wraps_temporary_error(monkeypatch):
    """`_count_owned_children` apiserver 500 must surface as kopf.TemporaryError."""
    from kubernetes_asyncio.client import ApiException

    _install_fake_k8s_for_rollup(
        monkeypatch,
        list_side_effect=ApiException(status=500, reason="Internal"),
    )
    with pytest.raises(kopf.TemporaryError):
        await child_rollup._count_owned_children("ns", "u-sweep", "s")


@pytest.mark.asyncio
async def test_patch_parent_status_404_returns_silently(monkeypatch):
    """Parent CR deleted between rollup and patch (404): handler returns None,
    no exception (the rollup is dead-letter and not retryable)."""
    from kubernetes_asyncio.client import ApiException

    _install_fake_k8s_for_rollup(
        monkeypatch,
        patch_status_side_effect=ApiException(status=404, reason="NotFound"),
    )
    # Must not raise.
    result = await child_rollup._patch_parent_status(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        plural="aiperfsweeps",
        name="missing",
        namespace="ns",
        body={"status": {"phase": "Aggregating"}},
    )
    assert result is None


@pytest.mark.asyncio
async def test_patch_parent_status_503_wraps_temporary_error(monkeypatch):
    """Patch on apiserver 503 must surface as kopf.TemporaryError."""
    from kubernetes_asyncio.client import ApiException

    _install_fake_k8s_for_rollup(
        monkeypatch,
        patch_status_side_effect=ApiException(status=503, reason="Unavailable"),
    )
    with pytest.raises(kopf.TemporaryError):
        await child_rollup._patch_parent_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            name="s",
            namespace="ns",
            body={"status": {"phase": "Aggregating"}},
        )


# ---------------------------------------------------------------------------
# lifecycle.on_delete — best-effort cooperative cancel
# ---------------------------------------------------------------------------


def _install_fake_k8s_for_lifecycle(
    monkeypatch,
    *,
    list_return=None,
    list_side_effect=None,
    patch_side_effect=None,
    delete_side_effect=None,
):
    """Install fake k8s_client + CustomObjectsApi for lifecycle handlers."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    list_mock = AsyncMock()
    if list_side_effect is not None:
        list_mock.side_effect = list_side_effect
    else:
        list_mock.return_value = list_return or {"items": []}

    patch_mock = AsyncMock()
    if patch_side_effect is not None:
        patch_mock.side_effect = patch_side_effect

    delete_mock = AsyncMock()
    if delete_side_effect is not None:
        delete_mock.side_effect = delete_side_effect

    custom = MagicMock()
    custom.list_namespaced_custom_object = list_mock
    custom.patch_namespaced_custom_object = patch_mock
    custom.delete_namespaced_custom_object = delete_mock

    fake_k8s_module = SimpleNamespace(CustomObjectsApi=lambda _api: custom)

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    import kubernetes_asyncio

    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module, raising=False)
    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    return list_mock, patch_mock, delete_mock


@pytest.mark.asyncio
async def test_on_delete_patches_each_child_with_cancel_true(monkeypatch):
    """`on_delete` lists children with the sweep label and patches each with
    spec.cancel=True for cooperative shutdown."""
    children = [
        {"metadata": {"name": "s-v00-t0"}},
        {"metadata": {"name": "s-v01-t0"}},
        {"metadata": {"name": "s-v02-t0"}},
    ]
    list_mock, patch_mock, _ = _install_fake_k8s_for_lifecycle(
        monkeypatch, list_return={"items": children}
    )
    await lifecycle.on_delete(name="s", namespace="ns")
    list_mock.assert_awaited_once()
    assert patch_mock.await_count == 3, (
        f"expected 3 patch calls, got {patch_mock.await_count}"
    )
    # Verify each call patches spec.cancel=True with the right name.
    patched_names = []
    for call in patch_mock.await_args_list:
        kwargs = call.kwargs
        assert kwargs["body"] == {"spec": {"cancel": True}}
        assert kwargs["_content_type"] == "application/merge-patch+json"
        assert kwargs["plural"] == "aiperfjobs"
        patched_names.append(kwargs["name"])
    assert sorted(patched_names) == ["s-v00-t0", "s-v01-t0", "s-v02-t0"]


@pytest.mark.asyncio
async def test_on_delete_swallows_apiexception_during_list(monkeypatch):
    """List fails with 503: on_delete must NOT raise (best-effort)."""
    from kubernetes_asyncio.client import ApiException

    _install_fake_k8s_for_lifecycle(
        monkeypatch,
        list_side_effect=ApiException(status=503, reason="Unavailable"),
    )
    # Must not raise.
    await lifecycle.on_delete(name="s", namespace="ns")


# ---------------------------------------------------------------------------
# lifecycle.maybe_reap_finished — TTL-based parent CR cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_reap_finished_terminal_age_exceeds_ttl_deletes(monkeypatch):
    """Succeeded sweep, completionTime 1h ago, ttl=1800: delete is invoked."""
    from datetime import datetime, timedelta

    one_hour_ago = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {"name": "s", "namespace": "ns"},
        "spec": {"ttlSecondsAfterFinished": 1800},
    }
    status = {"phase": "Succeeded", "completionTime": one_hour_ago}
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_awaited_once()
    kwargs = delete_mock.await_args.kwargs
    assert kwargs["plural"] == "aiperfsweeps"
    assert kwargs["name"] == "s"
    assert kwargs["namespace"] == "ns"


@pytest.mark.asyncio
async def test_maybe_reap_finished_terminal_age_below_ttl_does_not_delete(monkeypatch):
    """Succeeded sweep, completionTime 1h ago, ttl=7200: NOT yet eligible."""
    from datetime import datetime, timedelta

    one_hour_ago = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {"name": "s", "namespace": "ns"},
        "spec": {"ttlSecondsAfterFinished": 7200},
    }
    status = {"phase": "Succeeded", "completionTime": one_hour_ago}
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_reap_finished_handles_subsecond_completion_time(monkeypatch):
    """Sub-second RFC3339 `completionTime` must parse — not silently disable TTL.

    Pre-fix used `strptime("%Y-%m-%dT%H:%M:%SZ")` which rejected fractional
    seconds and `return`-ed silently from the reaper. The non-apiserver
    writers (sweep-controller `_now_iso` is whole-second today, but kopf
    bodies and JSON-patches may carry sub-second precision) would never
    reap.
    """
    from datetime import datetime, timedelta

    one_hour_ago_subsec = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.123456Z"
    )
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {"name": "s", "namespace": "ns"},
        "spec": {"ttlSecondsAfterFinished": 1800},
    }
    status = {"phase": "Succeeded", "completionTime": one_hour_ago_subsec}
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_reap_finished_non_terminal_phase_never_deletes(monkeypatch):
    """Pending phase + ttl=1: never reap, regardless of TTL."""
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {"name": "s", "namespace": "ns"},
        "spec": {"ttlSecondsAfterFinished": 1},
    }
    status = {"phase": "Pending", "completionTime": "2020-01-01T00:00:00Z"}
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_reap_finished_ttl_none_never_deletes(monkeypatch):
    """ttlSecondsAfterFinished unset: handler is a no-op."""
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {"metadata": {"name": "s", "namespace": "ns"}, "spec": {}}
    status = {"phase": "Succeeded", "completionTime": "2020-01-01T00:00:00Z"}
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Regression: TTL reaper must read CRD-declared `status.completionTime`,
# NOT the legacy/typo `status.completedAt`. Earlier code read `completedAt`
# but no writer ever populated it — TTL silently fell back to
# `metadata.creationTimestamp`, reaping long-running sweeps mid-flight when
# the user set a small ttlSecondsAfterFinished.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_reap_finished_reads_completion_time_not_completed_at(monkeypatch):
    """`status.completionTime` (CRD-declared) is the authoritative TTL anchor.

    With completionTime=1h-ago and ttl=1800, we MUST reap. If the reader
    were still looking at `status.completedAt`, the field would be missing,
    fall back to creationTimestamp=now, and never reap.
    """
    from datetime import datetime, timedelta

    one_hour_ago = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    now_iso = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {
            "name": "s",
            "namespace": "ns",
            "creationTimestamp": now_iso,
        },
        "spec": {"ttlSecondsAfterFinished": 1800},
    }
    status = {
        "phase": "Succeeded",
        "completionTime": one_hour_ago,
        # Stale legacy field; must be ignored.
        "completedAt": now_iso,
    }
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_reap_finished_falls_back_to_creation_timestamp(monkeypatch):
    """No completionTime present: fall back to metadata.creationTimestamp."""
    from datetime import datetime, timedelta

    one_hour_ago = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _list, _patch, delete_mock = _install_fake_k8s_for_lifecycle(monkeypatch)
    body = {
        "metadata": {"name": "s", "namespace": "ns", "creationTimestamp": one_hour_ago},
        "spec": {"ttlSecondsAfterFinished": 1800},
    }
    status = {"phase": "Succeeded"}  # no completionTime
    await lifecycle.maybe_reap_finished(
        body=body, status=status, name="s", namespace="ns"
    )
    delete_mock.assert_awaited_once()
