# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression-locking tests for AIPerfSweep handlers.

Locks in the recently-fixed bugs:

1. ``child_rollup.on_child_phase_transition``: ``PartiallyFailed`` is bucketed
   as ``failed``, not ``in_flight``; ``_read_parent_phase`` guards against
   clobbering terminal parent phases with ``Aggregating``; the API patch is
   sent with ``_content_type="application/merge-patch+json"``; the handler
   accepts arbitrary kopf-extra kwargs (``**_: Any``).
2. ``lifecycle.cancel``: terminal parent phase makes the handler a no-op even
   when ``spec.cancel=True``; ``spec.cancel=false`` clears any prior
   ``Cancelling`` condition (sticky-flag fix); the handler accepts arbitrary
   kopf-extra kwargs.
3. ``create.handle``: accepts arbitrary kopf-extra kwargs.
4. Sweep-identity fencing: a child only counts toward a parent when its
   controller ownerReference AND its sweep / sweep-uid / sweep-run-epoch
   labels all match, and the triggering child is re-read from the apiserver
   before any parent mutation so a delayed event from a previous run of a
   same-named sweep cannot corrupt the current one.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest
from aiperf.operator.handlers.sweep import child_rollup, lifecycle
from aiperf.operator.handlers.sweep import create as sweep_create
from pytest import param


@pytest.fixture(autouse=True)
def _hermetic_k8s_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to a stub ``k8s_client``.

    ``on_child_phase_transition`` opens ``k8s_client()`` itself (it holds one
    client for the whole rollup tick), so tests that monkeypatch only the inner
    ``_count_owned_children`` / ``_patch_parent_status`` / ``_read_parent_status``
    helpers still reach a live client-open, which falls through to
    ``load_kube_config()`` and depends on the developer's ``~/.kube/config``.
    Tests that assert on apiserver calls install their own fake via
    ``_install_fake_k8s`` — that ``monkeypatch.setattr`` re-patches over this
    default within the test body.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    import aiperf.kubernetes.client as kclient

    @asynccontextmanager
    async def _stub(*, kubeconfig: str | None = None, context: str | None = None):
        yield MagicMock(name="ApiClient")

    monkeypatch.setattr(kclient, "k8s_client", _stub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SWEEP_NAME = "s"
SWEEP_UID = "u-sweep"
RUN_EPOCH = "epoch-1"


def _owner_ref(*, uid: str = SWEEP_UID, name: str = SWEEP_NAME) -> dict[str, Any]:
    """Build the controller ownerReference the identity fence requires.

    ``_is_owned_child`` matches on all five of apiVersion / kind / name / uid /
    controller — a partial ref (the pre-fence shape) is treated as unowned.
    """
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfSweep",
        "name": name,
        "uid": uid,
        "controller": True,
    }


def _identity_labels(
    *, uid: str = SWEEP_UID, name: str = SWEEP_NAME, run_epoch: str = RUN_EPOCH
) -> dict[str, str]:
    """Build the sweep/sweep-uid/sweep-run-epoch label triple."""
    return {
        "aiperf.nvidia.com/sweep": name,
        "aiperf.nvidia.com/sweep-uid": uid,
        "aiperf.nvidia.com/sweep-run-epoch": run_epoch,
    }


def _child(
    phase: str | None,
    *,
    uid: str = SWEEP_UID,
    child_name: str = "child",
    child_uid: str = "child-uid",
    run_epoch: str = RUN_EPOCH,
) -> dict[str, Any]:
    """Build an AIPerfJob list-item that passes the owned-child identity fence."""
    return {
        "metadata": {
            "name": child_name,
            "uid": child_uid,
            "ownerReferences": [_owner_ref(uid=uid)],
            "labels": _identity_labels(uid=uid, run_epoch=run_epoch),
        },
        "status": ({"phase": phase} if phase is not None else {}),
    }


def _install_fake_k8s(
    monkeypatch: pytest.MonkeyPatch,
    *,
    list_items: list[dict[str, Any]] | None = None,
    parent_status: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Install a fake ``k8s_client``/``CustomObjectsApi`` pair on the kubernetes
    module that ``child_rollup`` imports lazily inside its helpers.

    Returns a SimpleNamespace exposing the captured ``patch_namespaced_custom_object_status``
    AsyncMock so tests can assert on call kwargs.
    """
    list_items = list_items or []
    parent_status = parent_status or {}

    captured = SimpleNamespace(
        patch_status=AsyncMock(return_value=None),
        list_objs=AsyncMock(return_value={"items": list_items}),
        get_obj=AsyncMock(return_value={"status": parent_status}),
    )

    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = captured.patch_status
    custom.list_namespaced_custom_object = captured.list_objs
    custom.get_namespaced_custom_object = captured.get_obj

    fake_k8s_module = SimpleNamespace(CustomObjectsApi=lambda _api: custom)

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    # Patch the modules that the handler imports lazily inside functions.
    import kubernetes_asyncio

    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module, raising=False)
    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    return captured


# ---------------------------------------------------------------------------
# A) child_rollup — PartiallyFailed regression-lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_owned_children_partially_failed_buckets_as_failed_not_in_flight(
    monkeypatch,
):
    """A PartiallyFailed child must count as ``failed`` (not ``in_flight``)."""
    fake = _install_fake_k8s(
        monkeypatch,
        list_items=[
            _child("PartiallyFailed"),
            _child("Succeeded"),
            _child("Succeeded"),
        ],
    )
    result = await child_rollup._count_owned_children(
        "ns", "u-sweep", "s", run_epoch="epoch-1"
    )
    assert result["failed"] == 1, "PartiallyFailed must bucket as failed"
    assert result["in_flight"] == 0, "PartiallyFailed must NOT bucket as in_flight"
    assert result["completed"] == 2
    # All terminal — Aggregating should be computed.
    assert result["total_terminal_phase"] == "Aggregating"
    fake.list_objs.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_owned_children_partially_failed_alone_yields_aggregating(
    monkeypatch,
):
    """A lone PartiallyFailed child with no in-flight peers triggers Aggregating."""
    _install_fake_k8s(monkeypatch, list_items=[_child("PartiallyFailed")])
    result = await child_rollup._count_owned_children(
        "ns", "u-sweep", "s", run_epoch="epoch-1"
    )
    assert result["in_flight"] == 0
    assert result["failed"] == 1
    assert result["total_terminal_phase"] == "Aggregating"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        param(
            lambda c: c["metadata"]["labels"].__setitem__(
                "aiperf.nvidia.com/sweep-run-epoch", "epoch-0"
            ),
            id="stale_run_epoch",
        ),
        param(
            lambda c: c["metadata"]["labels"].__setitem__(
                "aiperf.nvidia.com/sweep-uid", "u-previous"
            ),
            id="previous_sweep_uid",
        ),
        param(
            lambda c: c["metadata"]["ownerReferences"][0].__setitem__(
                "controller", False
            ),
            id="non_controller_owner_ref",
        ),
        param(
            lambda c: c["metadata"]["ownerReferences"][0].__setitem__(
                "apiVersion", "batch/v1"
            ),
            id="foreign_api_version",
        ),
        param(lambda c: c["metadata"].pop("uid"), id="missing_child_uid"),
    ],
)
async def test_count_owned_children_rejects_children_failing_identity_fence(
    monkeypatch, mutate
):
    """Label selectors are advisory; the in-process fence is authoritative.

    A recreated same-named sweep reuses the label values it can (``sweep``),
    so the tally must independently re-verify the immutable owner UID, the
    controller ownerReference, and the run epoch on every listed item.
    """
    stale = _child("Succeeded", child_uid="stale-uid")
    mutate(stale)
    _install_fake_k8s(monkeypatch, list_items=[stale, _child("Succeeded")])
    result = await child_rollup._count_owned_children(
        "ns", SWEEP_UID, SWEEP_NAME, run_epoch=RUN_EPOCH
    )
    assert result["completed"] == 1, f"unowned child leaked into tally: {result}"
    assert len(result["owned_children"]) == 1


# ---------------------------------------------------------------------------
# B) child_rollup — terminal-phase clobber guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_phase",
    [
        param("Succeeded", id="succeeded"),
        param("Failed", id="failed"),
        param("Cancelled", id="cancelled"),
        param("PartiallyFailed", id="partially_failed"),
    ],
)
async def test_on_child_phase_transition_does_not_clobber_terminal_parent_phase(
    monkeypatch, parent_phase
):
    """If parent is already terminal, the whole status merge-patch is skipped —
    no phase write and no count write.

    Once the sweep-controller has written authoritative counts at terminal time
    and the children have been harvested and deleted, ``_count_owned_children``
    sees zero, so patching counts would overwrite the authoritative values with
    zeros. The conditional phase setter must not fire either.
    """
    captured: list[dict[str, Any]] = []
    phase_calls: list[dict[str, Any]] = []

    async def fake_patch(*, body, **_kw):
        captured.append(body)

    async def fake_phase_set(**kwargs):
        phase_calls.append(kwargs)

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
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
    monkeypatch.setattr(
        child_rollup,
        "_read_parent_status",
        AsyncMock(return_value={"phase": parent_phase, "maxTotalRuns": 3}),
    )
    monkeypatch.setattr(child_rollup, "_ingest_sweep_aggregate", AsyncMock())

    body = _child("Succeeded")
    monkeypatch.setattr(
        child_rollup, "_read_current_child", AsyncMock(return_value=body)
    )
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    assert captured == [], (
        f"no status patch may be sent when parent is terminal ({parent_phase})"
    )
    # And the conditional phase setter must not have been invoked.
    assert phase_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_phase",
    [
        param("Pending", id="pending"),
        param(None, id="missing"),
        param("Running", id="running"),
        param("", id="empty_string"),
    ],
)
async def test_on_child_phase_transition_writes_aggregating_when_parent_non_terminal(
    monkeypatch, parent_phase
):
    """Non-terminal parent phase → ``Aggregating`` is written through
    ``_conditional_phase_set`` with the read-back phase as the test value."""
    captured: list[dict[str, Any]] = []
    phase_calls: list[dict[str, Any]] = []

    async def fake_patch(*, body, **_kw):
        captured.append(body)

    async def fake_phase_set(**kwargs):
        phase_calls.append(kwargs)

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 1,
                "failed": 0,
                "in_flight": 0,
                "total_terminal_phase": "Aggregating",
            }
        ),
    )
    monkeypatch.setattr(
        child_rollup,
        "_read_parent_status",
        AsyncMock(return_value={"phase": parent_phase, "maxTotalRuns": 1}),
    )

    body = _child("Succeeded", child_name="c")
    monkeypatch.setattr(
        child_rollup, "_read_current_child", AsyncMock(return_value=body)
    )
    await child_rollup.on_child_phase_transition(
        body=body, status={"phase": "Succeeded"}, name="c", namespace="ns"
    )
    # Counts ride the merge-patch; phase rides the conditional setter.
    assert "phase" not in captured[0]["status"]
    assert len(phase_calls) == 1
    assert phase_calls[0]["new_phase"] == "Aggregating"
    assert phase_calls[0]["expect_phase"] == (parent_phase or "")


# ---------------------------------------------------------------------------
# C) child_rollup — content-type regression-lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_parent_status_uses_merge_patch_content_type(monkeypatch):
    """``_patch_parent_status`` must call the API with merge-patch content-type.

    Locks the SSA revert: full Server-Side Apply was tried and reverted
    because relinquishment semantics drop a manager's previously-set
    fields between calls when the new apply body doesn't include them.
    Merge-patch is the right primitive for the imperative event-style
    rollup writes; ``field_manager`` here is observability metadata only.
    """
    fake = _install_fake_k8s(monkeypatch)

    await child_rollup._patch_parent_status(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        plural="aiperfsweeps",
        name="s",
        namespace="ns",
        body={"status": {"phase": "Aggregating"}},
    )

    fake.patch_status.assert_awaited_once()
    kwargs = fake.patch_status.await_args.kwargs
    assert kwargs.get("_content_type") == "application/merge-patch+json", (
        "patch must be sent as merge-patch+json (json-patch+json would expect a list of ops)"
    )
    assert kwargs.get("field_manager") == child_rollup.ROLLUP_FIELD_MANAGER, (
        "rollup must tag its writes with a distinct field_manager so kubectl "
        "shows which writer last touched each field"
    )
    # SSA-only kwargs must NOT leak in — they would change semantics.
    assert "force" not in kwargs or kwargs["force"] is None, (
        f"force={kwargs.get('force')!r} leaked from the SSA experiment; "
        "merge-patch must not pass force=True"
    )
    assert kwargs.get("name") == "s"
    assert kwargs.get("namespace") == "ns"
    assert kwargs.get("plural") == "aiperfsweeps"
    # Body is the merge-patch shape (no apiVersion/kind/metadata envelope).
    body = kwargs.get("body")
    assert body == {"status": {"phase": "Aggregating"}}, (
        f"merge-patch body must be the bare status diff, got {body!r}"
    )


# ---------------------------------------------------------------------------
# D) child_rollup — kwargs-trap regression-lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_child_phase_transition_accepts_kopf_extra_kwargs(monkeypatch):
    """Kopf passes a fixed kwarg set; handler must absorb extras via ``**_: Any``."""
    monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
    monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
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
    monkeypatch.setattr(
        child_rollup, "_read_parent_status", AsyncMock(return_value=None)
    )

    body = _child("Succeeded", child_name="c")
    monkeypatch.setattr(
        child_rollup, "_read_current_child", AsyncMock(return_value=body)
    )
    # Pass kopf-extra kwargs that are NOT in the handler signature.
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="c",
        namespace="ns",
        logger=MagicMock(),
        memo=MagicMock(),
        meta={"resourceVersion": "42"},
        retry=0,
        patch=kopf.Patch(),
        uid="child-uid",
        diff=(),
        old=None,
        new="Succeeded",
    )


# ---------------------------------------------------------------------------
# E) child_rollup — non-sweep child is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_child_phase_transition_no_owner_refs_is_noop(monkeypatch):
    patch_parent = AsyncMock()
    count_children = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(child_rollup, "_count_owned_children", count_children)

    body = {"metadata": {"name": "child", "ownerReferences": []}}
    await child_rollup.on_child_phase_transition(
        body=body, status={"phase": "Succeeded"}, name="child", namespace="ns"
    )
    patch_parent.assert_not_awaited()
    count_children.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_child_phase_transition_owner_kind_not_aiperfsweep_is_noop(
    monkeypatch,
):
    patch_parent = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(child_rollup, "_count_owned_children", AsyncMock())
    body = {
        "metadata": {
            "name": "child",
            "ownerReferences": [{"kind": "JobSet", "name": "jobset", "uid": "x"}],
        }
    }
    await child_rollup.on_child_phase_transition(
        body=body, status={"phase": "Succeeded"}, name="child", namespace="ns"
    )
    patch_parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_child_phase_transition_malformed_owner_ref_missing_uid_is_noop(
    monkeypatch,
):
    """An AIPerfSweep ownerRef missing ``uid`` must not crash; treat as no-op."""
    patch_parent = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(child_rollup, "_count_owned_children", AsyncMock())
    body = {
        "metadata": {
            "name": "child",
            "ownerReferences": [{"kind": "AIPerfSweep", "name": "s"}],  # no uid
        }
    }
    # Must NOT raise.
    await child_rollup.on_child_phase_transition(
        body=body, status={"phase": "Succeeded"}, name="child", namespace="ns"
    )
    patch_parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_child_phase_transition_missing_run_epoch_label_is_noop(monkeypatch):
    """No sweep-run-epoch label → the tick cannot be attributed to a run.

    Supersedes the pre-fence behaviour where a missing label degraded to an
    unscoped ``run_epoch=None`` count across every epoch of the sweep.
    """
    patch_parent = AsyncMock()
    count_children = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(child_rollup, "_count_owned_children", count_children)

    body = _child("Succeeded")
    body["metadata"]["labels"].pop("aiperf.nvidia.com/sweep-run-epoch")
    await child_rollup.on_child_phase_transition(
        body=body, status={"phase": "Succeeded"}, name="child", namespace="ns"
    )
    patch_parent.assert_not_awaited()
    count_children.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_child_phase_transition_vanished_child_skips_parent_mutation(
    monkeypatch,
):
    """The child is re-read before any parent write; a 404 aborts the tick."""
    patch_parent = AsyncMock()
    count_children = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(child_rollup, "_count_owned_children", count_children)
    monkeypatch.setattr(
        child_rollup, "_read_current_child", AsyncMock(return_value=None)
    )

    await child_rollup.on_child_phase_transition(
        body=_child("Succeeded"),
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    patch_parent.assert_not_awaited()
    count_children.assert_not_awaited()


# ---------------------------------------------------------------------------
# F) lifecycle.cancel — terminal-phase guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        param("Succeeded", id="succeeded"),
        param("Failed", id="failed"),
        param("Cancelled", id="cancelled"),
        param("PartiallyFailed", id="partially_failed"),
    ],
)
async def test_cancel_handler_terminal_parent_phase_is_noop(phase):
    """A terminal parent ignores Cancelling but acknowledges the new generation."""
    patch = kopf.Patch()
    body = {
        "metadata": {"name": "s", "generation": 17},
        "status": {"phase": phase},
    }
    await lifecycle.cancel(
        body=body,
        spec={"cancel": True},
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert "conditions" not in patch.status, (
        f"cancel on terminal phase {phase} must not touch patch.status['conditions']"
    )
    assert patch.status["observedGeneration"] == 17


# ---------------------------------------------------------------------------
# G) lifecycle.cancel — toggle off clears the condition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_handler_clears_cancelling_when_spec_toggles_false():
    """spec.cancel=False with a prior Cancelling condition clears the sticky flag."""
    patch = kopf.Patch()
    body = {
        "metadata": {"name": "s"},
        "status": {
            "phase": "Running",
            "conditions": [
                {
                    "type": "Cancelling",
                    "status": "True",
                    "reason": "UserRequested",
                    "lastTransitionTime": "2026-04-26T00:00:00Z",
                },
                {"type": "Other", "status": "False"},
            ],
        },
    }
    await lifecycle.cancel(
        body=body, spec={"cancel": False}, name="s", namespace="ns", patch=patch
    )
    conditions = patch.status.get("conditions")
    assert conditions is not None, "patch must clear the Cancelling condition"
    assert all(c.get("type") != "Cancelling" for c in conditions)
    # The non-Cancelling condition is preserved.
    assert any(c.get("type") == "Other" for c in conditions)


@pytest.mark.asyncio
async def test_cancel_handler_no_op_when_no_cancelling_and_spec_false():
    """spec.cancel=False with no prior Cancelling condition: handler must do nothing."""
    patch = kopf.Patch()
    body = {
        "metadata": {"name": "s"},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Other", "status": "False"}],
        },
    }
    await lifecycle.cancel(
        body=body, spec={"cancel": False}, name="s", namespace="ns", patch=patch
    )
    assert "conditions" not in patch.status


# ---------------------------------------------------------------------------
# H) lifecycle.cancel — toggle on appends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_handler_running_phase_appends_cancelling_condition():
    patch = kopf.Patch()
    body = {"metadata": {"name": "s"}, "status": {"phase": "Running"}}
    await lifecycle.cancel(
        body=body, spec={"cancel": True}, name="s", namespace="ns", patch=patch
    )
    conditions = patch.status["conditions"]
    cancellings = [c for c in conditions if c.get("type") == "Cancelling"]
    assert len(cancellings) == 1
    cond = cancellings[0]
    assert cond["status"] == "True"
    assert cond.get("lastTransitionTime"), "lastTransitionTime must be set"
    assert cond["lastTransitionTime"] != ""


@pytest.mark.asyncio
async def test_cancel_handler_dedupes_stale_cancelling_condition():
    """Re-firing with cancel=True against an existing Cancelling=True condition
    leaves exactly one Cancelling entry (no duplicate)."""
    patch = kopf.Patch()
    body = {
        "metadata": {"name": "s"},
        "status": {
            "phase": "Running",
            "conditions": [
                {
                    "type": "Cancelling",
                    "status": "True",
                    "reason": "Previous",
                    "lastTransitionTime": "2026-04-25T00:00:00Z",
                },
            ],
        },
    }
    await lifecycle.cancel(
        body=body, spec={"cancel": True}, name="s", namespace="ns", patch=patch
    )
    conditions = patch.status["conditions"]
    cancellings = [c for c in conditions if c.get("type") == "Cancelling"]
    assert len(cancellings) == 1, "exactly one Cancelling condition must remain"


# ---------------------------------------------------------------------------
# I) kwargs-trap regression-lock for lifecycle + create handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_handler_accepts_kopf_extra_kwargs():
    patch = kopf.Patch()
    await lifecycle.cancel(
        body={"metadata": {"name": "s"}, "status": {"phase": "Running"}},
        spec={"cancel": True},
        name="s",
        namespace="ns",
        patch=patch,
        logger=MagicMock(),
        memo=MagicMock(),
        meta={"resourceVersion": "1"},
        retry=0,
        uid="sweep-uid",
        diff=(),
        old=None,
        new={"cancel": True},
    )


@pytest.mark.asyncio
async def test_create_handle_accepts_kopf_extra_kwargs(monkeypatch):
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    body = {
        "metadata": {
            "name": "s",
            "namespace": "ns",
            "uid": "u",
            "creationTimestamp": "2024-04-25T18:22:03Z",
        },
        "spec": {
            "multiRun": {"numRuns": 1},
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1]},
            },
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
        },
    }
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=kopf.Patch(),
        logger=MagicMock(),
        memo=MagicMock(),
        meta={"resourceVersion": "1"},
        retry=0,
        uid="u",
        diff=(),
        old=None,
        new=body["spec"],
    )


# ---------------------------------------------------------------------------
# J) Property-style coverage — exhaustive bucketing
# ---------------------------------------------------------------------------


_PHASES = [
    "Succeeded",
    "Completed",
    "Failed",
    "Cancelled",
    "PartiallyFailed",
    "Running",
    "Pending",
    "",
    None,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase_pair",
    [
        param(p, id=f"{p[0] or 'none'}__{p[1] or 'none'}")
        for p in product(_PHASES, _PHASES)
    ],
)
async def test_count_owned_children_exhaustive_bucketing_no_loss(
    monkeypatch, phase_pair
):
    """For every pair of child phases, counts.{completed,failed,cancelled,in_flight}
    must sum to the total number of children — no child is dropped or double-counted."""
    items = [_child(phase_pair[0]), _child(phase_pair[1])]
    _install_fake_k8s(monkeypatch, list_items=items)
    result = await child_rollup._count_owned_children(
        "ns", "u-sweep", "s", run_epoch="epoch-1"
    )
    total = (
        result["completed"]
        + result["failed"]
        + result["cancelled"]
        + result["in_flight"]
    )
    assert total == len(items), (
        f"phases {phase_pair}: counts {result} sum to {total}, expected {len(items)}"
    )
