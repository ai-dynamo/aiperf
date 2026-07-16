# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.handlers.sweep.child_rollup``.

Covers the kopf rollup entry point ``on_child_phase_transition`` plus the
package-private helpers ``_find_sweep_owner``, ``_count_owned_children``,
``_patch_parent_status``, ``_read_parent_status``, ``_read_parent_phase``,
``_conditional_phase_set``, ``_ingest_sweep_aggregate``, and ``_api_or_new``.

Mocking strategy:
- ``aiperf.kubernetes.client.k8s_client`` is patched to an ``@asynccontextmanager``
  that yields a MagicMock ApiClient — no real apiserver socket is opened.
- ``kubernetes_asyncio.client.CustomObjectsApi`` is replaced with a factory that
  returns a MagicMock with AsyncMock-backed list/get/patch methods, so we can
  control returns and side-effects per test.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
import orjson
import pytest
from pytest import param

from aiperf.operator.handlers.sweep import child_rollup

# ============================================================
# Shared k8s mocking helper
# ============================================================


def _install_fake_k8s(
    monkeypatch: pytest.MonkeyPatch,
    *,
    list_return: dict[str, Any] | None = None,
    list_side_effect: BaseException | None = None,
    get_return: dict[str, Any] | None = None,
    get_side_effect: BaseException | None = None,
    patch_status_side_effect: BaseException | None = None,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Install fake ``k8s_client()`` and ``CustomObjectsApi`` for child_rollup.

    Returns ``(list_mock, get_mock, patch_status_mock)`` so individual tests can
    assert call counts and inspect kwargs.
    """
    list_mock = AsyncMock()
    if list_side_effect is not None:
        list_mock.side_effect = list_side_effect
    else:
        list_mock.return_value = (
            list_return if list_return is not None else {"items": []}
        )

    get_mock = AsyncMock()
    if get_side_effect is not None:
        get_mock.side_effect = get_side_effect
    else:
        get_mock.return_value = get_return if get_return is not None else {}

    patch_mock = AsyncMock()
    if patch_status_side_effect is not None:
        patch_mock.side_effect = patch_status_side_effect

    custom = MagicMock()
    custom.list_namespaced_custom_object = list_mock
    custom.get_namespaced_custom_object = get_mock
    custom.patch_namespaced_custom_object_status = patch_mock

    fake_k8s_module = SimpleNamespace(CustomObjectsApi=lambda _api: custom)

    fresh_client_opened = MagicMock(name="freshApiClient")

    @asynccontextmanager
    async def fake_k8s_client():
        yield fresh_client_opened

    import kubernetes_asyncio

    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module, raising=False)
    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)
    return list_mock, get_mock, patch_mock


def _child_body(
    *,
    owner_kind: str | None = "AIPerfSweep",
    owner_name: str | None = "s",
    owner_uid: str | None = "u",
    run_epoch: str | None = None,
    drop_owner_refs: bool = False,
) -> dict[str, Any]:
    """Build a minimal child AIPerfJob body for ``on_child_phase_transition``."""
    metadata: dict[str, Any] = {"name": "child", "namespace": "ns"}
    if not drop_owner_refs:
        ref: dict[str, Any] = {}
        if owner_kind is not None:
            ref["kind"] = owner_kind
        if owner_name is not None:
            ref["name"] = owner_name
        if owner_uid is not None:
            ref["uid"] = owner_uid
        metadata["ownerReferences"] = [ref] if ref else []
    if run_epoch is not None:
        metadata["labels"] = {"aiperf.nvidia.com/sweep-run-epoch": run_epoch}
    return {"metadata": metadata}


# ============================================================
# _find_sweep_owner
# ============================================================


class TestFindSweepOwner:
    """Verify owner-reference filtering."""

    def test_returns_name_and_uid_for_aiperfsweep_owner(self) -> None:
        body = _child_body(owner_kind="AIPerfSweep", owner_name="s", owner_uid="u-1")
        assert child_rollup._find_sweep_owner(body) == ("s", "u-1")

    @pytest.mark.parametrize(
        "body",
        [
            param({"metadata": {"ownerReferences": []}}, id="empty-refs"),
            param({"metadata": {}}, id="missing-refs-key"),
            param({}, id="missing-metadata"),
            param({"metadata": {"ownerReferences": None}}, id="null-refs"),
            param({"metadata": None}, id="null-metadata"),
        ],
    )  # fmt: skip
    def test_returns_none_when_no_owner_refs(self, body: dict[str, Any]) -> None:
        assert child_rollup._find_sweep_owner(body) is None

    def test_returns_none_when_kind_does_not_match(self) -> None:
        body = _child_body(owner_kind="Job", owner_name="s", owner_uid="u")
        assert child_rollup._find_sweep_owner(body) is None

    def test_returns_none_when_uid_missing(self) -> None:
        body = _child_body(owner_kind="AIPerfSweep", owner_name="s", owner_uid=None)
        assert child_rollup._find_sweep_owner(body) is None

    def test_returns_none_when_name_missing(self) -> None:
        body = _child_body(owner_kind="AIPerfSweep", owner_name=None, owner_uid="u")
        assert child_rollup._find_sweep_owner(body) is None

    def test_picks_first_matching_aiperfsweep_among_mixed_owners(self) -> None:
        body = {
            "metadata": {
                "ownerReferences": [
                    {"kind": "Job", "name": "j", "uid": "u-job"},
                    {"kind": "AIPerfSweep", "name": "swp-1", "uid": "u-swp-1"},
                    {"kind": "AIPerfSweep", "name": "swp-2", "uid": "u-swp-2"},
                ]
            }
        }
        assert child_rollup._find_sweep_owner(body) == ("swp-1", "u-swp-1")


# ============================================================
# on_child_phase_transition
# ============================================================


class TestOnChildPhaseTransition:
    """Verify the kopf entry point's dispatch logic."""

    @pytest.fixture(autouse=True)
    def _stub_append_run_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the runs[] append helper so terminal-phase tests don't await
        on the real two-step JSON-patch sequence (covered separately below)."""
        monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())

    @pytest.mark.asyncio
    async def test_standalone_child_no_sweep_owner_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No AIPerfSweep ownerRef → must not open k8s_client or call helpers."""
        opened = {"count": 0}

        @asynccontextmanager
        async def fake_k8s_client():
            opened["count"] += 1
            yield MagicMock()

        import aiperf.kubernetes.client as kclient

        monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)
        # Wire helpers so any accidental call would be detectable.
        count_mock = AsyncMock()
        patch_mock = AsyncMock()
        monkeypatch.setattr(child_rollup, "_count_owned_children", count_mock)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_mock)

        await child_rollup.on_child_phase_transition(
            body=_child_body(drop_owner_refs=True),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        assert opened["count"] == 0
        count_mock.assert_not_awaited()
        patch_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owned_child_patches_parent_with_counts_and_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owned child → counts + lastChildEvent on the merge-patch."""
        captured: dict[str, Any] = {}

        async def fake_count(
            namespace: str,
            sweep_uid: str,
            sweep_name: str,
            *,
            run_epoch: str | None = None,
            api: Any = None,
        ) -> dict[str, Any]:
            captured["count_kwargs"] = {
                "namespace": namespace,
                "sweep_uid": sweep_uid,
                "sweep_name": sweep_name,
                "run_epoch": run_epoch,
            }
            return {
                "pending": 1,
                "running": 2,
                "completed": 2,
                "failed": 1,
                "cancelled": 0,
                "in_flight": 3,
                "total_terminal_phase": None,
            }

        async def fake_patch(
            *, group, version, plural, name, namespace, body, api=None
        ):
            captured["patch_body"] = body
            captured["patch_name"] = name

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
        # Should not be reached (in_flight > 0).
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Failed"},
            name="child-A",
            namespace="ns",
        )
        assert captured["patch_name"] == "s"
        body_patch = captured["patch_body"]["status"]
        assert body_patch["completedRuns"] == 2
        assert body_patch["failedRuns"] == 1
        assert body_patch["runStates"] == {
            "pending": 1,
            "running": 2,
            "completed": 2,
            "failed": 1,
            "cancelled": 0,
        }
        assert body_patch["lastChildEvent"] == {"name": "child-A", "phase": "Failed"}

    @pytest.mark.asyncio
    async def test_owned_child_rollup_restamps_apiurl_from_current_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every rollup tick re-stamps ``status.apiUrl`` so AIPerfSweep CRs
        created before the URL-collapse cleanup self-heal post-upgrade —
        without this, an in-flight CR's stamped ``http://...:8080/api/v1/sweeps/...``
        from a pre-collapse install would persist forever (404 in production
        because the operator container has no FastAPI on 8080).
        """
        from aiperf.operator.environment import OperatorEnvironment

        captured: dict[str, Any] = {}

        async def fake_count(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {
                "pending": 0,
                "running": 1,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 1,
                "total_terminal_phase": None,
            }

        async def fake_patch(*, body, name, **_kw: Any) -> None:
            captured["patch_body"] = body

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)
        monkeypatch.setattr(
            OperatorEnvironment.SERVICE,
            "BASE_URL",
            "https://op.override.example:9091/",  # trailing slash on purpose
        )

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Running"},
            name="child-A",
            namespace="ns",
        )

        body_patch = captured["patch_body"]["status"]
        # rstrip("/") prevents a `//api/v1/...` leak.
        assert (
            body_patch["apiUrl"]
            == "https://op.override.example:9091/api/v1/sweeps/ns/s"
        )

    @pytest.mark.asyncio
    async def test_owned_child_unknown_phase_records_unknown_in_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """status.phase missing on the trigger → lastChildEvent.phase = 'Unknown'."""
        captured: dict[str, Any] = {}

        async def fake_patch(
            *, group, version, plural, name, namespace, body, api=None
        ):
            captured["body"] = body

        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "completed": 0,
                    "failed": 0,
                    "in_flight": 1,
                    "total_terminal_phase": None,
                }
            ),
        )
        monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={},
            name="child",
            namespace="ns",
        )
        assert captured["body"]["status"]["lastChildEvent"]["phase"] == "Unknown"

    @pytest.mark.asyncio
    async def test_run_epoch_label_propagates_to_count_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Child with epoch label → _count_owned_children receives run_epoch."""
        captured: dict[str, Any] = {}

        async def fake_count(
            namespace, sweep_uid, sweep_name, *, run_epoch=None, api=None
        ):
            captured["run_epoch"] = run_epoch
            return {
                "completed": 0,
                "failed": 0,
                "in_flight": 1,
                "total_terminal_phase": None,
            }

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(run_epoch="epoch-7"),
            status={"phase": "Running"},
            name="child",
            namespace="ns",
        )
        assert captured["run_epoch"] == "epoch-7"

    @pytest.mark.asyncio
    async def test_missing_run_epoch_label_propagates_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_count(
            namespace, sweep_uid, sweep_name, *, run_epoch=None, api=None
        ):
            captured["run_epoch"] = run_epoch
            return {
                "completed": 0,
                "failed": 0,
                "in_flight": 1,
                "total_terminal_phase": None,
            }

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Running"},
            name="child",
            namespace="ns",
        )
        assert captured["run_epoch"] is None

    @pytest.mark.asyncio
    async def test_in_flight_nonzero_skips_phase_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """in_flight > 0 → no phase-set, no parent-status read."""
        phase_set = AsyncMock()
        read_status = AsyncMock()
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", phase_set)
        monkeypatch.setattr(child_rollup, "_read_parent_status", read_status)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "completed": 1,
                    "failed": 0,
                    "in_flight": 2,
                    "total_terminal_phase": None,
                }
            ),
        )
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        phase_set.assert_not_awaited()
        read_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parent_already_terminal_calls_ingest_and_skips_phase_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All-children-terminal AND parent in PARENT_TERMINAL_PHASES →
        ingest sweep aggregate, do NOT call _conditional_phase_set."""
        ingest_mock = AsyncMock()
        phase_set = AsyncMock()
        monkeypatch.setattr(child_rollup, "_ingest_sweep_aggregate", ingest_mock)
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", phase_set)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": "Succeeded", "maxTotalRuns": 3}),
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
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        ingest_mock.assert_awaited_once_with("ns", "s")
        phase_set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accounted_below_max_total_runs_skips_phase_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All listed children terminal but accounted < maxTotalRuns →
        return without flipping phase (sweep-controller still creating cells)."""
        phase_set = AsyncMock()
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", phase_set)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 10}),
        )
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "completed": 2,
                    "failed": 1,
                    "in_flight": 0,
                    "total_terminal_phase": "Aggregating",
                }
            ),
        )
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        phase_set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accounted_meets_max_total_runs_calls_phase_set_aggregating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """accounted == maxTotalRuns → flip to Aggregating with TOCTOU guard."""
        phase_calls: list[dict[str, Any]] = []

        async def fake_phase_set(*, namespace, name, expect_phase, new_phase, api=None):
            phase_calls.append(
                {
                    "namespace": namespace,
                    "name": name,
                    "expect": expect_phase,
                    "new": new_phase,
                }
            )

        monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 4}),
        )
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "completed": 3,
                    "failed": 1,
                    "in_flight": 0,
                    "total_terminal_phase": "Aggregating",
                }
            ),
        )
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        assert phase_calls == [
            {"namespace": "ns", "name": "s", "expect": "Running", "new": "Aggregating"}
        ]

    @pytest.mark.asyncio
    async def test_empty_parent_phase_falls_through_to_phase_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """parent phase empty (initial create) and no maxTotalRuns gate set → fall through.

        With ``max_total_runs`` missing/zero, the int-isinstance gate doesn't
        block, so _conditional_phase_set fires with expect_phase="" and the
        helper itself handles the empty-expect fallback.
        """
        phase_calls: list[dict[str, Any]] = []

        async def fake_phase_set(*, namespace, name, expect_phase, new_phase, api=None):
            phase_calls.append({"expect": expect_phase, "new": new_phase})

        monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": ""}),
        )
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
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Succeeded"},
            name="child",
            namespace="ns",
        )
        assert phase_calls == [{"expect": "", "new": "Aggregating"}]

    @pytest.mark.asyncio
    async def test_no_total_terminal_phase_short_circuits_before_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """total_terminal_phase=None → return before reading parent status."""
        read_status = AsyncMock()
        phase_set = AsyncMock()
        monkeypatch.setattr(child_rollup, "_read_parent_status", read_status)
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", phase_set)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
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
        _install_fake_k8s(monkeypatch)

        await child_rollup.on_child_phase_transition(
            body=_child_body(),
            status={"phase": "Pending"},
            name="child",
            namespace="ns",
        )
        read_status.assert_not_awaited()
        phase_set.assert_not_awaited()


# ============================================================
# _count_owned_children
# ============================================================


class TestCountOwnedChildren:
    """Verify selector construction, owner-uid filtering, and phase tallies."""

    @pytest.mark.asyncio
    async def test_tallies_phases_into_completed_failed_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase strings tally into pending/running/completed/failed/cancelled buckets."""
        items = [
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Succeeded"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Completed"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Failed"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Cancelled"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "PartiallyFailed"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Running"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u"}]},
                "status": {"phase": "Pending"},
            },
            {"metadata": {"ownerReferences": [{"uid": "u"}]}, "status": {}},
            {"metadata": {"ownerReferences": [{"uid": "u"}]}},
        ]
        list_mock, _, _ = _install_fake_k8s(monkeypatch, list_return={"items": items})

        result = await child_rollup._count_owned_children("ns", "u", "s")
        assert result["completed"] == 2
        assert result["failed"] == 2
        assert result["cancelled"] == 1
        assert result["pending"] == 3
        assert result["running"] == 1
        assert result["in_flight"] == 4
        assert result["total_terminal_phase"] is None  # in_flight > 0
        # Selector: by default no run_epoch, so no epoch suffix.
        kwargs = list_mock.await_args.kwargs
        assert kwargs["label_selector"] == "aiperf.nvidia.com/sweep=s"
        assert kwargs["plural"] == "aiperfjobs"

    @pytest.mark.asyncio
    async def test_filters_out_items_with_mismatched_owner_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Items whose ownerReferences uid does not match must be skipped entirely."""
        items = [
            {
                "metadata": {"ownerReferences": [{"uid": "u-correct"}]},
                "status": {"phase": "Succeeded"},
            },
            {
                "metadata": {"ownerReferences": [{"uid": "u-OTHER"}]},
                "status": {"phase": "Succeeded"},
            },
            {"metadata": {"ownerReferences": []}, "status": {"phase": "Succeeded"}},
            {"metadata": {}, "status": {"phase": "Failed"}},
        ]
        _install_fake_k8s(monkeypatch, list_return={"items": items})

        result = await child_rollup._count_owned_children("ns", "u-correct", "s")
        assert result["completed"] == 1
        assert result["failed"] == 0
        assert result["in_flight"] == 0
        assert result["total_terminal_phase"] == "Aggregating"

    @pytest.mark.asyncio
    async def test_total_zero_yields_none_terminal_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(monkeypatch, list_return={"items": []})
        result = await child_rollup._count_owned_children("ns", "u", "s")
        assert result == {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
            "total_terminal_phase": None,
            "owned_children": [],
        }

    @pytest.mark.asyncio
    async def test_run_epoch_appended_to_label_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        list_mock, _, _ = _install_fake_k8s(monkeypatch, list_return={"items": []})
        await child_rollup._count_owned_children("ns", "u", "s", run_epoch="epoch-9")
        kwargs = list_mock.await_args.kwargs
        assert (
            kwargs["label_selector"]
            == "aiperf.nvidia.com/sweep=s,aiperf.nvidia.com/sweep-run-epoch=epoch-9"
        )

    @pytest.mark.asyncio
    async def test_apiexception_wraps_in_temporary_error_with_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch, list_side_effect=ApiException(status=500, reason="Internal")
        )
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._count_owned_children("ns", "u", "s")
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            param(ConnectionError("refused"), id="connection-error"),
            param(TimeoutError("slow apiserver"), id="timeout-error"),
        ],
    )  # fmt: skip
    async def test_network_errors_wrap_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        _install_fake_k8s(monkeypatch, list_side_effect=exc)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._count_owned_children("ns", "u", "s")
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    async def test_aiohttp_client_error_wraps_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiohttp

        _install_fake_k8s(monkeypatch, list_side_effect=aiohttp.ClientError("boom"))
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._count_owned_children("ns", "u", "s")
        assert exc_info.value.delay == 15


# ============================================================
# _patch_parent_status
# ============================================================


class TestPatchParentStatus:
    """Verify status merge-patch semantics, content type, and error wrapping."""

    @pytest.mark.asyncio
    async def test_calls_patch_with_field_manager_and_merge_patch_content_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, patch_mock = _install_fake_k8s(monkeypatch)
        body = {"status": {"completedRuns": 3, "failedRuns": 0}}
        await child_rollup._patch_parent_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            name="s",
            namespace="ns",
            body=body,
        )
        kwargs = patch_mock.await_args.kwargs
        assert kwargs["body"] == body
        assert kwargs["field_manager"] == child_rollup.ROLLUP_FIELD_MANAGER
        assert kwargs["_content_type"] == "application/merge-patch+json"
        assert kwargs["plural"] == "aiperfsweeps"
        assert kwargs["name"] == "s"
        assert kwargs["namespace"] == "ns"

    @pytest.mark.asyncio
    async def test_404_returns_silently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
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
    @pytest.mark.parametrize(
        "status_code",
        [
            param(409, id="conflict"),
            param(422, id="unprocessable"),
            param(500, id="internal-server-error"),
            param(503, id="service-unavailable"),
        ],
    )  # fmt: skip
    async def test_non_404_apiexception_wraps_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch,
            patch_status_side_effect=ApiException(status=status_code, reason="boom"),
        )
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._patch_parent_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                name="s",
                namespace="ns",
                body={"status": {}},
            )
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            param(ConnectionError("refused"), id="connection-error"),
            param(TimeoutError("slow"), id="timeout-error"),
        ],
    )  # fmt: skip
    async def test_network_errors_wrap_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        _install_fake_k8s(monkeypatch, patch_status_side_effect=exc)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._patch_parent_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                name="s",
                namespace="ns",
                body={"status": {}},
            )
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    async def test_aiohttp_client_error_wraps_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiohttp

        _install_fake_k8s(
            monkeypatch, patch_status_side_effect=aiohttp.ClientError("net")
        )
        with pytest.raises(kopf.TemporaryError):
            await child_rollup._patch_parent_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                name="s",
                namespace="ns",
                body={"status": {}},
            )


# ============================================================
# _read_parent_status
# ============================================================


class TestReadParentStatus:
    """Verify the GET → status-extraction helper."""

    @pytest.mark.asyncio
    async def test_returns_status_dict_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(
            monkeypatch,
            get_return={"status": {"phase": "Running", "maxTotalRuns": 9}},
        )
        result = await child_rollup._read_parent_status("ns", "s")
        assert result == {"phase": "Running", "maxTotalRuns": 9}

    @pytest.mark.asyncio
    async def test_returns_none_when_status_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(monkeypatch, get_return={"status": {}})
        assert await child_rollup._read_parent_status("ns", "s") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_status_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(monkeypatch, get_return={})
        assert await child_rollup._read_parent_status("ns", "s") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_status_is_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(monkeypatch, get_return={"status": None})
        assert await child_rollup._read_parent_status("ns", "s") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """404 → genuine not-found → None (safe unconditional set downstream)."""
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch, get_side_effect=ApiException(status=404, reason="NotFound")
        )
        assert await child_rollup._read_parent_status("ns", "s") is None

    @pytest.mark.asyncio
    async def test_non_404_apiexception_raises_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient read failure must NOT collapse into None — it would
        defeat the TOCTOU/maxTotalRuns guards and regress a terminal phase.
        It raises TemporaryError so the tick retries instead."""
        import kopf
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch, get_side_effect=ApiException(status=500, reason="boom")
        )
        with pytest.raises(kopf.TemporaryError):
            await child_rollup._read_parent_status("ns", "s")

    @pytest.mark.asyncio
    async def test_network_error_raises_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import kopf

        _install_fake_k8s(monkeypatch, get_side_effect=ConnectionError("refused"))
        with pytest.raises(kopf.TemporaryError):
            await child_rollup._read_parent_status("ns", "s")


# ============================================================
# _read_parent_phase
# ============================================================


class TestReadParentPhase:
    """Verify the thin .phase wrapper around _read_parent_status."""

    @pytest.mark.asyncio
    async def test_returns_phase_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_read(namespace, name, *, api=None):
            return {"phase": "Aggregating", "maxTotalRuns": 4}

        monkeypatch.setattr(child_rollup, "_read_parent_status", fake_read)
        assert await child_rollup._read_parent_phase("ns", "s") == "Aggregating"

    @pytest.mark.asyncio
    async def test_returns_none_when_status_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            child_rollup, "_read_parent_status", AsyncMock(return_value=None)
        )
        assert await child_rollup._read_parent_phase("ns", "s") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_phase_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            child_rollup, "_read_parent_status", AsyncMock(return_value={"phase": ""})
        )
        assert await child_rollup._read_parent_phase("ns", "s") is None


# ============================================================
# _conditional_phase_set
# ============================================================


class TestConditionalPhaseSet:
    """Verify JSON-patch test/replace race-safety and merge-patch fallback."""

    @pytest.mark.asyncio
    async def test_empty_expect_phase_falls_back_to_merge_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty expect_phase → call _patch_parent_status with merge-patch body."""
        captured: dict[str, Any] = {}

        async def fake_patch(
            *, group, version, plural, name, namespace, body, api=None
        ):
            captured["body"] = body
            captured["plural"] = plural

        monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
        # k8s_client must not be opened in this branch.
        await child_rollup._conditional_phase_set(
            namespace="ns",
            name="s",
            expect_phase="",
            new_phase="Aggregating",
        )
        assert captured["body"] == {"status": {"phase": "Aggregating"}}
        assert captured["plural"] == "aiperfsweeps"

    @pytest.mark.asyncio
    async def test_set_expect_phase_sends_jsonpatch_test_replace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """expect_phase set → JSON-patch with test+replace ops, json-patch+json content type."""
        _, _, patch_mock = _install_fake_k8s(monkeypatch)
        await child_rollup._conditional_phase_set(
            namespace="ns",
            name="s",
            expect_phase="Running",
            new_phase="Aggregating",
        )
        kwargs = patch_mock.await_args.kwargs
        assert kwargs["_content_type"] == "application/json-patch+json"
        assert kwargs["field_manager"] == child_rollup.ROLLUP_FIELD_MANAGER
        assert kwargs["body"] == [
            {"op": "test", "path": "/status/phase", "value": "Running"},
            {"op": "replace", "path": "/status/phase", "value": "Aggregating"},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [
            param(404, id="not-found"),
            param(422, id="test-op-failed"),
        ],
    )  # fmt: skip
    async def test_404_or_422_returns_silently(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch,
            patch_status_side_effect=ApiException(status=status_code, reason="x"),
        )
        # Must not raise.
        await child_rollup._conditional_phase_set(
            namespace="ns",
            name="s",
            expect_phase="Running",
            new_phase="Aggregating",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [
            param(409, id="conflict"),
            param(500, id="internal-server-error"),
            param(503, id="service-unavailable"),
        ],
    )  # fmt: skip
    async def test_other_apiexception_wraps_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch,
            patch_status_side_effect=ApiException(status=status_code, reason="boom"),
        )
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._conditional_phase_set(
                namespace="ns",
                name="s",
                expect_phase="Running",
                new_phase="Aggregating",
            )
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            param(ConnectionError("refused"), id="connection-error"),
            param(TimeoutError("slow"), id="timeout-error"),
        ],
    )  # fmt: skip
    async def test_network_errors_wrap_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        _install_fake_k8s(monkeypatch, patch_status_side_effect=exc)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            await child_rollup._conditional_phase_set(
                namespace="ns",
                name="s",
                expect_phase="Running",
                new_phase="Aggregating",
            )
        assert exc_info.value.delay == 15

    @pytest.mark.asyncio
    async def test_aiohttp_client_error_wraps_in_temporary_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiohttp

        _install_fake_k8s(
            monkeypatch, patch_status_side_effect=aiohttp.ClientError("x")
        )
        with pytest.raises(kopf.TemporaryError):
            await child_rollup._conditional_phase_set(
                namespace="ns",
                name="s",
                expect_phase="Running",
                new_phase="Aggregating",
            )


# ============================================================
# _api_or_new
# ============================================================


class TestApiOrNew:
    """Verify the share-or-open-fresh ApiClient context manager."""

    @pytest.mark.asyncio
    async def test_yields_passed_api_without_opening_fresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened_count = {"n": 0}

        @asynccontextmanager
        async def fake_k8s_client():
            opened_count["n"] += 1
            yield MagicMock(name="fresh")

        import aiperf.kubernetes.client as kclient

        monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

        existing = MagicMock(name="existing")
        async with child_rollup._api_or_new(existing) as got:
            assert got is existing
        assert opened_count["n"] == 0

    @pytest.mark.asyncio
    async def test_opens_fresh_when_api_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = MagicMock(name="freshClient")
        opened_count = {"n": 0}

        @asynccontextmanager
        async def fake_k8s_client():
            opened_count["n"] += 1
            yield sentinel

        import aiperf.kubernetes.client as kclient

        monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

        async with child_rollup._api_or_new(None) as got:
            assert got is sentinel
        assert opened_count["n"] == 1


# ============================================================
# _ingest_sweep_aggregate
# ============================================================


class TestIngestSweepAggregate:
    """Verify best-effort aggregate-ingest semantics."""

    @pytest.mark.asyncio
    async def test_returns_silently_when_resolve_sweep_dir_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No on-disk sweep dir → return without calling the index."""
        from aiperf.operator import results_layout, runs_index
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", "/fake/results")
        monkeypatch.setattr(results_layout, "resolve_sweep_dir", lambda *a, **k: None)
        index_mock = AsyncMock()
        monkeypatch.setattr(runs_index, "_index_sweep_from_disk", index_mock)

        await child_rollup._ingest_sweep_aggregate("ns", "s")
        index_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_index_sweep_when_dir_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Sweep dir resolved → call runs_index._index_sweep_from_disk with the dir."""
        from aiperf.operator import results_layout, runs_index
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", str(tmp_path))
        sweep_dir = tmp_path / "ns" / "s" / "epoch-1"
        sweep_dir.mkdir(parents=True)

        def fake_resolve(base, namespace, sweep_name):
            return sweep_dir

        monkeypatch.setattr(results_layout, "resolve_sweep_dir", fake_resolve)
        index_mock = AsyncMock()
        monkeypatch.setattr(runs_index, "_index_sweep_from_disk", index_mock)

        await child_rollup._ingest_sweep_aggregate("ns", "s")
        index_mock.assert_awaited_once_with("ns", "s", "epoch-1", sweep_dir)

    @pytest.mark.asyncio
    async def test_swallows_index_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Index call raising must NOT propagate (rollup must keep running)."""
        from aiperf.operator import results_layout, runs_index
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", str(tmp_path))
        sweep_dir = tmp_path / "ns" / "s" / "epoch-2"
        sweep_dir.mkdir(parents=True)
        monkeypatch.setattr(
            results_layout, "resolve_sweep_dir", lambda *a, **k: sweep_dir
        )

        async def boom(*args, **kwargs):
            raise RuntimeError("index disk full")

        monkeypatch.setattr(runs_index, "_index_sweep_from_disk", boom)
        # Must not raise.
        await child_rollup._ingest_sweep_aggregate("ns", "s")


# ============================================================
# Task 11 — runs[] terminal-entry append
# ============================================================


from aiperf.operator.handlers.sweep import _child_runs  # noqa: E402


class TestExtractSummaryMetrics:
    """``_child_runs.extract_summary_metrics`` shape contract."""

    def test_returns_empty_when_no_summary(self) -> None:
        assert _child_runs.extract_summary_metrics({}) == {}

    def test_pulls_scalar_metrics_from_summary(self) -> None:
        out = _child_runs.extract_summary_metrics(
            {
                "summary": {
                    "output_token_throughput": 42.0,
                    "request_throughput": 7.5,
                    "request_count": 100,
                    "error_request_count": 2,
                    "error_rate": 0.02,
                    "total_requests": 100,
                    "ignored_extra": "drop me",
                }
            }
        )
        assert out == {
            "output_token_throughput": 42.0,
            "request_throughput": 7.5,
            "request_count": 100,
            "error_request_count": 2,
            "error_rate": 0.02,
            "total_requests": 100,
        }

    def test_pulls_p50_p95_p99_from_ttft_and_itl(self) -> None:
        out = _child_runs.extract_summary_metrics(
            {
                "summary": {
                    "time_to_first_token": {
                        "p50": 1.0,
                        "p95": 2.0,
                        "p99": 3.0,
                        "p999": 4.0,
                    },
                    "inter_token_latency": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
                }
            }
        )
        assert out == {
            "time_to_first_token": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
            "inter_token_latency": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
        }

    def test_falls_back_to_liveSummary(self) -> None:
        out = _child_runs.extract_summary_metrics({"liveSummary": {"request_count": 5}})
        assert out == {"request_count": 5}

    def test_summary_takes_precedence_over_liveSummary(self) -> None:
        out = _child_runs.extract_summary_metrics(
            {
                "summary": {"request_count": 100},
                "liveSummary": {"request_count": 1},
            }
        )
        assert out == {"request_count": 100}


class TestBuildRunEntry:
    """``_child_runs.build_run_entry`` reads labels/annotations/status."""

    def test_full_entry_pulls_labels_annotations_and_status(self) -> None:
        body = {
            "metadata": {
                "name": "child-7",
                "labels": {
                    "aiperf.nvidia.com/variation-index": "3",
                    "aiperf.nvidia.com/variation-label": "concurrency_50",
                },
                "annotations": {
                    "aiperf.nvidia.com/variation-values": '{"concurrency": 50}',
                },
            }
        }
        status = {
            "phase": "Succeeded",
            "startTime": "2026-05-03T12:00:00Z",
            "completionTime": "2026-05-03T12:05:00Z",
            "summary": {"request_count": 100, "error_request_count": 0},
        }
        entry = _child_runs.build_run_entry(body=body, status=status, name="child-7")
        assert entry["index"] == 3
        assert entry["label"] == "concurrency_50"
        assert entry["values"] == '{"concurrency": 50}'
        assert entry["phase"] == "Succeeded"
        assert entry["childName"] == "child-7"
        assert entry["startedAt"] == "2026-05-03T12:00:00Z"
        assert entry["completedAt"] == "2026-05-03T12:05:00Z"
        assert entry["metrics"] == {"request_count": 100, "error_request_count": 0}

    def test_missing_labels_fall_back_to_index_minus_one(self) -> None:
        entry = _child_runs.build_run_entry(
            body={"metadata": {"name": "c"}},
            status={"phase": "Failed"},
            name="c",
        )
        assert entry["index"] == -1
        assert entry["label"] == ""
        assert entry["values"] == ""
        assert entry["phase"] == "Failed"
        assert entry["metrics"] == {}

    def test_garbage_index_label_falls_back_to_minus_one(self) -> None:
        entry = _child_runs.build_run_entry(
            body={
                "metadata": {
                    "name": "c",
                    "labels": {"aiperf.nvidia.com/variation-index": "not-a-number"},
                }
            },
            status={"phase": "Failed"},
            name="c",
        )
        assert entry["index"] == -1

    def test_small_scalar_variation_values_are_preserved(self) -> None:
        entry = _child_runs.build_run_entry(
            body={
                "metadata": {
                    "annotations": {
                        "aiperf.nvidia.com/variation-values": '{"concurrency": 50}'
                    }
                }
            },
            status={"phase": "Succeeded"},
            name="c",
        )
        assert entry["values"] == '{"concurrency": 50}'

    def test_status_variation_values_use_compact_bound(self) -> None:
        raw_values = orjson.dumps({"prompt_prefix": "x" * 2048}).decode()
        entry = _child_runs.build_run_entry(
            body={
                "metadata": {
                    "annotations": {"aiperf.nvidia.com/variation-values": raw_values}
                }
            },
            status={"phase": "Succeeded"},
            name="c",
        )

        values = entry["values"]
        payload = orjson.loads(values)
        assert payload == {
            "__aiperf_truncated__": True,
            "reason": "variation values exceeded status byte limit",
            "limitBytes": 256,
            "originalBytes": len(raw_values.encode()),
        }
        assert len(values.encode()) <= 256

    def test_status_runs_budget_leaves_inline_aggregate_headroom(self) -> None:
        from aiperf.sweep_controller.main import _AGGREGATE_INLINE_MAX_BYTES

        apiserver_safe_status_bytes = 1_000_000

        assert apiserver_safe_status_bytes >= (
            _child_runs._STATUS_RUNS_MAX_BYTES + _AGGREGATE_INLINE_MAX_BYTES
        )

        raw_values = orjson.dumps({"prompt_prefix": "x" * 2048}).decode()
        entries = [
            _child_runs.build_run_entry(
                body={
                    "metadata": {
                        "annotations": {
                            "aiperf.nvidia.com/variation-values": raw_values
                        }
                    }
                },
                status={"phase": "Succeeded"},
                name=f"c-{i}",
            )
            for i in range(1500)
        ]

        full_runs_payload_bytes = len(orjson.dumps({"runs": entries}))
        assert full_runs_payload_bytes > _child_runs._STATUS_RUNS_MAX_BYTES


class TestAppendRunEntryWiring:
    """End-to-end: ``on_child_phase_transition`` calls ``_append_run_entry``
    only on terminal phases, and forwards the right shape."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param("Succeeded", id="succeeded"),
            param("Completed", id="completed"),
            param("Failed", id="failed"),
            param("Cancelled", id="cancelled"),
            param("succeeded", id="lowercase"),
        ],
    )  # fmt: skip
    async def test_terminal_phase_triggers_append(
        self, monkeypatch: pytest.MonkeyPatch, phase: str
    ) -> None:
        """Each terminal phase must invoke `_append_run_entry` exactly once."""
        append_mock = AsyncMock()
        monkeypatch.setattr(child_rollup, "_append_run_entry", append_mock)

        async def fake_count(*_a, **_kw) -> dict[str, Any]:
            return {
                "pending": 0,
                "running": 1,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 1,
                "total_terminal_phase": None,
            }

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        body = {
            "metadata": {
                "name": "child-A",
                "ownerReferences": [
                    {"kind": "AIPerfSweep", "name": "swp", "uid": "u-1"}
                ],
                "labels": {
                    "aiperf.nvidia.com/variation-index": "0",
                    "aiperf.nvidia.com/variation-label": "v0",
                },
            }
        }
        await child_rollup.on_child_phase_transition(
            body=body,
            status={"phase": phase, "summary": {"request_count": 1}},
            name="child-A",
            namespace="ns",
        )
        append_mock.assert_awaited_once()
        # Positional: namespace, sweep_name, entry; api as kwarg.
        args, kwargs = append_mock.call_args
        assert args[0] == "ns"
        assert args[1] == "swp"
        entry = args[2]
        assert entry["phase"] == phase
        assert entry["childName"] == "child-A"
        assert entry["index"] == 0
        assert entry["metrics"] == {"request_count": 1}
        assert "api" in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param("Pending", id="pending"),
            param("Running", id="running"),
            param("Profiling", id="profiling"),
            param("", id="empty"),
            param(None, id="missing"),
        ],
    )  # fmt: skip
    async def test_non_terminal_phase_skips_append(
        self, monkeypatch: pytest.MonkeyPatch, phase: str | None
    ) -> None:
        append_mock = AsyncMock()
        monkeypatch.setattr(child_rollup, "_append_run_entry", append_mock)

        async def fake_count(*_a, **_kw) -> dict[str, Any]:
            return {
                "pending": 1,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 1,
                "total_terminal_phase": None,
            }

        monkeypatch.setattr(child_rollup, "_count_owned_children", fake_count)
        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        _install_fake_k8s(monkeypatch)

        status: dict[str, Any] = {} if phase is None else {"phase": phase}
        await child_rollup.on_child_phase_transition(
            body={
                "metadata": {
                    "name": "child-B",
                    "ownerReferences": [
                        {"kind": "AIPerfSweep", "name": "swp", "uid": "u-1"}
                    ],
                }
            },
            status=status,
            name="child-B",
            namespace="ns",
        )
        append_mock.assert_not_awaited()


class TestAppendRunEntryHelper:
    """Direct tests of ``_child_runs.append_run_entry`` JSON-patch behavior."""

    @pytest.mark.asyncio
    async def test_repeated_appends_preserve_runs_and_truncate_before_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fake apiserver applies JSON patches across repeated appends."""

        class FakeCustomObjectsApi:
            def __init__(self) -> None:
                self.cr: dict[str, Any] = {
                    "metadata": {"resourceVersion": "1"},
                    "status": {"totalVariations": 25},
                }
                self.status_patches: list[dict[str, Any]] = []

            async def get_namespaced_custom_object(
                self, **_kwargs: Any
            ) -> dict[str, Any]:
                return self.cr

            async def patch_namespaced_custom_object(self, **_kwargs: Any) -> None:
                raise AssertionError(
                    "runs status writes must use the status subresource"
                )

            async def patch_namespaced_custom_object_status(
                self, *, body: list[dict[str, Any]] | dict[str, Any], **_kwargs: Any
            ) -> None:
                self.status_patches.append(body)
                status = self.cr.setdefault("status", {})
                if isinstance(body, list):
                    for op in body:
                        if op.get("op") == "test":
                            continue
                        if op == {"op": "add", "path": "/status/runs", "value": []}:
                            status["runs"] = []
                            self.cr["metadata"]["resourceVersion"] = str(
                                int(self.cr["metadata"]["resourceVersion"]) + 1
                            )
                        elif (
                            op.get("op") == "add" and op.get("path") == "/status/runs/-"
                        ):
                            status.setdefault("runs", []).append(op["value"])
                            self.cr["metadata"]["resourceVersion"] = str(
                                int(self.cr["metadata"]["resourceVersion"]) + 1
                            )
                        else:  # pragma: no cover - defensive assertion aid
                            raise AssertionError(f"unexpected JSON patch op: {op}")
                    return
                status.update(body.get("status", {}))

        fake_custom = FakeCustomObjectsApi()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: fake_custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )
        monkeypatch.setattr(_child_runs, "_STATUS_RUNS_MAX_BYTES", 1000)

        accepted: list[dict[str, Any]] = []
        entries = [
            {"index": i, "phase": "Succeeded", "childName": f"swp-v{i:02d}"}
            for i in range(25)
        ]
        rejected_entry: dict[str, Any] | None = None
        for entry in entries:
            if _child_runs._runs_payload_would_exceed_budget(accepted, entry):
                rejected_entry = entry
                break
            await _child_runs.append_run_entry("ns", "swp", entry, api=MagicMock())
            accepted.append(entry)
            assert fake_custom.cr["status"]["runs"] == accepted

        assert rejected_entry is not None
        assert len(accepted) > 10
        assert len(orjson.dumps({"runs": fake_custom.cr["status"]["runs"]})) <= 1000

        await _child_runs.append_run_entry("ns", "swp", rejected_entry, api=MagicMock())

        assert fake_custom.cr["status"]["runs"] == accepted
        assert len(orjson.dumps({"runs": fake_custom.cr["status"]["runs"]})) <= 1000
        assert fake_custom.cr["status"]["runsTruncated"]["included"] == len(accepted)
        assert fake_custom.cr["status"]["runsTruncated"]["total"] == 25

    @pytest.mark.asyncio
    async def test_existing_truncation_marker_makes_later_append_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once truncation is declared, later small entries must not mutate the prefix."""
        marker = {
            "total": 25,
            "included": 12,
            "fetchURL": "http://aiperf-operator.aiperf-system:8081"
            "/api/v1/sweeps/ns/swp/children",
        }
        runs = [{"index": i, "childName": f"swp-v{i:02d}"} for i in range(12)]
        patch_mock = AsyncMock()
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.patch_namespaced_custom_object_status = patch_mock
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "9"},
                "status": {
                    "runs": runs,
                    "runsTruncated": marker,
                    "totalVariations": 25,
                },
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry(
            "ns",
            "swp",
            {"index": 12, "phase": "Succeeded", "childName": "swp-v12"},
            api=MagicMock(),
        )

        patch_mock.assert_not_awaited()
        status = custom.get_namespaced_custom_object.return_value["status"]
        assert status["runs"] == runs
        assert status["runsTruncated"] == marker

    @pytest.mark.asyncio
    async def test_retry_after_phase_patch_failure_does_not_duplicate_existing_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retried child event must see the existing childName and no-op.

        ``child_rollup`` appends ``status.runs[]`` before its later phase patch.
        If that later patch raises ``TemporaryError``, kopf replays the same
        child terminal event; this helper must make that replay idempotent.
        """
        entry = {
            "index": 0,
            "label": "v0",
            "values": '{"concurrency": 1}',
            "phase": "Succeeded",
            "childName": "swp-v00-t0",
        }
        patch_mock = AsyncMock()
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.patch_namespaced_custom_object_status = patch_mock
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "9"},
                "status": {"runs": [entry], "totalVariations": 1},
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry("ns", "swp", dict(entry), api=MagicMock())

        patch_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_runs_init_then_append_two_patches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Helper initializes absent ``runs`` once, then appends to /-."""
        patch_mock = AsyncMock()
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError(
                "runs status writes must use the status subresource"
            )
        )
        custom.patch_namespaced_custom_object_status = patch_mock
        # GET returns a CR with no runs[] so initialization is required
        # before the normal append path.
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "7"},
                "status": {"totalVariations": 1},
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry(
            "ns", "swp", {"index": 0, "phase": "Succeeded"}, api=MagicMock()
        )
        assert patch_mock.await_count == 2
        # First call: init runs[] = []
        first_body = patch_mock.await_args_list[0].kwargs["body"]
        assert first_body == [
            {"op": "test", "path": "/metadata/resourceVersion", "value": "7"},
            {"op": "add", "path": "/status/runs", "value": []},
        ]
        # Second call: append to /-
        second_body = patch_mock.await_args_list[1].kwargs["body"]
        assert second_body == [
            {"op": "test", "path": "/metadata/resourceVersion", "value": "7"},
            {
                "op": "add",
                "path": "/status/runs/-",
                "value": {"index": 0, "phase": "Succeeded"},
            },
        ]

    @pytest.mark.asyncio
    async def test_concurrent_absent_runs_initializers_preserve_both_appends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two stale absent readers must not let one init erase another append."""

        from kubernetes_asyncio.client.exceptions import ApiException

        class FakeCustomObjectsApi:
            def __init__(self) -> None:
                self.cr: dict[str, Any] = {
                    "metadata": {"resourceVersion": "1"},
                    "status": {"totalVariations": 2},
                }
                self.get_count = 0
                self.init_count = 0
                self.both_read_absent = asyncio.Event()
                self.first_append_done = asyncio.Event()

            async def get_namespaced_custom_object(
                self, **_kwargs: Any
            ) -> dict[str, Any]:
                self.get_count += 1
                if self.get_count == 2:
                    self.both_read_absent.set()
                await self.both_read_absent.wait()
                if self.get_count <= 2:
                    return {
                        "metadata": {"resourceVersion": "1"},
                        "status": {"totalVariations": 2},
                    }
                return self.cr

            async def patch_namespaced_custom_object(self, **_kwargs: Any) -> None:
                raise AssertionError(
                    "runs status writes must use the status subresource"
                )

            async def patch_namespaced_custom_object_status(
                self, *, body: list[dict[str, Any]], **_kwargs: Any
            ) -> None:
                status = self.cr.setdefault("status", {})
                op_index = 0
                if body[0].get("op") == "test":
                    expected = body[0]["value"]
                    actual = self.cr["metadata"]["resourceVersion"]
                    if expected != actual:
                        raise ApiException(status=409, reason="Conflict")
                    op_index = 1
                op = body[op_index]
                if op == {"op": "add", "path": "/status/runs", "value": []}:
                    self.init_count += 1
                    if self.init_count == 2:
                        await self.first_append_done.wait()
                    status["runs"] = []
                    self.cr["metadata"]["resourceVersion"] = str(
                        int(self.cr["metadata"]["resourceVersion"]) + 1
                    )
                    return
                if op.get("op") == "add" and op.get("path") == "/status/runs/-":
                    status.setdefault("runs", []).append(op["value"])
                    self.cr["metadata"]["resourceVersion"] = str(
                        int(self.cr["metadata"]["resourceVersion"]) + 1
                    )
                    self.first_append_done.set()
                    return
                raise AssertionError(f"unexpected JSON patch op: {op}")

        fake_custom = FakeCustomObjectsApi()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: fake_custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        entries = [
            {"index": 0, "phase": "Succeeded"},
            {"index": 1, "phase": "Succeeded"},
        ]
        await asyncio.gather(
            *(
                _child_runs.append_run_entry("ns", "swp", entry, api=MagicMock())
                for entry in entries
            )
        )

        assert (
            sorted(fake_custom.cr["status"]["runs"], key=lambda item: item["index"])
            == entries
        )

    @pytest.mark.asyncio
    async def test_many_concurrent_append_writers_preserve_all_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Many stale writers must retry until every child entry is appended."""

        from kubernetes_asyncio.client.exceptions import ApiException

        writer_count = 8

        class FakeCustomObjectsApi:
            def __init__(self) -> None:
                self.cr: dict[str, Any] = {
                    "metadata": {"resourceVersion": "1"},
                    "status": {"runs": [], "totalVariations": writer_count},
                }
                self.get_count = 0
                self.first_read_barrier = asyncio.Event()
                self.lock = asyncio.Lock()

            async def get_namespaced_custom_object(
                self, **_kwargs: Any
            ) -> dict[str, Any]:
                self.get_count += 1
                if self.get_count == writer_count:
                    self.first_read_barrier.set()
                if self.get_count <= writer_count:
                    await self.first_read_barrier.wait()
                    return {
                        "metadata": {"resourceVersion": "1"},
                        "status": {"runs": [], "totalVariations": writer_count},
                    }
                return {
                    "metadata": dict(self.cr["metadata"]),
                    "status": {
                        "runs": list(self.cr["status"]["runs"]),
                        "totalVariations": writer_count,
                    },
                }

            async def patch_namespaced_custom_object(self, **_kwargs: Any) -> None:
                raise AssertionError(
                    "runs status writes must use the status subresource"
                )

            async def patch_namespaced_custom_object_status(
                self, *, body: list[dict[str, Any]], **_kwargs: Any
            ) -> None:
                async with self.lock:
                    expected = body[0]["value"]
                    actual = self.cr["metadata"]["resourceVersion"]
                    if expected != actual:
                        raise ApiException(status=409, reason="Conflict")
                    self.cr["status"]["runs"].append(body[1]["value"])
                    self.cr["metadata"]["resourceVersion"] = str(int(actual) + 1)

        fake_custom = FakeCustomObjectsApi()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: fake_custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        entries = [{"index": i, "phase": "Succeeded"} for i in range(writer_count)]
        await asyncio.gather(
            *(
                _child_runs.append_run_entry("ns", "swp", entry, api=MagicMock())
                for entry in entries
            )
        )

        assert (
            sorted(fake_custom.cr["status"]["runs"], key=lambda item: item["index"])
            == entries
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "conflict_status",
        [param(409, id="conflict"), param(422, id="unprocessable-cas")],
    )  # fmt: skip
    async def test_absent_runs_init_cas_failure_still_attempts_append(
        self, monkeypatch: pytest.MonkeyPatch, conflict_status: int
    ) -> None:
        """Stale GET with absent runs plus init CAS failure still appends."""

        class FakeApiException(Exception):
            def __init__(self, status: int, reason: str, body: str = "") -> None:
                super().__init__(reason)
                self.status = status
                self.reason = reason
                self.body = body

        calls: list[Any] = []
        get_count = 0
        init_failed = False

        async def fake_get(**_kwargs: Any) -> dict[str, Any]:
            nonlocal get_count
            get_count += 1
            if get_count == 1:
                return {
                    "metadata": {"resourceVersion": "7"},
                    "status": {"totalVariations": 1},
                }
            return {
                "metadata": {"resourceVersion": "8"},
                "status": {"runs": [], "totalVariations": 1},
            }

        async def fake_patch(**kwargs: Any) -> None:
            nonlocal init_failed
            body = kwargs["body"]
            calls.append(body)
            if (
                body[-1] == {"op": "add", "path": "/status/runs", "value": []}
                and not init_failed
            ):
                init_failed = True
                raise FakeApiException(
                    conflict_status,
                    "Conflict",
                    body="json patch test failed for metadata.resourceVersion",
                )

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError(
                "runs status writes must use the status subresource"
            )
        )
        custom.patch_namespaced_custom_object_status = fake_patch
        custom.get_namespaced_custom_object = fake_get
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=FakeApiException),
        )

        import kubernetes_asyncio
        import kubernetes_asyncio.client.exceptions as kexc

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )
        monkeypatch.setattr(kexc, "ApiException", FakeApiException, raising=False)

        await _child_runs.append_run_entry("ns", "swp", {"index": 1}, api=MagicMock())
        assert calls == [
            [
                {"op": "test", "path": "/metadata/resourceVersion", "value": "7"},
                {"op": "add", "path": "/status/runs", "value": []},
            ],
            [
                {"op": "test", "path": "/metadata/resourceVersion", "value": "8"},
                {"op": "add", "path": "/status/runs/-", "value": {"index": 1}},
            ],
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "conflict_status",
        [param(409, id="conflict"), param(422, id="unprocessable-cas")],
    )  # fmt: skip
    async def test_append_run_entry_cas_retry_exhaustion_raises_retryable(
        self, monkeypatch: pytest.MonkeyPatch, conflict_status: int
    ) -> None:
        """Repeated resourceVersion conflicts must let kopf retry later."""

        from kubernetes_asyncio.client.exceptions import ApiException

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "1"},
                "status": {"runs": [], "totalVariations": 1},
            }
        )
        api_error = ApiException(status=conflict_status, reason="Conflict")
        api_error.body = "json patch test failed for metadata.resourceVersion"
        custom.patch_namespaced_custom_object_status = AsyncMock(side_effect=api_error)
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        with pytest.raises(kopf.TemporaryError, match="raced for 20"):
            await _child_runs.append_run_entry(
                "ns", "swp", {"index": 1}, api=MagicMock()
            )

        assert custom.get_namespaced_custom_object.await_count == 20
        assert custom.patch_namespaced_custom_object_status.await_count == 20
        for call in custom.patch_namespaced_custom_object_status.await_args_list:
            assert call.kwargs["body"] == [
                {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
                {"op": "add", "path": "/status/runs/-", "value": {"index": 1}},
            ]

    @pytest.mark.asyncio
    async def test_append_run_entry_transient_get_failure_raises_without_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown current status must not be treated as absent runs[]."""

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=TimeoutError("apiserver read timed out")
        )
        custom.patch_namespaced_custom_object_status = AsyncMock()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        with pytest.raises(kopf.TemporaryError, match="status read"):
            await _child_runs.append_run_entry(
                "ns", "swp", {"index": 1}, api=MagicMock()
            )

        custom.patch_namespaced_custom_object_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_append_run_entry_parent_get_404_noops_without_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleted parent sweeps must not make kopf retry child rollups forever."""

        from kubernetes_asyncio.client.exceptions import ApiException

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404, reason="Not Found")
        )
        custom.patch_namespaced_custom_object_status = AsyncMock()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry("ns", "swp", {"index": 1}, api=MagicMock())

        custom.get_namespaced_custom_object.assert_awaited_once()
        custom.patch_namespaced_custom_object_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_stale_append_rechecks_budget_after_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent stale readers CAS, retry, then truncate instead of overfilling."""

        from kubernetes_asyncio.client.exceptions import ApiException

        current_runs = [{"index": i, "payload": "x" * 50} for i in range(5)]
        safe_entry = {"index": 5, "payload": "safe"}
        rejected_entry = {"index": 6, "payload": "over"}
        budget = max(
            len(orjson.dumps({"runs": [*current_runs, safe_entry]})),
            len(orjson.dumps({"runs": [*current_runs, rejected_entry]})),
        )
        assert len(orjson.dumps({"runs": [*current_runs, safe_entry]})) <= budget
        assert len(orjson.dumps({"runs": [*current_runs, rejected_entry]})) <= budget
        assert (
            len(orjson.dumps({"runs": [*current_runs, safe_entry, rejected_entry]}))
            > budget
        )

        class FakeCustomObjectsApi:
            def __init__(self) -> None:
                self.cr: dict[str, Any] = {
                    "metadata": {"resourceVersion": "1"},
                    "status": {"runs": list(current_runs), "totalVariations": 7},
                }
                self.get_count = 0
                self.both_stale_reads_done = asyncio.Event()
                self.safe_append_done = asyncio.Event()

            async def get_namespaced_custom_object(
                self, **_kwargs: Any
            ) -> dict[str, Any]:
                self.get_count += 1
                if self.get_count == 2:
                    self.both_stale_reads_done.set()
                await self.both_stale_reads_done.wait()
                if self.get_count <= 2:
                    return {
                        "metadata": {"resourceVersion": "1"},
                        "status": {"runs": list(current_runs), "totalVariations": 7},
                    }
                return self.cr

            async def patch_namespaced_custom_object(self, **_kwargs: Any) -> None:
                raise AssertionError(
                    "runs status writes must use the status subresource"
                )

            async def patch_namespaced_custom_object_status(
                self, *, body: list[dict[str, Any]] | dict[str, Any], **_kwargs: Any
            ) -> None:
                if not isinstance(body, list):
                    self.cr.setdefault("status", {}).update(body.get("status", {}))
                    return

                expected = body[0]["value"]
                op = body[1]
                entry = op["value"]
                if entry["index"] == rejected_entry["index"] and expected == "1":
                    await self.safe_append_done.wait()
                actual = self.cr["metadata"]["resourceVersion"]
                if expected != actual:
                    raise ApiException(status=409, reason="Conflict")
                self.cr.setdefault("status", {}).setdefault("runs", []).append(entry)
                self.cr["metadata"]["resourceVersion"] = str(int(actual) + 1)
                if entry["index"] == safe_entry["index"]:
                    self.safe_append_done.set()

        fake_custom = FakeCustomObjectsApi()
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: fake_custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )
        monkeypatch.setattr(_child_runs, "_STATUS_RUNS_MAX_BYTES", budget)

        await asyncio.gather(
            _child_runs.append_run_entry("ns", "swp", safe_entry, api=MagicMock()),
            _child_runs.append_run_entry("ns", "swp", rejected_entry, api=MagicMock()),
        )

        assert fake_custom.cr["status"]["runs"] == [*current_runs, safe_entry]
        assert len(orjson.dumps({"runs": fake_custom.cr["status"]["runs"]})) <= budget
        assert fake_custom.cr["status"]["runsTruncated"] == {
            "total": 7,
            "included": len(current_runs) + 1,
            "fetchURL": "http://aiperf-operator.aiperf-system:8081"
            "/api/v1/sweeps/ns/swp/children",
        }

    @pytest.mark.asyncio
    async def test_append_run_entry_truncates_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At/above 1500 entries, stamp ``status.runsTruncated`` instead of
        appending. The threshold protects the AIPerfSweep CR from blowing
        past apiserver's 1 MiB limit on extremely large sweeps."""
        patch_mock = AsyncMock(
            side_effect=AssertionError(
                "runs status writes must use the status subresource"
            )
        )
        status_patch_mock = AsyncMock()
        custom = MagicMock()
        custom.patch_namespaced_custom_object = patch_mock
        custom.patch_namespaced_custom_object_status = status_patch_mock
        # Simulate a sweep that already has 1500 run entries — equal to
        # the threshold, so the next append must be truncated.
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "status": {
                    "runs": [{"index": i} for i in range(1500)],
                    "totalVariations": 2000,
                }
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry(
            "ns", "swp", {"index": 1500, "phase": "Succeeded"}, api=MagicMock()
        )
        # Init-patch fires once (idempotent runs[] init), but the
        # ``add to /-`` patch must NOT fire — only the truncated stamp.
        append_calls = [
            c
            for c in status_patch_mock.await_args_list
            if isinstance(c.kwargs["body"], list)
            and c.kwargs["body"][-1].get("path") == "/status/runs/-"
        ]
        assert append_calls == []
        # The merge-patch onto ``status.runsTruncated`` is the only
        # status-write that should land for over-threshold appends.
        status_patch_mock.assert_awaited_once()
        kwargs = status_patch_mock.await_args.kwargs
        assert kwargs["_content_type"] == "application/merge-patch+json"
        body = kwargs["body"]
        assert body["status"]["runsTruncated"]["included"] == 1500
        assert body["status"]["runsTruncated"]["total"] == 2000
        assert (
            body["status"]["runsTruncated"]["fetchURL"]
            == "http://aiperf-operator.aiperf-system:8081"
            "/api/v1/sweeps/ns/swp/children"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [param(500, id="internal-server-error"), param(503, id="service-unavailable")],
    )  # fmt: skip
    async def test_truncation_stamp_transient_failure_raises_retryable(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        """When no inline append happens, a failed truncation marker must retry."""
        from kubernetes_asyncio.client.exceptions import ApiException

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=AssertionError("runs status writes must use status subresource")
        )
        custom.patch_namespaced_custom_object_status = AsyncMock(
            side_effect=ApiException(status=status_code, reason="temporary")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "status": {
                    "runs": [{"index": i} for i in range(1500)],
                    "totalVariations": 2000,
                }
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        with pytest.raises(kopf.TemporaryError, match="runsTruncated stamp failed"):
            await _child_runs.append_run_entry(
                "ns", "swp", {"index": 1500, "phase": "Succeeded"}, api=MagicMock()
            )

    @pytest.mark.asyncio
    async def test_append_run_entry_truncates_before_serialized_runs_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byte-budget truncation uses the full serialized runs payload.

        Counting only values bytes misses labels, timestamps, metrics, and JSON
        overhead; this test fills ``status.runs`` below the byte budget and
        verifies the next representative entry is rejected before the serialized
        payload crosses it.
        """
        budget = _child_runs._STATUS_RUNS_MAX_BYTES
        representative_entry = {
            "index": 0,
            "label": "prompt_prefix_0000",
            "values": orjson.dumps(
                _child_runs._status_variation_values_truncated_payload(2056)
            ).decode(),
            "phase": "Succeeded",
            "childName": "swp-v0000-t0",
            "startedAt": "2026-05-03T12:00:00Z",
            "completedAt": "2026-05-03T12:05:00Z",
            "metrics": {
                "output_token_throughput": 12345.678,
                "request_throughput": 987.654,
                "request_count": 100000,
                "error_count": 0,
                "ttft": {"p50": 1.23, "p95": 4.56, "p99": 7.89},
                "itl": {"p50": 0.12, "p95": 0.34, "p99": 0.56},
            },
        }
        current_runs: list[dict[str, Any]] = []
        while True:
            next_entry = representative_entry | {
                "index": len(current_runs),
                "label": f"prompt_prefix_{len(current_runs):04d}",
                "childName": f"swp-v{len(current_runs):04d}-t0",
            }
            if len(orjson.dumps({"runs": [*current_runs, next_entry]})) > budget:
                rejected_entry = next_entry
                break
            current_runs.append(next_entry)

        assert len(current_runs) < _child_runs._RUNS_SAFETY_THRESHOLD
        assert len(orjson.dumps({"runs": current_runs})) <= budget
        assert len(orjson.dumps({"runs": [*current_runs, rejected_entry]})) > budget

        patch_mock = AsyncMock(
            side_effect=AssertionError(
                "runs status writes must use the status subresource"
            )
        )
        status_patch_mock = AsyncMock()
        custom = MagicMock()
        custom.patch_namespaced_custom_object = patch_mock
        custom.patch_namespaced_custom_object_status = status_patch_mock
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "status": {
                    "runs": current_runs,
                    "totalVariations": len(current_runs) + 1,
                }
            }
        )
        fake_k8s_module = SimpleNamespace(
            CustomObjectsApi=lambda _api: custom,
            exceptions=SimpleNamespace(ApiException=Exception),
        )
        import kubernetes_asyncio

        monkeypatch.setattr(
            kubernetes_asyncio, "client", fake_k8s_module, raising=False
        )

        await _child_runs.append_run_entry("ns", "swp", rejected_entry, api=MagicMock())

        append_calls = [
            c
            for c in status_patch_mock.await_args_list
            if isinstance(c.kwargs["body"], list)
            and c.kwargs["body"][-1].get("path") == "/status/runs/-"
        ]
        assert append_calls == []
        status_patch_mock.assert_awaited_once()
        body = status_patch_mock.await_args.kwargs["body"]
        assert body["status"]["runsTruncated"]["included"] == len(current_runs)
        assert body["status"]["runsTruncated"]["total"] == len(current_runs) + 1
