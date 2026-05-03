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

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
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
                "completed": 2,
                "failed": 1,
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
        assert body_patch["lastChildEvent"] == {"name": "child-A", "phase": "Failed"}

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
        assert result["failed"] == 3
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
            "completed": 0,
            "failed": 0,
            "in_flight": 0,
            "total_terminal_phase": None,
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
    async def test_returns_none_on_apiexception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive: any exception during GET → None (best-effort read)."""
        from kubernetes_asyncio.client import ApiException

        _install_fake_k8s(
            monkeypatch, get_side_effect=ApiException(status=500, reason="boom")
        )
        assert await child_rollup._read_parent_status("ns", "s") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_k8s(monkeypatch, get_side_effect=ConnectionError("refused"))
        assert await child_rollup._read_parent_status("ns", "s") is None


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
