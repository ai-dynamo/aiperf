# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the JobSet-not-found phase-stomp race.

Race shape: ``handle_completion`` writes ``phase=Completed`` then
``_maybe_delete_jobset_after_success`` deletes the JobSet. A subsequent
``monitor_progress`` timer tick fires with a *stale* body snapshot
(kopf's local cache hasn't yet caught the watch event for our own
patch), sees a non-terminal ``current_phase``, fetches the JobSet,
gets a 404, and ``_reconcile_missing_jobset`` would historically stamp
``Phase.FAILED`` over an already-Completed CR.

The defenses are in :func:`_reconcile_missing_jobset`:
    1. The completion-claim annotation on the body is authoritative —
       ``try_claim_completion`` stamps it BEFORE ``handle_completion``,
       and only the success path deletes the JobSet, so claim+404 means
       "success handler owns this CR".
    2. Fresh-read failure must not stamp FAILED (apiserver hiccup is
       not evidence of benchmark failure).
    3. Fresh body's annotations are re-checked too — a peer operator
       pod (HA) or concurrent tick may have stamped the claim between
       our body snapshot and the fresh read.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.phase import Phase
from aiperf.operator.handlers.monitor import _reconcile_missing_jobset
from aiperf.operator.status import StatusBuilder


def _make_status_builder() -> tuple[StatusBuilder, Any]:
    patch = MagicMock()
    patch.status = {}
    return StatusBuilder(patch, {}), patch


def _body(*, claimed: bool = False, phase: str | None = None) -> dict[str, Any]:
    """Build an AIPerfJob CR body, optionally with claim annotation / phase."""
    metadata: dict[str, Any] = {}
    if claimed:
        metadata["annotations"] = {
            Annotations.COMPLETION_CLAIMED: "2026-04-29T08:18:22Z"
        }
    body: dict[str, Any] = {"metadata": metadata}
    if phase is not None:
        body["status"] = {"phase": phase}
    return body


@pytest.mark.asyncio
async def test_short_circuits_when_current_phase_is_terminal() -> None:
    """If the cached phase is already terminal, no fresh read is needed."""
    sb, patch = _make_status_builder()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock()  # should never be called

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(),
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.COMPLETED,
        sb=sb,
    )

    assert result is True
    assert "phase" not in patch.status
    custom.get_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_short_circuits_on_completion_claim_with_stale_body() -> None:
    """The race regression: stale non-terminal phase but claim annotation is set.

    Reproduces the live capture (job stress-1 stomped at 08:18:38 over a
    08:18:22 success): kopf timer fires with a body snapshot taken before
    the success handler's phase patch was visible. Without the
    completion-claim gate, the monitor would re-read, possibly observe
    the stale phase too, and stamp FAILED.
    """
    sb, patch = _make_status_builder()
    custom = MagicMock()
    # Even if a fresh read returned the stale Running phase, the claim
    # annotation alone is sufficient evidence to short-circuit. Assert
    # the fresh-read path is never even reached.
    custom.get_namespaced_custom_object = AsyncMock()

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(claimed=True, phase="Running"),
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.RUNNING,
        sb=sb,
    )

    assert result is True
    assert "phase" not in patch.status
    custom.get_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_short_circuits_when_fresh_read_shows_terminal_phase() -> None:
    """Fresh read sees the success patch; short-circuit without stamping FAILED."""
    sb, patch = _make_status_builder()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"phase": str(Phase.COMPLETED)}}
    )

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(phase="Running"),  # cached body still stale
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.RUNNING,
        sb=sb,
    )

    assert result is True
    assert "phase" not in patch.status


@pytest.mark.asyncio
async def test_short_circuits_when_fresh_read_carries_claim_annotation() -> None:
    """Peer operator pod (HA) stamped claim between body snapshot and fresh read.

    The cached body has no claim annotation, but a fresh read after the
    2s sleep returns a CR with the claim annotation set. Phase may still
    be non-terminal in the fresh read if the success handler is mid-flight
    on a peer pod, but the claim alone is sufficient evidence.
    """
    sb, patch = _make_status_builder()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "annotations": {Annotations.COMPLETION_CLAIMED: "2026-04-29T08:18:22Z"}
            },
            "status": {"phase": str(Phase.RUNNING)},
        }
    )

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(phase="Running"),
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.RUNNING,
        sb=sb,
    )

    assert result is True
    assert "phase" not in patch.status


@pytest.mark.asyncio
async def test_does_not_stamp_failed_when_fresh_read_raises() -> None:
    """Apiserver hiccup must not be misread as benchmark failure.

    The previous implementation logged the exception and then fell
    through to ``sb.set_phase(Phase.FAILED)``. That's the second arm of
    the JobSet-not-found phase-stomp bug: any GET error on a CR whose
    JobSet was just deleted (success path) would re-stamp FAILED.
    """
    sb, patch = _make_status_builder()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=503, reason="Service Unavailable")
    )

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(phase="Running"),
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.RUNNING,
        sb=sb,
    )

    assert result is True
    assert "phase" not in patch.status


@pytest.mark.asyncio
async def test_stamps_failed_for_genuine_orphan_jobset() -> None:
    """Legitimate JobSet-deleted-out-of-band case still stamps FAILED.

    No claim annotation on cached or fresh body, fresh phase still
    non-terminal — that means the JobSet really did disappear without
    a completion handler having owned the cleanup. Mark FAILED so the
    CR doesn't sit in Pending forever.
    """
    sb, patch = _make_status_builder()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {"annotations": {}},
            "status": {"phase": str(Phase.RUNNING)},
        }
    )

    result = await _reconcile_missing_jobset(
        custom,
        body=_body(phase="Running"),
        namespace="ns",
        name="job",
        jobset_name="js",
        current_phase=Phase.RUNNING,
        sb=sb,
    )

    assert result is False
    assert patch.status["phase"] == str(Phase.FAILED)
    assert patch.status["error"] == "JobSet not found"
