# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the observedGeneration stamping path.

Ensures ``StatusBuilder.set_observed_generation`` flows through:
  - ``monitor.monitor_progress`` success path (sets observedGeneration AFTER
    ``_monitor_tick`` has populated phase/progress fields)
  - ``create._finalize_success`` end-to-end (stamps observedGeneration plus
    the four other status fields)

Tests use kopf.Patch directly and mock the heavy collaborators
(``_monitor_tick``, k8s_client) so we observe the patch shape that would
land on the apiserver.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import pytest

from aiperf.operator.handlers import create as create_handler
from aiperf.operator.handlers import monitor as monitor_handler
from aiperf.operator.status import StatusBuilder


def _ajob_body(generation: int, phase: str = "Running") -> dict[str, Any]:
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "j",
            "namespace": "ns",
            "generation": generation,
            "annotations": {},
        },
        "spec": {},
        "status": {
            "phase": phase,
            "jobId": "j",
            "jobSetName": "aiperf-j",
        },
    }


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_monitor_progress_stamps_observed_generation_alongside_phase() -> None:
    """A full monitor tick stamps observedGeneration AFTER ``_monitor_tick`` has
    written phase/progress fields. Both must end up in the same kopf.Patch.
    """
    body = _ajob_body(generation=7)

    async def fake_tick(api: Any, **kwargs: Any) -> None:
        # Simulate _monitor_tick doing real work via the StatusBuilder.
        sb = kwargs["sb"]
        kwargs["patch"].status["progress"] = {"completed": 42, "total": 100}
        sb.set_phase(monitor_handler.Phase.RUNNING)
        sb.finalize()

    p = kopf.Patch()
    with (
        patch.object(monitor_handler, "_monitor_tick", new=fake_tick),
        patch.object(
            monitor_handler,
            "k8s_client",
            new=lambda: _AsyncCtx(),
        ),
    ):
        await monitor_handler.monitor_progress(
            body=body,
            status=body["status"],
            spec=body["spec"],
            name="j",
            namespace="ns",
            patch=p,
        )

    assert p.status.get("observedGeneration") == 7, (
        "observedGeneration must be stamped on the success path"
    )
    assert p.status.get("phase") == "Running"
    assert p.status.get("progress") == {"completed": 42, "total": 100}


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_observed_generation_increments_with_spec_edits() -> None:
    """After a spec edit bumps generation, the next successful tick must stamp
    the new value (not retain the prior one).
    """

    async def fake_tick(api: Any, **kwargs: Any) -> None:
        kwargs["sb"].finalize()

    with (
        patch.object(monitor_handler, "_monitor_tick", new=fake_tick),
        patch.object(monitor_handler, "k8s_client", new=lambda: _AsyncCtx()),
    ):
        # Tick 1: generation=7
        body_v1 = _ajob_body(generation=7)
        p1 = kopf.Patch()
        await monitor_handler.monitor_progress(
            body=body_v1,
            status=body_v1["status"],
            spec=body_v1["spec"],
            name="j",
            namespace="ns",
            patch=p1,
        )
        assert p1.status.get("observedGeneration") == 7

        # Tick 2: same CR, but spec edit bumped generation to 8.
        body_v2 = _ajob_body(generation=8)
        p2 = kopf.Patch()
        await monitor_handler.monitor_progress(
            body=body_v2,
            status=body_v2["status"],
            spec=body_v2["spec"],
            name="j",
            namespace="ns",
            patch=p2,
        )
        assert p2.status.get("observedGeneration") == 8


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_monitor_progress_omits_observed_generation_when_absent_from_body() -> (
    None
):
    """A malformed body lacking ``metadata.generation`` must not crash and
    must not stamp ``observedGeneration`` (since there's no value to stamp).
    """
    body = _ajob_body(generation=1)
    del body["metadata"]["generation"]

    async def fake_tick(api: Any, **kwargs: Any) -> None:
        kwargs["sb"].finalize()

    p = kopf.Patch()
    with (
        patch.object(monitor_handler, "_monitor_tick", new=fake_tick),
        patch.object(monitor_handler, "k8s_client", new=lambda: _AsyncCtx()),
    ):
        await monitor_handler.monitor_progress(
            body=body,
            status=body["status"],
            spec=body["spec"],
            name="j",
            namespace="ns",
            patch=p,
        )

    assert "observedGeneration" not in p.status


@pytest.mark.component_integration
def test_finalize_success_writes_observed_generation_alongside_status_fields() -> None:
    """End-to-end ``_finalize_success`` writes:
      - observedGeneration
      - phase=Pending
      - jobId, jobSetName, startTime
      - workers (ready, total)
    onto the kopf.Patch in a single shot.
    """
    body = _ajob_body(generation=12)
    p = kopf.Patch()
    sb = StatusBuilder(p)

    deployment = MagicMock()
    deployment.jobset_name = "aiperf-j"
    deploy_config = MagicMock()
    deploy_config.results_ttl_days = 0  # falsy → resultsTtlDays not set

    with (
        patch.object(create_handler.events, "resources_created"),
        patch.object(create_handler.events, "created"),
    ):
        create_handler._finalize_success(
            patch=p,
            status=sb,
            body=body,
            deployment=deployment,
            deploy_config=deploy_config,
            configmap_name="configmap-j",
            jobset_name="aiperf-j",
            job_id="j",
            total_workers=8,
        )

    s = p.status
    assert s["observedGeneration"] == 12, "observedGeneration must be stamped"
    assert s["phase"] == "Pending"
    assert s["jobId"] == "j"
    assert s["jobSetName"] == "aiperf-j"
    assert "startTime" in s
    assert s["workers"]["ready"] == 0
    assert s["workers"]["total"] == 8
    # Conditions are flushed via finalize → must contain ResourcesCreated.
    cond_types = {c["type"] for c in s.get("conditions", [])}
    assert "ResourcesCreated" in cond_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Minimal async context manager standing in for ``k8s_client()``."""

    async def __aenter__(self) -> Any:
        return MagicMock()

    async def __aexit__(self, *_a: Any) -> None:
        return None


# Quiet unused-import lint — AsyncMock is reserved for future test additions.
_ = AsyncMock
