# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the JobSet → AIPerfJob terminal-condition chain.

These tests exercise:
    handle_jobset_conditions()
        → _lookup_aiperfjob_body() → fake apiserver get
        → _set_benchmark_complete_annotation() → fake apiserver patch

against an in-memory fake apiserver. Unit tests at tests/unit/operator/
already cover the decision logic with mocked helpers; here we verify the
helper code paths actually issue the apiserver round-trips with the
correct group/version/plural/name and patch shape.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from aiperf.kubernetes.constants import AIPerfLabels, Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_API_VERSION,
    AIPERF_PLURAL,
)
from aiperf.operator.handlers.jobset_terminal import handle_jobset_conditions
from tests.component_integration.operator._fake_apiserver import FakeApiserver

NS = "bench"
COMPLETED_TRUE = [{"type": "Completed", "status": "True", "reason": "AllJobsCompleted"}]


def _ajob_body(name: str, annotations: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a realistic AIPerfJob CR body."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": name,
            "namespace": NS,
            "uid": f"uid-{name}",
            "resourceVersion": "1",
            "generation": 1,
            "annotations": dict(annotations or {}),
        },
        "spec": {},
        "status": {"phase": "Running", "jobId": name, "jobSetName": f"aiperf-{name}"},
    }


def _jobset_body(name: str) -> dict[str, Any]:
    """Build a JobSet body that proves AIPerfJob ownership.

    The handler's ``_is_trusted_aiperf_jobset`` gate requires the JobSet to
    carry the AIPerf app labels, the parent ``job-id`` label, and an
    ownerReference back to the parent AIPerfJob (matching name + uid).
    """
    return {
        "metadata": {
            "name": f"aiperf-{name}",
            "labels": {
                AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
                AIPerfLabels.JOB_ID: name,
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": name,
                    "uid": f"uid-{name}",
                }
            ],
        }
    }


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_completed_condition_preserves_controller_completion_handshake() -> None:
    """JobSet completion does not bypass the controller's artifact-ready handshake."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "alpha", _ajob_body("alpha"))

    with fake.context():
        await handle_jobset_conditions(
            old=[],
            new=COMPLETED_TRUE,
            namespace=NS,
            jobset_name="aiperf-alpha",
            jobset_body=_jobset_body("alpha"),
        )

    assert fake.get_call_count("alpha") == 0
    assert fake.patch_call_count("alpha") == 0


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_completed_condition_skips_already_complete_controller_state() -> None:
    """The JobSet observer never writes completion state, even when it is present."""
    fake = FakeApiserver()
    fake.add_cr(
        NS,
        AIPERF_PLURAL,
        "beta",
        _ajob_body("beta", {Annotations.BENCHMARK_COMPLETE: "true"}),
    )

    with fake.context():
        await handle_jobset_conditions(
            old=[],
            new=COMPLETED_TRUE,
            namespace=NS,
            jobset_name="aiperf-beta",
            jobset_body=_jobset_body("beta"),
        )

    assert fake.get_call_count("beta") == 0
    assert fake.patch_call_count("beta") == 0


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_aiperfjob_deleted_mid_chain_silently_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed JobSet never reads or patches a deleted parent CR."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "gamma", _ajob_body("gamma"))
    fake.patch_404.add((NS, AIPERF_PLURAL, "gamma"))

    import logging

    with fake.context(), caplog.at_level(logging.WARNING):
        # Must not raise.
        await handle_jobset_conditions(
            old=[],
            new=COMPLETED_TRUE,
            namespace=NS,
            jobset_name="aiperf-gamma",
            jobset_body=_jobset_body("gamma"),
        )

    assert fake.get_call_count("gamma") == 0
    assert fake.patch_call_count("gamma") == 0


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_concurrent_completed_jobsets_do_not_bypass_controller_handshake() -> (
    None
):
    """Concurrent terminal events leave controller-owned completion untouched."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "left", _ajob_body("left"))
    fake.add_cr(NS, AIPERF_PLURAL, "right", _ajob_body("right"))

    with fake.context():
        await asyncio.gather(
            handle_jobset_conditions(
                old=[],
                new=COMPLETED_TRUE,
                namespace=NS,
                jobset_name="aiperf-left",
                jobset_body=_jobset_body("left"),
            ),
            handle_jobset_conditions(
                old=[],
                new=COMPLETED_TRUE,
                namespace=NS,
                jobset_name="aiperf-right",
                jobset_body=_jobset_body("right"),
            ),
        )

    assert fake.get_call_count("left") == 0
    assert fake.get_call_count("right") == 0
    assert fake.patches == []


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_completion_annotation_dispatches_on_benchmark_complete() -> None:
    """The controller-written annotation dispatches ``on_benchmark_complete``.

    Verify the lifecycle handler runs end-to-end through ``try_claim_completion``
    when invoked with the controller's annotated body.

    We mock ``handle_completion`` itself (its internals are integration-tested
    elsewhere) and the controller-shutdown POST to keep the test on the
    handler-chain spine.
    """
    from unittest.mock import AsyncMock

    from aiperf.operator import client_cache
    from aiperf.operator.handlers import lifecycle

    client_cache._reset_for_testing()

    fake = FakeApiserver()
    fake.add_cr(
        NS,
        AIPERF_PLURAL,
        "delta",
        _ajob_body("delta", {Annotations.BENCHMARK_COMPLETE: "true"}),
    )

    # The controller writes this annotation only after its results-ready
    # handshake. Kopf subsequently dispatches lifecycle handling with that CR.
    cr = fake.crs[(NS, AIPERF_PLURAL, "delta")]
    assert cr["metadata"]["annotations"][Annotations.BENCHMARK_COMPLETE] == "true"

    # Step 2: simulate kopf dispatching on_benchmark_complete with that body.
    # The body kopf would deliver carries the annotation; the claim patch will
    # land on the same fake apiserver and ``handle_completion`` is mocked so
    # we observe only the orchestration spine.
    import kopf

    patch_obj = kopf.Patch()
    handle_completion_mock = AsyncMock()
    progress_client_mock = AsyncMock()
    progress_client_mock.send_shutdown = AsyncMock()
    with (
        fake.context(),
        patch(
            "aiperf.operator.handlers.lifecycle.handle_completion",
            new=handle_completion_mock,
        ),
        patch(
            "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
            new=AsyncMock(return_value=progress_client_mock),
        ),
        patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            new=AsyncMock(),
        ),
    ):
        await lifecycle.on_benchmark_complete(
            body=cr,
            status=cr["status"],
            name="delta",
            namespace=NS,
            patch=patch_obj,
        )

    # ``try_claim_completion`` writes the COMPLETION_CLAIMED annotation via
    # JSON-patch — the fake records it under fake.patches.
    claim_patches = [
        (k, b)
        for k, b in fake.patches
        if k == (NS, AIPERF_PLURAL, "delta") and isinstance(b, list)
    ]
    assert len(claim_patches) == 1, (
        "exactly one JSON-patch claim attempt expected against the AIPerfJob CR"
    )
    handle_completion_mock.assert_awaited_once()
    client_cache._reset_for_testing()
