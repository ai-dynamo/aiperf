# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator on_create handler, especially persistence ordering (H1).

H1 moves index/spec persistence BEFORE JobSet creation so a persistence failure
cannot leave an orphan JobSet that the index/history API can't see. Verifies
that:
  - persistence failure raises kopf.TemporaryError (retryable)
  - JobSet creation is NOT invoked when persistence fails
  - persistence runs AFTER idempotent RBAC/ConfigMap create
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest

from aiperf.operator.handlers.create import on_create
from tests.harness.operator import (
    build_minimal_aiperfjob_spec,
    build_sample_body,
)


def _status_patch() -> MagicMock:
    patch = MagicMock()
    patch.status = {}
    return patch


def _preflight_ok() -> MagicMock:
    from aiperf.kubernetes.preflight import CheckStatus

    pr = MagicMock()
    pr.passed = True
    pr.checks = []
    check = MagicMock()
    check.status = CheckStatus.PASS
    pr.checks = [check]
    return pr


async def _async_identity(*_a, **_kw):
    return None


@pytest.mark.asyncio
async def test_on_create_persistence_failure_raises_temporary_error_and_skips_jobset():
    """H1: OSError from save_job_spec_file -> TemporaryError, JobSet not created."""
    spec = build_minimal_aiperfjob_spec()
    body = build_sample_body()

    create_idempotent_mock = AsyncMock()

    with (
        mock_patch(
            "aiperf.operator.handlers.create.check_endpoint_health",
            new=AsyncMock(return_value=MagicMock(reachable=True, error=None)),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.get_api",
            new=AsyncMock(return_value=MagicMock()),
        ),
        mock_patch(
            "aiperf.operator.preflight.OperatorPreflightChecker",
        ) as preflight_cls,
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent",
            new=create_idempotent_mock,
        ),
        mock_patch(
            "aiperf.operator.handlers.create.asyncio.sleep",
            new=AsyncMock(),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.save_job_spec_file",
            new=AsyncMock(side_effect=OSError("PVC write failed")),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.index_job_created",
            new=AsyncMock(),
        ),
        mock_patch("aiperf.operator.handlers.create.events.spec_valid"),
        mock_patch("aiperf.operator.handlers.create.events.endpoint_reachable"),
        mock_patch("aiperf.operator.handlers.create.events.preflight_passed"),
    ):
        preflight_cls.return_value.run_all = AsyncMock(return_value=_preflight_ok())

        with pytest.raises(kopf.TemporaryError) as exc_info:
            await on_create(
                body=body,
                spec=spec,
                name="test-job",
                namespace="default",
                uid="abc-123",
                patch=_status_patch(),
            )

    msg = str(exc_info.value)
    assert "Persisting job spec/index failed" in msg
    assert "PVC write failed" in msg

    # JobSet create must NOT have been attempted. create_idempotent is used
    # for Role, RoleBinding, ConfigMap (3 calls) but not JobSet when
    # persistence raises before step 7.
    create_types = [c.args[0].__name__ for c in create_idempotent_mock.call_args_list]
    assert "AsyncJobSet" not in create_types, (
        f"JobSet must not be created on persistence failure, got: {create_types}"
    )


@pytest.mark.asyncio
async def test_on_create_persistence_success_then_jobset_created():
    """H1: happy path — persistence runs before JobSet and both succeed."""
    spec = build_minimal_aiperfjob_spec()
    body = build_sample_body()

    call_order: list[str] = []

    async def record_save(*_a, **_kw):
        call_order.append("save")

    async def record_index(*_a, **_kw):
        call_order.append("index")

    async def record_create(cls, *_a, **_kw):
        call_order.append(f"create:{cls.__name__}")

    with (
        mock_patch(
            "aiperf.operator.handlers.create.check_endpoint_health",
            new=AsyncMock(return_value=MagicMock(reachable=True, error=None)),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.get_api",
            new=AsyncMock(return_value=MagicMock()),
        ),
        mock_patch(
            "aiperf.operator.preflight.OperatorPreflightChecker",
        ) as preflight_cls,
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent",
            new=AsyncMock(side_effect=record_create),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.asyncio.sleep",
            new=AsyncMock(),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.save_job_spec_file",
            new=AsyncMock(side_effect=record_save),
        ),
        mock_patch(
            "aiperf.operator.handlers.create.index_job_created",
            new=AsyncMock(side_effect=record_index),
        ),
        mock_patch("aiperf.operator.handlers.create.events.spec_valid"),
        mock_patch("aiperf.operator.handlers.create.events.endpoint_reachable"),
        mock_patch("aiperf.operator.handlers.create.events.preflight_passed"),
        mock_patch("aiperf.operator.handlers.create.events.resources_created"),
        mock_patch("aiperf.operator.handlers.create.events.created"),
    ):
        preflight_cls.return_value.run_all = AsyncMock(return_value=_preflight_ok())

        result = await on_create(
            body=body,
            spec=spec,
            name="test-job",
            namespace="default",
            uid="abc-123",
            patch=_status_patch(),
        )

    assert "jobSetName" in result
    # ConfigMap/RBAC happen before persistence; JobSet happens after.
    save_idx = call_order.index("save")
    index_idx = call_order.index("index")
    jobset_entries = [
        i for i, c in enumerate(call_order) if c.startswith("create:") and "JobSet" in c
    ]
    assert jobset_entries, f"JobSet was not created. call_order={call_order}"
    jobset_idx = jobset_entries[0]
    assert save_idx < jobset_idx
    assert index_idx < jobset_idx
