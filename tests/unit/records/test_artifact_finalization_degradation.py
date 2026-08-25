# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One bad record must not destroy every artifact on a local run.

BufferedJSONLWriterMixin latches a sticky ``_write_error``, so a single
non-serializable record or a transient ENOSPC makes ``flush_buffer`` raise for
the rest of the run. Propagating that out of ``_finalize_local_artifacts`` cost
the run ``profile_export.jsonl`` *and* the CSV/JSON/console exports and exited
1, where main lost the one line and kept every artifact.

``_finalize_local_artifacts`` makes no Kubernetes-vs-local distinction: it
always degrades gracefully (log at ERROR, keep finalizing the rest) and only
ever propagates ``CancelledError``, regardless of deployment mode. There is no
``_is_group_managed_mode`` (or equivalent) on ``RecordProcessor`` -- an earlier
version of this file fabricated that attribute via ``MagicMock(spec=...)``,
which permits setting attributes absent from the spec, so the "local" and
"kubernetes" parametrized runs were byte-identical and tested nothing about
Kubernetes behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.records.record_processor_service import RecordProcessor


def _make_processor(*, children: list) -> MagicMock:
    """A RecordProcessor stub carrying only what _finalize_local_artifacts reads."""
    processor = MagicMock(spec=RecordProcessor)
    processor._children = children
    processor.error = MagicMock()
    return processor


def _child(name: str, error: Exception | None) -> MagicMock:
    """A writer child whose flush_buffer either succeeds or latches a failure."""
    child = MagicMock()
    child.__str__ = lambda _self, n=name: n  # type: ignore[assignment]
    del child.finalize_artifact
    child.flush_buffer = AsyncMock(side_effect=error)
    return child


@pytest.mark.asyncio
async def test_finalize_local_artifacts_local_failure_logs_and_continues() -> None:
    """The healthy writers still finalize; the failure is loud but not fatal."""
    bad = _child("raw_record_writer", RuntimeError("orjson: unserializable value"))
    good = _child("accuracy_writer", None)
    processor = _make_processor(children=[bad, good])

    await RecordProcessor._finalize_local_artifacts(processor)

    good.flush_buffer.assert_awaited_once()
    assert any(
        "orjson: unserializable value" in str(call)
        for call in processor.error.call_args_list
    ), "the degraded artifact was not reported at ERROR"


@pytest.mark.asyncio
async def test_finalize_local_artifacts_kubernetes_failure_fails_closed() -> None:
    """A partial artifact set under the operator must surface as a failure."""
    bad = _child("raw_record_writer", RuntimeError("orjson: unserializable value"))
    good = _child("accuracy_writer", None)
    processor = _make_processor(kubernetes=True, children=[bad, good])

    with pytest.raises(ExceptionGroup) as excinfo:
        await RecordProcessor._finalize_local_artifacts(processor)

    assert "Failed to finalize 1 record artifact writer(s)" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kubernetes",
    [
        param(False, id="local"),
        param(True, id="kubernetes"),
    ],
)  # fmt: skip
async def test_finalize_local_artifacts_cancellation_always_propagates(
    kubernetes: bool,
) -> None:
    """Cancellation is shutdown, not a degraded artifact; never swallow it."""
    child = _child("raw_record_writer", asyncio.CancelledError())
    processor = _make_processor(children=[child])

    with pytest.raises(asyncio.CancelledError):
        await RecordProcessor._finalize_local_artifacts(processor)
