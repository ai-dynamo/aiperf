# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records-manager handling of per-row endpoint routing.

Records produced by an endpoint other than the run-level one are counted but
never fed to the accumulators: metric applicability is a property of the
endpoint (embeddings declares ``produces_tokens: false``), so mixing them into
one metric set would compute token metrics over requests that produce no
tokens.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import MetricRecordsData, RecordsMessage
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager import RecordsManager


def _make_manager(run_endpoint_type: str = "chat") -> RecordsManager:
    """Build a RecordsManager stub with only the fields _on_records touches."""
    manager = RecordsManager.__new__(RecordsManager)
    manager.debug = MagicMock()
    manager.error = MagicMock()
    manager.trace = MagicMock()
    manager.is_enabled_for = MagicMock(return_value=False)
    manager._dataset_configured_event = asyncio.Event()
    manager._dataset_configured_event.set()
    manager._records_tracker = MagicMock()
    manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
    manager._error_tracker = ErrorTracker()
    manager._complete_credit_phases = set()
    manager._warned_missing_cache_reporting = False
    manager._failed_request_threshold = None
    manager._failed_request_abort_triggered = False
    manager._skipped_context_overflow_counts_by_phase = {
        CreditPhase.WARMUP: 0,
        CreditPhase.PROFILING: 0,
    }
    manager._unmeasured_endpoint_counts = {}
    manager._dispatch_record = AsyncMock(return_value=[])

    run = MagicMock()
    run.cfg.endpoint.type = run_endpoint_type
    manager.run = run
    return manager


def _records_message(
    endpoint_type: str | None, error: ErrorDetails | None = None
) -> RecordsMessage:
    metadata = MetricRecordMetadata(
        session_num=0,
        conversation_id="test",
        turn_index=0,
        request_start_ns=1_000,
        request_end_ns=2_000,
        worker_id="worker-1",
        record_processor_id="processor-1",
        benchmark_phase=CreditPhase.PROFILING,
        endpoint_type=endpoint_type,
    )
    return RecordsMessage(
        service_id="rp",
        metadata=metadata,
        records=[MetricRecordsData(metadata=metadata, metrics={})],
        error=error,
    )


@pytest.mark.asyncio
async def test_secondary_endpoint_record_is_counted_not_measured() -> None:
    manager = _make_manager()

    await manager._on_records(_records_message("embeddings"))

    assert manager._unmeasured_endpoint_counts == {
        CreditPhase.PROFILING: {"embeddings": 1}
    }
    manager._dispatch_record.assert_not_called()


@pytest.mark.asyncio
async def test_primary_endpoint_record_still_measured() -> None:
    """A record tagged with the run-level endpoint is measured normally."""
    manager = _make_manager()

    await manager._on_records(_records_message("chat"))

    assert manager._unmeasured_endpoint_counts == {}
    manager._dispatch_record.assert_called_once()


@pytest.mark.asyncio
async def test_untagged_record_still_measured() -> None:
    """Records from datasets without per-row routing carry no endpoint_type."""
    manager = _make_manager()

    await manager._on_records(_records_message(None))

    assert manager._unmeasured_endpoint_counts == {}
    manager._dispatch_record.assert_called_once()


@pytest.mark.asyncio
async def test_counts_accumulate_per_endpoint() -> None:
    manager = _make_manager()

    await manager._on_records(_records_message("embeddings"))
    await manager._on_records(_records_message("embeddings"))
    await manager._on_records(_records_message("rankings"))

    assert manager._unmeasured_endpoint_counts == {
        CreditPhase.PROFILING: {"embeddings": 2, "rankings": 1}
    }


@pytest.mark.asyncio
async def test_failure_on_secondary_endpoint_is_still_an_error() -> None:
    """Unlike the context-overflow skip, these errors are real failures.

    Not measuring a request's latency is not the same as forgiving its failure;
    a 500 from the embeddings endpoint must still count against
    --failed-request-threshold.
    """
    manager = _make_manager()
    error = ErrorDetails(code=500, type="ServerError", message="boom")

    await manager._on_records(_records_message("embeddings", error=error))

    summary = manager._error_tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
    assert any(item.error_details == error for item in summary)
    manager._records_tracker.update_from_request.assert_called_once()
    assert manager._records_tracker.update_from_request.call_args.args[1] == error


@pytest.mark.asyncio
async def test_counts_are_partitioned_by_phase() -> None:
    """Warmup routing must not inflate the profiling-phase report.

    _process_results reads the counts for the phase it is summarizing; a flat
    dict would report warmup requests as profiling ones.
    """
    manager = _make_manager()

    warmup = _records_message("embeddings")
    warmup.metadata.benchmark_phase = CreditPhase.WARMUP
    await manager._on_records(warmup)
    await manager._on_records(_records_message("embeddings"))

    assert manager._unmeasured_endpoint_counts == {
        CreditPhase.WARMUP: {"embeddings": 1},
        CreditPhase.PROFILING: {"embeddings": 1},
    }
