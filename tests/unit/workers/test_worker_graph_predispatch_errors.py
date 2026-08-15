# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-dispatch graph credit failures must emit a synthetic error record (WK2)."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.models import ErrorDetails
from aiperf.credit.structs import CreditContext
from aiperf.graph.dynamic_pool import GraphPoolMissingError
from aiperf.workers.worker import Worker
from tests.unit.workers.conftest import (
    graph_client_with_node,
    make_graph_credit_context,
    make_graph_worker,
    sole_sent_record,
)


@pytest.mark.asyncio
async def test_store_missing_credit_emits_error_record() -> None:
    """A missing graph store sends a synthetic error InferenceResults record."""
    store_error = ErrorDetails(
        type="GraphStoreUnavailable", message="no store could be opened"
    )

    def _reader(credit_context: CreditContext) -> None:
        # Mirrors the real _graph_store_reader failure contract: attribute the
        # error on the context and return None.
        credit_context.error = store_error
        return None

    self = make_graph_worker(MagicMock(side_effect=_reader))
    ctx = make_graph_credit_context()
    await self._process_graph_credit(ctx, "x-req-1", None)

    record = sole_sent_record(self)
    assert record.error is store_error
    assert record.valid is False
    assert record.request_info.x_request_id == "x-req-1"
    assert record.request_info.x_correlation_id == ctx.credit.x_correlation_id
    # Credit-return semantics preserved: the context still carries the error.
    assert ctx.error is store_error
    # Nothing was dispatched to the inference server.
    self.inference_client.send_request.assert_not_awaited()
    # The drop-message path appends a SECOND record for the same credit unless
    # this early return marks the record as already emitted, which desyncs the
    # RecordsManager completion barrier.
    assert ctx.record_emitted is True


@pytest.mark.asyncio
async def test_envelope_missing_credit_emits_error_record(tmp_path: Path) -> None:
    """An unaddressable node ordinal sends a GraphEnvelopeMissing error record."""
    client = await graph_client_with_node(tmp_path)
    self = make_graph_worker(MagicMock(return_value=client))
    ctx = make_graph_credit_context(node_ordinal=7)
    await self._process_graph_credit(ctx, "x-req-2", None)

    record = sole_sent_record(self)
    assert record.error is not None
    assert record.error.type == "GraphEnvelopeMissing"
    assert record.valid is False
    assert ctx.error is record.error
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_missing_credit_emits_error_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dynamic-pool entry sends an error record carrying the sniffable prefix."""
    # Patch where the name is USED, not where it is defined: worker.py binds
    # these helpers at module scope, so patching aiperf.graph.worker_materialize
    # would leave the worker's own reference untouched.
    import aiperf.workers.worker as worker_mod

    client = await graph_client_with_node(tmp_path, envelope_extra={"items": []})

    def _raise_pool_missing(*args, **kwargs):
        raise GraphPoolMissingError(3)

    monkeypatch.setattr(
        worker_mod, "materialize_graph_request_unified", _raise_pool_missing
    )
    self = make_graph_worker(MagicMock(return_value=client))
    ctx = make_graph_credit_context()
    await self._process_graph_credit(ctx, "x-req-3", None)

    record = sole_sent_record(self)
    assert record.error is not None
    assert record.valid is False
    assert "aiperf.graph.pool_missing:" in record.error.message
    # The context keeps the raw prefixed string the dispatch adapter sniffs.
    assert isinstance(ctx.error, str)
    assert ctx.error.startswith("aiperf.graph.pool_missing:")
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_dispatch_sends_single_record(tmp_path: Path) -> None:
    """Control: the happy path still sends exactly one (dispatch-built) record."""
    client = await graph_client_with_node(tmp_path)
    self = make_graph_worker(MagicMock(return_value=client))
    self._dispatch_graph_request = AsyncMock()
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    ctx = make_graph_credit_context()
    await self._process_graph_credit(ctx, "x-req-4", None)

    self._dispatch_graph_request.assert_awaited_once()
    # No synthetic pre-dispatch record on the happy path.
    self._send_inference_result_message.assert_not_awaited()
    assert ctx.error is None


@pytest.mark.asyncio
async def test_unexpected_exception_emits_error_record_and_credit_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unanticipated raiser still emits one error record AND attributes the credit return."""
    # Without the catch-all the credit returns error=None (counted as a success)
    # while no record flows, starving the RecordsManager barrier. The raiser here
    # is a corrupt envelope failing orjson.loads inside read_node_envelope.
    # Patch where the name is USED (see the note in the pool-missing test above).
    import aiperf.workers.worker as worker_mod

    def _raise_decode_error(*args, **kwargs):
        raise orjson.JSONDecodeError("corrupt envelope bytes", "", 0)

    monkeypatch.setattr(worker_mod, "read_node_envelope", _raise_decode_error)
    self = make_graph_worker(MagicMock(return_value=MagicMock()))
    self._prefill_concurrency_enabled = False
    self.credit_return_push_client.send = AsyncMock()
    # Drive the full credit task so the CreditReturn built in its finally is
    # observable alongside the record.
    self._process_credit = types.MethodType(Worker._process_credit, self)
    self._on_credit_drop_message_task = types.MethodType(
        Worker._on_credit_drop_message_task, self
    )
    ctx = make_graph_credit_context()

    await self._on_credit_drop_message_task(ctx)

    record = sole_sent_record(self)
    assert record.error is not None
    assert record.valid is False
    assert "corrupt envelope bytes" in record.error.message
    # Credit-return attribution: the barrier-side counterpart must see an error.
    assert ctx.error is not None
    credit_return = self.credit_return_push_client.send.await_args.args[0]
    assert credit_return.error is not None, (
        "an escaped exception must not be counted as a completed request"
    )
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_after_dispatch_record_does_not_double_emit(
    tmp_path: Path,
) -> None:
    """A raiser after dispatch resolved attributes the error but emits no second record."""
    # The raiser here is the pool bracket close; the dispatch path already owns
    # the record for this credit.
    client = await graph_client_with_node(tmp_path)
    self = make_graph_worker(MagicMock(return_value=client))
    self._dispatch_graph_request = AsyncMock()
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    self._graph_dynamic_pool.credit_finished = MagicMock(
        side_effect=RuntimeError("post-record boom")
    )
    ctx = make_graph_credit_context()

    await self._process_graph_credit(ctx, "x-req-5", None)

    self._dispatch_graph_request.assert_awaited_once()
    self._send_inference_result_message.assert_not_awaited()
    assert ctx.error is not None


def test_worker_module_imports_graph_helpers_at_module_scope() -> None:
    """The lazy-import workaround is obsolete: graph/__init__.py has no imports.

    Guards against reintroducing function-scope imports on the hot graph
    dispatch path.
    """
    import aiperf.workers.worker as worker_mod

    assert worker_mod.read_node_envelope is not None
    assert worker_mod.materialize_graph_request_unified is not None
    assert worker_mod.materialize_graph_request_unified_bytes is not None
    assert worker_mod.apply_run_level_payload_options is not None
    assert worker_mod.stamp_cache_bust_marker is not None
    assert worker_mod.GraphPoolMissingError is not None


@pytest.mark.asyncio
async def test_fail_graph_credit_always_marks_record_emitted(tmp_path: Path) -> None:
    """The barrier invariant lives in one place, not five copies.

    An errored credit that emits no record starves the RecordsManager
    completion barrier and hangs the run, so record_emitted must be set by the
    same helper that emits.
    """
    self = make_graph_worker(
        MagicMock(return_value=await graph_client_with_node(tmp_path))
    )
    ctx = make_graph_credit_context()
    ctx.record_emitted = False

    await self._fail_graph_credit(ctx, "x-req-fail", error="boom")

    assert ctx.record_emitted is True
    assert ctx.error == "boom"
    assert sole_sent_record(self).error is not None


@pytest.mark.asyncio
async def test_fail_graph_credit_preserves_preset_error(tmp_path: Path) -> None:
    """error=None keeps an error the caller already attributed."""
    self = make_graph_worker(
        MagicMock(return_value=await graph_client_with_node(tmp_path))
    )
    ctx = make_graph_credit_context()
    ctx.error = "preset"
    ctx.record_emitted = False

    await self._fail_graph_credit(ctx, "x-req-preset")

    assert ctx.error == "preset"
    assert ctx.record_emitted is True
