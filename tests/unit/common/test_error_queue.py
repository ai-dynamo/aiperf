# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the subprocess error backchannel queue."""

import multiprocessing
import queue as queue_mod
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiperf.common.bootstrap import _report_service_errors
from aiperf.common.environment import Environment
from aiperf.common.error_queue import (
    ErrorCollector,
    drain_error_queue,
    report_errors,
)
from aiperf.common.models.error_models import ErrorDetails, ExitErrorInfo


def _make_error(
    service_id: str = "worker_abc",
    operation: str = "service_run",
    message: str = "something broke",
) -> ExitErrorInfo:
    return ExitErrorInfo(
        error_details=ErrorDetails(type="RuntimeError", message=message),
        operation=operation,
        service_id=service_id,
    )


def _wait_for_items(q: multiprocessing.Queue, count: int) -> None:
    """Wait until the queue's feeder thread has flushed *count* items.

    multiprocessing.Queue.put_nowait() hands off to a background feeder
    thread, so a caller that immediately inspects the queue can race it.
    Bounded by iteration count rather than a wall-clock deadline because the
    unit-test suite patches clock functions.
    """
    for _ in range(5000):
        if q.qsize() >= count:
            return
        time.sleep(0.001)


def _drain_queue() -> queue_mod.Queue:
    """Return a queue with ``get_nowait`` semantics but no feeder-thread race.

    ``drain_error_queue`` only requires ``get_nowait``; using a thread queue
    here keeps the non-blocking drain tests deterministic.
    """
    return queue_mod.Queue()


def _report_and_wait(q: multiprocessing.Queue, errors: list[ExitErrorInfo]) -> None:
    report_errors(q, errors)
    _wait_for_items(q, len(errors))


class TestReportErrors:
    def test_report_errors_puts_serialized_errors_on_queue(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        _report_and_wait(q, [_make_error(service_id="worker_abc")])

        item = q.get(timeout=5)
        assert isinstance(item, dict)
        error_info = ExitErrorInfo.model_validate(item)
        assert error_info.service_id == "worker_abc"
        assert error_info.operation == "service_run"
        assert "something broke" in error_info.error_details.message

    def test_report_errors_multiple(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        _report_and_wait(
            q,
            [
                _make_error(service_id="svc_a", operation="initialize"),
                _make_error(service_id="svc_b", operation="start"),
            ],
        )

        first = ExitErrorInfo.model_validate(q.get(timeout=5))
        second = ExitErrorInfo.model_validate(q.get(timeout=5))
        assert first.service_id == "svc_a"
        assert second.service_id == "svc_b"

    def test_report_errors_drops_when_queue_full(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
        q.put("filler")
        _wait_for_items(q, 1)

        report_errors(q, [_make_error()])

        assert q.get(timeout=5) == "filler"
        assert q.empty()

    def test_report_errors_empty_list_is_noop(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        report_errors(q, [])
        assert q.empty()


class TestDrainErrorQueue:
    def test_drain_returns_empty_list_on_empty_queue(self) -> None:
        assert drain_error_queue(_drain_queue()) == []

    def test_drain_returns_all_errors(self) -> None:
        q = _drain_queue()
        report_errors(q, [_make_error(service_id=f"worker_{i}") for i in range(3)])

        drained = drain_error_queue(q)
        assert len(drained) == 3
        assert all(isinstance(e, ExitErrorInfo) for e in drained)
        assert {e.service_id for e in drained} == {"worker_0", "worker_1", "worker_2"}

    def test_drain_accepts_exit_error_info_objects(self) -> None:
        q = _drain_queue()
        q.put(_make_error(service_id="test_svc"))

        errors = drain_error_queue(q)
        assert len(errors) == 1
        assert errors[0].service_id == "test_svc"

    def test_drain_skips_malformed_items(self) -> None:
        q = _drain_queue()
        q.put("not a valid error")
        q.put({"error_details": "invalid"})
        report_errors(q, [_make_error(service_id="good")])

        errors = drain_error_queue(q)
        assert len(errors) == 1
        assert errors[0].service_id == "good"


class TestErrorQueueMaxsize:
    def test_maxsize_is_bounded(self) -> None:
        assert 0 < Environment.SERVICE.ERROR_QUEUE_MAXSIZE <= 1024


class TestErrorCollector:
    def test_drain_into_logs_and_extends_exit_errors(self, monkeypatch) -> None:
        q = _drain_queue()
        monkeypatch.setattr(
            "aiperf.common.error_queue.get_global_error_queue", lambda: q
        )
        logger = MagicMock()
        exit_errors: list[ExitErrorInfo] = []
        collector = ErrorCollector(logger=logger, exit_errors=exit_errors)

        report_errors(q, [_make_error(service_id="worker_9")])
        drained = collector.drain_into()

        assert len(drained) == 1
        assert exit_errors == drained
        assert "worker_9" in logger.error.call_args.args[0]


class TestBootstrapReportsToQueue:
    """The producer half of the backchannel: children put errors on the queue."""

    def test_report_service_errors_puts_accumulated_errors(self) -> None:
        """A failed child's ``_exit_errors`` reach the parent before it exits."""
        service = SimpleNamespace(
            _exit_errors=[_make_error(service_id="worker_0", message="boom")]
        )
        queue_ = queue_mod.Queue()
        _report_service_errors(service, queue_)

        drained = drain_error_queue(queue_)
        assert [e.service_id for e in drained] == ["worker_0"]

    def test_report_service_errors_without_queue_is_noop(self) -> None:
        """Local multiprocessing mode supplies no queue; that must not raise."""
        service = SimpleNamespace(
            _exit_errors=[_make_error(service_id="worker_0", message="boom")]
        )
        _report_service_errors(service, None)

    def test_report_service_errors_healthy_service_puts_nothing(self) -> None:
        queue_ = queue_mod.Queue()
        _report_service_errors(SimpleNamespace(_exit_errors=[]), queue_)
        assert drain_error_queue(queue_) == []
