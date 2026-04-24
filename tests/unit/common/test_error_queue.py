# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import time

import msgspec

from aiperf.common.error_queue import (
    _ERROR_QUEUE_MAXSIZE,
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


def _report_and_wait(q: multiprocessing.Queue, errors: list[ExitErrorInfo]) -> None:
    """Report errors and wait for them to be enqueued by the background thread."""
    report_errors(q, errors)
    time.sleep(0.001)


class TestReportErrors:
    def test_report_errors_puts_serialized_errors_on_queue(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        err = _make_error(service_id="worker_abc", message="something broke")

        _report_and_wait(q, [err])

        item = q.get(timeout=1)
        assert isinstance(item, dict)
        error_info = msgspec.convert(item, ExitErrorInfo)
        assert error_info.service_id == "worker_abc"
        assert error_info.operation == "service_run"
        assert "something broke" in error_info.error_details.message

    def test_report_errors_multiple(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        errors = [
            _make_error(service_id="svc_a", operation="initialize"),
            _make_error(service_id="svc_b", operation="start"),
        ]

        _report_and_wait(q, errors)

        item1 = q.get(timeout=1)
        item2 = q.get(timeout=1)
        assert msgspec.convert(item1, ExitErrorInfo).service_id == "svc_a"
        assert msgspec.convert(item2, ExitErrorInfo).service_id == "svc_b"

    def test_report_errors_drops_when_queue_full(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
        q.put("filler")
        time.sleep(0.001)

        # Should not raise
        report_errors(q, [_make_error()])
        time.sleep(0.001)

        assert q.get(timeout=1) == "filler"
        assert q.empty()

    def test_report_errors_empty_list_is_noop(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        report_errors(q, [])
        assert q.empty()


class TestDrainErrorQueue:
    def test_drain_returns_empty_list_on_empty_queue(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        assert drain_error_queue(q) == []

    def test_drain_returns_all_errors(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        errors = [_make_error(service_id=f"worker_{i}") for i in range(3)]
        _report_and_wait(q, errors)

        drained = drain_error_queue(q)
        assert len(drained) == 3
        assert all(isinstance(e, ExitErrorInfo) for e in drained)
        assert {e.service_id for e in drained} == {
            "worker_0",
            "worker_1",
            "worker_2",
        }

    def test_drain_accepts_exit_error_info_objects(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        error_info = _make_error(service_id="test_svc")
        q.put(error_info)
        time.sleep(0.001)

        errors = drain_error_queue(q)
        assert len(errors) == 1
        assert errors[0].service_id == "test_svc"

    def test_drain_skips_malformed_items(self) -> None:
        q: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        q.put("not a valid error")
        q.put({"error_details": "invalid"})
        time.sleep(0.001)

        _report_and_wait(q, [_make_error(service_id="good")])

        errors = drain_error_queue(q)
        assert len(errors) == 1
        assert errors[0].service_id == "good"


class TestErrorQueueMaxsize:
    def test_maxsize_is_reasonable(self) -> None:
        assert _ERROR_QUEUE_MAXSIZE > 0
        assert _ERROR_QUEUE_MAXSIZE <= 1024
