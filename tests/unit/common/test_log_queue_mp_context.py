# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the global log queue must match the subprocess start method.

CPython refuses to share a SemLock across start methods ("A SemLock created in
a fork context is being shared with a process in a spawn context"). Service
processes are spawned via ``get_mp_context()``, so the log queue handed to them
must be built from the same context. Building it with the bare
``multiprocessing.Queue`` constructor picks the process-global default (``fork``
on Linux) and breaks every interactive Dashboard run at service spawn.
"""

import multiprocessing
import multiprocessing.queues
from unittest.mock import MagicMock, patch

import pytest

import aiperf.common.logging as logging_module
from aiperf.common.logging import get_global_log_queue
from aiperf.common.mp_context import get_mp_context


@pytest.fixture(autouse=True)
def _reset_global_log_queue():
    """Drop the module singleton so each test builds a fresh queue."""
    logging_module._global_log_queue = None
    yield
    queue = logging_module._global_log_queue
    if isinstance(queue, multiprocessing.queues.Queue):
        queue.close()
        queue.join_thread()
    logging_module._global_log_queue = None


def test_global_log_queue_built_from_subprocess_mp_context() -> None:
    """The queue must come from get_mp_context(), not the bare constructor.

    The bare ``multiprocessing.Queue`` uses the process-global default context
    (``fork`` on Linux) while services are spawned from ``get_mp_context()``
    (``forkserver`` on Linux), and CPython rejects the cross-context SemLock.
    """
    sentinel = MagicMock(name="context_queue")
    mock_ctx = MagicMock()
    mock_ctx.Queue.return_value = sentinel

    with patch("aiperf.common.mp_context.get_mp_context", return_value=mock_ctx):
        log_queue = get_global_log_queue()

    assert log_queue is sentinel
    mock_ctx.Queue.assert_called_once()


def _child_touches_queue(log_queue: multiprocessing.Queue) -> None:
    """Child entrypoint: proves the queue survived the start-method handoff."""
    log_queue.put("ok")


def test_global_log_queue_accepted_by_spawned_child() -> None:
    """A child started via get_mp_context() must accept the log queue.

    This is the end-to-end shape of the production failure: the parent passes
    the log queue as a Process kwarg, and a context mismatch raises RuntimeError
    inside ``Process.start()`` before the child ever runs.
    """
    log_queue = get_global_log_queue()

    process = get_mp_context().Process(
        target=_child_touches_queue,
        kwargs={"log_queue": log_queue},
        daemon=True,
    )
    process.start()
    try:
        assert log_queue.get(timeout=30) == "ok"
    finally:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=30)

    assert process.exitcode == 0
