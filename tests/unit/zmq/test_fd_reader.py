# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct unit tests for the FdEdgeReader edge-triggered ZMQ FD driver.

The client tests exercise the driver indirectly through the busy-loop-safe
subclass in ``conftest._TestSafeFdEdgeReader``; these drive the real
:class:`FdEdgeReader` against a minimal fake socket so the send path, the
``_pump`` send-drain branch, the error boundary, and the ``_rearm`` reschedule
are covered without a live ZMQ context.
"""

import asyncio
import socket as stdlib_socket
from collections import deque

import pytest
import zmq

from aiperf.zmq.fd_reader import FdEdgeReader


class _FakeSocket:
    """Stand-in exposing ``getsockopt(FD/EVENTS)`` for the FD driver.

    ``events`` is the current ``ZMQ_EVENTS`` bitmask (POLLIN/POLLOUT) the driver
    re-reads each pass. Set ``events_error`` to make ``getsockopt(EVENTS)`` raise.
    """

    def __init__(self, fd: int = -1, events: int = 0) -> None:
        self._fd = fd
        self.events = events
        self.events_error: BaseException | None = None

    def getsockopt(self, opt: int) -> int:
        if opt == zmq.FD:
            return self._fd
        if opt == zmq.EVENTS:
            if self.events_error is not None:
                raise self.events_error
            return self.events
        return 0


def _make_reader(sock, **kwargs):
    """Build a reader with no-op defaults so tests override only what they need."""
    kwargs.setdefault("recv_one", lambda: (_ for _ in ()).throw(zmq.Again()))
    kwargs.setdefault("dispatch", lambda item: None)
    kwargs.setdefault("batch_limit", 10)
    return FdEdgeReader(socket=sock, **kwargs)


# =============================================================================
# start() / stop()
# =============================================================================


@pytest.mark.asyncio
async def test_start_registers_fd_and_bootstrap_drains():
    """start() installs the reader and the bootstrap pump drains queued items."""
    r_sock, w_sock = stdlib_socket.socketpair()
    r_sock.setblocking(False)
    sock = _FakeSocket(fd=r_sock.fileno(), events=zmq.POLLIN)
    received: list[bytes] = []
    items = iter([b"a", b"b"])

    def recv_one():
        try:
            return next(items)
        except StopIteration:
            sock.events = 0  # buffer drained -> POLLIN clears, mirrors ZMQ_EVENTS
            raise zmq.Again() from None

    reader = _make_reader(sock, recv_one=recv_one, dispatch=received.append)
    try:
        reader.start()
        await asyncio.sleep(0)  # let any rearm reschedule run
        assert received == [b"a", b"b"]
        assert reader._fd == r_sock.fileno()
    finally:
        reader.stop()
        r_sock.close()
        w_sock.close()

    assert reader._stopped is True
    assert reader._fd is None


def test_stop_without_start_is_safe():
    """stop() before start() (no fd/loop) is a no-op that still marks stopped."""
    reader = _make_reader(_FakeSocket())
    reader.stop()
    assert reader._stopped is True


def test_batch_limit_nonpositive_falls_back_to_256():
    """A <=0 batch_limit clamps to the 256 default."""
    assert _make_reader(_FakeSocket(), batch_limit=0)._batch_limit == 256
    assert _make_reader(_FakeSocket(), batch_limit=-5)._batch_limit == 256


# =============================================================================
# send()
# =============================================================================


def test_send_without_send_one_raises():
    """send() with no send_one callable is a programming error."""
    reader = _make_reader(_FakeSocket())
    with pytest.raises(RuntimeError, match="without a send_one"):
        reader.send(b"x")


def test_send_immediate_when_buffer_empty():
    """An empty buffer sends synchronously and does not buffer."""
    sent: list[bytes] = []
    reader = _make_reader(_FakeSocket(events=0), send_one=sent.append)
    reader.send(b"x")
    assert sent == [b"x"]
    assert not reader._send_buf


def test_send_buffers_on_hwm_again():
    """A NOBLOCK send hitting the HWM (zmq.Again) buffers the item instead."""

    def send_one(_item):
        raise zmq.Again()

    reader = _make_reader(_FakeSocket(events=0), send_one=send_one)
    reader.send(b"x")
    assert list(reader._send_buf) == [b"x"]


def test_send_preserves_order_when_already_buffered():
    """Once something is queued behind a full HWM, later sends append in order."""
    sent: list[bytes] = []
    reader = _make_reader(_FakeSocket(events=0), send_one=sent.append)
    reader._send_buf.append(b"first")
    reader.send(b"second")
    # send_one must not be called while a backlog exists; ordering preserved.
    assert sent == []
    assert list(reader._send_buf) == [b"first", b"second"]


# =============================================================================
# _pump()
# =============================================================================


def test_pump_noop_when_stopped():
    """A stopped reader pumps nothing."""
    dispatched: list = []
    reader = _make_reader(
        _FakeSocket(events=zmq.POLLIN),
        recv_one=lambda: b"x",
        dispatch=dispatched.append,
    )
    reader._stopped = True
    reader._pump()
    assert dispatched == []


def test_pump_breaks_on_recv_again():
    """POLLIN set but recv immediately Again -> no dispatch, clean exit."""
    dispatched: list = []
    reader = _make_reader(
        _FakeSocket(events=zmq.POLLIN),
        recv_one=lambda: (_ for _ in ()).throw(zmq.Again()),
        dispatch=dispatched.append,
    )
    reader._pump()
    assert dispatched == []


def test_pump_drains_send_backlog_on_pollout():
    """With POLLOUT and a backlog, _pump flushes buffered sends until Again."""
    sock = _FakeSocket(events=zmq.POLLOUT)
    sent: list[bytes] = []
    calls = {"n": 0}

    def send_one(item):
        calls["n"] += 1
        if calls["n"] == 1:
            sent.append(item)
            return
        raise zmq.Again()  # HWM hit again on the second buffered item

    reader = _make_reader(sock, send_one=send_one)
    reader._send_buf.extend([b"a", b"b"])
    reader._pump()

    assert sent == [b"a"]
    # b"a" popped, b"b" stayed buffered after the HWM Again.
    assert list(reader._send_buf) == [b"b"]


def test_pump_swallows_zmq_error():
    """ContextTerminated / ZMQError during a pass ends the pass without on_error."""
    errors: list[Exception] = []
    reader = _make_reader(
        _FakeSocket(events=zmq.POLLIN),
        recv_one=lambda: (_ for _ in ()).throw(zmq.ContextTerminated()),
        on_error=errors.append,
    )
    reader._pump()
    assert errors == []


def test_pump_reports_unexpected_exception_to_on_error():
    """A non-ZMQ exception is reported via on_error and ends the pass."""
    errors: list[Exception] = []
    boom = RuntimeError("boom")
    reader = _make_reader(
        _FakeSocket(events=zmq.POLLIN),
        recv_one=lambda: (_ for _ in ()).throw(boom),
        on_error=errors.append,
    )
    reader._pump()
    assert errors == [boom]


def test_pump_unexpected_exception_without_on_error_is_swallowed():
    """Without an on_error callback the exception is still contained."""
    reader = _make_reader(
        _FakeSocket(events=zmq.POLLIN),
        recv_one=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    reader._pump()  # must not raise


@pytest.mark.asyncio
async def test_pump_rearms_after_error_so_drain_continues():
    """After a non-ZMQ error mid-drain, _pump re-arms (POLLIN still set) and the
    rescheduled pass drains the messages queued behind the poisoned one.

    Without the re-arm, the edge-triggered FD would never re-fire for the already
    queued backlog and those messages would be stranded.
    """
    sock = _FakeSocket(events=zmq.POLLIN)
    errors: list[Exception] = []
    dispatched: list[int] = []
    calls = {"n": 0}

    def recv_one():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("poison")  # decode-style failure; frame consumed
        if calls["n"] <= 3:
            return calls["n"]  # two good messages queued behind the poison
        sock.events = 0  # buffer drained
        raise zmq.Again()

    reader = _make_reader(
        sock, recv_one=recv_one, dispatch=dispatched.append, on_error=errors.append
    )
    reader._loop = asyncio.get_running_loop()

    reader._pump()
    # error reported AND a continuation scheduled (not stranded)
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert reader._rearm_pending is True

    for _ in range(5):
        await asyncio.sleep(0)
        if not reader._rearm_pending and sock.events == 0:
            break

    assert dispatched == [2, 3]


# =============================================================================
# _rearm() / _run()
# =============================================================================


def test_rearm_noop_without_loop():
    """No loop -> nothing scheduled."""
    reader = _make_reader(_FakeSocket(events=zmq.POLLIN))
    reader._rearm(zmq.POLLIN)
    assert reader._rearm_pending is False


@pytest.mark.asyncio
async def test_rearm_noop_when_already_pending():
    """An in-flight rearm is not double-scheduled."""
    reader = _make_reader(_FakeSocket(events=zmq.POLLIN))
    reader._loop = asyncio.get_running_loop()
    reader._rearm_pending = True
    reader._rearm(zmq.POLLIN)
    # Still exactly one pending flag, no extra callback churn.
    assert reader._rearm_pending is True


@pytest.mark.asyncio
async def test_rearm_swallows_getsockopt_error():
    """A ZMQError while re-reading EVENTS ends the rearm quietly."""
    sock = _FakeSocket(events=zmq.POLLIN)
    sock.events_error = zmq.ZMQError()
    reader = _make_reader(sock)
    reader._loop = asyncio.get_running_loop()
    reader._rearm()  # events=None -> getsockopt raises -> swallowed
    assert reader._rearm_pending is False


@pytest.mark.asyncio
async def test_rearm_schedules_run_on_pollin():
    """POLLIN pending schedules exactly one _run, which pumps and clears pending."""
    sock = _FakeSocket(events=zmq.POLLIN)
    dispatched: list = []
    buf = deque([b"only"])

    def recv_one():
        if not buf:
            raise zmq.Again()
        item = buf.popleft()
        if not buf:
            sock.events = 0  # buffer now empty: ZMQ_EVENTS clears POLLIN at once
        return item

    reader = _make_reader(sock, recv_one=recv_one, dispatch=dispatched.append)
    reader._loop = asyncio.get_running_loop()

    reader._rearm(zmq.POLLIN)
    assert reader._rearm_pending is True

    await asyncio.sleep(0)  # _run executes

    assert reader._rearm_pending is False
    assert dispatched == [b"only"]
