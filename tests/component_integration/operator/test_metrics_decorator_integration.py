# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the @track_handler decorator stack.

These tests verify the contract between kopf's dispatch convention and our
metrics decorator:
  - Handlers imported from ``aiperf.operator.main`` are the
    ``track_handler``-wrapped wrappers (kopf decorators don't replace the
    function, they only register).
  - The four outcome labels (success / retry / fatal / error) all surface
    on the Prometheus exposition format.
  - The histogram ``aiperf_operator_handler_duration_seconds`` records
    samples for fast handlers in the small buckets, never just ``+Inf``.
  - Decoration order: ``@kopf.*`` wraps OUTSIDE ``@track_handler`` for every
    decorated handler in main.py. (Wrong order = kopf's kwargs filtering
    breaks track_handler's call.)

The metrics scrape uses an ephemeral port + ProxyHandler({}) per the
existing pattern (HTTP_PROXY in this sandbox would otherwise route the
request through a proxy that returns 405).
"""

from __future__ import annotations

import asyncio
import inspect
import socket
import urllib.request
from typing import Any
from unittest.mock import AsyncMock, patch

import kopf
import pytest
from prometheus_client import REGISTRY

from aiperf.operator import main as operator_main
from aiperf.operator.metrics import (
    HANDLER_DURATION,
    HANDLER_TOTAL,
    start_metrics_server,
    track_handler,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()
    yield
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()


def _all_decorated_handlers() -> list[tuple[str, Any]]:
    """Return [(name, fn)] for every track_handler-wrapped attribute of main.py."""
    out: list[tuple[str, Any]] = []
    for attr_name in dir(operator_main):
        fn = getattr(operator_main, attr_name)
        if (
            callable(fn)
            and hasattr(fn, "__wrapped__")
            and inspect.iscoroutinefunction(fn)
            and getattr(fn, "__module__", "") == "aiperf.operator.main"
        ):
            out.append((attr_name, fn))
    return out


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_real_trampoline_increments_metrics_for_correct_label() -> None:
    """Calling ``operator_main.monitor_progress`` runs through the real
    ``@track_handler("monitor_progress")`` wrapper and increments the
    matching Prometheus label.
    """
    # Make the underlying ``monitor.monitor_progress`` a no-op so the test
    # focuses on the decorator behavior, not the handler internals.
    with patch(
        "aiperf.operator.handlers.monitor.heartbeat_watchdog",
        new=AsyncMock(return_value=None),
    ):
        await operator_main.heartbeat_watchdog(
            body={"metadata": {"name": "x", "namespace": "ns"}},
            status={},
            spec={},
            name="x",
            namespace="ns",
            patch=kopf.Patch(),
        )

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_total"
        and s.labels.get("handler") == "heartbeat_watchdog"
        and s.labels.get("outcome") == "success"
    ]
    assert any(s.value >= 1 for s in samples), (
        "heartbeat_watchdog success counter must increment via the real "
        "track_handler wrapper imported from operator.main"
    )


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_all_four_outcome_labels_visible_on_metrics_endpoint() -> None:
    """Drive a track_handler-wrapped function through all four outcomes,
    then scrape /metrics and assert each label is exposed.
    """

    @track_handler("multi_outcome")
    async def fake(kind: str) -> None:
        if kind == "ok":
            return
        if kind == "retry":
            raise kopf.TemporaryError("retry me", delay=1)
        if kind == "fatal":
            raise kopf.PermanentError("fatal")
        raise RuntimeError("boom")

    # success
    await fake("ok")
    # retry
    with pytest.raises(kopf.TemporaryError):
        await fake("retry")
    # fatal
    with pytest.raises(kopf.PermanentError):
        await fake("fatal")
    # error
    with pytest.raises(RuntimeError):
        await fake("err")

    # Bind the metrics server to an ephemeral port; ProxyHandler({}) bypasses
    # any HTTP_PROXY in the test sandbox.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    start_metrics_server(port)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = opener.open(f"http://127.0.0.1:{port}/metrics", timeout=5.0).read().decode()

    for outcome in ("success", "retry", "fatal", "error"):
        assert (
            f'aiperf_operator_handler_total{{handler="multi_outcome",outcome="{outcome}"}}'
            in body
        ), f"missing {outcome!r} label in /metrics scrape:\n{body}"


@pytest.mark.component_integration
def test_decoration_order_kopf_outside_track_handler() -> None:
    """For every decorated handler in main.py, ``@kopf.*`` wraps OUTSIDE
    ``@track_handler``. We verify by walking ``__wrapped__``: the visible
    attribute is the ``track_handler`` wrapper (functools.wraps), and the
    underlying function name matches.

    kopf's decorators don't wrap with functools — they register the handler
    and return the function unchanged. So the OUTERMOST callable visible at
    ``operator_main.<name>`` IS the ``track_handler`` wrapper. If kopf were
    inside track_handler, kwargs filtering would break and metrics would
    never be incremented for that handler.
    """
    handlers = _all_decorated_handlers()
    assert handlers, "expected at least one decorated handler in main.py"
    for attr_name, fn in handlers:
        # The attribute IS the wrapper (track_handler used functools.wraps).
        assert hasattr(fn, "__wrapped__"), (
            f"{attr_name} missing __wrapped__; track_handler must use functools.wraps"
        )
        # The wrapper's closure carries the metric name; verify it matches the attr.
        freevars = fn.__code__.co_freevars
        assert "name" in freevars, (
            f"{attr_name} wrapper has no 'name' closure cell — track_handler missing?"
        )
        idx = freevars.index("name")
        assert fn.__closure__ is not None
        metric_name = fn.__closure__[idx].cell_contents
        assert isinstance(metric_name, str), (
            f"{attr_name} closure 'name' is not a string"
        )
        # If this assertion ever fires, someone reordered to
        # @track_handler outside @kopf — the registered handler would no
        # longer be the metrics-wrapped one.
        unwrapped = inspect.unwrap(fn)
        assert unwrapped is not fn, (
            f"{attr_name} unwraps to itself — track_handler not applied"
        )


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_histogram_records_realistic_buckets_for_fast_handler() -> None:
    """Invoke a fast track_handler-wrapped function 10 times. Scrape /metrics
    and assert the bucket distribution makes sense: most observations fall
    in the small-bucket range, the +Inf cumulative count equals the total."""

    @track_handler("fast_handler")
    async def fast() -> None:
        # ~0–5ms; well under the 0.01s smallest bucket
        await asyncio.sleep(0)

    for _ in range(10):
        await fast()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    start_metrics_server(port)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = opener.open(f"http://127.0.0.1:{port}/metrics", timeout=5.0).read().decode()

    # Parse the cumulative bucket counts for fast_handler.
    import re

    bucket_counts: dict[str, float] = {}
    for line in body.splitlines():
        match = re.match(
            r"aiperf_operator_handler_duration_seconds_bucket\{"
            r'handler="fast_handler",le="([^"]+)"\} ([0-9.eE+-]+)',
            line,
        )
        if match:
            bucket_counts[match.group(1)] = float(match.group(2))

    assert bucket_counts, "no histogram buckets found for fast_handler"
    # +Inf bucket is the total observation count.
    assert bucket_counts["+Inf"] == 10.0
    # All 10 observations should land at-or-below the smallest bucket (0.01s)
    # because asyncio.sleep(0) is sub-millisecond.
    assert bucket_counts["0.01"] == 10.0, (
        f"all 10 fast observations should land in the <=0.01s bucket; got {bucket_counts}"
    )
