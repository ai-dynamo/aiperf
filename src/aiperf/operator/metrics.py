# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for the AIPerf kopf operator.

Exposes a /metrics HTTP endpoint on a dedicated port (default 9090) so cluster
operators can scrape reconcile latency, error rate, and per-handler outcome
counts without going through the FastAPI sidecar (which runs in a separate
container and cannot see the kopf process's in-memory state).

Metrics:
    aiperf_operator_handler_duration_seconds{handler}      Histogram
    aiperf_operator_handler_total{handler, outcome}        Counter
    aiperf_operator_completion_claim_races_total           Counter

Usage:
    @track_handler("monitor_progress")
    async def monitor_progress(...): ...

    # in main.py @kopf.on.startup:
    start_metrics_server(port=OperatorEnvironment.METRICS_PORT)
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

T = TypeVar("T")

HANDLER_DURATION = Histogram(
    "aiperf_operator_handler_duration_seconds",
    "Wall-clock duration of an AIPerf kopf handler invocation, by handler name.",
    labelnames=("handler",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

HANDLER_TOTAL = Counter(
    "aiperf_operator_handler_total",
    "Total number of AIPerf kopf handler invocations, by handler name and outcome.",
    labelnames=("handler", "outcome"),
)

COMPLETION_CLAIM_RACES = Counter(
    "aiperf_operator_completion_claim_races_total",
    "Number of times try_claim_completion lost the race to a peer tick or replica.",
)


def track_handler(name: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that records duration and outcome for an async kopf handler."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            outcome = "success"
            try:
                return await func(*args, **kwargs)
            except Exception:
                outcome = "error"
                raise
            finally:
                HANDLER_DURATION.labels(handler=name).observe(time.perf_counter() - start)
                HANDLER_TOTAL.labels(handler=name, outcome=outcome).inc()

        return wrapper

    return decorator


def start_metrics_server(port: int) -> None:
    """Start the Prometheus /metrics HTTP server on the configured port.

    Port 0 disables the server (used in tests and when the chart sets metrics off).
    """
    if port <= 0:
        logger.info("Operator Prometheus metrics disabled (port <= 0)")
        return
    start_http_server(port)
    logger.info("Operator Prometheus metrics listening on :%d/metrics", port)
