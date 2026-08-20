# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UI dispatch and progress streaming helpers for kube commands."""

from __future__ import annotations

from typing import Any, TypedDict

from aiperf.common.enums import MessageType


class WSProgressMessage(TypedDict, total=False):
    """Shape of a progress-streaming WebSocket payload.

    All fields are optional (``total=False``) because the server emits
    different subsets per ``message_type``. The values below are every key
    this module inspects via ``data.get(...)`` or ``data[...]``; other keys
    may be present and are ignored.
    """

    # One of ``MessageType.*`` values (serialized as the enum's str value) or
    # the sentinel string ``"subscribed"`` sent on WS handshake.
    message_type: str
    # Populated on ``CREDIT_PHASE_START`` / ``_PROGRESS`` / ``_COMPLETE``.
    # Keys observed here: ``phase``, ``requests_completed``,
    # ``total_expected_requests``.
    stats: dict[str, Any]
    # Populated on ``WORKER_STATUS_SUMMARY``. Maps worker id -> worker dict
    # containing at least ``status``.
    workers: dict[str, Any]
    # Populated on ``REALTIME_METRICS``. List of metric dicts with keys
    # ``tag``, ``value``/``avg``/``current``, ``display_unit``/``unit``.
    metrics: list[dict[str, Any]]


# WebSocket subscription message types for progress streaming
WS_MESSAGE_TYPES = [
    MessageType.CREDIT_PHASE_START,
    MessageType.CREDIT_PHASE_PROGRESS,
    MessageType.CREDIT_PHASE_COMPLETE,
    MessageType.REALTIME_METRICS,
    MessageType.WORKER_STATUS_SUMMARY,
    MessageType.ALL_RECORDS_RECEIVED,
]

# WebSocket reconnection settings
WS_MAX_RETRIES = 10

# API path segments used by CLI commands
API_WS_PATH = "/ws"


async def stream_progress(ws_url: str) -> None:
    """Stream progress messages from the benchmark via WebSocket.

    Args:
        ws_url: WebSocket URL for progress streaming.
    """
    from aiperf.kubernetes.console import logger, print_step
    from aiperf.kubernetes.port_forward import stream_progress_from_api

    print_step("Streaming progress...")
    logger.info("")

    async def handle_message(data: WSProgressMessage) -> bool:
        print_progress_message(data)
        return data.get("message_type") == MessageType.ALL_RECORDS_RECEIVED

    await stream_progress_from_api(
        ws_url,
        on_message=handle_message,
        message_types=WS_MESSAGE_TYPES,
        max_retries=WS_MAX_RETRIES,
    )


def _handle_credit_phase_start(logger: Any, data: WSProgressMessage) -> None:
    stats = data.get("stats", {})
    phase = stats.get("phase", "unknown")
    logger.info(f"[bold cyan]\\[PHASE][/bold cyan] Starting {phase} phase")


def _handle_credit_phase_progress(logger: Any, data: WSProgressMessage) -> None:
    stats = data.get("stats", {})
    phase = stats.get("phase", "")
    completed = stats.get("requests_completed", 0)
    total = stats.get("total_expected_requests", 0)
    percent = (completed / total * 100) if total > 0 else 0
    logger.info(
        f"[bold cyan]\\[PROGRESS][/bold cyan] {phase} "
        f"{completed}/{total} requests ({percent:.1f}%)"
    )


def _handle_credit_phase_complete(logger: Any, data: WSProgressMessage) -> None:
    stats = data.get("stats", {})
    phase = stats.get("phase", "unknown")
    logger.info(f"[bold cyan]\\[PHASE][/bold cyan] Completed {phase} phase")


def _handle_worker_status_summary(logger: Any, data: WSProgressMessage) -> None:
    workers = data.get("workers", {})
    total = len(workers)
    healthy = sum(
        1
        for w in workers.values()
        if isinstance(w, dict) and w.get("status", "").upper() == "HEALTHY"
    )
    logger.info(f"[bold cyan]\\[WORKERS][/bold cyan] {healthy}/{total} healthy")


def _handle_all_records_received(logger: Any, data: WSProgressMessage) -> None:
    logger.info(
        "[bold green]\\[COMPLETE][/bold green] All records received, benchmark finishing..."
    )


def _handle_realtime_metrics(logger: Any, data: WSProgressMessage) -> None:
    print_realtime_metrics(data)


_PROGRESS_HANDLERS = {
    MessageType.CREDIT_PHASE_START: _handle_credit_phase_start,
    MessageType.CREDIT_PHASE_PROGRESS: _handle_credit_phase_progress,
    MessageType.CREDIT_PHASE_COMPLETE: _handle_credit_phase_complete,
    MessageType.REALTIME_METRICS: _handle_realtime_metrics,
    MessageType.WORKER_STATUS_SUMMARY: _handle_worker_status_summary,
    MessageType.ALL_RECORDS_RECEIVED: _handle_all_records_received,
}


def print_progress_message(data: WSProgressMessage) -> None:
    """Log a progress message."""
    from aiperf.kubernetes.console import logger

    msg_type = data.get("message_type", "")
    if msg_type == "subscribed":
        return
    handler = _PROGRESS_HANDLERS.get(msg_type)
    if handler is not None:
        handler(logger, data)


def print_realtime_metrics(data: WSProgressMessage) -> None:
    """Log key metrics from realtime metrics message."""
    from aiperf.kubernetes.console import logger

    metrics = data.get("metrics", [])
    key_keywords = ["throughput", "latency", "ttft", "token"]
    found = []
    for m in metrics:
        tag = m.get("tag", "")
        tag_lower = tag.lower()
        if not any(kw in tag_lower for kw in key_keywords):
            continue
        value = m.get("value", m.get("avg", m.get("current", 0))) or 0
        unit = m.get("display_unit", m.get("unit", ""))
        found.append((tag, value, unit))
    for tag, value, unit in found:
        logger.info(f"[dim]\\[METRIC][/dim] {tag}: {value:.2f} {unit}".rstrip())
