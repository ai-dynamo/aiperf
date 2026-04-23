# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-pod analysis helpers for the benchmark watchdog.

Each function here is a single check (phase transition, crash loop, etc.)
extracted out of ``BenchmarkWatchdog`` so the main class stays a thin
coordinator. They take the watchdog instance via the ``WatchdogLike``
protocol and mutate its recorded problems / timelines through its public
helper methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Protocol

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.watchdog_models import (
    PodTimeline,
    ProblemSeverity,
    WatchdogDataSource,
    WatchdogPodSnapshot,
)
from aiperf.kubernetes.watchdog_render import _fmt_duration

if TYPE_CHECKING:
    from collections.abc import Callable


class WatchdogLike(Protocol):
    """Narrow interface the pod-check helpers need from the watchdog."""

    namespace: str
    pending_threshold: float
    pending_critical_threshold: float
    crashloop_threshold: int
    _log: AIPerfLogger
    _source: WatchdogDataSource
    _pod_timelines: dict[str, PodTimeline]
    _completed_pods: set[str]
    _start_time: float
    _add_problem: Callable[..., None]


def track_phase_transition(
    wd: WatchdogLike, pod: WatchdogPodSnapshot, tl: PodTimeline
) -> None:
    """Record phase changes with timing."""
    if pod.phase == tl.last_phase:
        return

    old_phase = tl.last_phase
    now = time.time()
    tl.phase_history.append((now, pod.phase))
    elapsed = now - wd._start_time
    time_in_old = 0.0
    if len(tl.phase_history) >= 2:
        time_in_old = tl.phase_history[-1][0] - tl.phase_history[-2][0]

    short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

    wd._log.info(
        lambda: f"[WATCHDOG] {tl.role}({short}): "
        f"{old_phase} -> {pod.phase}  "
        f"(in {old_phase} for {_fmt_duration(time_in_old)}, "
        f"total +{_fmt_duration(elapsed)})"
    )

    tl.last_phase = pod.phase


def check_pending_too_long(
    wd: WatchdogLike, pod: WatchdogPodSnapshot, tl: PodTimeline
) -> None:
    """Escalating warnings for pods stuck in Pending."""
    if pod.phase != "Pending":
        return

    pending_duration = time.time() - tl.first_seen

    if (
        pending_duration > wd.pending_critical_threshold
        and not tl.pending_critical_warned
    ):
        tl.pending_critical_warned = True
        wd._add_problem(
            ProblemSeverity.CRITICAL,
            "pod-pending-critical",
            f"Pod {pod.name} stuck Pending for "
            f"{_fmt_duration(pending_duration)}! "
            f"Likely resource exhaustion or scheduling constraint.",
            pod_name=pod.name,
            suggestion=(
                f"1) kubectl describe pod -n {wd.namespace} {pod.name} "
                f"| tail -20\n"
                f"  2) kubectl get ns | grep aiperf | wc -l  "
                f"(check stale namespaces)\n"
                f"  3) kubectl describe node | grep -A 10 'Allocated'"
            ),
        )
    elif pending_duration > wd.pending_threshold and not tl.pending_warned:
        tl.pending_warned = True
        wd._add_problem(
            ProblemSeverity.WARNING,
            "pod-pending",
            f"Pod {pod.name} Pending for "
            f"{_fmt_duration(pending_duration)} "
            f"(threshold: {_fmt_duration(wd.pending_threshold)}). "
            f"May be waiting for resources.",
            pod_name=pod.name,
            suggestion=(
                f"kubectl describe pod -n {wd.namespace} {pod.name} | tail -20"
            ),
        )


def check_crash_loop(
    wd: WatchdogLike, pod: WatchdogPodSnapshot, tl: PodTimeline
) -> None:
    """Detect restart count increases and crash loops."""
    if pod.restarts <= tl.last_restart_count:
        tl.last_restart_count = pod.restarts
        return

    old_count = tl.last_restart_count
    tl.last_restart_count = pod.restarts
    tl.restart_count = pod.restarts
    wd._log.info(
        lambda: f"[WATCHDOG] Restart detected: "
        f"{tl.role}({pod.name.split('-')[-1]}) "
        f"restarts {old_count} -> {pod.restarts}"
    )

    if pod.restarts >= wd.crashloop_threshold and not tl.crashloop_warned:
        tl.crashloop_warned = True
        wd._add_problem(
            ProblemSeverity.CRITICAL,
            "crash-loop",
            f"Pod {pod.name} restarted {pod.restarts}x - likely CrashLoopBackOff.",
            pod_name=pod.name,
            suggestion=(f"kubectl -n {wd.namespace} logs {pod.name} --previous"),
        )


def check_pod_completion(
    wd: WatchdogLike, pod: WatchdogPodSnapshot, pod_role_fn: Callable[[str], str]
) -> None:
    """Log when a pod reaches terminal state for the first time."""
    if pod.name in wd._completed_pods:
        return
    if pod.phase not in ("Succeeded", "Failed"):
        return

    wd._completed_pods.add(pod.name)
    elapsed = time.time() - wd._start_time
    tl = wd._pod_timelines.get(pod.name)
    pod_age = (time.time() - tl.first_seen) if tl else 0
    role = tl.role if tl else pod_role_fn(pod.name)
    short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

    if pod.phase == "Succeeded":
        wd._log.info(
            f"[WATCHDOG] {role}({short}) completed successfully "
            f"(age={_fmt_duration(pod_age)}, +{_fmt_duration(elapsed)})"
        )
        return

    wd._add_problem(
        ProblemSeverity.CRITICAL,
        "pod-failed",
        f"Pod {pod.name} FAILED after {_fmt_duration(pod_age)}.",
        pod_name=pod.name,
        suggestion=(f"kubectl -n {wd.namespace} logs {pod.name} --all-containers"),
    )
    with contextlib.suppress(RuntimeError):
        asyncio.create_task(fetch_failure_logs(wd, pod.name))


async def fetch_failure_logs(wd: WatchdogLike, pod_name: str) -> None:
    """Best-effort fetch of logs from a failed pod."""
    try:
        logs = await wd._source.get_pod_logs(pod_name, wd.namespace, tail=20)
        if logs.strip():
            wd._log.error(
                lambda logs=logs, pod_name=pod_name: (
                    f"[WATCHDOG] Last logs from {pod_name}:\n{logs}"
                )
            )
    except Exception:  # noqa: BLE001, S110 - watchdog must never die on a single-check failure
        # Best-effort: logs already swallowed to empty by get_pod_logs in
        # normal paths; this is a defensive net for unexpected failures.
        pass
