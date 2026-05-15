# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event classification helpers for the benchmark watchdog.

Each ``_handle_*`` function converts a single K8s ``EventInfo`` into either
a ``WatchdogProblem`` (recorded via ``recorder``) or a log entry. Kept
separate from the monitoring loop so ``BenchmarkWatchdog`` stays focused
on control flow.
"""

from __future__ import annotations

from typing import Protocol

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.watchdog_models import (
    ContainerInfo,
    EventInfo,
    ProblemSeverity,
    WatchdogPodSnapshot,
)

_FATAL_WAITING_REASONS = frozenset(
    {
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "ErrImageNeverPull",
        "CreateContainerConfigError",
        "InvalidImageName",
    }
)

_IMAGE_PULL_REASONS = frozenset(
    {"ImagePullBackOff", "ErrImagePull", "ErrImageNeverPull"}
)


class _ProblemRecorder(Protocol):
    def __call__(
        self,
        severity: ProblemSeverity,
        category: str,
        message: str,
        *,
        pod_name: str | None = None,
        suggestion: str | None = None,
    ) -> None: ...


def handle_failed_scheduling_event(
    event: EventInfo, recorder: _ProblemRecorder
) -> None:
    """Record a FailedScheduling event with severity based on cause."""
    severity = (
        ProblemSeverity.CRITICAL
        if "Insufficient" in event.message
        else ProblemSeverity.WARNING
    )
    recorder(
        severity,
        "scheduling-failure",
        f"{event.involved_object}: FailedScheduling - {event.message[:120]}",
        pod_name=event.involved_object,
        suggestion=(
            "kubectl get ns | grep aiperf- | wc -l  (clean up stale namespaces if > 5)"
        ),
    )


def handle_volume_issue_event(event: EventInfo, recorder: _ProblemRecorder) -> None:
    """Record a FailedMount / FailedAttachVolume event."""
    recorder(
        ProblemSeverity.WARNING,
        "volume-issue",
        f"{event.involved_object}: {event.reason} - {event.message[:120]}",
        pod_name=event.involved_object,
    )


def handle_backoff_event(event: EventInfo, recorder: _ProblemRecorder) -> None:
    """Record a container BackOff warning event."""
    recorder(
        ProblemSeverity.WARNING,
        "container-backoff",
        f"{event.involved_object}: BackOff - {event.message[:120]}",
        pod_name=event.involved_object,
    )


def handle_evicted_event(event: EventInfo, recorder: _ProblemRecorder) -> None:
    """Record a pod eviction event."""
    recorder(
        ProblemSeverity.CRITICAL,
        "pod-evicted",
        f"{event.involved_object}: Evicted - {event.message[:120]}",
        pod_name=event.involved_object,
        suggestion=(
            "Node under memory/disk pressure. Reduce worker count or pod memory limits."
        ),
    )


def handle_killing_event(event: EventInfo, log: AIPerfLogger) -> None:
    """Log a container-being-killed event at INFO level."""
    log.info(
        f"[WATCHDOG] Event: {event.involved_object} being killed "
        f"- {event.message[:100]}"
    )


def handle_unhealthy_event(event: EventInfo, log: AIPerfLogger) -> None:
    """Log a debug message for a Warning-type Unhealthy probe event."""
    short = (
        event.involved_object.split("-")[-1]
        if "-" in event.involved_object
        else event.involved_object
    )
    log.debug(
        lambda short=short, msg=event.message: (
            f"[WATCHDOG] Probe failure on {short}: {msg[:80]}"
        )
    )


def classify_event(
    event: EventInfo, recorder: _ProblemRecorder, log: AIPerfLogger
) -> bool:
    """Dispatch ``event`` to the appropriate handler.

    Returns True if the event matched a handler (and thus should be
    fingerprinted by the caller), False if it was ignored.
    """
    if event.reason == "FailedScheduling":
        handle_failed_scheduling_event(event, recorder)
        return True
    if event.reason in ("FailedMount", "FailedAttachVolume"):
        handle_volume_issue_event(event, recorder)
        return True
    if event.reason == "BackOff" and event.type == "Warning":
        handle_backoff_event(event, recorder)
        return True
    if event.reason == "Evicted":
        handle_evicted_event(event, recorder)
        return True
    if event.reason == "Killing":
        handle_killing_event(event, log)
        return True
    if event.reason == "Unhealthy" and event.type == "Warning":
        handle_unhealthy_event(event, log)
        return True
    return False


def classify_container_states(
    pod: WatchdogPodSnapshot,
    *,
    namespace: str,
    seen_fingerprints: set[str],
    recorder: _ProblemRecorder,
    log: AIPerfLogger,
) -> None:
    """Detect problematic container states on ``pod``.

    Adds entries to ``seen_fingerprints`` for each (pod, container, reason)
    so repeated observations aren't re-reported.
    """
    for c in pod.container_statuses:
        if c.state == "waiting" and c.reason in _FATAL_WAITING_REASONS:
            _report_waiting_failure(pod, c, seen_fingerprints, recorder)
            continue
        if c.state != "terminated":
            continue
        if c.reason == "OOMKilled":
            _report_oom_killed(pod, c, seen_fingerprints, recorder)
        elif c.exit_code == 137:
            _report_sigkill(
                pod, c, namespace=namespace, seen=seen_fingerprints, recorder=recorder
            )
        elif c.exit_code is not None and c.exit_code != 0:
            _log_nonzero_exit(pod, c, seen_fingerprints, log)


def _report_waiting_failure(
    pod: WatchdogPodSnapshot,
    c: ContainerInfo,
    seen: set[str],
    recorder: _ProblemRecorder,
) -> None:
    fp = f"{pod.name}/{c.name}/{c.reason}"
    if fp in seen:
        return
    seen.add(fp)
    msg_detail = (c.message or "N/A")[:100]
    hint = ""
    if c.reason in _IMAGE_PULL_REASONS:
        hint = (
            " -- For locally built images, use: "
            "--image-pull-policy IfNotPresent "
            "(or imagePullPolicy: IfNotPresent in YAML)"
        )
    reason_slug = (c.reason or "unknown").lower()
    recorder(
        ProblemSeverity.CRITICAL,
        f"container-{reason_slug}",
        f"{c.name} in {pod.name}: {c.reason} - {msg_detail}{hint}",
        pod_name=pod.name,
    )


def _report_oom_killed(
    pod: WatchdogPodSnapshot,
    c: ContainerInfo,
    seen: set[str],
    recorder: _ProblemRecorder,
) -> None:
    fp = f"{pod.name}/{c.name}/OOMKilled"
    if fp in seen:
        return
    seen.add(fp)
    recorder(
        ProblemSeverity.CRITICAL,
        "oom-killed",
        f"{c.name} in {pod.name}: OOMKilled. Process exceeded memory limits.",
        pod_name=pod.name,
        suggestion="Increase memory limits in benchmark config.",
    )


def _report_sigkill(
    pod: WatchdogPodSnapshot,
    c: ContainerInfo,
    *,
    namespace: str,
    seen: set[str],
    recorder: _ProblemRecorder,
) -> None:
    fp = f"{pod.name}/{c.name}/sigkill-137"
    if fp in seen:
        return
    seen.add(fp)
    recorder(
        ProblemSeverity.WARNING,
        "sigkill",
        f"{c.name} in {pod.name}: killed by SIGKILL (exit 137). "
        f"May be pod eviction due to node memory pressure.",
        pod_name=pod.name,
        suggestion=(
            f"1) kubectl describe pod -n {namespace} {pod.name}"
            f" | grep -A5 'Status'\n"
            f"  2) kubectl get events -n {namespace}"
            f" --field-selector involvedObject.name={pod.name}"
            f" | grep Evict\n"
            f"  3) kubectl describe node | grep -A5 'Conditions'"
        ),
    )


def _log_nonzero_exit(
    pod: WatchdogPodSnapshot,
    c: ContainerInfo,
    seen: set[str],
    log: AIPerfLogger,
) -> None:
    fp = f"{pod.name}/{c.name}/exit-{c.exit_code}"
    if fp in seen:
        return
    seen.add(fp)
    log.warning(
        f"[WATCHDOG] {c.name} in {pod.name}: "
        f"exit code {c.exit_code} "
        f"(reason: {c.reason or 'Completed'})"
    )
