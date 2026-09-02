# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast detection of unrecoverable pod conditions for GPU E2E tests.

Readiness loops in the GPU suites poll for many minutes before giving up. A pod
that can never start -- wrong node pool, missing Secret, unpullable image --
would otherwise burn the entire budget and then surface as a misleading
``TimeoutError``. These detectors turn those cases into an immediate, actionable
``RuntimeError`` instead.
"""

from __future__ import annotations

import re

from tests.kubernetes.helpers.kubectl import KubectlClient, PodStatus

__all__ = [
    "check_fatal_pod_conditions",
    "detect_fatal_image_conditions",
    "detect_fatal_pod_conditions",
    "detect_fatal_scheduling_conditions",
]

_SECRET_NOT_FOUND_RE = re.compile(r'secret\s+"?([\w.\-]+)"?\s+not found')
_CONFIGMAP_NOT_FOUND_RE = re.compile(r'configmap\s+"?([\w.\-]+)"?\s+not found')
_NO_NODES_AVAILABLE_RE = re.compile(r"0/(\d+) nodes are available")

_FATAL_WAITING_REASONS = frozenset(
    {
        "CreateContainerConfigError",
        "CrashLoopBackOff",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
    }
)


def _waiting_states(pods: list[PodStatus]) -> list[tuple[str, str, str, str]]:
    """Collect ``(pod, container, reason, message)`` for every waiting container."""
    states: list[tuple[str, str, str, str]] = []
    for pod in pods:
        for container_name, container in pod.containers.items():
            waiting = (container.get("state") or {}).get("waiting") or {}
            reason = waiting.get("reason") or ""
            if reason:
                states.append(
                    (pod.name, container_name, reason, waiting.get("message") or "")
                )
    return states


def _describe_config_error(
    pod: str, container: str, message: str, namespace: str
) -> str:
    """Explain a ``CreateContainerConfigError`` in terms of the missing object."""
    secret = _SECRET_NOT_FOUND_RE.search(message)
    if secret:
        name = secret.group(1)
        return (
            f"Pod {pod!r} (container {container!r}) is stuck in "
            f"CreateContainerConfigError because Secret {name!r} does not exist in "
            f"namespace {namespace!r}. The container references it via env/volume, "
            f"so the kubelet retries forever and the pod never starts. Create it, "
            f"e.g. `kubectl -n {namespace} create secret generic {name} "
            f"--from-literal=HF_TOKEN=$HF_TOKEN`, or point the suite at an existing "
            f"secret (GPU_TEST_HF_TOKEN_SECRET)."
        )

    configmap = _CONFIGMAP_NOT_FOUND_RE.search(message)
    if configmap:
        name = configmap.group(1)
        return (
            f"Pod {pod!r} (container {container!r}) is stuck in "
            f"CreateContainerConfigError because ConfigMap {name!r} does not exist "
            f"in namespace {namespace!r}. Create the ConfigMap before deploying, or "
            f"remove the reference from the manifest."
        )

    return (
        f"Pod {pod!r} (container {container!r}) is stuck in "
        f"CreateContainerConfigError in namespace {namespace!r}: {message.strip()!r}. "
        f"A referenced Secret/ConfigMap key is missing or misspelled; the kubelet "
        f"retries this forever. Fix the reference in the pod spec or create the "
        f"missing object."
    )


def _describe_image_error(
    pod: str, container: str, reason: str, message: str, namespace: str
) -> str:
    """Explain an image pull failure and name the likely misconfiguration."""
    if reason == "InvalidImageName":
        return (
            f"Pod {pod!r} (container {container!r}) has an invalid image reference "
            f"in namespace {namespace!r}: {message.strip()!r}. The image string is "
            f"malformed (bad tag/digest or stray whitespace) and will never pull. "
            f"Fix the image override for this suite."
        )
    return (
        f"Pod {pod!r} (container {container!r}) cannot pull its image in namespace "
        f"{namespace!r} ({reason}): {message.strip()!r}. Either the image does not "
        f"exist at that reference or the namespace has no working imagePullSecret. "
        f"Verify the tag exists and that the pull secret (GPU_TEST_IMAGE_PULL_SECRET) "
        f"is present in {namespace!r} and referenced by the pod spec."
    )


def _describe_crash_loop(pod: str, container: str, message: str, namespace: str) -> str:
    """Explain a container that repeatedly exits before becoming ready."""
    return (
        f"Pod {pod!r} (container {container!r}) is in CrashLoopBackOff in "
        f"namespace {namespace!r}: {message.strip()!r}. The container repeatedly "
        "exits before it can serve traffic; inspect its previous logs and fix the "
        "startup error before re-running."
    )


def detect_fatal_image_conditions(pods: list[PodStatus], namespace: str) -> str | None:
    """Return a message for image/secret failures that never self-heal.

    Args:
        pods: Pod statuses for the deployment being waited on.
        namespace: Namespace the pods live in, used in the remediation text.

    Returns:
        A diagnostic string, or None when no such condition is present.
    """
    for pod, container, reason, message in _waiting_states(pods):
        if reason not in _FATAL_WAITING_REASONS:
            continue
        if reason == "CreateContainerConfigError":
            return _describe_config_error(pod, container, message, namespace)
        if reason == "CrashLoopBackOff":
            return _describe_crash_loop(pod, container, message, namespace)
        return _describe_image_error(pod, container, reason, message, namespace)
    return None


def detect_fatal_scheduling_conditions(
    events: str, pods: list[PodStatus], namespace: str
) -> str | None:
    """Return a message for placement failures that can never resolve themselves.

    Args:
        events: Recent namespace events as rendered by ``kubectl get events``.
        pods: Pod statuses for the deployment being waited on.
        namespace: Namespace the pods live in, used in the remediation text.

    Returns:
        A diagnostic string, or None when no such condition is present.
    """
    names = sorted(p.name for p in pods) or ["<none>"]

    if 'no runtime for "nvidia" is configured' in events:
        return (
            f"Pods {names} in namespace {namespace!r} were scheduled onto node(s) "
            f"that do not provide the 'nvidia' RuntimeClass, so pod sandbox creation "
            f"fails permanently (FailedCreatePodSandBox). This happens when the pod "
            f"sets runtimeClassName: nvidia but requests no nvidia.com/gpu resource, "
            f"so nothing pins it to a GPU node. Either request nvidia.com/gpu, or pin "
            f'the pods with a node selector such as \'{{"nvidia.com/gpu.present": '
            f'"true"}}\'.'
        )

    if "Insufficient nvidia.com/gpu" in events:
        return (
            f"Pods {names} in namespace {namespace!r} are unschedulable: no node has "
            f"free nvidia.com/gpu capacity (Insufficient nvidia.com/gpu). The GPU "
            f"pool is saturated by other workloads or the request exceeds any single "
            f"node. Free GPUs, lower GPU_TEST_GPU_COUNT, or target a pool with "
            f"capacity."
        )

    if "didn't match Pod's node affinity/selector" in events:
        return (
            f"Pods {names} in namespace {namespace!r} are unschedulable: no node "
            f"matches the pod's nodeSelector/affinity. The selector labels or "
            f"tolerations do not exist on this cluster's nodes. Check the node "
            f"selector and GPU_TEST_TOLERATIONS against `kubectl get nodes "
            f"--show-labels`."
        )

    no_nodes = _NO_NODES_AVAILABLE_RE.search(events)
    if no_nodes:
        reason_line = next(
            (
                line.strip()
                for line in events.splitlines()
                if "0/" in line and "nodes are available" in line
            ),
            "",
        )
        return (
            f"Pods {names} in namespace {namespace!r} are unschedulable across all "
            f"{no_nodes.group(1)} nodes. Scheduler said: {reason_line!r}. Resolve the "
            f"stated predicate (taints/tolerations, resources, selectors) before "
            f"re-running."
        )

    return None


def detect_fatal_pod_conditions(
    events: str, pods: list[PodStatus], namespace: str
) -> str | None:
    """Return a message for any unrecoverable pod condition, or None.

    Checks image/secret failures first: they are unambiguous and reported on the
    container status itself, whereas scheduling text has to be inferred from
    free-form event lines.

    Args:
        events: Recent namespace events as rendered by ``kubectl get events``.
        pods: Pod statuses for the deployment being waited on.
        namespace: Namespace the pods live in, used in the remediation text.

    Returns:
        A diagnostic string naming the symptom, the cause and the fix, or None.
    """
    return detect_fatal_image_conditions(
        pods, namespace
    ) or detect_fatal_scheduling_conditions(events, pods, namespace)


async def check_fatal_pod_conditions(
    kubectl: KubectlClient,
    namespace: str,
    pods: list[PodStatus],
    fatal_checks: int,
    settle_polls: int = 3,
) -> tuple[str | None, int]:
    """Readiness-loop step: report a fatal condition and carry the poll counter.

    Image and Secret failures are reported on the first poll -- they are stated
    on the container status and never self-heal. Scheduling failures wait
    ``settle_polls`` consecutive not-ready polls, because a transient
    ``FailedCreatePodSandBox`` is normal while a sandbox is still coming up.

    Args:
        kubectl: Client used to fetch events only when the settle window elapses.
        namespace: Namespace the pods live in.
        pods: Pod statuses for the deployment being waited on.
        fatal_checks: Consecutive not-ready poll count from the previous call.
        settle_polls: Not-ready polls required before judging scheduling fatal.

    Returns:
        ``(message, fatal_checks)`` where message is None unless the condition is
        unrecoverable, and fatal_checks should be fed back into the next call.
    """
    if not pods or any(p.is_ready for p in pods):
        return None, 0

    fatal = detect_fatal_image_conditions(pods, namespace)
    if fatal:
        return fatal, fatal_checks

    fatal_checks += 1
    if fatal_checks < settle_polls:
        return None, fatal_checks

    events = await kubectl.get_events(namespace, limit=40)
    return detect_fatal_scheduling_conditions(events, pods, namespace), 0
