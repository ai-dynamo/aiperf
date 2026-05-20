# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Dynamo D608-D614 multi-node/rank chaos tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)

_WORKER_LABEL_KEYS = (
    "nvidia.com/dynamo-sub-component-type",
    "nvidia.com/dynamo-component-type",
    "app.kubernetes.io/component",
)
_TOPOLOGY_LABEL_NEEDLES = ("grove", "leaderworkerset", "lws")
_RANK_LABEL_KEYS = (
    "leaderworkerset.sigs.k8s.io/worker-index",
    "apps.kubernetes.io/pod-index",
    "statefulset.kubernetes.io/pod-name",
    "batch.kubernetes.io/job-completion-index",
    "jobset.sigs.k8s.io/job-index",
)
_PULL_REASONS = ("ImagePullBackOff", "ErrImagePull", "InvalidImageName")


@dataclass(slots=True)
class RankPod:
    """Ready Dynamo worker/rank pod selected from the live deployment."""

    name: str
    namespace: str
    node: str
    container: str
    image: str
    labels: dict[str, str]
    owner_kind: str
    owner_name: str
    rank_index: int | None

    @property
    def is_leader(self) -> bool:
        """Whether labels/name identify this pod as rank zero / leader."""
        if self.labels.get("leaderworkerset.sigs.k8s.io/leader") == "true":
            return True
        if self.rank_index == 0:
            return True
        return self.name.endswith("-0")


@dataclass(slots=True)
class MultiNodeRankTopology:
    """Detected rank-aware topology for D608-D614 self-skip gating."""

    pods: list[RankPod]
    reasons: list[str]

    def non_leader_pod(self) -> RankPod | None:
        """Return a non-leader rank when the topology exposes one."""
        for pod in self.pods:
            if not pod.is_leader:
                return pod
        return None


async def require_multinode_rank_topology(
    kubectl: KubectlClient,
    namespace: str,
    dynamo_config: Any,  # noqa: ANN401 - pytest fixture type is intentionally erased
    *,
    case_id: str,
) -> MultiNodeRankTopology:
    """Skip unless the live Dynamo deployment exposes multi-node/Grove/LWS/TP ranks."""
    topology = await discover_multinode_rank_topology(kubectl, namespace, dynamo_config)
    if topology is None:
        pytest.skip(
            f"{case_id}: requires multi-node, Grove/LWS, or tensor-parallel "
            f"Dynamo topology in namespace {namespace!r}; current deployment does "
            "not expose rank-aware worker pods"
        )
    return topology


async def discover_multinode_rank_topology(
    kubectl: KubectlClient,
    namespace: str,
    dynamo_config: Any,  # noqa: ANN401 - pytest fixture type is intentionally erased
) -> MultiNodeRankTopology | None:
    """Return topology details, or ``None`` when D608-D614 should self-skip."""
    pods = await _list_ready_rank_pods(kubectl, namespace)
    if not pods:
        return None

    reasons: list[str] = []
    nodes = {pod.node for pod in pods if pod.node}
    if len(nodes) > 1:
        reasons.append(f"multi-node workers on {sorted(nodes)!r}")
    if any(_has_grove_or_lws_label(pod.labels) for pod in pods):
        reasons.append("Grove/LWS labels present")
    tp_size = getattr(dynamo_config, "tensor_parallel_size", None)
    if isinstance(tp_size, int) and tp_size > 1:
        reasons.append(f"tensor_parallel_size={tp_size}")
    if len(pods) > 1 and any(pod.rank_index is not None for pod in pods):
        reasons.append("rank-indexed worker pods present")

    if not reasons:
        return None
    return MultiNodeRankTopology(pods=pods, reasons=reasons)


async def send_chat_completion(endpoint_url: str, *, case_id: str) -> None:
    """Issue a small non-streaming chat completion and assert HTTP 200."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": f"{case_id}: say ready"}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
        assert resp.status == 200, (
            f"{case_id}: expected HTTP 200 from Dynamo frontend, got "
            f"{resp.status}; body={body[:512]!r}"
        )


async def wait_until_chat_serves(
    endpoint_url: str,
    *,
    case_id: str,
    timeout: float = 120.0,
) -> None:
    """Poll chat completions until the frontend serves again after a rank fault."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_error = "<not attempted>"
    while True:
        try:
            await send_chat_completion(endpoint_url, case_id=case_id)
            return
        except (aiohttp.ClientError, AssertionError, asyncio.TimeoutError) as exc:
            last_error = repr(exc)
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"{case_id}: frontend did not serve within {timeout}s after fault; "
                f"last_error={last_error}"
            )
        await asyncio.sleep(3.0)


async def find_non_leader_rank_pid(
    kubectl: KubectlClient,
    pod: RankPod,
    *,
    case_id: str,
) -> int | None:
    """Best-effort discovery of a non-PID-1 rank child inside a worker container."""
    script = (
        "ps -eo pid=,ppid=,comm=,args= | "
        "grep -E 'dynamo|vllm|trtllm|sglang|worker|rank' | "
        "grep -v grep | awk '$1 != 1 {print $1; exit}'"
    )
    result = await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "sh",
        "-c",
        script,
        check=False,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        logger.info(
            lambda pod=pod.name, err=result.stderr: (
                f"{case_id}: no non-leader rank child PID found in {pod}: {err!r}"
            )
        )
        return None
    try:
        return int(text.splitlines()[0].strip())
    except ValueError:
        logger.info(lambda text=text: f"{case_id}: rank PID parse failed: {text!r}")
        return None


async def wait_for_replacement_ready(
    kubectl: KubectlClient,
    old_pod: RankPod,
    *,
    case_id: str,
    timeout: float = 180.0,
) -> RankPod:
    """Wait for a ready worker pod different from ``old_pod`` on the same rank."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        pods = await _list_ready_rank_pods(kubectl, old_pod.namespace)
        for pod in pods:
            if pod.name != old_pod.name and _same_rank_or_owner(pod, old_pod):
                return pod
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"{case_id}: no ready replacement for {old_pod.namespace}/"
                f"{old_pod.name} appeared within {timeout}s"
            )
        await asyncio.sleep(3.0)


@asynccontextmanager
async def temporary_network_policy(
    kubectl: KubectlClient,
    pod: RankPod,
    *,
    name: str,
    policy_spec: dict[str, Any],
) -> AsyncIterator[None]:
    """Apply a NetworkPolicy selected to one rank pod and delete it on exit."""
    manifest = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": name, "namespace": pod.namespace},
        "spec": {
            "podSelector": {"matchLabels": _stable_match_labels(pod)},
            **policy_spec,
        },
    }
    await kubectl.apply(orjson.dumps(manifest).decode())
    try:
        yield
    finally:
        await kubectl.run(
            "delete",
            "networkpolicy",
            name,
            "-n",
            pod.namespace,
            "--ignore-not-found",
            check=False,
        )


async def fill_dev_shm(kubectl: KubectlClient, pod: RankPod, *, case_id: str) -> bool:
    """Fill most of /dev/shm in a rank container; return False when unsupported."""
    script = (
        "set -eu; avail=$(df -Pm /dev/shm | awk 'NR==2 {print $4}'); "
        "count=$((avail > 32 ? avail - 8 : avail)); "
        'test "$count" -gt 0; '
        "dd if=/dev/zero of=/dev/shm/d611-fill bs=1M count=$count status=none"
    )
    result = await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "sh",
        "-c",
        script,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        logger.info(
            lambda err=result.stderr: f"{case_id}: /dev/shm fill skipped: {err!r}"
        )
        return False
    return True


async def cleanup_dev_shm(kubectl: KubectlClient, pod: RankPod) -> None:
    """Remove the D611 /dev/shm fill file if the pod is still reachable."""
    await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "rm",
        "-f",
        "/dev/shm/d611-fill",
        check=False,
    )


async def maybe_skew_clock(
    kubectl: KubectlClient,
    pod: RankPod,
    *,
    seconds: int,
) -> bool:
    """Try to skew a container clock; return False when CAP_SYS_TIME is absent."""
    probe = await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "sh",
        "-c",
        "date -u +%s >/tmp/d612-clock-original && date -u -s @$(cat /tmp/d612-clock-original)",
        check=False,
    )
    if probe.returncode != 0:
        return False
    skew = await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "sh",
        "-c",
        f"date -u -s @$(( $(cat /tmp/d612-clock-original) + {seconds} ))",
        check=False,
    )
    return skew.returncode == 0


async def restore_clock(kubectl: KubectlClient, pod: RankPod) -> None:
    """Restore a clock skew made by ``maybe_skew_clock`` when possible."""
    await kubectl.run(
        "exec",
        pod.name,
        "-c",
        pod.container,
        "-n",
        pod.namespace,
        "--",
        "sh",
        "-c",
        "test -f /tmp/d612-clock-original && date -u -s @$(cat /tmp/d612-clock-original)",
        check=False,
    )


async def find_placement_object(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str] | None:
    """Find a Grove/LWS placement object that can be deleted for D613."""
    for resource in ("podgroup", "podgroups.scheduling.x-k8s.io", "leaderworkerset"):
        result = await kubectl.run(
            "get",
            resource,
            "-n",
            namespace,
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        name = result.stdout.strip()
        if result.returncode == 0 and name:
            return resource, name
    return None


async def wait_for_object_recreated(
    kubectl: KubectlClient,
    namespace: str,
    resource: str,
    name: str,
    *,
    case_id: str,
    timeout: float = 120.0,
) -> None:
    """Wait for a placement object to reappear after deletion."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await kubectl.run("get", resource, name, "-n", namespace, check=False)
        if result.returncode == 0:
            return
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"{case_id}: placement object {resource}/{name} was not recreated "
                f"within {timeout}s after deletion"
            )
        await asyncio.sleep(3.0)


async def set_owner_image(
    kubectl: KubectlClient,
    pod: RankPod,
    image: str,
    *,
    case_id: str,
) -> bool:
    """Patch the rank pod's immediate workload owner image; return False if unsupported."""
    if pod.owner_kind.lower() not in {"deployment", "statefulset", "daemonset", "job"}:
        logger.info(
            lambda kind=pod.owner_kind: f"{case_id}: unsupported owner kind {kind!r}"
        )
        return False
    result = await kubectl.run(
        "set",
        "image",
        f"{pod.owner_kind.lower()}/{pod.owner_name}",
        f"{pod.container}={image}",
        "-n",
        pod.namespace,
        check=False,
    )
    return result.returncode == 0


async def wait_for_image_pull_failure(
    kubectl: KubectlClient,
    namespace: str,
    *,
    case_id: str,
    timeout: float = 90.0,
) -> str | None:
    """Wait until any pod in ``namespace`` reports an image-pull waiting reason."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await kubectl.run(
            "get", "pods", "-n", namespace, "-o", "json", check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            data = orjson.loads(result.stdout)
            for item in data.get("items", []):
                name = item.get("metadata", {}).get("name", "")
                statuses = item.get("status", {}).get("containerStatuses", []) or []
                for status in statuses:
                    waiting = (status.get("state") or {}).get("waiting") or {}
                    if waiting.get("reason") in _PULL_REASONS:
                        return name
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"{case_id}: no pod reported {_PULL_REASONS!r} within {timeout}s"
            )
        await asyncio.sleep(2.0)


async def _list_ready_rank_pods(
    kubectl: KubectlClient, namespace: str
) -> list[RankPod]:
    result = await kubectl.run("get", "pods", "-n", namespace, "-o", "json", check=True)
    data = orjson.loads(result.stdout)
    pods: list[RankPod] = []
    for item in data.get("items", []):
        labels = _labels(item)
        if not _is_worker_pod(item, labels) or not _pod_is_ready(item):
            continue
        container, image = _main_container(item)
        if not container:
            continue
        owner_kind, owner_name = _owner(item)
        pods.append(
            RankPod(
                name=_metadata_name(item),
                namespace=namespace,
                node=_node_name(item),
                container=container,
                image=image,
                labels=labels,
                owner_kind=owner_kind,
                owner_name=owner_name,
                rank_index=_rank_index(item, labels),
            )
        )
    return pods


def _is_worker_pod(item: dict[str, Any], labels: dict[str, str]) -> bool:
    values = " ".join(labels.get(key, "") for key in _WORKER_LABEL_KEYS).lower()
    name = _metadata_name(item).lower()
    return any(
        needle in values or needle in name for needle in ("worker", "decode", "prefill")
    )


def _pod_is_ready(item: dict[str, Any]) -> bool:
    status = item.get("status", {})
    if not isinstance(status, dict) or status.get("phase") != "Running":
        return False
    conditions = status.get("conditions", [])
    return any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in conditions
    )


def _metadata_name(item: dict[str, Any]) -> str:
    name = item.get("metadata", {}).get("name", "")
    return name if isinstance(name, str) else ""


def _node_name(item: dict[str, Any]) -> str:
    node = item.get("spec", {}).get("nodeName", "")
    return node if isinstance(node, str) else ""


def _labels(item: dict[str, Any]) -> dict[str, str]:
    raw = item.get("metadata", {}).get("labels", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _main_container(item: dict[str, Any]) -> tuple[str, str]:
    containers = item.get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return "", ""
    for container in containers:
        if not isinstance(container, dict):
            continue
        name = str(container.get("name", ""))
        if name not in {"istio-proxy", "linkerd-proxy"}:
            return name, str(container.get("image", ""))
    return "", ""


def _owner(item: dict[str, Any]) -> tuple[str, str]:
    owners = item.get("metadata", {}).get("ownerReferences", [])
    if not isinstance(owners, list) or not owners:
        return "", ""
    owner = owners[0]
    if not isinstance(owner, dict):
        return "", ""
    return str(owner.get("kind", "")), str(owner.get("name", ""))


def _rank_index(item: dict[str, Any], labels: dict[str, str]) -> int | None:
    for key in _RANK_LABEL_KEYS:
        value = labels.get(key, "")
        if value.isdigit():
            return int(value)
    tail = _metadata_name(item).rsplit("-", maxsplit=1)[-1]
    return int(tail) if tail.isdigit() else None


def _has_grove_or_lws_label(labels: dict[str, str]) -> bool:
    text = " ".join((*labels.keys(), *labels.values())).lower()
    return any(needle in text for needle in _TOPOLOGY_LABEL_NEEDLES)


def _same_rank_or_owner(left: RankPod, right: RankPod) -> bool:
    if left.rank_index is not None and left.rank_index == right.rank_index:
        return True
    return bool(left.owner_name and left.owner_name == right.owner_name)


def _stable_match_labels(pod: RankPod) -> dict[str, str]:
    value = pod.labels.get("statefulset.kubernetes.io/pod-name")
    if value:
        return {"statefulset.kubernetes.io/pod-name": value}
    pytest.skip(
        f"cannot safely select only rank pod {pod.namespace}/{pod.name}: "
        "no pod-unique label was found"
    )
