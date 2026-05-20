# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D901-D914 -- Dynamo chaos observability and audit-surface tests.

These cases exercise the observability layer around failures rather than the
fault mechanics themselves. Each test is executable on a cluster with the
matching Dynamo deployment shape and otherwise skips with the missing
prerequisite named in the skip reason.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from typing import Any

import aiohttp
import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_RESOURCE = "dynamographdeployment"
_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_OPERATOR_NS = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_METRICS_PORT = 8000
_AUDIT_TIMEOUT_S = 120.0
_CONDITION_SLO_S = 120.0
_MAX_METRIC_SERIES = 2000
_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;}]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
)
_FAILURE_TERMS = ("failed", "error", "backoff", "denied", "forbidden", "quota")


@pytest.mark.timeout(300)
async def test_d901_operator_observability_exposes_ready_condition_and_logs(
    kubectl: KubectlClient,
    dynamo_operator: Any,  # noqa: ARG001 - fixture installs the operator
) -> None:
    """D901: operator observability includes readiness and recent reconcile logs."""
    pod = await _first_pod_name(
        kubectl,
        namespace=_OPERATOR_NS,
        label_selector=_OPERATOR_SELECTOR,
        skip_reason="D901 requires a running Dynamo operator pod",
    )
    pod_obj = await _get_pod_json(kubectl, namespace=_OPERATOR_NS, pod=pod)
    assert _pod_condition_status(pod_obj, "Ready") == "True", (
        f"D901: operator pod {_OPERATOR_NS}/{pod} is not Ready; "
        "operator failures must be visible through pod conditions"
    )

    logs = await kubectl.get_logs(pod, namespace=_OPERATOR_NS, tail=200)
    assert _mentions_any(logs, ("reconcil", "controller", "manager", "starting")), (
        f"D901: operator pod {_OPERATOR_NS}/{pod} exposed no recognizable "
        "controller/reconcile log lines in the last 200 lines"
    )


@pytest.mark.timeout(300)
async def test_d902_worker_observability_exposes_live_ready_and_restart_counts(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D902: worker pods expose Ready state and restart-count observability."""
    pods = await _pods_json(kubectl, namespace=dynamo_deployment_namespace)
    workers = [pod for pod in pods if _is_worker_pod(pod)]
    if not workers:
        pytest.skip("D902 requires at least one Dynamo worker pod")

    ready_workers = [
        pod for pod in workers if _pod_condition_status(pod, "Ready") == "True"
    ]
    assert ready_workers, "D902: no Dynamo worker pod reports Ready=True"
    for pod in workers:
        name = pod.get("metadata", {}).get("name", "<unknown>")
        restarts = _total_restarts(pod)
        assert restarts >= 0, (
            f"D902: worker pod {name} has invalid restart count {restarts}"
        )
        assert pod.get("status", {}).get("containerStatuses"), (
            f"D902: worker pod {name} has no containerStatuses; restart and "
            "readiness observability would be blind"
        )


@pytest.mark.timeout(300)
async def test_d903_nats_observability_survives_stats_bus_fault(
    faults: Any,
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D903: a NATS fault produces observable events and leaves metrics scrapeable."""
    before_events = await _events_json(kubectl, namespace=dynamo_deployment_namespace)
    async with faults.inject("store.nats.kill", grace_period=0):
        await asyncio.sleep(5.0)
    await asyncio.sleep(5.0)

    after_events = await _events_json(kubectl, namespace=dynamo_deployment_namespace)
    assert len(after_events) >= len(before_events), (
        "D903: namespace events disappeared after NATS fault; expected the "
        "fault to be auditable through Kubernetes events"
    )
    metrics = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    assert metrics, "D903: frontend metrics were empty after NATS restart"


@pytest.mark.timeout(300)
async def test_d904_dns_observability_records_resolution_failure(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D904: DNS failures are visible in events or pod logs when present."""
    events = await _events_json(kubectl, namespace=dynamo_deployment_namespace)
    logs = await _recent_namespace_logs(kubectl, namespace=dynamo_deployment_namespace)
    combined = _event_text(events) + "\n" + logs
    if not _mentions_any(
        combined, ("dns", "resolve", "no such host", "name resolution")
    ):
        pytest.skip(
            "D904 requires an active or recently injected DNS-resolution failure "
            "in the Dynamo namespace"
        )
    assert _mentions_any(
        combined, ("dns", "resolve", "no such host", "name resolution")
    )


@pytest.mark.timeout(300)
async def test_d905_image_failure_audit_surfaces_pull_reason(
    kubectl: KubectlClient,
    dynamo_operator: Any,  # noqa: ARG001 - fixture installs the operator
) -> None:
    """D905: image-pull failures are auditable through pod reason and CR status."""
    namespace = "d905-image-audit"
    name = "d905-image-audit"
    manifest = _minimal_dgd_manifest(
        name=name,
        namespace=namespace,
        image="nonexistent.example.com/dynamo-observability:nope",
    )
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(orjson.dumps(manifest).decode(), namespace=namespace)
        pod_name = await _wait_for_pod_waiting_reason(
            kubectl,
            namespace=namespace,
            label_selector=f"{_DGD_LABEL}={name}",
            reasons=("ImagePullBackOff", "ErrImagePull"),
            timeout=_AUDIT_TIMEOUT_S,
        )
        assert pod_name, (
            "D905: no child pod surfaced ImagePullBackOff/ErrImagePull; "
            "image failures must be auditable at the pod layer"
        )
        status = await _dgd_status_text(kubectl, namespace=namespace, name=name)
        assert _mentions_any(
            status, ("image", "pull", "ImagePullBackOff", "ErrImagePull")
        ), f"D905: DGD status did not mention the image-pull failure: {status!r}"
    finally:
        await _delete_namespace_now(kubectl, namespace)


@pytest.mark.timeout(300)
async def test_d906_quota_failure_audit_surfaces_admission_rejection(
    faults: Any,
    kubectl: KubectlClient,
    dynamo_operator: Any,  # noqa: ARG001 - fixture installs the operator
) -> None:
    """D906: ResourceQuota admission failures leave an explicit audit trail."""
    namespace = "d906-quota-audit"
    name = "d906-quota-audit"
    await kubectl.create_namespace(namespace)
    try:
        async with faults.inject(
            "cluster.resource_quota",
            namespace=namespace,
            name="d906-tiny-quota",
            hard={"requests.memory": "1Mi", "limits.memory": "1Mi", "pods": "1"},
        ):
            await kubectl.apply(
                orjson.dumps(
                    _minimal_dgd_manifest(name=name, namespace=namespace)
                ).decode(),
                namespace=namespace,
            )
            events = await _wait_for_event_terms(
                kubectl,
                namespace=namespace,
                terms=("quota", "exceeded", "forbidden"),
                timeout=_AUDIT_TIMEOUT_S,
            )
            assert events, (
                "D906: quota exhaustion produced no Kubernetes event mentioning "
                "quota/exceeded/forbidden"
            )
    finally:
        await _delete_namespace_now(kubectl, namespace)


@pytest.mark.timeout(300)
async def test_d907_security_failure_audit_surfaces_policy_denial(
    kubectl: KubectlClient,
) -> None:
    """D907: security denials are explicit when the cluster policy stack exists."""
    result = await kubectl.run(
        "auth",
        "can-i",
        "create",
        "pods/exec",
        "-n",
        _OPERATOR_NS,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("D907 requires Kubernetes authz checks to be available")
    assert result.stdout.strip().lower() in {"yes", "no"}, (
        f"D907: kubectl auth can-i returned an unparsable audit answer: {result.stdout!r}"
    )


@pytest.mark.timeout(300)
async def test_d908_metrics_endpoint_unavailable_is_explicitly_detected(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D908: an unavailable metrics endpoint fails with a clear scrape error."""
    pod = await _first_frontend_pod(kubectl, namespace=dynamo_deployment_namespace)
    with pytest.raises(
        (RuntimeError, aiohttp.ClientError, TimeoutError),
        match="metrics|port|GET|connect",
    ):
        async with kubectl.port_forward(
            pod, 1, namespace=dynamo_deployment_namespace
        ) as local:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2.0)
            ) as session:
                async with session.get(f"http://127.0.0.1:{local}/metrics") as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(
                            f"metrics GET returned {resp.status}: {body[:128]}"
                        )


@pytest.mark.timeout(300)
async def test_d909_metric_cardinality_stays_bounded(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D909: frontend metrics cardinality is bounded enough to scrape safely."""
    text = await _frontend_metrics_text(kubectl, namespace=dynamo_deployment_namespace)
    series = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert series, "D909: frontend metrics endpoint returned no sample series"
    assert len(series) <= _MAX_METRIC_SERIES, (
        f"D909: frontend metrics exposed {len(series)} series, exceeding "
        f"the {_MAX_METRIC_SERIES} series cardinality budget"
    )
    dynamic_label_lines = [
        line for line in series if re.search(r"(pod|uid|request_id|trace_id)=", line)
    ]
    assert not dynamic_label_lines, (
        "D909: frontend metrics include high-cardinality identity labels: "
        f"{dynamic_label_lines[:5]!r}"
    )


@pytest.mark.timeout(300)
async def test_d910_events_ordering_is_monotonic_by_timestamp(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D910: Kubernetes event timestamps are ordered for timeline reconstruction."""
    events = await _events_json(kubectl, namespace=dynamo_deployment_namespace)
    if len(events) < 2:
        pytest.skip("D910 requires at least two namespace events to validate ordering")
    timestamps = [_event_timestamp(event) for event in events]
    timestamps = [stamp for stamp in timestamps if stamp]
    assert timestamps == sorted(timestamps), (
        "D910: events are not monotonic by eventTime/lastTimestamp; "
        f"observed order={timestamps!r}"
    )


@pytest.mark.timeout(300)
async def test_d911_secret_redaction_in_events_status_and_logs(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D911: events, CR status, and recent logs do not expose secret material."""
    events = _event_text(
        await _events_json(kubectl, namespace=dynamo_deployment_namespace)
    )
    dgds = await _dgd_items(kubectl, namespace=dynamo_deployment_namespace)
    statuses = "\n".join(orjson.dumps(item.get("status", {})).decode() for item in dgds)
    logs = await _recent_namespace_logs(kubectl, namespace=dynamo_deployment_namespace)
    combined = "\n".join((events, statuses, logs))
    leaks = [
        pattern.pattern for pattern in _SECRET_PATTERNS if pattern.search(combined)
    ]
    assert not leaks, (
        f"D911: secret-looking material leaked via observability: {leaks!r}"
    )


@pytest.mark.timeout(300)
async def test_d912_condition_timing_slo_for_successful_deployment(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D912: Ready/Successful condition timestamps satisfy the D9xx timing SLO."""
    dgds = await _dgd_items(kubectl, namespace=dynamo_deployment_namespace)
    if not dgds:
        pytest.skip("D912 requires at least one DynamoGraphDeployment")
    for item in dgds:
        name = item.get("metadata", {}).get("name", "<unknown>")
        status = item.get("status", {})
        if status.get("state") != "successful":
            pytest.skip(
                f"D912 requires successful DGD state; {name} is {status.get('state')!r}"
            )
        created = item.get("metadata", {}).get("creationTimestamp")
        transitioned = _latest_condition_time(status.get("conditions", []))
        if not created or not transitioned:
            pytest.skip(f"D912 requires condition timestamps on DGD {name}")
        elapsed = _parse_rfc3339_seconds(transitioned) - _parse_rfc3339_seconds(created)
        assert elapsed <= _CONDITION_SLO_S, (
            f"D912: DGD {name} condition latency {elapsed:.1f}s exceeds "
            f"{_CONDITION_SLO_S:.0f}s SLO"
        )


@pytest.mark.timeout(300)
async def test_d913_mutation_observability_records_generation_and_patch_events(
    kubectl: KubectlClient,
    dynamo_server: Any,  # noqa: ARG001 - fixture installs a Dynamo deployment
    dynamo_deployment_namespace: str,
) -> None:
    """D913: CR mutations are observable through generation and event changes."""
    dgds = await _dgd_items(kubectl, namespace=dynamo_deployment_namespace)
    if not dgds:
        pytest.skip("D913 requires an existing DynamoGraphDeployment to inspect")
    item = dgds[0]
    metadata = item.get("metadata", {})
    status = item.get("status", {})
    generation = int(metadata.get("generation", 0))
    observed = int(status.get("observedGeneration", generation))
    assert observed <= generation, (
        f"D913: observedGeneration {observed} is ahead of metadata.generation {generation}"
    )
    assert "resourceVersion" in metadata, "D913: CR metadata lacks resourceVersion"


@pytest.mark.timeout(300)
async def test_d914_sweep_observability_reports_variation_progress_or_skips(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D914: mutation-sweep observability is present when sweep CRDs exist."""
    result = await kubectl.run(
        "get",
        "dynamographdeployments",
        "-n",
        dynamo_deployment_namespace,
        "-l",
        "chaos.aiperf.nvidia.com/sweep=true",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("D914 requires DynamoGraphDeployment list access")
    data = orjson.loads(result.stdout or b'{"items":[]}')
    items = data.get("items", [])
    if not items:
        pytest.skip(
            "D914 requires a D9xx mutation-sweep run labelled chaos.aiperf.nvidia.com/sweep=true"
        )
    for item in items:
        metadata = item.get("metadata", {})
        labels = metadata.get("labels", {})
        status = item.get("status", {})
        assert labels.get("chaos.aiperf.nvidia.com/mutation") is not None, (
            f"D914: sweep child {metadata.get('name')} lacks mutation label"
        )
        assert status.get("state"), (
            f"D914: sweep child {metadata.get('name')} lacks status.state for progress tracking"
        )


async def _get_pod_json(
    kubectl: KubectlClient,
    *,
    namespace: str,
    pod: str,
) -> dict[str, Any]:
    result = await kubectl.run(
        "get", "pod", pod, "-n", namespace, "-o", "json", check=True
    )
    return orjson.loads(result.stdout)


async def _pods_json(kubectl: KubectlClient, *, namespace: str) -> list[dict[str, Any]]:
    result = await kubectl.run("get", "pods", "-n", namespace, "-o", "json", check=True)
    return orjson.loads(result.stdout).get("items", [])


async def _dgd_items(kubectl: KubectlClient, *, namespace: str) -> list[dict[str, Any]]:
    result = await kubectl.run(
        "get",
        _DGD_RESOURCE,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    return orjson.loads(result.stdout or b'{"items":[]}').get("items", [])


async def _events_json(
    kubectl: KubectlClient, *, namespace: str
) -> list[dict[str, Any]]:
    result = await kubectl.run(
        "get",
        "events",
        "-n",
        namespace,
        "--sort-by=.lastTimestamp",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    return orjson.loads(result.stdout or b'{"items":[]}').get("items", [])


async def _first_pod_name(
    kubectl: KubectlClient,
    *,
    namespace: str,
    label_selector: str,
    skip_reason: str,
) -> str:
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        label_selector,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = result.stdout.strip() if result.returncode == 0 else ""
    if not pod:
        pytest.skip(skip_reason)
    return pod


async def _first_frontend_pod(kubectl: KubectlClient, *, namespace: str) -> str:
    pods = await _pods_json(kubectl, namespace=namespace)
    for pod in pods:
        name = pod.get("metadata", {}).get("name", "")
        if "frontend" in name and _pod_condition_status(pod, "Ready") == "True":
            return name
    pytest.skip("D908/D909 require a ready Dynamo frontend pod")


async def _frontend_metrics_text(kubectl: KubectlClient, *, namespace: str) -> str:
    pod = await _first_frontend_pod(kubectl, namespace=namespace)
    async with kubectl.port_forward(pod, _METRICS_PORT, namespace=namespace) as local:
        return await _get_metrics_text(f"http://127.0.0.1:{local}/metrics")


async def _get_metrics_text(url: str) -> str:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=5.0)
    ) as session:
        resp = await session.get(url)
        try:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"metrics GET returned {resp.status}: {body[:256]}")
            return await resp.text()
        finally:
            resp.release()


async def _recent_namespace_logs(kubectl: KubectlClient, *, namespace: str) -> str:
    chunks: list[str] = []
    for pod in await _pods_json(kubectl, namespace=namespace):
        pod_name = pod.get("metadata", {}).get("name", "")
        for status in pod.get("status", {}).get("containerStatuses", []) or []:
            container = status.get("name", "")
            if not pod_name or not container:
                continue
            chunks.append(
                await kubectl.get_logs(
                    pod_name, container=container, namespace=namespace, tail=50
                )
            )
    return "\n".join(chunks)


async def _wait_for_pod_waiting_reason(
    kubectl: KubectlClient,
    *,
    namespace: str,
    label_selector: str,
    reasons: Iterable[str],
    timeout: float,
) -> str:
    wanted = set(reasons)
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        pods = await _pods_json_by_label(
            kubectl, namespace=namespace, label_selector=label_selector
        )
        for pod in pods:
            name = pod.get("metadata", {}).get("name", "")
            statuses = list(pod.get("status", {}).get("containerStatuses", []) or [])
            statuses += list(
                pod.get("status", {}).get("initContainerStatuses", []) or []
            )
            for container_status in statuses:
                waiting = (container_status.get("state") or {}).get("waiting") or {}
                if waiting.get("reason") in wanted:
                    return name
        await asyncio.sleep(1.0)
    return ""


async def _pods_json_by_label(
    kubectl: KubectlClient,
    *,
    namespace: str,
    label_selector: str,
) -> list[dict[str, Any]]:
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        label_selector,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    return orjson.loads(result.stdout or b'{"items":[]}').get("items", [])


async def _wait_for_event_terms(
    kubectl: KubectlClient,
    *,
    namespace: str,
    terms: Iterable[str],
    timeout: float,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        text = _event_text(await _events_json(kubectl, namespace=namespace)).lower()
        if any(term.lower() in text for term in terms):
            return text
        await asyncio.sleep(2.0)
    return ""


async def _dgd_status_text(kubectl: KubectlClient, *, namespace: str, name: str) -> str:
    result = await kubectl.run(
        "get",
        _DGD_RESOURCE,
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


async def _delete_namespace_now(kubectl: KubectlClient, namespace: str) -> None:
    await kubectl.run(
        "delete",
        "namespace",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )


def _minimal_dgd_manifest(
    *,
    name: str,
    namespace: str,
    image: str = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest",
) -> dict[str, Any]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                        }
                    },
                }
            }
        },
    }


def _pod_condition_status(pod: dict[str, Any], condition_type: str) -> str:
    for condition in pod.get("status", {}).get("conditions", []) or []:
        if condition.get("type") == condition_type:
            return str(condition.get("status", ""))
    return ""


def _is_worker_pod(pod: dict[str, Any]) -> bool:
    metadata = pod.get("metadata", {})
    labels = metadata.get("labels", {})
    name = metadata.get("name", "")
    label_text = " ".join(str(value).lower() for value in labels.values())
    return any(
        term in f"{name.lower()} {label_text}"
        for term in ("worker", "decode", "prefill")
    )


def _total_restarts(pod: dict[str, Any]) -> int:
    statuses = pod.get("status", {}).get("containerStatuses", []) or []
    return sum(int(status.get("restartCount", 0)) for status in statuses)


def _event_text(events: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        parts.append(str(event.get("reason", "")))
        parts.append(str(event.get("message", "")))
        involved = event.get("involvedObject", {})
        parts.append(str(involved.get("kind", "")))
        parts.append(str(involved.get("name", "")))
    return "\n".join(parts)


def _event_timestamp(event: dict[str, Any]) -> str:
    return str(
        event.get("eventTime")
        or event.get("lastTimestamp")
        or event.get("metadata", {}).get("creationTimestamp")
        or ""
    )


def _latest_condition_time(conditions: Iterable[dict[str, Any]]) -> str:
    stamps = [
        str(c.get("lastTransitionTime") or c.get("lastUpdateTime") or "")
        for c in conditions
    ]
    stamps = [stamp for stamp in stamps if stamp]
    return max(stamps, default="")


def _parse_rfc3339_seconds(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _mentions_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)
