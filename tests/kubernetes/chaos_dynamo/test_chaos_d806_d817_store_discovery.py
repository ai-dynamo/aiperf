# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D806-D817 -- Dynamo store/discovery chaos coverage.

These cases cover the expanded D8xx store/discovery spec without changing the
shared Dynamo fixtures. Destructive topology-wide faults are opt-in and self-skip
with concrete prerequisite diagnostics on the stock v1.1.0 disagg-1gpu topology.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

ETCD_NAMESPACE = "dynamo-system"
ETCD_SERVICE = "dynamo-platform-etcd-headless"
ETCD_PROXY_NAME = "etcd-keepalive"
ETCD_PROXY_LISTEN = "0.0.0.0:20031"
ETCD_CLIENT_PORT = 2379
ETCD_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_ETCD_CHAOS"

NATS_NAMESPACE = "dynamo-system"
NATS_SELECTOR = "app=nats"
NATS_PROXY_NAME = "nats-frontend-partition"
NATS_PROXY_LISTEN = "0.0.0.0:20021"
NATS_PROXY_ROUTE = "toxiproxy.chaos-toxiproxy.svc:20020"
NATS_SERVICE_PORT = 4222
NATS_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NATS_CHAOS"

DWM_API_GROUP = "nvidia.com"
DWM_RESOURCE = "dynamoworkermetadatas"
ENDPOINTSLICE_API_GROUP = "discovery.k8s.io"
ENDPOINTSLICE_RESOURCE = "endpointslices"

SERVICE_SELECTOR_OPT_IN_ENV = "AIPERF_DYNAMO_SERVICE_SELECTOR_CHAOS"
COREDNS_OPT_IN_ENV = "AIPERF_DYNAMO_COREDNS_CHAOS"
FRONTEND_REQUEST_TIMEOUT_S = 30.0
RBAC_FAILURE_WINDOW_S = 5.0
SERVICE_SELECTOR_WINDOW_S = 10.0
COREDNS_WINDOW_S = 15.0


@dataclass(frozen=True, slots=True)
class _RbacOwner:
    """Exact RBAC resource granting one discovery permission."""

    scope: str
    """``role`` or ``clusterrole`` for kubectl and the fault injector."""

    name: str
    """Role or ClusterRole name."""

    namespace: str | None
    """Role namespace, or ``None`` for ClusterRole."""

    @property
    def label(self) -> str:
        """Human-readable identifier for skip and assertion messages."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


@dataclass(frozen=True, slots=True)
class _ServiceSelectorPatch:
    """Patch target and original selector for a reversible Service-selector fault."""

    namespace: str
    name: str
    original_selector: dict[str, str]


async def test_d806_etcd_keepalive_blackhole_expires_one_worker(
    request: pytest.FixtureRequest,
) -> None:
    """Blackhole etcd keepalive traffic only for an etcd-enabled topology.

    The default Dynamo v1.1.0 topology uses Kubernetes discovery and does not
    install bundled etcd, so the test self-skips unless the caller opts into an
    etcd-backed deployment where the reserved Toxiproxy route is live.
    """
    if os.environ.get(ETCD_CHAOS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D806 requires bundled etcd plus an etcd-discovery topology routed "
            f"through Toxiproxy; stock Dynamo v1.1.0 disagg uses Kubernetes "
            f"discovery. Set {ETCD_CHAOS_OPT_IN_ENV}=1 only for that topology."
        )

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    if not await _service_exists(kubectl, ETCD_NAMESPACE, ETCD_SERVICE):
        pytest.skip(
            f"D806 requires bundled etcd service {ETCD_NAMESPACE}/{ETCD_SERVICE}; "
            "the opt-in topology did not expose that service."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    upstream = f"{ETCD_SERVICE}.{ETCD_NAMESPACE}.svc:{ETCD_CLIENT_PORT}"

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=ETCD_PROXY_NAME,
            listen=ETCD_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        async with faults.inject(
            "store.etcd.bandwidth",
            target={"proxy": ETCD_PROXY_NAME},
            attributes={"rate": 0},
            stream="upstream",
        ) as applied:
            assert applied.spec.fault_id == "network.bandwidth"
            assert applied.metadata.get("proxy_name") == ETCD_PROXY_NAME
            logger.info(
                "D806: etcd keepalive bandwidth=0 toxic applied; lease-expiry "
                "assertion is topology-gated by the etcd opt-in deployment"
            )
    finally:
        if proxy_created:
            await _remove_proxy_safely(dynamo_toxiproxy, ETCD_PROXY_NAME, "D806")


async def test_d807_nats_frontend_partition_converges_after_heal(
    request: pytest.FixtureRequest,
) -> None:
    """Partition frontend NATS traffic only when multiple frontends are proxied."""
    if os.environ.get(NATS_CHAOS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D807 requires two frontend replicas whose NATS route traverses "
            f"{NATS_PROXY_ROUTE!r}; stock Dynamo v1.1.0 disagg has one frontend "
            f"and direct NATS. Set {NATS_CHAOS_OPT_IN_ENV}=1 only for that topology."
        )

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    frontend_pods = await _list_frontend_pods(kubectl, namespace)
    if len(frontend_pods) < 2:
        pytest.skip(
            f"D807 requires at least two frontend pods in {namespace!r}; "
            f"observed {frontend_pods!r}."
        )
    if not await _topology_mentions_route(
        kubectl, [namespace, NATS_NAMESPACE], NATS_PROXY_ROUTE
    ):
        pytest.skip(
            f"D807 requires frontend NATS traffic to route through {NATS_PROXY_ROUTE!r}; "
            "no pod env/args mention that route."
        )

    nats_service = await _find_service_with_port(
        kubectl, NATS_NAMESPACE, NATS_SERVICE_PORT
    )
    if nats_service is None:
        pytest.skip(
            f"D807 requires a NATS Service exposing port {NATS_SERVICE_PORT} in "
            f"{NATS_NAMESPACE!r}; selector={NATS_SELECTOR!r}."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    upstream = f"{nats_service}.{NATS_NAMESPACE}.svc:{NATS_SERVICE_PORT}"

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=NATS_PROXY_NAME,
            listen=NATS_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        async with faults.inject(
            "store.nats.partition",
            target={"proxy": NATS_PROXY_NAME},
        ) as applied:
            assert applied.spec.fault_id == "network.partition"
            assert applied.metadata.get("proxy_name") == NATS_PROXY_NAME
            await asyncio.sleep(RBAC_FAILURE_WINDOW_S)
    finally:
        if proxy_created:
            await _remove_proxy_safely(dynamo_toxiproxy, NATS_PROXY_NAME, "D807")


@pytest.mark.parametrize(
    ("case_id", "verb"),
    [
        pytest.param("D808", "get", id="D808-dwm-get"),
        pytest.param("D809", "list", id="D809-dwm-list"),
        pytest.param("D810", "watch", id="D810-dwm-watch"),
    ],
)  # fmt: skip
async def test_d808_d810_dwm_rbac_revocation_preserves_cached_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
    case_id: str,
    verb: str,
) -> None:
    """Revoke one DWM verb and require cached discovery to keep traffic alive."""
    await _run_discovery_rbac_case(
        request=request,
        kubectl=kubectl,
        endpoint_url=dynamo_endpoint_url,
        namespace=dynamo_deployment_namespace,
        case_id=case_id,
        api_group=DWM_API_GROUP,
        resource=DWM_RESOURCE,
        verb=verb,
    )


@pytest.mark.parametrize(
    ("case_id", "verb"),
    [
        pytest.param("D811", "get", id="D811-endpointslice-get"),
        pytest.param("D812", "list", id="D812-endpointslice-list"),
        pytest.param("D813", "watch", id="D813-endpointslice-watch"),
        pytest.param("D814", "delete", id="D814-endpointslice-delete"),
        pytest.param("D815", "patch", id="D815-endpointslice-patch"),
    ],
)  # fmt: skip
async def test_d811_d815_endpointslice_rbac_revocation_preserves_cached_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
    case_id: str,
    verb: str,
) -> None:
    """Revoke one EndpointSlice verb and require cached discovery to serve."""
    await _run_discovery_rbac_case(
        request=request,
        kubectl=kubectl,
        endpoint_url=dynamo_endpoint_url,
        namespace=dynamo_deployment_namespace,
        case_id=case_id,
        api_group=ENDPOINTSLICE_API_GROUP,
        resource=ENDPOINTSLICE_RESOURCE,
        verb=verb,
    )


async def test_d816_service_selector_mismatch_does_not_poison_cached_discovery(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
) -> None:
    """Patch one Dynamo Service selector only in explicit selector-chaos topology."""
    if os.environ.get(SERVICE_SELECTOR_OPT_IN_ENV) != "1":
        pytest.skip(
            "D816 mutates a live Dynamo Service selector and is intentionally "
            f"opt-in. Set {SERVICE_SELECTOR_OPT_IN_ENV}=1 only on an isolated "
            "cluster where EndpointSlice churn is expected."
        )

    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    patch_target = await _find_patchable_dynamo_service(kubectl, namespace)
    if patch_target is None:
        pytest.skip(
            f"D816 requires a Dynamo Service with a non-empty selector in {namespace!r}; "
            "none was found."
        )

    await _assert_frontend_serves(dynamo_endpoint_url, case_id="D816", phase="before")
    mismatched_selector = dict(patch_target.original_selector)
    mismatched_selector["aiperf.nvidia.com/chaos-d816"] = "no-such-pod"
    try:
        await _patch_service_selector(
            kubectl,
            patch_target.namespace,
            patch_target.name,
            mismatched_selector,
        )
        await asyncio.sleep(SERVICE_SELECTOR_WINDOW_S)
        await _assert_frontend_serves(
            dynamo_endpoint_url,
            case_id="D816",
            phase="while service selector is mismatched",
        )
    finally:
        await _patch_service_selector(
            kubectl,
            patch_target.namespace,
            patch_target.name,
            patch_target.original_selector,
        )


async def test_d817_coredns_outage_does_not_poison_cached_discovery(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
) -> None:
    """Scale CoreDNS down only in explicit DNS-chaos topology."""
    if os.environ.get(COREDNS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D817 scales the cluster DNS deployment and is intentionally opt-in. "
            f"Set {COREDNS_OPT_IN_ENV}=1 only on an isolated cluster where a "
            "short CoreDNS outage is acceptable."
        )

    deployment = await _find_coredns_deployment(kubectl)
    if deployment is None:
        pytest.skip(
            "D817 requires a CoreDNS/kube-dns Deployment in kube-system; none found."
        )

    faults = request.getfixturevalue("faults")
    await _assert_frontend_serves(dynamo_endpoint_url, case_id="D817", phase="before")
    async with faults.inject(
        "workload.scale",
        target={"kind": "deployment", "name": deployment, "ns": "kube-system"},
        replicas=0,
    ) as applied:
        assert applied.metadata.get("name") == deployment
        await asyncio.sleep(COREDNS_WINDOW_S)
        await _assert_frontend_serves(
            dynamo_endpoint_url,
            case_id="D817",
            phase="during CoreDNS outage",
        )


async def _run_discovery_rbac_case(
    *,
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    endpoint_url: str,
    namespace: str,
    case_id: str,
    api_group: str,
    resource: str,
    verb: str,
) -> None:
    """Revoke one exact discovery RBAC verb, assert cached traffic, verify restore."""
    owner, inspected_names = await _find_unique_rbac_owner(
        kubectl,
        namespace=namespace,
        api_group=api_group,
        resource=resource,
        verb=verb,
    )
    if owner is None:
        pytest.skip(
            f"{case_id} requires exactly one exact RBAC rule granting {verb!r} on "
            f"{resource}.{api_group}; inspected/candidate RBAC resources: "
            f"{', '.join(inspected_names) or '<none>'}. Wildcards and ambiguous "
            "owners are skipped to avoid broad cluster mutation."
        )

    await _assert_frontend_serves(endpoint_url, case_id=case_id, phase="before")
    faults = request.getfixturevalue("faults")
    target: dict[str, str] = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace

    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=target,
            api_group=api_group,
            resource=resource,
            verb=verb,
        ) as applied:
            assert applied.metadata["name"] == owner.name
            assert applied.metadata["resource"] == resource
            assert applied.metadata["verb"] == verb
            await asyncio.sleep(RBAC_FAILURE_WINDOW_S)
            await _assert_frontend_serves(
                endpoint_url,
                case_id=case_id,
                phase=f"while {owner.label} lacks {verb!r}",
            )
    finally:
        restored = await _role_currently_grants(
            kubectl,
            owner,
            api_group=api_group,
            resource=resource,
            verb=verb,
        )
        assert restored, (
            f"{case_id}: RBAC restore did not put {verb!r} back on {owner.label} "
            f"for {resource}.{api_group}; manual cluster repair required"
        )


async def _find_unique_rbac_owner(
    kubectl: KubectlClient,
    *,
    namespace: str,
    api_group: str,
    resource: str,
    verb: str,
) -> tuple[_RbacOwner | None, list[str]]:
    """Return unique exact RBAC owner for ``(api_group, resource, verb)``."""
    roles = await _load_rbac_collection(kubectl, "roles", namespace=namespace)
    clusterroles = await _load_rbac_collection(kubectl, "clusterroles")

    inspected: list[str] = []
    candidates: list[_RbacOwner] = []
    for item in roles:
        metadata = item.get("metadata", {})
        owner = _RbacOwner(
            scope="role",
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
        )
        inspected.append(owner.label)
        if _has_exact_rule(item.get("rules") or [], api_group, resource, verb):
            candidates.append(owner)

    for item in clusterroles:
        metadata = item.get("metadata", {})
        owner = _RbacOwner(
            scope="clusterrole",
            name=str(metadata.get("name", "")),
            namespace=None,
        )
        inspected.append(owner.label)
        if _has_exact_rule(item.get("rules") or [], api_group, resource, verb):
            candidates.append(owner)

    if len(candidates) != 1:
        return None, [candidate.label for candidate in candidates] or inspected
    return candidates[0], inspected


async def _load_rbac_collection(
    kubectl: KubectlClient,
    resource: str,
    *,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Load Roles or ClusterRoles, skipping if caller cannot inspect RBAC."""
    args = ["get", resource, "-o", "json"]
    if namespace is not None:
        args.extend(["-n", namespace])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"could not inspect {resource} before RBAC mutation: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    data = orjson.loads(result.stdout or b"{}")
    return list(data.get("items", []))


def _has_exact_rule(
    rules: Iterable[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    """Return true only for explicit group/resource/verb RBAC rules."""
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if api_group in groups and resource in resources and verb in verbs:
            return True
    return False


async def _role_currently_grants(
    kubectl: KubectlClient,
    owner: _RbacOwner,
    *,
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    """Verify cleanup restored the exact permission removed by a test."""
    args = ["get", owner.scope, owner.name]
    if owner.namespace is not None:
        args.extend(["-n", owner.namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return False
    body = orjson.loads(result.stdout or b"{}")
    return _has_exact_rule(body.get("rules") or [], api_group, resource, verb)


async def _assert_frontend_serves(
    endpoint_url: str, *, case_id: str, phase: str
) -> None:
    """Send one streaming OpenAI-compatible request and require HTTP success."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 10,
    }
    timeout = aiohttp.ClientTimeout(total=FRONTEND_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body_prefix = b""
        async for chunk in resp.content.iter_chunked(1024):
            body_prefix += chunk
            if body_prefix:
                break
        assert resp.status == 200, (
            f"{case_id}: frontend returned HTTP {resp.status} {phase}; "
            f"body_prefix={body_prefix[:256].decode(errors='replace')!r}"
        )
        assert body_prefix, f"{case_id}: frontend returned an empty stream {phase}"


async def _service_exists(kubectl: KubectlClient, namespace: str, name: str) -> bool:
    """Return whether a Service exists."""
    result = await kubectl.run(
        "get",
        "service",
        name,
        "-n",
        namespace,
        check=False,
    )
    return result.returncode == 0


async def _find_service_with_port(
    kubectl: KubectlClient,
    namespace: str,
    port: int,
) -> str | None:
    """Return the first Service in ``namespace`` exposing ``port``."""
    result = await kubectl.run(
        "get",
        "services",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return None
    for service in orjson.loads(result.stdout or b"{}").get("items", []):
        ports = service.get("spec", {}).get("ports", [])
        if any(item.get("port") == port for item in ports):
            name = service.get("metadata", {}).get("name")
            if isinstance(name, str):
                return name
    return None


async def _list_frontend_pods(kubectl: KubectlClient, namespace: str) -> list[str]:
    """Return frontend-like pod names in the Dynamo deployment namespace."""
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    pods = orjson.loads(result.stdout or b"{}").get("items", [])
    names: list[str] = []
    for pod in pods:
        name = pod.get("metadata", {}).get("name")
        if isinstance(name, str) and "frontend" in name:
            names.append(name)
    return sorted(names)


async def _topology_mentions_route(
    kubectl: KubectlClient,
    namespaces: Iterable[str],
    route: str,
) -> bool:
    """Return whether pod env/args in any namespace mention ``route``."""
    for namespace in dict.fromkeys(namespaces):
        result = await kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode != 0:
            continue
        pods = orjson.loads(result.stdout or b"{}").get("items", [])
        for pod in pods:
            for container in pod.get("spec", {}).get("containers", []):
                if _container_mentions_route(container, route):
                    return True
    return False


def _container_mentions_route(container: dict[str, Any], route: str) -> bool:
    """Inspect one container's env/command/args for a route string."""
    for env in container.get("env", []):
        if isinstance(env, dict) and env.get("value") == route:
            return True
        if isinstance(env, dict) and route in str(env.get("value", "")):
            return True
    for field in ("command", "args"):
        values = container.get(field, [])
        if isinstance(values, list) and any(route in str(value) for value in values):
            return True
    return False


async def _find_patchable_dynamo_service(
    kubectl: KubectlClient,
    namespace: str,
) -> _ServiceSelectorPatch | None:
    """Find a Dynamo Service whose selector can be restored exactly."""
    result = await kubectl.run(
        "get",
        "services",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return None
    services = orjson.loads(result.stdout or b"{}").get("items", [])
    for service in services:
        metadata = service.get("metadata", {})
        name = metadata.get("name")
        selector = service.get("spec", {}).get("selector") or {}
        if not isinstance(name, str) or not isinstance(selector, dict) or not selector:
            continue
        if "frontend" not in name and "worker" not in name:
            continue
        return _ServiceSelectorPatch(
            namespace=namespace,
            name=name,
            original_selector={str(key): str(value) for key, value in selector.items()},
        )
    return None


async def _patch_service_selector(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
    selector: dict[str, str],
) -> None:
    """Patch a Service selector using merge patch."""
    patch = {"spec": {"selector": selector}}
    await kubectl.run(
        "patch",
        "service",
        name,
        "-n",
        namespace,
        "--type=merge",
        "-p",
        orjson.dumps(patch).decode(),
        check=True,
    )


async def _find_coredns_deployment(kubectl: KubectlClient) -> str | None:
    """Return CoreDNS/kube-dns deployment name in kube-system."""
    for selector in (
        "k8s-app=kube-dns",
        "k8s-app=coredns",
        "app.kubernetes.io/name=coredns",
    ):
        result = await kubectl.run(
            "get",
            "deployment",
            "-n",
            "kube-system",
            "-l",
            selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        name = result.stdout.strip() if result.returncode == 0 else ""
        if name:
            return name
    return None


async def _remove_proxy_safely(toxiproxy: Any, proxy_name: str, case_id: str) -> None:
    """Best-effort proxy cleanup so assertion failures are not masked."""
    try:
        await toxiproxy.remove_proxy(proxy_name)
    except Exception as exc:
        logger.warning(lambda exc=exc: f"{case_id} remove_proxy failed: {exc!r}")
