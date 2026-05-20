"""Dynamo store/discovery chaos cases D818-D835.

These tests are intentionally opt-in because they mutate a live Dynamo control-plane.
Set ``AIPERF_DYNAMO_CHAOS=1`` plus the per-topology variables named in each
skip message to run them against a disposable cluster.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import pytest

DYNAMO_CHAOS_ENV = "AIPERF_DYNAMO_CHAOS"
DEFAULT_TIMEOUT_S = 60

ChaosArea = Literal["nats", "etcd", "metadata"]


@dataclass(frozen=True, slots=True)
class Prerequisites:
    """Environment gates required before mutating a live Dynamo topology."""

    env_vars: tuple[str, ...] = ()
    requires_proxy: bool = False
    requires_ha: bool = False

    def missing(self, area: ChaosArea) -> list[str]:
        missing = [name for name in self.env_vars if not os.environ.get(name)]
        if self.requires_proxy:
            proxy_url_var = f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_URL"
            proxy_name_var = f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_NAME"
            missing.extend(
                name
                for name in (proxy_url_var, proxy_name_var)
                if not os.environ.get(name)
            )
        if self.requires_ha:
            ha_var = f"AIPERF_DYNAMO_{area.upper()}_HA"
            if os.environ.get(ha_var) != "1":
                missing.append(f"{ha_var}=1")
        return missing


@dataclass(frozen=True, slots=True)
class ChaosCase:
    """Single store/discovery chaos case from the expanded Dynamo matrix."""

    case_id: str
    area: ChaosArea
    name: str
    prerequisites: Prerequisites
    run: Callable[[Cluster], None]

    @property
    def id(self) -> str:
        return f"{self.case_id}-{self.area}-{self.name.replace(' ', '-')}"


@dataclass(frozen=True, slots=True)
class Cluster:
    """Minimal kubectl/toxiproxy facade for destructive opt-in tests."""

    namespace: str
    context: str | None

    def kubectl(
        self, *args: str, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        command = ["kubectl"]
        if self.context:
            command.extend(("--context", self.context))
        command.extend(("-n", self.namespace, *args))
        return subprocess.run(
            command,
            input=stdin,
            text=True,
            check=True,
            capture_output=True,
            timeout=DEFAULT_TIMEOUT_S,
        )

    def wait_for_selector(self, selector: str) -> None:
        self.kubectl(
            "wait",
            "pod",
            "--for=condition=Ready",
            f"--selector={selector}",
            "--timeout=180s",
        )

    def pods_for_selector(self, selector: str) -> list[str]:
        result = self.kubectl(
            "get",
            "pod",
            f"--selector={selector}",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        return [pod for pod in result.stdout.split() if pod]

    def service_yaml(self, service_name: str) -> str:
        return self.kubectl("get", "service", service_name, "-o", "yaml").stdout

    def apply_yaml(self, yaml_text: str) -> None:
        self.kubectl("apply", "-f", "-", stdin=yaml_text)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"missing prerequisite environment variable {name}")
    return value


def _toxiproxy_request(
    area: ChaosArea,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> None:
    base_url = _env(f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_URL")
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if method == "DELETE" and exc.code == 404:
            return
        raise


def _toxiproxy_path(area: ChaosArea, suffix: str) -> str:
    proxy_name = _env(f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_NAME")
    return f"/proxies/{proxy_name}{suffix}"


def _add_toxic(
    area: ChaosArea,
    toxic_name: str,
    toxic_type: str,
    attributes: dict[str, object],
) -> None:
    _toxiproxy_request(
        area,
        "POST",
        _toxiproxy_path(area, "/toxics"),
        {
            "name": toxic_name,
            "type": toxic_type,
            "stream": "downstream",
            "toxicity": 1.0,
            "attributes": attributes,
        },
    )


def _remove_toxic(area: ChaosArea, toxic_name: str) -> None:
    _toxiproxy_request(area, "DELETE", _toxiproxy_path(area, f"/toxics/{toxic_name}"))


def _with_toxic(
    area: ChaosArea,
    toxic_name: str,
    toxic_type: str,
    attributes: dict[str, object],
) -> None:
    _remove_toxic(area, toxic_name)
    try:
        _add_toxic(area, toxic_name, toxic_type, attributes)
        time.sleep(2)
    finally:
        _remove_toxic(area, toxic_name)


def _nats_partition(_: Cluster) -> None:
    _with_toxic("nats", "d818_partition", "timeout", {"timeout": 0})


def _nats_latency(_: Cluster) -> None:
    _with_toxic("nats", "d819_latency", "latency", {"latency": 750, "jitter": 100})


def _nats_bandwidth(_: Cluster) -> None:
    _with_toxic("nats", "d820_bandwidth", "bandwidth", {"rate": 16})


def _nats_reset(_: Cluster) -> None:
    _with_toxic("nats", "d821_reset", "reset_peer", {"timeout": 1000})


def _nats_restart(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_NATS_SELECTOR")
    pods_before = set(cluster.pods_for_selector(selector))
    assert pods_before, f"selector {selector!r} did not match any NATS pods"
    cluster.kubectl("delete", "pod", f"--selector={selector}", "--wait=true")
    cluster.wait_for_selector(selector)
    pods_after = set(cluster.pods_for_selector(selector))
    assert pods_after and pods_after != pods_before


def _nats_selector_delete(cluster: Cluster) -> None:
    service = _env("AIPERF_DYNAMO_NATS_SERVICE")
    original_yaml = cluster.service_yaml(service)
    try:
        cluster.kubectl(
            "patch",
            "service",
            service,
            "--type=merge",
            "-p",
            '{"spec":{"selector":{"aiperf.nvidia.com/chaos":"missing"}}}',
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original_yaml)


def _nats_service_delete(cluster: Cluster) -> None:
    service = _env("AIPERF_DYNAMO_NATS_SERVICE")
    original_yaml = cluster.service_yaml(service)
    try:
        cluster.kubectl("delete", "service", service, "--wait=true")
    finally:
        cluster.apply_yaml(original_yaml)


def _etcd_bandwidth(_: Cluster) -> None:
    _with_toxic("etcd", "d825_bandwidth", "bandwidth", {"rate": 32})


def _etcd_reset(_: Cluster) -> None:
    _with_toxic("etcd", "d826_reset", "reset_peer", {"timeout": 1000})


def _etcd_partition(_: Cluster) -> None:
    _with_toxic("etcd", "d827_partition", "timeout", {"timeout": 0})


def _etcd_compaction(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any etcd pods"
    cluster.kubectl("exec", pods[0], "--", "etcdctl", "compact", "1")


def _etcd_leader_restart(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any etcd pods"
    cluster.kubectl("delete", "pod", pods[0], "--wait=true")
    cluster.wait_for_selector(selector)


def _etcd_quorum_loss(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert len(pods) >= 3, "etcd quorum-loss chaos requires at least three etcd pods"
    victims = pods[:2]
    try:
        cluster.kubectl("delete", "pod", *victims, "--wait=true")
        time.sleep(2)
    finally:
        cluster.wait_for_selector(selector)


def _metadata_configmap_name() -> str:
    return _env("AIPERF_DYNAMO_METADATA_CONFIGMAP")


def _metadata_malformed(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            '{"data":{"endpoints":"not: [valid"}}',
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_duplicate(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            json.dumps(
                {"data": {"duplicate-endpoints": '[{"name":"dup"},{"name":"dup"}]'}}
            ),
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_delete(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl("delete", "configmap", configmap, "--wait=true")
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_freeze(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_METADATA_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any metadata pods"
    cluster.kubectl(
        "exec", pods[0], "--", "sh", "-c", "kill -STOP 1; sleep 2; kill -CONT 1"
    )


def _metadata_watch_storm(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    for index in range(10):
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            f'{{"metadata":{{"annotations":{{"aiperf.nvidia.com/chaos-watch-storm":"{index}"}}}}}}',
        )


NATS_PROXY = Prerequisites(requires_proxy=True)
NATS_TOPOLOGY = Prerequisites(("AIPERF_DYNAMO_NATS_SELECTOR",))
NATS_SERVICE = Prerequisites(
    ("AIPERF_DYNAMO_NATS_SELECTOR", "AIPERF_DYNAMO_NATS_SERVICE")
)
ETCD_PROXY_HA = Prerequisites(requires_proxy=True, requires_ha=True)
ETCD_HA = Prerequisites(("AIPERF_DYNAMO_ETCD_SELECTOR",), requires_ha=True)
METADATA_TOPOLOGY = Prerequisites(
    ("AIPERF_DYNAMO_METADATA_CONFIGMAP", "AIPERF_DYNAMO_METADATA_SELECTOR")
)
METADATA_CONFIG = Prerequisites(("AIPERF_DYNAMO_METADATA_CONFIGMAP",))

CASES: tuple[ChaosCase, ...] = (
    ChaosCase("D818", "nats", "partition", NATS_PROXY, _nats_partition),
    ChaosCase("D819", "nats", "latency", NATS_PROXY, _nats_latency),
    ChaosCase("D820", "nats", "bandwidth", NATS_PROXY, _nats_bandwidth),
    ChaosCase("D821", "nats", "reset", NATS_PROXY, _nats_reset),
    ChaosCase("D822", "nats", "restart", NATS_TOPOLOGY, _nats_restart),
    ChaosCase(
        "D823", "nats", "selector service delete", NATS_SERVICE, _nats_selector_delete
    ),
    ChaosCase("D824", "nats", "service delete", NATS_SERVICE, _nats_service_delete),
    ChaosCase("D825", "etcd", "bandwidth", ETCD_PROXY_HA, _etcd_bandwidth),
    ChaosCase("D826", "etcd", "reset", ETCD_PROXY_HA, _etcd_reset),
    ChaosCase("D827", "etcd", "partition", ETCD_PROXY_HA, _etcd_partition),
    ChaosCase("D828", "etcd", "compaction", ETCD_HA, _etcd_compaction),
    ChaosCase("D829", "etcd", "leader restart", ETCD_HA, _etcd_leader_restart),
    ChaosCase("D830", "etcd", "quorum loss", ETCD_HA, _etcd_quorum_loss),
    ChaosCase("D831", "metadata", "malformed", METADATA_CONFIG, _metadata_malformed),
    ChaosCase("D832", "metadata", "duplicate", METADATA_CONFIG, _metadata_duplicate),
    ChaosCase("D833", "metadata", "delete", METADATA_CONFIG, _metadata_delete),
    ChaosCase("D834", "metadata", "freeze", METADATA_TOPOLOGY, _metadata_freeze),
    ChaosCase(
        "D835", "metadata", "watch storm", METADATA_CONFIG, _metadata_watch_storm
    ),
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case" in metafunc.fixturenames:
        metafunc.parametrize("case", CASES, ids=[case.id for case in CASES])


@pytest.fixture
def cluster() -> Iterator[Cluster]:
    if os.environ.get(DYNAMO_CHAOS_ENV) != "1":
        pytest.skip(f"set {DYNAMO_CHAOS_ENV}=1 to run destructive Dynamo chaos tests")
    if not shutil.which("kubectl"):
        pytest.skip("kubectl is required for Dynamo chaos tests")
    namespace = os.environ.get("AIPERF_DYNAMO_NAMESPACE")
    if not namespace:
        pytest.skip("set AIPERF_DYNAMO_NAMESPACE to the disposable Dynamo namespace")
    yield Cluster(
        namespace=namespace, context=os.environ.get("AIPERF_DYNAMO_KUBE_CONTEXT")
    )


def test_expanded_spec_cases_cover_d818_through_d835() -> None:
    assert [case.case_id for case in CASES] == [
        f"D{case_id}" for case_id in range(818, 836)
    ]


def test_store_discovery_chaos_case_has_explicit_prerequisites(case: ChaosCase) -> None:
    missing = case.prerequisites.missing(case.area)
    assert (
        case.prerequisites.env_vars
        or case.prerequisites.requires_proxy
        or case.prerequisites.requires_ha
    )
    assert all(item for item in missing)


def test_store_discovery_chaos_case_executes_with_required_topology(
    cluster: Cluster, case: ChaosCase
) -> None:
    missing = case.prerequisites.missing(case.area)
    if missing:
        pytest.skip(
            f"{case.case_id} requires explicit prerequisites: {', '.join(missing)}"
        )
    case.run(cluster)
