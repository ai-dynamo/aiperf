# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The audit CI Kind-cluster CPU budget must fit under the advertised floor.

Regression coverage for the weekly "Run Kubernetes Audit Tests" workflow,
which failed every scheduled run with operator pods stuck Pending on
``FailedScheduling: 0/1 nodes available: Insufficient cpu``. The test
fixtures pinned production-scale requests (worker pod 3350m + controller pod
2600m = 5950m per AIPerfJob) onto a single-node Kind cluster whose
ubuntu-latest runner has only 4 vCPUs, so no benchmark pod ever scheduled.

The audit workflow opts into two complementary mechanisms:

- ``K8S_TEST_KIND_SIZED_JOB_PODS=true``: KIND_JOB_POD_ENV right-sizes the
  job-pod requests (worker pod 1000m, controller containers at the
  operator's calibrated small defaults).
- ``K8S_TEST_NODE_CPU_OVERCOMMIT=32``: the local_cluster fixture raises the
  node's advertised CPU capacity/allocatable to a 32-core floor so the
  scheduler admits the workload (re-applied on an interval; kubelet
  re-asserts real capacity within seconds).

The audit suite runs serially and sweeps execute cells sequentially, so peak
benchmark-pod demand is one AIPerfJob (controller pod + worker pod) plus,
during sweeps, one sweep-controller pod. That peak must fit alongside the
fixed tenants under the advertised floor:

- kube-system static/daemon pods: 950m (measured on kind v0.27 / k8s 1.32:
  apiserver 250m, controller-manager 200m, coredns x2 200m, etcd 100m,
  kindnet 100m, scheduler 100m; kind v0.31 / k8s 1.35 measures the same)
- jobset-controller-manager: 500m (upstream v0.8.0 manifests.yaml)
- mock server Deployment: parsed from dev/deploy/mock-server.yaml
- operator pod: parsed from the chart's values.yaml (operator +
  results-server containers; dashboard disabled by default)

Both the kind-sized and the production-default request sets are checked
against the floor, so the suite stays schedulable whether or not the
right-sizing opt-in is set. Requests only gate scheduling
(resource_mode=burstable sets no limits), so containers still burst to real
node CPU at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from aiperf.common.environment import Environment
from aiperf.kubernetes.constants import WORKER_IMPORT_MEMORY_FLOOR_MIB
from aiperf.kubernetes.environment import (
    CONTROLLER_RESOURCE_KEYS,
    _K8sEnvironment,
)
from aiperf.kubernetes.jobset import split_worker_pod_resources
from aiperf.kubernetes.utils import parse_memory_mib
from tests.kubernetes.helpers.cluster import KIND_CPU_OVERCOMMIT_FLOOR
from tests.kubernetes.helpers.operator import KIND_JOB_POD_ENV

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ADVERTISED_FLOOR_MCPU = KIND_CPU_OVERCOMMIT_FLOOR * 1000
"""Advertised CPU floor the audit CI workflow fakes on its single Kind node."""

_KUBE_SYSTEM_CPU_MCPU = 950
"""Measured on a fresh kind v0.31.0 / k8s v1.35.0 control-plane node:
apiserver 250m + controller-manager 200m + coredns 2x100m + etcd 100m +
kindnet 100m + scheduler 100m (kind v0.27 / k8s 1.32 measures the same)."""

_JOBSET_CONTROLLER_CPU_MCPU = 500
"""jobset v0.8.0 manifests.yaml: controller-manager requests.cpu."""

_SWEEP_CONTROLLER_CPU_MCPU = 75
"""K8sEnvironment.SWEEP_CONTROLLER default; runs alongside one sweep cell."""


def _parse_cpu_mcpu(quantity: str) -> int:
    """Parse a Kubernetes CPU quantity into millicores."""
    if quantity.endswith("m"):
        return int(quantity[:-1])
    return int(float(quantity) * 1000)


def _fixture_overrides() -> dict[str, str]:
    """KIND_JOB_POD_ENV as {AIPERF_K8S_<NAME>_<FIELD>: value}."""
    pairs = dict(entry.split("=", 1) for entry in KIND_JOB_POD_ENV)
    assert len(pairs) == len(KIND_JOB_POD_ENV), "duplicate keys in KIND_JOB_POD_ENV"
    return pairs


@pytest.fixture
def _clean_k8s_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ambient AIPERF_K8S_* so _K8sEnvironment() reflects shipped defaults."""
    for var in list(os.environ):
        if var.startswith("AIPERF_K8S_"):
            monkeypatch.delenv(var, raising=False)


def _controller_pod_cpu_mcpu(
    overrides: dict[str, str], defaults: _K8sEnvironment
) -> int:
    """Controller-pod request total: fixture override or shipped default per
    container."""
    return sum(
        _parse_cpu_mcpu(
            overrides.get(f"AIPERF_K8S_{name}_CPU", getattr(defaults, name).CPU)
        )
        for name in CONTROLLER_RESOURCE_KEYS
    )


def _mock_server_cpu_mcpu() -> int:
    docs = yaml.safe_load_all(
        (_REPO_ROOT / "dev" / "deploy" / "mock-server.yaml").read_text()
    )
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    return _parse_cpu_mcpu(container["resources"]["requests"]["cpu"])


def _operator_pod_cpu_mcpu() -> int:
    values = yaml.safe_load(
        (_REPO_ROOT / "deploy" / "helm" / "aiperf-operator" / "values.yaml").read_text()
    )
    assert values["dashboard"]["enabled"] is False, (
        "dashboard sidecar adds requests this budget does not account for"
    )
    operator = _parse_cpu_mcpu(values["operator"]["resources"]["requests"]["cpu"])
    results_server = _parse_cpu_mcpu(
        values["resultsServer"]["resources"]["requests"]["cpu"]
    )
    return operator + results_server


def _fixed_tenants_mcpu() -> int:
    return (
        _KUBE_SYSTEM_CPU_MCPU
        + _JOBSET_CONTROLLER_CPU_MCPU
        + _mock_server_cpu_mcpu()
        + _operator_pod_cpu_mcpu()
    )


def test_kind_job_pod_env_is_parseable_and_positive() -> None:
    for key, value in _fixture_overrides().items():
        assert key.startswith("AIPERF_K8S_")
        if key.rsplit("_", 1)[-1] == "CPU":
            assert _parse_cpu_mcpu(value) > 0, key


@pytest.mark.usefixtures("_clean_k8s_env")
def test_kind_sized_peak_fits_advertised_floor() -> None:
    """Audit CI configuration: kind-sized job pods + fixed tenants must fit
    under the advertised CPU floor the fixture fakes."""
    defaults = _K8sEnvironment()
    overrides = _fixture_overrides()

    peak_benchmark_mcpu = (
        _controller_pod_cpu_mcpu(overrides, defaults)
        + _parse_cpu_mcpu(overrides["AIPERF_K8S_WORKER_POD_CPU"])
        + _SWEEP_CONTROLLER_CPU_MCPU
    )
    total_mcpu = peak_benchmark_mcpu + _fixed_tenants_mcpu()

    assert total_mcpu <= _ADVERTISED_FLOOR_MCPU, (
        f"kind-sized peak {peak_benchmark_mcpu}m + fixed tenants "
        f"{_fixed_tenants_mcpu()}m = {total_mcpu}m exceeds the advertised "
        f"{_ADVERTISED_FLOOR_MCPU}m floor; pods would sit Pending with "
        "FailedScheduling"
    )


@pytest.mark.usefixtures("_clean_k8s_env")
def test_production_default_peak_also_fits_advertised_floor() -> None:
    """With the floor alone (right-sizing opt-in off), production-calibrated
    job pods plus fixed tenants must still fit — the floor is the mechanism
    that lets production defaults schedule on a 4-vCPU runner."""
    defaults = _K8sEnvironment()

    peak_benchmark_mcpu = (
        _controller_pod_cpu_mcpu({}, defaults)
        + _parse_cpu_mcpu(defaults.WORKER_POD.CPU)
        + _SWEEP_CONTROLLER_CPU_MCPU
    )
    total_mcpu = peak_benchmark_mcpu + _fixed_tenants_mcpu()

    assert total_mcpu <= _ADVERTISED_FLOOR_MCPU, (
        f"production-default peak {peak_benchmark_mcpu}m + fixed tenants "
        f"{_fixed_tenants_mcpu()}m = {total_mcpu}m exceeds the advertised "
        f"{_ADVERTISED_FLOOR_MCPU}m floor; the overcommit floor no longer "
        "covers production-sized pods"
    )


@pytest.mark.usefixtures("_clean_k8s_env")
def test_worker_pod_budget_renders_at_default_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture's WORKER_POD budget must survive render-time split validation.

    The operator divides the worker-pod budget across 1 worker-pod-manager +
    DEFAULT_WORKERS_PER_POD workers + that many record processors and refuses
    shares below the per-worker import-memory floor. A too-small memory budget
    therefore fails JobSet rendering with ValueError instead of merely
    scheduling tight — a failure mode the CPU budget check above cannot see.
    """
    for key, value in _fixture_overrides().items():
        monkeypatch.setenv(key, value)
    env = _K8sEnvironment()
    worker_count = Environment.WORKER.DEFAULT_WORKERS_PER_POD
    record_processors = max(1, worker_count // env.RECORD_PROCESSOR_SCALE_FACTOR)

    split = split_worker_pod_resources(
        env.WORKER_POD.to_k8s_resources(burstable=True),
        worker_count,
        record_processors,
        None,
        burstable=True,
    )

    worker_shares = split[1 : 1 + worker_count]
    floor = WORKER_IMPORT_MEMORY_FLOOR_MIB
    assert all(
        parse_memory_mib(share["requests"]["memory"]) >= floor
        for share in worker_shares
    ), (
        f"worker containers fall below the {floor}Mi import-memory floor at "
        f"the default {worker_count}-worker/{record_processors}-record-"
        "processor topology"
    )
