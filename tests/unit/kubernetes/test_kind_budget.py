# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kind test-cluster CPU budget must fit a 4-vCPU CI runner.

Regression coverage for the weekly "Run Kubernetes Audit Tests" workflow,
which failed every scheduled run with operator pods stuck Pending on
``FailedScheduling: 0/1 nodes are available: Insufficient cpu``. The test
fixtures pinned production-scale requests (worker pod 3350m + controller pod
2600m = 5950m per AIPerfJob) onto a single-node Kind cluster whose
ubuntu-latest runner has only 4 vCPUs, so no benchmark pod ever scheduled.

The audit/chaos suites run serially and sweeps execute cells sequentially,
so peak benchmark-pod demand is one AIPerfJob (controller pod + worker pod)
plus, during sweeps, one sweep-controller pod. That peak must fit alongside
the fixed tenants:

- kube-system static/daemon pods: 950m (measured on kind v0.27 / k8s 1.32:
  apiserver 250m, controller-manager 200m, coredns x2 200m, etcd 100m,
  kindnet 100m, scheduler 100m)
- jobset-controller-manager: 500m (upstream v0.8.0 manifests.yaml)
- mock server Deployment: parsed from dev/deploy/mock-server.yaml
- operator pod: parsed from the chart's values.yaml (operator +
  results-server containers; dashboard disabled by default)

Requests only gate scheduling (resource_mode=burstable sets no limits), so
containers still burst to full node CPU at runtime; shrinking requests does
not slow the benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from aiperf.kubernetes.environment import (
    CONTROLLER_RESOURCE_KEYS,
    _K8sEnvironment,
)
from tests.kubernetes.helpers.operator import KIND_JOB_POD_ENV

_REPO_ROOT = Path(__file__).resolve().parents[3]

_KIND_NODE_CPU_MCPU = 4000
"""ubuntu-latest GitHub-hosted runner: 4 vCPUs; Kind reserves nothing."""

_KUBE_SYSTEM_CPU_MCPU = 950
"""Measured on a fresh kind v0.27.0 / k8s v1.32.2 control-plane node."""

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


def _effective_container_cpu_mcpu(
    overrides: dict[str, str], defaults: _K8sEnvironment, name: str
) -> int:
    """Effective request for one controller container: fixture override or
    the operator's own shipped default."""
    override = overrides.get(f"AIPERF_K8S_{name}_CPU")
    if override is not None:
        return _parse_cpu_mcpu(override)
    return _parse_cpu_mcpu(getattr(defaults, name).CPU)


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


def test_kind_job_pod_env_is_parseable_and_positive() -> None:
    for key, value in _fixture_overrides().items():
        assert key.startswith("AIPERF_K8S_")
        if key.rsplit("_", 1)[-1] == "CPU":
            assert _parse_cpu_mcpu(value) > 0, key


@pytest.mark.usefixtures("_clean_k8s_env")
def test_peak_benchmark_pod_cpu_fits_kind_node() -> None:
    """One AIPerfJob at peak fan-out + sweep controller must fit beside the
    fixed tenants on a 4-vCPU Kind node."""
    overrides = _fixture_overrides()
    defaults = _K8sEnvironment()

    controller_mcpu = sum(
        _effective_container_cpu_mcpu(overrides, defaults, name)
        for name in CONTROLLER_RESOURCE_KEYS
    )
    worker_pod_mcpu = _parse_cpu_mcpu(overrides["AIPERF_K8S_WORKER_POD_CPU"])

    peak_benchmark_mcpu = controller_mcpu + worker_pod_mcpu + _SWEEP_CONTROLLER_CPU_MCPU
    fixed_mcpu = (
        _KUBE_SYSTEM_CPU_MCPU
        + _JOBSET_CONTROLLER_CPU_MCPU
        + _mock_server_cpu_mcpu()
        + _operator_pod_cpu_mcpu()
    )

    budget_mcpu = _KIND_NODE_CPU_MCPU - fixed_mcpu
    assert peak_benchmark_mcpu <= budget_mcpu, (
        f"peak benchmark-pod request {peak_benchmark_mcpu}m (controller "
        f"{controller_mcpu}m + worker pod {worker_pod_mcpu}m + sweep controller "
        f"{_SWEEP_CONTROLLER_CPU_MCPU}m) exceeds the {budget_mcpu}m left on a "
        f"4-vCPU Kind node after fixed tenants ({fixed_mcpu}m); pods would sit "
        "Pending with FailedScheduling"
    )
