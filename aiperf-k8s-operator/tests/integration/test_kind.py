# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from typing import Any

import pytest

LIVE_CLUSTER_SELECTED = bool(
    os.environ.get("KUBECONFIG") or os.environ.get("AIPERF_K8S_INTEGRATION")
)


def kubectl(*args: str) -> dict[str, Any]:
    """Run one read-only live-cluster query and decode its JSON document."""
    completed = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(
    not LIVE_CLUSTER_SELECTED,
    reason="requires a configured Kubernetes target via KUBECONFIG or AIPERF_K8S_INTEGRATION",
)
def test_chart_installation_is_live_and_exposes_only_the_shipped_crd() -> None:
    """Prove the installed chart has a ready operator and its one supported CRD."""
    crds = kubectl("get", "customresourcedefinitions.apiextensions.k8s.io")
    aiperf_crds = [
        item for item in crds["items"] if item["spec"]["group"] == "aiperf.nvidia.com"
    ]
    assert [item["metadata"]["name"] for item in aiperf_crds] == [
        "aiperfjobs.aiperf.nvidia.com"
    ]
    assert any(
        condition.get("type") == "Established" and condition.get("status") == "True"
        for condition in aiperf_crds[0]["status"]["conditions"]
    )

    deployments = kubectl(
        "get",
        "deployments.apps",
        "--all-namespaces",
        "--selector",
        "app.kubernetes.io/name=aiperf-k8s-operator",
    )
    assert len(deployments["items"]) == 1
    deployment = deployments["items"][0]
    assert deployment["status"].get("availableReplicas") == 1
