# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RBAC = ROOT / "deploy" / "aiperf-k8s-operator"


@pytest.mark.parametrize(
    "manifest",
    [
        RBAC / "rbac" / "operator-clusterrole.yaml",
        RBAC / "helm" / "aiperf-k8s-operator" / "templates" / "clusterrole.yaml",
    ],
)
def test_operator_secret_access_is_limited_to_reconciled_capability_lifecycle(
    manifest: Path,
) -> None:
    source = manifest.read_text().replace("{{ .Release.Name }}", "aiperf-operator")
    source = "\n".join(
        line for line in source.splitlines() if not line.startswith("{{-")
    )
    cluster_role = yaml.safe_load(source)
    assert cluster_role["kind"] == "ClusterRole"
    secret_rules = [
        rule
        for rule in cluster_role["rules"]
        if rule["apiGroups"] == [""] and "secrets" in rule["resources"]
    ]

    assert secret_rules == [
        {
            "apiGroups": [""],
            "resources": ["secrets"],
            "verbs": ["delete", "get"],
        }
    ]
    assert cluster_role["rules"] == [
        {
            "apiGroups": ["aiperf.nvidia.com"],
            "resources": ["aiperfjobs"],
            "verbs": ["get", "list", "watch", "patch"],
        },
        {
            "apiGroups": ["aiperf.nvidia.com"],
            "resources": ["aiperfjobs/status"],
            "verbs": ["patch"],
        },
        {
            "apiGroups": ["jobset.x-k8s.io"],
            "resources": ["jobsets"],
            "verbs": ["create", "delete", "get", "list", "watch"],
        },
        {
            "apiGroups": [""],
            "resources": ["secrets"],
            "verbs": ["delete", "get"],
        },
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "verbs": ["create", "delete", "get"],
        },
        {
            "apiGroups": [""],
            "resources": ["serviceaccounts"],
            "verbs": ["create", "delete", "get"],
        },
        {
            "apiGroups": ["rbac.authorization.k8s.io"],
            "resources": ["roles", "rolebindings"],
            "verbs": ["create", "delete", "get"],
        },
    ]
