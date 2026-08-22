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
def test_operator_secret_access_is_cluster_scoped_get_only(manifest: Path) -> None:
    source = manifest.read_text().replace("{{ .Release.Name }}", "aiperf-operator")
    cluster_role = yaml.safe_load(source)
    assert cluster_role["kind"] == "ClusterRole"
    secret_rules = [
        rule
        for rule in cluster_role["rules"]
        if rule["apiGroups"] == [""] and "secrets" in rule["resources"]
    ]

    assert secret_rules == [
        {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get"]}
    ]
