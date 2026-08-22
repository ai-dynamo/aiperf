# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CHART = ROOT / "deploy" / "aiperf-k8s-operator" / "helm" / "aiperf-k8s-operator"


def render_chart(*values: str) -> list[dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "operator",
            str(CHART),
            "--namespace",
            "control-plane",
            "--include-crds",
            *(argument for value in values for argument in ("--set", value)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [resource for resource in yaml.safe_load_all(result.stdout) if resource]


def one(resources: list[dict], kind: str, name: str) -> dict:
    return next(
        resource
        for resource in resources
        if resource.get("kind") == kind
        and resource.get("metadata", {}).get("name") == name
    )


def test_fresh_chart_renders_jobset_prerequisite_and_durable_results_service() -> None:
    resources = render_chart()
    crd_names = {
        resource["metadata"]["name"]
        for resource in resources
        if resource.get("kind") == "CustomResourceDefinition"
    }
    assert "jobsets.jobset.x-k8s.io" in crd_names

    deployment = one(resources, "Deployment", "operator")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    environment = {entry["name"]: entry["value"] for entry in container["env"]}
    assert environment == {
        "AIPERF_K8S_OPERATOR_ARTIFACT_ROOT": "/var/lib/aiperf/results",
        "AIPERF_K8S_OPERATOR_API_PORT": "8080",
        "AIPERF_K8S_OPERATOR_RESULTS_UPLOAD_BASE_URL": "http://aiperf-k8s-operator.control-plane.svc:8080",
    }
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/name": "aiperf-k8s-operator",
        "app.kubernetes.io/instance": "operator",
    }
    assert deployment["spec"]["template"]["metadata"]["labels"] == {
        "app.kubernetes.io/name": "aiperf-k8s-operator",
        "app.kubernetes.io/instance": "operator",
    }
    assert container["ports"] == [{"name": "http", "containerPort": 8080}]
    assert container["readinessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": "http",
    }
    assert container["livenessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": "http",
    }
    assert container["volumeMounts"] == [
        {"name": "results", "mountPath": "/var/lib/aiperf/results"}
    ]
    assert pod["volumes"] == [
        {"name": "results", "persistentVolumeClaim": {"claimName": "operator-results"}}
    ]

    service = one(resources, "Service", "aiperf-k8s-operator")
    assert service["spec"]["selector"] == {
        "app.kubernetes.io/name": "aiperf-k8s-operator",
        "app.kubernetes.io/instance": "operator",
    }
    assert service["spec"]["ports"] == [
        {"name": "http", "port": 8080, "targetPort": "http"}
    ]
    claim = one(resources, "PersistentVolumeClaim", "operator-results")
    assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]

    aiperfjob_crd = one(
        resources,
        "CustomResourceDefinition",
        "aiperfjobs.aiperf.nvidia.com",
    )
    spec_schema = aiperfjob_crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
        "properties"
    ]["spec"]
    assert spec_schema["x-kubernetes-validations"] == [
        {"rule": "self == oldSelf", "message": "AIPerfJob spec is immutable"}
    ]
    status_schema = aiperfjob_crd["spec"]["versions"][0]["schema"][
        "openAPIV3Schema"
    ]["properties"]["status"]
    assert status_schema["additionalProperties"] is False
    assert status_schema["required"] == ["phase", "runId", "jobSet"]
    assert status_schema["properties"]["phase"]["enum"] == [
        "Pending",
        "PublishingResults",
        "Completed",
        "Failed",
    ]
    assert status_schema["x-kubernetes-validations"] == [
        {
            "rule": "!has(oldSelf.runId) || self.runId == oldSelf.runId",
            "message": "status runId is immutable",
        },
        {
            "rule": "!has(oldSelf.jobSet) || self.jobSet == oldSelf.jobSet",
            "message": "status jobSet is immutable",
        },
        {
            "rule": "!has(oldSelf.resultsReady) || !oldSelf.resultsReady || (has(self.resultsReady) && self.resultsReady)",
            "message": "resultsReady cannot return to false",
        },
        {
            "rule": "(!has(self.resultsReady) || !self.resultsReady || self.phase == 'Completed') && (self.phase != 'Completed' || (has(self.resultsReady) && self.resultsReady))",
            "message": "Completed and resultsReady must be published together",
        },
        {
            "rule": "!has(oldSelf.phase) || self.phase == oldSelf.phase || (oldSelf.phase == 'Pending' && self.phase in ['PublishingResults', 'Failed']) || (oldSelf.phase == 'PublishingResults' && self.phase in ['Completed', 'Failed'])",
            "message": "status phase transition is not allowed",
        },
    ]


def test_operator_rbac_can_provision_only_required_namespaced_workload_identity() -> (
    None
):
    role = one(render_chart(), "ClusterRole", "operator")
    selected = {
        tuple(rule["resources"]): tuple(rule["verbs"])
        for rule in role["rules"]
        if rule["apiGroups"] in ([""], ["rbac.authorization.k8s.io"])
    }
    assert selected[("serviceaccounts",)] == ("create", "delete", "get")
    assert selected[("roles", "rolebindings")] == ("create", "delete", "get")
    assert selected[("secrets",)] == ("create", "delete", "get")
    assert selected[("configmaps",)] == ("create", "delete", "get")


def test_admin_managed_operator_identity_omits_chart_owned_rbac() -> None:
    resources = render_chart("rbac.create=false", "serviceAccount.create=false")
    assert not any(
        resource.get("kind") in {"ClusterRole", "ClusterRoleBinding"}
        and resource.get("metadata", {}).get("name") == "operator"
        for resource in resources
    )
    assert not any(
        resource.get("kind") == "ServiceAccount"
        and resource.get("metadata", {}).get("name") == "aiperf-k8s-operator"
        for resource in resources
    )
