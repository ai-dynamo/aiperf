# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v1beta1 Dynamo manifest emission path (no cluster required).

Covers the new ``spec.components`` list + native ``corev1.PodTemplateSpec``
shape and verifies that the v1alpha1 path still works when explicitly
selected (regression guard for tests/operators pinned to alpha).
"""

from __future__ import annotations

import pytest
import yaml
from pytest import param

from tests.kubernetes.gpu.dynamo.helpers import (
    MAIN_CONTAINER_NAME,
    DynamoBackend,
    DynamoConfig,
    DynamoDeployer,
    DynamoMode,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


@pytest.fixture
def kubectl() -> KubectlClient:
    """Create a kubectl client (not used for real calls in these tests)."""
    return KubectlClient()


def _parse_manifest(deployer: DynamoDeployer) -> list[dict]:
    """Generate manifest and parse all YAML documents."""
    raw = deployer.generate_manifest()
    return list(yaml.safe_load_all(raw))


def _crd_doc(deployer: DynamoDeployer) -> dict:
    """Extract the DynamoGraphDeployment doc from a generated manifest."""
    docs = _parse_manifest(deployer)
    crd_docs = [d for d in docs if d.get("kind") == "DynamoGraphDeployment"]
    assert len(crd_docs) == 1, "Expected exactly one DynamoGraphDeployment doc"
    return crd_docs[0]


def _find_component(crd: dict, name: str) -> dict:
    """Look up a component by name in v1beta1 ``spec.components``."""
    matches = [c for c in crd["spec"]["components"] if c["name"] == name]
    assert len(matches) == 1, f"expected exactly one component named {name!r}"
    return matches[0]


class TestDynamoConfigDefaults:
    """The v1beta1 default flip itself."""

    def test_default_api_version_is_v1beta1(self) -> None:
        config = DynamoConfig()
        assert config.api_version == "v1beta1"

    def test_explicit_v1alpha1_preserved(self) -> None:
        config = DynamoConfig(api_version="v1alpha1")
        assert config.api_version == "v1alpha1"


class TestV1Beta1ManifestShape:
    """Structural assertions for the new ``spec.components`` shape."""

    def test_emits_v1beta1_api_version(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig())
        crd = _crd_doc(deployer)
        assert crd["apiVersion"] == "nvidia.com/v1beta1"
        assert crd["kind"] == "DynamoGraphDeployment"

    def test_namespace_doc_still_emitted(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig(namespace="my-ns"))
        docs = _parse_manifest(deployer)
        ns_docs = [d for d in docs if d.get("kind") == "Namespace"]
        assert len(ns_docs) == 1
        assert ns_docs[0]["metadata"]["name"] == "my-ns"

    def test_components_is_a_list(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig())
        crd = _crd_doc(deployer)
        assert isinstance(crd["spec"]["components"], list)
        # And no legacy v1alpha1 fields:
        assert "services" not in crd["spec"]

    def test_every_component_has_a_name(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig())
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            assert "name" in component
            assert isinstance(component["name"], str)
            assert component["name"]

    def test_inference_container_named_main(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig())
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            containers = component["podTemplate"]["spec"]["containers"]
            assert containers, "podTemplate.spec.containers must be non-empty"
            names = [c["name"] for c in containers]
            assert MAIN_CONTAINER_NAME in names
            # The first container is the one the operator merges defaults onto.
            assert containers[0]["name"] == MAIN_CONTAINER_NAME


class TestV1Beta1ComponentTypes:
    """Component-type emission across the three modes."""

    def test_aggregated_emits_frontend_and_worker(self, kubectl: KubectlClient) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig(mode=DynamoMode.AGGREGATED))
        crd = _crd_doc(deployer)
        types = [c["type"] for c in crd["spec"]["components"]]
        assert types == ["frontend", "worker"]

    def test_aggregated_router_emits_frontend_and_worker(
        self, kubectl: KubectlClient
    ) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER, router_mode="kv")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        types = [c["type"] for c in crd["spec"]["components"]]
        assert types == ["frontend", "worker"]

    def test_disaggregated_emits_frontend_decode_prefill(
        self, kubectl: KubectlClient
    ) -> None:
        deployer = DynamoDeployer(kubectl, DynamoConfig(mode=DynamoMode.DISAGGREGATED))
        crd = _crd_doc(deployer)
        types = [c["type"] for c in crd["spec"]["components"]]
        # Decode + prefill are first-class types in v1beta1; no subComponentType.
        assert "decode" in types
        assert "prefill" in types
        assert types[0] == "frontend"

    def test_v1beta1_has_no_sub_component_type_field(
        self, kubectl: KubectlClient
    ) -> None:
        """``subComponentType`` is removed in v1beta1 — only the label remains."""
        deployer = DynamoDeployer(kubectl, DynamoConfig(mode=DynamoMode.DISAGGREGATED))
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            assert "subComponentType" not in component


class TestV1Beta1PodTemplate:
    """Pod-template-level fields land in the right corev1 slots."""

    def test_runtime_class_on_pod_spec(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(runtime_class_name="nvidia")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            assert component["podTemplate"]["spec"]["runtimeClassName"] == "nvidia"

    def test_tolerations_on_pod_spec(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}]
        )
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            tolerations = component["podTemplate"]["spec"]["tolerations"]
            assert tolerations == [{"key": "nvidia.com/gpu", "operator": "Exists"}]

    def test_node_selector_on_pod_spec(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(node_selector={"nvidia.com/gpu.product": "H100"})
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            assert component["podTemplate"]["spec"]["nodeSelector"] == {
                "nvidia.com/gpu.product": "H100"
            }

    def test_image_pull_secrets_on_pod_spec(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(image_pull_secrets=["ngc-pull", "ghcr-pull"])
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        for component in crd["spec"]["components"]:
            assert component["podTemplate"]["spec"]["imagePullSecrets"] == [
                {"name": "ngc-pull"},
                {"name": "ghcr-pull"},
            ]

    def test_gpu_resource_on_main_container(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED, gpu_count=2)
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        worker = _find_component(crd, "VllmDecodeWorker")
        main = worker["podTemplate"]["spec"]["containers"][0]
        assert main["name"] == MAIN_CONTAINER_NAME
        assert main["resources"]["limits"] == {"nvidia.com/gpu": "2"}

    def test_gpu_count_zero_omits_resources(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED, gpu_count=0)
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        worker = _find_component(crd, "VllmDecodeWorker")
        main = worker["podTemplate"]["spec"]["containers"][0]
        assert "resources" not in main

    def test_component_type_label_stamped(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.DISAGGREGATED)
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)

        frontend = _find_component(crd, "Frontend")
        assert (
            frontend["podTemplate"]["metadata"]["labels"][
                "nvidia.com/dynamo-component-type"
            ]
            == "frontend"
        )

        decode = _find_component(crd, "VllmDecodeWorker")
        decode_labels = decode["podTemplate"]["metadata"]["labels"]
        assert decode_labels["nvidia.com/dynamo-component-type"] == "decode"
        assert decode_labels["nvidia.com/dynamo-sub-component-type"] == "decode"

        prefill = _find_component(crd, "VllmPrefillWorker")
        prefill_labels = prefill["podTemplate"]["metadata"]["labels"]
        assert prefill_labels["nvidia.com/dynamo-component-type"] == "prefill"
        assert prefill_labels["nvidia.com/dynamo-sub-component-type"] == "prefill"

    def test_worker_args_carry_model(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED, model_name="Qwen/Qwen3-8B")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        worker = _find_component(crd, "VllmDecodeWorker")
        main = worker["podTemplate"]["spec"]["containers"][0]
        assert "--model" in main["args"]
        assert "Qwen/Qwen3-8B" in main["args"]

    def test_frontend_router_mode_env(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER, router_mode="kv")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        frontend = _find_component(crd, "Frontend")
        envs = frontend["podTemplate"]["spec"]["containers"][0]["env"]
        router_env = next(e for e in envs if e["name"] == "DYN_ROUTER_MODE")
        assert router_env["value"] == "kv"

    def test_replicas_propagate(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            mode=DynamoMode.DISAGGREGATED,
            frontend_replicas=2,
            decode_replicas=3,
            prefill_replicas=4,
        )
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        assert _find_component(crd, "Frontend")["replicas"] == 2
        assert _find_component(crd, "VllmDecodeWorker")["replicas"] == 3
        assert _find_component(crd, "VllmPrefillWorker")["replicas"] == 4


class TestV1Beta1Modes:
    """All three modes round-trip through the v1beta1 emitter."""

    @pytest.mark.parametrize(
        "mode,expected_component_names",
        [
            param(
                DynamoMode.AGGREGATED,
                ["Frontend", "VllmDecodeWorker"],
                id="aggregated",
            ),
            param(
                DynamoMode.AGGREGATED_ROUTER,
                ["Frontend", "VllmDecodeWorker"],
                id="agg-router",
            ),
            param(
                DynamoMode.DISAGGREGATED,
                ["Frontend", "VllmDecodeWorker", "VllmPrefillWorker"],
                id="disaggregated",
            ),
        ],
    )  # fmt: skip
    def test_components_per_mode(
        self,
        kubectl: KubectlClient,
        mode: DynamoMode,
        expected_component_names: list[str],
    ) -> None:
        config = DynamoConfig(mode=mode)
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        names = [c["name"] for c in crd["spec"]["components"]]
        assert names == expected_component_names

    def test_single_gpu_disagg_preset_emits_v1beta1_no_gpu_request(
        self, kubectl: KubectlClient
    ) -> None:
        """``DynamoConfig.single_gpu_disagg()`` is the chaos suite's primary fixture.

        It defaults to v1beta1 (Phase 0a flip), ``gpu_count=0`` (so K8s does not
        block scheduling on single-GPU dev boxes), and ``runtimeClassName=nvidia``
        (so the NVIDIA runtime still mounts the GPU driver). Both prefill and
        decode workers must emit cleanly under those constraints.
        """
        config = DynamoConfig.single_gpu_disagg()
        # Sanity: the preset really is a disaggregated mode.
        assert config.mode.is_disaggregated

        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        assert crd["apiVersion"] == "nvidia.com/v1beta1"

        types_by_name = {c["name"]: c["type"] for c in crd["spec"]["components"]}
        assert types_by_name.get("VllmPrefillWorker") == "prefill"
        assert types_by_name.get("VllmDecodeWorker") == "decode"

        for worker_name in ("VllmPrefillWorker", "VllmDecodeWorker"):
            worker = _find_component(crd, worker_name)
            pod_spec = worker["podTemplate"]["spec"]
            main = pod_spec["containers"][0]
            assert main["name"] == MAIN_CONTAINER_NAME
            assert "resources" not in main, (
                f"{worker_name} should omit resources when gpu_count=0"
            )
            assert pod_spec["runtimeClassName"] == "nvidia"


class TestV1Alpha1RegressionGuard:
    """Explicit ``api_version="v1alpha1"`` still produces the legacy shape."""

    def test_alpha_emits_v1alpha1_api_version(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(api_version="v1alpha1")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        assert crd["apiVersion"] == "nvidia.com/v1alpha1"

    def test_alpha_emits_services_map(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(api_version="v1alpha1")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        services = crd["spec"]["services"]
        assert isinstance(services, dict)
        assert "Frontend" in services
        assert "components" not in crd["spec"]

    def test_alpha_uses_extra_pod_spec_main_container(
        self, kubectl: KubectlClient
    ) -> None:
        config = DynamoConfig(api_version="v1alpha1")
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        frontend = crd["spec"]["services"]["Frontend"]
        assert "extraPodSpec" in frontend
        assert "mainContainer" in frontend["extraPodSpec"]

    @pytest.mark.parametrize(
        "backend",
        [
            param(DynamoBackend.VLLM, id="vllm"),
            param(DynamoBackend.TRTLLM, id="trtllm"),
            param(DynamoBackend.SGLANG, id="sglang"),
        ],
    )  # fmt: skip
    def test_alpha_disaggregated_three_backends(
        self, kubectl: KubectlClient, backend: DynamoBackend
    ) -> None:
        config = DynamoConfig(
            api_version="v1alpha1",
            backend=backend,
            mode=DynamoMode.DISAGGREGATED,
        )
        deployer = DynamoDeployer(kubectl, config)
        crd = _crd_doc(deployer)
        services = crd["spec"]["services"]
        prefix = {
            DynamoBackend.VLLM: "Vllm",
            DynamoBackend.TRTLLM: "Trtllm",
            DynamoBackend.SGLANG: "Sglang",
        }[backend]
        assert f"{prefix}DecodeWorker" in services
        assert f"{prefix}PrefillWorker" in services
