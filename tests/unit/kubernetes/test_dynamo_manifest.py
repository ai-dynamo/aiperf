# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Dynamo manifest generation (no cluster required)."""

from __future__ import annotations

import pytest
import yaml
from pytest import param

from tests.kubernetes.gpu.dynamo.helpers import (
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


class TestDynamoManifestAggregated:
    """Test aggregated mode manifest generation."""

    def test_generates_namespace_and_crd(self, kubectl: KubectlClient) -> None:
        """Manifest should produce a Namespace + DynamoGraphDeployment."""
        config = DynamoConfig()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        assert len(docs) == 2
        assert docs[0]["kind"] == "Namespace"
        assert docs[1]["kind"] == "DynamoGraphDeployment"
        assert docs[1]["apiVersion"] == "nvidia.com/v1alpha1"

    def test_namespace_matches_config(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(namespace="my-ns")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        assert docs[0]["metadata"]["name"] == "my-ns"
        assert docs[1]["metadata"]["namespace"] == "my-ns"

    def test_aggregated_has_frontend_and_decode_worker(
        self, kubectl: KubectlClient
    ) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert "Frontend" in services
        assert "VllmDecodeWorker" in services
        assert "VllmPrefillWorker" not in services

    def test_default_image_from_backend(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        frontend = docs[1]["spec"]["services"]["Frontend"]
        assert (
            frontend["extraPodSpec"]["mainContainer"]["image"]
            == DynamoBackend.VLLM.default_image
        )

    def test_explicit_image_overrides_backend(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(image="custom:latest")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        frontend = docs[1]["spec"]["services"]["Frontend"]
        assert frontend["extraPodSpec"]["mainContainer"]["image"] == "custom:latest"

    def test_worker_has_model_args(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(model_name="Qwen/Qwen3-8B")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert "--model" in args
        assert "Qwen/Qwen3-8B" in args

    def test_worker_has_gpu_resources(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(gpu_count=2)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        assert worker["resources"]["limits"]["gpu"] == "2"

    def test_worker_has_probes(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        container = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]
        assert "startupProbe" in container
        assert "livenessProbe" in container
        assert container["startupProbe"]["httpGet"]["path"] == "/live"
        assert container["startupProbe"]["httpGet"]["port"] == 9090

    def test_hf_token_secret(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(hf_token_secret="hf-token-secret")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        assert worker["envFromSecret"] == "hf-token-secret"

    def test_replicas(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(frontend_replicas=2, decode_replicas=4)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert services["Frontend"]["replicas"] == 2
        assert services["VllmDecodeWorker"]["replicas"] == 4


class TestDynamoManifestDisaggregated:
    """Test disaggregated mode manifest generation."""

    def test_disaggregated_has_prefill_and_decode(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.DISAGGREGATED)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert "Frontend" in services
        assert "VllmDecodeWorker" in services
        assert "VllmPrefillWorker" in services

    def test_decode_worker_has_decode_flag(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.DISAGGREGATED)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert "--is-decode-worker" in args
        assert worker["subComponentType"] == "decode"

    def test_prefill_worker_has_prefill_flag(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.DISAGGREGATED)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmPrefillWorker"]
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert "--is-prefill-worker" in args
        assert worker["subComponentType"] == "prefill"

    def test_connectors_on_prefill_worker(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            mode=DynamoMode.DISAGGREGATED,
            connectors=["kvbm", "nixl"],
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmPrefillWorker"]
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert "--connector" in args
        idx = args.index("--connector")
        assert args[idx + 1] == "kvbm"
        assert args[idx + 2] == "nixl"

    def test_kvbm_cpu_cache_on_prefill(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            mode=DynamoMode.DISAGGREGATED,
            kvbm_cpu_cache_gb=50,
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmPrefillWorker"]
        envs = worker["extraPodSpec"]["mainContainer"]["env"]
        kvbm_env = next(e for e in envs if e["name"] == "DYN_KVBM_CPU_CACHE_GB")
        assert kvbm_env["value"] == "50"

    def test_kvbm_enables_metrics(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            mode=DynamoMode.DISAGGREGATED,
            kvbm_cpu_cache_gb=10,
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmPrefillWorker"]
        envs = worker["extraPodSpec"]["mainContainer"]["env"]
        metrics_env = next(e for e in envs if e["name"] == "DYN_KVBM_METRICS")
        assert metrics_env["value"] == "true"

    def test_prefill_replicas(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            mode=DynamoMode.DISAGGREGATED,
            decode_replicas=2,
            prefill_replicas=3,
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert services["VllmDecodeWorker"]["replicas"] == 2
        assert services["VllmPrefillWorker"]["replicas"] == 3


class TestDynamoManifestOptions:
    """Test optional configuration fields."""

    def test_router_mode(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(router_mode="kv")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        frontend = docs[1]["spec"]["services"]["Frontend"]
        envs = frontend["envs"]
        router_env = next(e for e in envs if e["name"] == "DYN_ROUTER_MODE")
        assert router_env["value"] == "kv"

    def test_pvc_mount(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(pvc_name="models-cache")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        spec = docs[1]["spec"]
        assert {"name": "models-cache"} in spec["pvcs"]

        frontend = spec["services"]["Frontend"]
        assert frontend["volumeMounts"][0]["name"] == "models-cache"
        assert frontend["volumeMounts"][0]["mountPoint"] == "/models"

    def test_max_model_len(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(max_model_len=32000)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        args = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--max-model-len" in args
        assert "32000" in args

    def test_tensor_parallel_size(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(tensor_parallel_size=4)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        args = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--tensor-parallel-size" in args
        assert "4" in args

    def test_enforce_eager(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(enforce_eager=True)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        args = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--enforce-eager" in args

    def test_extra_envs(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            extra_envs=[{"name": "HF_HOME", "value": "/models"}],
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        spec = docs[1]["spec"]
        assert {"name": "HF_HOME", "value": "/models"} in spec["envs"]

    def test_extra_worker_args(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(extra_worker_args=["--quantization", "awq"])
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        args = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--quantization" in args
        assert "awq" in args

    @pytest.mark.parametrize(
        "mode,expected_name",
        [
            param(DynamoMode.AGGREGATED, "dynamo-agg", id="agg"),
            param(DynamoMode.AGGREGATED_ROUTER, "dynamo-agg-router", id="agg-router"),
            param(DynamoMode.DISAGGREGATED, "dynamo-disagg", id="disagg"),
        ],
    )  # fmt: skip
    def test_deployment_name(
        self, kubectl: KubectlClient, mode: DynamoMode, expected_name: str
    ) -> None:
        config = DynamoConfig(mode=mode)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        assert docs[1]["metadata"]["name"] == expected_name

    def test_gpu_memory_utilization(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(gpu_memory_utilization=0.5)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        args = docs[1]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--gpu-memory-utilization" in args
        assert "0.5" in args


class TestDynamoManifestAggregatedRouter:
    """Test aggregated-router mode (single GPU with KV-aware routing)."""

    def test_has_frontend_and_worker_no_prefill(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER, router_mode="kv")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert "Frontend" in services
        assert "VllmDecodeWorker" in services
        assert "VllmPrefillWorker" not in services

    def test_worker_has_no_decode_flag(self, kubectl: KubectlClient) -> None:
        """Aggregated-router workers are full agg workers, not decode-only."""
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert "--is-decode-worker" not in args
        assert "--is-prefill-worker" not in args
        assert "subComponentType" not in worker

    def test_router_mode_env(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER, router_mode="kv")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        frontend = docs[1]["spec"]["services"]["Frontend"]
        router_env = next(e for e in frontend["envs"] if e["name"] == "DYN_ROUTER_MODE")
        assert router_env["value"] == "kv"


class TestDynamoSingleGpuDisagg:
    """Test single_gpu_disagg factory for shared-GPU local testing."""

    def test_factory_defaults(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig.single_gpu_disagg()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        assert config.mode == DynamoMode.DISAGGREGATED_1GPU
        assert config.gpu_count == 0
        assert config.enforce_eager is True
        assert config.gpu_memory_utilization == 0.12
        assert config.max_model_len == 4096
        assert config.runtime_class_name == "nvidia"

        # Workers should have NO gpu resources block
        for svc_name in ("VllmDecodeWorker", "VllmPrefillWorker"):
            worker = docs[1]["spec"]["services"][svc_name]
            assert "resources" not in worker

    def test_factory_has_runtime_class(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig.single_gpu_disagg()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        for svc_name in ("Frontend", "VllmDecodeWorker", "VllmPrefillWorker"):
            svc = docs[1]["spec"]["services"][svc_name]
            assert svc["extraPodSpec"]["runtimeClassName"] == "nvidia"

    def test_factory_workers_have_low_memory_utilization(
        self, kubectl: KubectlClient
    ) -> None:
        config = DynamoConfig.single_gpu_disagg()
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        for svc_name in ("VllmDecodeWorker", "VllmPrefillWorker"):
            args = docs[1]["spec"]["services"][svc_name]["extraPodSpec"][
                "mainContainer"
            ]["args"]
            assert "--gpu-memory-utilization" in args
            assert "0.12" in args
            assert "--enforce-eager" in args
            assert "--max-model-len" in args
            assert "4096" in args

    def test_factory_overrides(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig.single_gpu_disagg(
            model_name="meta-llama/Llama-3.2-1B",
            gpu_memory_utilization=0.4,
            namespace="my-test",
        )
        assert config.model_name == "meta-llama/Llama-3.2-1B"
        assert config.gpu_memory_utilization == 0.4
        assert config.namespace == "my-test"
        assert config.mode == DynamoMode.DISAGGREGATED_1GPU

    def test_factory_auto_defaults_kvbm(self) -> None:
        """single_gpu_disagg with kvbm connector should auto-set kvbm_cpu_cache_gb."""
        config = DynamoConfig.single_gpu_disagg(connectors=["kvbm"])
        assert config.kvbm_cpu_cache_gb == 1
        assert config.connectors == ["kvbm"]

    def test_factory_kvbm_explicit_overrides_auto(self) -> None:
        """Explicit kvbm_cpu_cache_gb should not be overwritten by auto-default."""
        config = DynamoConfig.single_gpu_disagg(
            connectors=["kvbm"], kvbm_cpu_cache_gb=4
        )
        assert config.kvbm_cpu_cache_gb == 4

    def test_gpu_count_zero_omits_resources(self, kubectl: KubectlClient) -> None:
        """gpu_count=0 should produce no resources block at all."""
        config = DynamoConfig(gpu_count=0)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["VllmDecodeWorker"]
        assert "resources" not in worker

    def test_runtime_class_name(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(runtime_class_name="nvidia")
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        for svc_name in ("Frontend", "VllmDecodeWorker"):
            svc = docs[1]["spec"]["services"][svc_name]
            assert svc["extraPodSpec"]["runtimeClassName"] == "nvidia"


class TestDynamoBackend:
    """Test backend selection (vLLM vs TRT-LLM)."""

    def test_vllm_default_image(self) -> None:
        assert "vllm-runtime" in DynamoBackend.VLLM.default_image

    def test_trtllm_default_image(self) -> None:
        assert "trtllm-runtime" in DynamoBackend.TRTLLM.default_image

    def test_vllm_worker_command(self) -> None:
        assert DynamoBackend.VLLM.worker_command == ["python3", "-m", "dynamo.vllm"]

    def test_trtllm_worker_command(self) -> None:
        assert DynamoBackend.TRTLLM.worker_command == ["python3", "-m", "dynamo.trtllm"]

    def test_vllm_working_dir(self) -> None:
        assert "backends/vllm" in DynamoBackend.VLLM.worker_working_dir

    def test_trtllm_working_dir(self) -> None:
        assert "backends/trtllm" in DynamoBackend.TRTLLM.worker_working_dir

    def test_trtllm_service_keys(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(
            backend=DynamoBackend.TRTLLM, mode=DynamoMode.DISAGGREGATED
        )
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        services = docs[1]["spec"]["services"]
        assert "TrtllmDecodeWorker" in services
        assert "TrtllmPrefillWorker" in services
        assert "VllmDecodeWorker" not in services

    def test_trtllm_worker_command_in_manifest(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(backend=DynamoBackend.TRTLLM)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["TrtllmDecodeWorker"]
        container = worker["extraPodSpec"]["mainContainer"]
        assert container["command"] == ["python3", "-m", "dynamo.trtllm"]
        assert "backends/trtllm" in container["workingDir"]

    def test_trtllm_default_image_in_manifest(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(backend=DynamoBackend.TRTLLM)
        deployer = DynamoDeployer(kubectl, config)
        docs = _parse_manifest(deployer)

        worker = docs[1]["spec"]["services"]["TrtllmDecodeWorker"]
        assert "trtllm-runtime" in worker["extraPodSpec"]["mainContainer"]["image"]

    def test_effective_image_uses_backend_default(self) -> None:
        config = DynamoConfig(backend=DynamoBackend.TRTLLM)
        assert config.effective_image == DynamoBackend.TRTLLM.default_image

    def test_effective_image_explicit_overrides(self) -> None:
        config = DynamoConfig(backend=DynamoBackend.TRTLLM, image="custom:v1")
        assert config.effective_image == "custom:v1"


class TestDynamoEndpointUrl:
    """Test endpoint URL generation."""

    def test_aggregated_endpoint(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED, namespace="test-ns")
        deployer = DynamoDeployer(kubectl, config)
        url = deployer.get_endpoint_url()
        assert url == "http://dynamo-agg-frontend.test-ns.svc.cluster.local:8000/v1"

    def test_aggregated_router_endpoint(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.AGGREGATED_ROUTER, namespace="test-ns")
        deployer = DynamoDeployer(kubectl, config)
        url = deployer.get_endpoint_url()
        assert (
            url == "http://dynamo-agg-router-frontend.test-ns.svc.cluster.local:8000/v1"
        )

    def test_disaggregated_endpoint(self, kubectl: KubectlClient) -> None:
        config = DynamoConfig(mode=DynamoMode.DISAGGREGATED, namespace="prod")
        deployer = DynamoDeployer(kubectl, config)
        url = deployer.get_endpoint_url()
        assert url == "http://dynamo-disagg-frontend.prod.svc.cluster.local:8000/v1"
