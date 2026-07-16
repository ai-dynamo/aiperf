# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v1beta1 Dynamo manifest emission path (no cluster required).

Covers the new ``spec.components`` list + native ``corev1.PodTemplateSpec``
shape and verifies that the v1alpha1 path still works when explicitly
selected. The default ``api_version`` is now ``v1alpha1`` (matches the
shipped helm chart), so each test here either pins ``api_version="v1beta1"``
via the local ``DynamoConfig`` shim or constructs the real config directly
with an explicit kwarg.
"""

from __future__ import annotations

import pytest
import yaml
from pytest import param

from tests.kubernetes.chaos_dynamo.d7_status_helpers import (
    dgd_state_diagnostic_from_status_text,
    dgd_state_from_status_text,
    mentions_any,
    minimal_v1alpha1_frontend_dgd_manifest,
    wait_for_namespace_event_terms,
)
from tests.kubernetes.chaos_dynamo.frontend_request_helpers import (
    append_sse_data_lines,
    append_sse_events,
    chat_completion_url,
    chat_payload,
)
from tests.kubernetes.chaos_dynamo.metrics_helpers import metric_delta
from tests.kubernetes.chaos_dynamo.rbac_helpers import (
    RbacOwner,
    rbac_revoke_target,
    rbac_rule_grants,
)
from tests.kubernetes.chaos_dynamo.test_chaos_d3xx_nixl_kvbm import (
    _nixl_route_skip_reason,
)
from tests.kubernetes.chaos_dynamo.test_chaos_d8xx_store_discovery import (
    _d802_static_skip_reason,
    _find_decode_component,
)
from tests.kubernetes.gpu.conftest import GPUTestSettings
from tests.kubernetes.gpu.dynamo import helpers as _dynamo_helpers
from tests.kubernetes.gpu.dynamo.conftest import dynamo_config as _dynamo_config_fixture
from tests.kubernetes.gpu.dynamo.helpers import (
    MAIN_CONTAINER_NAME,
    DynamoBackend,
    DynamoDeployer,
    DynamoMode,
    is_dynamo_webhook_warmup_error,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


def DynamoConfig(**kwargs: object) -> _dynamo_helpers.DynamoConfig:
    """Shim: construct a ``DynamoConfig`` pinned to ``api_version="v1beta1"``.

    These tests assert the v1beta1 manifest shape; the dataclass default
    is ``v1alpha1`` (matches the shipped helm chart) so we override unless
    an individual test passes its own value.
    """
    kwargs.setdefault("api_version", "v1beta1")
    return _dynamo_helpers.DynamoConfig(**kwargs)  # type: ignore[arg-type]


def _single_gpu_disagg_beta(**overrides: object) -> _dynamo_helpers.DynamoConfig:
    """``DynamoConfig.single_gpu_disagg`` with the v1beta1 manifest shape."""
    overrides.setdefault("api_version", "v1beta1")
    return _dynamo_helpers.DynamoConfig.single_gpu_disagg(**overrides)


DynamoConfig.single_gpu_disagg = _single_gpu_disagg_beta  # type: ignore[attr-defined]


@pytest.fixture
def kubectl() -> KubectlClient:
    """Create a kubectl client (not used for real calls in these tests)."""
    return KubectlClient()


class TestDynamoMetricHelpers:
    """Shared metric helpers preserve Dynamo chaos counter semantics."""

    def test_metric_delta_defaults_missing_values_to_zero(self) -> None:
        assert metric_delta({"requests": 3.0}, {}, "errors") == 0.0
        assert metric_delta({}, {"errors": 2.0}, "errors") == -2.0

    def test_metric_delta_returns_after_minus_before(self) -> None:
        assert metric_delta({"requests": 9.5}, {"requests": 4.0}, "requests") == 5.5

    def test_metric_delta_can_floor_negative_values_at_zero(self) -> None:
        assert metric_delta({"requests": 1.0}, {"requests": 4.0}, "requests") == -3.0
        assert (
            metric_delta(
                {"requests": 1.0}, {"requests": 4.0}, "requests", floor_at_zero=True
            )
            == 0.0
        )


class TestDynamoFrontendRequestHelpers:
    """Shared frontend request helpers preserve low-level HTTP/SSE semantics."""

    def test_chat_completion_url_uses_alias_path_by_default(self) -> None:
        assert chat_completion_url("http://frontend.example/") == (
            "http://frontend.example/chat/completions"
        )

    def test_chat_completion_url_can_include_v1_prefix(self) -> None:
        assert chat_completion_url("http://frontend.example/v1", include_v1=True) == (
            "http://frontend.example/v1/v1/chat/completions"
        )

    def test_chat_payload_omits_temperature_none_and_preserves_extra(self) -> None:
        extra: dict[str, object] = {"stream_options": {"include_usage": True}}

        payload = chat_payload(
            "hello",
            model="chaos-model",
            stream=True,
            max_tokens=16,
            temperature=None,
            extra=extra,
        )

        assert payload == {
            "model": "chaos-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 16,
            "stream_options": {"include_usage": True},
        }
        assert extra == {"stream_options": {"include_usage": True}}

    def test_chat_payload_extra_overrides_base_fields_without_mutating_extra(
        self,
    ) -> None:
        extra: dict[str, object] = {"model": "override", "metadata": {"case": "D232"}}

        payload = chat_payload("hello", extra=extra)

        assert payload["model"] == "override"
        assert payload["temperature"] == 0.0
        assert extra == {"model": "override", "metadata": {"case": "D232"}}

    def test_append_sse_data_lines_keeps_incomplete_suffix(self) -> None:
        payloads: list[str] = []

        suffix = append_sse_data_lines(
            "data: one\r\nevent: ignored\ndata: tw", payloads
        )
        suffix = append_sse_data_lines(suffix + "o\n", payloads)

        assert payloads == ["one", "two"]
        assert suffix == ""

    def test_append_sse_events_keeps_partial_event_until_blank_line(self) -> None:
        payloads: list[str] = []

        suffix = append_sse_events("data: one\n\ndata: tw", payloads)
        suffix = append_sse_events(suffix + "o\r\n\r\ndata: three", payloads)

        assert payloads == ["one", "two"]
        assert suffix == "data: three"


class TestDynamoD7StatusHelpers:
    """Shared D7 helpers preserve DGD status and manifest semantics."""

    @pytest.mark.parametrize(
        "status_text,expected",
        [
            param('{"state":"failed","message":"boom"}', "failed", id="state"),
            param('{"message":"boom"}', "", id="missing-state"),
            param("not-json", "", id="unparsable"),
            param("", "", id="empty"),
        ],
    )  # fmt: skip
    def test_dgd_state_from_status_text_extracts_state_or_empty(
        self, status_text: str, expected: str
    ) -> None:
        assert dgd_state_from_status_text(status_text) == expected

    @pytest.mark.parametrize(
        "status_text,expected",
        [
            param('{"state":"failed","message":"boom"}', "failed", id="state"),
            param('{"message":"boom"}', "", id="missing-state"),
            param("not-json", "<unparsable>", id="unparsable"),
            param("", "<empty>", id="empty"),
        ],
    )  # fmt: skip
    def test_dgd_state_diagnostic_from_status_text_names_empty_and_unparsable(
        self, status_text: str, expected: str
    ) -> None:
        assert dgd_state_diagnostic_from_status_text(status_text) == expected

    def test_mentions_any_matches_case_insensitively(self) -> None:
        assert mentions_any(
            "FailedScheduling: Node Selector mismatch", ("node selector",)
        )
        assert not mentions_any("FailedScheduling: Node Selector mismatch", ("quota",))

    @pytest.mark.asyncio
    async def test_wait_for_namespace_event_terms_returns_only_matching_events(
        self,
    ) -> None:
        class FakeKubectl:
            async def run(self, *args: str, check: bool) -> object:
                assert args == ("get", "events", "-n", "d7-ns", "-o", "json")
                assert check is False
                return type(
                    "Result",
                    (),
                    {
                        "returncode": 0,
                        "stdout": (
                            '{"items":[{"reason":"FailedScheduling",'
                            '"message":"node selector mismatch"}]}'
                        ),
                    },
                )()

        events = await wait_for_namespace_event_terms(
            FakeKubectl(),
            namespace="d7-ns",
            needles=("node selector",),
            timeout_s=1.0,
            poll_interval_s=0.01,
        )

        assert events == "FailedScheduling: node selector mismatch"

    def test_minimal_v1alpha1_frontend_dgd_manifest_preserves_extra_pod_spec(
        self,
    ) -> None:
        extra_pod_spec = {
            "nodeSelector": {"aiperf.nvidia.com/missing": "true"},
            "mainContainer": {"image": "busybox:1.36"},
        }

        manifest = yaml.safe_load(
            minimal_v1alpha1_frontend_dgd_manifest(
                "d7-test",
                "d7-namespace",
                extra_pod_spec=extra_pod_spec,
            )
        )

        assert manifest["apiVersion"] == "nvidia.com/v1alpha1"
        assert manifest["kind"] == "DynamoGraphDeployment"
        assert manifest["metadata"] == {"name": "d7-test", "namespace": "d7-namespace"}
        frontend = manifest["spec"]["services"]["Frontend"]
        assert frontend["componentType"] == "frontend"
        assert frontend["replicas"] == 1
        assert frontend["extraPodSpec"] == extra_pod_spec


class TestDynamoRbacHelpers:
    """Shared RBAC helper primitives preserve Dynamo chaos target semantics."""

    def test_owner_label_formats_cluster_and_namespaced_roles(self) -> None:
        assert RbacOwner(
            scope="clusterrole", name="dynamo-operator", namespace=None
        ).label == ("clusterrole/dynamo-operator")
        assert RbacOwner(
            scope="role", name="dynamo-writer", namespace="dynamo-system"
        ).label == ("role/dynamo-system/dynamo-writer")

    def test_revoke_target_omits_namespace_for_clusterrole(self) -> None:
        owner = RbacOwner(scope="clusterrole", name="dynamo-operator", namespace=None)

        assert rbac_revoke_target(owner) == {
            "scope": "clusterrole",
            "name": "dynamo-operator",
        }

    def test_revoke_target_includes_namespace_for_role(self) -> None:
        owner = RbacOwner(scope="role", name="dynamo-writer", namespace="dynamo-system")

        assert rbac_revoke_target(owner) == {
            "scope": "role",
            "name": "dynamo-writer",
            "ns": "dynamo-system",
        }

    def test_rule_grants_rejects_wildcards_when_exact_match_required(self) -> None:
        rules = [
            {"apiGroups": ["*"], "resources": ["deployments"], "verbs": ["create"]},
            {"apiGroups": ["apps"], "resources": ["*"], "verbs": ["create"]},
            {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["*"]},
        ]

        assert not rbac_rule_grants(
            rules,
            api_group="apps",
            resource="deployments",
            verb="create",
            reject_wildcards=True,
        )

    def test_rule_grants_accepts_wildcards_when_allowed(self) -> None:
        rules = [{"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["*"]}]

        assert rbac_rule_grants(
            rules,
            api_group="apps",
            resource="deployments",
            verb="create",
            reject_wildcards=False,
        )


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


class TestDynamoD301Preconditions:
    """D301 static preconditions avoid expensive cluster setup for invalid topology."""

    def test_default_single_gpu_disagg_skips_without_toxiproxy_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIPERF_DYNAMO_NIXL_CHAOS", raising=False)
        config = _dynamo_helpers.DynamoConfig.single_gpu_disagg()

        reason = _nixl_route_skip_reason(config)

        assert reason is not None
        assert "VLLM_NIXL_SIDE_CHANNEL_HOST" in reason
        assert "toxiproxy.chaos-toxiproxy.svc" in reason

    def test_opt_in_allows_external_topology_assertions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DYNAMO_NIXL_CHAOS", "1")
        config = _dynamo_helpers.DynamoConfig.single_gpu_disagg()

        assert _nixl_route_skip_reason(config) is None

    def test_toxiproxy_route_allows_runtime_cluster_assertions(self) -> None:
        config = _dynamo_helpers.DynamoConfig.single_gpu_disagg(
            extra_envs=[
                {
                    "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
                    "value": "toxiproxy.chaos-toxiproxy.svc",
                }
            ]
        )

        assert _nixl_route_skip_reason(config) is None


class TestDynamoD801Helpers:
    """D801 helper compatibility across Dynamo CRD schema versions."""

    def test_find_decode_component_supports_v1alpha1_services_map(self) -> None:
        dgd = {
            "spec": {
                "services": {
                    "Frontend": {"componentType": "frontend"},
                    "VllmDecodeWorker": {
                        "componentType": "worker",
                        "subComponentType": "decode",
                        "replicas": 1,
                    },
                }
            }
        }
        assert _find_decode_component(dgd) == {
            "componentType": "worker",
            "subComponentType": "decode",
            "replicas": 1,
        }


class TestDynamoD802Preconditions:
    """D802 static preconditions encode Dynamo v1.1.0's no-etcd default."""

    def test_v1_1_default_skips_without_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIPERF_DYNAMO_ETCD_CHAOS", raising=False)

        reason = _d802_static_skip_reason("1.1.0")

        assert reason is not None
        assert "global.etcd.install=false" in reason
        assert "AIPERF_DYNAMO_ETCD_CHAOS=1" in reason

    def test_opt_in_allows_service_precondition_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DYNAMO_ETCD_CHAOS", "1")

        assert _d802_static_skip_reason("1.1.0") is None


class TestDynamoWebhookWarmup:
    """Detect transient admission-webhook startup failures."""

    @pytest.mark.parametrize(
        "message",
        [
            'failed calling webhook "mdynamographdeployment.kb.io": connect: connection refused',
            'failed calling webhook "vdynamographdeployment.kb.io": no endpoints available for service',
        ],
    )
    def test_webhook_warmup_errors_are_retryable(self, message: str) -> None:
        assert is_dynamo_webhook_warmup_error(RuntimeError(message))

    def test_non_webhook_errors_are_not_retryable(self) -> None:
        err = RuntimeError("strict decoding error: unknown field spec.components")
        assert not is_dynamo_webhook_warmup_error(err)


class TestDynamoConfigDefaults:
    """The default api_version + opt-in v1beta1."""

    def test_default_api_version_is_v1alpha1(self) -> None:
        config = _dynamo_helpers.DynamoConfig()
        assert config.api_version == "v1alpha1"

    def test_explicit_v1beta1_preserved(self) -> None:
        config = _dynamo_helpers.DynamoConfig(api_version="v1beta1")
        assert config.api_version == "v1beta1"

    def test_explicit_v1alpha1_preserved(self) -> None:
        config = _dynamo_helpers.DynamoConfig(api_version="v1alpha1")
        assert config.api_version == "v1alpha1"

    def test_single_gpu_disagg_fixture_preserves_gpu_settings_overrides(self) -> None:
        settings = GPUTestSettings(mem_util=0.05, max_model_len=2048)

        config = _dynamo_config_fixture.__wrapped__(settings)

        assert config.gpu_memory_utilization == 0.05
        assert config.max_model_len == 2048


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

        Under the v1beta1 shim used in this file, it emits the new list-based
        ``spec.components`` shape with ``gpu_count=0`` (so K8s does not block
        scheduling on single-GPU dev boxes) and ``runtimeClassName=nvidia``
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
