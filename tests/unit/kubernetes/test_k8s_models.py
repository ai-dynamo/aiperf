# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.k8s_models module.

Focuses on:
- camelCase alias generation and serialization
- populate_by_name (construction from snake_case or camelCase)
- to_k8s_dict() output: camelCase keys, None exclusion, JSON mode
- Subclass inheritance of config behavior
"""

import pytest
from pydantic import Field
from pydantic.alias_generators import to_camel
from pytest import param

from aiperf.kubernetes.k8s_models import K8sCamelModel
from aiperf.operator.models import OwnerReference, PhaseProgress

# ============================================================
# Test subclass used across tests
# ============================================================


class SampleK8sModel(K8sCamelModel):
    """Minimal subclass for testing base behavior."""

    api_version: str = Field(description="API version string")
    container_port: int = Field(description="Port number")
    is_ready: bool = Field(default=False, description="Readiness flag")


class NestedK8sModel(K8sCamelModel):
    """Model with a nested K8sCamelModel field."""

    metadata_name: str = Field(description="Name in metadata")
    owner_ref: SampleK8sModel | None = Field(
        default=None, description="Optional owner reference"
    )


# ============================================================
# Camel-case alias generation
# ============================================================


class TestCamelCaseAliasing:
    """Verify snake_case fields get camelCase aliases."""

    def test_model_dump_by_alias_produces_camel_case(self) -> None:
        model = SampleK8sModel(api_version="v1", container_port=8080)
        dumped = model.model_dump(by_alias=True)
        assert "apiVersion" in dumped
        assert "containerPort" in dumped
        assert "isReady" in dumped

    def test_model_dump_without_alias_produces_snake_case(self) -> None:
        model = SampleK8sModel(api_version="v1", container_port=8080)
        dumped = model.model_dump(by_alias=False)
        assert "api_version" in dumped
        assert "container_port" in dumped
        assert "is_ready" in dumped

    @pytest.mark.parametrize(
        "snake_field,camel_field",
        [
            ("api_version", "apiVersion"),
            ("container_port", "containerPort"),
            ("is_ready", "isReady"),
        ],
    )  # fmt: skip
    def test_alias_mapping_correct(self, snake_field: str, camel_field: str) -> None:
        field_info = SampleK8sModel.model_fields[snake_field]
        assert field_info.alias == camel_field


# ============================================================
# Construction: populate_by_name
# ============================================================


class TestPopulateByName:
    """Verify models can be constructed from both snake_case and camelCase."""

    def test_construct_from_snake_case(self) -> None:
        model = SampleK8sModel(api_version="v1", container_port=8080, is_ready=True)
        assert model.api_version == "v1"
        assert model.container_port == 8080
        assert model.is_ready is True

    def test_construct_from_camel_case(self) -> None:
        model = SampleK8sModel(apiVersion="v1", containerPort=8080, isReady=True)
        assert model.api_version == "v1"
        assert model.container_port == 8080
        assert model.is_ready is True

    def test_construct_from_camel_case_dict(self) -> None:
        data = {"apiVersion": "v1", "containerPort": 443, "isReady": True}
        model = SampleK8sModel.model_validate(data)
        assert model.api_version == "v1"
        assert model.container_port == 443

    def test_construct_from_snake_case_dict(self) -> None:
        data = {"api_version": "v1", "container_port": 443}
        model = SampleK8sModel.model_validate(data)
        assert model.api_version == "v1"
        assert model.container_port == 443


# ============================================================
# to_k8s_dict()
# ============================================================


class TestToK8sDict:
    """Verify to_k8s_dict serialization behavior."""

    def test_to_k8s_dict_uses_camel_case_keys(self) -> None:
        model = SampleK8sModel(api_version="v1", container_port=8080)
        result = model.to_k8s_dict()
        assert "apiVersion" in result
        assert "containerPort" in result
        assert "api_version" not in result
        assert "container_port" not in result

    def test_to_k8s_dict_excludes_none_values(self) -> None:
        model = NestedK8sModel(metadata_name="test", owner_ref=None)
        result = model.to_k8s_dict()
        assert "metadataName" in result
        assert "ownerRef" not in result

    def test_to_k8s_dict_includes_non_none_values(self) -> None:
        inner = SampleK8sModel(api_version="v1", container_port=80)
        model = NestedK8sModel(metadata_name="test", owner_ref=inner)
        result = model.to_k8s_dict()
        assert "ownerRef" in result
        assert result["ownerRef"]["apiVersion"] == "v1"

    def test_to_k8s_dict_preserves_falsy_non_none_values(self) -> None:
        """False, 0, empty string should NOT be excluded (only None)."""
        model = SampleK8sModel(api_version="", container_port=0, is_ready=False)
        result = model.to_k8s_dict()
        assert result["apiVersion"] == ""
        assert result["containerPort"] == 0
        assert result["isReady"] is False

    def test_to_k8s_dict_returns_plain_dict(self) -> None:
        model = SampleK8sModel(api_version="v1", container_port=8080)
        result = model.to_k8s_dict()
        assert type(result) is dict

    def test_to_k8s_dict_json_mode_serializes_complex_types(self) -> None:
        """JSON mode should produce JSON-compatible primitives."""
        model = SampleK8sModel(api_version="v1", container_port=8080, is_ready=True)
        result = model.to_k8s_dict()
        # All values should be JSON primitives
        assert isinstance(result["apiVersion"], str)
        assert isinstance(result["containerPort"], int)
        assert isinstance(result["isReady"], bool)


# ============================================================
# Inheritance
# ============================================================


class TestInheritance:
    """Verify subclasses inherit camelCase config properly."""

    def test_subclass_inherits_alias_generator(self) -> None:
        """A subclass that adds fields should still get camelCase aliases."""

        class ExtendedModel(SampleK8sModel):
            extra_field: str = Field(
                default="default", description="Extra field for testing"
            )

        model = ExtendedModel(api_version="v1", container_port=80)
        result = model.to_k8s_dict()
        assert "extraField" in result
        assert "apiVersion" in result

    def test_real_subclass_owner_reference(self) -> None:
        """Test with the actual OwnerReference model from the operator."""
        ref = OwnerReference(
            api_version="aiperf.nvidia.com/v1alpha1",
            kind="AIPerfJob",
            name="my-job",
            uid="abc-123",
        )
        result = ref.to_k8s_dict()
        assert result["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert result["kind"] == "AIPerfJob"
        assert result["blockOwnerDeletion"] is True
        assert "block_owner_deletion" not in result

    def test_real_subclass_phase_progress(self) -> None:
        """Test with the actual PhaseProgress model from the operator."""
        progress = PhaseProgress(
            requests_completed=50,
            requests_sent=55,
            requests_total=100,
            requests_cancelled=0,
            requests_errors=2,
            requests_in_flight=3,
            requests_per_second=10.5,
            requests_progress_percent=50.0,
            sessions_sent=10,
            sessions_completed=8,
            sessions_cancelled=0,
            sessions_in_flight=2,
            records_success=40,
            records_error=1,
            records_per_second=8.0,
            records_progress_percent=41.0,
            sending_complete=False,
            timeout_triggered=False,
            was_cancelled=False,
        )
        result = progress.to_k8s_dict()
        assert result["requestsCompleted"] == 50
        assert result["requestsPerSecond"] == 10.5
        assert result["requestsProgressPercent"] == 50.0


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    """Verify behavior with unusual but valid inputs."""

    def test_model_roundtrip_camel_to_snake(self) -> None:
        """Serialize to camelCase dict, then reconstruct from that dict."""
        original = SampleK8sModel(api_version="v1", container_port=443, is_ready=True)
        camel_dict = original.to_k8s_dict()
        reconstructed = SampleK8sModel.model_validate(camel_dict)
        assert reconstructed.api_version == original.api_version
        assert reconstructed.container_port == original.container_port
        assert reconstructed.is_ready == original.is_ready

    def test_nested_model_roundtrip(self) -> None:
        inner = SampleK8sModel(api_version="v1", container_port=80)
        original = NestedK8sModel(metadata_name="test", owner_ref=inner)
        camel_dict = original.to_k8s_dict()
        reconstructed = NestedK8sModel.model_validate(camel_dict)
        assert reconstructed.metadata_name == "test"
        assert reconstructed.owner_ref is not None
        assert reconstructed.owner_ref.container_port == 80

    def test_to_k8s_dict_deeply_nested_none_excluded(self) -> None:
        """None in nested model should also be excluded."""
        model = NestedK8sModel(metadata_name="test", owner_ref=None)
        result = model.to_k8s_dict()
        assert len(result) == 1
        assert result == {"metadataName": "test"}

    @pytest.mark.parametrize(
        "field_name,camel_name",
        [
            param("a_b", "aB", id="simple-two-part"),
            param("my_long_field_name", "myLongFieldName", id="multi-part"),
        ],
    )  # fmt: skip
    def test_alias_generation_various_patterns(
        self, field_name: str, camel_name: str
    ) -> None:
        """Verify camelCase conversion across various naming patterns."""
        assert to_camel(field_name) == camel_name
