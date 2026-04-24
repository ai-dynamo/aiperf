# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Kueue/gang-scheduling integration across the Kubernetes modules.

Tests cover:
- KueueLabels constants integrity
- AIPerfJobSetSpec Kueue label injection and suspend behavior
- KubernetesDeployment passthrough of queue_name / priority_class
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest
from pytest import param

from aiperf.config.deployment import DeploymentConfig, SchedulingConfig
from aiperf.kubernetes.constants import KueueLabels
from aiperf.kubernetes.jobset import AIPerfJobSetSpec
from aiperf.kubernetes.resources import KubernetesDeployment

# ---------------------------------------------------------------------------
# KueueLabels constant integrity
# ---------------------------------------------------------------------------


class TestKueueLabels:
    """Verify KueueLabels dataclass constants are correct and immutable."""

    def test_queue_name_value(self) -> None:
        labels = KueueLabels()
        assert labels.QUEUE_NAME == "kueue.x-k8s.io/queue-name"

    def test_priority_class_value(self) -> None:
        labels = KueueLabels()
        assert labels.PRIORITY_CLASS == "kueue.x-k8s.io/priority-class"

    def test_all_values_are_nonempty_strings(self) -> None:
        instance = KueueLabels()
        for f in fields(instance):
            val = getattr(instance, f.name)
            assert isinstance(val, str), f"KueueLabels.{f.name} is not a str"
            assert val != "", f"KueueLabels.{f.name} is empty"

    def test_no_duplicate_values(self) -> None:
        instance = KueueLabels()
        values = [getattr(instance, f.name) for f in fields(instance)]
        assert len(values) == len(set(values)), (
            f"KueueLabels has duplicate values: {values}"
        )

    def test_frozen_immutability(self) -> None:
        instance = KueueLabels()
        with pytest.raises(FrozenInstanceError):
            instance.QUEUE_NAME = "tampered"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "field_name",
        [
            param("QUEUE_NAME", id="QUEUE_NAME"),
            param("PRIORITY_CLASS", id="PRIORITY_CLASS"),
        ],
    )  # fmt: skip
    def test_kueue_labels_use_kueue_domain_prefix(self, field_name: str) -> None:
        value = getattr(KueueLabels(), field_name)
        assert value.startswith("kueue.x-k8s.io/"), (
            f"KueueLabels.{field_name} = {value!r} missing kueue domain prefix"
        )


# ---------------------------------------------------------------------------
# Helpers to build a minimal AIPerfJobSetSpec
# ---------------------------------------------------------------------------


def _make_jobset_spec(**overrides: Any) -> AIPerfJobSetSpec:
    """Create a minimal AIPerfJobSetSpec with optional overrides."""
    queue_name = overrides.pop("queue_name", None)
    priority_class = overrides.pop("priority_class", None)
    scheduling_kwargs: dict[str, Any] = {}
    if queue_name is not None:
        scheduling_kwargs["queue_name"] = queue_name
    if priority_class is not None:
        scheduling_kwargs["priority_class"] = priority_class
    if scheduling_kwargs:
        overrides.setdefault("scheduling", SchedulingConfig(**scheduling_kwargs))

    defaults: dict[str, Any] = {
        "name": "test-js",
        "job_id": "abc123",
        "image": "test:latest",
    }
    defaults.update(overrides)
    return AIPerfJobSetSpec(**defaults)


# ---------------------------------------------------------------------------
# AIPerfJobSetSpec Kueue integration
# ---------------------------------------------------------------------------


class TestJobSetSpecKueue:
    """Tests for Kueue-related behavior in AIPerfJobSetSpec.to_k8s_manifest()."""

    def test_to_k8s_manifest_without_queue_name_has_no_kueue_labels(self) -> None:
        spec = _make_jobset_spec()
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME not in labels
        assert KueueLabels.PRIORITY_CLASS not in labels

    def test_to_k8s_manifest_with_queue_name_adds_kueue_label(self) -> None:
        spec = _make_jobset_spec(queue_name="my-queue")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels[KueueLabels.QUEUE_NAME] == "my-queue"

    def test_to_k8s_manifest_with_queue_name_sets_suspend_true(self) -> None:
        spec = _make_jobset_spec(queue_name="my-queue")
        manifest = spec.to_k8s_manifest()
        assert manifest["spec"]["suspend"] is True

    def test_to_k8s_manifest_with_priority_class_adds_priority_label(self) -> None:
        spec = _make_jobset_spec(queue_name="my-queue", priority_class="high-priority")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels[KueueLabels.PRIORITY_CLASS] == "high-priority"

    def test_to_k8s_manifest_with_both_queue_and_priority_adds_both_labels(
        self,
    ) -> None:
        spec = _make_jobset_spec(queue_name="team-queue", priority_class="batch-low")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels[KueueLabels.QUEUE_NAME] == "team-queue"
        assert labels[KueueLabels.PRIORITY_CLASS] == "batch-low"

    def test_to_k8s_manifest_without_queue_name_no_suspend(self) -> None:
        spec = _make_jobset_spec()
        manifest = spec.to_k8s_manifest()
        assert "suspend" not in manifest["spec"]

    def test_to_k8s_manifest_priority_class_without_queue_name_no_suspend(
        self,
    ) -> None:
        """priority_class alone does not trigger suspend (no queue_name)."""
        spec = _make_jobset_spec(priority_class="high-priority")
        manifest = spec.to_k8s_manifest()
        assert "suspend" not in manifest["spec"]

    def test_to_k8s_manifest_priority_class_without_queue_name_still_adds_label(
        self,
    ) -> None:
        spec = _make_jobset_spec(priority_class="high-priority")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels[KueueLabels.PRIORITY_CLASS] == "high-priority"

    def test_to_k8s_manifest_base_labels_preserved_with_kueue(self) -> None:
        """Kueue labels do not remove the standard AIPerf labels."""
        spec = _make_jobset_spec(queue_name="q")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels["app"] == "aiperf"
        assert labels["aiperf.nvidia.com/job-id"] == "abc123"

    def test_to_k8s_manifest_name_label_preserved_with_kueue(self) -> None:
        spec = _make_jobset_spec(queue_name="q", name_label="my-bench")
        manifest = spec.to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels["aiperf.nvidia.com/name"] == "my-bench"
        assert KueueLabels.QUEUE_NAME in labels


# ---------------------------------------------------------------------------
# KubernetesDeployment passthrough
# ---------------------------------------------------------------------------


class TestKubernetesDeploymentKueue:
    """Tests that KubernetesDeployment passes Kueue fields to AIPerfJobSetSpec."""

    @pytest.fixture
    def _deployment_kwargs(self, sample_config) -> dict[str, Any]:
        return {
            "job_id": "ktest1",
            "deployment": DeploymentConfig(image="aiperf:latest"),
            "config": sample_config,
        }

    def test_get_jobset_spec_passes_queue_name(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        _deployment_kwargs["deployment"] = DeploymentConfig(
            image="aiperf:latest",
            scheduling=SchedulingConfig(queue_name="team-queue"),
        )
        deployment = KubernetesDeployment(**_deployment_kwargs)
        js_spec = deployment.get_jobset_spec()
        assert js_spec.scheduling.queue_name == "team-queue"

    def test_get_jobset_spec_passes_priority_class(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        _deployment_kwargs["deployment"] = DeploymentConfig(
            image="aiperf:latest",
            scheduling=SchedulingConfig(priority_class="high-priority"),
        )
        deployment = KubernetesDeployment(**_deployment_kwargs)
        js_spec = deployment.get_jobset_spec()
        assert js_spec.scheduling.priority_class == "high-priority"

    def test_get_jobset_spec_passes_none_by_default(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        deployment = KubernetesDeployment(**_deployment_kwargs)
        js_spec = deployment.get_jobset_spec()
        assert js_spec.scheduling.queue_name is None
        assert js_spec.scheduling.priority_class is None

    def test_get_jobset_spec_both_fields_propagate(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        _deployment_kwargs["deployment"] = DeploymentConfig(
            image="aiperf:latest",
            scheduling=SchedulingConfig(
                queue_name="shared-queue", priority_class="low"
            ),
        )
        deployment = KubernetesDeployment(**_deployment_kwargs)
        js_spec = deployment.get_jobset_spec()
        assert js_spec.scheduling.queue_name == "shared-queue"
        assert js_spec.scheduling.priority_class == "low"

    def test_manifest_includes_kueue_labels_when_set(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        """End-to-end: KubernetesDeployment -> AIPerfJobSetSpec -> manifest labels."""
        _deployment_kwargs["deployment"] = DeploymentConfig(
            image="aiperf:latest",
            scheduling=SchedulingConfig(
                queue_name="prod-queue", priority_class="urgent"
            ),
        )
        deployment = KubernetesDeployment(**_deployment_kwargs)
        manifest = deployment.get_jobset_spec().to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert labels[KueueLabels.QUEUE_NAME] == "prod-queue"
        assert labels[KueueLabels.PRIORITY_CLASS] == "urgent"
        assert manifest["spec"]["suspend"] is True

    def test_manifest_no_kueue_artifacts_when_unset(
        self, _deployment_kwargs: dict[str, Any]
    ) -> None:
        deployment = KubernetesDeployment(**_deployment_kwargs)
        manifest = deployment.get_jobset_spec().to_k8s_manifest()
        labels = manifest["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME not in labels
        assert KueueLabels.PRIORITY_CLASS not in labels
        assert "suspend" not in manifest["spec"]
