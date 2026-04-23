# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.jobset_specs.

Covers the pod-spec assembly details inside AIPerfReplicatedJobSpec that
aren't exercised by the broader JobSet integration tests: pod-level
security context, annotation/label merging, service account plumbing,
image pull secrets, and the top-level job-spec shape.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.enums import RestartPolicy
from aiperf.kubernetes.jobset_specs import (
    AIPerfContainerSpec,
    AIPerfReplicatedJobSpec,
)


@pytest.fixture
def minimal_container() -> AIPerfContainerSpec:
    """A bare container spec usable by any replicated-job test."""
    return AIPerfContainerSpec(name="worker", image="nginx:latest")


class TestPodSpecSecurityContext:
    """Pod-level security context is emitted on every replicated job."""

    def test_pod_security_context_defaults(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """fsGroup and non-root settings apply across all pods."""
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        ctx = pod_spec["securityContext"]
        assert ctx["runAsNonRoot"] is True
        assert ctx["runAsUser"] == 1000
        assert ctx["runAsGroup"] == 1000
        assert ctx["fsGroup"] == 1000
        assert ctx["seccompProfile"] == {"type": "RuntimeDefault"}


class TestPodSpecSchedulingOverrides:
    """PodTemplateConfig overrides flow into the k8s pod spec."""

    def test_node_selector_applied(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        template = PodTemplateConfig(node_selector={"gpu": "true"})
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"] == {"gpu": "true"}

    def test_tolerations_applied(self, minimal_container: AIPerfContainerSpec) -> None:
        template = PodTemplateConfig(
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}]
        )
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert pod_spec["tolerations"] == [
            {"key": "nvidia.com/gpu", "operator": "Exists"}
        ]

    def test_image_pull_secrets_converted_to_objects(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """Bare secret names must be wrapped as ``{"name": ...}`` objects."""
        template = PodTemplateConfig(image_pull_secrets=["secret-a", "secret-b"])
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert pod_spec["imagePullSecrets"] == [
            {"name": "secret-a"},
            {"name": "secret-b"},
        ]

    def test_service_account_name_applied(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        template = PodTemplateConfig(service_account_name="aiperf-sa")
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "aiperf-sa"

    def test_no_pod_template_means_no_scheduling_keys(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """Empty pod template must not leak empty scheduling fields into the spec."""
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        for key in (
            "nodeSelector",
            "tolerations",
            "imagePullSecrets",
            "serviceAccountName",
        ):
            assert key not in pod_spec

    def test_empty_pod_template_omits_scheduling_keys(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """An explicit but empty PodTemplateConfig still omits unset scheduling keys."""
        job = AIPerfReplicatedJobSpec(
            name="workers",
            containers=[minimal_container],
            pod_template=PodTemplateConfig(),
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert "nodeSelector" not in pod_spec
        assert "tolerations" not in pod_spec
        assert "imagePullSecrets" not in pod_spec


class TestPodAnnotationsMerge:
    """Pod annotations come from the template, merged with extras."""

    def test_no_annotations_key_when_none(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """Absence of annotations must not materialize an empty dict."""
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        metadata = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]
        assert "annotations" not in metadata

    def test_template_annotations_alone(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        template = PodTemplateConfig(annotations={"team": "perf"})
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        metadata = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]
        assert metadata["annotations"] == {"team": "perf"}

    def test_extra_annotations_alone(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        job = AIPerfReplicatedJobSpec(
            name="controller",
            containers=[minimal_container],
            extra_annotations={"prometheus.io/scrape": "true"},
        )
        metadata = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]
        assert metadata["annotations"] == {"prometheus.io/scrape": "true"}

    def test_extra_annotations_override_template_annotations(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """extra_annotations must win on key collisions (applied after template)."""
        template = PodTemplateConfig(annotations={"shared": "template"})
        job = AIPerfReplicatedJobSpec(
            name="controller",
            containers=[minimal_container],
            pod_template=template,
            extra_annotations={"shared": "extra"},
        )
        metadata = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]
        assert metadata["annotations"]["shared"] == "extra"


class TestPodLabels:
    """Pod labels include AIPerf baseline and optional job id."""

    def test_base_label_always_present(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        labels = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]["labels"]
        assert labels[AIPerfLabels.APP_KEY] == AIPerfLabels.APP_VALUE

    def test_job_id_label_added_when_set(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], job_id="bench-42"
        )
        labels = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]["labels"]
        assert labels[AIPerfLabels.JOB_ID] == "bench-42"

    def test_job_id_label_absent_when_unset(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        labels = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]["labels"]
        assert AIPerfLabels.JOB_ID not in labels

    def test_custom_pod_template_labels_merge_in(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        template = PodTemplateConfig(labels={"team": "perf"})
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], pod_template=template
        )
        labels = job.to_k8s_spec()["template"]["spec"]["template"]["metadata"]["labels"]
        assert labels["team"] == "perf"
        assert labels[AIPerfLabels.APP_KEY] == AIPerfLabels.APP_VALUE


class TestJobSpecShape:
    """The generated ``jobSpec`` structure matches Kubernetes conventions."""

    def test_job_spec_uses_indexed_completion_mode(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """JobSet requires Indexed completion mode so job-index labels are stable."""
        job = AIPerfReplicatedJobSpec(name="workers", containers=[minimal_container])
        job_spec = job.to_k8s_spec()["template"]["spec"]
        assert job_spec["completionMode"] == "Indexed"
        assert job_spec["parallelism"] == 1
        assert job_spec["completions"] == 1

    def test_backoff_limit_is_propagated(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[minimal_container], backoff_limit=7
        )
        job_spec = job.to_k8s_spec()["template"]["spec"]
        assert job_spec["backoffLimit"] == 7

    @pytest.mark.parametrize(
        "ttl,has_ttl_key",
        [
            param(None, False, id="none-omits-key"),
            param(0, True, id="zero-emits-key"),
            param(300, True, id="positive-emits-key"),
        ],
    )  # fmt: skip
    def test_ttl_emission(
        self,
        minimal_container: AIPerfContainerSpec,
        ttl: int | None,
        has_ttl_key: bool,
    ) -> None:
        """``ttlSecondsAfterFinished`` is only emitted when an explicit value is set."""
        job = AIPerfReplicatedJobSpec(
            name="workers",
            containers=[minimal_container],
            job_ttl_seconds=ttl,
        )
        job_spec = job.to_k8s_spec()["template"]["spec"]
        assert ("ttlSecondsAfterFinished" in job_spec) == has_ttl_key
        if has_ttl_key:
            assert job_spec["ttlSecondsAfterFinished"] == ttl

    def test_restart_policy_serialized_as_string(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """RestartPolicy enum must serialize to its string value (not the enum repr)."""
        job = AIPerfReplicatedJobSpec(
            name="workers",
            containers=[minimal_container],
            restart_policy=RestartPolicy.NEVER,
        )
        pod_spec = job.to_k8s_spec()["template"]["spec"]["template"]["spec"]
        assert pod_spec["restartPolicy"] == "Never"

    def test_replicas_and_name_at_top_level(
        self, minimal_container: AIPerfContainerSpec
    ) -> None:
        """Top-level ``name`` and ``replicas`` must appear in the manifest fragment."""
        job = AIPerfReplicatedJobSpec(
            name="myjob", replicas=4, containers=[minimal_container]
        )
        result = job.to_k8s_spec()
        assert result["name"] == "myjob"
        assert result["replicas"] == 4
