# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes deployment configuration models."""

import pytest
from pydantic import ValidationError

from aiperf.config.deployment import (
    DeploymentConfig,
    PodTemplateConfig,
    SchedulingConfig,
    privilege_escalating_keys,
)


def test_deployment_config_defaults_are_conservative():
    cfg = DeploymentConfig()
    assert cfg.pod_template is not None
    assert cfg.pod_template.share_process_namespace is False


def test_pod_template_config_accepts_node_selector():
    # Node placement lives on podTemplate; SchedulingConfig is Kueue-only.
    pod_template = PodTemplateConfig(node_selector={"nvidia.com/gpu.present": "true"})
    assert pod_template.node_selector["nvidia.com/gpu.present"] == "true"


def test_scheduling_config_is_kueue_only():
    sched = SchedulingConfig(queue_name="bench-queue", priority_class="high")
    assert sched.queue_name == "bench-queue"
    assert sched.priority_class == "high"


def test_deployment_config_serializes_camel_case():
    cfg = DeploymentConfig(pod_template=PodTemplateConfig(node_selector={"a": "b"}))
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert dumped["podTemplate"]["nodeSelector"] == {"a": "b"}
    assert "ttlSecondsAfterFinished" in dumped


def test_privilege_escalating_keys_truthy_non_bool_privileged_flagged():
    # YAML `privileged: 1` parses as int 1, not bool True. The forbidden-value
    # check must not silently miss it via bool identity comparison.
    assert privilege_escalating_keys({"privileged": 1}) == ["privileged"]


def test_pod_template_config_container_security_context_truthy_int_privileged_rejected():
    with pytest.raises(ValidationError, match="privileged"):
        PodTemplateConfig(container_security_context={"privileged": 1})
