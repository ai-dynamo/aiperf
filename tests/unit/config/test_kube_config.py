# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes CLI option models."""

import pytest
from cyclopts import App

from aiperf.config.kube import KubeManageOptions, KubeOptions, SecretMountConfig


def test_kube_manage_options_defaults():
    opts = KubeManageOptions()
    assert opts.namespace is None
    assert opts.kubeconfig is None
    assert opts.kube_context is None


def test_kube_options_extends_manage_options():
    assert issubclass(KubeOptions, KubeManageOptions)


def test_kube_options_cli_binding_does_not_require_image():
    app = App(name="test_kube")

    @app.default
    def _runner(*, kube_options: KubeOptions | None = None) -> None:
        del kube_options

    app.parse_args([], exit_on_error=False, print_error=False)


def test_secret_mount_requires_name_and_path():
    mount = SecretMountConfig(name="hf-token-secret", mount_path="/secrets/hf")
    assert mount.name == "hf-token-secret"
    with pytest.raises(ValueError):
        SecretMountConfig(name="only-name")


def test_kube_options_node_selector_accepts_key_value_cli_tokens():
    opts = KubeOptions(image="aiperf:latest", node_selector_cli=["gpu=true"])
    assert opts.node_selector == {"gpu": "true"}


def test_kube_options_rejects_env_from_secrets_without_key():
    with pytest.raises(ValueError, match="secret_name/key"):
        KubeOptions(image="aiperf:latest", env_from_secrets={"HF_TOKEN": "hf-secret"})


def test_kube_options_to_deployment_config_expands_secret_mounts():
    opts = KubeOptions(
        image="aiperf:latest",
        env_from_secrets={"HF_TOKEN": "hf-secret/token"},
        secret_mounts=[SecretMountConfig(name="hf", mount_path="/secrets/hf")],
    )
    deployment = opts.to_deployment_config()
    assert deployment.image == "aiperf:latest"
    assert deployment.pod_template.volumes == [
        {"name": "secret-hf", "secret": {"secretName": "hf"}}
    ]
    assert deployment.pod_template.env[0]["valueFrom"]["secretKeyRef"] == {
        "name": "hf-secret",
        "key": "token",
    }


def test_kube_options_to_deployment_config_omits_unauthored_defaults():
    opts = KubeOptions(image="aiperf:latest")

    deployment = opts.to_deployment_config()

    assert opts.model_fields_set == {"image"}
    assert deployment.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    ) == {"image": "aiperf:latest"}


def test_kube_options_to_deployment_config_omits_unauthored_image_override():
    opts = KubeOptions()

    deployment = opts.to_deployment_config()

    assert opts.model_fields_set == set()
    assert deployment.image == "nvcr.io/nvidia/aiperf:latest"
    assert "image" not in deployment.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )


def test_kube_options_to_deployment_config_preserves_explicit_default_values():
    opts = KubeOptions(
        image="aiperf:latest",
        ttl_seconds=300,
        node_selector={},
        tolerations=[],
    )

    deployment = opts.to_deployment_config()
    dumped = deployment.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )

    assert dumped["ttlSecondsAfterFinished"] == 300
    assert dumped["podTemplate"]["nodeSelector"] == {}
    assert dumped["podTemplate"]["tolerations"] == []


def test_kube_options_to_deployment_config_preserves_cli_node_selector_provenance():
    opts = KubeOptions(image="aiperf:latest", node_selector_cli=["gpu=true"])

    dumped = opts.to_deployment_config().model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )

    assert dumped["podTemplate"]["nodeSelector"] == {"gpu": "true"}


def test_kube_options_total_workers_stamps_runtime_workers() -> None:
    from aiperf.config import AIPerfConfig

    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 10,
                    }
                ],
            }
        }
    )

    spec = KubeOptions(image="aiperf:latest", total_workers=6).to_crd_spec(config)

    assert spec["benchmark"]["runtime"]["workers"] == 6
    # The ratio remains an autosizing hint, but runtime.workers carries the
    # exact total when concurrency is not evenly divisible.
    assert spec["connectionsPerWorker"] == 2


def test_kube_options_rejects_negative_ttl():
    with pytest.raises(ValueError):
        KubeOptions(image="aiperf:latest", ttl_seconds=-1)


def test_kube_options_accepts_valid_dns_label_name():
    opts = KubeOptions(image="aiperf:latest", name="aiperf-bench-7f2a")
    assert opts.name == "aiperf-bench-7f2a"
