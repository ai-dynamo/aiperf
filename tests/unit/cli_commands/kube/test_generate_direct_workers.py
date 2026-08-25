# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-count routing for ``aiperf kube generate --no-operator``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.generate import (
    _dump_raw_manifests,
    _reject_orchestrated_direct_workload,
)
from aiperf.config.kube import KubeOptions


@pytest.mark.parametrize(
    ("options", "expected_workers"),
    [
        param({"total_workers": 7}, 7, id="explicit-total"),
        param({}, 25, id="omitted-derives-from-connections"),
    ],
)  # fmt: skip
def test_dump_raw_manifests_resolves_direct_worker_count(
    options: dict[str, int], expected_workers: int
) -> None:
    source_config = MagicMock()
    source_config.model_dump.return_value = {"benchmark": {}}

    phase = MagicMock()
    phase.concurrency = 100
    resolved_config = MagicMock()
    resolved_config.benchmark.phases = [phase]

    deployment_config = MagicMock()
    deployment_config.connections_per_worker = 4
    deployment_config.pod_template.env = []
    kube_options = KubeOptions(image="aiperf:test", ttl_seconds=10, **options)

    deployment = MagicMock()
    deployment.get_all_manifests.return_value = []
    with (
        patch(
            "aiperf.config.AIPerfConfig.model_validate", return_value=resolved_config
        ),
        patch("aiperf.kubernetes.spec_converter.apply_k8s_runtime_config"),
        patch(
            "aiperf.kubernetes.spec_converter.apply_worker_config",
            return_value=1,
        ) as apply_workers,
        patch(
            "aiperf.config.kube.KubeOptions.to_deployment_config",
            return_value=deployment_config,
        ),
        patch(
            "aiperf.kubernetes.resources.KubernetesDeployment",
            return_value=deployment,
        ),
        patch(
            "aiperf.common.endpoint_credentials.validate_kubernetes_credential_transport"
        ),
    ):
        _dump_raw_manifests(
            config=source_config,
            kube_options=kube_options,
            name="bench",
            namespace="ns",
            yaml=MagicMock(),
        )

    apply_workers.assert_called_once_with(resolved_config, expected_workers)


def test_direct_generation_rejects_sweep_or_multi_run_workload() -> None:
    config = MagicMock()
    with (
        patch(
            "aiperf.kubernetes.sweep_routing.requires_sweep_controller",
            return_value=True,
        ),
        patch(
            "aiperf.cli_utils.raise_startup_error_and_exit",
            side_effect=SystemExit(1),
        ) as fail,
        pytest.raises(SystemExit) as exc_info,
    ):
        _reject_orchestrated_direct_workload(config)

    assert exc_info.value.code == 1
    message = fail.call_args.args[0]
    assert "--no-operator" in message
    assert "parameter-sweep or multi-run" in message
    assert "aiperf kube generate --operator" in message
    assert "aiperf kube sweep" in message


def test_direct_generation_accepts_single_run_workload() -> None:
    config = MagicMock()
    with (
        patch(
            "aiperf.kubernetes.sweep_routing.requires_sweep_controller",
            return_value=False,
        ),
        patch("aiperf.cli_utils.raise_startup_error_and_exit") as fail,
    ):
        _reject_orchestrated_direct_workload(config)

    fail.assert_not_called()


def test_dump_raw_manifests_preserves_cr_deployment_spec() -> None:
    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig

    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://svc:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 4,
                    }
                ],
            }
        }
    )
    deployment = MagicMock()
    deployment.get_all_manifests.return_value = []
    captured: dict[str, object] = {}

    def _deployment_factory(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return deployment

    with patch(
        "aiperf.kubernetes.resources.KubernetesDeployment",
        side_effect=_deployment_factory,
    ):
        _dump_raw_manifests(
            config=config,
            kube_options=KubeOptions(image="aiperf:test"),
            name="bench",
            namespace="ns",
            yaml=MagicMock(),
            deployment_spec={
                "image": "aiperf:test",
                "resourceMode": "none",
                "keepFailedPods": True,
                "ttlSecondsAfterFinished": 999,
                "podTemplate": {"nodeSelector": {"region": "west"}},
            },
        )

    resolved = captured["deployment"]
    assert isinstance(resolved, DeploymentConfig)
    assert resolved.resource_mode == "none"
    assert resolved.keep_failed_pods is True
    assert resolved.ttl_seconds_after_finished == 999
    assert resolved.pod_template.node_selector == {"region": "west"}


def test_dump_raw_manifests_includes_requested_namespace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ruamel.yaml

    from aiperf.config import AIPerfConfig

    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://svc:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 4,
                    }
                ],
            }
        }
    )

    _dump_raw_manifests(
        config=config,
        kube_options=KubeOptions(image="aiperf:test"),
        name="bench",
        namespace="tenant-a",
        yaml=ruamel.yaml.YAML(),
    )

    parser = ruamel.yaml.YAML(typ="safe")
    manifests = list(parser.load_all(capsys.readouterr().out))
    assert [manifest["kind"] for manifest in manifests] == [
        "Namespace",
        "Role",
        "RoleBinding",
        "ConfigMap",
        "JobSet",
    ]
    assert manifests[0]["metadata"]["name"] == "tenant-a"
