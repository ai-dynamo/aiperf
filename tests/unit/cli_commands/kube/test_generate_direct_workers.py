# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-count routing for ``aiperf kube generate --no-operator``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube.generate import (
    _dump_raw_manifests,
    _reject_orchestrated_direct_workload,
)
from aiperf.config.kube import KubeOptions


@pytest.mark.parametrize(
    ("options", "expected_workers"),
    [
        param({"total_workers": 7}, 7, id="explicit-total"),
        param({}, 30, id="omitted-derives-then-rounds-to-whole-pods"),
    ],
)  # fmt: skip
def test_dump_raw_manifests_resolves_direct_worker_count(
    options: dict[str, int], expected_workers: int
) -> None:
    """An explicit total is passed through; a derived one is rounded up.

    ceil(100/4) = 25 workers, which cannot fill uniform 10-worker pods and
    would be rejected downstream. Nobody typed 25, so it becomes 30 rather than
    failing the run on a number the user never chose.
    """
    source_config = MagicMock()
    source_config.model_dump.return_value = {"benchmark": {}}

    phase = MagicMock()
    phase.concurrency = 100
    resolved_config = MagicMock()
    resolved_config.benchmark.phases = [phase]
    resolved_config.benchmark.runtime.workers = None
    resolved_config.benchmark.runtime.workers_per_pod = None

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


def test_generate_memory_estimate_delegates_to_shared_helper() -> None:
    config = MagicMock()
    phase = MagicMock()
    phase.concurrency = 1
    config.benchmark.phases = [phase]
    config.benchmark.runtime.workers = None
    config.benchmark.runtime.workers_per_pod = None
    kube_options = KubeOptions()
    spec = {"connectionsPerWorker": 17}

    with (
        patch(
            "aiperf.cli_commands.kube._kube_common.print_memory_estimate"
        ) as print_memory_estimate,
        patch("aiperf.kubernetes.memory_estimator.estimate_memory"),
        patch("aiperf.kubernetes.memory_estimator.format_estimate", return_value=""),
    ):
        generate_cmd._print_memory_estimate(config, kube_options, spec)

    print_memory_estimate.assert_called_once_with(config, kube_options, spec)


def test_default_direct_manifests_pin_image_and_pull_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ruamel.yaml

    from aiperf import __version__
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
        kube_options=KubeOptions(),
        name="bench",
        namespace="ns",
        yaml=ruamel.yaml.YAML(),
    )

    parser = ruamel.yaml.YAML(typ="safe")
    jobset = next(
        manifest
        for manifest in parser.load_all(capsys.readouterr().out)
        if manifest["kind"] == "JobSet"
    )
    containers = [
        container
        for replicated_job in jobset["spec"]["replicatedJobs"]
        for container in replicated_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
    ]
    assert {container["image"] for container in containers} == {
        f"nvcr.io/nvidia/aiperf:{__version__}"
    }
    assert {container["imagePullPolicy"] for container in containers} == {
        "IfNotPresent"
    }


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


def test_dump_raw_manifests_targets_requested_namespace_without_creating_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Generated YAML must never contain a Namespace manifest.

    The benchmark namespace is named by the user and must already exist;
    emitting one would make applying the output require cluster-scoped
    namespace-create rights.
    """
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
        "Role",
        "RoleBinding",
        "ConfigMap",
        "JobSet",
    ]
    assert {manifest["metadata"]["namespace"] for manifest in manifests} == {"tenant-a"}
