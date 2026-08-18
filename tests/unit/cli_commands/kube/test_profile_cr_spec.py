# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube profile` must not silently rewrite a CR's own fields."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli import app as aiperf_app
from aiperf.cli_commands.kube.profile import (
    _build_cr_spec_and_config,
    _print_memory_estimate,
)
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions


def _raw_cr(**spec_extra) -> dict:
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "test-job"},
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "kind": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 8,
                    }
                ],
            },
            **spec_extra,
        },
    }


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ["kube", "profile", "--config", "aiperfjob.yaml"],
            id="profile-aiperfjob",
        ),
        pytest.param(
            ["kube", "sweep", "--config", "aiperfsweep.yaml"],
            id="sweep-config",
        ),
        pytest.param(
            ["kube", "generate", "--operator", "--config", "aiperfsweep.yaml"],
            id="generate-aiperfsweep",
        ),
    ],
)
def test_kube_config_commands_bind_without_image(argv: list[str]) -> None:
    aiperf_app.parse_args(argv, exit_on_error=False, print_error=False)


class TestConnectionsPerWorkerPassthrough:
    """`total_workers` defaults to 10, so deriving unconditionally clobbers the CR.

    The hardened twin KubeOptions.to_crd_spec guards on model_fields_set with
    a comment saying not to override; this copy had no guard, so a CR
    declaring connectionsPerWorker: 500 was submitted with a derived value.
    """

    def test_cr_value_survives_when_total_workers_not_passed(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(connectionsPerWorker=500),
            KubeOptions(image="nvcr.io/nvidia/aiperf:latest"),
        )
        assert spec["connectionsPerWorker"] == 500

    def test_explicit_total_workers_still_derives(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(connectionsPerWorker=500),
            KubeOptions(image="nvcr.io/nvidia/aiperf:latest", total_workers=4),
        )
        # concurrency 8 over 4 workers.
        assert spec["connectionsPerWorker"] == 2


class TestTtlPassthrough:
    """Default-valued CLI options override a CR only when explicitly authored."""

    def test_cr_value_survives_when_ttl_not_passed(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(ttlSecondsAfterFinished=999),
            KubeOptions(image="nvcr.io/nvidia/aiperf:latest"),
        )

        assert spec["ttlSecondsAfterFinished"] == 999

    def test_explicit_default_ttl_overrides_cr_value(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(ttlSecondsAfterFinished=999),
            KubeOptions(
                image="nvcr.io/nvidia/aiperf:latest",
                ttl_seconds=300,
            ),
        )

        assert spec["ttlSecondsAfterFinished"] == 300


class TestImagePassthrough:
    """Raw workload images are authoritative unless CLI authored --image."""

    def test_cr_image_survives_when_image_not_passed(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(image="registry.example/aiperf:from-yaml"),
            KubeOptions(),
        )

        assert spec["image"] == "registry.example/aiperf:from-yaml"

    def test_explicit_cli_image_overrides_cr_image(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(image="registry.example/aiperf:from-yaml"),
            KubeOptions(image="registry.example/aiperf:from-cli"),
        )

        assert spec["image"] == "registry.example/aiperf:from-cli"


class TestCrCliOverlayParity:
    """CR input uses the same benchmark and nested deployment precedence."""

    def test_benchmark_cli_values_override_cr_yaml(self) -> None:
        spec, config = _build_cr_spec_and_config(
            _raw_cr(randomSeed=91),
            KubeOptions(image="nvcr.io/nvidia/aiperf:latest"),
            cli_config=CLIConfig(request_count=37, random_seed=0),
        )

        assert spec["benchmark"]["phases"][0]["requests"] == 37
        assert spec["randomSeed"] == 0
        assert config.benchmark.phases[0].requests == 37
        assert config.random_seed == 0

    def test_nested_cli_deployment_overlay_preserves_unrelated_cr_fields(self) -> None:
        raw = _raw_cr(
            resourceMode="none",
            podTemplate={
                "affinity": {"nodeAffinity": {"required": "yaml"}},
                "nodeSelector": {"region": "west"},
            },
        )

        spec, _ = _build_cr_spec_and_config(
            raw,
            KubeOptions(
                image="nvcr.io/nvidia/aiperf:latest",
                node_selector_cli=["gpu=true"],
            ),
        )

        assert spec["resourceMode"] == "none"
        assert spec["podTemplate"]["affinity"] == {"nodeAffinity": {"required": "yaml"}}
        assert spec["podTemplate"]["nodeSelector"] == {
            "region": "west",
            "gpu": "true",
        }

    def test_total_workers_uses_canonical_runtime_field(self) -> None:
        spec, _ = _build_cr_spec_and_config(
            _raw_cr(),
            KubeOptions(
                image="nvcr.io/nvidia/aiperf:latest",
                total_workers=6,
            ),
        )

        assert spec["benchmark"]["runtime"]["workers"] == 6


def test_memory_estimate_uses_stderr_not_manifest_stdout() -> None:
    spec, config = _build_cr_spec_and_config(
        _raw_cr(),
        KubeOptions(image="nvcr.io/nvidia/aiperf:latest"),
    )
    with (
        patch(
            "aiperf.kubernetes.memory_estimator.estimate_memory",
            return_value=MagicMock(),
        ),
        patch(
            "aiperf.kubernetes.memory_estimator.format_estimate",
            return_value="memory estimate",
        ),
        patch("aiperf.kubernetes.console.console.print") as stdout_print,
        patch("aiperf.kubernetes.console.stderr_console.print") as stderr_print,
    ):
        _print_memory_estimate(
            config,
            KubeOptions(image="nvcr.io/nvidia/aiperf:latest"),
            spec,
        )

    stdout_print.assert_not_called()
    stderr_print.assert_called_once_with("memory estimate", highlight=False)
