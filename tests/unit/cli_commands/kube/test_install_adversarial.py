# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the kube operator-install surface that exists today.

The branch does not expose ``aiperf kube install``, ``upgrade``, or
``uninstall`` modules. Operator lifecycle management is documented as direct
Helm usage, while the CLI install-adjacent paths are:
- ``aiperf kube generate --operator`` renders AIPerfJob/AIPerfSweep CRs for an
  already-installed operator.
- ``aiperf kube profile`` probes for the AIPerfJob CRD and falls back to direct
  manifests when the operator is absent.
- ``deploy/helm/aiperf-operator`` is the chart users install/upgrade/uninstall
  with Helm.
"""

from __future__ import annotations

import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import ruamel.yaml
import yaml

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube._app import app as kube_app
from aiperf.cli_commands.kube.profile_deploy import operator_available
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions

PROJECT_ROOT = Path(__file__).parents[4]
CHART_PATH = PROJECT_ROOT / "deploy" / "helm" / "aiperf-operator"

_VALID_BENCHMARK: dict[str, object] = {
    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
    "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 16,
            "prompts": {"isl": 64, "osl": 32},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 8,
            "concurrency": 2,
        }
    ],
}


# ---------------------------------------------------------------------------
# Command contract: no invented lifecycle commands
# ---------------------------------------------------------------------------


class TestKubeOperatorLifecycleSurface:
    """The CLI must not register unimplemented lifecycle commands."""

    @pytest.mark.parametrize("subcommand", ["install", "upgrade", "uninstall"])
    def test_helm_lifecycle_subcommands_are_not_registered(
        self, subcommand: str
    ) -> None:
        assert subcommand not in kube_app
        assert (
            importlib.util.find_spec(f"aiperf.cli_commands.kube.{subcommand}") is None
        )

    def test_existing_install_adjacent_commands_are_registered(self) -> None:
        for subcommand in ("generate", "profile", "preflight", "dashboard"):
            assert subcommand in kube_app


class TestHelmChartInstallContract:
    """Operator install/upgrade/uninstall is a Helm-chart workflow, not a kube CLI module."""

    def test_operator_chart_contains_installable_core_templates(self) -> None:
        expected_paths = [
            "Chart.yaml",
            "values.yaml",
            "values.schema.json",
            "templates/deployment.yaml",
            "templates/service.yaml",
            "templates/crd-aiperfjob.yaml",
            "templates/crd-aiperfsweep.yaml",
        ]
        missing = [path for path in expected_paths if not (CHART_PATH / path).exists()]
        assert missing == []


# ---------------------------------------------------------------------------
# Actual CLI install-adjacent behavior
# ---------------------------------------------------------------------------


def _write_config(path: Path) -> Path:
    path.write_text(yaml.safe_dump({"benchmark": _VALID_BENCHMARK}, sort_keys=False))
    return path


def _parse_single_yaml(stdout: str) -> dict[str, object]:
    parsed = ruamel.yaml.YAML(typ="safe").load(stdout)
    assert isinstance(parsed, dict)
    return parsed


def _kube_options(**overrides: object) -> KubeOptions:
    data: dict[str, object] = {"image": "nvcr.io/nvidia/aiperf:ci"}
    data.update(overrides)
    return KubeOptions.model_validate(data)


class TestGenerateOperatorManifests:
    """``aiperf kube generate --operator`` emits CRs consumed by the installed operator."""

    @pytest.mark.asyncio
    async def test_generate_operator_outputs_aiperfjob_cr_not_helm_invocation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_file = _write_config(tmp_path / "operator-job.yaml")

        with patch("aiperf.cli_commands.kube.generate._print_memory_estimate"):
            await generate_cmd.generate(
                cli_config=CLIConfig(config_file=config_file),
                kube_options=_kube_options(
                    name="operator-job",
                    namespace="aiperf-ci",
                    kube_context="kind-aiperf-ci",
                    kubeconfig="/tmp/aiperf-ci.kubeconfig",
                ),
                operator=True,
            )

        cr = _parse_single_yaml(capsys.readouterr().out)
        assert cr["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert cr["kind"] == "AIPerfJob"
        assert cr["metadata"] == {"name": "operator-job", "namespace": "aiperf-ci"}
        assert cr["spec"]["image"] == "nvcr.io/nvidia/aiperf:ci"
        assert "helm" not in yaml.safe_dump(cr).lower()
        assert "kubeContext" not in cr["spec"]
        assert "kubeconfig" not in cr["spec"]


class TestProfileOperatorDetection:
    """``aiperf kube profile`` selects operator mode by probing the installed CRD."""

    @pytest.mark.asyncio
    async def test_operator_available_probes_crd_with_selected_cluster_options(
        self,
    ) -> None:
        api = MagicMock()
        seen_client_kwargs: dict[str, object] = {}

        @asynccontextmanager
        async def _fake_client(**kwargs: object):
            seen_client_kwargs.update(kwargs)
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            return_value=MagicMock()
        )

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api", return_value=fake_apiext
            ),
        ):
            available = await operator_available(
                _kube_options(
                    kubeconfig="/tmp/aiperf-ci.kubeconfig",
                    kube_context="kind-aiperf-ci",
                )
            )

        assert available is True
        assert seen_client_kwargs == {
            "kubeconfig": "/tmp/aiperf-ci.kubeconfig",
            "context": "kind-aiperf-ci",
        }
        fake_apiext.read_custom_resource_definition.assert_awaited_once_with(
            "aiperfjobs.aiperf.nvidia.com"
        )
