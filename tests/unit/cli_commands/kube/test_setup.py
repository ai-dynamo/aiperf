# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube setup` -- the cluster bootstrap path (A19).

Lost at the March squash-transplant, leaving no way to prepare a cluster from
the CLI even though every install helper it needs (JobSet URL resolution, the
Helm chart, namespace constants) survived. Must be idempotent: each step
checks first, so re-running against a prepared cluster reports rather than
fails.
"""

from __future__ import annotations

import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.cli_commands.kube.setup import _default_chart_path, setup
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.subproc import CommandResult


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "boom") -> CommandResult:
    return CommandResult(returncode=1, stdout="", stderr=stderr)


@asynccontextmanager
async def _fake_client(**_: Any):
    yield MagicMock()


def _cluster(*, jobset: bool, namespaces_exist: bool) -> tuple[MagicMock, MagicMock]:
    custom = MagicMock(
        list_cluster_custom_object=AsyncMock(
            return_value={"items": []} if jobset else None,
            side_effect=None if jobset else ApiException(status=404),
        )
    )
    core = MagicMock(
        read_namespace=AsyncMock(
            side_effect=None if namespaces_exist else ApiException(status=404)
        )
    )
    return custom, core


def _patched(custom: MagicMock, core: MagicMock, run: AsyncMock):
    return (
        patch("aiperf.kubernetes.client.k8s_client", _fake_client),
        patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
        patch("kubernetes_asyncio.client.CoreV1Api", return_value=core),
        patch("aiperf.kubernetes.subproc.run_command", run),
        patch(
            "aiperf.kubernetes.jobset_urls.get_latest_jobset_version",
            AsyncMock(return_value="v0.9.9"),
        ),
    )


async def _run_setup(custom, core, run, **kwargs) -> None:
    a, b, c, d, e = _patched(custom, core, run)
    with a, b, c, d, e:
        await setup(**kwargs)


class TestKubeSetup:
    @pytest.mark.asyncio
    async def test_installs_jobset_when_absent(self) -> None:
        custom, core = _cluster(jobset=False, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, skip_operator=True)
        applied = [c.args[0] for c in run.await_args_list]
        assert any("apply" in cmd and "v0.9.9" in " ".join(cmd) for cmd in applied)

    @pytest.mark.asyncio
    async def test_skips_jobset_when_present(self) -> None:
        """Idempotence: a prepared cluster must not be re-applied."""
        custom, core = _cluster(jobset=True, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, skip_operator=True)
        assert not any("apply" in c.args[0] for c in run.await_args_list)

    @pytest.mark.asyncio
    async def test_explicit_version_wins_over_lookup(self) -> None:
        custom, core = _cluster(jobset=False, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, skip_operator=True, jobset_version="v0.5.2")
        joined = " ".join(" ".join(c.args[0]) for c in run.await_args_list)
        assert "v0.5.2" in joined and "v0.9.9" not in joined

    @pytest.mark.asyncio
    async def test_creates_missing_namespaces(self) -> None:
        custom, core = _cluster(jobset=True, namespaces_exist=False)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, skip_operator=True)
        created = [c.args[0][-1] for c in run.await_args_list if "create" in c.args[0]]
        assert "aiperf-system" in created

    @pytest.mark.asyncio
    async def test_dry_run_changes_nothing(self) -> None:
        custom, core = _cluster(jobset=False, namespaces_exist=False)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, dry_run=True)
        run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_jobset_install_failure_is_fatal(self) -> None:
        """A half-prepared cluster must not be reported as ready."""
        custom, core = _cluster(jobset=False, namespaces_exist=True)
        run = AsyncMock(return_value=_fail("no such host"))
        with pytest.raises(SystemExit):
            await _run_setup(custom, core, run, skip_operator=True)

    @pytest.mark.asyncio
    async def test_missing_chart_is_fatal(self) -> None:
        """Setup must not report success after skipping the operator install."""
        custom, core = _cluster(jobset=True, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        with pytest.raises(SystemExit) as exc_info:
            await _run_setup(custom, core, run, chart=Path("/nonexistent/chart"))
        assert exc_info.value.code == 1
        assert not any("helm" in c.args[0][0] for c in run.await_args_list)

    @pytest.mark.asyncio
    async def test_installs_operator_via_helm(self, tmp_path: Path) -> None:
        custom, core = _cluster(jobset=True, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        await _run_setup(custom, core, run, chart=tmp_path)
        helm_calls = [c.args[0] for c in run.await_args_list if c.args[0][0] == "helm"]
        assert helm_calls, "operator install never ran"
        helm = helm_calls[0]
        assert "upgrade" in helm and "--install" in helm
        assert "benchmarkNamespace.name=aiperf-benchmarks" in helm
        assert "benchmarkNamespace.create=false" in helm

    @pytest.mark.asyncio
    async def test_custom_benchmark_namespace_propagates_to_helm(
        self, tmp_path: Path
    ) -> None:
        custom, core = _cluster(jobset=True, namespaces_exist=True)
        run = AsyncMock(return_value=_ok())
        await _run_setup(
            custom,
            core,
            run,
            chart=tmp_path,
            manage_options=KubeManageOptions(namespace="team-benchmarks"),
        )

        helm = next(c.args[0] for c in run.await_args_list if c.args[0][0] == "helm")
        assert "benchmarkNamespace.name=team-benchmarks" in helm
        assert "benchmarkNamespace.create=false" in helm

    def test_default_chart_prefers_packaged_wheel_data(self, tmp_path: Path) -> None:
        packaged = tmp_path / "site-packages" / "aiperf" / "kubernetes" / "helm"
        packaged.mkdir(parents=True)
        source = tmp_path / "checkout" / "deploy" / "helm"
        source.mkdir(parents=True)

        with (
            patch("aiperf.cli_commands.kube.setup.PACKAGED_CHART_PATH", packaged),
            patch("aiperf.cli_commands.kube.setup.SOURCE_CHART_PATH", source),
        ):
            assert _default_chart_path() == packaged

    def test_default_chart_falls_back_to_source_checkout(self, tmp_path: Path) -> None:
        packaged = tmp_path / "missing-packaged-chart"
        source = tmp_path / "checkout" / "deploy" / "helm"
        source.mkdir(parents=True)

        with (
            patch("aiperf.cli_commands.kube.setup.PACKAGED_CHART_PATH", packaged),
            patch("aiperf.cli_commands.kube.setup.SOURCE_CHART_PATH", source),
        ):
            assert _default_chart_path() == source

    def test_wheel_build_includes_operator_chart(self) -> None:
        """The installed CLI needs the chart outside a source checkout."""
        project_root = Path(__file__).resolve().parents[4]
        config = tomllib.loads((project_root / "pyproject.toml").read_text())

        force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        assert force_include["deploy/helm/aiperf-operator"] == (
            "aiperf/kubernetes/helm/aiperf-operator"
        )

    def test_docker_builds_copy_operator_chart(self) -> None:
        """Container package builds must supply every forced chart input."""
        project_root = Path(__file__).resolve().parents[4]
        dockerfile = (project_root / "Dockerfile").read_text()
        mock_dockerfile = (
            project_root / "dev" / "deploy" / "Dockerfile.mock-server"
        ).read_text()

        assert (
            "COPY deploy/helm/aiperf-operator/ "
            "/workspace/deploy/helm/aiperf-operator/" in dockerfile
        )
        assert (
            "COPY deploy/helm/aiperf-operator/ ./deploy/helm/aiperf-operator/"
            in mock_dockerfile
        )

    def test_runtime_image_uses_the_locked_botorch_dependency_set(self) -> None:
        """Runtime builds must not re-resolve the adaptive-search dependencies."""
        project_root = Path(__file__).resolve().parents[4]
        dockerfile = (project_root / "Dockerfile").read_text()
        dockerignore = (project_root / ".dockerignore").read_text()
        gitignore = (project_root / ".gitignore").read_text()

        assert (project_root / "uv.lock").is_file()
        assert "uv.lock" not in dockerignore
        assert "uv.lock" not in gitignore
        assert "COPY pyproject.toml uv.lock ." in dockerfile
        assert (
            "uv sync --active --locked --no-install-project --no-default-groups "
            "--extra botorch" in dockerfile
        )
        assert (
            'uv pip install --no-deps "aiperf[botorch] @ file://${WHEEL}"' in dockerfile
        )

    def test_runtime_lock_uses_cpu_torch_and_preserves_setuptools(self) -> None:
        """The runtime dependency closure must pass its license and metadata gates."""
        project_root = Path(__file__).resolve().parents[4]
        dockerfile = (project_root / "Dockerfile").read_text()
        project = tomllib.loads((project_root / "pyproject.toml").read_text())
        lock = tomllib.loads((project_root / "uv.lock").read_text())
        packages = {package["name"]: package for package in lock["package"]}

        assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
        assert {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        } in project["tool"]["uv"]["index"]
        assert packages["torch"]["source"] == {
            "registry": "https://download.pytorch.org/whl/cpu"
        }
        assert "setuptools" in packages
        assert {name for name in packages if name.startswith("cuda-")} == set()
        assert {name for name in packages if name.startswith("nvidia-")} == {
            "nvidia-ml-py"
        }
        assert "uv pip uninstall setuptools" not in dockerfile
