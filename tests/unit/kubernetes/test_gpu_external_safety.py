# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safety contracts for live external GPU test runs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.kubernetes.gpu.conftest import _OPTIONS, _release_gpu, _resolve_settings
from tests.kubernetes.helpers.benchmark import BenchmarkDeployer


def _config() -> SimpleNamespace:
    """Build a pytest-config-shaped object with no explicit CLI options."""
    return SimpleNamespace(option=SimpleNamespace())


def _set_safe_external_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure every mutable GPU test namespace for the external-cluster guard."""
    monkeypatch.setenv("GPU_TEST_CONTEXT", "external")
    monkeypatch.setenv("GPU_TEST_EXTERNAL_EXISTING_OPERATOR", "1")
    for suffix in (
        "BENCHMARK_NAMESPACE",
        "VLLM_NAMESPACE",
        "TRTLLM_NAMESPACE",
        "SGLANG_NAMESPACE",
        "DYNAMO_NAMESPACE",
    ):
        monkeypatch.setenv(f"GPU_TEST_{suffix}", "acasagrande-gpu-e2e")


def test_resolve_settings_external_cluster_requires_user_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External execution must reject the historic shared GPU namespaces."""
    monkeypatch.setenv("GPU_TEST_CONTEXT", "external")
    monkeypatch.setenv("GPU_TEST_EXTERNAL_EXISTING_OPERATOR", "1")

    with pytest.raises(pytest.UsageError, match="acasagrande-"):
        _resolve_settings(_config())


def test_resolve_settings_external_cluster_accepts_explicit_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External execution accepts only the fully explicit user-owned scope."""
    _set_safe_external_environment(monkeypatch)

    settings = _resolve_settings(_config())

    assert settings.benchmark_namespace == "acasagrande-gpu-e2e"
    assert settings.vllm_namespace == "acasagrande-gpu-e2e"
    assert settings.dynamo_namespace == "acasagrande-gpu-e2e"
    assert settings.external_existing_operator is True


@pytest.mark.asyncio
async def test_release_gpu_external_cluster_never_lists_or_deletes_namespaces() -> None:
    """The historical GPU release loop must be inert against a shared cluster."""

    class ExternalKubectl:
        context = "external"

        async def run(self, *args: str, **kwargs: object) -> None:
            raise AssertionError(f"unexpected namespace lookup: {args}, {kwargs}")

        async def delete_namespace(self, *args: str, **kwargs: object) -> None:
            raise AssertionError(f"unexpected namespace deletion: {args}, {kwargs}")

    await _release_gpu(ExternalKubectl(), "acasagrande-gpu-e2e")


def test_gpu_option_surface_contains_all_external_namespace_controls() -> None:
    """The safe external invocation does not rely on hidden fixture defaults."""
    flags = {flag for flag, *_rest in _OPTIONS}

    assert {
        "--gpu-benchmark-namespace",
        "--gpu-vllm-namespace",
        "--gpu-trtllm-namespace",
        "--gpu-sglang-namespace",
        "--gpu-dynamo-namespace",
        "--gpu-external-existing-operator",
    } <= flags


@pytest.mark.asyncio
async def test_pull_secret_copy_uses_only_explicit_source_namespace() -> None:
    """A safe external run must not enumerate or read other users' namespaces."""

    calls: list[tuple[str, ...]] = []

    class Kubectl:
        async def run(self, *args: str, **kwargs: object) -> SimpleNamespace:
            calls.append(args)
            if args[:3] == ("get", "secret", "nvcr-pull") and args[-1] == "target":
                return SimpleNamespace(returncode=1, stdout="")
            assert args == (
                "get",
                "secret",
                "nvcr-pull",
                "-n",
                "acasagrande-aiperf-bench",
                "-o",
                "yaml",
            )
            return SimpleNamespace(
                returncode=0, stdout="apiVersion: v1\nmetadata:\n  name: nvcr-pull\n"
            )

        async def apply(self, manifest: str, namespace: str) -> None:
            assert namespace == "target"
            assert "namespace:" not in manifest

    deployer = BenchmarkDeployer(
        kubectl=Kubectl(),  # type: ignore[arg-type]
        project_root=Path.cwd(),
        default_image_pull_secret_source_namespace="acasagrande-aiperf-bench",
    )

    await deployer._ensure_pull_secrets_in_namespace("target", ["nvcr-pull"])

    assert (
        "get",
        "namespaces",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ) not in calls
