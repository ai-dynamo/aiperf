# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube profile` deploy helpers.

`test_profile.py` already covers the top-level `profile()` command's
`--skip-endpoint-check` wiring. This file targets the helpers that
`profile` delegates to:
    - `profile._try_load_aiperfjob_cr`   — CR-vs-plain-config detection
    - `profile.generate_benchmark_name`  — deterministic DNS-safe name
    - `profile_deploy._build_cr`         — CR envelope construction
    - `profile_deploy.operator_available`— CRD probe (404 vs other)
    - `profile_deploy.wait_or_detach`    — interactive/detach split
    - `profile_deploy_direct._apply_manifest` — kind-dispatch table
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.profile import (
    _try_load_aiperfjob_cr,
    generate_benchmark_name,
)
from aiperf.cli_commands.kube.profile_deploy import (
    _build_cr,
    operator_available,
    wait_or_detach,
)
from aiperf.cli_commands.kube.profile_deploy_direct import _apply_manifest
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

# =============================================================================
# _try_load_aiperfjob_cr
# =============================================================================


class TestTryLoadAiperfjobCr:
    """Tests for the CR detection heuristic."""

    def test_valid_aiperfjob_cr_returns_dict(self, tmp_path) -> None:
        """A well-formed AIPerfJob YAML file is recognised."""
        cr_file = tmp_path / "job.yaml"
        cr_file.write_text(
            "apiVersion: aiperf.nvidia.com/v1alpha1\nkind: AIPerfJob\nspec: {}\n"
        )
        raw = _try_load_aiperfjob_cr(cr_file)
        assert raw is not None
        assert raw["kind"] == "AIPerfJob"

    @pytest.mark.parametrize(
        "content",
        [
            param("not-valid: yaml: [[[", id="malformed-yaml"),
            param("kind: Pod\napiVersion: v1\n", id="wrong-kind"),
            param("kind: AIPerfJob\napiVersion: other.io/v1\n", id="wrong-api-version"),
            param("just-a-string", id="not-a-mapping"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    def test_non_cr_returns_none(self, tmp_path, content: str) -> None:
        """Non-AIPerfJob / malformed YAML paths return None."""
        cr_file = tmp_path / "other.yaml"
        cr_file.write_text(content)
        assert _try_load_aiperfjob_cr(cr_file) is None


# =============================================================================
# generate_benchmark_name
# =============================================================================


class TestGenerateBenchmarkName:
    """Tests for the benchmark-name generator."""

    def _stub_config(
        self,
        *,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        endpoint_type: str = "chat",
        phase_type: str = "throughput",
    ) -> Any:
        """Build a stub config with the three fields the helper reads."""
        phase = MagicMock()
        phase.type = phase_type
        config = MagicMock()
        config.get_model_names.return_value = [model]
        config.endpoint.type = endpoint_type
        config.phases.values.return_value = [phase]
        # first phase iteration uses values() iterator
        config.phases.__iter__ = lambda self_: iter(["default"])
        return config

    def test_basic_name(self) -> None:
        """Assembles model + endpoint-type + phase-type into a DNS-safe name."""
        config = self._stub_config()
        name = generate_benchmark_name(config)
        # Dots in the model tag are replaced with hyphens by the sanitizer
        assert name == "llama-3-1-8b-instruct-chat-throughput"
        # DNS-label shape: lower, hyphen-only, <=40
        assert len(name) <= 40
        assert all(c.islower() or c.isdigit() or c == "-" for c in name)

    def test_truncates_to_40(self) -> None:
        """Very long model names are truncated at 40 chars."""
        config = self._stub_config(model="x" * 100)
        assert len(generate_benchmark_name(config)) <= 40

    def test_strips_leading_and_trailing_hyphens(self) -> None:
        """Leading/trailing invalid chars are sanitized to hyphens then stripped."""
        config = self._stub_config(model="--weird--")
        out = generate_benchmark_name(config)
        assert not out.startswith("-")
        assert not out.endswith("-")


# =============================================================================
# _build_cr
# =============================================================================


class TestBuildCr:
    """Tests for the CR envelope builder."""

    def test_build_cr_shape(self) -> None:
        """`_build_cr` wraps spec with correct apiVersion/kind/metadata."""
        cr = _build_cr("my-bench", "ns-1", {"benchmark": {"foo": "bar"}})
        assert cr["apiVersion"] == AIPERF_API_VERSION
        assert cr["kind"] == "AIPerfJob"
        assert cr["metadata"] == {"name": "my-bench", "namespace": "ns-1"}
        assert cr["spec"] == {"benchmark": {"foo": "bar"}}


# =============================================================================
# operator_available
# =============================================================================


class _StubKubeOpts:
    """Minimal KubeOptions-shaped stub for operator_available()."""

    def __init__(self) -> None:
        self.kubeconfig: str | None = None
        self.kube_context: str | None = None


class TestOperatorAvailable:
    """Tests for the CRD-existence probe."""

    async def test_returns_true_when_crd_exists(self, capsys) -> None:
        """If read_custom_resource_definition succeeds, operator mode is selected."""
        from contextlib import asynccontextmanager

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            return_value=MagicMock()
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
        ):
            assert await operator_available(_StubKubeOpts()) is True

        assert "operator mode" in capsys.readouterr().out

    async def test_returns_false_on_404(self, capsys) -> None:
        """404 from the API -> direct mode (no operator)."""
        from contextlib import asynccontextmanager

        from kubernetes_asyncio.client.exceptions import ApiException

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=404, reason="NotFound")
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
        ):
            assert await operator_available(_StubKubeOpts()) is False

        assert "no operator" in capsys.readouterr().out

    async def test_returns_false_on_unexpected_exception(self, capsys) -> None:
        """Any unrecognised error falls back to direct mode with a labeled message."""
        from contextlib import asynccontextmanager

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            side_effect=RuntimeError("network down")
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
        ):
            assert await operator_available(_StubKubeOpts()) is False

        out = capsys.readouterr().out
        assert "RuntimeError" in out
        assert "network down" in out


# =============================================================================
# wait_or_detach
# =============================================================================


class TestWaitOrDetach:
    """Tests for the post-submit interactive/detach dispatcher."""

    async def test_detach_flag_prints_info_and_returns(self, capsys) -> None:
        """`detach=True` short-circuits the attach workflow."""
        opts = _StubKubeOpts()
        opts.name = "my-bench"

        with patch("sys.stdout.isatty", return_value=True):
            await wait_or_detach(
                "my-bench",
                "ns",
                opts,
                detach=True,
                no_wait=False,
                attach_port=0,
                hint="Retrieve results: aiperf kube results",
            )

        out = capsys.readouterr().out
        assert "my-bench" in out
        assert "Retrieve results" in out

    async def test_non_interactive_forces_detach_with_warning(self, capsys) -> None:
        """Non-TTY stdout auto-enables detach mode and emits a warning."""
        opts = _StubKubeOpts()
        opts.name = "bench-ci"

        attach_mock = AsyncMock()
        with (
            patch("sys.stdout.isatty", return_value=False),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=attach_mock,
            ),
        ):
            await wait_or_detach(
                "bench-ci",
                "ns",
                opts,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        attach_mock.assert_not_awaited()
        assert "Non-interactive" in capsys.readouterr().out

    async def test_interactive_calls_auto_attach_workflow(self) -> None:
        """Interactive + `detach=False` invokes the attach workflow."""
        opts = _StubKubeOpts()
        opts.name = "bench"

        attach_mock = AsyncMock()
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=attach_mock,
            ),
        ):
            await wait_or_detach(
                "bench",
                "ns",
                opts,
                detach=False,
                no_wait=True,
                attach_port=7777,
            )

        attach_mock.assert_awaited_once()

    async def test_keyboard_interrupt_prints_interrupt_info(self, capsys) -> None:
        """Ctrl-C during attach is caught and `print_interrupt_info` fires."""
        opts = _StubKubeOpts()
        opts.name = "bench"

        with (
            patch("sys.stdout.isatty", return_value=True),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=AsyncMock(side_effect=KeyboardInterrupt),
            ),
        ):
            await wait_or_detach(
                "bench",
                "ns",
                opts,
                detach=False,
                no_wait=False,
                attach_port=0,
                hint="a-hint",
            )

        out = capsys.readouterr().out
        # print_interrupt_info prints job info; hint follows
        assert "a-hint" in out


# =============================================================================
# _apply_manifest
# =============================================================================


class TestApplyManifest:
    """Tests for the kind->api-call dispatch table in direct-mode deploy."""

    async def _run(self, kind: str) -> tuple[str | None, dict]:
        """Call _apply_manifest and return (label, which-client-method-was-called)."""
        manifest = {
            "kind": kind,
            "metadata": {"name": "res1", "namespace": "ns1"},
        }
        core = MagicMock()
        core.create_namespace = AsyncMock()
        core.create_namespaced_config_map = AsyncMock()
        rbac = MagicMock()
        rbac.create_namespaced_role = AsyncMock()
        rbac.create_namespaced_role_binding = AsyncMock()
        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock()

        label = await _apply_manifest(
            manifest, core=core, rbac=rbac, custom=custom, default_namespace="ns1"
        )
        return label, {
            "create_namespace": core.create_namespace.await_count,
            "create_configmap": core.create_namespaced_config_map.await_count,
            "create_role": rbac.create_namespaced_role.await_count,
            "create_rolebinding": rbac.create_namespaced_role_binding.await_count,
            "create_custom": custom.create_namespaced_custom_object.await_count,
        }

    @pytest.mark.parametrize(
        "kind, expected_call",
        [
            param("Namespace", "create_namespace", id="namespace"),
            param("ConfigMap", "create_configmap", id="configmap"),
            param("Role", "create_role", id="role"),
            param("RoleBinding", "create_rolebinding", id="rolebinding"),
            param("JobSet", "create_custom", id="jobset"),
        ],
    )  # fmt: skip
    async def test_kind_dispatches_to_correct_api_call(
        self, kind: str, expected_call: str
    ) -> None:
        """Each known kind routes to its matching kubernetes_asyncio call."""
        label, counts = await self._run(kind)
        assert label == f"{kind}/res1"
        assert counts[expected_call] == 1
        # Exactly one client method was called
        assert sum(counts.values()) == 1

    async def test_unknown_kind_returns_none(self) -> None:
        """Unrecognised kinds return None (no API call)."""
        label, counts = await self._run("Deployment")
        assert label is None
        assert sum(counts.values()) == 0
