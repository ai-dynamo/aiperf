# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the `aiperf kube generate` CLI command.

Covers the CR-vs-raw mutual exclusion, `--operator` / `--no-operator`
dispatch, and the spec+name resolution wiring. Full end-to-end manifest
generation is exercised in `tests/kubernetes/test_cli_commands.py`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.kube.generate import (
    AIPERF_KIND,
    _resolve_spec_and_name,
    generate,
)
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION


class _StubKubeOptions:
    """Minimal stand-in for KubeOptions used by the generate command."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.namespace: str | None = None
        self.kubeconfig: str | None = None
        self.kube_context: str | None = None
        self.image: str | None = "aiperf:latest"
        self.workers = 1
        self.model_fields_set = set()

    def to_deployment_config(self) -> Any:  # pragma: no cover - not used in happy paths
        raise NotImplementedError


class TestGenerateMutualExclusion:
    """Tests for --operator / --no-operator required-exactly-one validation."""

    async def test_neither_operator_nor_no_operator_exits(self) -> None:
        """Must specify one of --operator or --no-operator."""
        with pytest.raises(SystemExit, match="Specify --operator"):
            await generate(
                cli_model=object(),
                kube_options=_StubKubeOptions(),
                operator=False,
                no_operator=False,
            )

    async def test_both_operator_and_no_operator_exits(self) -> None:
        """Cannot pass both --operator and --no-operator simultaneously."""
        with pytest.raises(SystemExit, match="Cannot use both"):
            await generate(
                cli_model=object(),
                kube_options=_StubKubeOptions(),
                operator=True,
                no_operator=True,
            )


class TestGenerateOperatorMode:
    """Tests for `--operator` CR-generation path."""

    async def test_operator_mode_writes_aiperfjob_cr_to_stdout(self, capsys) -> None:
        """In --operator mode a single AIPerfJob CR is emitted to stdout."""
        kube_options = _StubKubeOptions()
        fake_config = MagicMock()
        fake_spec = {"benchmark": {"endpoint": {"url": "http://x"}}}

        with (
            patch(
                "aiperf.cli_commands.kube.generate._resolve_spec_and_name",
                return_value=(fake_spec, fake_config, "bench-xyz"),
            ),
            patch(
                "aiperf.cli_commands.kube.generate._print_memory_estimate",
            ),
        ):
            await generate(
                cli_model=object(),
                kube_options=kube_options,
                operator=True,
                no_operator=False,
            )

        out = capsys.readouterr().out
        # CR structure keys
        assert "apiVersion" in out
        assert AIPERF_API_VERSION in out
        assert f"kind: {AIPERF_KIND}" in out
        assert "bench-xyz" in out

    async def test_operator_mode_uses_default_namespace_when_unset(
        self, capsys
    ) -> None:
        """Default benchmark namespace is applied when kube_options.namespace=None."""
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        kube_options = _StubKubeOptions()
        fake_config = MagicMock()
        fake_spec: dict[str, Any] = {}

        with (
            patch(
                "aiperf.cli_commands.kube.generate._resolve_spec_and_name",
                return_value=(fake_spec, fake_config, "bench-name"),
            ),
            patch(
                "aiperf.cli_commands.kube.generate._print_memory_estimate",
            ),
        ):
            await generate(
                cli_model=object(),
                kube_options=kube_options,
                operator=True,
                no_operator=False,
            )

        assert DEFAULT_BENCHMARK_NAMESPACE in capsys.readouterr().out


class TestGenerateNoOperatorMode:
    """Tests for `--no-operator` raw-manifests path."""

    async def test_no_operator_mode_dumps_raw_manifests(self) -> None:
        """In --no-operator mode, `_dump_raw_manifests` is called and CR path skipped."""
        kube_options = _StubKubeOptions()
        fake_config = MagicMock()
        fake_spec: dict[str, Any] = {}

        with (
            patch(
                "aiperf.cli_commands.kube.generate._resolve_spec_and_name",
                return_value=(fake_spec, fake_config, "bench-name"),
            ),
            patch(
                "aiperf.cli_commands.kube.generate._dump_raw_manifests",
                return_value=fake_config,
            ) as mock_dump,
            patch(
                "aiperf.cli_commands.kube.generate._print_memory_estimate",
            ) as mock_mem,
        ):
            await generate(
                cli_model=object(),
                kube_options=kube_options,
                operator=False,
                no_operator=True,
            )

        mock_dump.assert_called_once()
        mock_mem.assert_called_once()


class TestResolveSpecAndName:
    """Tests for the `_resolve_spec_and_name` CR-or-flags dispatch."""

    def test_flag_format_uses_cli_converter_when_no_cr_file(self) -> None:
        """No config_file -> build via CLI flags, name via generate_benchmark_name."""
        kube_options = _StubKubeOptions()
        cli_model = MagicMock()
        cli_model.config_file = None
        fake_config = MagicMock()
        fake_spec = {"benchmark": {}}

        with (
            patch(
                "aiperf.cli_commands.kube.profile._try_load_aiperfjob_cr",
                return_value=None,
            ),
            patch(
                "aiperf.cli_commands.kube.profile._resolve_config",
                return_value=fake_config,
            ),
            patch.object(
                type(kube_options), "to_crd_spec", create=True, return_value=fake_spec
            ),
            patch(
                "aiperf.cli_commands.kube.profile.generate_benchmark_name",
                return_value="gen-bench",
            ),
        ):
            spec, config, name = _resolve_spec_and_name(cli_model, kube_options)

        assert spec == fake_spec
        assert config is fake_config
        assert name == "gen-bench"

    def test_cr_format_uses_cr_name_when_present(self) -> None:
        """If config_file is an AIPerfJob CR, use its metadata.name when kube_options.name is None."""
        kube_options = _StubKubeOptions()
        cli_model = MagicMock()
        cli_model.config_file = MagicMock()  # truthy
        fake_config = MagicMock()
        fake_spec = {"benchmark": {}}
        cr_raw = {"metadata": {"name": "cr-bench"}, "spec": {}}

        with (
            patch(
                "aiperf.cli_commands.kube.profile._try_load_aiperfjob_cr",
                return_value=cr_raw,
            ),
            patch(
                "aiperf.cli_commands.kube.profile._build_cr_spec_and_config",
                return_value=(fake_spec, fake_config),
            ),
            patch(
                "aiperf.cli_commands.kube.profile.generate_benchmark_name",
                return_value="fallback",
            ),
        ):
            _, _, name = _resolve_spec_and_name(cli_model, kube_options)

        assert name == "cr-bench"
