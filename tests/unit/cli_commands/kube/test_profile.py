# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube profile` flag wiring.

Covers the `--skip-endpoint-check` flag: verifies it forwards into
`deploy_via_operator`/`deploy_direct` kwargs and, in operator mode, lands
on the submitted CR spec as `skipEndpointCheck=True` so the operator's
`_check_endpoint_reachable` handler can honor it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.cli_commands.kube.profile import profile
from aiperf.cli_commands.kube.profile_deploy import deploy_via_operator
from aiperf.operator.models import AIPerfJobSpec


class _StubKubeOptions:
    """Minimal stand-in for KubeOptions used by the profile command."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.namespace: str | None = None
        self.kubeconfig: str | None = None
        self.kube_context: str | None = None
        self.image: str | None = "aiperf:latest"
        self.workers = 1


@pytest.mark.asyncio
async def test_profile_forwards_skip_endpoint_check_to_deploy_via_operator() -> None:
    """--skip-endpoint-check must arrive as a kwarg on deploy_via_operator."""
    kube_options = _StubKubeOptions()
    cli_model = object()
    fake_spec = {"benchmark": {"endpoint": {"url": "http://x"}}}
    fake_config = object()
    captured: dict[str, Any] = {}

    async def _capture_via_operator(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch(
            "aiperf.cli_commands.kube.profile._resolve_spec_and_name",
            return_value=(fake_spec, fake_config, "bench-1"),
        ),
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
        patch(
            "aiperf.cli_commands.kube.profile_deploy.operator_available",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "aiperf.cli_commands.kube.profile_deploy.deploy_via_operator",
            new=_capture_via_operator,
        ),
    ):
        await profile(
            cli_model=cli_model,
            kube_options=kube_options,
            skip_endpoint_check=True,
            dry_run=True,  # short-circuit operator_available so no cluster probe
        )

    assert captured["kwargs"].get("skip_endpoint_check") is True


@pytest.mark.asyncio
async def test_profile_forwards_skip_endpoint_check_to_deploy_direct() -> None:
    """--skip-endpoint-check must arrive as a kwarg on deploy_direct too."""
    kube_options = _StubKubeOptions()
    cli_model = object()
    fake_spec: dict[str, Any] = {}
    fake_config = object()
    captured: dict[str, Any] = {}

    async def _capture_direct(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch(
            "aiperf.cli_commands.kube.profile._resolve_spec_and_name",
            return_value=(fake_spec, fake_config, "bench-1"),
        ),
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
        patch(
            "aiperf.cli_commands.kube.profile_deploy_direct.deploy_direct",
            new=_capture_direct,
        ),
    ):
        await profile(
            cli_model=cli_model,
            kube_options=kube_options,
            skip_endpoint_check=True,
            dry_run=True,
            no_operator=True,  # force direct path
        )

    assert captured["kwargs"].get("skip_endpoint_check") is True


@pytest.mark.asyncio
async def test_deploy_via_operator_injects_skip_endpoint_check_into_cr() -> None:
    """When skip_endpoint_check=True, the submitted CR spec carries skipEndpointCheck=True."""
    kube_options = _StubKubeOptions()
    spec: dict[str, Any] = {"benchmark": {"endpoint": {"url": "http://x"}}}
    config = type(
        "C",
        (),
        {
            "endpoint": type("E", (), {"urls": []})(),
            "get_model_names": lambda self: ["m"],
        },
    )()

    captured_cr: dict[str, Any] = {}

    def _capture_print(*args: Any, **kwargs: Any) -> None:
        captured_cr["printed"] = args[0] if args else kwargs.get("data")

    with patch("aiperf.kubernetes.console.console") as mock_console:
        mock_console.print.side_effect = _capture_print
        await deploy_via_operator(
            spec,
            kube_options,
            config,
            "bench-1",
            "aiperf",
            dry_run=True,  # take the json-print branch, no cluster
            detach=False,
            no_wait=False,
            attach_port=0,
            skip_endpoint_check=True,
        )

    assert spec.get("skipEndpointCheck") is True


def test_aiperfjobspec_reads_skip_endpoint_check_from_crd() -> None:
    """AIPerfJobSpec.from_crd_spec must honor skipEndpointCheck from the raw CR."""
    crd_spec = {
        "image": "aiperf:latest",
        "skipEndpointCheck": True,
        "benchmark": {"endpoint": {"url": "http://x"}},
    }
    validated = AIPerfJobSpec.from_crd_spec(crd_spec)
    assert validated.skip_endpoint_check is True


def test_aiperfjobspec_skip_endpoint_check_defaults_false() -> None:
    """Absent skipEndpointCheck defaults to False (preserves prior behaviour)."""
    crd_spec = {
        "image": "aiperf:latest",
        "benchmark": {"endpoint": {"url": "http://x"}},
    }
    validated = AIPerfJobSpec.from_crd_spec(crd_spec)
    assert validated.skip_endpoint_check is False
