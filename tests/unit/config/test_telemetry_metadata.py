# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for plugin-metadata-driven GPU telemetry classification.

Proves the refactor's premise: classifier, conflict check, import probe, and
the "Invalid GPU telemetry item" error message all derive from
``GPUTelemetryCollectorMetadata`` on the plugin rather than hard-coded
collector names in either ``_converter_telemetry`` or ``GpuTelemetryConfig``.
"""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import EndpointType, GPUTelemetryCollectorType
from tests.harness import mock_plugin


def _make_aiperf_config(**overrides):
    """Build a minimal CLIConfig and convert to AIPerfConfig."""
    cli = CLIConfig(
        model_names=["llama"],
        url="http://localhost:8000",
        endpoint_type=EndpointType.CHAT,
        request_count=10,
        concurrency=1,
        **overrides,
    )
    return convert_cli_to_aiperf(cli)


def test_local_collector_discovered_dynamically_from_plugin_metadata() -> None:
    """A runtime-registered local collector flows through --gpu-telemetry end-to-end.

    Registers a fake local collector via ``mock_plugin``, extends the dynamic
    enum to mirror the registry, then walks the behaviors that should fall
    out for free from the plugin-metadata-driven design:

    1. Selection — the new keyword resolves to the new enum member.
    2. Conflict detection — mixing two local collectors raises.
    3. Local-vs-URL guardrail — local + URLs raises.
    4. Invalid-item error message advertises the new keyword.
    """
    fake_name = "fake_local_gpu"
    fake_enum_member = "FAKE_LOCAL_GPU"

    class FakeLocalCollector:
        """Placeholder class - never instantiated; only the registration matters."""

        @classmethod
        def validate_environment(cls) -> None:
            """No-op: the fake collector has no native binding to probe."""

    # The enum is built at import time, so runtime registration via
    # mock_plugin alone is not visible to ``for member in
    # GPUTelemetryCollectorType``. Mirror the registry into the enum and
    # tear it back down afterwards.
    GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
    try:
        with mock_plugin(
            "gpu_telemetry_collector",
            fake_name,
            FakeLocalCollector,
            metadata={"is_local": True},
        ):
            # 1. Selection: keyword resolves to the new enum member.
            config = _make_aiperf_config(gpu_telemetry=[fake_name])
            assert (
                config.benchmark.gpu_telemetry.collector
                == GPUTelemetryCollectorType(fake_name)
            )

            # 2. Conflict detection generalizes to the new collector.
            sys.modules["pynvml"] = type(sys)("pynvml")
            try:
                with pytest.raises(
                    (ValidationError, ValueError),
                    match="Conflicting local GPU telemetry collectors",
                ):
                    _make_aiperf_config(gpu_telemetry=["pynvml", fake_name])
            finally:
                sys.modules.pop("pynvml", None)

            # 3. Local-vs-URL guardrail generalizes to the new collector.
            with pytest.raises(
                (ValidationError, ValueError),
                match=f"Cannot use {fake_name} with DCGM URLs",
            ):
                _make_aiperf_config(gpu_telemetry=[fake_name, "http://localhost:9400"])

            # 4. Invalid-item error message advertises the new keyword.
            with pytest.raises(
                (ValidationError, ValueError),
                match=r"Invalid GPU telemetry item.*not_a_real_keyword",
            ) as exc_info:
                _make_aiperf_config(gpu_telemetry=["not_a_real_keyword"])
            assert f"'{fake_name}'" in str(exc_info.value)
    finally:
        GPUTelemetryCollectorType._extensions.pop(fake_enum_member, None)
