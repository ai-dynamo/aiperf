# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the network-latency CLI converter and NetworkLatencyConfig model.

Covers every branch of ``build_network_latency`` (mean / automatic / mutex /
disabled), the ``NetworkLatencyConfig`` Field constraints and default factory,
and the ``should_probe`` property.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.environment import Environment
from aiperf.config.flags._converter_telemetry import build_network_latency
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.network_latency import NetworkLatencyConfig


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


class TestBuildNetworkLatency:
    """Branch coverage for the CLIConfig -> network-latency dict converter."""

    def test_neither_flag_disabled(self) -> None:
        assert build_network_latency(_make_cli()) == {"enabled": False}

    def test_automatic_only_enables(self) -> None:
        cli = _make_cli(network_latency_automatic=True)
        assert build_network_latency(cli) == {"enabled": True}

    def test_automatic_with_ping_interval_sets_interval(self) -> None:
        cli = _make_cli(
            network_latency_automatic=True, network_latency_ping_interval=0.25
        )
        assert build_network_latency(cli) == {
            "enabled": True,
            "ping_interval": 0.25,
        }

    def test_mean_only_enables_with_rtt(self) -> None:
        cli = _make_cli(network_latency_mean=5.0)
        assert build_network_latency(cli) == {
            "enabled": True,
            "mean_ms": 5.0,
        }

    def test_mean_without_automatic_flag_still_enables(self) -> None:
        cli = _make_cli(network_latency_mean=0.0)
        assert build_network_latency(cli) == {
            "enabled": True,
            "mean_ms": 0.0,
        }

    def test_automatic_and_mean_together_raises(self) -> None:
        cli = _make_cli(network_latency_automatic=True, network_latency_mean=5.0)
        with pytest.raises(
            ValueError,
            match="Cannot use both --network-latency-automatic and --network-latency-mean",
        ):
            build_network_latency(cli)

    def test_mean_with_ping_interval_raises(self) -> None:
        cli = _make_cli(network_latency_mean=5.0, network_latency_ping_interval=0.5)
        with pytest.raises(
            ValueError,
            match="--network-latency-ping-interval only applies with --network-latency-automatic",
        ):
            build_network_latency(cli)

    def test_ping_interval_without_automatic_raises(self) -> None:
        cli = _make_cli(network_latency_ping_interval=0.5)
        with pytest.raises(
            ValueError, match="only applies when --network-latency-automatic"
        ):
            build_network_latency(cli)


class TestNetworkLatencyConfigDefaults:
    """Default values and the default-factory wiring for ping_interval."""

    def test_defaults(self) -> None:
        config = NetworkLatencyConfig()
        assert config.enabled is False
        assert config.mean_ms is None
        assert config.ping_interval == pytest.approx(
            Environment.NETWORK_LATENCY.DEFAULT_PROBE_INTERVAL
        )


class TestNetworkLatencyConfigConstraints:
    """Field constraints reject invalid ping_interval / mean_ms."""

    @pytest.mark.parametrize("ping_interval", [0.0, -1.0])
    def test_non_positive_ping_interval_rejected(self, ping_interval: float) -> None:
        with pytest.raises(ValidationError):
            NetworkLatencyConfig(ping_interval=ping_interval)

    def test_negative_mean_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NetworkLatencyConfig(mean_ms=-1.0)

    def test_zero_mean_accepted(self) -> None:
        assert NetworkLatencyConfig(mean_ms=0.0).mean_ms == 0.0


class TestShouldProbe:
    """should_probe is True only when enabled and no manual mean is set."""

    @pytest.mark.parametrize(
        ("enabled", "mean_ms", "expected"),
        [
            (False, None, False),
            (True, None, True),
            (True, 5.0, False),
            (False, 5.0, False),
        ],
    )
    def test_should_probe(
        self, enabled: bool, mean_ms: float | None, expected: bool
    ) -> None:
        config = NetworkLatencyConfig(enabled=enabled, mean_ms=mean_ms)
        assert config.should_probe is expected
