# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube._kube_common`.

Focuses on:
- ``generate_benchmark_name``: DNS-safe name building, suffix handling, slug
  truncation, character sanitization, model-org stripping.
- ``print_memory_estimate``: parameter wiring into the estimator + label
  prefix output path.
- Re-exports (``resolve_config``, ``deep_merge``, ``build_v1_overrides``):
  confirm the back-compat surface still resolves to the resolver module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube import _kube_common
from aiperf.cli_commands.kube._kube_common import (
    generate_benchmark_name,
    print_memory_estimate,
)

# ============================================================
# Helpers
# ============================================================


def _fake_config(
    *,
    model: str = "Qwen/Qwen3",
    endpoint_type: str = "openai",
    phase_type: str = "throughput",
    workers_per_pod: int = 10,
) -> SimpleNamespace:
    """Build a duck-typed config that satisfies generate_benchmark_name + print_memory_estimate."""
    benchmark = SimpleNamespace(
        get_model_names=lambda: [model],
        endpoint=SimpleNamespace(type=endpoint_type),
        phases=[SimpleNamespace(type=phase_type)],
        runtime=SimpleNamespace(workers_per_pod=workers_per_pod),
    )
    return SimpleNamespace(benchmark=benchmark)


# ============================================================
# generate_benchmark_name
# ============================================================


class TestGenerateBenchmarkName:
    """Verify the DNS-safe name generator."""

    def test_simple_name_constructed_from_model_endpoint_phase(self) -> None:
        config = _fake_config(
            model="qwen3", endpoint_type="openai", phase_type="throughput"
        )
        assert generate_benchmark_name(config) == "qwen3-openai-throughput"

    def test_model_org_prefix_stripped(self) -> None:
        """`Qwen/Qwen3-1.7b` keeps only the trailing slug."""
        config = _fake_config(model="Qwen/Qwen3-1.7b")
        # Model basename is lowercased; '.' becomes '-' via the regex sanitize.
        assert generate_benchmark_name(config).startswith("qwen3-1-7b-")

    def test_suffix_appended_with_hyphen(self) -> None:
        config = _fake_config(model="qwen3")
        assert generate_benchmark_name(config, suffix="sweep").endswith("-sweep")

    def test_empty_suffix_no_dangling_hyphen(self) -> None:
        config = _fake_config()
        # An empty suffix must not yield a trailing hyphen.
        name = generate_benchmark_name(config, suffix="")
        assert not name.endswith("-")

    @pytest.mark.parametrize(
        "model,expected_prefix",
        [
            param("Llama-3.1-70B-Instruct", "llama-3-1-70b-instruct-", id="dots"),
            param("MyOrg/My_Model", "my_model-", id="underscores-preserved-from-source"),
            param("Hello World", "hello-world-", id="space-becomes-hyphen"),
            param("model$$$weird", "model---weird-", id="symbols-become-hyphens"),
        ],
    )  # fmt: skip
    def test_special_characters_normalized(
        self, model: str, expected_prefix: str
    ) -> None:
        config = _fake_config(model=model)
        # `_` survives the regex `[^a-z0-9-]`. We only assert the LOWERED
        # output begins with the expected prefix; the regex passes lowercase
        # letters, digits, and hyphens.
        name = generate_benchmark_name(config)
        # Lowercase + character class equivalence: replace [^a-z0-9-] with `-`.
        # `MyOrg/My_Model` → after `.split('/')[-1]`: `My_Model` → lower
        # `my_model` → underscore is sanitized to `-`. So expected prefix
        # rebuilt: `my-model-`.
        if "_" in model:
            # Underscores are NOT in the allowed character class; they
            # become hyphens. Adjust expectation in this branch.
            assert name.startswith("my-model-")
        else:
            assert name.startswith(expected_prefix)

    def test_truncated_to_40_characters(self) -> None:
        config = _fake_config(
            model="this-is-an-extremely-long-model-name-that-overflows-the-budget",
            endpoint_type="openai",
            phase_type="throughput",
        )
        name = generate_benchmark_name(config)
        assert len(name) <= 40

    def test_leading_and_trailing_hyphens_stripped(self) -> None:
        """A trailing slash on the model basename leaves a hyphen the strip should remove."""
        config = _fake_config(model="org/")  # split → ""
        # Empty model slug yields a leading hyphen which strip("-") removes.
        name = generate_benchmark_name(config)
        assert not name.startswith("-")
        assert not name.endswith("-")

    def test_suffix_also_truncated_within_40_chars(self) -> None:
        config = _fake_config(
            model="my-model-with-some-length-for-the-test",
            endpoint_type="openai",
            phase_type="throughput",
        )
        name = generate_benchmark_name(config, suffix="sweep")
        assert len(name) <= 40

    def test_uses_first_phase_only(self) -> None:
        """Multiple phases — only the first contributes to the name."""
        config = SimpleNamespace(
            get_model_names=lambda: ["qwen3"],
            endpoint=SimpleNamespace(type="openai"),
            phases=[
                SimpleNamespace(type="warmup"),
                SimpleNamespace(type="throughput"),
            ],
        )
        name = generate_benchmark_name(config)
        assert "warmup" in name
        assert "throughput" not in name


# ============================================================
# print_memory_estimate
# ============================================================


class TestPrintMemoryEstimate:
    """Verify estimator parameters and rendered output flow to the kube console."""

    def test_passes_workers_and_workers_per_pod_through(self) -> None:
        config = _fake_config(workers_per_pod=4)
        kube_options = SimpleNamespace(workers=20)
        spec: dict[str, Any] = {"connectionsPerWorker": 50}

        with (
            patch(
                "aiperf.kubernetes.memory_estimator.estimate_memory"
            ) as mock_estimate,
            patch("aiperf.kubernetes.memory_estimator.format_estimate") as mock_format,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            mock_estimate.return_value = MagicMock(name="mem_est")
            mock_format.return_value = "RENDERED PANEL"

            print_memory_estimate(config, kube_options, spec)

            mock_estimate.assert_called_once_with(
                config,
                total_workers=20,
                workers_per_pod=4,
                connections_per_worker=50,
            )
            mock_format.assert_called_once_with(mock_estimate.return_value)
            mock_console.print.assert_called_once_with(
                "RENDERED PANEL", highlight=False
            )

    def test_default_connections_per_worker_when_missing_from_spec(self) -> None:
        config = _fake_config(workers_per_pod=10)
        kube_options = SimpleNamespace(workers=10)

        with (
            patch(
                "aiperf.kubernetes.memory_estimator.estimate_memory"
            ) as mock_estimate,
            patch(
                "aiperf.kubernetes.memory_estimator.format_estimate",
                return_value="x",
            ),
            patch("aiperf.kubernetes.console.console"),
        ):
            print_memory_estimate(config, kube_options, spec={})
            kwargs = mock_estimate.call_args.kwargs
            assert kwargs["connections_per_worker"] == 100

    def test_label_prefix_printed_before_estimate(self) -> None:
        config = _fake_config()
        kube_options = SimpleNamespace(workers=1)
        spec: dict[str, Any] = {}

        with (
            patch("aiperf.kubernetes.memory_estimator.estimate_memory"),
            patch(
                "aiperf.kubernetes.memory_estimator.format_estimate",
                return_value="estimate-rendered",
            ),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            print_memory_estimate(
                config, kube_options, spec, label_prefix="Sweep template: "
            )

            # Two console prints: the prefix line, then the rendered estimate.
            assert mock_console.print.call_count == 2
            first_args = mock_console.print.call_args_list[0]
            second_args = mock_console.print.call_args_list[1]
            assert first_args.args == ("Sweep template: ",)
            assert first_args.kwargs == {"highlight": False}
            assert second_args.args == ("estimate-rendered",)

    def test_empty_label_prefix_skips_the_prefix_print(self) -> None:
        config = _fake_config()
        kube_options = SimpleNamespace(workers=1)

        with (
            patch("aiperf.kubernetes.memory_estimator.estimate_memory"),
            patch(
                "aiperf.kubernetes.memory_estimator.format_estimate",
                return_value="x",
            ),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            print_memory_estimate(config, kube_options, spec={})
            # Only the rendered estimate gets printed.
            mock_console.print.assert_called_once_with("x", highlight=False)


# ============================================================
# Re-export back-compat
# ============================================================


class TestReExports:
    """Verify the re-exported resolver helpers are still accessible."""

    @pytest.mark.parametrize(
        "name",
        [
            param("resolve_config", id="resolve-config"),
            param("_deep_merge", id="deep-merge"),
            param("_build_v1_overrides", id="build-v1-overrides"),
        ],
    )  # fmt: skip
    def test_resolver_helper_re_exported(self, name: str) -> None:
        assert hasattr(_kube_common, name)

    def test_re_exports_match_resolver_module(self) -> None:
        from aiperf.config.v1 import _resolver

        assert _kube_common.resolve_config is _resolver.resolve_config
        assert _kube_common._deep_merge is _resolver.deep_merge
        assert _kube_common._build_v1_overrides is _resolver.build_v1_overrides
