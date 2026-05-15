# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for ``AccuracyConfig``.

Exercises the ``_reject_stub_plugins`` model validator that surfaces
``--accuracy-benchmark`` / ``--accuracy-grader`` errors at config-parse
time instead of letting an unimplemented stub raise
``NotImplementedError`` deep inside async dataset loading.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.config.accuracy_config import AccuracyConfig

# Stub names match the ``is_implemented: false`` entries in plugins.yaml.
# Update both lists together when a follow-up branch lands an
# implementation (and remove the ``is_implemented: false`` from the YAML).
# This branch (AIP-874) implements ``aime``, ``math``, and ``code_execution``,
# so those names are absent from the stub lists.
STUB_BENCHMARKS = (
    "hellaswag",
    "bigbench",
    "aime24",
    "aime25",
    "math_500",
    "gpqa_diamond",
    "lcb_codegeneration",
)
STUB_GRADERS = ("exact_match",)


class TestAcceptsImplemented:
    def test_accuracyconfig_no_benchmark_returns_defaults(self) -> None:
        cfg = AccuracyConfig()
        assert cfg.benchmark is None
        assert cfg.grader is None
        assert cfg.enabled is False

    def test_accuracyconfig_with_implemented_benchmark_enables_config(self) -> None:
        cfg = AccuracyConfig(benchmark="mmlu")
        assert str(cfg.benchmark) == "mmlu"
        assert cfg.enabled is True

    def test_accuracyconfig_with_grader_override_sets_grader(self) -> None:
        cfg = AccuracyConfig(benchmark="mmlu", grader="multiple_choice")
        assert str(cfg.grader) == "multiple_choice"


class TestRejectsStubBenchmark:
    @pytest.mark.parametrize(
        "name",
        [param(n, id=n) for n in STUB_BENCHMARKS],
    )  # fmt: skip
    def test_accuracyconfig_with_stub_benchmark_raises_validationerror(
        self, name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc:
            AccuracyConfig(benchmark=name)
        msg = str(exc.value)
        assert "--accuracy-benchmark" in msg
        assert name in msg
        assert "not yet implemented" in msg
        assert "Available:" in msg
        # ``mmlu`` is the one always-implemented benchmark; the message
        # must surface at least that as a usable alternative.
        assert "mmlu" in msg.split("Available:")[-1]

    def test_accuracyconfig_with_hyphenated_stub_name_raises_validationerror(
        self,
    ) -> None:
        """Reproduces the original bug: ``--accuracy-benchmark lcb-codegeneration``
        used the hyphen-tolerant enum lookup and reached the loader."""
        with pytest.raises(ValidationError) as exc:
            AccuracyConfig(benchmark="lcb-codegeneration")
        msg = str(exc.value)
        # Enum normalization runs first → message references the canonical
        # snake-case form, not the user's hyphenated input.
        assert "lcb_codegeneration" in msg
        assert "not yet implemented" in msg

    def test_accuracyconfig_with_uppercase_stub_name_raises_validationerror(
        self,
    ) -> None:
        """Case-insensitive enum lookup must not bypass the validator."""
        with pytest.raises(ValidationError) as exc:
            AccuracyConfig(benchmark="HELLASWAG")
        assert "hellaswag" in str(exc.value)


class TestRejectsStubGrader:
    @pytest.mark.parametrize(
        "name",
        [param(n, id=n) for n in STUB_GRADERS],
    )  # fmt: skip
    def test_accuracyconfig_with_stub_grader_override_raises_validationerror(
        self, name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc:
            AccuracyConfig(benchmark="mmlu", grader=name)
        msg = str(exc.value)
        assert "--accuracy-grader" in msg
        assert name in msg
        assert "not yet implemented" in msg
        # ``multiple_choice`` is the one always-implemented grader.
        assert "multiple_choice" in msg.split("Available:")[-1]

    def test_accuracyconfig_grader_unset_allows_default(self) -> None:
        """Leaving ``grader`` unset must not trigger the stub check.

        AccuracyConfig stays neutral about which grader the benchmark
        defaults to — the dataset loader resolves that. This test pins
        that behavior so the validator only ever inspects an explicit
        ``--accuracy-grader`` override.
        """
        cfg = AccuracyConfig(benchmark="mmlu")
        assert cfg.grader is None
