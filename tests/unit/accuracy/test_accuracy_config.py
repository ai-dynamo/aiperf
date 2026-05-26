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

from aiperf.config.accuracy import AccuracyConfig

# Stub names match the ``is_implemented: false`` entries in plugins.yaml.
# Update both lists together when a follow-up branch lands an
# implementation (and remove the ``is_implemented: false`` from the YAML).
# This branch (AIP-874) implements ``aime``, ``math``, and ``code_execution``,
# so those names are absent from the stub lists.
STUB_BENCHMARKS = (
    "aime24",
    "aime25",
    "math_500",
    "gpqa_diamond",
    "lcb_codegeneration",
)
STUB_GRADERS: tuple[str, ...] = ()


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
            AccuracyConfig(benchmark="LCB_CODEGENERATION")
        assert "lcb_codegeneration" in str(exc.value)


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


class TestRequiresBenchmarkWhenAccuracyFieldsSet:
    """Regression tests for silent no-op when ``--accuracy-tasks`` (or any
    other ``--accuracy-*`` flag) is set without ``--accuracy-benchmark``.

    Without this validator, the v2 ``AccuracyConfig`` would accept
    ``tasks=["mmlu"]`` with ``benchmark=None``; ``enabled`` would be
    ``False``; the entire accuracy pipeline would self-disable; and the
    user would get a normal perf benchmark with no ``accuracy_results.csv``
    and no warning that their flags were ignored.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            param({"tasks": ["mmlu"]}, id="tasks-only"),
            param({"tasks": ["abstract_algebra", "anatomy"]}, id="tasks-multi"),
            param({"n_shots": 3}, id="n_shots-only"),
            param({"enable_cot": True}, id="enable_cot-true"),
            param({"enable_cot": False}, id="enable_cot-false"),
            param({"grader": "multiple_choice"}, id="grader-only"),
            param({"system_prompt": "you are an expert"}, id="system_prompt-only"),
            param({"verbose": True}, id="verbose-true"),
            param({"verbose": False}, id="verbose-false"),
            param(
                {"tasks": ["mmlu"], "n_shots": 5, "verbose": True},
                id="multiple-fields",
            ),
        ],
    )  # fmt: skip
    def test_accuracy_field_without_benchmark_rejected(
        self, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError) as exc:
            AccuracyConfig(**kwargs)
        msg = str(exc.value)
        assert "--accuracy-benchmark" in msg
        assert "silently ignored" in msg
        # Surface at least one available benchmark name so users have an
        # immediate next step.
        assert "mmlu" in msg

    def test_empty_config_still_valid(self) -> None:
        """No accuracy flags at all is the default and must not error."""
        cfg = AccuracyConfig()
        assert cfg.enabled is False

    def test_benchmark_set_with_tasks_passes(self) -> None:
        """The companion path: tasks plus a real benchmark is the
        intended usage and must validate cleanly."""
        cfg = AccuracyConfig(benchmark="mmlu", tasks=["abstract_algebra"])
        assert cfg.enabled is True
        assert cfg.tasks == ["abstract_algebra"]
