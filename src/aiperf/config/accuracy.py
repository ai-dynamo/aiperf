# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmarking configuration.

Hosts `AccuracyConfig` — the optional per-benchmark block that enables
accuracy evaluation (MMLU, AIME, etc.) alongside performance profiling.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.loader.parsing import parse_str_or_list
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    AccuracyGraderType,
    PluginType,
)


def _list_implemented(category: PluginType) -> list[str]:
    """Names of plugins in ``category`` whose metadata does not flag them as stubs.

    A plugin is treated as implemented when ``metadata.is_implemented`` is
    truthy or absent (the default). Stubs explicitly opt out with
    ``is_implemented: false`` in plugins.yaml so the
    ``AccuracyConfig`` validator can reject them before any service starts.
    """
    return sorted(
        entry.name
        for entry in plugins.list_entries(category)
        if entry.metadata.get("is_implemented", True)
    )


def _check_implemented(category: PluginType, name: str, *, flag_name: str) -> None:
    """Raise ``ValueError`` if ``name`` resolves to a stub plugin.

    Why: registering stubs in plugins.yaml makes them visible to the CLI
    enum so we can keep their names stable across releases, but running
    them would only surface ``NotImplementedError`` deep inside async
    dataset loading. We fail at config-validation time instead, with a
    message that lists what IS available.
    """
    metadata = plugins.get_metadata(category, name)
    if metadata.get("is_implemented", True):
        return
    available = _list_implemented(category)
    raise ValueError(
        f"{flag_name} '{name}' is registered but not yet implemented "
        f"in this release. Available: {', '.join(available) or '(none)'}."
    )


class AccuracyConfig(BaseConfig):
    """Configuration for accuracy benchmarking mode.

    When benchmark is set, enables accuracy evaluation alongside
    performance profiling using standard benchmarks (MMLU, AIME, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _reject_stub_plugins(self) -> AccuracyConfig:
        """Reject benchmark/grader names that point at unimplemented stubs."""
        if self.benchmark is not None:
            _check_implemented(
                PluginType.ACCURACY_BENCHMARK,
                str(self.benchmark),
                flag_name="--accuracy-benchmark",
            )
        if self.grader is not None:
            _check_implemented(
                PluginType.ACCURACY_GRADER,
                str(self.grader),
                flag_name="--accuracy-grader",
            )
        return self

    @model_validator(mode="after")
    def _require_benchmark_when_accuracy_fields_set(self) -> AccuracyConfig:
        """Reject accuracy-only flags when ``--accuracy-benchmark`` is unset.

        Why: ``enabled`` keys off ``benchmark is not None``. A user passing
        only ``--accuracy-tasks`` (or any other ``--accuracy-*`` flag) would
        otherwise get a silent no-op: the accuracy pipeline self-disables
        and the run completes as a normal perf benchmark, with no warning
        and no ``accuracy_results.csv``. Surface the misconfiguration here
        so it fails before any service starts.
        """
        if self.benchmark is not None:
            return self
        dependent_fields = {
            "tasks": self.tasks,
            "n_shots": self.n_shots,
            "grader": self.grader,
            "system_prompt": self.system_prompt,
        }
        explicitly_set_boolean_fields = {
            "enable_cot",
            "verbose",
        } & self.model_fields_set
        set_fields = sorted(
            [k for k, v in dependent_fields.items() if v is not None]
            + list(explicitly_set_boolean_fields)
        )
        if not set_fields:
            return self
        flag_names = ", ".join(f"--accuracy-{k.replace('_', '-')}" for k in set_fields)
        available = _list_implemented(PluginType.ACCURACY_BENCHMARK)
        raise ValueError(
            f"Accuracy options {flag_names} were set but --accuracy-benchmark "
            f"is not. Accuracy mode requires --accuracy-benchmark to select a "
            f"benchmark; otherwise these flags are silently ignored. "
            f"Available benchmarks: {', '.join(available) or '(none)'}."
        )

    benchmark: Annotated[
        AccuracyBenchmarkType | None,
        Field(
            default=None,
            description="Accuracy benchmark to run. When set, enables accuracy "
            "benchmarking alongside performance profiling. AIME variants: 'aime' "
            "is the legacy combined set (deprecated for new runs); prefer the "
            "year-pinned 'aime24' or 'aime25' for reproducibility.",
        ),
    ]

    tasks: Annotated[
        list[str] | None,
        BeforeValidator(parse_str_or_list),
        Field(
            default=None,
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). Accepts comma-separated values "
            "(e.g. abstract_algebra,anatomy) or repeated flags. If not set, all tasks are included.",
        ),
    ]

    n_shots: Annotated[
        int | None,
        Field(
            ge=0,
            le=32,
            default=None,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation, None uses the benchmark default (e.g. MMLU=5). Maximum 32.",
        ),
    ]

    enable_cot: Annotated[
        bool | None,
        Field(
            default=None,
            description="Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt. Defaults to the benchmark's "
            "``default_enable_cot`` metadata when unset (e.g. AIME defaults to True).",
        ),
    ]

    grader: Annotated[
        AccuracyGraderType | None,
        Field(
            default=None,
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader.",
        ),
    ]

    system_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
    ]

    verbose: Annotated[
        bool,
        Field(
            default=False,
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
    ]

    @property
    def enabled(self) -> bool:
        """Whether accuracy benchmarking mode is enabled."""
        return self.benchmark is not None
