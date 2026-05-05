# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.groups import Groups
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
    """Configuration for accuracy benchmarking mode."""

    @model_validator(mode="after")
    def _reject_stub_plugins(self) -> "AccuracyConfig":
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

    benchmark: Annotated[
        AccuracyBenchmarkType | None,
        Field(
            description="Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode alongside performance profiling.",
        ),
        CLIParameter(
            name=("--accuracy-benchmark",),
            group=Groups.ACCURACY,
        ),
    ] = None

    tasks: Annotated[
        list[str] | None,
        BeforeValidator(parse_str_or_list),
        Field(
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). Accepts comma-separated values "
            "(e.g. abstract_algebra,anatomy) or repeated flags. If not set, all tasks are included.",
        ),
        CLIParameter(
            name=("--accuracy-tasks",),
            group=Groups.ACCURACY,
        ),
    ] = None

    n_shots: Annotated[
        int | None,
        Field(
            ge=0,
            le=32,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation, None uses the benchmark default (e.g. MMLU=5). Maximum 32.",
        ),
        CLIParameter(
            name=("--accuracy-n-shots",),
            group=Groups.ACCURACY,
        ),
    ] = None

    enable_cot: Annotated[
        bool,
        Field(
            description="Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt.",
        ),
        CLIParameter(
            name=("--accuracy-enable-cot",),
            group=Groups.ACCURACY,
        ),
    ] = False

    grader: Annotated[
        AccuracyGraderType | None,
        Field(
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader.",
        ),
        CLIParameter(
            name=("--accuracy-grader",),
            group=Groups.ACCURACY,
        ),
    ] = None

    system_prompt: Annotated[
        str | None,
        Field(
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
        CLIParameter(
            name=("--accuracy-system-prompt",),
            group=Groups.ACCURACY,
        ),
    ] = None

    verbose: Annotated[
        bool,
        Field(
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
        CLIParameter(
            name=("--accuracy-verbose",),
            group=Groups.ACCURACY,
        ),
    ] = False

    @property
    def enabled(self) -> bool:
        """Whether accuracy benchmarking mode is enabled."""
        return self.benchmark is not None
