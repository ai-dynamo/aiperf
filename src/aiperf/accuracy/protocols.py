# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.accuracy.models import BenchmarkProblem, GradingResult
    from aiperf.common.config import UserConfig


@runtime_checkable
class AccuracyGraderProtocol(Protocol):
    """Protocol for accuracy graders that evaluate LLM responses against ground truth."""

    def __init__(self, user_config: UserConfig, **kwargs) -> None: ...

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult: ...

    def extract_answer(self, response_text: str, **kwargs) -> str: ...


@runtime_checkable
class AccuracyBenchmarkProtocol(Protocol):
    """Protocol for accuracy benchmark loaders that provide problems from standard datasets."""

    def __init__(self, user_config: UserConfig, **kwargs) -> None: ...

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]: ...
