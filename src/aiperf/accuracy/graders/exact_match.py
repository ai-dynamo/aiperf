# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ExactMatchGrader(BaseGrader):
    """Grades responses by exact string matching against ground truth."""

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult:
        raise NotImplementedError(
            "exact_match grader is not yet implemented; only 'multiple_choice' is available in this release."
        )

    def extract_answer(self, response_text: str, **kwargs) -> str:
        raise NotImplementedError(
            "exact_match grader is not yet implemented; only 'multiple_choice' is available in this release."
        )
