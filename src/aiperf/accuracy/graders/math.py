# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.config import BenchmarkRun


class MathGrader(BaseGrader):
    """Grades responses by evaluating mathematical expressions for equivalence."""

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult:
        raise NotImplementedError

    def extract_answer(self, response_text: str, **kwargs) -> str:
        raise NotImplementedError
