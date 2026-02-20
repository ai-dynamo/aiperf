# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig


class CodeExecutionGrader(BaseGrader):
    """Grades responses by executing generated code and comparing output against expected results."""

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        super().__init__(user_config=user_config, **kwargs)

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult:
        raise NotImplementedError

    def extract_answer(self, response_text: str, **kwargs) -> str:
        raise NotImplementedError
