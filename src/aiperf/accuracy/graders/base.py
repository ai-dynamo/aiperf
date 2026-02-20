# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin


class BaseGrader(AIPerfLoggerMixin):
    """Base class for accuracy graders.

    Provides common initialization and interface for all grader implementations.
    Subclasses must override `grade()` and `extract_answer()`.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult:
        """Grade a model response against ground truth.

        Args:
            response_text: The raw text response from the LLM.
            ground_truth: The expected correct answer.
            **kwargs: Additional grading parameters.

        Returns:
            GradingResult with correctness, confidence, and reasoning.
        """
        raise NotImplementedError

    def extract_answer(self, response_text: str, **kwargs) -> str:
        """Extract the answer portion from a model response.

        Args:
            response_text: The raw text response from the LLM.
            **kwargs: Additional extraction parameters.

        Returns:
            The extracted answer string.
        """
        raise NotImplementedError
