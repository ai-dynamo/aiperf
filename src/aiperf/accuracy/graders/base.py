# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.accuracy.models import GradingResult
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseGrader(AIPerfLoggerMixin):
    """Base class for accuracy graders.

    Provides common initialization and interface for all grader implementations.
    Subclasses must override `grade()` and `extract_answer()`.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run = run

    @classmethod
    def check_available(cls) -> None:
        """Raise if this grader's optional dependencies are missing.

        Called from the main-process preflight (``cli_runner._preflight``)
        BEFORE any service is spawned, so a missing optional dependency (e.g.
        lighteval) surfaces as a clean ``ConfigurationError`` instead of
        crashing the daemon record-processor mid-run — which prints a raw
        multiprocessing traceback and hangs the parent waiting for records
        that never arrive.

        Default is a no-op: graders with no optional dependencies (the
        hand-ported ``exact_match`` / ``multiple_choice`` / ``gsm8k`` / ``math``
        graders) are always available. Graders backed by an optional package
        override this to check their import flag.
        """
        return

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
