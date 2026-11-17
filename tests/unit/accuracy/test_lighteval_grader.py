# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from aiperf.accuracy.graders import lighteval_grader
from aiperf.accuracy.graders.lighteval_grader import (
    LightevalGPQAGrader,
    _LightevalBaseGrader,
)


class _MinimalLightevalGrader(_LightevalBaseGrader):
    def _build_metric(self) -> Any:
        return object()


def test_lighteval_base_grader_accepts_benchmark_run(
    monkeypatch, benchmark_run
) -> None:
    monkeypatch.setattr(lighteval_grader, "_HAS_LIGHTEVAL", True)

    grader = _MinimalLightevalGrader(run=benchmark_run)

    assert grader.run is benchmark_run
    assert grader.accuracy_config is benchmark_run.cfg.accuracy


@pytest.mark.skipif(not lighteval_grader._HAS_LIGHTEVAL, reason="needs lighteval")
@pytest.mark.asyncio
async def test_gpqa_grade_malformed_gold_is_grader_error(benchmark_run) -> None:
    grader = LightevalGPQAGrader(run=benchmark_run)

    result = await grader.grade("The answer is A", "Z")

    assert result.grader_error is True
    assert result.unparsed is False
    assert result.correct is False
    assert "invalid ground truth" in result.reasoning
