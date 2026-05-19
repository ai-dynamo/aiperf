# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.accuracy.graders import lighteval_grader
from aiperf.accuracy.graders.lighteval_grader import _LightevalBaseGrader


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
