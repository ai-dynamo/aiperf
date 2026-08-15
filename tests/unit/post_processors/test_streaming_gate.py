# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-level STREAMING_ONLY gate."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import MetricFlags
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


def _streaming_only_disallowed(run: BenchmarkRun) -> bool:
    """Whether the run-level gate suppresses STREAMING_ONLY metrics."""
    _, disallowed = BaseMetricsProcessor(run).get_filters()
    return bool(disallowed & MetricFlags.STREAMING_ONLY)


class TestStreamingOnlyGate:
    """Gate matrix for MetricFlags.STREAMING_ONLY over the global flag."""

    @pytest.mark.parametrize(
        ("streaming", "expect_disallowed"),
        [
            param(True, False, id="global_on_allows"),
            param(False, True, id="global_off_disallows"),
        ],
    )  # fmt: skip
    def test_gate_follows_global_flag(
        self, mock_run, streaming: bool, expect_disallowed: bool
    ) -> None:
        """Streaming applicability is decided once, by the global streaming flag."""
        mock_run.cfg.endpoint.streaming = streaming
        assert _streaming_only_disallowed(mock_run) is expect_disallowed
