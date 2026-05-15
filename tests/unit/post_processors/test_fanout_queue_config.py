# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: AIPERF_OTEL_MAX_BUFFERED_RECORDS controls fanout queue maxsize."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiperf.common.environment import Environment
from aiperf.config import (
    ArtifactsConfig,
    BenchmarkConfig,
    EndpointConfig,
    OTelConfig,
)
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.otel_metrics_results_processor import (
    OTelMetricsResultsProcessor,
)


@pytest.mark.asyncio
async def test_fanout_queue_maxsize_reads_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_artifact_dir: Path,
    fake_otel: dict[str, object],
) -> None:
    """Queue maxsize must equal AIPERF_OTEL_MAX_BUFFERED_RECORDS (Req 7.4)."""
    monkeypatch.setattr(Environment.OTEL, "MAX_BUFFERED_RECORDS", 1)

    user_config = BenchmarkConfig(
        model="test-model",
        endpoint=EndpointConfig(
            urls=["http://localhost:8000"],
            type=EndpointType.CHAT,
        ),
        dataset={"type": "synthetic"},
        profiling={"type": "concurrency", "requests": 1, "concurrency": 1},
        artifacts=ArtifactsConfig(dir=tmp_artifact_dir),
        otel=OTelConfig(metrics_url="collector:4318"),
    )

    processor = OTelMetricsResultsProcessor(
        service_id="records-manager",
        run=SimpleNamespace(cfg=user_config, benchmark_id="bench-fanout"),
    )

    assert processor._fanout_queue_maxsize == 1

    # Mock the fanout target so we don't actually spawn a real child process.
    with patch(
        "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
    ):
        await processor._start_fanout_process()

    try:
        assert processor._fanout_queue is not None
        assert processor._fanout_queue._maxsize == 1
    finally:
        if (
            processor._fanout_process is not None
            and processor._fanout_process.is_alive()
        ):
            processor._fanout_process.terminate()
            processor._fanout_process.join(timeout=5)
        if processor._fanout_queue is not None:
            processor._fanout_queue.close()
