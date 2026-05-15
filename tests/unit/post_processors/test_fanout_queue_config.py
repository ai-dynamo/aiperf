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

    cfg = BenchmarkConfig(
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
        run=SimpleNamespace(cfg=cfg, benchmark_id="bench-fanout"),
    )

    assert processor._fanout_queue_maxsize == 1

    class _FakeProcess:
        def __init__(
            self,
            *,
            target: object = None,
            args: tuple = (),
            name: str = "",
            daemon: bool = True,
        ) -> None:
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self._alive = False

        def start(self) -> None:
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            return None

    class _FakeContext:
        def Queue(self, maxsize: int = 0):  # noqa: N802 - mirror mp API
            from queue import Queue

            q: Queue = Queue(maxsize=maxsize)
            q._maxsize = maxsize  # type: ignore[attr-defined]
            return q

        def Process(self, **kwargs):  # noqa: N802 - mirror mp API
            return _FakeProcess(**kwargs)

    # Patch the fanout target and the multiprocessing context so no real
    # subprocess is forked from the unit test.
    with (
        patch(
            "aiperf.post_processors.otel_metrics_results_processor.run_otel_streaming_fanout"
        ),
        patch(
            "aiperf.post_processors.otel_metrics_results_processor.mp.get_context",
            return_value=_FakeContext(),
        ),
    ):
        await processor._start_fanout_process()

    assert processor._fanout_queue is not None
    assert processor._fanout_queue._maxsize == 1
