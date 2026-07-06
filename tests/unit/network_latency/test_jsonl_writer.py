# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NetworkLatencyJSONLWriter.

Mirrors ``tests/unit/server_metrics/test_jsonl_writer.py``: a v2 BenchmarkRun is
built with probing enabled, the processor is driven through its lifecycle (so the
buffered writer flushes), and the JSONL artifact is asserted on disk. Also covers
the self-disable (PostProcessorDisabled) path when probing is not active.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import NetworkLatencySample
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.network_latency.jsonl_writer import NetworkLatencyJSONLWriter
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli
from tests.unit.post_processors.conftest import aiperf_lifecycle


@pytest.fixture
def cfg_probing_enabled(tmp_artifact_dir: Path) -> BenchmarkRun:
    """BenchmarkRun with active RTT probing (export enabled)."""
    user_cfg = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        urls=["http://localhost:8000/v1/chat"],
        artifact_directory=tmp_artifact_dir,
        network_latency_automatic=True,
    )
    return make_run_from_cli(user_cfg)


def _sample(
    rtt_ns: int | None = 1_234_567, success: bool = True
) -> NetworkLatencySample:
    return NetworkLatencySample(
        timestamp_ns=1_000_000_000,
        target_url="http://localhost:8000/v1/chat",
        target_host="localhost",
        target_port=8000,
        probe_type="tcp_connect",
        rtt_ns=rtt_ns,
        success=success,
    )


class TestInitialization:
    def test_output_file_is_network_latency_jsonl(
        self, cfg_probing_enabled: BenchmarkRun
    ) -> None:
        processor = NetworkLatencyJSONLWriter(
            run=cfg_probing_enabled, service_id="records-manager"
        )
        assert (
            processor.output_file
            == cfg_probing_enabled.cfg.artifacts.network_latency_export_jsonl_file
        )
        assert processor.output_file.name == "profile_export_network_latency.jsonl"

    def test_disabled_when_probing_not_active(self, tmp_artifact_dir: Path) -> None:
        """Manual mean (no active probing) self-disables the export processor."""
        user_cfg = CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat"],
            artifact_directory=tmp_artifact_dir,
            network_latency_mean=2.5,  # manual mean -> should_probe is False
        )
        run = make_run_from_cli(user_cfg)

        with pytest.raises(PostProcessorDisabled):
            NetworkLatencyJSONLWriter(run=run, service_id="records-manager")


class TestSampleProcessing:
    @pytest.mark.asyncio
    async def test_process_single_sample_writes_one_line(
        self, cfg_probing_enabled: BenchmarkRun
    ) -> None:
        processor = NetworkLatencyJSONLWriter(
            run=cfg_probing_enabled, service_id="records-manager"
        )

        async with aiperf_lifecycle(processor):
            await processor.process_network_latency_sample(_sample())

        output_file = (
            cfg_probing_enabled.cfg.artifacts.network_latency_export_jsonl_file
        )
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        data = orjson.loads(lines[0])
        assert data["target_host"] == "localhost"
        assert data["target_port"] == 8000
        assert data["rtt_ns"] == 1_234_567
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_process_multiple_samples_writes_multiple_lines(
        self, cfg_probing_enabled: BenchmarkRun
    ) -> None:
        processor = NetworkLatencyJSONLWriter(
            run=cfg_probing_enabled, service_id="records-manager"
        )

        async with aiperf_lifecycle(processor):
            for i in range(5):
                await processor.process_network_latency_sample(
                    _sample(rtt_ns=1_000 + i)
                )

        output_file = (
            cfg_probing_enabled.cfg.artifacts.network_latency_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_summarize_returns_empty(
        self, cfg_probing_enabled: BenchmarkRun
    ) -> None:
        processor = NetworkLatencyJSONLWriter(
            run=cfg_probing_enabled, service_id="records-manager"
        )
        assert await processor.summarize() == []
