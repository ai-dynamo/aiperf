# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ConsoleArchetypeMetricsExporter."""

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.image_config import ImageHeightConfig, ImageWidthConfig
from aiperf.common.config.media_mix_config import (
    ImageProfileConfig,
    MediaMixArchetype,
    ModalityEntry,
)
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.console_archetype_metrics_exporter import (
    ConsoleArchetypeMetricsExporter,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import EndpointType


def _build_user_config(*archetype_names_and_weights: tuple[str, float]) -> UserConfig:
    media_mix = [
        MediaMixArchetype(
            weight=weight,
            name=name,
            modalities=[
                ModalityEntry(
                    modality="image",
                    profiles=[
                        ImageProfileConfig(
                            weight=1.0,
                            width=ImageWidthConfig(mean=64),
                            height=ImageHeightConfig(mean=64),
                        )
                    ],
                )
            ],
        )
        for name, weight in archetype_names_and_weights
    ]
    return UserConfig(
        endpoint=EndpointConfig(
            type=EndpointType.CHAT, streaming=True, model_names=["test-model"]
        ),
        input={"media_mix": media_mix},
    )


def _records() -> list[MetricResult]:
    return [
        MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=120.0,
            min=55.0,
            max=450.0,
            p95=280.0,
            p50=110.0,
        ),
    ]


class TestConsoleArchetypeMetricsExporter:
    """Per-archetype console rendering for media mix benchmarks."""

    def test_disabled_when_no_archetype_data(self):
        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                streaming=True,
                model_names=["test-model"],
            )
        )
        config = ExporterConfig(
            results=ProfileResults(records=[], start_ns=0, end_ns=0, completed=0),
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        with pytest.raises(ConsoleExporterDisabled, match="no archetype metric"):
            ConsoleArchetypeMetricsExporter(config)

    @pytest.mark.asyncio
    async def test_prints_one_table_per_archetype(self, capsys):
        user_config = _build_user_config(("image-only", 0.4), ("video-only", 0.6))
        archetype_results = {
            "image-only": _records(),
            "video-only": _records(),
        }
        config = ExporterConfig(
            results=ProfileResults(
                records=[],
                archetype_metric_results=archetype_results,
                start_ns=0,
                end_ns=0,
                completed=0,
            ),
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = ConsoleArchetypeMetricsExporter(config)
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out

        # Both archetype names appear as table titles, with traffic share.
        assert "image-only" in output
        assert "video-only" in output
        assert "40%" in output
        assert "60%" in output
        # The metric content shows up in both tables.
        assert output.count("Request Latency") >= 2
