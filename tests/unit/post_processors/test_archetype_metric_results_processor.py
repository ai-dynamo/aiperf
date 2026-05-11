# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.config.image_config import ImageHeightConfig, ImageWidthConfig
from aiperf.common.config.media_mix_config import (
    ImageProfileConfig,
    MediaMixArchetype,
    ModalityEntry,
)
from aiperf.common.enums import MetricType
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.post_processors.archetype_metric_results_processor import (
    ArchetypeMetricResultsProcessor,
)
from tests.unit.post_processors.conftest import create_metric_records_message


def _media_mix_for_test(*names: str) -> list[MediaMixArchetype]:
    """Build a media_mix list with one image archetype per name."""
    return [
        MediaMixArchetype(
            weight=1.0,
            name=name,
            modalities=[
                ModalityEntry(
                    modality="image",
                    profiles=[
                        ImageProfileConfig(
                            weight=1.0,
                            width=ImageWidthConfig(mean=256),
                            height=ImageHeightConfig(mean=256),
                        )
                    ],
                )
            ],
        )
        for name in names
    ]


class TestArchetypeMetricResultsProcessor:
    """Per-archetype metric grouping for media mix benchmarks."""

    def test_initialization_without_media_mix_raises(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        assert mock_user_config.input.media_mix is None
        with pytest.raises(PostProcessorDisabled, match="requires media_mix"):
            ArchetypeMetricResultsProcessor(mock_user_config)

    def test_initialization_with_media_mix_succeeds(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        mock_user_config.input.media_mix = _media_mix_for_test("image-only")
        processor = ArchetypeMetricResultsProcessor(mock_user_config)
        assert processor.result_kind == "archetype"
        assert hasattr(processor, "_archetype_instances_maps")
        assert hasattr(processor, "_archetype_results")

    @pytest.mark.asyncio
    async def test_process_result_separates_by_archetype(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        mock_user_config.input.media_mix = _media_mix_for_test("alpha", "beta")
        processor = ArchetypeMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        msg_alpha = create_metric_records_message(
            x_request_id="r-1",
            request_start_ns=1000,
            results=[{"test_record": 11.0}],
        )
        msg_alpha.metadata.archetype_name = "alpha"
        await processor.process_result(msg_alpha.to_data())

        msg_beta = create_metric_records_message(
            x_request_id="r-2",
            request_start_ns=2000,
            results=[{"test_record": 22.0}],
        )
        msg_beta.metadata.archetype_name = "beta"
        await processor.process_result(msg_beta.to_data())

        assert "alpha" in processor._archetype_results
        assert "beta" in processor._archetype_results
        assert list(processor._archetype_results["alpha"]["test_record"].data) == [11.0]
        assert list(processor._archetype_results["beta"]["test_record"].data) == [22.0]

    @pytest.mark.asyncio
    async def test_summarize_returns_per_archetype_results(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        from aiperf.metrics.metric_dicts import MetricArray
        from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric

        mock_user_config.input.media_mix = _media_mix_for_test("alpha", "beta")
        processor = ArchetypeMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}

        # Seed buckets directly with realistic metric data (values in
        # internal nanosecond units; summarize converts to display unit ms).
        processor._archetype_results["alpha"][RequestLatencyMetric.tag] = MetricArray()
        processor._archetype_results["alpha"][RequestLatencyMetric.tag].append(
            42_000_000.0
        )
        processor._archetype_results["beta"][RequestLatencyMetric.tag] = MetricArray()
        processor._archetype_results["beta"][RequestLatencyMetric.tag].append(
            84_000_000.0
        )

        # Parent class _create_metric_result reads from self._instances_map.
        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        summary = await processor.summarize()
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {"alpha", "beta"}
        for results in summary.values():
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].tag == RequestLatencyMetric.tag

    @pytest.mark.asyncio
    async def test_synthetic_named_archetypes_grouped_correctly(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Synthetic _archetype_{i} names from the InputConfig validator land in distinct buckets."""
        mock_user_config.input.media_mix = _media_mix_for_test(
            "_archetype_0", "_archetype_1"
        )
        processor = ArchetypeMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        for name in ("_archetype_0", "_archetype_1"):
            msg = create_metric_records_message(
                x_request_id=name,
                request_start_ns=1,
                results=[{"test_record": 7.0}],
            )
            msg.metadata.archetype_name = name
            await processor.process_result(msg.to_data())

        assert "_archetype_0" in processor._archetype_results
        assert "_archetype_1" in processor._archetype_results
        # Confirm the two synthetic buckets are independent objects, not aliased.
        assert (
            processor._archetype_results["_archetype_0"]
            is not processor._archetype_results["_archetype_1"]
        )
