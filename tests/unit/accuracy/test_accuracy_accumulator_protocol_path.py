# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive AccuracyAccumulator through the registered plugin protocol path.

Mirrors how RecordsManager actually uses the accumulator: class resolution
via ``accumulator:accuracy_results``, construction with the exact kwargs
``load_accumulators`` passes, record-type routing via plugin metadata, and
summarize invocation through ``generate_realtime_metrics`` (the engine's
realtime path calls ``acc.summarize()`` with no arguments). Direct-method
unit tests stay green when any of these seams drift; these tests do not.
"""

import pytest

from aiperf.accuracy.accuracy_accumulator import AccuracyAccumulator
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    AccuracyBenchmarkType,
    DatasetSamplingStrategy,
    EndpointType,
    PluginType,
)
from aiperf.records.records_manager_processing import (
    accumulators_for_record_type,
    generate_realtime_metrics,
)
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_registered_accumulator() -> AccuracyAccumulator:
    AccumulatorClass = plugins.get_class(PluginType.ACCUMULATOR, "accuracy_results")
    run = make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    )
    # Exact kwarg shape used by records_manager_processing.load_accumulators.
    return AccumulatorClass(service_id="records_manager", run=run, pub_client=None)


class TestAccuracyAccumulatorProtocolPath:
    def test_registry_resolves_accuracy_accumulator_class(self) -> None:
        AccumulatorClass = plugins.get_class(PluginType.ACCUMULATOR, "accuracy_results")
        assert AccumulatorClass is AccuracyAccumulator

    def test_metadata_routes_to_metric_records_dispatch(self) -> None:
        accumulator = _make_registered_accumulator()
        accumulators = {AccumulatorType("accuracy_results"): accumulator}

        assert accumulators_for_record_type(accumulators, "metric_records") == [
            accumulator
        ]
        assert accumulators_for_record_type(accumulators, "gpu_telemetry") == []

    def test_disabled_accuracy_raises_post_processor_disabled(self) -> None:
        """load_accumulators skips PostProcessorDisabled - the self-disable contract."""
        AccumulatorClass = plugins.get_class(PluginType.ACCUMULATOR, "accuracy_results")
        run = make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
        )

        with pytest.raises(PostProcessorDisabled):
            AccumulatorClass(service_id="records_manager", run=run, pub_client=None)

    @pytest.mark.asyncio
    async def test_process_record_and_summarize_through_engine_helper(self) -> None:
        accumulator = _make_registered_accumulator()
        accumulator.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="conv-0",
                        accuracy_ground_truth="A",
                        accuracy_task="algebra",
                    ),
                    ConversationMetadata(
                        conversation_id="conv-1",
                        accuracy_ground_truth="B",
                        accuracy_task="history",
                    ),
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )

        await accumulator.process_record(
            MetricRecordsData(
                metadata=create_metric_metadata(session_num=0),
                metrics={"accuracy.correct": 1.0, "accuracy.unparsed": 0.0},
            )
        )
        await accumulator.process_record(
            MetricRecordsData(
                metadata=create_metric_metadata(session_num=1),
                metrics={"accuracy.correct": 0.0, "accuracy.unparsed": 1.0},
            )
        )

        # generate_realtime_metrics calls acc.summarize() exactly like the
        # RecordsManager realtime path and flattens list-shaped results.
        results = await generate_realtime_metrics([accumulator])

        by_tag = {r.tag: r for r in results}
        assert by_tag["accuracy.overall"].current == pytest.approx(0.5)
        assert by_tag["accuracy.task.algebra"].current == pytest.approx(1.0)
        assert by_tag["accuracy.task.history"].current == pytest.approx(0.0)
        assert by_tag["accuracy.unparsed"].sum == 1
