# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive AccuracyAccumulator through the registered plugin protocol path.

Direct-method unit tests stay green when registry wiring, record-type metadata,
or realtime summarize invocation drift. These tests exercise the same seams the
records pipeline uses for ``accumulator:accuracy_results``.
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
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_manager_processing import generate_realtime_metrics
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_registered_accumulator() -> AccuracyAccumulator:
    accumulator_class = plugins.get_class(PluginType.ACCUMULATOR, "accuracy_results")
    run = make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    )
    return accumulator_class(service_id="records_manager", run=run, pub_client=None)


class TestAccuracyAccumulatorProtocolPath:
    def test_registry_resolves_accuracy_accumulator_class(self) -> None:
        accumulator_class = plugins.get_class(
            PluginType.ACCUMULATOR, "accuracy_results"
        )

        assert accumulator_class is AccuracyAccumulator

    def test_metadata_routes_to_metric_records_dispatch(self) -> None:
        accumulator = _make_registered_accumulator()
        manager = RecordsManager.__new__(RecordsManager)
        manager._accumulators = {AccumulatorType.ACCURACY_RESULTS: accumulator}
        manager._stream_exporters = {}

        table = manager._build_routing_table()

        assert table["metric_records"] == [accumulator]
        assert accumulator not in table.get("gpu_telemetry", [])

    def test_disabled_accuracy_raises_post_processor_disabled(self) -> None:
        accumulator_class = plugins.get_class(
            PluginType.ACCUMULATOR, "accuracy_results"
        )
        run = make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
        )

        with pytest.raises(PostProcessorDisabled):
            accumulator_class(service_id="records_manager", run=run, pub_client=None)

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

        results = await generate_realtime_metrics([accumulator])

        by_tag = {r.tag: r for r in results}
        assert by_tag["accuracy.overall"].current == pytest.approx(0.5)
        assert by_tag["accuracy.task.algebra"].current == pytest.approx(1.0)
        assert by_tag["accuracy.task.history"].current == pytest.approx(0.0)
        assert by_tag["accuracy.unparsed"].sum == 1
