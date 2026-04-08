# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.models import BenchmarkProblem, GradingResult
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
    )


def _make_processor(monkeypatch, user_config: UserConfig) -> AccuracyRecordProcessor:
    mock_grader_cls = MagicMock()
    mock_grader_cls.return_value = MagicMock()

    mock_benchmark_cls = MagicMock()

    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_class",
        lambda plugin_type, name: mock_benchmark_cls
        if "benchmark" in str(plugin_type).lower()
        else mock_grader_cls,
    )
    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_metadata",
        lambda *_args, **_kwargs: {"default_grader": "multiple_choice"},
    )

    return AccuracyRecordProcessor(service_id="test", user_config=user_config)


def _make_problem(ground_truth: str = "A") -> BenchmarkProblem:
    return BenchmarkProblem(
        prompt="Which is correct?",
        ground_truth=ground_truth,
        task="test_task",
    )


@pytest.mark.asyncio
class TestAccuracyRecordProcessorSessionBounds:
    async def test_process_record_session_num_out_of_range_raises_value_error(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [_make_problem()]

        metadata = create_metric_metadata(session_num=1)

        with pytest.raises(
            ValueError,
            match="session_num 1 is out of range for dataset with 1 problems",
        ):
            await processor.process_record(sample_parsed_record, metadata)

    async def test_process_record_session_num_out_of_range_error_includes_counts(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [_make_problem(), _make_problem()]

        metadata = create_metric_metadata(session_num=5)

        with pytest.raises(ValueError, match="session_num 5.*2 problems"):
            await processor.process_record(sample_parsed_record, metadata)

    async def test_process_record_last_valid_session_num_succeeds(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [
            _make_problem(ground_truth="A"),
            _make_problem(ground_truth="B"),
        ]

        grading_result = GradingResult(
            correct=True,
            confidence=1.0,
            reasoning="Correct",
            extracted_answer="B",
            ground_truth="B",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        metadata = create_metric_metadata(session_num=1)
        result = await processor.process_record(sample_parsed_record, metadata)

        assert result["accuracy.correct"] == 1.0
