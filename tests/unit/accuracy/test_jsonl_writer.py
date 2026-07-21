# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import orjson
import pytest

from aiperf.accuracy.jsonl_writer import AccuracyJSONLWriter
from aiperf.accuracy.models import AccuracyRecordsData
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _record(*, timestamp_ns: int, task: str | None = "math") -> AccuracyRecordsData:
    return AccuracyRecordsData(
        session_num=0,
        worker_id="worker-0",
        benchmark_phase=CreditPhase.PROFILING,
        timestamp_ns=timestamp_ns,
        task=task,
        grader_name="exact_match",
        passed=True,
        unparsed=False,
        confidence=0.42,
        expected="A",
        actual="A",
        explanation="the answer is A",
    )


@pytest.mark.asyncio
class TestAccuracyJSONLWriter:
    async def test_disabled_raises(self) -> None:
        with pytest.raises(PostProcessorDisabled):
            AccuracyJSONLWriter(
                run=make_benchmark_run(
                    model_names=["test-model"],
                    endpoint_type=EndpointType.COMPLETIONS,
                    streaming=False,
                )
            )

    async def test_records_round_trip(self) -> None:
        run = make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
        )
        writer = AccuracyJSONLWriter(run=run)
        await writer.initialize()
        await writer.start()

        records = [_record(timestamp_ns=10), _record(timestamp_ns=20, task="algebra")]
        for record in records:
            await writer.process_record(record)
        await writer.finalize()
        await writer.stop()

        lines = writer.output_file.read_text().splitlines()
        assert len(lines) == 2

        parsed = [orjson.loads(line) for line in lines]
        assert parsed[0]["timestamp_ns"] == 10
        assert parsed[0]["explanation"] == "the answer is A"
        assert parsed[0]["confidence"] == pytest.approx(0.42)
        assert parsed[1]["task"] == "algebra"
