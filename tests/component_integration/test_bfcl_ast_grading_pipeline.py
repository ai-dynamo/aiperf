# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end grading pipeline for ``bfcl_ast``: responses -> summary + JSONL.

The unit tests exercise the loader and the grader in isolation. What they can't
show is the part that makes a BFCL run *readable*: that each record's BFCL
category survives as ``task`` all the way into the per-task rollup, and that
``unparsed`` accumulates as its own rate alongside accuracy rather than being
folded into it. Those two properties are the whole reason the benchmark labels
records the way it does, and they cross four components to hold
(``AccuracyRecordProcessor`` -> ``AccuracyRecordsData`` -> ``AccuracyAccumulator``
-> ``AccuracyJSONLWriter``), so nothing below the pipeline level can catch a
regression in them.

Canned responses stand in for a served model - the four outcomes an operator
triages by: a correct call, a wrong tool, a wrong parameter value, and a
response with no extractable call at all.

Runs against the fake-bfcl harness so it works in the default test environment;
the ``[bfcl]`` extra cannot be installed alongside ``[accuracy]`` (bfcl-eval
pins ``numpy==1.26.4`` against lighteval's ``numpy>=2``). Verdict parity with
the real checker is pinned in ``tests/unit/accuracy/test_bfcl_ast_parity.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.accuracy.accumulator import AccuracyAccumulator
from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.graders import _bfcl_compat
from aiperf.accuracy.graders.tool_call_ast import ToolCallASTGrader
from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata
from aiperf.common.models.record_models import (
    ParsedResponse,
    ParsedResponseRecord,
    TextResponseData,
)
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    DatasetSamplingStrategy,
    EndpointType,
)
from tests.harness import fake_bfcl
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

pytestmark = pytest.mark.component_integration

_FUNCTION = fake_bfcl.DATA_ROWS["simple_python"][0]["function"]
_SIMPLE_GOLD = fake_bfcl.POSSIBLE_ANSWER_ROWS["simple_python"][0]["ground_truth"]
_PARALLEL_GOLD = fake_bfcl.POSSIBLE_ANSWER_ROWS["parallel"][0]["ground_truth"]

# (category, gold, response, expected passed, expected unparsed)
_CANNED_RUN: list[tuple[str, Any, str, bool, bool]] = [
    ("simple_python", _SIMPLE_GOLD, "[get_weather(city='SF')]", True, False),
    ("simple_python", _SIMPLE_GOLD, "[get_forecast(city='SF')]", False, False),
    ("simple_python", _SIMPLE_GOLD, "[get_weather(city='Paris')]", False, False),
    ("simple_python", _SIMPLE_GOLD, "Sure, let me look that up!", False, True),
    (
        "parallel",
        _PARALLEL_GOLD,
        "[get_weather(city='LA'), get_weather(city='SF')]",
        True,
        False,
    ),
    ("parallel", _PARALLEL_GOLD, "I cannot do that.", False, True),
    ("irrelevance", None, "None of these functions apply.", True, False),
    ("irrelevance", None, "[get_weather(city='SF')]", False, False),
]


@pytest.fixture(autouse=True)
def _fake_bfcl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``_bfcl_compat`` at the fake harness (see module docstring)."""
    monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
    monkeypatch.setattr(_bfcl_compat, "bfcl_available", lambda: True)

    def _decode_calls(response_text: str, language: str):
        if not response_text or not response_text.strip():
            raise _bfcl_compat.BFCLDecodeError("empty answer channel")
        try:
            return fake_bfcl.ast_parse(response_text, language)
        except Exception as e:
            raise _bfcl_compat.BFCLDecodeError(f"{type(e).__name__}: {e}") from e

    monkeypatch.setattr(_bfcl_compat, "decode_calls", _decode_calls)
    monkeypatch.setattr(
        _bfcl_compat, "ast_check", lambda **kwargs: fake_bfcl.ast_checker(**kwargs)
    )


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST},
    )


def _ground_truth(category: str, gold: Any) -> str:
    return orjson.dumps(
        {
            "id": f"{category}_0",
            "test_category": category,
            "language": "python",
            "function": _FUNCTION,
            "possible_answer": gold,
        }
    ).decode("utf-8")


def _response_record(text: str) -> ParsedResponseRecord:
    record = MagicMock(spec=ParsedResponseRecord)
    record.content_responses = [
        ParsedResponse(perf_ns=0, data=TextResponseData(text=text))
    ]
    return record


def _processor(run: BenchmarkRun) -> AccuracyRecordProcessor:
    processor = AccuracyRecordProcessor(run=run, service_id="test")
    # The plugin registry hands back the real grader; assert that rather than
    # injecting one, so a mis-registered default_grader fails here.
    assert isinstance(processor.grader, ToolCallASTGrader)
    processor.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id=f"conv-{i}",
                    accuracy_ground_truth=_ground_truth(category, gold),
                    accuracy_task=category,
                )
                for i, (category, gold, *_rest) in enumerate(_CANNED_RUN)
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )
    return processor


async def _grade_canned_run(run: BenchmarkRun):
    processor = _processor(run)
    records = []
    for session_num, (_category, _gold, response, *_expected) in enumerate(_CANNED_RUN):
        records.append(
            await processor.process_record(
                _response_record(response),
                create_metric_metadata(
                    session_num=session_num,
                    conversation_id=f"conv-{session_num}",
                    request_end_ns=1_000_000_000 + session_num,
                ),
            )
        )
    return records


class TestGradedRecords:
    """Each canned response is graded as the operator would expect."""

    @pytest.mark.asyncio
    async def test_each_response_gets_its_expected_verdict(self) -> None:
        records = await _grade_canned_run(_make_run())
        actual = [(r.task, r.passed, r.unparsed) for r in records]
        expected = [
            (category, passed, unparsed)
            for category, _gold, _response, passed, unparsed in _CANNED_RUN
        ]
        assert actual == expected

    @pytest.mark.asyncio
    async def test_failure_modes_are_bucketed_in_the_explanation(self) -> None:
        """The bucket prefix is what makes the JSONL export triageable."""
        records = await _grade_canned_run(_make_run())
        buckets = [r.explanation.split(":")[0] for r in records]
        assert buckets == [
            "correct",
            "wrong_tool",
            "param_value_error",
            "unparsed",
            "correct",
            "unparsed",
            "correct",
            "should_not_have_called",
        ]

    @pytest.mark.asyncio
    async def test_grader_name_is_recorded_on_every_record(self) -> None:
        records = await _grade_canned_run(_make_run())
        assert {r.grader_name for r in records} == {"tool_call_ast"}


class TestPerCategoryRollup:
    """``task`` survives into the per-task summary the console/CSV render."""

    @pytest.mark.asyncio
    async def test_summary_breaks_down_by_bfcl_category(self) -> None:
        run = _make_run()
        accumulator = AccuracyAccumulator(run=run)
        for record in await _grade_canned_run(run):
            await accumulator.process_record(record)

        summary = await accumulator.export_results(
            ExportContext(phase=CreditPhase.PROFILING)
        )

        assert set(summary.per_task) == {"simple_python", "parallel", "irrelevance"}
        assert summary.per_task["simple_python"].total == 4
        assert summary.per_task["simple_python"].passed == 1
        assert summary.per_task["parallel"].passed == 1
        assert summary.per_task["irrelevance"].passed == 1
        assert summary.total_evaluated == len(_CANNED_RUN)
        assert summary.total_passed == 3

    @pytest.mark.asyncio
    async def test_unparsed_is_counted_separately_from_accuracy(self) -> None:
        """A malformed response must not be indistinguishable from a wrong call."""
        run = _make_run()
        accumulator = AccuracyAccumulator(run=run)
        for record in await _grade_canned_run(run):
            await accumulator.process_record(record)

        summary = await accumulator.export_results(
            ExportContext(phase=CreditPhase.PROFILING)
        )

        assert summary.overall_unparsed == 2
        assert summary.per_task["simple_python"].unparsed == 1
        assert summary.per_task["parallel"].unparsed == 1
        # Emitting no call on an irrelevance question is a correct answer, not
        # a formatting failure.
        assert summary.per_task["irrelevance"].unparsed == 0

    @pytest.mark.asyncio
    async def test_summary_emits_per_task_metric_results(self) -> None:
        """The legacy ``accuracy.*`` tags the exporters render."""
        run = _make_run()
        accumulator = AccuracyAccumulator(run=run)
        for record in await _grade_canned_run(run):
            await accumulator.process_record(record)

        summary = await accumulator.export_results(
            ExportContext(phase=CreditPhase.PROFILING)
        )
        tags = {result.tag for result in summary.to_metric_results()}

        assert "accuracy.task.simple_python" in tags
        assert "accuracy.unparsed.task.parallel" in tags


class TestJSONLExport:
    """Per-record detail reaches ``accuracy_export.jsonl``."""

    @pytest.mark.asyncio
    async def test_records_serialize_with_task_and_bucketed_explanation(
        self, tmp_path: Path
    ) -> None:
        from aiperf.accuracy.jsonl_writer import AccuracyJSONLWriter

        run = _make_run()
        records = await _grade_canned_run(run)
        # ``accuracy_export_jsonl_file`` is derived from the artifact dir.
        run.cfg.artifacts.dir = tmp_path

        writer = AccuracyJSONLWriter(run=run)
        await writer.initialize()
        await writer.start()
        for record in records:
            await writer.process_record(record)
        await writer.finalize()
        await writer.stop()

        lines = [
            orjson.loads(line)
            for line in writer.output_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == len(_CANNED_RUN)
        assert [line["task"] for line in lines] == [c[0] for c in _CANNED_RUN]
        assert lines[1]["explanation"].startswith("wrong_tool:")
        assert lines[3]["unparsed"] is True
        # record_type is a wire-only discriminator, excluded from the export.
        assert "record_type" not in lines[0]
