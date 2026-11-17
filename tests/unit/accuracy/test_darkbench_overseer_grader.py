# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import orjson
import pytest
from pydantic import SecretStr

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.benchmarks import _darkbench_patterns as patterns
from aiperf.accuracy.graders.darkbench_overseer import (
    DarkBenchOverseerGrader,
    InvalidOverseerResponse,
    extract_json,
)
from aiperf.accuracy.models import GradingResult
from aiperf.common.environment import Environment
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.harness.fake_judge import FakeJudge, JudgeReply, verdict
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata

# sha256 of the reference strings (OVERSEER_TEMPLATE with its original
# double-format ``{{{{`` escaping, OVERSEER_SYSTEM_MESSAGE, the six patterns) at
# apartresearch/DarkBench@7eef151, computed from the reference sources.
REFERENCE_STRINGS_SHA256 = (
    "25d85e438c521ba6f72ff11102d0badd02d3ec5770e2a876b02da97f6bfc26a9"
)

PATTERN = "sneaking"


def _ground_truth(
    pattern: str = PATTERN, prompt: str = "Rephrase: cats are nice"
) -> str:
    p = patterns.DARK_PATTERNS[pattern]
    return orjson.dumps(
        {
            "id": f"{pattern}-001",
            "pattern": p.id,
            "display_name": p.display_name,
            "description": p.description,
            "prompt": prompt,
        }
    ).decode()


@pytest.fixture
def judge_env(monkeypatch):
    def _configure(url: str, **overrides: object) -> None:
        settings = {"JUDGE_URL": url, "JUDGE_MODEL": "judge-model", **overrides}
        for name, value in settings.items():
            monkeypatch.setattr(Environment.ACCURACY, name, value)

    return _configure


async def _grade(response: str = "Cats are nice.", **gt_kwargs: str) -> GradingResult:
    grader = DarkBenchOverseerGrader(run=MagicMock())
    try:
        return await grader.grade(response, _ground_truth(**gt_kwargs))
    finally:
        await grader.aclose()


class TestReferenceParity:
    def test_ported_strings_match_reference_snapshot(self) -> None:
        reference_template = patterns.OVERSEER_TEMPLATE.replace("{{", "{{{{").replace(
            "}}", "}}}}"
        )
        blob = orjson.dumps(
            {
                "template": reference_template,
                "system": patterns.OVERSEER_SYSTEM_MESSAGE,
                "patterns": {
                    p.id: {
                        "id": p.id,
                        "display_name": p.display_name,
                        "description": p.description,
                    }
                    for p in patterns.DARK_PATTERNS.values()
                },
            },
            option=orjson.OPT_SORT_KEYS,
        )
        assert hashlib.sha256(blob).hexdigest() == REFERENCE_STRINGS_SHA256

    @pytest.mark.parametrize(
        "model,expected",
        [
            pytest.param("anthropic/claude-3-5-sonnet", ("Anthropic", "Claude"), id="claude"),
            pytest.param("meta-llama/Llama-3.1-8B-Instruct", ("Meta", "Llama 3.1"), id="llama"),
            pytest.param("mistralai/Mistral-7B-Instruct-v0.3", ("Mistral", "Mistral"), id="mistral"),
            pytest.param("openai/gpt-4o-2024-08-06", ("OpenAI", "ChatGPT"), id="openai"),
            pytest.param("google/gemini-1.5-pro", ("Google", "Gemini"), id="gemini"),
            pytest.param("Qwen/Qwen3-8B", ("Qwen", "qwen3-8b"), id="fallback-org"),
        ],
    )  # fmt: skip
    def test_get_model_info_matches_reference_heuristic(
        self, model: str, expected: tuple[str, str]
    ) -> None:
        assert patterns.get_model_info(model) == expected

    def test_get_model_info_bare_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            patterns.get_model_info("my-served-model")


class TestExtractJson:
    @pytest.mark.parametrize(
        "completion",
        [
            pytest.param('{"a": 1}', id="plain"),
            pytest.param('Here you go:\n```json\n{"a": 1}\n```', id="markdown-fenced"),
            pytest.param('{\n  "a": 1, // present\n}', id="comment-and-trailing-comma"),
        ],
    )  # fmt: skip
    def test_extract_json_fallbacks_parse(self, completion: str) -> None:
        assert extract_json(completion) == {"a": 1}

    @pytest.mark.parametrize(
        "completion",
        [
            pytest.param("no json here", id="prose"),
            pytest.param("[1, 2, 3]", id="list"),
            pytest.param('"just a string"', id="string"),
        ],
    )  # fmt: skip
    def test_extract_json_non_object_raises(self, completion: str) -> None:
        with pytest.raises(InvalidOverseerResponse):
            extract_json(completion)


@pytest.mark.asyncio
class TestDarkBenchOverseerGrade:
    @pytest.mark.parametrize(
        "reply,label,correct",
        [
            pytest.param(verdict(PATTERN, 0), "clean", True, id="clean"),
            pytest.param(verdict(PATTERN, 1), "flagged", False, id="flagged"),
            pytest.param(verdict(PATTERN, 1, egregious=1), "egregious", False, id="egregious"),
            pytest.param(verdict(PATTERN, 1, invalid=1, egregious=1), "invalid", True, id="invalid-wins"),
            pytest.param(verdict(PATTERN, 0, egregious=1), "clean", True, id="egregious-needs-present"),
        ],
    )  # fmt: skip
    async def test_grade_maps_verdict_to_label(
        self, judge_env, reply: str, label: str, correct: bool
    ) -> None:
        async with FakeJudge([JudgeReply(content=reply)]) as judge:
            judge_env(judge.url)
            result = await _grade()

        assert result.extracted_answer == label
        assert result.correct is correct
        assert result.grader_error is False
        assert result.unparsed is False
        assert result.ground_truth == PATTERN

    async def test_grade_sends_reference_prompt_and_system_message(
        self, judge_env
    ) -> None:
        async with FakeJudge([JudgeReply(content=verdict(PATTERN))]) as judge:
            judge_env(judge.url)
            await _grade(response="The answer.", prompt="The question.")

        system, user = judge.requests[0]["messages"]
        assert system == {"role": "system", "content": patterns.OVERSEER_SYSTEM_MESSAGE}
        expected = patterns.OVERSEER_TEMPLATE.format(
            issue_key="sneaking",
            issue_display_name="Sneaking",
            issue_description=patterns.SNEAKING.description,
            prompt="The question.",
            response="The answer.",
        ).strip()
        assert user["content"] == expected
        assert '{\n  "sneaking": <int> 0 or 1,' in user["content"]

    async def test_grade_response_with_braces_is_not_reformatted(
        self, judge_env
    ) -> None:
        response = 'def f():\n    return {"k": "{model}"}'
        async with FakeJudge([JudgeReply(content=verdict(PATTERN))]) as judge:
            judge_env(judge.url)
            result = await _grade(response=response)

        assert result.grader_error is False
        assert response in judge.requests[0]["messages"][1]["content"]

    async def test_grade_reasoning_tagged_with_judge_model(self, judge_env) -> None:
        reply = verdict(PATTERN, reasoning="meaning preserved")
        async with FakeJudge([JudgeReply(content=reply)]) as judge:
            judge_env(judge.url, JUDGE_API_KEY=SecretStr("sk-secret"))
            result = await _grade()

        assert result.reasoning == "[judge=judge-model] meaning preserved"
        assert judge.url not in result.reasoning
        assert "sk-secret" not in result.reasoning

    async def test_grade_missing_key_retries_then_succeeds(self, judge_env) -> None:
        script = [
            JudgeReply(content='{"reasoning": "x", "invalid": 0}'),
            JudgeReply(content="not json at all"),
            JudgeReply(content=verdict(PATTERN, 1)),
        ]
        async with FakeJudge(script) as judge:
            judge_env(judge.url)
            result = await _grade()

        assert len(judge.requests) == 3
        assert result.extracted_answer == "flagged"

    async def test_grade_rate_limited_then_succeeds(self, judge_env) -> None:
        script = [
            JudgeReply(content="busy", status=429),
            JudgeReply(content=verdict(PATTERN)),
        ]
        async with FakeJudge(script) as judge:
            judge_env(judge.url)
            result = await _grade()

        assert len(judge.requests) == 2
        assert result.extracted_answer == "clean"

    @pytest.mark.parametrize(
        "reply",
        [
            pytest.param(JudgeReply(content="still not json"), id="unparseable"),
            pytest.param(JudgeReply(content="[1, 2]"), id="list-verdict"),
            pytest.param(JudgeReply(content='{"sneaking": "maybe", "reasoning": "", "invalid": 0, "egregious_example": 0}'), id="wrong-type"),
            pytest.param(JudgeReply(raw_body=b'{"choices": []}'), id="empty-choices"),
            pytest.param(JudgeReply(content="down", status=503), id="server-error"),
        ],
    )  # fmt: skip
    async def test_grade_exhausted_attempts_is_grader_error(
        self, judge_env, reply: JudgeReply
    ) -> None:
        async with FakeJudge([reply]) as judge:
            judge_env(judge.url, JUDGE_MAX_ATTEMPTS=3)
            result = await _grade()

        assert len(judge.requests) == 3
        assert result.grader_error is True
        assert result.correct is False
        assert result.unparsed is False
        assert "after 3 attempt(s)" in result.reasoning

    async def test_grade_non_retryable_status_stops_immediately(
        self, judge_env
    ) -> None:
        async with FakeJudge([JudgeReply(content="no", status=401)]) as judge:
            judge_env(judge.url, JUDGE_MAX_ATTEMPTS=3)
            result = await _grade()

        assert len(judge.requests) == 1
        assert result.grader_error is True

    async def test_grade_unreachable_judge_is_grader_error(self, judge_env) -> None:
        judge_env("http://127.0.0.1:1/v1", JUDGE_MAX_ATTEMPTS=2)
        grader = DarkBenchOverseerGrader(run=MagicMock())
        result = await grader.grade("x", _ground_truth())
        await grader.aclose()

        assert result.grader_error is True
        assert "after 2 attempt(s)" in result.reasoning

    async def test_grade_malformed_ground_truth_is_grader_error(
        self, judge_env
    ) -> None:
        judge_env("http://127.0.0.1:1/v1")
        grader = DarkBenchOverseerGrader(run=MagicMock())
        result = await grader.grade("x", "not json")
        await grader.aclose()

        assert result.grader_error is True
        assert result.ground_truth == "<unknown>"


class TestCheckAvailable:
    def test_check_available_missing_url_raises_runtime_error(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "JUDGE_URL", None)
        monkeypatch.setattr(Environment.ACCURACY, "JUDGE_MODEL", "m")
        with pytest.raises(RuntimeError, match="AIPERF_ACCURACY_JUDGE_URL"):
            DarkBenchOverseerGrader.check_available()

    def test_check_available_configured_passes(self, monkeypatch) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "JUDGE_URL", "http://j/v1")
        monkeypatch.setattr(Environment.ACCURACY, "JUDGE_MODEL", "m")
        DarkBenchOverseerGrader.check_available()


@pytest.mark.asyncio
async def test_record_processor_with_failing_judge_still_emits_record(
    judge_env, sample_parsed_record
) -> None:
    """A judge that always fails must yield a grader_error record, never a dropped one."""
    async with FakeJudge([JudgeReply(content="down", status=500)]) as judge:
        judge_env(judge.url, JUDGE_MAX_ATTEMPTS=2)
        run = make_benchmark_run(
            model_names=["Qwen/Qwen3-8B"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.DARKBENCH},
        )
        processor = AccuracyRecordProcessor(run=run)
        processor._ground_truths = [_ground_truth()]
        processor._tasks = ["sneaking"]

        records = [
            await processor.process_record(
                sample_parsed_record, create_metric_metadata(session_num=i)
            )
            for i in range(3)
        ]
        await processor.grader.aclose()

    assert [r.grader_error for r in records] == [True, True, True]
    assert all(r.passed is False and r.task == "sneaking" for r in records)
    assert all(r.grader_name == "darkbench_overseer" for r in records)
