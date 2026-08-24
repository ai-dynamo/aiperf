# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ToolCallASTGrader`` (BFCL Prompt mode).

Pins:
1. ``correct`` and ``unparsed`` are independent signals - a decoded-but-wrong
   call is ``correct=False, unparsed=False``, and only an undecodable response
   sets ``unparsed``.
2. Upstream ``error_type`` strings normalize into the four operator-facing
   buckets, with unknown values surfacing as ``unclassified`` rather than being
   folded into a real bucket.
3. The hallucination categories are graded on call-or-no-call, before the AST
   checker is consulted.
4. Malformed ground truth degrades to an unparsed failure instead of raising
   into the record processor.

These run against the fake-bfcl harness (see ``conftest.py``); verdict parity
with the real checker is pinned separately in ``test_bfcl_ast_parity.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson
import pytest
from pytest import param

from aiperf.accuracy.graders._bfcl_compat import require_bfcl as _real_require_bfcl
from aiperf.accuracy.graders.tool_call_ast import (
    PARAM_TYPE_ERROR,
    PARAM_VALUE_ERROR,
    UNCLASSIFIED,
    WRONG_TOOL,
    ToolCallASTGrader,
    classify_error,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_WEATHER_FUNCTION = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "dict",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "days": {"type": "integer", "description": "Forecast horizon."},
            },
            "required": ["city"],
        },
    }
]


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST},
    )


def _grader() -> ToolCallASTGrader:
    return ToolCallASTGrader(run=_make_run())


def _ground_truth(
    *,
    test_category: str = "simple_python",
    language: str = "python",
    possible_answer: Any = None,
    function: Any = None,
) -> str:
    payload = {
        "id": f"{test_category}_0",
        "test_category": test_category,
        "language": language,
        "function": _WEATHER_FUNCTION if function is None else function,
        "possible_answer": (
            [{"get_weather": {"city": ["SF"], "days": [1, ""]}}]
            if possible_answer is None
            else possible_answer
        ),
    }
    return orjson.dumps(payload).decode("utf-8")


class TestASTVerdicts:
    """Decoded calls are scored by the AST checker."""

    @pytest.mark.asyncio
    async def test_grade_correct_simple_call_returns_correct(self) -> None:
        result = await _grader().grade(
            "[get_weather(city='SF', days=1)]", _ground_truth()
        )
        assert result.correct is True
        assert result.unparsed is False
        assert result.confidence == 1.0
        assert result.reasoning == "correct"

    @pytest.mark.asyncio
    async def test_grade_omitted_optional_param_returns_correct(self) -> None:
        """``""`` in the accepted list marks a parameter the model may omit."""
        result = await _grader().grade("[get_weather(city='SF')]", _ground_truth())
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_grade_wrong_function_name_returns_wrong_tool_bucket(self) -> None:
        result = await _grader().grade("[get_forecast(city='SF')]", _ground_truth())
        assert result.correct is False
        assert result.unparsed is False
        assert result.reasoning.startswith(f"{WRONG_TOOL}:")

    @pytest.mark.asyncio
    async def test_grade_wrong_param_type_returns_param_type_bucket(self) -> None:
        result = await _grader().grade(
            "[get_weather(city='SF', days='1')]", _ground_truth()
        )
        assert result.correct is False
        assert result.reasoning.startswith(f"{PARAM_TYPE_ERROR}:")

    @pytest.mark.asyncio
    async def test_grade_wrong_param_value_returns_param_value_bucket(self) -> None:
        result = await _grader().grade("[get_weather(city='LA')]", _ground_truth())
        assert result.correct is False
        assert result.reasoning.startswith(f"{PARAM_VALUE_ERROR}:")

    @pytest.mark.asyncio
    async def test_grade_missing_required_param_returns_param_value_bucket(
        self,
    ) -> None:
        result = await _grader().grade("[get_weather(days=1)]", _ground_truth())
        assert result.correct is False
        assert result.reasoning.startswith(f"{PARAM_VALUE_ERROR}:")

    @pytest.mark.asyncio
    async def test_grade_parallel_calls_out_of_order_returns_correct(self) -> None:
        """BFCL compares parallel calls without regard to order."""
        ground_truth = _ground_truth(
            test_category="parallel",
            possible_answer=[
                {"get_weather": {"city": ["SF"]}},
                {"get_weather": {"city": ["LA"]}},
            ],
        )
        result = await _grader().grade(
            "[get_weather(city='LA'), get_weather(city='SF')]", ground_truth
        )
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_grade_records_decoded_calls_as_extracted_answer(self) -> None:
        result = await _grader().grade("[get_weather(city='SF')]", _ground_truth())
        assert orjson.loads(result.extracted_answer) == [
            {"get_weather": {"city": "SF"}}
        ]


class TestDecodeAndUnparsed:
    """``unparsed`` means no call list was extractable - nothing else."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            param("Sure! I will call get_weather for you.", id="prose"),
            param("", id="empty"),
            param("   \n  ", id="whitespace"),
            param("[get_weather(city=", id="truncated"),
            param('{"name": "get_weather"}', id="json_not_call_list"),
            param("[1, 2, 3]", id="list_of_non_calls"),
        ],
    )  # fmt: skip
    async def test_grade_undecodable_response_sets_unparsed(
        self, response: str
    ) -> None:
        result = await _grader().grade(response, _ground_truth())
        assert result.unparsed is True
        assert result.correct is False
        assert result.reasoning.startswith("unparsed:")

    def test_extract_answer_undecodable_returns_empty_without_raising(self) -> None:
        assert _grader().extract_answer("not a call list") == ""

    def test_extract_answer_returns_decoded_calls_as_json(self) -> None:
        extracted = _grader().extract_answer("[get_weather(city='SF')]")
        assert orjson.loads(extracted) == [{"get_weather": {"city": "SF"}}]


class TestIrrelevanceCategories:
    """Hallucination categories are graded on call-or-no-call."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category",
        [
            param("irrelevance", id="irrelevance"),
            param("live_irrelevance", id="live_irrelevance"),
        ],
    )  # fmt: skip
    async def test_grade_irrelevance_no_call_returns_correct(
        self, category: str
    ) -> None:
        ground_truth = _ground_truth(test_category=category, possible_answer=None)
        result = await _grader().grade(
            "None of the available functions can answer that.", ground_truth
        )
        assert result.correct is True
        # A prose refusal is the expected answer here, not a format failure.
        assert result.unparsed is False

    @pytest.mark.asyncio
    async def test_grade_irrelevance_with_call_returns_should_not_have_called(
        self,
    ) -> None:
        ground_truth = _ground_truth(test_category="irrelevance", possible_answer=None)
        result = await _grader().grade("[get_weather(city='SF')]", ground_truth)
        assert result.correct is False
        assert result.unparsed is False
        assert result.reasoning.startswith("should_not_have_called:")

    @pytest.mark.asyncio
    async def test_grade_relevance_with_call_returns_correct(self) -> None:
        ground_truth = _ground_truth(
            test_category="live_relevance", possible_answer=None
        )
        result = await _grader().grade("[get_weather(city='SF')]", ground_truth)
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_grade_relevance_without_call_returns_should_have_called(
        self,
    ) -> None:
        ground_truth = _ground_truth(
            test_category="live_relevance", possible_answer=None
        )
        result = await _grader().grade("I cannot help with that.", ground_truth)
        assert result.correct is False
        assert result.reasoning.startswith("should_have_called:")


class TestErrorBucketMapping:
    """Upstream ``error_type`` strings normalize into operator-facing buckets."""

    @pytest.mark.parametrize(
        "error_type,expected",
        [
            param("type_error:simple", PARAM_TYPE_ERROR, id="type_simple"),
            param("type_error:nested", PARAM_TYPE_ERROR, id="type_nested"),
            param("type_error:java", PARAM_TYPE_ERROR, id="type_java"),
            param("type_error:js", PARAM_TYPE_ERROR, id="type_js"),
            param("value_error:others", PARAM_VALUE_ERROR, id="value_others"),
            param("value_error:dict_key", PARAM_VALUE_ERROR, id="value_dict_key"),
            param("value_error:list/tuple", PARAM_VALUE_ERROR, id="value_list"),
            param("simple_function_checker:wrong_func_name", WRONG_TOOL, id="wrong_name"),
            param("simple_function_checker:wrong_count", WRONG_TOOL, id="wrong_count"),
            param("multiple_function_checker:wrong_count", WRONG_TOOL, id="multi_count"),
            param("parallel_function_checker_no_order:cannot_find_match", WRONG_TOOL, id="no_match"),
            param("simple_function_checker:missing_required", PARAM_VALUE_ERROR, id="missing_required"),
            param("simple_function_checker:missing_optional", PARAM_VALUE_ERROR, id="missing_optional"),
            param("simple_function_checker:unexpected_param", PARAM_VALUE_ERROR, id="unexpected_param"),
            param("simple_function_checker:unclear", UNCLASSIFIED, id="unclear"),
            param("brand_new_checker:brand_new_detail", UNCLASSIFIED, id="unknown"),
            param("", UNCLASSIFIED, id="empty"),
        ],
    )  # fmt: skip
    def test_classify_error_maps_upstream_error_type_to_bucket(
        self, error_type: str, expected: str
    ) -> None:
        assert classify_error(error_type) == expected


class TestMalformedGroundTruth:
    """A bad payload degrades to unparsed rather than raising."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "ground_truth",
        [
            param("not json at all", id="not_json"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    async def test_grade_non_json_ground_truth_returns_failure(
        self, ground_truth: str
    ) -> None:
        result = await _grader().grade("[get_weather(city='SF')]", ground_truth)
        assert result.correct is False
        assert result.unparsed is True
        assert "not orjson" in result.reasoning

    @pytest.mark.asyncio
    async def test_grade_ground_truth_missing_fields_returns_failure(self) -> None:
        result = await _grader().grade(
            "[get_weather(city='SF')]", orjson.dumps({"id": "x"}).decode("utf-8")
        )
        assert result.correct is False
        assert result.unparsed is True
        assert "malformed ground_truth" in result.reasoning


class TestGraderContract:
    """Contract details the record processor and preflight depend on."""

    def test_check_available_without_bfcl_raises_actionable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.accuracy.graders import _bfcl_compat

        # Restore the real gate (conftest's autouse fixture stubs it out) and
        # make the dependency look absent.
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", _real_require_bfcl)
        monkeypatch.setattr(_bfcl_compat, "bfcl_available", lambda: False)
        with pytest.raises(RuntimeError, match=r"aiperf\[bfcl\]"):
            ToolCallASTGrader.check_available()

    @pytest.mark.asyncio
    async def test_grade_does_not_mutate_grader_state(self) -> None:
        """Records are graded concurrently; the grader must stay stateless."""
        grader = _grader()
        before = dict(grader.__dict__)
        await grader.grade("[get_weather(city='SF')]", _ground_truth())
        await grader.grade("nonsense", _ground_truth())
        assert grader.__dict__ == before


class TestCheckerCrashSafety:
    """A raising checker must not take down the daemon record processor."""

    @pytest.mark.asyncio
    async def test_grade_when_checker_raises_returns_unparsed_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Grading runs in a daemon process; an unhandled raise there hangs the
        parent waiting on records that never arrive."""
        from aiperf.accuracy.graders import _bfcl_compat

        def _explode(**_kwargs: Any) -> dict[str, Any]:
            raise TypeError("'NoneType' object is not subscriptable")

        monkeypatch.setattr(_bfcl_compat, "ast_check", _explode)

        result = await _grader().grade("[get_weather(city='SF')]", _ground_truth())

        assert result.correct is False
        assert result.unparsed is True
        assert "AST checker raised" in result.reasoning

    @pytest.mark.asyncio
    async def test_grade_with_null_possible_answer_does_not_raise(self) -> None:
        """The shape that makes upstream index into ``None``."""
        ground_truth = orjson.dumps(
            {
                "id": "simple_python_0",
                "test_category": "simple_python",
                "language": "python",
                "function": _WEATHER_FUNCTION,
                "possible_answer": None,
            }
        ).decode("utf-8")

        result = await _grader().grade("[get_weather(city='SF')]", ground_truth)

        assert result.correct is False


class TestEmptyAnswerChannelOnAbstainCategories:
    """An empty response is not an abstention."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category",
        [
            param("irrelevance", id="irrelevance"),
            param("live_irrelevance", id="live_irrelevance"),
            param("live_relevance", id="live_relevance"),
        ],
    )  # fmt: skip
    @pytest.mark.parametrize(
        "response",
        [
            param("", id="empty"),
            param("   \n ", id="whitespace"),
        ],
    )  # fmt: skip
    async def test_empty_response_is_unparsed_not_correct(
        self, category: str, response: str
    ) -> None:
        """Scoring silence as a correct abstention would inflate the
        240-problem irrelevance category whenever generations are truncated."""
        result = await _grader().grade(
            response, _ground_truth(test_category=category, possible_answer=None)
        )
        assert result.unparsed is True
        assert result.correct is False
        assert "empty answer channel" in result.reasoning

    @pytest.mark.asyncio
    async def test_prose_refusal_is_still_a_correct_abstention(self) -> None:
        """The distinction only applies to an empty channel, not to prose."""
        result = await _grader().grade(
            "I cannot answer that with the available functions.",
            _ground_truth(test_category="irrelevance", possible_answer=None),
        )
        assert result.correct is True
        assert result.unparsed is False


class TestExplanationRendering:
    """The explanation string is the export's whole triage surface."""

    @pytest.mark.asyncio
    async def test_non_list_error_field_is_still_rendered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upstream returns ``error`` as a list, but a future version returning
        a bare string must not render as ``bucket: type:`` with nothing after."""
        from aiperf.accuracy.graders import _bfcl_compat

        monkeypatch.setattr(
            _bfcl_compat,
            "ast_check",
            lambda **_kwargs: {
                "valid": False,
                "error": "Function name 'get_weather' not found.",
                "error_type": "simple_function_checker:wrong_func_name",
            },
        )

        result = await _grader().grade("[get_weather(city='SF')]", _ground_truth())

        assert result.reasoning.startswith(f"{WRONG_TOOL}:")
        assert "not found" in result.reasoning

    @pytest.mark.asyncio
    async def test_empty_error_field_still_names_the_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.accuracy.graders import _bfcl_compat

        monkeypatch.setattr(
            _bfcl_compat,
            "ast_check",
            lambda **_kwargs: {
                "valid": False,
                "error": [],
                "error_type": "type_error:simple",
            },
        )

        result = await _grader().grade("[get_weather(city='SF')]", _ground_truth())

        assert result.reasoning == f"{PARAM_TYPE_ERROR}: type_error:simple"

    def test_dumps_falls_back_to_str_for_unserializable_values(self) -> None:
        """``extracted_answer`` must never raise on its way to the export."""
        from aiperf.accuracy.graders.tool_call_ast import _dumps

        assert _dumps(object()).startswith("<object object")
