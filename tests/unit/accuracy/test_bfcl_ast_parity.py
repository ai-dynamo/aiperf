# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: our BFCL integration vs the real ``bfcl-eval``.

``ToolCallASTGrader`` does not reimplement BFCL's semantics - it decodes the
response and delegates the verdict to upstream's ``ast_checker`` through
``_bfcl_compat``. These tests lock that delegation down so it can never
silently drift into a local reimplementation or a mis-bound call:

- our ``correct`` verdict must equal upstream's ``valid`` for the same inputs,
- our normalized failure bucket must match the ``error_type`` upstream returns,
- upstream's ``ast_checker`` signature must still accept every keyword the
  compat shim binds (the reorder guard - the reason the shim never calls
  positionally),
- our category tuples must still equal upstream's category lists.

Unlike the rest of the accuracy unit tests, this file uses the real
``bfcl-eval`` on purpose: a fake harness cannot serve as a reference oracle. It
is skipped when the ``[bfcl]`` extra is not installed.

Reference:
    bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker
    bfcl_eval.model_handler.utils.ast_parse
"""

from __future__ import annotations

import inspect

import pytest
from pytest import param

# This file is a parity oracle against the real dependency; skip cleanly when
# bfcl-eval isn't installed rather than faking it.
pytest.importorskip("bfcl_eval")

from bfcl_eval.constants.category_mapping import (  # noqa: E402
    LIVE_CATEGORY,
    NON_LIVE_CATEGORY,
)
from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl_eval.model_handler.utils import ast_parse  # noqa: E402

from aiperf.accuracy.benchmarks.bfcl_ast import (  # noqa: E402
    LIVE_CATEGORIES,
    NON_LIVE_CATEGORIES,
)
from aiperf.accuracy.graders import _bfcl_compat  # noqa: E402
from aiperf.accuracy.graders.tool_call_ast import (  # noqa: E402
    PARAM_TYPE_ERROR,
    PARAM_VALUE_ERROR,
    WRONG_TOOL,
    ToolCallASTGrader,
    classify_error,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType  # noqa: E402
from tests.unit.conftest import make_benchmark_run  # noqa: E402

pytestmark = pytest.mark.requires_bfcl


def _grader() -> ToolCallASTGrader:
    return ToolCallASTGrader(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST},
        )
    )


def _ground_truth(category: str, function, gold) -> str:
    import orjson

    return orjson.dumps(
        {
            "id": f"{category}_0",
            "test_category": category,
            "language": "python",
            "function": function,
            "possible_answer": gold,
        }
    ).decode("utf-8")


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
_WEATHER_GOLD = [{"get_weather": {"city": ["SF"], "days": [1, ""]}}]

_PARALLEL_GOLD = [
    {"get_weather": {"city": ["SF"], "days": [1, ""]}},
    {"get_weather": {"city": ["LA"], "days": [2, ""]}},
]

# (response, function docs, gold, category, expected bucket or None if correct)
_GOLDEN_CASES = [
    param(
        "[get_weather(city='SF', days=1)]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        None,
        id="correct",
    ),
    param(
        "[get_weather(city='SF')]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        None,
        id="omitted_optional",
    ),
    param(
        "[get_forecast(city='SF')]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        WRONG_TOOL,
        id="wrong_tool",
    ),
    param(
        "[get_weather(city='SF', days='1')]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        PARAM_TYPE_ERROR,
        id="param_type_error",
    ),
    param(
        "[get_weather(city='LA')]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        PARAM_VALUE_ERROR,
        id="param_value_error",
    ),
    param(
        "[get_weather(days=1)]",
        _WEATHER_FUNCTION,
        _WEATHER_GOLD,
        "simple_python",
        PARAM_VALUE_ERROR,
        id="missing_required",
    ),
    param(
        "[get_weather(city='LA', days=2), get_weather(city='SF', days=1)]",
        _WEATHER_FUNCTION,
        _PARALLEL_GOLD,
        "parallel",
        None,
        id="parallel_out_of_order",
    ),
    param(
        "[get_weather(city='SF', days=1)]",
        _WEATHER_FUNCTION,
        _PARALLEL_GOLD,
        "parallel",
        WRONG_TOOL,
        id="parallel_wrong_count",
    ),
]


def _upstream_verdict(response, function, gold, category):
    """Run the exact upstream pipeline: decode, then check."""
    decoded = ast_parse(response, ReturnFormat.PYTHON)
    return ast_checker(function, decoded, gold, Language.PYTHON, category, "aiperf")


class TestVerdictParity:
    """Our verdict must equal upstream's for the same inputs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response,function,gold,category,expected_bucket", _GOLDEN_CASES
    )
    async def test_grade_matches_upstream_ast_checker(
        self, response, function, gold, category, expected_bucket
    ) -> None:
        upstream = _upstream_verdict(response, function, gold, category)
        result = await _grader().grade(
            response, _ground_truth(category, function, gold)
        )
        assert result.correct is bool(upstream["valid"])
        assert result.unparsed is False
        if expected_bucket is None:
            assert result.correct is True
        else:
            assert result.reasoning.startswith(f"{expected_bucket}:")
            assert classify_error(str(upstream["error_type"])) == expected_bucket

    @pytest.mark.asyncio
    async def test_undecodable_response_is_unparsed_not_a_verdict(self) -> None:
        """Upstream raises on prose; we must translate that into unparsed."""
        with pytest.raises((SyntaxError, ValueError, AssertionError)):
            ast_parse("Sure, I'll get the weather.", ReturnFormat.PYTHON)
        result = await _grader().grade(
            "Sure, I'll get the weather.",
            _ground_truth("simple_python", _WEATHER_FUNCTION, _WEATHER_GOLD),
        )
        assert result.unparsed is True
        assert result.correct is False


class TestCompatShimBinding:
    """The shim's keyword binding is the guard against a silent reorder."""

    def test_ast_checker_signature_accepts_every_bound_keyword(self) -> None:
        parameters = inspect.signature(ast_checker).parameters
        for canonical, aliases in _bfcl_compat._ARG_ALIASES.items():
            assert any(alias in parameters for alias in aliases), (
                f"no alias of {canonical!r} matches upstream's signature "
                f"{list(parameters)}"
            )

    def test_ast_check_returns_upstream_verdict_shape(self) -> None:
        result = _bfcl_compat.ast_check(
            func_description=_WEATHER_FUNCTION,
            model_output=[{"get_weather": {"city": "SF"}}],
            possible_answer=_WEATHER_GOLD,
            language="python",
            test_category="simple_python",
            model_name="aiperf",
        )
        assert set(result) >= {"valid", "error"}
        assert result["valid"] is True

    def test_ast_check_omits_error_type_on_a_passing_verdict(self) -> None:
        """Upstream only sets ``error_type`` on some paths, and not on success.

        Pinned because the grader reads it with ``.get`` for exactly this
        reason: indexing it would crash the record processor on every correct
        answer.
        """
        passing = _bfcl_compat.ast_check(
            func_description=_WEATHER_FUNCTION,
            model_output=[{"get_weather": {"city": "SF"}}],
            possible_answer=_WEATHER_GOLD,
            language="python",
            test_category="simple_python",
            model_name="aiperf",
        )
        failing = _bfcl_compat.ast_check(
            func_description=_WEATHER_FUNCTION,
            model_output=[{"get_weather": {"city": "LA"}}],
            possible_answer=_WEATHER_GOLD,
            language="python",
            test_category="simple_python",
            model_name="aiperf",
        )
        assert "error_type" not in passing
        assert failing["error_type"]

    def test_decode_calls_matches_upstream_ast_parse(self) -> None:
        response = "[get_weather(city='SF', days=1)]"
        assert _bfcl_compat.decode_calls(response, "python") == ast_parse(
            response, ReturnFormat.PYTHON
        )


class TestCategoryParity:
    """Our category tuples mirror upstream's lists."""

    def test_non_live_categories_match_upstream(self) -> None:
        assert tuple(NON_LIVE_CATEGORY) == NON_LIVE_CATEGORIES

    def test_live_categories_match_upstream(self) -> None:
        assert tuple(LIVE_CATEGORY) == LIVE_CATEGORIES

    def test_single_turn_categories_resolve_through_compat(self) -> None:
        assert _bfcl_compat.single_turn_categories() == (
            NON_LIVE_CATEGORIES + LIVE_CATEGORIES
        )


class TestBundledDataLayout:
    """The wheel really does ship the dataset where the loader looks for it."""

    def test_package_root_locates_the_installed_package(self) -> None:
        root = _bfcl_compat.package_root()
        assert root.is_dir()
        assert root.name == "bfcl_eval"

    def test_bundled_question_and_answer_files_exist(self) -> None:
        """The whole no-download design rests on this layout (gorilla PR #504)."""
        prefix = _bfcl_compat.version_prefix()
        questions = _bfcl_compat.data_dir() / f"{prefix}_simple_python.json"
        answers = _bfcl_compat.possible_answer_dir() / f"{prefix}_simple_python.json"
        assert questions.is_file()
        assert answers.is_file()

    def test_every_default_category_ships_a_question_file(self) -> None:
        """A missing file would surface as a load-time error mid-run; catching
        it here names the category instead."""
        from aiperf.accuracy.benchmarks.bfcl_ast import DEFAULT_CATEGORIES

        prefix = _bfcl_compat.version_prefix()
        missing = [
            category
            for category in DEFAULT_CATEGORIES
            if not (_bfcl_compat.data_dir() / f"{prefix}_{category}.json").is_file()
        ]
        assert not missing
