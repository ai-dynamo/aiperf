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

import ast
import builtins
import inspect
import os

import pytest
from pytest import param

# This file is a parity oracle against the real dependency; skip cleanly when
# bfcl-eval isn't installed rather than faking it.
pytest.importorskip("bfcl_eval")

import orjson  # noqa: E402
from bfcl_eval.constants.category_mapping import (  # noqa: E402
    LIVE_CATEGORY,
    NON_LIVE_CATEGORY,
)
from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl_eval.model_handler.utils import (  # noqa: E402
    ast_parse,
    default_decode_ast_prompting,
    resolve_ast_by_type,
)

from aiperf.accuracy.benchmarks.bfcl_ast import (  # noqa: E402
    DEFAULT_CATEGORIES,
    LIVE_CATEGORIES,
    NON_LIVE_CATEGORIES,
)
from aiperf.accuracy.graders import _bfcl_compat  # noqa: E402
from aiperf.accuracy.graders.tool_call_ast import (  # noqa: E402
    CHECKER_MODEL_NAME,
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
    return ast_checker(
        function, decoded, gold, Language.PYTHON, category, CHECKER_MODEL_NAME
    )


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
            model_name=CHECKER_MODEL_NAME,
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
            model_name=CHECKER_MODEL_NAME,
        )
        failing = _bfcl_compat.ast_check(
            func_description=_WEATHER_FUNCTION,
            model_output=[{"get_weather": {"city": "LA"}}],
            possible_answer=_WEATHER_GOLD,
            language="python",
            test_category="simple_python",
            model_name=CHECKER_MODEL_NAME,
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
        prefix = _bfcl_compat.version_prefix()
        missing = [
            category
            for category in DEFAULT_CATEGORIES
            if not (_bfcl_compat.data_dir() / f"{prefix}_{category}.json").is_file()
        ]
        assert not missing


class TestDottedFunctionNames:
    """Dotted gold function names, the case that reaches upstream's registry.

    ``convert_func_name`` indexes ``MODEL_CONFIG_MAPPING`` with a bare
    subscript whenever the gold function name contains a dot, and it runs
    unconditionally before the name match. Roughly a third of the gradeable
    dataset has such a name, so an unregistered ``CHECKER_MODEL_NAME`` raises
    ``KeyError`` there and the crash guard turns it into a wall of failed
    records. Every fixture elsewhere in this file uses ``get_weather``, which
    has no dot, so nothing else here can catch it.
    """

    _MATH_FUNCTION = [
        {
            "name": "math.factorial",
            "description": "Calculate the factorial of a number.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "number": {"type": "integer", "description": "The number."}
                },
                "required": ["number"],
            },
        }
    ]
    _MATH_GOLD = [{"math.factorial": {"number": [5]}}]

    def test_checker_model_name_is_registered_upstream(self) -> None:
        """The guard that would have caught this before it shipped."""
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        from aiperf.accuracy.graders.tool_call_ast import CHECKER_MODEL_NAME

        assert CHECKER_MODEL_NAME in MODEL_CONFIG_MAPPING

    def test_checker_model_name_leaves_dotted_names_untouched(self) -> None:
        """``underscore_to_dot=True`` would rewrite ``math.factorial`` to
        ``math_factorial`` and fail every dotted-name entry on the name match."""
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        from aiperf.accuracy.graders.tool_call_ast import CHECKER_MODEL_NAME

        assert MODEL_CONFIG_MAPPING[CHECKER_MODEL_NAME].underscore_to_dot is False

    @pytest.mark.asyncio
    async def test_grade_dotted_function_name_matches_upstream(self) -> None:
        upstream = _upstream_verdict(
            "[math.factorial(number=5)]",
            self._MATH_FUNCTION,
            self._MATH_GOLD,
            "simple_python",
        )
        result = await _grader().grade(
            "[math.factorial(number=5)]",
            _ground_truth("simple_python", self._MATH_FUNCTION, self._MATH_GOLD),
        )
        assert upstream["valid"] is True
        assert result.correct is True
        assert result.unparsed is False

    @pytest.mark.asyncio
    async def test_grade_dotted_name_from_the_bundled_dataset(self) -> None:
        """``simple_python_1``'s gold function really is ``math.factorial``.

        Reads the installed wheel rather than a local fixture, so a dataset
        reshuffle that removes dotted names from this category is visible.
        """
        prefix = _bfcl_compat.version_prefix()
        entries = {
            orjson.loads(line)["id"]: orjson.loads(line)
            for line in (_bfcl_compat.data_dir() / f"{prefix}_simple_python.json")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        answers = {
            orjson.loads(line)["id"]: orjson.loads(line)["ground_truth"]
            for line in (
                _bfcl_compat.possible_answer_dir() / f"{prefix}_simple_python.json"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        entry, gold = entries["simple_python_1"], answers["simple_python_1"]
        assert any("." in name for call in gold for name in call)

        # Build the call straight from the gold answer: this isolates the
        # registry lookup from any text-rendering of the model's response.
        decoded = [
            {name: {k: v[0] for k, v in args.items() if v[0] != ""}}
            for call in gold
            for name, args in call.items()
        ]
        verdict = _bfcl_compat.ast_check(
            func_description=entry["function"],
            model_output=decoded,
            possible_answer=gold,
            language="python",
            test_category="simple_python",
            model_name=CHECKER_MODEL_NAME,
        )
        assert verdict["valid"] is True


class TestDecoderDoesNotExecuteModelOutput:
    """The decoder must never execute the response it is decoding.

    ``bfcl-eval``'s own ``resolve_ast_by_type`` resolves ``BinOp`` and
    ``Lambda`` argument values with ``eval(ast.unparse(node))``. The AST-node
    check in front of it only restricts the *shape* of the expression; the
    source text handed to ``eval`` is then re-parsed and executed with no
    further restriction. Inference-server output is attacker-controlled in
    aiperf's threat model, so routing it through that function would let the
    model under test run Python with aiperf's credentials and filesystem
    access.

    ``_bfcl_compat`` therefore reimplements the value-resolution step without
    ``eval``. This oracle documents that the upstream function really does
    execute, so the local decoder is load-bearing, not redundant. The local
    decoder's refusals and bounds are stdlib-only and are pinned without
    ``bfcl-eval`` in ``test_bfcl_safe_decoder.py``.
    """

    @staticmethod
    def _tripwire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record any execution of the payload's side effect."""
        fired: list[str] = []
        real_print = builtins.print

        def watched_print(*args: object, **kwargs: object) -> None:
            if args and args[0] == "PROBE":
                fired.append("print")
            real_print(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "print", watched_print)
        monkeypatch.setattr(os, "system", lambda cmd: fired.append("os.system") or 0)
        return fired

    def test_upstream_resolver_executes_binop_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Oracle: upstream really does execute, which is why we do not call it.

        If a future ``bfcl-eval`` stops executing here this test fails, which
        is the signal to re-evaluate whether the local decoder is still needed
        - not a reason to route model output back through upstream.
        """
        fired = self._tripwire(monkeypatch)
        node = ast.parse(
            "(__import__('builtins').print('PROBE') or 10)+0", mode="eval"
        ).body
        resolve_ast_by_type(node)
        assert fired, (
            "upstream resolve_ast_by_type no longer executes BinOp arguments; "
            "re-check whether the non-executing decoder is still required"
        )


_NORMALIZATION_CASES = [
    param("[get_weather(city='SF')]", "python", id="python_bracketed"),
    param("get_weather(city='SF')", "python", id="python_unbracketed"),
    param("```\n[get_weather(city='SF')]\n```", "python", id="python_fenced"),
    param(
        "```python\n[get_weather(city='SF')]\n```", "python", id="python_labelled_fence"
    ),
    param("```", "python", id="python_bare_fence"),
    param("I cannot help with that.", "python", id="python_prose"),
    param(
        "GeometryPresentation.createPresentation(controller=mapController, parent=mapArea)",
        "java",
        id="java_unbracketed",
    ),
    param(
        "```\n[GeometryPresentation.createPresentation(controller=mapController, parent=mapArea)]\n```",
        "java",
        id="java_fenced",
    ),
    param(
        "validateUserInput(inputField=userInputField, isComplete=true)",
        "javascript",
        id="javascript_unbracketed",
    ),
    param(
        "```\nvalidateUserInput(inputField=userInputField, isComplete=true)\n```",
        "javascript",
        id="javascript_fenced",
    ),
]

_RETURN_FORMATS = {
    "python": ReturnFormat.PYTHON,
    "java": ReturnFormat.JAVA,
    "javascript": ReturnFormat.JAVASCRIPT,
}
_LANGUAGES = {
    "python": Language.PYTHON,
    "java": Language.JAVA,
    "javascript": Language.JAVASCRIPT,
}


def _bundled_entry(category: str, index: int = 0) -> tuple[list, list]:
    """``(function docs, possible answer)`` for one entry of the bundled data."""
    prefix = _bfcl_compat.version_prefix()
    entry = orjson.loads(
        (_bfcl_compat.data_dir() / f"{prefix}_{category}.json")
        .read_text()
        .splitlines()[index]
    )
    answer = orjson.loads(
        (_bfcl_compat.possible_answer_dir() / f"{prefix}_{category}.json")
        .read_text()
        .splitlines()[index]
    )
    return entry["function"], answer["ground_truth"]


class TestPromptModeNormalizationParity:
    """Decoding must match upstream's Prompt-mode handler, not bare ``ast_parse``.

    Upstream's handler runs ``default_decode_ast_prompting``: strip backticks,
    newlines and spaces, then bracket, then ``ast_parse``. Its Java and
    JavaScript branches drop the first and last character on the assumption
    that the bracketing already happened.
    """

    @pytest.mark.parametrize("response,language", _NORMALIZATION_CASES)
    def test_decode_calls_matches_upstream_prompting_decoder(
        self, response: str, language: str
    ) -> None:
        try:
            expected = default_decode_ast_prompting(response, _RETURN_FORMATS[language])
        except Exception:
            with pytest.raises(_bfcl_compat.BFCLDecodeError):
                _bfcl_compat.decode_calls(response, language)
            return
        assert _bfcl_compat.decode_calls(response, language) == expected

    @pytest.mark.asyncio
    async def test_grade_fenced_call_on_irrelevance_matches_upstream(self) -> None:
        """Upstream decodes the fenced call, so it is a hallucinated call."""
        response = "```\n[get_weather(city='SF')]\n```"
        assert default_decode_ast_prompting(response, ReturnFormat.PYTHON)
        result = await _grader().grade(
            response, _ground_truth("irrelevance", _WEATHER_FUNCTION, None)
        )
        assert result.correct is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category,language,response",
        [
            param(
                "simple_java",
                "java",
                "GeometryPresentation.createPresentation(controller=mapController, parent=mapArea)",
                id="java",
            ),
            param(
                "simple_javascript",
                "javascript",
                "validateUserInput(inputField=userInputField, isComplete=true)",
                id="javascript",
            ),
        ],
    )  # fmt: skip
    async def test_grade_unbracketed_call_matches_upstream_verdict(
        self, category: str, language: str, response: str
    ) -> None:
        function, gold = _bundled_entry(category)
        upstream = ast_checker(
            function,
            default_decode_ast_prompting(response, _RETURN_FORMATS[language]),
            gold,
            _LANGUAGES[language],
            category,
            CHECKER_MODEL_NAME,
        )
        payload = orjson.dumps(
            {
                "id": f"{category}_0",
                "test_category": category,
                "language": language,
                "function": function,
                "possible_answer": gold,
            }
        ).decode("utf-8")
        result = await _grader().grade(response, payload)
        assert upstream["valid"] is True
        assert result.correct is upstream["valid"]
