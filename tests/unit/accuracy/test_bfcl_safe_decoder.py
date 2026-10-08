# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The non-executing BFCL Python decoder, exercised without ``bfcl-eval``.

``_bfcl_compat``'s Python decode path (strip-and-wrap normalization, AST value
resolution, bounded arithmetic) is stdlib-only, so its safety boundary must be
covered by the default unit run rather than only by the opt-in real-wheel job.
The accuracy conftest swaps ``decode_calls`` for the fake when ``bfcl_eval`` is
absent; every test here restores the real function first. Upstream parity of
the same paths lives in ``test_bfcl_ast_parity.py``.
"""

from __future__ import annotations

import builtins
import os
import time

import orjson
import pytest
from pytest import param

from aiperf.accuracy.graders import _bfcl_compat
from aiperf.accuracy.graders._bfcl_compat import BFCLDecodeError
from aiperf.accuracy.graders._bfcl_compat import decode_calls as _real_decode_calls
from aiperf.accuracy.graders.tool_call_ast import ToolCallASTGrader
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

#: Wall-clock ceiling for refusing an oversized expression. A refusal is a
#: size check on the operands, so anything near this means the operator ran.
_REFUSAL_BUDGET_S = 0.5

_WEATHER_FUNCTION = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "dict",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    }
]
_WEATHER_GOLD = [{"get_weather": {"city": ["SF"]}}]


@pytest.fixture(autouse=True)
def _real_python_decoder(
    _patch_bfcl_compat_names: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_bfcl_compat, "decode_calls", _real_decode_calls)


@pytest.fixture
def tripwire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any execution of a payload's side effect."""
    fired: list[str] = []
    real_print = builtins.print

    def watched_print(*args: object, **kwargs: object) -> None:
        if args and args[0] == "PROBE":
            fired.append("print")
        real_print(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "print", watched_print)
    monkeypatch.setattr(os, "system", lambda cmd: fired.append("os.system") or 0)
    return fired


def _grader() -> ToolCallASTGrader:
    return ToolCallASTGrader(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST},
        )
    )


def _ground_truth(category: str, gold: object) -> str:
    return orjson.dumps(
        {
            "id": f"{category}_0",
            "test_category": category,
            "language": "python",
            "function": _WEATHER_FUNCTION,
            "possible_answer": gold,
        }
    ).decode("utf-8")


#: Shapes whose resolution upstream routes through ``eval``. Each embeds a
#: distinct execution vector; all must be refused without running.
EXECUTING_PAYLOADS = [
    param(
        "[calculate_triangle_area(base=(__import__('builtins').print('PROBE') or 10)+0, height=5)]",
        id="binop_over_boolop_call",
    ),
    param(
        "[calculate_triangle_area(base=(lambda: __import__('builtins').print('PROBE'))()+0, height=5)]",
        id="binop_over_called_lambda",
    ),
    param(
        "[calculate_triangle_area(base=lambda: __import__('builtins').print('PROBE'), height=5)]",
        id="lambda_argument",
    ),
    param(
        "[f(x=(__import__('os').system('id') or 1)+0)]",
        id="binop_over_os_system",
    ),
]

#: Arithmetic whose result (or an intermediate) is too large to compute on the
#: event loop. Every exponent is within ``_MAX_POW_EXPONENT``.
OVERSIZED_PAYLOADS = [
    param("[f(x=(((3**1024)**1024)**64))]", id="nested_pow"),
    param("[f(x=1<<100000000)]", id="huge_lshift"),
    param("[f(x=10**70*10**70*10**70*10**70)]", id="repeated_mult"),
    param("[f(x=2**256)]", id="pow_one_bit_over"),
    param("[f(x=-(2**200)*2**200)]", id="negative_mult"),
    param("[f(x=1e308*10)]", id="float_overflow_to_inf"),
    param("[f(x=2.0**1024)]", id="float_pow_overflow"),
]


class TestDecoderDoesNotExecuteModelOutput:
    """The decoder refuses every shape upstream would hand to ``eval``."""

    @pytest.mark.parametrize("payload", EXECUTING_PAYLOADS)
    def test_decode_calls_executing_shape_raises_decode_error(
        self, payload: str, tripwire: list[str]
    ) -> None:
        with pytest.raises(BFCLDecodeError):
            _bfcl_compat.decode_calls(payload, "python")
        assert not tripwire, f"decoding executed model output: {tripwire}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", EXECUTING_PAYLOADS)
    async def test_grade_executing_shape_is_unparsed_without_executing(
        self, payload: str, tripwire: list[str]
    ) -> None:
        result = await _grader().grade(
            payload, _ground_truth("simple_python", _WEATHER_GOLD)
        )
        assert not tripwire, f"grading executed model output: {tripwire}"
        assert result.correct is False
        assert result.unparsed is True


class TestArithmeticIsBounded:
    """Oversized arithmetic is refused before it is computed."""

    @pytest.mark.parametrize("payload", OVERSIZED_PAYLOADS)
    def test_decode_calls_oversized_arithmetic_raises_quickly(
        self, payload: str
    ) -> None:
        start = time.perf_counter()
        with pytest.raises(BFCLDecodeError):
            _bfcl_compat.decode_calls(payload, "python")
        assert time.perf_counter() - start < _REFUSAL_BUDGET_S

    def test_decode_calls_deeply_chained_arithmetic_raises_decode_error(
        self,
    ) -> None:
        payload = "[f(x=" + "+".join(["1"] * 20_000) + ")]"
        with pytest.raises(BFCLDecodeError):
            _bfcl_compat.decode_calls(payload, "python")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", OVERSIZED_PAYLOADS)
    async def test_grade_oversized_arithmetic_on_abstention_returns_a_record(
        self, payload: str
    ) -> None:
        """The abstention path must produce a serializable result, not raise."""
        result = await _grader().grade(payload, _ground_truth("irrelevance", None))
        assert result.correct is True
        assert orjson.dumps(result.model_dump())

    @pytest.mark.asyncio
    async def test_grade_largest_allowed_integer_on_abstention_serializes(
        self,
    ) -> None:
        result = await _grader().grade(
            "[f(x=2**255)]", _ground_truth("irrelevance", None)
        )
        assert result.correct is False
        assert str(2**255) in result.extracted_answer

    @pytest.mark.parametrize(
        "payload,expected",
        [
            param("[f(x=5+5, y=2*3)]", {"x": 10, "y": 6}, id="add_mult"),
            param("[f(x=2**10)]", {"x": 1024}, id="pow"),
            param("[f(x=1<<8, y=256>>4)]", {"x": 256, "y": 16}, id="shifts"),
            param("[f(x=3.5*2, y=7/2)]", {"x": 7.0, "y": 3.5}, id="floats"),
            param("[f(x=-(2**100))]", {"x": -(2**100)}, id="negative_large"),
            param("[f(x=2**255)]", {"x": 2**255}, id="at_bit_bound"),
        ],
    )  # fmt: skip
    def test_decode_calls_benign_arithmetic_resolves(
        self, payload: str, expected: dict[str, object], tripwire: list[str]
    ) -> None:
        assert _bfcl_compat.decode_calls(payload, "python") == [{"f": expected}]
        assert not tripwire


class TestPromptModeNormalization:
    """Upstream's strip-and-wrap runs before the Python decoder."""

    @pytest.mark.parametrize(
        "response",
        [
            param("[get_weather(city='SF')]", id="bracketed"),
            param("get_weather(city='SF')", id="unbracketed"),
            param("```\n[get_weather(city='SF')]\n```", id="unlabelled_fence"),
            param("`get_weather(city='SF')`", id="inline_backticks"),
            param("  \n[get_weather(city='SF')]\n  ", id="padded"),
        ],
    )  # fmt: skip
    def test_decode_calls_normalizes_to_the_same_call(self, response: str) -> None:
        assert _bfcl_compat.decode_calls(response, "python") == [
            {"get_weather": {"city": "SF"}}
        ]

    def test_decode_calls_labelled_fence_raises_like_upstream(self) -> None:
        """Upstream does not strip a language label; neither may we."""
        with pytest.raises(BFCLDecodeError):
            _bfcl_compat.decode_calls(
                "```python\n[get_weather(city='SF')]\n```", "python"
            )

    def test_decode_calls_bare_fence_decodes_to_no_calls(self) -> None:
        assert _bfcl_compat.decode_calls("```", "python") == []

    @pytest.mark.asyncio
    async def test_grade_fenced_call_on_irrelevance_is_not_an_abstention(
        self,
    ) -> None:
        result = await _grader().grade(
            "```\n[get_weather(city='SF')]\n```", _ground_truth("irrelevance", None)
        )
        assert result.correct is False
        assert result.unparsed is False

    @pytest.mark.asyncio
    async def test_grade_prose_on_irrelevance_is_an_abstention(self) -> None:
        result = await _grader().grade(
            "I cannot answer that with these tools.",
            _ground_truth("irrelevance", None),
        )
        assert result.correct is True
