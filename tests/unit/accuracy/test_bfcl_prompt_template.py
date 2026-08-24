# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-equality pin: our BFCL prompt vs the real ``bfcl-eval``'s.

This is the single biggest parity risk in the ``bfcl_ast`` integration. BFCL
v4's own format-sensitivity work shows models are highly sensitive to the exact
prompt wording - some tool-trained models drop to near-zero when the requested
return format or tool-call tagging changes - so a prompt that merely *looks*
right would silently understate every score.

``BFCLASTBenchmark`` therefore never builds the template itself; it delegates
to upstream's ``system_prompt_pre_processing_chat_model`` through
``_bfcl_compat``. These tests pin that delegation to byte equality, and pin the
landmark sentences so a silent upstream rewrite is loud rather than invisible.

If one of these fails after a ``bfcl-eval`` bump, the *scores changed*: move
``AIPERF_ACCURACY_BFCL_VERSION_PIN`` and rebaseline deliberately. Do not adjust
the expectation to match.

Reference: bfcl_eval.model_handler.utils.system_prompt_pre_processing_chat_model
"""

from __future__ import annotations

import pytest

# This file is a parity oracle against the real dependency; skip cleanly when
# bfcl-eval isn't installed rather than faking it.
pytest.importorskip("bfcl_eval")

from bfcl_eval.model_handler.utils import (  # noqa: E402
    system_prompt_pre_processing_chat_model,
)

from aiperf.accuracy.benchmarks.bfcl_ast import BFCLASTBenchmark  # noqa: E402
from aiperf.accuracy.graders import _bfcl_compat  # noqa: E402
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType  # noqa: E402
from tests.unit.conftest import make_benchmark_run  # noqa: E402

pytestmark = pytest.mark.requires_bfcl

_ENTRY_ID = "simple_python_0"
_QUESTION = [{"role": "user", "content": "What is the weather in SF?"}]
_FUNCTION = [
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


def _upstream_messages() -> list[dict]:
    """What upstream produces for this entry, from a fresh input list.

    Upstream mutates the list it is handed, so each call gets its own copy.
    """
    return system_prompt_pre_processing_chat_model(
        [dict(m) for m in _QUESTION], _FUNCTION, _ENTRY_ID
    )


class TestPromptByteEquality:
    """Our messages must be byte-identical to upstream's."""

    def test_compat_shim_matches_upstream_builder(self) -> None:
        assert (
            _bfcl_compat.build_chat_messages(_QUESTION, _FUNCTION, _ENTRY_ID)
            == _upstream_messages()
        )

    def test_loader_system_prompt_is_byte_identical(self) -> None:
        benchmark = BFCLASTBenchmark(
            run=make_benchmark_run(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                streaming=False,
                accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST},
            )
        )
        problem = benchmark._build_problem(
            "simple_python",
            {"id": _ENTRY_ID, "question": [_QUESTION], "function": _FUNCTION},
            [{"get_weather": {"city": ["SF"]}}],
        )
        expected = _upstream_messages()
        assert [m["content"] for m in problem.raw_messages] == [
            m["content"] for m in expected
        ]
        assert [m["role"] for m in problem.raw_messages] == [
            m["role"] for m in expected
        ]

    def test_build_chat_messages_does_not_mutate_caller_input(self) -> None:
        """Upstream inserts into the list it is given; we must pass a copy."""
        question = [dict(m) for m in _QUESTION]
        _bfcl_compat.build_chat_messages(question, _FUNCTION, _ENTRY_ID)
        assert question == _QUESTION


class TestPromptLandmarks:
    """Landmark sentences a silent upstream rewrite would move."""

    def test_system_prompt_opens_with_the_expert_persona(self) -> None:
        system_prompt = _upstream_messages()[0]["content"]
        assert system_prompt.startswith("You are an expert in composing functions.")

    def test_system_prompt_requests_the_python_call_list_format(self) -> None:
        system_prompt = _upstream_messages()[0]["content"]
        assert (
            "[func_name1(params_name1=params_value1, params_name2=params_value2...), "
            "func_name2(params)]" in system_prompt
        )

    def test_system_prompt_forbids_other_text(self) -> None:
        """The instruction that makes the response decodable at all."""
        system_prompt = _upstream_messages()[0]["content"]
        assert "SHOULD NOT include any other text in the response" in system_prompt

    def test_system_prompt_embeds_the_entry_tool_schemas(self) -> None:
        system_prompt = _upstream_messages()[0]["content"]
        assert "get_weather" in system_prompt
        assert "Get the weather for a city." in system_prompt
