# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks import mmlu as mmlu_mod
from aiperf.accuracy.benchmarks.mmlu import MMLUBenchmark
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _run():
    return make_benchmark_run(
        model_names=["m"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    )


def _row():
    return {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": "B"}


def _split(rows):
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


async def _load(enable_cot: bool):
    dd = {"test": _split([_row()]), "dev": _split([])}
    with patch("aiperf.accuracy.benchmarks.mmlu.load_dataset", return_value=dd):
        return await MMLUBenchmark(run=_run()).load_problems(
            tasks=["abstract_algebra"], n_shots=0, enable_cot=enable_cot
        )


@pytest.mark.asyncio
async def test_non_cot_generation_size_unchanged():
    problems = await _load(enable_cot=False)
    assert problems[0].metadata["generation_size"] == 5


@pytest.mark.asyncio
async def test_cot_generation_size_is_large():
    problems = await _load(enable_cot=True)
    assert problems[0].metadata["generation_size"] == mmlu_mod.COT_GENERATION_SIZE
    assert mmlu_mod.COT_GENERATION_SIZE == 4000


@pytest.mark.asyncio
async def test_cot_prompt_requests_answer_format():
    problems = await _load(enable_cot=True)
    assert "The answer is (X)" in problems[0].prompt


@pytest.mark.asyncio
async def test_non_cot_prompt_has_no_answer_format_instruction():
    problems = await _load(enable_cot=False)
    assert "The answer is (X)" not in problems[0].prompt


@pytest.mark.asyncio
async def test_cot_chat_messages_also_request_answer_format():
    # Chat-endpoint parity: the instruction must be added to raw_messages too,
    # not just the flat completions prompt (mmlu.py::_build_chat_messages).
    problems = await _load(enable_cot=True)
    msgs = problems[0].raw_messages
    assert msgs is not None
    joined = "".join(m["content"] for m in msgs)
    assert "The answer is (X)" in joined


@pytest.mark.asyncio
async def test_non_cot_chat_messages_have_no_answer_format():
    problems = await _load(enable_cot=False)
    msgs = problems[0].raw_messages
    assert msgs is not None
    joined = "".join(m["content"] for m in msgs)
    assert "The answer is (X)" not in joined
