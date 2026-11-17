# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp streams cumulative ``timings`` on every chunk; the parser keeps the last.

Other engines' per-chunk payloads must still be suppressed, and an ``n > 1``
request must stay suppressed even though its timings collapse the same way.
"""

from typing import Any

import pytest

from aiperf.common.models import ParsedResponse
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.spec_decode.test_llamacpp_adapter import TIMINGS_PAYLOAD
from tests.unit.spec_decode.test_vllm_adapter import SUMMARY_PAYLOAD as VLLM_PAYLOAD


def _cumulative_chunks() -> list[ParsedResponse]:
    """Three chunks whose timings only grow, as llama.cpp reports them."""
    totals = [(2, 1), (4, 3), (8, 6)]
    chunks = [
        ParsedResponse(
            perf_ns=i,
            spec_decode_stats={
                **TIMINGS_PAYLOAD,
                "draft_n": drafted,
                "draft_n_accepted": accepted,
            },
        )
        for i, (drafted, accepted) in enumerate(totals, start=1)
    ]
    chunks.append(ParsedResponse(perf_ns=4, usage={"completion_tokens": 20}))
    return chunks


def test_cumulative_timings_collapse_to_last_chunk() -> None:
    record = InferenceResultParser._extract_spec_decode_acceptance(_cumulative_chunks())
    assert record is not None
    assert record.engine == "llamacpp"
    assert record.num_draft_tokens == 8
    assert record.num_accepted_draft_tokens == 6
    assert record.completion_tokens == 20


@pytest.mark.parametrize("num_choices", [2, 4])
def test_multi_sequence_cumulative_timings_stay_suppressed(num_choices: int) -> None:
    record = InferenceResultParser._extract_spec_decode_acceptance(
        _cumulative_chunks(), num_choices=num_choices
    )
    assert record is None


def test_llamacpp_payload_beside_other_engine_payload_stays_suppressed() -> None:
    responses: list[Any] = [
        ParsedResponse(perf_ns=1, spec_decode_stats=TIMINGS_PAYLOAD),
        ParsedResponse(perf_ns=2, spec_decode_stats=VLLM_PAYLOAD),
    ]
    assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None
