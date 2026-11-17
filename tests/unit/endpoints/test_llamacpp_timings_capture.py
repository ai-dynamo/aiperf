# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that ``extract_spec_decode_stats`` captures llama.cpp's top-level ``timings``.

llama.cpp places its draft counters at the response root, and only the other
engines' placements may take precedence over them. A multi-choice response is
suppressed before ``timings`` is consulted, because llama.cpp keeps only the
first result's timings for that response.
"""

from typing import Any

import pytest
from pytest import param

from aiperf.endpoints.base_endpoint import BaseEndpoint
from tests.unit.spec_decode.test_llamacpp_adapter import TIMINGS_PAYLOAD
from tests.unit.spec_decode.test_sglang_adapter import SUMMARY_PAYLOAD
from tests.unit.spec_decode.test_trtllm_adapter import SUMMARY_PAYLOAD as TRT_PAYLOAD
from tests.unit.spec_decode.test_vllm_adapter import SUMMARY_PAYLOAD as VLLM_PAYLOAD


def test_top_level_timings_are_captured() -> None:
    body = {
        "object": "text_completion",
        "choices": [{"text": "hi"}],
        "timings": TIMINGS_PAYLOAD,
    }
    assert BaseEndpoint.extract_spec_decode_stats(body) == TIMINGS_PAYLOAD


@pytest.mark.parametrize(
    ("placement", "payload"),
    [
        param(
            {"metrics": {"speculative_decoding": VLLM_PAYLOAD}},
            VLLM_PAYLOAD,
            id="vllm_root",
        ),
        param(
            {"choices": [{"index": 0, "speculative_decoding": TRT_PAYLOAD}]},
            TRT_PAYLOAD,
            id="trtllm_choice",
        ),
        param(
            {"sglext": {"spec_tokens_details": SUMMARY_PAYLOAD}},
            SUMMARY_PAYLOAD,
            id="sglang_extension",
        ),
    ],
)  # fmt: skip
def test_other_engine_placements_win_over_timings(
    placement: dict[str, Any], payload: dict[str, Any]
) -> None:
    body = {**placement, "timings": TIMINGS_PAYLOAD}
    assert BaseEndpoint.extract_spec_decode_stats(body) == payload


def test_two_choice_response_with_timings_is_suppressed() -> None:
    body = {
        "object": "text_completion",
        "choices": [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}],
        "timings": TIMINGS_PAYLOAD,
    }
    assert BaseEndpoint.extract_spec_decode_stats(body) is None


@pytest.mark.parametrize(
    "timings",
    [
        param({"predicted_n": 20}, id="no_draft_counters"),
        param({"draft_n": 4}, id="missing_accepted"),
        param(None, id="null"),
        param("bad", id="string"),
    ],
)  # fmt: skip
def test_malformed_timings_are_ignored(timings: Any) -> None:
    body = {
        "object": "text_completion",
        "choices": [{"text": "hi"}],
        "timings": timings,
    }
    assert BaseEndpoint.extract_spec_decode_stats(body) is None
