# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp aggregate draft-counter normalization and malformed-payload handling."""

from typing import Any

import pytest
from pytest import param

from aiperf.common.models import ParsedResponse
from aiperf.spec_decode.llamacpp_adapter import LlamaCppSpecDecodeAdapter

# A request that drafted 16 tokens and had 12 accepted. llama.cpp reports only
# these counters plus timing fields, so the payload is the full timings object.
TIMINGS_PAYLOAD: dict[str, Any] = {
    "cache_n": 0,
    "prompt_n": 10,
    "prompt_ms": 5.0,
    "predicted_n": 20,
    "predicted_ms": 40.0,
    "draft_n": 16,
    "draft_n_accepted": 12,
}


def _responses(payload: dict[str, Any] | None) -> list[ParsedResponse]:
    return [
        ParsedResponse(perf_ns=1, spec_decode_stats=payload),
        ParsedResponse(perf_ns=2, usage={"completion_tokens": 20}),
    ]


def test_typical_timings_fill_aggregate_record() -> None:
    record = LlamaCppSpecDecodeAdapter.adapt(_responses(TIMINGS_PAYLOAD))
    assert record is not None
    assert record.engine == "llamacpp"
    assert record.draft_acceptance_rate == 0.75
    assert record.num_accepted_draft_tokens == 12
    assert record.num_draft_tokens == 16
    assert record.completion_tokens == 20
    assert record.num_spec_tokens is None


def test_typical_timings_leave_step_fields_unset() -> None:
    record = LlamaCppSpecDecodeAdapter.adapt(_responses(TIMINGS_PAYLOAD))
    assert record is not None
    assert record.mean_acceptance_length is None
    assert record.acceptance_histogram is None
    assert record.num_spec_steps is None
    assert record.per_step_accepted is None
    assert record.per_step_drafted is None


def test_completion_tokens_none_without_usage() -> None:
    record = LlamaCppSpecDecodeAdapter.adapt(
        [ParsedResponse(perf_ns=1, spec_decode_stats=TIMINGS_PAYLOAD)]
    )
    assert record is not None
    assert record.completion_tokens is None


def test_zero_drafted_returns_no_record() -> None:
    payload = {**TIMINGS_PAYLOAD, "draft_n": 0, "draft_n_accepted": 0}
    assert LlamaCppSpecDecodeAdapter.adapt(_responses(payload)) is None


def test_missing_draft_counters_are_not_claimed() -> None:
    payload = {"predicted_n": 20, "predicted_ms": 40.0}
    assert not LlamaCppSpecDecodeAdapter.can_adapt(_responses(payload))
    assert LlamaCppSpecDecodeAdapter.adapt(_responses(payload)) is None


@pytest.mark.parametrize(
    "override",
    [
        param({"draft_n_accepted": 17}, id="accepted_exceeds_drafted"),
        param({"draft_n": -1}, id="negative_drafted"),
        param({"draft_n": True, "draft_n_accepted": 1}, id="boolean_drafted"),
        param({"draft_n_accepted": "12"}, id="string_accepted"),
        param({"draft_n": 16.0}, id="float_drafted"),
    ],
)  # fmt: skip
def test_malformed_counters_degrade_to_none(override: dict[str, Any]) -> None:
    payload = {**TIMINGS_PAYLOAD, **override}
    assert LlamaCppSpecDecodeAdapter.adapt(_responses(payload)) is None
