# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TensorRT-LLM per-request spec-decode adapter.

Covers the engine-neutral ``SpecDecodeAcceptanceRecord`` filled from TRT-LLM's
per-choice ``speculative_decoding`` payload: present, absent, fully-rejected,
fully-accepted, streaming and non-streaming, and malformed degradation.

Two differences from vLLM drive most of what is tested here. TRT-LLM does not
send a mean acceptance length -- it reports that per choice as
``avg_decoded_tokens_per_iter`` -- so the adapter derives it from the counts.
And it sends ``num_spec_tokens: null`` under ``draft_len_schedule``, where the
per-step draft bound varies by batch size; that must relax the histogram-length
cross-check without relaxing any of the record's identity validators.
"""

from typing import Any

import pytest
from pytest import param

from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord
from aiperf.spec_decode.trtllm_adapter import TRTLLMSpecDecodeAdapter

# A representative TRT-LLM payload. The dense acceptance_histogram [8, 0, 6, 6]
# means 8 steps accepted 0 drafts, 6 accepted 2, 6 accepted all 3:
#   num_spec_steps       = 8 + 0 + 6 + 6           = 20
#   accepted             = 0*8 + 1*0 + 2*6 + 3*6   = 30
#   mean acceptance len  = 1 + 30 / 20             = 2.5   (derived, not sent)
SUMMARY_PAYLOAD: dict[str, Any] = {
    "acceptance_rate": 0.5,
    "total_accepted_draft_tokens": 30,
    "total_draft_tokens": 60,
    "num_spec_steps": 20,
    "acceptance_histogram": [8, 0, 6, 6],
    "num_spec_tokens": 3,
}

# draft_len_schedule: no fixed per-step bound, so num_spec_tokens is null and
# the histogram is sized to the depth actually reached.
ADAPTIVE_PAYLOAD: dict[str, Any] = {
    "acceptance_rate": 0.5,
    "total_accepted_draft_tokens": 30,
    "total_draft_tokens": 60,
    "num_spec_steps": 20,
    "acceptance_histogram": [8, 0, 6, 6],
    "num_spec_tokens": None,
}


def _response(
    *,
    spec_decode_stats: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> ParsedResponse:
    return ParsedResponse(perf_ns=123, usage=usage, spec_decode_stats=spec_decode_stats)


def _non_streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    """Single response carrying both the stats and usage."""
    return [_response(spec_decode_stats=payload, usage={"completion_tokens": 50})]


def _streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    """Streaming layout: content chunks, then the terminal chunk's choice.

    Unlike vLLM -- whose payload rides the trailing empty-choices usage chunk --
    TRT-LLM attaches it to the choice bearing ``finish_reason``, so the payload
    and the usage arrive on different responses.
    """
    return [
        _response(),
        _response(spec_decode_stats=payload),
        _response(usage={"completion_tokens": 50}),
    ]


class TestCanAdapt:
    def test_claims_its_own_payload(self) -> None:
        assert TRTLLMSpecDecodeAdapter.can_adapt(_non_streaming(SUMMARY_PAYLOAD))

    def test_defers_when_no_payload(self) -> None:
        assert not TRTLLMSpecDecodeAdapter.can_adapt([_response()])

    def test_defers_on_empty_payload(self) -> None:
        assert not TRTLLMSpecDecodeAdapter.can_adapt([_response(spec_decode_stats={})])


class TestAdapt:
    @pytest.mark.parametrize(
        "build",
        [
            param(_non_streaming, id="non_streaming"),
            param(_streaming, id="streaming"),
        ],
    )  # fmt: skip
    def test_fills_the_record(self, build) -> None:
        record = TRTLLMSpecDecodeAdapter.adapt(build(SUMMARY_PAYLOAD))

        assert isinstance(record, SpecDecodeAcceptanceRecord)
        assert record.engine == "tensorrt_llm"
        assert record.num_spec_steps == 20
        assert record.num_accepted_draft_tokens == 30
        assert record.num_draft_tokens == 60
        assert record.draft_acceptance_rate == 0.5
        assert record.num_spec_tokens == 3
        assert record.completion_tokens == 50
        # Zero-count buckets are dropped by the dense -> sparse inflation.
        assert record.acceptance_histogram == {0: 8, 2: 6, 3: 6}

    def test_mean_acceptance_length_is_derived_from_counts(self) -> None:
        """TRT-LLM sends no mean acceptance length; it comes from the counts.

        Deriving it rather than reading a transmitted float means the reported
        length can never contradict the histogram it is reported beside.
        """
        record = TRTLLMSpecDecodeAdapter.adapt(_non_streaming(SUMMARY_PAYLOAD))
        assert record.mean_acceptance_length == 2.5

    def test_per_step_arrays_are_never_populated(self) -> None:
        """TRT-LLM keeps per-position vectors, not per-step sequences."""
        record = TRTLLMSpecDecodeAdapter.adapt(_non_streaming(SUMMARY_PAYLOAD))
        assert record.per_step_accepted is None
        assert record.per_step_drafted is None

    def test_missing_usage_yields_no_completion_tokens(self) -> None:
        record = TRTLLMSpecDecodeAdapter.adapt(
            [_response(spec_decode_stats=SUMMARY_PAYLOAD)]
        )
        assert record is not None
        assert record.completion_tokens is None

    def test_all_rejected(self) -> None:
        payload = {
            "acceptance_rate": 0.0,
            "total_accepted_draft_tokens": 0,
            "total_draft_tokens": 30,
            "num_spec_steps": 10,
            "acceptance_histogram": [10, 0, 0, 0],
            "num_spec_tokens": 3,
        }
        record = TRTLLMSpecDecodeAdapter.adapt(_non_streaming(payload))
        assert record.acceptance_histogram == {0: 10}
        assert record.mean_acceptance_length == 1.0

    def test_all_accepted(self) -> None:
        payload = {
            "acceptance_rate": 1.0,
            "total_accepted_draft_tokens": 15,
            "total_draft_tokens": 15,
            "num_spec_steps": 5,
            "acceptance_histogram": [0, 0, 0, 5],
            "num_spec_tokens": 3,
        }
        record = TRTLLMSpecDecodeAdapter.adapt(_non_streaming(payload))
        assert record.acceptance_histogram == {3: 5}
        assert record.mean_acceptance_length == 4.0


class TestAdaptiveDrafting:
    """``draft_len_schedule`` runs report no fixed per-step bound."""

    def test_null_num_spec_tokens_is_accepted(self) -> None:
        record = TRTLLMSpecDecodeAdapter.adapt(_non_streaming(ADAPTIVE_PAYLOAD))
        assert record is not None
        assert record.num_spec_tokens is None
        assert record.acceptance_histogram == {0: 8, 2: 6, 3: 6}

    def test_null_num_spec_tokens_relaxes_only_the_length_check(self) -> None:
        """A null bound must not disable the record's identity validators.

        Without a bound there is no expected histogram length, so that one
        cross-check cannot run -- but a payload whose counts contradict each
        other is still corrupt and must be dropped.
        """
        broken = {**ADAPTIVE_PAYLOAD, "num_spec_steps": 19}  # histogram sums to 20
        assert TRTLLMSpecDecodeAdapter.adapt(_non_streaming(broken)) is None


class TestMalformedDegradesToNone:
    """A signature-matching but broken payload must not raise -- no record."""

    @pytest.mark.parametrize(
        "bad_payload",
        [
            param(
                {"total_accepted_draft_tokens": 1, "total_draft_tokens": 2},
                id="signature_only_missing_rest",
            ),
            param(
                {**SUMMARY_PAYLOAD, "acceptance_histogram": [8, 0, "x", 6]},
                id="non_integer_histogram_element",
            ),
            param(
                {**SUMMARY_PAYLOAD, "acceptance_histogram": {"0": 8}},
                id="histogram_not_a_list",
            ),
            param(
                {**SUMMARY_PAYLOAD, "acceptance_histogram": [8, 0, 6, 6, 0]},
                id="histogram_longer_than_num_spec_tokens",
            ),
            param(
                {**SUMMARY_PAYLOAD, "num_spec_steps": 19},
                id="histogram_does_not_sum_to_steps",
            ),
            param(
                {**SUMMARY_PAYLOAD, "total_accepted_draft_tokens": 29},
                id="weighted_sum_disagrees_with_accepted",
            ),
            param(
                {**SUMMARY_PAYLOAD, "total_draft_tokens": 29},
                id="accepted_exceeds_drafted",
            ),
            param(
                {
                    **SUMMARY_PAYLOAD,
                    "num_spec_steps": 0,
                    "acceptance_histogram": [0, 0, 0, 0],
                    "total_accepted_draft_tokens": 0,
                },
                id="zero_steps_would_divide_by_zero",
            ),
        ],
    )  # fmt: skip
    def test_degrades_to_none(self, bad_payload: dict[str, Any]) -> None:
        responses = [_response(spec_decode_stats=bad_payload)]
        assert TRTLLMSpecDecodeAdapter.adapt(responses) is None
