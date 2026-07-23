# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM per-request spec-decode adapter.

Covers the engine-neutral ``SpecDecodeAcceptanceRecord`` filled from vLLM's
per-choice ``speculative_decoding_stats`` payload across the shapes the ticket
calls out: present, absent, zero-step, and fully-rejected, in both streaming
and non-streaming layouts, plus the detailed per-step arrays and malformed
degradation.

The sample payloads mirror the wire format from vLLM PR
https://github.com/vllm-project/vllm/pull/48915.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord
from aiperf.spec_decode.vllm_adapter import VLLMSpecDecodeAdapter

# A representative vLLM ``summary`` payload (histogram keys are JSON strings).
# 39 steps accepted 0 drafts, 1 step accepted 1, 3 steps accepted 3:
#   num_spec_steps      = 39 + 1 + 3            = 43
#   num_accepted_draft  = 0*39 + 1*1 + 3*3      = 10
#   mean_acceptance_len = 1 + 10 / 43           = 1.2325...
#   draft_acceptance    = 10 / 129              = 0.0775...
SUMMARY_PAYLOAD: dict[str, Any] = {
    "mean_acceptance_length": 1.2325581395348837,
    "draft_acceptance_rate": 0.07751937984496124,
    "acceptance_histogram": {"0": 39, "1": 1, "3": 3},
    "num_spec_steps": 43,
    "num_accepted_draft_tokens": 10,
    "num_draft_tokens": 129,
    "num_spec_tokens": 3,
}


def _response(
    *,
    spec_decode_stats: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse carrying only the fields under test."""
    return ParsedResponse(
        perf_ns=123,
        usage=usage,
        spec_decode_stats=spec_decode_stats,
    )


def _non_streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    """Single response carrying both the stats and usage (non-streaming)."""
    return [_response(spec_decode_stats=payload, usage={"completion_tokens": 50})]


def _streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    """Streaming layout: content chunk, terminal stats chunk, usage-only chunk.

    Mirrors vLLM: acceptance stats ride the finish-reason chunk's choice while
    the trailing ``include_usage`` chunk carries usage on empty choices.
    """
    return [
        _response(),
        _response(spec_decode_stats=payload),
        _response(usage={"completion_tokens": 50}),
    ]


class TestVLLMSpecDecodeAdapter:
    @pytest.mark.parametrize(
        "make_responses",
        [
            param(_non_streaming, id="non_streaming"),
            param(_streaming, id="streaming"),
        ],
    )  # fmt: skip
    def test_adapt_summary_payload_fills_record(
        self, make_responses: Callable[[dict[str, Any]], list[ParsedResponse]]
    ) -> None:
        responses = make_responses(SUMMARY_PAYLOAD)

        assert VLLMSpecDecodeAdapter.can_adapt(responses) is True
        record = VLLMSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.engine == "vllm"
        assert record.mean_acceptance_length == pytest.approx(1.2325581395348837)
        assert record.draft_acceptance_rate == pytest.approx(0.07751937984496124)
        # Histogram string keys are int-cast into the neutral record.
        assert record.acceptance_histogram == {0: 39, 1: 1, 3: 3}
        assert record.num_spec_steps == 43
        assert record.num_accepted_draft_tokens == 10
        assert record.num_draft_tokens == 129
        assert record.num_spec_tokens == 3
        # completion_tokens is copied from the response usage, not the payload.
        assert record.completion_tokens == 50
        # summary level carries no per-step arrays.
        assert record.per_step_accepted is None
        assert record.per_step_drafted is None

    @pytest.mark.parametrize(
        "responses",
        [
            param([], id="no_responses"),
            param([_response()], id="response_without_stats"),
            param(
                [_response(usage={"completion_tokens": 5})],
                id="usage_only_no_stats",
            ),
            param([_response(spec_decode_stats={})], id="empty_stats_dict"),
            param(
                # A payload present but NOT vLLM-shaped (e.g. another engine's
                # histogram key): the signature check must reject it so this
                # adapter defers rather than greedily claiming it.
                [
                    _response(
                        spec_decode_stats={
                            "spec_correct_drafts_histogram": {"0": 5},
                            "steps": 5,
                        }
                    )
                ],
                id="non_vllm_shaped_payload",
            ),
        ],
    )  # fmt: skip
    def test_adapt_absent_or_unrecognized_payload_yields_no_record(
        self, responses: list[ParsedResponse]
    ) -> None:
        """No field, or a non-vLLM payload: can_adapt False, adapt None."""
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is False
        assert VLLMSpecDecodeAdapter.adapt(responses) is None

    def test_adapt_zero_step_payload_fills_record(self) -> None:
        """No verify steps: empty histogram, mean 1.0, rate 0.0 (server-computed)."""
        payload = {
            "mean_acceptance_length": 1.0,
            "draft_acceptance_rate": 0.0,
            "acceptance_histogram": {},
            "num_spec_steps": 0,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 0,
            "num_spec_tokens": 3,
        }
        record = VLLMSpecDecodeAdapter.adapt(_non_streaming(payload))

        assert record is not None
        assert record.mean_acceptance_length == 1.0
        assert record.draft_acceptance_rate == 0.0
        assert record.acceptance_histogram == {}
        assert record.num_spec_steps == 0
        assert record.num_accepted_draft_tokens == 0
        assert record.num_draft_tokens == 0

    def test_adapt_fully_rejected_payload_fills_record(self) -> None:
        """Every draft rejected: all steps land in the j=0 bucket, mean 1.0."""
        payload = {
            "mean_acceptance_length": 1.0,
            "draft_acceptance_rate": 0.0,
            "acceptance_histogram": {"0": 20},
            "num_spec_steps": 20,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 60,
            "num_spec_tokens": 3,
        }
        record = VLLMSpecDecodeAdapter.adapt(_non_streaming(payload))

        assert record is not None
        assert record.acceptance_histogram == {0: 20}
        assert record.num_spec_steps == 20
        assert record.num_accepted_draft_tokens == 0
        assert record.num_draft_tokens == 60
        assert record.draft_acceptance_rate == 0.0

    def test_adapt_detailed_payload_carries_per_step_arrays(self) -> None:
        # Self-consistent detailed payload: the per-step arrays agree with the
        # histogram and counts -- 4 steps, accepted [0,1,3,0] -> histogram
        # {0:2, 1:1, 3:1} and 4 accepted; drafted [3,3,3,3] -> 12 proposed.
        payload = {
            "mean_acceptance_length": 2.0,
            "draft_acceptance_rate": 4 / 12,
            "acceptance_histogram": {"0": 2, "1": 1, "3": 1},
            "num_spec_steps": 4,
            "num_accepted_draft_tokens": 4,
            "num_draft_tokens": 12,
            "num_spec_tokens": 3,
            "per_step_accepted": [0, 1, 3, 0],
            "per_step_drafted": [3, 3, 3, 3],
        }
        record = VLLMSpecDecodeAdapter.adapt(_non_streaming(payload))

        assert record is not None
        assert record.per_step_accepted == [0, 1, 3, 0]
        assert record.per_step_drafted == [3, 3, 3, 3]

    def test_adapt_missing_num_spec_tokens_is_optional(self) -> None:
        payload = {k: v for k, v in SUMMARY_PAYLOAD.items() if k != "num_spec_tokens"}
        record = VLLMSpecDecodeAdapter.adapt(_non_streaming(payload))

        assert record is not None
        assert record.num_spec_tokens is None

    def test_adapt_no_usage_leaves_completion_tokens_none(self) -> None:
        responses = [_response(spec_decode_stats=SUMMARY_PAYLOAD)]
        record = VLLMSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.completion_tokens is None

    @pytest.mark.parametrize(
        "bad_payload",
        [
            # Signature keys present (so can_adapt matches) but the rest of the
            # required body is missing.
            param(
                {"acceptance_histogram": {"0": 1}, "num_spec_steps": 1},
                id="signature_only_missing_rest",
            ),
            param(
                {**SUMMARY_PAYLOAD, "acceptance_histogram": {"x": 1}},
                id="non_integer_histogram_key",
            ),
            param(
                {**SUMMARY_PAYLOAD, "acceptance_histogram": [1, 2, 3]},
                id="histogram_wrong_type",
            ),
        ],
    )  # fmt: skip
    def test_adapt_malformed_payload_degrades_to_none(
        self, bad_payload: dict[str, Any]
    ) -> None:
        """A signature-matching but broken payload must not raise -- no record."""
        responses = [_response(spec_decode_stats=bad_payload)]
        # can_adapt matches the vLLM signature; adapt must swallow the bad shape.
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is True
        assert VLLMSpecDecodeAdapter.adapt(responses) is None

    def test_adapt_last_payload_wins_across_chunks(self) -> None:
        """If more than one chunk carried stats, the last is authoritative."""
        first = {**SUMMARY_PAYLOAD, "num_spec_steps": 1}
        responses = [
            _response(spec_decode_stats=first),
            _response(spec_decode_stats=SUMMARY_PAYLOAD),
        ]
        record = VLLMSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.num_spec_steps == 43

    def test_adapt_negative_count_payload_degrades_to_none(self) -> None:
        """A signature-matching payload with a negative count is rejected by the
        record's ge=0 constraints, and the adapter degrades to None."""
        bad = {**SUMMARY_PAYLOAD, "acceptance_histogram": {"0": -1}}
        responses = [_response(spec_decode_stats=bad)]
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is True
        assert VLLMSpecDecodeAdapter.adapt(responses) is None

    @pytest.mark.parametrize(
        "bad_payload",
        [
            # All keys present, valid types, non-negative -- but the aggregates
            # contradict each other, so the record's invariants reject them.
            param({**SUMMARY_PAYLOAD, "num_spec_steps": 99}, id="histogram_sum_mismatch"),
            param({**SUMMARY_PAYLOAD, "num_accepted_draft_tokens": 11}, id="weighted_sum_mismatch"),
            param({**SUMMARY_PAYLOAD, "num_draft_tokens": 5}, id="accepted_exceeds_drafted"),
        ],
    )  # fmt: skip
    def test_adapt_inconsistent_aggregate_payload_degrades_to_none(
        self, bad_payload: dict[str, Any]
    ) -> None:
        """Contradictory (but non-negative) aggregates fail the record's integer
        invariants; the adapter degrades to None rather than emit a bad record."""
        responses = [_response(spec_decode_stats=bad_payload)]
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is True
        assert VLLMSpecDecodeAdapter.adapt(responses) is None


class TestRecordConstraints:
    """The record rejects negative counts (ge=0), including inside containers."""

    def _valid_kwargs(self) -> dict[str, Any]:
        return {
            "engine": "vllm",
            "mean_acceptance_length": 1.0,
            "draft_acceptance_rate": 0.0,
            "acceptance_histogram": {0: 1},
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 1,
            "num_spec_steps": 1,
        }

    @pytest.mark.parametrize(
        "override",
        [
            param({"num_draft_tokens": -1}, id="negative_scalar_count"),
            param({"acceptance_histogram": {0: -1}}, id="negative_histogram_value"),
            param({"acceptance_histogram": {-1: 1}}, id="negative_histogram_key"),
            param({"per_step_accepted": [0, -1]}, id="negative_per_step_value"),
        ],
    )  # fmt: skip
    def test_record_rejects_negative_counts(self, override: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            SpecDecodeAcceptanceRecord(**{**self._valid_kwargs(), **override})

    @pytest.mark.parametrize(
        "kwargs",
        [
            param(
                {"num_spec_steps": 2},  # histogram {0:1} sums to 1, not 2
                id="histogram_sum_mismatch",
            ),
            param(
                {"acceptance_histogram": {1: 1}},  # weighted 1 != accepted 0
                id="weighted_sum_mismatch",
            ),
            param(
                # weighted 2 == accepted 2, sum 1 == steps 1, but accepted > draft
                {
                    "acceptance_histogram": {2: 1},
                    "num_accepted_draft_tokens": 2,
                    "num_draft_tokens": 1,
                },
                id="accepted_exceeds_drafted",
            ),
        ],
    )  # fmt: skip
    def test_record_rejects_inconsistent_aggregates(
        self, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            SpecDecodeAcceptanceRecord(**{**self._valid_kwargs(), **kwargs})
