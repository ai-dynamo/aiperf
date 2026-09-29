# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the llama.cpp per-request spec-decode adapter.

Mirrors the structure of ``test_vllm_adapter.py``. Covers the
engine-neutral ``SpecDecodeAcceptanceRecord`` filled from llama.cpp's
top-level ``timings`` field (``draft_n``, ``draft_n_accepted``,
``predicted_n``) across representative shapes: normal, fully rejected,
all accepted, and various malformed payloads.

Wire-format reference: llama.cpp ``timings`` object as returned by
``llama-server`` when running with a draft model.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pytest import param

from aiperf.common.models import ParsedResponse
from aiperf.spec_decode.llamacpp_adapter import LlamaCppSpecDecodeAdapter
from aiperf.spec_decode.vllm_adapter import VLLMSpecDecodeAdapter

# Representative llama.cpp timings payload from a real server response.
# predicted_n=20, draft_n=15, draft_n_accepted=14 =>
#   num_spec_steps      = 20 - 14          = 6
#   draft_acceptance    = 14 / 15          = 0.933...
#   mean_accept_length  = 1 + 14 / 6       = 3.333...
#   histogram (base=2, remainder=2)        = {3: 2, 2: 4}
TIMINGS_PAYLOAD: dict[str, Any] = {
    "predicted_n": 20,
    "draft_n": 15,
    "draft_n_accepted": 14,
    # extra timing fields that llama.cpp always includes; adapter must ignore them
    "predicted_ms": 76.214,
    "predicted_per_second": 262.4,
    "prompt_n": 11,
    "prompt_ms": 224.7,
}

# Fully-rejected case: no drafts accepted.
REJECTED_PAYLOAD: dict[str, Any] = {
    "predicted_n": 10,
    "draft_n": 30,
    "draft_n_accepted": 0,
}

# All-accepted case: every draft token accepted.
# predicted_n=22, draft_n=20, draft_n_accepted=20 =>
#   num_spec_steps = 22 - 20 = 2
#   histogram (base=10, remainder=0) = {10: 2}
ALL_ACCEPTED_PAYLOAD: dict[str, Any] = {
    "predicted_n": 22,
    "draft_n": 20,
    "draft_n_accepted": 20,
}

# Signature payload for vLLM -- must NOT be claimed by the llama.cpp adapter.
VLLM_PAYLOAD: dict[str, Any] = {
    "acceptance_histogram": {"0": 39, "1": 1, "3": 3},
    "num_spec_steps": 43,
    "mean_acceptance_length": 1.23,
    "draft_acceptance_rate": 0.08,
    "num_accepted_draft_tokens": 10,
    "num_draft_tokens": 129,
    "num_spec_tokens": 3,
}


def _response(
    *,
    spec_decode_stats: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> ParsedResponse:
    return ParsedResponse(perf_ns=123, usage=usage, spec_decode_stats=spec_decode_stats)


def _non_streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    return [_response(spec_decode_stats=payload, usage={"completion_tokens": 20})]


def _streaming(payload: dict[str, Any]) -> list[ParsedResponse]:
    """Streaming layout: content chunks followed by finish-reason chunk with timings."""
    return [
        _response(),
        _response(spec_decode_stats=payload),
        _response(usage={"completion_tokens": 20}),
    ]


class TestLlamaCppSpecDecodeAdapter:
    @pytest.mark.parametrize(
        "make_responses",
        [
            param(_non_streaming, id="non_streaming"),
            param(_streaming, id="streaming"),
        ],
    )  # fmt: skip
    def test_adapt_typical_payload_fills_record(
        self, make_responses: Callable[[dict[str, Any]], list[ParsedResponse]]
    ) -> None:
        responses = make_responses(TIMINGS_PAYLOAD)

        assert LlamaCppSpecDecodeAdapter.can_adapt(responses) is True
        record = LlamaCppSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.engine == "llamacpp"
        assert record.num_draft_tokens == 15
        assert record.num_accepted_draft_tokens == 14
        assert record.num_spec_steps == 6  # 20 - 14
        assert record.draft_acceptance_rate == pytest.approx(14 / 15)
        assert record.mean_acceptance_length == pytest.approx(1.0 + 14 / 6)
        assert record.completion_tokens == 20
        assert record.num_spec_tokens is None
        # per-step arrays are not available from the timings payload
        assert record.per_step_accepted is None
        assert record.per_step_drafted is None

    def test_adapt_histogram_invariants_hold(self) -> None:
        """Reconstructed histogram satisfies both SpecDecodeAcceptanceRecord invariants."""
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(TIMINGS_PAYLOAD))

        assert record is not None
        h = record.acceptance_histogram
        assert sum(h.values()) == record.num_spec_steps
        assert (
            sum(j * count for j, count in h.items()) == record.num_accepted_draft_tokens
        )

    def test_adapt_histogram_two_bucket_distribution(self) -> None:
        """integer-division bucketing: 14 accepted / 6 steps -> base=2, rem=2 -> {2:4, 3:2}."""
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(TIMINGS_PAYLOAD))

        assert record is not None
        assert record.acceptance_histogram == {2: 4, 3: 2}

    def test_adapt_fully_rejected_fills_record(self) -> None:
        """All drafts rejected: rate 0.0, all steps in j=0 bucket."""
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(REJECTED_PAYLOAD))

        assert record is not None
        assert record.num_accepted_draft_tokens == 0
        assert record.draft_acceptance_rate == 0.0
        assert record.mean_acceptance_length == pytest.approx(1.0)
        assert record.num_spec_steps == 10  # predicted_n(10) - accepted(0)
        assert record.acceptance_histogram == {0: 10}

    def test_adapt_all_accepted_fills_record(self) -> None:
        """All drafts accepted: rate 1.0, histogram is a single even bucket."""
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(ALL_ACCEPTED_PAYLOAD))

        assert record is not None
        assert record.num_accepted_draft_tokens == 20
        assert record.draft_acceptance_rate == pytest.approx(1.0)
        assert record.num_spec_steps == 2  # 22 - 20
        assert record.mean_acceptance_length == pytest.approx(1.0 + 20 / 2)
        assert record.acceptance_histogram == {10: 2}

    def test_adapt_no_usage_leaves_completion_tokens_none(self) -> None:
        responses = [_response(spec_decode_stats=TIMINGS_PAYLOAD)]
        record = LlamaCppSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.completion_tokens is None

    def test_adapt_last_payload_wins_across_chunks(self) -> None:
        """If more than one chunk carries timings, the last non-None is authoritative."""
        early = {
            **TIMINGS_PAYLOAD,
            "draft_n": 5,
            "draft_n_accepted": 3,
            "predicted_n": 8,
        }
        responses = [
            _response(spec_decode_stats=early),
            _response(spec_decode_stats=TIMINGS_PAYLOAD),
        ]
        record = LlamaCppSpecDecodeAdapter.adapt(responses)

        assert record is not None
        assert record.num_draft_tokens == 15  # from TIMINGS_PAYLOAD, not early

    @pytest.mark.parametrize(
        "responses",
        [
            param([], id="no_responses"),
            param([_response()], id="response_without_stats"),
            param([_response(spec_decode_stats={})], id="empty_stats_dict"),
            param(
                [_response(spec_decode_stats=VLLM_PAYLOAD)],
                id="vllm_shaped_payload",
            ),
            param(
                [_response(spec_decode_stats={"draft_n": 5})],
                id="only_draft_n_missing_accepted",
            ),
        ],
    )  # fmt: skip
    def test_can_adapt_rejects_non_llamacpp_shapes(
        self, responses: list[ParsedResponse]
    ) -> None:
        assert LlamaCppSpecDecodeAdapter.can_adapt(responses) is False
        assert LlamaCppSpecDecodeAdapter.adapt(responses) is None

    def test_vllm_adapter_rejects_llamacpp_payload(self) -> None:
        """Adapters must not greedily claim each other's payloads."""
        responses = [_response(spec_decode_stats=TIMINGS_PAYLOAD)]
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is False

    @pytest.mark.parametrize(
        "bad_payload",
        [
            # accepted > draft: impossible count
            param(
                {"predicted_n": 20, "draft_n": 10, "draft_n_accepted": 15},
                id="accepted_exceeds_drafted",
            ),
            # num_spec_steps derived as 0 (predicted_n == draft_n_accepted)
            param(
                {"predicted_n": 14, "draft_n": 15, "draft_n_accepted": 14},
                id="zero_spec_steps",
            ),
            # negative num_spec_steps (predicted_n < draft_n_accepted)
            param(
                {"predicted_n": 5, "draft_n": 15, "draft_n_accepted": 14},
                id="negative_spec_steps",
            ),
            # missing predicted_n entirely
            param(
                {"draft_n": 15, "draft_n_accepted": 14},
                id="missing_predicted_n",
            ),
        ],
    )  # fmt: skip
    def test_adapt_malformed_payload_degrades_to_none(
        self, bad_payload: dict[str, Any]
    ) -> None:
        """Signature matches but the payload is invalid: adapter yields None, no raise."""
        responses = [_response(spec_decode_stats=bad_payload)]
        # can_adapt matches the llama.cpp signature (draft_n + draft_n_accepted present)
        assert LlamaCppSpecDecodeAdapter.can_adapt(responses) is True
        assert LlamaCppSpecDecodeAdapter.adapt(responses) is None

    def test_adapt_extra_timings_fields_are_ignored(self) -> None:
        """Extra llama.cpp timing fields (prompt_ms, predicted_per_second, etc.) are ignored."""
        payload = {**TIMINGS_PAYLOAD, "cache_n": 7, "prompt_per_second": 49.0}
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(payload))
        assert record is not None
        assert record.num_draft_tokens == 15

    def test_adapt_num_spec_tokens_is_always_none(self) -> None:
        """llama.cpp timings never carry a per-step draft bound, so num_spec_tokens is None."""
        record = LlamaCppSpecDecodeAdapter.adapt(_non_streaming(TIMINGS_PAYLOAD))
        assert record is not None
        assert record.num_spec_tokens is None

    @pytest.mark.parametrize(
        "bad_payload",
        [
            param(
                {"predicted_n": 20, "draft_n": -1, "draft_n_accepted": 0},
                id="negative_draft_n",
            ),
            param(
                {"predicted_n": 20, "draft_n": 15, "draft_n_accepted": -1},
                id="negative_draft_n_accepted",
            ),
        ],
    )  # fmt: skip
    def test_adapt_negative_count_payload_degrades_to_none(
        self, bad_payload: dict[str, Any]
    ) -> None:
        """Negative counter values are caught by the record's ge=0 constraints."""
        responses = [_response(spec_decode_stats=bad_payload)]
        assert LlamaCppSpecDecodeAdapter.can_adapt(responses) is True
        assert LlamaCppSpecDecodeAdapter.adapt(responses) is None
