# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang counter normalization and malformed-payload handling."""

from typing import Any

import pytest
from pytest import param

from aiperf.common.models import ParsedResponse
from aiperf.spec_decode.sglang_adapter import SGLangSpecDecodeAdapter

# Histogram: 20 steps, 30 accepted drafts, neutral acceptance length 2.5.
SUMMARY_PAYLOAD: dict[str, Any] = {
    "spec_accept_length": 2.55,
    "spec_accept_rate": 0.5,
    "spec_num_correct_drafts": 30,
    "spec_num_proposed_drafts": 60,
    "spec_verify_ct": 20,
    "spec_correct_drafts_histogram": [8, 0, 6, 6],
}


def test_normalizes_counts_without_prefill_token() -> None:
    record = SGLangSpecDecodeAdapter.adapt(
        [
            ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=2, usage={"completion_tokens": 51}),
        ]
    )
    assert record is not None
    assert record.engine == "sglang"
    assert record.mean_acceptance_length == 2.5
    assert record.draft_acceptance_rate == 0.5
    assert record.acceptance_histogram == {0: 8, 2: 6, 3: 6}
    assert record.num_accepted_draft_tokens == 30
    assert record.num_draft_tokens == 60
    assert record.num_spec_steps == 20
    assert record.num_spec_tokens is None
    assert record.completion_tokens == 51
    assert record.per_step_accepted is None
    assert record.per_step_drafted is None


@pytest.mark.parametrize(
    "steps", [param(0, id="zero_steps"), param(5, id="all_rejected")],
)  # fmt: skip
def test_zero_steps_and_all_rejected_drafts(steps: int) -> None:
    # Zero-step counts are defensive coverage; v0.5.19 normally omits the details.
    payload = {
        "spec_num_correct_drafts": 0,
        "spec_num_proposed_drafts": 3 * steps,
        "spec_verify_ct": steps,
        "spec_correct_drafts_histogram": [steps] if steps else [],
    }
    record = SGLangSpecDecodeAdapter.adapt(
        [ParsedResponse(perf_ns=1, spec_decode_stats=payload)]
    )
    assert record is not None
    assert record.mean_acceptance_length == 1.0
    assert record.draft_acceptance_rate == 0.0
    assert record.num_spec_steps == steps
    assert record.completion_tokens is None


@pytest.mark.parametrize(
    "override",
    [
        param({"spec_verify_ct": 19}, id="steps_mismatch"),
        param({"spec_num_correct_drafts": 31}, id="accepted_mismatch"),
        param({"spec_num_proposed_drafts": 29}, id="accepted_exceeds_proposed"),
        param({"spec_num_proposed_drafts": -1}, id="negative_count"),
        param({"spec_num_proposed_drafts": True}, id="boolean_count"),
        param({"spec_num_proposed_drafts": "60"}, id="string_count"),
        param({"spec_verify_ct": 20.0}, id="float_count"),
        param({"spec_num_correct_drafts": 10**309}, id="overflowing_count"),
        param({"spec_correct_drafts_histogram": None}, id="null_histogram"),
        param({"spec_correct_drafts_histogram": {}}, id="dict_histogram"),
        param({"spec_correct_drafts_histogram": []}, id="empty_histogram"),
        param({"spec_correct_drafts_histogram": [True, 0, 6, 6]}, id="boolean_bucket"),
        param({"spec_correct_drafts_histogram": [-1, 9, 6, 6]}, id="negative_bucket"),
    ],
)  # fmt: skip
def test_malformed_payload_is_ignored(override: dict[str, Any]) -> None:
    responses = [
        ParsedResponse(perf_ns=1, spec_decode_stats={**SUMMARY_PAYLOAD, **override})
    ]
    assert SGLangSpecDecodeAdapter.adapt(responses) is None


def test_missing_histogram_is_ignored() -> None:
    payload = {k: v for k, v in SUMMARY_PAYLOAD.items() if "histogram" not in k}
    assert (
        SGLangSpecDecodeAdapter.adapt(
            [ParsedResponse(perf_ns=1, spec_decode_stats=payload)]
        )
        is None
    )


def test_absent_payload_is_ignored() -> None:
    assert SGLangSpecDecodeAdapter.can_adapt([]) is False
    assert SGLangSpecDecodeAdapter.adapt([]) is None
