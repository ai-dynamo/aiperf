# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for WekaTrace Pydantic models."""

import math

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaSubagentEntry,
    WekaTrace,
)

_VALID = {
    "id": "t1",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1},
    ],
}


def _trace_with_request(req: dict) -> dict:
    """Build a WekaTrace dict with a single inner request."""
    return {
        "id": "t1",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [req],
    }


def _valid_subagent(inner_requests: list[dict]) -> dict:
    """Build a minimal WekaSubagentEntry dict with provided inner requests."""
    return {
        "t": 0.0,
        "type": "subagent",
        "agent_id": "a",
        "subagent_type": "Explore",
        "status": "completed",
        "requests": inner_requests,
        "models": ["m"],
    }


# Group A: discriminator attacks


@pytest.mark.parametrize(
    "req",
    [
        param({"t": 0.0, "type": "x", "model": "m", "in": 10, "out": 1}, id="unknown_type"),
        param({"t": 0.0, "type": None, "model": "m", "in": 10, "out": 1}, id="null_type"),
        param({"t": 0.0, "model": "m", "in": 10, "out": 1}, id="missing_type"),
        param({"t": 0.0, "type": "N", "model": "m", "in": 10, "out": 1}, id="uppercase_type"),
        param({"t": 0.0, "type": "", "model": "m", "in": 10, "out": 1}, id="empty_string_type"),
    ],
)  # fmt: skip
def test_discriminator_invalid_type_rejected(req: dict):
    """Pin: invalid discriminator tags must fail tagged-union discrimination."""
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(_trace_with_request(req))


def test_discriminator_nested_subagent_rejected():
    """Pin: WekaSubagentEntry.requests is list[WekaNormalRequest], so a nested subagent is rejected (no tagged union at the inner level)."""
    inner_subagent = {
        "t": 0.0,
        "type": "subagent",
        "agent_id": "a2",
        "subagent_type": "Explore",
        "status": "completed",
        "requests": [],
        "models": ["m"],
    }
    d = _valid_subagent([inner_subagent])
    with pytest.raises(ValidationError):
        WekaSubagentEntry.model_validate(d)


def test_discriminator_streaming_inside_subagent_accepted():
    """A subagent's inner requests accept both normal and streaming API calls
    (``list[WekaNormalRequest | WekaStreamingRequest]``), matching the top-level
    trace which already allows streaming. A single streaming inner request must
    not abort the whole dataset load under ``extra='forbid'``."""
    inner_streaming = {
        "t": 0.0,
        "type": "s",
        "model": "m",
        "in": 10,
        "out": 1,
        "ttft": 0.2,
    }
    d = _valid_subagent([inner_streaming])
    entry = WekaSubagentEntry.model_validate(d)
    assert [r.type for r in entry.requests] == ["s"]


def test_discriminator_ttft_on_normal_request_rejected():
    """Pin: WekaNormalRequest has extra='forbid', so the streaming-only ttft field is rejected on a normal request."""
    bad = _trace_with_request(
        {"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1, "ttft": 0.2}
    )
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


# Group B: numeric boundary + non-finite (currently accepted)


@pytest.mark.parametrize(
    "in_value",
    [
        param(-1, id="negative"),
        param(0, id="zero"),
        param(10**9, id="huge"),
    ],
)  # fmt: skip
def test_normal_request_input_length_no_bounds(in_value: int):
    """Pin: input_length has no lower/upper bound; negative, zero, huge all parse."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": in_value, "out": 1}
    )
    assert req.input_length == in_value


def test_normal_request_negative_output_length_accepted():
    """Pin: no lower bound on output_length; negative int parses."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": 1, "out": -5}
    )
    assert req.output_length == -5


def test_normal_request_nan_timestamp_accepted():
    """Pin: timestamp is a plain float; NaN is accepted by pydantic float."""
    req = WekaNormalRequest.model_validate(
        {"t": math.nan, "type": "n", "model": "m", "in": 1, "out": 1}
    )
    assert math.isnan(req.t)


def test_normal_request_pos_inf_timestamp_accepted():
    """Pin: +inf timestamp is accepted by pydantic float."""
    req = WekaNormalRequest.model_validate(
        {"t": math.inf, "type": "n", "model": "m", "in": 1, "out": 1}
    )
    assert req.t == math.inf


def test_normal_request_neg_inf_timestamp_accepted():
    """Pin: -inf timestamp is accepted by pydantic float."""
    req = WekaNormalRequest.model_validate(
        {"t": -math.inf, "type": "n", "model": "m", "in": 1, "out": 1}
    )
    assert req.t == -math.inf


# Group C: type coercion probes


def test_normal_request_string_input_coerced_to_int():
    """Pin: pydantic lax mode coerces numeric strings to int for 'in'."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": "10", "out": 1}
    )
    assert req.input_length == 10
    assert isinstance(req.input_length, int)


def test_normal_request_float_input_rejected():
    """Pin: a non-whole float input (10.5) is rejected by pydantic v2 lax int coercion, which only coerces whole-valued floats."""
    with pytest.raises(ValidationError):
        WekaNormalRequest.model_validate(
            {"t": 0.0, "type": "n", "model": "m", "in": 10.5, "out": 1}
        )


def test_normal_request_whole_float_input_coerced():
    """Pin: whole-valued float (10.0) coerces to int under pydantic v2 lax."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": 10.0, "out": 1}
    )
    assert req.input_length == 10
    assert isinstance(req.input_length, int)


def test_hash_ids_with_fractional_float_rejected():
    """Pin: hash_ids: list[int]; a fractional float (1.5) must be rejected."""
    with pytest.raises(ValidationError):
        WekaNormalRequest.model_validate(
            {
                "t": 0.0,
                "type": "n",
                "model": "m",
                "in": 1,
                "out": 1,
                "hash_ids": [1.5],
            }
        )


# Group D: required-field and Literal edge


@pytest.mark.parametrize(
    "field", [param("id", id="id"), param("block_size", id="block_size")]
)
def test_weka_trace_missing_required_field_rejected(field: str):
    """Pin: 'id' and 'block_size' are required at the trace level."""
    bad = {k: v for k, v in _VALID.items() if k != field}
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


@pytest.mark.parametrize("block_size", [0, -1, -64])
def test_weka_trace_non_positive_block_size_rejected(block_size: int):
    """Pin: block_size must be > 0, so the gt=0 constraint turns tiling-divisor zeros and corrupting negatives into a parse-time ValidationError."""
    bad = dict(_VALID)
    bad["block_size"] = block_size
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_weka_trace_hash_id_scope_global_rejected_by_schema():
    """A 'global' hash_id_scope is rejected at schema level since the v1 loader only implements per-trace local-scope synthesis."""
    d = dict(_VALID)
    d["hash_id_scope"] = "global"
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(d)


def test_weka_subagent_missing_required_agent_id_rejected():
    """Pin: 'agent_id' is required on WekaSubagentEntry."""
    d = _valid_subagent([])
    del d["agent_id"]
    with pytest.raises(ValidationError):
        WekaSubagentEntry.model_validate(d)
