# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for WekaTrace Pydantic models."""

import math

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# Group A: discriminator attacks
# ---------------------------------------------------------------------------


def test_discriminator_unknown_type_rejected():
    """Pin: unknown type tag 'x' must fail tagged-union discrimination."""
    bad = _trace_with_request({"t": 0.0, "type": "x", "model": "m", "in": 10, "out": 1})
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_discriminator_null_type_rejected():
    """Pin: null type must fail discrimination (not coerce to a variant)."""
    bad = _trace_with_request(
        {"t": 0.0, "type": None, "model": "m", "in": 10, "out": 1}
    )
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_discriminator_missing_type_rejected():
    """Pin: absent type field must fail discrimination."""
    bad = _trace_with_request({"t": 0.0, "model": "m", "in": 10, "out": 1})
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_discriminator_uppercase_type_rejected():
    """Pin: discriminator is case-sensitive; 'N' must not match 'n'."""
    bad = _trace_with_request({"t": 0.0, "type": "N", "model": "m", "in": 10, "out": 1})
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_discriminator_empty_string_type_rejected():
    """Pin: empty-string discriminator must fail (no variant matches '')."""
    bad = _trace_with_request({"t": 0.0, "type": "", "model": "m", "in": 10, "out": 1})
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_discriminator_nested_subagent_rejected():
    """Pin: WekaSubagentEntry.requests is list[WekaNormalRequest]; a nested
    subagent must be rejected (no tagged union at the inner level)."""
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


def test_discriminator_streaming_inside_subagent_rejected():
    """Pin: inner list accepts only WekaNormalRequest; a streaming request
    with type='s' must be rejected (Literal['n'] mismatch)."""
    inner_streaming = {
        "t": 0.0,
        "type": "s",
        "model": "m",
        "in": 10,
        "out": 1,
        "ttft": 0.2,
    }
    d = _valid_subagent([inner_streaming])
    with pytest.raises(ValidationError):
        WekaSubagentEntry.model_validate(d)


def test_discriminator_ttft_on_normal_request_rejected():
    """Pin: WekaNormalRequest has extra='forbid'; ttft is streaming-only
    and must be rejected on a normal request."""
    bad = _trace_with_request(
        {"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1, "ttft": 0.2}
    )
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


# ---------------------------------------------------------------------------
# Group B: numeric boundary + non-finite (currently accepted)
# ---------------------------------------------------------------------------


def test_normal_request_negative_input_length_accepted():
    """Pin: no lower bound on input_length; negative int parses."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": -1, "out": 1}
    )
    assert req.input_length == -1


def test_normal_request_zero_input_length_accepted():
    """Pin: zero input_length parses (no ge=1 constraint)."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": 0, "out": 1}
    )
    assert req.input_length == 0


def test_normal_request_huge_input_length_accepted():
    """Pin: no upper bound on input_length; 10**9 parses."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": 10**9, "out": 1}
    )
    assert req.input_length == 10**9


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


# ---------------------------------------------------------------------------
# Group C: type coercion probes
# ---------------------------------------------------------------------------


def test_normal_request_string_input_coerced_to_int():
    """Pin: pydantic lax mode coerces numeric strings to int for 'in'."""
    req = WekaNormalRequest.model_validate(
        {"t": 0.0, "type": "n", "model": "m", "in": "10", "out": 1}
    )
    assert req.input_length == 10
    assert isinstance(req.input_length, int)


def test_normal_request_float_input_rejected():
    """Pin: non-whole float input (10.5) is rejected by pydantic v2 lax
    int coercion; only whole-valued floats coerce."""
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


# ---------------------------------------------------------------------------
# Group D: required-field and Literal edge
# ---------------------------------------------------------------------------


def test_weka_trace_missing_required_id_rejected():
    """Pin: 'id' is required at the trace level."""
    bad = {k: v for k, v in _VALID.items() if k != "id"}
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_weka_trace_missing_required_block_size_rejected():
    """Pin: 'block_size' is required at the trace level."""
    bad = {k: v for k, v in _VALID.items() if k != "block_size"}
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


@pytest.mark.parametrize("block_size", [0, -1, -64])
def test_weka_trace_non_positive_block_size_rejected(block_size: int):
    """Pin: block_size must be > 0. Zero reaches token-tiling divisors
    (weka_trace.py _chain_init_tokens; weka_synth_buf.py init_turn_0/
    advance_turn) as a ZeroDivisionError, and negatives silently corrupt
    tiling — the gt=0 constraint converts both into a parse-time
    ValidationError that _load_single_file wraps as DatasetLoaderError.
    """
    bad = dict(_VALID)
    bad["block_size"] = block_size
    with pytest.raises(ValidationError):
        WekaTrace.model_validate(bad)


def test_weka_trace_hash_id_scope_global_rejected_by_schema():
    """'global' hash_id_scope is rejected at schema level: v1 loader only
    implements local-scope synthesis (hashes scoped per-trace). Accepting
    'global' at the schema would let misconfigured traces load and silently
    misbehave — global-scope support is a future feature, and until it is
    implemented, the schema rejects.
    """
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
