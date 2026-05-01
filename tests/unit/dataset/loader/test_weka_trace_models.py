# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaSubagentEntry,
    WekaTrace,
)


def test_weka_normal_request_parses_with_alias_fields():
    d = {
        "t": 0.0,
        "type": "n",
        "model": "claude-opus-4-5-20251101",
        "in": 71175,
        "out": 169,
        "hash_ids": [1, 2, 3],
        "input_types": ["text"],
        "output_types": ["text", "thinking"],
        "stop": "tool_use",
        "api_time": 7.34,
        "think_time": 0.0,
    }
    req = WekaNormalRequest.model_validate(d)
    assert req.t == 0.0
    assert req.model == "claude-opus-4-5-20251101"
    assert req.input_length == 71175
    assert req.output_length == 169
    assert req.hash_ids == [1, 2, 3]
    assert req.api_time == 7.34
    assert req.think_time == 0.0


def test_weka_normal_request_rejects_extra_fields():
    d = {"t": 0.0, "type": "n", "model": "m", "in": 1, "out": 1, "extra": "nope"}
    with pytest.raises(ValidationError):
        WekaNormalRequest.model_validate(d)


def test_weka_subagent_entry_parses_nested_requests():
    d = {
        "t": 134.227,
        "type": "subagent",
        "agent_id": "agent_001",
        "subagent_type": "Explore",
        "duration_ms": 126015,
        "total_tokens": 39427,
        "tool_use_count": 27,
        "status": "completed",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-haiku-4-5-20251001",
                "in": 9526,
                "out": 363,
                "hash_ids": [1, 2],
            }
        ],
        "models": ["claude-haiku-4-5-20251001"],
        "tool_tokens": 8306,
        "system_tokens": 735,
    }
    sa = WekaSubagentEntry.model_validate(d)
    assert sa.agent_id == "agent_001"
    assert sa.subagent_type == "Explore"
    assert sa.duration_ms == 126015
    assert len(sa.requests) == 1
    assert sa.requests[0].model == "claude-haiku-4-5-20251001"


def test_weka_trace_discriminates_request_union():
    d = {
        "id": "t1",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1},
            {
                "t": 1.0,
                "type": "subagent",
                "agent_id": "a",
                "subagent_type": "X",
                "duration_ms": 100,
                "total_tokens": 0,
                "tool_use_count": 0,
                "status": "completed",
                "requests": [],
                "models": ["m2"],
            },
        ],
    }
    tr = WekaTrace.model_validate(d)
    assert len(tr.requests) == 2
    assert isinstance(tr.requests[0], WekaNormalRequest)
    assert isinstance(tr.requests[1], WekaSubagentEntry)


def test_weka_trace_totals_optional():
    d = {
        "id": "t1",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [],
        "totals": {"x": 1},
    }
    tr = WekaTrace.model_validate(d)
    assert tr.totals == {"x": 1}


def test_weka_streaming_request_carries_ttft():
    d = {
        "t": 0.0,
        "type": "s",
        "model": "m",
        "in": 100,
        "out": 10,
        "hash_ids": [1],
        "ttft": 0.25,
        "api_time": 1.0,
        "think_time": 0.0,
    }
    req = WekaStreamingRequest.model_validate(d)
    assert req.ttft == 0.25
    assert req.type == "s"


def test_weka_trace_accepts_streaming_top_level():
    d = {
        "id": "t1",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {"t": 0.0, "type": "s", "model": "m", "in": 100, "out": 10, "ttft": 0.2}
        ],
    }
    tr = WekaTrace.model_validate(d)
    assert len(tr.requests) == 1
    assert isinstance(tr.requests[0], WekaStreamingRequest)
