# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.cli_commands.dynamo_trace import dynamo_trace
from aiperf.dataset.loader.weka_trace_models import WekaTrace


def _record(
    session_id: str,
    received_ms: int,
    hashes: list[int],
    *,
    parent_session_id: str | None = None,
    input_length: int | None = None,
) -> dict:
    agent_context = {"session_id": session_id}
    if parent_session_id is not None:
        agent_context["parent_session_id"] = parent_session_id
    return {
        "event": {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "agent_context": agent_context,
            "request": {
                "model": "test-model",
                "output_tokens": 8,
                "request_received_ms": received_ms,
                "total_time_ms": 100,
                "replay": {
                    "trace_block_size": 16,
                    "input_length": input_length
                    if input_length is not None
                    else len(hashes) * 16,
                    "input_sequence_hashes": hashes,
                },
            },
        }
    }


@pytest.mark.parametrize(
    "compressed",
    [param(False, id="jsonl"), param(True, id="jsonl-gz")],
)  # fmt: skip
def test_dynamo_trace_writes_weka_subagents_from_agent_context(
    tmp_path: Path, compressed: bool
) -> None:
    input_file = tmp_path / f"trace.jsonl{'.gz' if compressed else ''}"
    records = [
        _record("root", 1_000, [10, 20, 99], input_length=33),
        _record("child", 1_500, [10, 30], parent_session_id="root"),
        _record("root", 5_000, [10, 40]),
    ]
    writer = gzip.open if compressed else Path.open
    with writer(input_file, "wb") as trace_file:
        for record in records:
            trace_file.write(orjson.dumps(record) + b"\n")

    output = tmp_path / "weka"
    dynamo_trace(input_file, output=output)

    traces = [
        WekaTrace.model_validate(orjson.loads(path.read_bytes()))
        for path in sorted(output.glob("trace_*.json"))
    ]
    assert [trace.id for trace in traces] == ["root"]
    root = traces[0]
    assert root.hash_id_scope == "local"
    assert [request.type for request in root.requests] == ["n", "subagent", "n"]
    assert [root.requests[0].t, root.requests[-1].t] == [0.0, 4.0]
    assert root.requests[0].hash_ids == [10, 20, 99]
    child = root.requests[1]
    assert child.agent_id == "child"
    assert child.t == 0.5
    assert child.requests[0].hash_ids == [10, 30]
