# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Python-side detection for runner-owned recorded graph inputs."""

from __future__ import annotations

import gzip
from pathlib import Path

import orjson
import pytest

from aiperf.dataset.loader.recorded_graph import (
    DynamoTraceNativeLoader,
    WekaTraceNativeLoader,
)


def _weka(trace_id: str = "trace") -> dict[str, object]:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": 16,
        "hash_id_scope": "local",
        "requests": [],
    }


def _dynamo(request_id: str = "r") -> dict[str, object]:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1,
        "agent_context": {"session_id": "s"},
        "request": {"request_id": request_id},
    }


def test_weka_detects_inline_file_directory_and_mixed_case_suffix(
    tmp_path: Path,
) -> None:
    assert WekaTraceNativeLoader.can_load(data=_weka())
    path = tmp_path / "TRACE.JSON"
    path.write_bytes(orjson.dumps(_weka()))
    assert WekaTraceNativeLoader.can_load(filename=str(path))
    assert WekaTraceNativeLoader.can_load(filename=str(tmp_path))


def test_weka_detection_rejects_foreign_keys_bool_block_size_and_bad_json(
    tmp_path: Path,
) -> None:
    foreign = _weka()
    foreign["messages"] = []
    assert not WekaTraceNativeLoader.can_load(data=foreign)
    invalid_block = _weka()
    invalid_block["block_size"] = True
    assert not WekaTraceNativeLoader.can_load(data=invalid_block)
    path = tmp_path / "bad.json"
    path.write_text("{not-json")
    assert not WekaTraceNativeLoader.can_load(filename=str(path))


def test_dynamo_detects_bare_enveloped_plain_gzip_and_directory(
    tmp_path: Path,
) -> None:
    assert DynamoTraceNativeLoader.can_load(data=_dynamo())
    assert DynamoTraceNativeLoader.can_load(data={"timestamp": 1, "event": _dynamo()})

    plain = tmp_path / "trace.jsonl"
    plain.write_bytes(orjson.dumps(_dynamo("plain")) + b"\n")
    assert DynamoTraceNativeLoader.can_load(filename=str(plain))
    plain.unlink()

    gzip_path = tmp_path / "trace.000000.jsonl.gz"
    with gzip.open(gzip_path, "wb") as stream:
        stream.write(orjson.dumps({"timestamp": 1, "event": _dynamo("gzip")}) + b"\n")
    assert DynamoTraceNativeLoader.can_load(filename=str(gzip_path))
    assert DynamoTraceNativeLoader.can_load(filename=str(tmp_path))


def test_dynamo_directory_uses_numeric_segment_order_and_fails_closed(
    tmp_path: Path,
) -> None:
    with gzip.open(tmp_path / "trace.1000000.jsonl.gz", "wb") as stream:
        stream.write(orjson.dumps(_dynamo("later")) + b"\n")
    with gzip.open(tmp_path / "trace.999999.jsonl.gz", "wb") as stream:
        stream.write(b"not-json\n")
    assert not DynamoTraceNativeLoader.can_load(filename=str(tmp_path))

    corrupt = tmp_path / "corrupt.jsonl.gz"
    corrupt.write_bytes(b"not gzip")
    assert not DynamoTraceNativeLoader.can_load(filename=str(corrupt))


def test_configuration_only_loaders_cannot_enter_python_data_plane() -> None:
    loader = object.__new__(WekaTraceNativeLoader)
    with pytest.raises(RuntimeError, match="aiperf runner"):
        loader.load_dataset()
    with pytest.raises(RuntimeError, match="legacy linear"):
        loader.convert_to_conversations({})
