# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded JSON serialization preserves bytes and yields during real encoding."""

import asyncio
from pathlib import Path
from typing import Any

import orjson
import pytest
import zstandard
from pytest import param

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.models import Conversation, InputsFile, SessionPayloads, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config.artifacts import OutputDefaults
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.json_stream import _CHUNK_SIZE, _STRING_CHARS, iter_json_bytes
from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore


@pytest.mark.parametrize("pretty", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        param({"empty": [{}, [], ""], "null": None}, id="empty"),
        param([True, False, -(2**63), 2**64 - 1, 1e-7, -0.0, 1e20], id="scalars"),
        param({"text": '\"\\\n\t\x00é雪𝄞' * 20_000}, id="large-string"),
        param({"k\n" * 30_000: [1, {"x": (2, 3)}]}, id="large-key"),
        param({"rows": [{"x": i} for i in range(20_000)]}, id="many-rows"),
        param({"nested": [list(range(100_000))]}, id="batched-array"),
        param({"mixed": [0] * 1024 + [True, None, [1, 2]] * 20_000}, id="mixed-array"),
    ],
)  # fmt: skip
async def test_stream_matches_orjson(value: Any, pretty: bool) -> None:
    """Compact and indented chunks reproduce the original encoder bytes."""
    chunks = [chunk async for chunk in iter_json_bytes(value, pretty=pretty)]
    assert all(0 < len(chunk) <= _CHUNK_SIZE for chunk in chunks)
    assert b"".join(chunks) == orjson.dumps(
        value, option=orjson.OPT_INDENT_2 if pretty else 0
    )


@pytest.mark.parametrize("value", [{1: "bad key"}, "\ud800", 2**64, object()])
async def test_stream_rejects_invalid_values(value: Any) -> None:
    """Invalid JSON values retain the original encoder error type."""
    with pytest.raises(orjson.JSONEncodeError):
        _ = [chunk async for chunk in iter_json_bytes(value)]


async def test_stream_rejects_cycle() -> None:
    """Container cycles fail instead of emitting an endless stream."""
    value = []
    value.append(value)
    with pytest.raises(orjson.JSONEncodeError, match="Recursion limit"):
        _ = [chunk async for chunk in iter_json_bytes(value)]


async def test_deep_indentation_keeps_native_calls_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size estimate includes opening and closing indentation."""
    value = [[0] for _ in range(2100)]
    for _ in range(100):
        value = [value]
    expected = orjson.dumps(value, option=orjson.OPT_INDENT_2)
    original = orjson.dumps

    def bounded_dumps(value: Any, **kwargs: Any) -> bytes:
        data = original(value, **kwargs)
        assert len(data) <= _CHUNK_SIZE
        return data

    monkeypatch.setattr(orjson, "dumps", bounded_dumps)
    chunks = [chunk async for chunk in iter_json_bytes(value, pretty=True)]
    assert all(len(chunk) <= _CHUNK_SIZE for chunk in chunks)
    assert b"".join(chunks) == expected


@pytest.mark.parametrize("operation", ["inputs", "payload", "compressed-payload"])
async def test_large_payload_encoding_allows_heartbeats(
    operation: str,
    populated_dataset_manager: DatasetManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real encoding bounds every orjson call and allows heartbeat progress."""
    payload = {"text": 'é\n\\"' * 500_000, "items": list(range(10_000))}
    conversation = Conversation(
        session_id="large", turns=[Turn(role="user", raw_payload=payload)]
    )
    manager = populated_dataset_manager
    manager.dataset = {"large": conversation}
    expected = (
        orjson.dumps(
            manager._generate_input_payloads(
                ModelEndpointInfo.from_run(manager.run)
            ).model_dump(exclude_none=True, mode="json"),
            option=orjson.OPT_INDENT_2,
        )
        if operation == "inputs"
        else orjson.dumps(payload)
    )
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    store = MemoryMapDatasetBackingStore(
        format=MemoryMapFormat.PAYLOAD_BYTES,
        compress_only=operation == "compressed-payload",
    )
    await store.initialize()
    loop = asyncio.get_running_loop()
    heartbeats = 0
    samples = []
    original = orjson.dumps

    def heartbeat() -> None:
        nonlocal heartbeats, handle
        heartbeats += 1
        handle = loop.call_soon(heartbeat)

    def bounded_dumps(value: Any, **kwargs: Any) -> bytes:
        if isinstance(value, str):
            assert len(value) <= _STRING_CHARS
        samples.append(heartbeats)
        encoded = original(value, **kwargs)
        assert len(encoded) <= _CHUNK_SIZE
        return encoded

    handle = loop.call_soon(heartbeat)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(orjson, "dumps", bounded_dumps)
            if operation == "inputs":
                await manager._generate_inputs_json_file()
            else:
                await store.add_conversation("large", conversation)
        assert samples[-1] > samples[0]
        if operation == "inputs":
            data = (
                manager.run.cfg.artifacts.dir / OutputDefaults.INPUTS_JSON_FILE
            ).read_bytes()
        else:
            await store.finalize()
            offset = store._payload_offsets["large"][0]
            assert offset.offset == 0
            assert offset.size == len(expected)
            if store._compress_only:
                with (
                    store._compressed_data_path.open("rb") as file,
                    zstandard.ZstdDecompressor().stream_reader(file) as reader,
                ):
                    data = reader.read()
            else:
                data = store._data_path.read_bytes()
        assert data == expected
    finally:
        handle.cancel()
        await store.stop()


async def test_inputs_stream_preserves_model_serialization(
    populated_dataset_manager: DatasetManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session omission and nested nulls match the previous Pydantic export."""
    inputs = InputsFile(
        data=[SessionPayloads(payloads=[{"value": None}]) for _ in range(2000)]
    )
    manager = populated_dataset_manager
    monkeypatch.setattr(manager, "_generate_input_payloads", lambda _: inputs)
    await manager._generate_inputs_json_file()
    data = (
        manager.run.cfg.artifacts.dir / OutputDefaults.INPUTS_JSON_FILE
    ).read_bytes()
    assert data == orjson.dumps(
        inputs.model_dump(exclude_none=True, mode="json"), option=orjson.OPT_INDENT_2
    )
