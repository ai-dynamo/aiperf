# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the chunked inputs.json encoder and the streamed DatasetManager writer."""

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.models import Conversation, InputsFile, SessionPayloads, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.payload_formatting import iter_inputs_json_chunks


def _single_dump(inputs: InputsFile) -> bytes:
    return orjson.dumps(
        inputs.model_dump(exclude_none=True, mode="json"), option=orjson.OPT_INDENT_2
    )


def _payload(content: str) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }


def _session(
    session_id: str | None, num_payloads: int, content: str
) -> SessionPayloads:
    return SessionPayloads(
        session_id=session_id,
        payloads=[_payload(f"{content} {i}") for i in range(num_payloads)],
    )


class TestIterInputsJsonChunks:
    @pytest.mark.parametrize(
        "inputs",
        [
            param(InputsFile(), id="empty"),
            param(InputsFile(data=[_session("s1", 1, "hi")]), id="single-session"),
            param(InputsFile(data=[_session("s1", 0, "")]), id="session-without-payloads"),
            param(InputsFile(data=[_session(None, 2, "hi")]), id="session-without-id"),
            param(
                InputsFile(
                    data=[
                        _session("s1", 1, "before"),
                        SessionPayloads.model_validate(
                            {
                                "session_id": "s2",
                                "payloads": [_payload("extra 0")],
                                "trace_meta": {"origin": "unit", "ids": [1, 2]},
                                "note": "extension field",
                                "dropped_if_none": None,
                            }
                        ),
                        _session("s3", 1, "after"),
                    ]
                ),
                id="session-with-extra-fields",
            ),
            param(
                InputsFile(
                    data=[
                        _session("s1", 2, 'line1\nline2\ttab "quoted" \\ slash'),
                        _session(None, 0, ""),
                        _session("s3", 3, "unicode é 中 \U0001f600"),
                        SessionPayloads(
                            session_id="s4",
                            payloads=[
                                {},
                                {
                                    "top": None,
                                    "nested": {"none": None, "list": [None, 1.5, True]},
                                    "n": 0,
                                },
                            ],
                        ),
                    ]
                ),
                id="escapes-unicode-none-values-mixed",
            ),
        ],
    )  # fmt: skip
    def test_iter_inputs_json_chunks_concatenation_matches_single_dump(
        self, inputs: InputsFile
    ) -> None:
        assert b"".join(iter_inputs_json_chunks(inputs)) == _single_dump(inputs)

    def test_iter_inputs_json_chunks_flushes_at_threshold(self) -> None:
        chunk_bytes = 4096
        inputs = InputsFile(data=[_session(f"s{i}", 2, "x" * 200) for i in range(100)])

        chunks = list(iter_inputs_json_chunks(inputs, chunk_bytes=chunk_bytes))

        assert len(chunks) > 1
        assert all(len(chunk) == chunk_bytes for chunk in chunks[:-1])
        assert 0 < len(chunks[-1]) <= chunk_bytes
        assert b"".join(chunks) == _single_dump(inputs)

    def test_iter_inputs_json_chunks_bounds_chunks_for_oversized_session(self) -> None:
        chunk_bytes = 4096
        inputs = InputsFile(data=[_session("big", 64, "y" * 1024)])

        chunks = list(iter_inputs_json_chunks(inputs, chunk_bytes=chunk_bytes))

        assert len(chunks) > 1
        assert all(len(chunk) == chunk_bytes for chunk in chunks[:-1])
        assert 0 < len(chunks[-1]) <= chunk_bytes
        assert b"".join(chunks) == _single_dump(inputs)

    def test_iter_inputs_json_chunks_bounds_chunks_for_oversized_extra_field(
        self,
    ) -> None:
        chunk_bytes = 4096
        inputs = InputsFile(
            data=[
                _session("s1", 1, "small"),
                SessionPayloads.model_validate(
                    {
                        "session_id": "huge-extra",
                        "payloads": [_payload("p")],
                        "blob": "z" * (4 * chunk_bytes),
                    }
                ),
            ]
        )

        chunks = list(iter_inputs_json_chunks(inputs, chunk_bytes=chunk_bytes))

        assert len(chunks) > 4
        assert all(len(chunk) == chunk_bytes for chunk in chunks[:-1])
        assert 0 < len(chunks[-1]) <= chunk_bytes
        assert b"".join(chunks) == _single_dump(inputs)

    @pytest.mark.parametrize("chunk_bytes", [0, -1])
    def test_iter_inputs_json_chunks_rejects_non_positive_chunk_bytes(
        self, chunk_bytes: int
    ) -> None:
        with pytest.raises(ValueError, match="chunk_bytes must be positive"):
            next(iter_inputs_json_chunks(InputsFile(), chunk_bytes=chunk_bytes))

    @pytest.mark.parametrize(
        "inputs",
        [
            param(
                InputsFile.model_validate(
                    {
                        "data": [],
                        "export_meta": {"origin": "unit", "ids": [1, 2]},
                    }
                ),
                id="top-level-extras-empty-data",
            ),
            param(
                InputsFile.model_validate(
                    {
                        "data": [
                            {"session_id": "s1", "payloads": [_payload("hi 0")]},
                        ],
                        "note": "extension field",
                        "dropped_if_none": None,
                    }
                ),
                id="top-level-extras-with-data",
            ),
        ],
    )  # fmt: skip
    def test_iter_inputs_json_chunks_preserves_top_level_extra_fields(
        self, inputs: InputsFile
    ) -> None:
        chunk_bytes = 64
        chunks = list(iter_inputs_json_chunks(inputs, chunk_bytes=chunk_bytes))

        assert all(len(chunk) == chunk_bytes for chunk in chunks[:-1])
        assert 0 < len(chunks[-1]) <= chunk_bytes
        assert b"".join(chunks) == _single_dump(inputs)


class TestGenerateInputsJsonFileStreaming:
    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_writes_multiple_bounded_chunks(
        self, benchmark_run, tmp_path: Path
    ) -> None:
        manager = DatasetManager(run=benchmark_run, service_id="test_dataset_manager")
        manager.dataset = {
            f"session_{i}": Conversation(
                session_id=f"session_{i}",
                turns=[
                    Turn(role="user", raw_payload=_payload("y" * 1024))
                    for _ in range(2)
                ],
            )
            for i in range(1200)
        }
        expected = _single_dump(
            manager._generate_input_payloads(ModelEndpointInfo.from_run(benchmark_run))
        )
        assert len(expected) > 2 * BYTES_PER_MIB
        write_sizes: list[int] = []

        class _RecordingFile:
            def __init__(self, path: Path) -> None:
                self._file = path.open("wb")

            async def write(self, data: bytes) -> None:
                write_sizes.append(len(data))
                self._file.write(data)

            async def __aenter__(self) -> "_RecordingFile":
                return self

            async def __aexit__(self, *exc: object) -> None:
                self._file.close()

        with patch("aiofiles.open", lambda path, mode: _RecordingFile(Path(path))):
            await manager._generate_inputs_json_file()

        assert len(write_sizes) > 1
        assert all(size == BYTES_PER_MIB for size in write_sizes[:-1])
        assert 0 < write_sizes[-1] <= BYTES_PER_MIB
        assert (tmp_path / "inputs.json").read_bytes() == expected
        assert not (tmp_path / "inputs.tmp").exists()
