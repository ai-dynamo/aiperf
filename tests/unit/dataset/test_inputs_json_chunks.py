# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the chunked inputs.json encoder and the streamed DatasetManager writer."""

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.models import Conversation, InputsFile, SessionPayloads, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.payload_formatting import iter_inputs_json_chunks


def _single_dump(inputs: InputsFile) -> bytes:
    """The whole-document encoding the chunked encoder must reproduce byte for byte."""
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
            param(
                InputsFile(
                    data=[
                        _session("s1", 2, 'line1\nline2\ttab "quoted" \\ slash'),
                        _session(None, 0, ""),
                        _session("s3", 3, "unicode é 中 \U0001f600"),
                    ]
                ),
                id="escapes-unicode-no-session-id",
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
        assert all(len(chunk) >= chunk_bytes for chunk in chunks[:-1])
        assert all(len(chunk) < 2 * chunk_bytes for chunk in chunks)
        assert b"".join(chunks) == _single_dump(inputs)


class TestGenerateInputsJsonFileStreaming:
    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_large_dataset_matches_single_dump(
        self, benchmark_run, tmp_path: Path
    ) -> None:
        """A dataset several times the flush threshold is written across multiple
        chunks and still lands on disk byte-identical to the one-shot encoding."""
        manager = DatasetManager(run=benchmark_run, service_id="test_dataset_manager")
        manager.dataset = {
            f"session_{i}": Conversation(
                session_id=f"session_{i}",
                turns=[
                    Turn(role="user", raw_payload=_payload("y" * 1024))
                    for _ in range(2)
                ],
            )
            for i in range(2000)
        }
        expected = _single_dump(
            manager._generate_input_payloads(ModelEndpointInfo.from_run(benchmark_run))
        )
        assert len(expected) > 2 * BYTES_PER_MIB

        await manager._generate_inputs_json_file()

        assert (tmp_path / "inputs.json").read_bytes() == expected
        assert not (tmp_path / "inputs.tmp").exists()
