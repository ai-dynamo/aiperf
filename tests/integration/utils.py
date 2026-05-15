# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for integration tests."""

import base64
from pathlib import Path

import orjson


def create_mooncake_trace_file(
    tmp_path: Path,
    traces: list[dict],
    filename: str = "traces.jsonl",
) -> Path:
    """Create a Mooncake trace JSONL file for testing.

    Args:
        tmp_path: Temporary directory path
        traces: List of trace dictionaries to write
        filename: Name of the trace file

    Returns:
        Path to the created trace file
    """
    trace_file = tmp_path / filename
    with open(trace_file, "wb") as f:
        for trace in traces:
            f.write(orjson.dumps(trace) + b"\n")
    return trace_file


def create_sagemaker_capture_record(
    messages: list[dict] | None = None,
    max_tokens: int | None = 50,
    prompt_tokens: int = 28,
    completion_tokens: int = 15,
    inference_time: str = "2026-04-29T00:03:18Z",
    event_id: str = "e4378ff2-0000-0000-0000-000000000000",
    encoding: str = "JSON",
) -> dict:
    """Build a SageMaker Data Capture record dict for testing."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]

    input_payload: dict = {"messages": messages}
    if max_tokens is not None:
        input_payload["max_tokens"] = max_tokens

    output_payload = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    def _encode_data(payload: dict, enc: str) -> tuple[str, str]:
        raw = orjson.dumps(payload).decode()
        if enc == "BASE64":
            return base64.b64encode(orjson.dumps(payload)).decode(), "BASE64"
        if enc == "JSON":
            return raw, "JSON"
        raise ValueError(f"Unsupported encoding for test helper: {enc}")

    input_data, input_enc = _encode_data(input_payload, encoding)
    output_data, output_enc = _encode_data(output_payload, encoding)

    return {
        "captureData": {
            "endpointInput": {
                "observedContentType": "application/json",
                "mode": "INPUT",
                "data": input_data,
                "encoding": input_enc,
            },
            "endpointOutput": {
                "observedContentType": "application/json",
                "mode": "OUTPUT",
                "data": output_data,
                "encoding": output_enc,
            },
        },
        "eventMetadata": {
            "eventId": event_id,
            "inferenceTime": inference_time,
        },
        "eventVersion": "0",
    }


def create_sagemaker_capture_file(
    tmp_path: Path,
    records: list[dict],
    filename: str = "capture.jsonl",
) -> Path:
    """Create a SageMaker Data Capture JSONL file for testing.

    Args:
        tmp_path: Temporary directory path
        records: List of capture record dicts (from create_sagemaker_capture_record)
        filename: Name of the capture file

    Returns:
        Path to the created capture file
    """
    capture_file = tmp_path / filename
    with open(capture_file, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")
    return capture_file


def create_rankings_dataset(tmp_path: Path, num_entries: int) -> Path:
    """Create a rankings dataset for testing.

    Args:
        tmp_path: Temporary directory path
        num_entries: Number of entries to create in the dataset

    Returns:
        Path to the created dataset file
    """
    dataset_path = tmp_path / "rankings.jsonl"
    with open(dataset_path, "w") as f:
        for i in range(num_entries):
            entry = {
                "texts": [
                    {"name": "query", "contents": [f"What is AI topic {i}?"]},
                    {"name": "passages", "contents": [f"AI passage {i}"]},
                ]
            }
            f.write(orjson.dumps(entry).decode("utf-8") + "\n")
    return dataset_path
