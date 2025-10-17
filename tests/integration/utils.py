# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for integration tests."""

import base64
import json
import subprocess
from pathlib import Path

import orjson

from tests.integration.models import VideoDetails


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


def extract_base64_video_details(base64_data: str) -> VideoDetails:
    """Decode base64 MP4 data and extract file details using ffprobe via stdin.

    Args:
        base64_data: Base64-encoded video data

    Returns:
        VideoDetails object containing video metadata
    """
    video_bytes = base64.b64decode(base64_data)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "pipe:0",
    ]
    result = subprocess.run(cmd, input=video_bytes, capture_output=True, check=True)

    probe_data = json.loads(result.stdout)
    format_info = probe_data["format"]
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    fps_parts = video_stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    return VideoDetails(
        format_name=format_info["format_name"],
        duration=float(format_info["duration"]),
        codec_name=video_stream["codec_name"],
        width=video_stream["width"],
        height=video_stream["height"],
        fps=fps,
        pix_fmt=video_stream.get("pix_fmt"),
    )
