# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for integration tests."""

import base64
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.harness.utils import AIPerfResults, VideoDetails

logger = AIPerfLogger(__name__)


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


def _check_mp4_fragmentation(video_bytes: bytes) -> bool:
    """Check if MP4 video is fragmented by looking for moof (movie fragment) boxes.

    Fragmented MP4s contain 'moof' boxes instead of a single 'moov' box.
    Non-fragmented MP4s with faststart have 'moov' before 'mdat'.

    Args:
        video_bytes: Raw video file bytes

    Returns:
        True if the MP4 is fragmented, False otherwise
    """
    # Look for 'moof' (movie fragment) box which indicates fragmentation
    # MP4 boxes are: [4 bytes size][4 bytes type][data]
    # We search for b'moof' in the first 10KB which should contain the header structure
    header_size = min(len(video_bytes), 10240)
    return b"moof" in video_bytes[:header_size]


def extract_base64_video_details(base64_data: str) -> VideoDetails:
    """Decode base64 video data and extract file details using ffprobe via stdin.

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
        "-count_frames",
        "pipe:0",
    ]
    result = subprocess.run(cmd, input=video_bytes, capture_output=True, check=True)

    probe_data = orjson.loads(result.stdout)
    format_info = probe_data["format"]
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    fps_parts = video_stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    # Try to get duration from format first, fallback to stream, or calculate from frames
    duration = format_info.get("duration")
    if not duration:
        duration = video_stream.get("duration")
    if not duration:
        # Use nb_read_frames (from -count_frames) or nb_frames if available
        frame_count = video_stream.get("nb_read_frames") or video_stream.get(
            "nb_frames"
        )
        if frame_count and fps:
            duration = float(frame_count) / fps

    # Check for MP4 fragmentation
    is_fragmented = False
    format_name = format_info.get("format_name", "unknown")
    if "mp4" in format_name.lower():
        is_fragmented = _check_mp4_fragmentation(video_bytes)

    # Extract audio stream info if present
    audio_stream = next(
        (s for s in probe_data["streams"] if s["codec_type"] == "audio"), None
    )
    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    audio_sample_rate = (
        int(audio_stream["sample_rate"])
        if audio_stream and "sample_rate" in audio_stream
        else None
    )
    audio_channels = audio_stream.get("channels") if audio_stream else None

    try:
        return VideoDetails(
            format_name=format_name,
            duration=float(duration) if duration else 0.0,
            codec_name=video_stream.get("codec_name", "unknown"),
            width=video_stream.get("width", 0),
            height=video_stream.get("height", 0),
            fps=fps,
            pix_fmt=video_stream.get("pix_fmt"),
            is_fragmented=is_fragmented,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
        )
    except Exception as e:
        if result.stderr:
            logger.error(result.stderr.decode())
        if result.stdout:
            logger.error(result.stdout.decode())
        raise RuntimeError(f"Failed to extract video details: {e!r}") from e


def iter_video_details(result: AIPerfResults) -> Iterator[VideoDetails]:
    """Yield VideoDetails for every video found in the result payloads."""
    if (
        result.inputs is None
        or not hasattr(result.inputs, "data")
        or not isinstance(result.inputs.data, Iterable)
    ):
        return
    for session in result.inputs.data:
        if not hasattr(session, "payloads") or not isinstance(
            session.payloads, Iterable
        ):
            continue
        for payload in session.payloads:
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []):
                if not isinstance(message, dict):
                    continue
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict) or "video_url" not in item:
                            continue
                        video_url = item["video_url"]
                        if not isinstance(video_url, dict):
                            continue
                        url = video_url.get("url")
                        if not isinstance(url, str) or "," not in url:
                            continue
                        video_data = url.split(",", 1)[1]
                        yield extract_base64_video_details(video_data)


def first_video_details(result: AIPerfResults) -> VideoDetails | None:
    """Return VideoDetails for the first video in the result, or None."""
    return next(iter_video_details(result), None)
