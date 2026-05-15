# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import subprocess
from collections.abc import Iterable, Iterator

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.harness.utils import AIPerfResults, VideoDetails

logger = AIPerfLogger(__name__)


def check_mp4_fragmentation(video_bytes: bytes) -> bool:
    header_size = min(len(video_bytes), 10240)
    return b"moof" in video_bytes[:header_size]


def extract_base64_video_details(base64_data: str) -> VideoDetails:
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

    duration = format_info.get("duration")
    if not duration:
        duration = video_stream.get("duration")
    if not duration:
        frame_count = video_stream.get("nb_read_frames") or video_stream.get(
            "nb_frames"
        )
        if frame_count and fps:
            duration = float(frame_count) / fps

    is_fragmented = False
    format_name = format_info.get("format_name", "unknown")
    if "mp4" in format_name.lower():
        is_fragmented = check_mp4_fragmentation(video_bytes)

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
    return next(iter_video_details(result), None)
