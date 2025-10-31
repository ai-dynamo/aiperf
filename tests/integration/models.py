# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for integration tests."""

from asyncio.subprocess import Process
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aiperf.common.models import (
    InputsFile,
    JsonExportData,
    MetricRecordInfo,
    RawRecordInfo,
    SessionPayloads,
)


@dataclass
class AIPerfSubprocessResult:
    """AIPerf subprocess result."""

    exit_code: int
    output_dir: Path


@dataclass
class AIPerfMockServer:
    """AIPerfMockServer server info."""

    host: str
    port: int
    url: str
    process: Process

    @property
    def dcgm_urls(self) -> list[str]:
        """AIPerfMockServer server DCGM metrics URLs."""
        return [f"{self.url}/dcgm{i}/metrics" for i in [1, 2]]


class VideoDetails(BaseModel):
    """Video file metadata extracted from ffprobe."""

    format_name: str
    duration: float
    codec_name: str
    width: int
    height: int
    fps: float
    pix_fmt: str | None = None


class AIPerfResults:
    """Simple wrapper for AIPerf results.

    All JSON-based artifacts are loaded as Pydantic models for type safety and validation.
    """

    def __init__(self, result: AIPerfSubprocessResult) -> None:
        self.artifacts_dir = result.output_dir
        self.exit_code = result.exit_code

        self.json = self._load_json_export()
        self.csv = self._load_text_file("**/*aiperf.csv")
        self.inputs = self._load_inputs()
        self.jsonl = self._load_jsonl_records()
        self.raw_records = self._load_raw_records()
        self.log = self._load_text_file("**/logs/aiperf.log")

    def _find_file(self, pattern: str) -> Path | None:
        """Find first file matching pattern in artifacts directory."""
        return next(self.artifacts_dir.glob(pattern), None)

    def _load_text_file(self, pattern: str) -> str:
        """Load text file content or return empty string."""
        file_path = self._find_file(pattern)
        return file_path.read_text() if file_path else ""

    def _load_json_export(self) -> JsonExportData | None:
        """Load JSON export as Pydantic model."""
        file_path = self._find_file("**/*aiperf.json")
        if not file_path:
            return None
        return JsonExportData.model_validate_json(file_path.read_text())

    def _load_inputs(self) -> InputsFile | None:
        """Load inputs file as Pydantic model."""
        file_path = self._find_file("**/inputs.json")
        return (
            InputsFile.model_validate_json(file_path.read_text()) if file_path else None
        )

    def _load_jsonl_records(self) -> list[MetricRecordInfo] | None:
        """Load JSONL records as Pydantic models."""
        file_path = self._find_file("**/*profile_export.jsonl")
        if not file_path:
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(MetricRecordInfo.model_validate_json(line))
        return records

    def _load_raw_records(self) -> list[RawRecordInfo] | None:
        """Load raw records as Pydantic models."""
        file_path = self._find_file("**/*profile_export_raw.jsonl")
        if not file_path:
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(RawRecordInfo.model_validate_json(line))
        return records

    @property
    def has_all_outputs(self) -> bool:
        """Check if all outputs exist."""
        return all(
            (
                self.json is not None,
                bool(self.csv),
                self.inputs is not None,
                self.jsonl is not None,
            )
        )

    def validate_pydantic_models(self) -> None:
        """Validate that all Pydantic models are properly loaded."""
        if self.json:
            assert isinstance(self.json, JsonExportData), (
                "json should be JsonExportData"
            )

        if self.inputs:
            assert isinstance(self.inputs, InputsFile), "inputs should be InputsFile"
            if self.inputs.data:
                assert all(isinstance(s, SessionPayloads) for s in self.inputs.data), (
                    "All inputs.data entries should be SessionPayloads"
                )

        if self.jsonl:
            assert all(isinstance(r, MetricRecordInfo) for r in self.jsonl), (
                "All jsonl records should be MetricRecordInfo"
            )

        if self.raw_records:
            assert all(isinstance(r, RawRecordInfo) for r in self.raw_records), (
                "All raw records should be RawRecordInfo"
            )

    @property
    def request_count(self) -> int:
        """Get number of completed requests from JsonExportData Pydantic model."""
        if not self.json or not self.json.request_count:
            return 0
        return int(self.json.request_count.avg)

    @property
    def has_streaming_metrics(self) -> bool:
        """Check if streaming metrics exist."""
        return self._has_all_metrics(
            (
                "time_to_first_token",
                "inter_token_latency",
                "inter_chunk_latency",
                "time_to_second_token",
            )
        )

    @property
    def has_non_streaming_metrics(self) -> bool:
        """Check if non-streaming metrics exist."""
        return self._has_all_metrics(
            (
                "request_latency",
                "request_throughput",
                "output_token_throughput",
                "output_token_throughput_per_user",
                "output_sequence_length",
                "input_sequence_length",
            )
        )

    def _has_all_metrics(self, metrics: tuple[str, ...]) -> bool:
        """Check if all specified metrics exist in the JsonExportData Pydantic model."""
        return bool(self.json) and all(
            getattr(self.json, metric, None) is not None for metric in metrics
        )

    def _has_input_media(self, media_attr: str) -> bool:
        """Check if inputs contain media of the specified type."""
        if not (self.inputs and self.inputs.data):
            return False

        media_type_map = {
            "images": "image_url",
            "audios": "input_audio",
            "videos": "video_url",
        }
        content_type = media_type_map.get(media_attr, media_attr)

        for session in self.inputs.data:
            if not session.payloads:
                continue

            for payload in session.payloads:
                if self._has_openai_media(payload, content_type):
                    return True
                if self._has_top_level_media(payload, media_attr):
                    return True

        return False

    def _has_openai_media(self, payload: dict[str, Any], content_type: str) -> bool:
        """Check for media in OpenAI message format."""
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get(content_type):
                        return True
        return False

    def _has_top_level_media(self, payload: dict[str, Any], media_attr: str) -> bool:
        """Check for media at top level of payload."""
        media_list = payload.get(media_attr, [])
        return bool(media_list)

    @property
    def has_input_images(self) -> bool:
        """Check if inputs contain images."""
        return self._has_input_media("images")

    @property
    def has_input_audio(self) -> bool:
        """Check if inputs contain audio."""
        return self._has_input_media("audios")

    @property
    def has_input_videos(self) -> bool:
        """Check if inputs contain videos."""
        return self._has_input_media("videos")

    @property
    def has_gpu_telemetry(self) -> bool:
        """Check if GPU telemetry exists."""
        return self.json is not None and self.json.telemetry_data is not None

    def assert_valid(self) -> None:
        """Assert that the results are valid and all Pydantic models are properly loaded."""
        assert self.has_all_outputs, "Not all output files exist"
        assert self.request_count > 0, "Request count should be greater than 0"
        self.validate_pydantic_models()
