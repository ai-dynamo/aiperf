# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageMaker Data Capture trace loader.

Parses JSONL files produced by Amazon SageMaker real-time endpoint data capture.
Extracts request timing, literal prompts (messages), and token counts for
trace replay. Supports both single-file and directory input (recursive globbing).

Only OpenAI-compatible chat endpoints are supported — the captured payload must
contain a ``messages`` array.

See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture-endpoint.html
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Turn
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import SageMakerDataCaptureTrace


def _parse_iso8601_to_ms(iso_str: str) -> float:
    """Convert ISO 8601 inferenceTime to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp() * 1000.0


def _decode_payload(capture_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Decode a captureData entry (endpointInput or endpointOutput).

    Handles JSON (raw) and BASE64 encoded payloads. Returns None for
    CSV or unknown encodings.
    """
    data = capture_entry.get("data")
    if data is None:
        return None
    encoding = capture_entry.get("encoding", "BASE64")
    if encoding == "JSON":
        return orjson.loads(data)
    if encoding == "BASE64":
        return orjson.loads(base64.b64decode(data))
    return None


class SageMakerDataCaptureLoader(
    BaseTraceDatasetLoader[SageMakerDataCaptureTrace],
):
    """Loader for SageMaker Data Capture JSONL files.

    Accepts a single ``.jsonl`` file or a directory of capture files
    (recursive ``**/*.jsonl`` globbing). Records are sorted by timestamp
    after loading all files.
    """

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Detect SageMaker Data Capture format.

        Content-based: checks for ``captureData`` and ``eventMetadata`` keys.
        Path-based: peeks at the first line of a file or the first ``.jsonl``
        file in a directory.
        """
        if data is not None:
            return (
                isinstance(data, dict)
                and "captureData" in data
                and "eventMetadata" in data
            )
        if filename is not None:
            path = Path(filename)
            if path.is_dir():
                target = next(path.rglob("*.jsonl"), None)
                if target is None:
                    return False
            elif path.is_file():
                target = path
            else:
                return False
            try:
                with open(target) as f:
                    for line in f:
                        if not (line := line.strip()):
                            continue
                        record = orjson.loads(line)
                        return "captureData" in record and "eventMetadata" in record
            except (OSError, orjson.JSONDecodeError, UnicodeDecodeError):
                return False
        return False

    # ------------------------------------------------------------------
    # Template-method hooks
    # ------------------------------------------------------------------

    def _parse_trace(self, line: str) -> SageMakerDataCaptureTrace:
        """Parse a single JSONL line from a SageMaker Data Capture file."""
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as e:
            raise DatasetLoaderError(f"Invalid JSON in capture record: {e}") from e

        event_id = record.get("eventMetadata", {}).get("eventId", "<unknown>")

        try:
            timestamp_ms = _parse_iso8601_to_ms(
                record["eventMetadata"]["inferenceTime"]
            )
        except KeyError as e:
            raise DatasetLoaderError(
                f"Capture record {event_id} missing required field: {e}"
            ) from e

        try:
            input_data = _decode_payload(record["captureData"]["endpointInput"])
        except KeyError as e:
            raise DatasetLoaderError(
                f"Capture record {event_id} missing captureData.endpointInput: {e}"
            ) from e

        if not isinstance(input_data, dict) or "messages" not in input_data:
            raise DatasetLoaderError(
                f"Capture record {event_id} has no 'messages' key in payload. "
                "Only OpenAI-compatible chat endpoints are supported."
            )

        output_data = _decode_payload(record["captureData"].get("endpointOutput", {}))
        usage = output_data.get("usage", {}) if isinstance(output_data, dict) else {}

        max_tokens = input_data.get("max_tokens")
        if max_tokens is None:
            max_tokens = input_data.get("max_completion_tokens")

        try:
            return SageMakerDataCaptureTrace(
                timestamp=timestamp_ms,
                input_length=usage.get("prompt_tokens"),
                output_length=max_tokens,
                messages=input_data["messages"],
                tools=input_data.get("tools"),
                event_id=record["eventMetadata"].get("eventId"),
            )
        except (TypeError, ValueError) as e:
            raise DatasetLoaderError(f"Capture record {event_id} malformed: {e}") from e

    def _preprocess_trace(self, trace: SageMakerDataCaptureTrace) -> None:
        pass

    def _group_traces(
        self, items: list[SageMakerDataCaptureTrace]
    ) -> dict[str, list[SageMakerDataCaptureTrace]]:
        """Each captured record is an independent request."""
        return {self.session_id_generator.next(): [trace] for trace in items}

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _get_text_input(self, trace: SageMakerDataCaptureTrace) -> str | None:
        """Return empty string to signal literal-messages mode.

        The empty-string sentinel causes the base class to skip hash-id
        decoding and pass the value to ``_build_turn``, which ignores
        the prompt arg and uses ``trace.messages`` via ``Turn.raw_messages``.
        """
        return ""

    def _build_turn(self, trace: SageMakerDataCaptureTrace, prompt: str) -> Turn:
        """Build a Turn with raw_messages for literal replay.

        The ``prompt`` arg is unused — it is the empty-string sentinel
        from ``_get_text_input``. The captured messages array is set
        directly on ``Turn.raw_messages``, bypassing endpoint message
        construction.
        """
        del prompt
        return Turn(
            timestamp=trace.timestamp,
            max_tokens=trace.output_length,
            raw_messages=trace.messages,
            raw_tools=trace.tools,
        )

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset({"event_id", "messages", "tools"})

    def _reconstruct_traces(
        self,
        originals: list[SageMakerDataCaptureTrace],
        synth_dicts: list[dict[str, Any]],
    ) -> list[SageMakerDataCaptureTrace]:
        result: list[SageMakerDataCaptureTrace] = []
        for i, synth_dict in enumerate(synth_dicts):
            original = originals[i] if i < len(originals) else originals[-1]
            result.append(
                SageMakerDataCaptureTrace(
                    timestamp=synth_dict.get("timestamp", original.timestamp),
                    input_length=synth_dict.get("input_length", original.input_length),
                    output_length=synth_dict.get(
                        "output_length", original.output_length
                    ),
                    messages=original.messages,
                    tools=original.tools,
                    event_id=original.event_id,
                )
            )
        return result

    def _resolve_files(self) -> list[Path]:
        """Resolve input path to a list of JSONL files."""
        path = self.filename
        if path.is_dir():
            files = sorted(path.rglob("*.jsonl"))
            if not files:
                raise DatasetLoaderError(f"No .jsonl files found in directory '{path}'")
            self.info(f"Found {len(files)} capture files in {path}")
            return files
        return [path]

    def _read_all_traces(self, files: list[Path]) -> list[SageMakerDataCaptureTrace]:
        """Parse all records from the given files."""
        items: list[SageMakerDataCaptureTrace] = []
        for file in files:
            with open(file) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    trace = self._parse_trace(line)
                    self._preprocess_trace(trace)
                    items.append(trace)
        return items

    # ------------------------------------------------------------------
    # load_dataset — override for directory support
    # ------------------------------------------------------------------

    def load_dataset(
        self,
    ) -> dict[str, list[SageMakerDataCaptureTrace]]:
        """Load SageMaker Data Capture traces from a file or directory.

        When given a directory, recursively globs ``**/*.jsonl`` files.
        Timestamps are zero-aligned (earliest becomes 0) so that
        ``--fixed-schedule-end-offset`` works with relative offsets.
        """
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0

        files = self._resolve_files()
        raw_items = self._read_all_traces(files)

        if not raw_items:
            self._log_filtering_summary()
            return self._group_traces([])

        # Zero-align timestamps so offset filters work with relative values
        min_ts = min(t.timestamp for t in raw_items)
        for trace in raw_items:
            trace.timestamp -= min_ts

        # Apply offset/ISL/OSL filters on zero-aligned timestamps
        items = [t for t in raw_items if self._filter_and_cap_trace(t)]
        items.sort(key=lambda t: t.timestamp)

        self._log_filtering_summary()
        self.info(f"Loaded {len(items):,} traces from {len(files)} file(s)")

        data = self._group_traces(items)

        if self.user_config.input.synthesis.should_synthesize():
            data = self._apply_synthesis(data)

        return data
