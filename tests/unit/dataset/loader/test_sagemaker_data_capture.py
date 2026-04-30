# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SageMaker Data Capture trace loader."""

import base64
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader.sagemaker_data_capture import (
    SageMakerDataCaptureLoader,
    _decode_payload,
    _parse_iso8601_to_ms,
)


def _make_capture_record(
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int | None = 50,
    prompt_tokens: int = 28,
    completion_tokens: int = 15,
    inference_time: str = "2026-04-29T00:03:18Z",
    event_id: str = "e4378ff2-2b43-4031-a21f-401bb3c3e038",
    input_encoding: str = "JSON",
    output_encoding: str = "JSON",
) -> str:
    """Build a SageMaker Data Capture JSONL line for testing."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]

    input_payload: dict[str, Any] = {"messages": messages}
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

    def _encode(payload: dict, encoding: str) -> tuple[str, str]:
        raw = orjson.dumps(payload).decode()
        if encoding == "JSON":
            return raw, "JSON"
        return base64.b64encode(orjson.dumps(payload)).decode(), "BASE64"

    input_data, input_enc = _encode(input_payload, input_encoding)
    output_data, output_enc = _encode(output_payload, output_encoding)

    record = {
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
    return orjson.dumps(record).decode()


class TestParseIso8601ToMs:
    """Tests for _parse_iso8601_to_ms helper."""

    def test_parse_utc_z_suffix(self) -> None:
        result = _parse_iso8601_to_ms("2026-04-29T00:03:18Z")
        assert isinstance(result, float)
        assert result > 0

    def test_parse_utc_offset_suffix(self) -> None:
        result = _parse_iso8601_to_ms("2026-04-29T00:03:18+00:00")
        z_result = _parse_iso8601_to_ms("2026-04-29T00:03:18Z")
        assert result == z_result

    def test_parse_preserves_ordering(self) -> None:
        t1 = _parse_iso8601_to_ms("2026-04-29T00:03:18Z")
        t2 = _parse_iso8601_to_ms("2026-04-29T00:03:19Z")
        assert t2 > t1
        assert t2 - t1 == pytest.approx(1000.0, abs=1)


class TestDecodePayload:
    """Tests for _decode_payload helper."""

    def test_decode_json_encoding(self) -> None:
        payload = {"messages": [{"role": "user", "content": "test"}]}
        entry = {
            "data": orjson.dumps(payload).decode(),
            "encoding": "JSON",
        }
        result = _decode_payload(entry)
        assert result == payload

    def test_decode_base64_encoding(self) -> None:
        payload = {"messages": [{"role": "user", "content": "test"}]}
        entry = {
            "data": base64.b64encode(orjson.dumps(payload)).decode(),
            "encoding": "BASE64",
        }
        result = _decode_payload(entry)
        assert result == payload

    def test_decode_csv_encoding_returns_none(self) -> None:
        entry = {"data": "1,2,3", "encoding": "CSV"}
        assert _decode_payload(entry) is None

    def test_decode_missing_data_returns_none(self) -> None:
        entry = {"encoding": "JSON"}
        assert _decode_payload(entry) is None

    def test_decode_defaults_to_base64_when_encoding_missing(self) -> None:
        payload = {"messages": [{"role": "user", "content": "test"}]}
        entry = {"data": base64.b64encode(orjson.dumps(payload)).decode()}
        result = _decode_payload(entry)
        assert result == payload


class TestCanLoad:
    """Tests for SageMakerDataCaptureLoader.can_load."""

    def test_can_load_with_valid_data(self) -> None:
        data = {
            "captureData": {"endpointInput": {}, "endpointOutput": {}},
            "eventMetadata": {
                "eventId": "test",
                "inferenceTime": "2026-01-01T00:00:00Z",
            },
        }
        assert SageMakerDataCaptureLoader.can_load(data=data) is True

    def test_can_load_rejects_non_capture_data(self) -> None:
        data = {"messages": [{"role": "user", "content": "test"}]}
        assert SageMakerDataCaptureLoader.can_load(data=data) is False

    def test_can_load_rejects_none(self) -> None:
        assert SageMakerDataCaptureLoader.can_load() is False

    def test_can_load_with_directory(self, tmp_path: Path) -> None:
        subdir = tmp_path / "2026" / "04" / "29" / "00"
        subdir.mkdir(parents=True)
        capture_file = subdir / "test.jsonl"
        capture_file.write_text(_make_capture_record() + "\n")
        assert SageMakerDataCaptureLoader.can_load(filename=tmp_path) is True

    def test_can_load_rejects_empty_directory(self, tmp_path: Path) -> None:
        assert SageMakerDataCaptureLoader.can_load(filename=tmp_path) is False

    def test_can_load_rejects_directory_with_non_capture_jsonl(
        self, tmp_path: Path
    ) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"input_length": 100, "output_length": 50}\n')
        assert SageMakerDataCaptureLoader.can_load(filename=tmp_path) is False


class TestParseTrace:
    """Tests for _parse_trace method."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        prompt_gen = MagicMock()
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=prompt_gen,
        )

    def test_parse_trace_extracts_messages(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]
        line = _make_capture_record(messages=messages)
        trace = loader._parse_trace(line)
        assert trace.messages == messages

    def test_parse_trace_extracts_timestamp(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(inference_time="2026-04-29T00:03:18Z")
        trace = loader._parse_trace(line)
        assert trace.timestamp == _parse_iso8601_to_ms("2026-04-29T00:03:18Z")

    def test_parse_trace_extracts_max_tokens_from_input(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(max_tokens=100, completion_tokens=15)
        trace = loader._parse_trace(line)
        assert trace.output_length == 100

    def test_parse_trace_output_length_is_none_when_no_max_tokens(
        self, tmp_path: Path
    ) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(max_tokens=None, completion_tokens=42)
        trace = loader._parse_trace(line)
        assert trace.output_length is None

    def test_parse_trace_extracts_prompt_tokens(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(prompt_tokens=128)
        trace = loader._parse_trace(line)
        assert trace.input_length == 128

    def test_parse_trace_extracts_event_id(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(event_id="abc-123")
        trace = loader._parse_trace(line)
        assert trace.event_id == "abc-123"

    def test_parse_trace_handles_base64_encoding(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record(input_encoding="BASE64", output_encoding="BASE64")
        trace = loader._parse_trace(line)
        assert trace.messages == [{"role": "user", "content": "Hello"}]

    def test_parse_trace_rejects_missing_messages(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        record = orjson.loads(_make_capture_record())
        record["captureData"]["endpointInput"]["data"] = orjson.dumps(
            {"inputs": "plain text prompt"}
        ).decode()
        line = orjson.dumps(record).decode()
        with pytest.raises(DatasetLoaderError, match="no 'messages' key"):
            loader._parse_trace(line)


class TestLoadDataset:
    """Tests for load_dataset with file and directory input."""

    def _make_loader(
        self, path: Path, user_config: Any = None
    ) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = user_config or UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"])
        )
        prompt_gen = MagicMock()
        return SageMakerDataCaptureLoader(
            filename=path,
            user_config=config,
            prompt_generator=prompt_gen,
        )

    def test_load_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "capture.jsonl"
        f.write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:01Z")
            + "\n"
            + _make_capture_record(inference_time="2026-04-29T00:00:02Z")
            + "\n"
        )
        loader = self._make_loader(f)
        data = loader.load_dataset()
        assert len(data) == 2

    def test_load_directory_globs_recursively(self, tmp_path: Path) -> None:
        hour1 = tmp_path / "2026" / "04" / "29" / "00"
        hour2 = tmp_path / "2026" / "04" / "29" / "01"
        hour1.mkdir(parents=True)
        hour2.mkdir(parents=True)
        (hour1 / "a.jsonl").write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:01Z") + "\n"
        )
        (hour2 / "b.jsonl").write_text(
            _make_capture_record(inference_time="2026-04-29T01:00:01Z") + "\n"
        )
        loader = self._make_loader(tmp_path)
        data = loader.load_dataset()
        assert len(data) == 2

    def test_load_directory_sorts_by_timestamp(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "hour_a"
        dir_b = tmp_path / "hour_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "a.jsonl").write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:03Z") + "\n"
        )
        (dir_b / "b.jsonl").write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:01Z") + "\n"
        )
        (dir_a / "c.jsonl").write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:02Z") + "\n"
        )
        loader = self._make_loader(tmp_path)
        data = loader.load_dataset()
        timestamps = [traces[0].timestamp for traces in data.values()]
        assert timestamps == sorted(timestamps)

    def test_load_single_file_sorts_by_timestamp(self, tmp_path: Path) -> None:
        f = tmp_path / "capture.jsonl"
        f.write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:03Z")
            + "\n"
            + _make_capture_record(inference_time="2026-04-29T00:00:01Z")
            + "\n"
            + _make_capture_record(inference_time="2026-04-29T00:00:02Z")
            + "\n"
        )
        loader = self._make_loader(f)
        data = loader.load_dataset()
        timestamps = [traces[0].timestamp for traces in data.values()]
        assert timestamps == sorted(timestamps)

    def test_load_empty_directory_raises_error(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        with pytest.raises(DatasetLoaderError, match=r"No \.jsonl files"):
            loader.load_dataset()

    def test_load_skips_empty_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "capture.jsonl"
        f.write_text(
            "\n" + _make_capture_record() + "\n" + "\n" + _make_capture_record() + "\n"
        )
        loader = self._make_loader(f)
        data = loader.load_dataset()
        assert len(data) == 2


class TestBuildTurn:
    """Tests for _build_turn producing Turn with raw_messages."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        prompt_gen = MagicMock()
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=prompt_gen,
        )

    def test_build_turn_sets_raw_messages(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        messages = [{"role": "user", "content": "test"}]
        trace = SageMakerDataCaptureTrace(
            timestamp=1000.0,
            messages=messages,
            output_length=50,
        )
        turn = loader._build_turn(trace, "")
        assert turn.raw_messages == messages
        assert turn.max_tokens == 50
        assert turn.timestamp == 1000.0

    def test_build_turn_preserves_system_message(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        trace = SageMakerDataCaptureTrace(
            timestamp=1000.0,
            messages=messages,
            output_length=30,
        )
        turn = loader._build_turn(trace, "")
        assert turn.raw_messages == messages
        assert len(turn.raw_messages) == 2
        assert turn.raw_messages[0]["role"] == "system"
        assert turn.raw_messages[1]["role"] == "user"

    def test_build_turn_without_output_length(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        trace = SageMakerDataCaptureTrace(
            timestamp=2000.0,
            messages=[{"role": "user", "content": "hi"}],
        )
        turn = loader._build_turn(trace, "")
        assert turn.max_tokens is None

    def test_get_text_input_always_returns_empty_string(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        trace = SageMakerDataCaptureTrace(
            timestamp=1000.0,
            messages=[{"role": "user", "content": "test"}],
        )
        assert loader._get_text_input(trace) == ""


class TestGroupTraces:
    """Tests for _group_traces assigning unique session IDs."""

    def test_group_traces_assigns_unique_session_ids(self, tmp_path: Path) -> None:
        from aiperf.common.config import EndpointConfig, UserConfig
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        loader = SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=MagicMock(),
        )
        traces = [
            SageMakerDataCaptureTrace(
                timestamp=1000.0, messages=[{"role": "user", "content": "a"}]
            ),
            SageMakerDataCaptureTrace(
                timestamp=2000.0, messages=[{"role": "user", "content": "b"}]
            ),
        ]
        grouped = loader._group_traces(traces)
        assert len(grouped) == 2
        session_ids = list(grouped.keys())
        assert session_ids[0] != session_ids[1]
        for traces_list in grouped.values():
            assert len(traces_list) == 1


class TestParseTraceEdgeCases:
    """Tests for _parse_trace edge cases."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=MagicMock(),
        )

    def test_parse_trace_missing_endpoint_output_sets_none_lengths(
        self, tmp_path: Path
    ) -> None:
        """When endpointOutput is missing (streaming), output_length and input_length are None."""
        loader = self._make_loader(tmp_path)
        record = {
            "captureData": {
                "endpointInput": {
                    "data": orjson.dumps(
                        {"messages": [{"role": "user", "content": "hi"}]}
                    ).decode(),
                    "encoding": "JSON",
                },
            },
            "eventMetadata": {
                "eventId": "test-id",
                "inferenceTime": "2026-04-29T00:00:00Z",
            },
            "eventVersion": "0",
        }
        line = orjson.dumps(record).decode()
        trace = loader._parse_trace(line)
        assert trace.output_length is None
        assert trace.input_length is None
        assert trace.messages == [{"role": "user", "content": "hi"}]

    def test_parse_trace_uses_max_completion_tokens_field(self, tmp_path: Path) -> None:
        """Newer OpenAI API uses max_completion_tokens instead of max_tokens."""
        loader = self._make_loader(tmp_path)
        input_payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 200,
        }
        output_payload = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
        }
        record = {
            "captureData": {
                "endpointInput": {
                    "data": orjson.dumps(input_payload).decode(),
                    "encoding": "JSON",
                },
                "endpointOutput": {
                    "data": orjson.dumps(output_payload).decode(),
                    "encoding": "JSON",
                },
            },
            "eventMetadata": {
                "eventId": "test",
                "inferenceTime": "2026-04-29T00:00:00Z",
            },
            "eventVersion": "0",
        }
        line = orjson.dumps(record).decode()
        trace = loader._parse_trace(line)
        assert trace.output_length == 200

    def test_parse_trace_csv_output_encoding_gives_none_usage(
        self, tmp_path: Path
    ) -> None:
        """CSV-encoded output (non-LLM endpoint) results in None token counts."""
        loader = self._make_loader(tmp_path)
        record = {
            "captureData": {
                "endpointInput": {
                    "data": orjson.dumps(
                        {"messages": [{"role": "user", "content": "hi"}]}
                    ).decode(),
                    "encoding": "JSON",
                },
                "endpointOutput": {
                    "data": "0.95",
                    "encoding": "CSV",
                },
            },
            "eventMetadata": {
                "eventId": "test",
                "inferenceTime": "2026-04-29T00:00:00Z",
            },
            "eventVersion": "0",
        }
        line = orjson.dumps(record).decode()
        trace = loader._parse_trace(line)
        assert trace.input_length is None
        assert trace.output_length is None

    def test_can_load_returns_true_for_single_file_path(self, tmp_path: Path) -> None:
        """can_load with filename pointing to a capture file returns True."""
        f = tmp_path / "capture.jsonl"
        f.write_text(_make_capture_record() + "\n")
        assert SageMakerDataCaptureLoader.can_load(filename=f) is True

    def test_can_load_returns_false_for_non_capture_file(self, tmp_path: Path) -> None:
        """can_load with a non-capture JSONL file returns False."""
        f = tmp_path / "other.jsonl"
        f.write_text('{"input_length": 100}\n')
        assert SageMakerDataCaptureLoader.can_load(filename=f) is False


class TestSynthesisHooks:
    """Tests for synthesis-related hooks."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=MagicMock(),
        )

    def test_reconstruct_traces_preserves_messages_from_originals(
        self, tmp_path: Path
    ) -> None:
        """Synthesis modifies timestamps/lengths but messages must come from originals."""
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        loader = self._make_loader(tmp_path)
        originals = [
            SageMakerDataCaptureTrace(
                timestamp=1000.0,
                input_length=50,
                output_length=20,
                messages=[{"role": "user", "content": "original prompt"}],
                event_id="evt-1",
            ),
        ]
        synth_dicts = [
            {"timestamp": 500.0, "input_length": 50, "output_length": 20},
        ]
        result = loader._reconstruct_traces(originals, synth_dicts)
        assert len(result) == 1
        assert result[0].timestamp == 500.0
        assert result[0].messages == [{"role": "user", "content": "original prompt"}]
        assert result[0].event_id == "evt-1"

    def test_reconstruct_traces_uses_synth_lengths(self, tmp_path: Path) -> None:
        """Synthesis may modify input_length and output_length."""
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        loader = self._make_loader(tmp_path)
        originals = [
            SageMakerDataCaptureTrace(
                timestamp=1000.0,
                input_length=100,
                output_length=50,
                messages=[{"role": "user", "content": "test"}],
            ),
        ]
        synth_dicts = [
            {"timestamp": 1000.0, "input_length": 200, "output_length": 100},
        ]
        result = loader._reconstruct_traces(originals, synth_dicts)
        assert result[0].input_length == 200
        assert result[0].output_length == 100

    def test_synthesis_exclude_fields_includes_messages_and_event_id(
        self, tmp_path: Path
    ) -> None:
        loader = self._make_loader(tmp_path)
        excluded = loader._synthesis_exclude_fields()
        assert "event_id" in excluded
        assert "messages" in excluded


class TestCanLoadEdgeCases:
    """Tests for can_load edge cases to improve branch coverage."""

    def test_can_load_returns_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        assert SageMakerDataCaptureLoader.can_load(filename=nonexistent) is False

    def test_can_load_returns_false_for_corrupt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupt.jsonl"
        f.write_bytes(b"\x80\x81\x82")
        assert SageMakerDataCaptureLoader.can_load(filename=f) is False

    def test_can_load_skips_empty_lines_in_file(self, tmp_path: Path) -> None:
        f = tmp_path / "capture.jsonl"
        f.write_text("\n\n" + _make_capture_record() + "\n")
        assert SageMakerDataCaptureLoader.can_load(filename=f) is True


class TestTimestampZeroAlignment:
    """Tests for zero-alignment of absolute epoch timestamps."""

    def _make_loader(
        self, path: Path, user_config: Any = None
    ) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = user_config or UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"])
        )
        return SageMakerDataCaptureLoader(
            filename=path,
            user_config=config,
            prompt_generator=MagicMock(),
        )

    def test_timestamps_are_zero_aligned(self, tmp_path: Path) -> None:
        f = tmp_path / "capture.jsonl"
        f.write_text(
            _make_capture_record(inference_time="2026-04-29T00:00:10Z")
            + "\n"
            + _make_capture_record(inference_time="2026-04-29T00:00:12Z")
            + "\n"
            + _make_capture_record(inference_time="2026-04-29T00:00:14Z")
            + "\n"
        )
        loader = self._make_loader(f)
        data = loader.load_dataset()
        timestamps = [traces[0].timestamp for traces in data.values()]
        assert timestamps[0] == 0.0
        assert timestamps[1] == pytest.approx(2000.0, abs=1)
        assert timestamps[2] == pytest.approx(4000.0, abs=1)


class TestToolsSupport:
    """Tests for tools field extraction and replay."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=MagicMock(),
        )

    def test_parse_trace_extracts_tools(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        input_payload = {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "max_tokens": 50,
            "tools": tools,
        }
        output_payload = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }
        record = {
            "captureData": {
                "endpointInput": {
                    "data": orjson.dumps(input_payload).decode(),
                    "encoding": "JSON",
                },
                "endpointOutput": {
                    "data": orjson.dumps(output_payload).decode(),
                    "encoding": "JSON",
                },
            },
            "eventMetadata": {
                "eventId": "tools-test",
                "inferenceTime": "2026-04-29T00:00:00Z",
            },
            "eventVersion": "0",
        }
        line = orjson.dumps(record).decode()
        trace = loader._parse_trace(line)
        assert trace.tools == tools

    def test_parse_trace_tools_none_when_absent(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        line = _make_capture_record()
        trace = loader._parse_trace(line)
        assert trace.tools is None

    def test_build_turn_passes_raw_tools(self, tmp_path: Path) -> None:
        from aiperf.dataset.loader.models import SageMakerDataCaptureTrace

        loader = self._make_loader(tmp_path)
        tools = [{"type": "function", "function": {"name": "test"}}]
        trace = SageMakerDataCaptureTrace(
            timestamp=0.0,
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            output_length=50,
        )
        turn = loader._build_turn(trace, "")
        assert turn.raw_tools == tools
        assert turn.raw_messages == trace.messages

    def test_synthesis_excludes_tools(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        assert "tools" in loader._synthesis_exclude_fields()


class TestParseTraceErrorPaths:
    """Tests for _parse_trace error handling with DatasetLoaderError."""

    def _make_loader(self, tmp_path: Path) -> SageMakerDataCaptureLoader:
        from aiperf.common.config import EndpointConfig, UserConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        return SageMakerDataCaptureLoader(
            filename=str(tmp_path / "test.jsonl"),
            user_config=config,
            prompt_generator=MagicMock(),
        )

    def test_parse_trace_invalid_json_raises_dataset_loader_error(
        self, tmp_path: Path
    ) -> None:
        loader = self._make_loader(tmp_path)
        with pytest.raises(DatasetLoaderError, match="Invalid JSON"):
            loader._parse_trace("not valid json {{{")

    def test_parse_trace_missing_event_metadata_raises_dataset_loader_error(
        self, tmp_path: Path
    ) -> None:
        loader = self._make_loader(tmp_path)
        record = orjson.dumps({"captureData": {}}).decode()
        with pytest.raises(DatasetLoaderError, match="missing required field"):
            loader._parse_trace(record)

    def test_parse_trace_missing_capture_data_raises_dataset_loader_error(
        self, tmp_path: Path
    ) -> None:
        loader = self._make_loader(tmp_path)
        record = orjson.dumps(
            {
                "eventMetadata": {
                    "eventId": "test",
                    "inferenceTime": "2026-04-29T00:00:00Z",
                }
            }
        ).decode()
        with pytest.raises(DatasetLoaderError, match="captureData.endpointInput"):
            loader._parse_trace(record)

    def test_load_dataset_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        loader = self._make_loader(tmp_path)
        loader.filename = f
        data = loader.load_dataset()
        assert len(data) == 0


class TestCountDatasetEntriesDirectory:
    """Test that _count_dataset_entries handles directories for SageMaker captures."""

    def test_count_entries_in_directory(self, tmp_path: Path) -> None:
        from aiperf.common.config import EndpointConfig, UserConfig

        hour_00 = tmp_path / "2026" / "04" / "29" / "00"
        hour_01 = tmp_path / "2026" / "04" / "29" / "01"
        hour_00.mkdir(parents=True)
        hour_01.mkdir(parents=True)
        (hour_00 / "a.jsonl").write_text(
            _make_capture_record() + "\n" + _make_capture_record() + "\n"
        )
        (hour_01 / "b.jsonl").write_text(_make_capture_record() + "\n")

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.file = tmp_path
        assert config._count_dataset_entries() == 3

    def test_count_entries_single_file(self, tmp_path: Path) -> None:
        from aiperf.common.config import EndpointConfig, UserConfig

        f = tmp_path / "capture.jsonl"
        f.write_text(_make_capture_record() + "\n" + _make_capture_record() + "\n")

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.file = f
        assert config._count_dataset_entries() == 2
