# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import wave

import pytest

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    AudioResponseData,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_audio_duration_metric import (
    OutputAudioDurationMetric,
    decode_audio_duration_seconds,
)

_SAMPLE_RATE = 16000


def _make_wav(duration_s: float, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Build a mono 16-bit WAV of the given duration."""
    num_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _audio_record(chunks: list[bytes], start_ns: int = 100) -> ParsedResponseRecord:
    request = RequestRecord(
        model_name="tts-1",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 1000,
    )
    responses = [
        ParsedResponse(perf_ns=start_ns + 50 + i, data=AudioResponseData(audio_bytes=c))
        for i, c in enumerate(chunks)
    ]
    return ParsedResponseRecord(request=request, responses=responses)


class TestDecodeAudioDurationSeconds:
    def test_decodes_wav_duration(self):
        wav = _make_wav(2.0)
        assert decode_audio_duration_seconds(wav) == pytest.approx(2.0, rel=1e-3)

    def test_undecodable_bytes_returns_none(self):
        assert decode_audio_duration_seconds(b"not audio at all") is None


class TestOutputAudioDurationMetric:
    def test_duration_from_single_clip(self):
        record = _audio_record([_make_wav(1.5)])
        result = OutputAudioDurationMetric().parse_record(record, MetricRecordDict())
        assert result == pytest.approx(1.5, rel=1e-3)

    def test_duration_from_concatenated_stream_chunks(self):
        """Streamed chunks split a single clip; the concatenation decodes."""
        wav = _make_wav(2.0)
        third = len(wav) // 3
        chunks = [wav[:third], wav[third : 2 * third], wav[2 * third :]]
        record = _audio_record(chunks)
        result = OutputAudioDurationMetric().parse_record(record, MetricRecordDict())
        assert result == pytest.approx(2.0, rel=1e-3)

    def test_no_audio_bytes_raises(self):
        record = _audio_record([b""])
        with pytest.raises(NoMetricValue, match="no audio"):
            OutputAudioDurationMetric().parse_record(record, MetricRecordDict())

    def test_undecodable_audio_raises(self):
        record = _audio_record([b"garbage-bytes-here"])
        with pytest.raises(NoMetricValue, match="decode"):
            OutputAudioDurationMetric().parse_record(record, MetricRecordDict())

    def test_metric_properties(self):
        metric = OutputAudioDurationMetric()
        assert metric.tag == "output_audio_duration"
        # All four headline TTS metrics share the DEFAULT console table.
        assert metric.console_group == MetricConsoleGroup.DEFAULT
        assert MetricFlags.PRODUCES_AUDIO_ONLY in metric.flags
