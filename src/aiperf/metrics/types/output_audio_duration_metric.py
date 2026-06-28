# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import soundfile as sf

from aiperf.common.enums import MetricConsoleGroup, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import AudioResponseData, ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


def decode_audio_duration_seconds(audio_bytes: bytes) -> float | None:
    """Decode the duration in seconds of a self-describing audio payload.

    Reads only the header via ``soundfile.info`` (no full decode), so it is
    cheap even for long clips. Works for container formats that carry their
    own sample rate and frame count (wav, flac, ogg/opus, mp3 with a
    libsndfile build that supports it). Headerless ``pcm`` cannot be decoded
    without an assumed sample rate and returns ``None``.

    Returns ``None`` on any decode failure so callers can raise
    ``NoMetricValue`` rather than aborting the whole record.
    """
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except (OSError, ValueError, RuntimeError):
        return None
    return float(info.duration)


def _concat_audio_bytes(record: ParsedResponseRecord) -> bytes:
    """Concatenate the audio bytes of every audio chunk in the record.

    For streamed responses the chunks split the clip at arbitrary byte
    boundaries, so individual chunks are not independently decodable; the
    concatenation reconstructs the original audio payload.
    """
    return b"".join(
        response.data.audio_bytes
        for response in record.content_responses
        if isinstance(response.data, AudioResponseData) and response.data.audio_bytes
    )


class OutputAudioDurationMetric(BaseRecordMetric[float]):
    """Per-request duration of the generated (output) audio in seconds.

    Decodes the audio returned by a text-to-speech endpoint to recover how
    much audio was synthesized. This is the TTS counterpart of the ASR
    ``audio_duration`` metric (which measures *input* audio), and is the
    basis for the output real-time factor and audio throughput metrics.

    Example:
        A request that returns a 7.3s clip produces
        ``output_audio_duration = 7.3``.

    Raises:
        NoMetricValue: when the record carries no audio bytes, or the audio
            cannot be decoded (e.g. a headerless ``pcm`` payload).
    """

    tag = "output_audio_duration"
    header = "Output Audio Duration"
    short_header = "Audio Dur"
    unit = MetricTimeUnit.SECONDS
    display_order = 870
    flags = MetricFlags.PRODUCES_AUDIO_ONLY
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        audio_bytes = _concat_audio_bytes(record)
        if not audio_bytes:
            raise NoMetricValue(
                "Record has no audio response bytes; output audio duration unavailable."
            )

        duration = decode_audio_duration_seconds(audio_bytes)
        if duration is None or duration <= 0:
            raise NoMetricValue(
                "Could not decode output audio duration (unsupported or headerless "
                "format such as raw pcm)."
            )
        return duration


class TotalOutputAudioDurationMetric(
    DerivedSumMetric[float, OutputAudioDurationMetric]
):
    """Total seconds of audio synthesized across the whole benchmark.

    Formula:
        Total Output Audio Duration = Sum(Output Audio Durations)
    """

    tag = "total_output_audio_duration"
    header = "Total Output Audio Duration"
    short_header = "Total Audio Dur"
    short_header_hide_unit = True
    flags = MetricFlags.PRODUCES_AUDIO_ONLY
    console_group = MetricConsoleGroup.NONE
