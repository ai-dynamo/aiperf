# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/audio/speech (text-to-speech) endpoint.

Based on: docs/tutorials/tts.md
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSpeechEndpoint:
    """Tests for the OpenAI-compatible text-to-speech endpoint."""

    def test_non_streaming_speech(self, cli: AIPerfCLI):
        """Text-to-speech with synthetic inputs (non-streaming binary audio).

        Validates the audio output metrics: output audio duration (decoded
        from the returned clip), real-time factor, and audio throughput.
        Time-to-first-audio is streaming-only and must be absent here.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model tts-1 \
                --tokenizer gpt2 \
                --endpoint-type speech \
                --synthetic-input-tokens-mean 30 \
                --synthetic-input-tokens-stddev 5 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        assert result.json.output_audio_duration is not None
        assert result.json.output_audio_duration.avg > 0
        assert result.json.output_rtf is not None
        assert result.json.audio_throughput is not None
        assert result.json.audio_throughput.avg > 0

        # TTS produces audio, not tokens: token metrics must be absent.
        assert result.json.output_token_throughput is None
        # Non-streaming: no time-to-first-audio.
        assert result.json.time_to_first_audio is None

    def test_streaming_speech_time_to_first_audio(self, cli: AIPerfCLI):
        """Streaming text-to-speech yields time-to-first-audio (TTFA)."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model tts-1 \
                --tokenizer gpt2 \
                --endpoint-type speech \
                --streaming \
                --synthetic-input-tokens-mean 30 \
                --synthetic-input-tokens-stddev 5 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        assert result.json.time_to_first_audio is not None
        assert result.json.time_to_first_audio.avg > 0
        assert result.json.output_audio_duration is not None
        assert result.json.output_audio_duration.avg > 0
        assert result.json.audio_throughput is not None

    def test_speech_with_extra_inputs(self, cli: AIPerfCLI):
        """Voice and response_format pass through via --extra-inputs."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model tts-1 \
                --tokenizer gpt2 \
                --endpoint-type speech \
                --extra-inputs voice:echo response_format:wav \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.output_audio_duration is not None
