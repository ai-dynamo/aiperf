# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for the audio_transcription endpoint type."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestAudioTranscriptionEndpoint:
    """Smoke tests for the audio_transcription endpoint against the mock server."""

    def test_synthetic_audio_transcription(self, cli: AIPerfCLI) -> None:
        """Audio transcription with synthetically generated WAV audio."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model openai/whisper-large-v3 \
                --endpoint-type audio_transcription \
                --audio-batch-size 1 \
                --audio-length-mean 3.0 \
                --audio-length-stddev 0.0 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    def test_audio_transcription_with_extra_inputs(self, cli: AIPerfCLI) -> None:
        """--extra-inputs (language, temperature) flow through to multipart form fields."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model openai/whisper-large-v3 \
                --endpoint-type audio_transcription \
                --audio-batch-size 1 \
                --audio-length-mean 3.0 \
                --audio-length-stddev 0.0 \
                --extra-inputs language:en temperature:0.0 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
