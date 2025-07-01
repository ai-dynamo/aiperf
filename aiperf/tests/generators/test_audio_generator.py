# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import random

import numpy as np
import pytest
import soundfile as sf

from aiperf.common.config import AudioConfig, AudioLengthConfig
from aiperf.common.enums import AudioFormat
from aiperf.services.dataset.generator.audio import (
    AudioGenerator,
    GeneratorConfigurationError,
)


def decode_audio(data_uri: str) -> tuple[np.ndarray, int]:
    """Helper function to decode audio from data URI format.

    Args:
        data_uri: Data URI string in format "format,b64_data"

    Returns:
        Tuple of (audio_data: np.ndarray, sample_rate: int)
    """
    # Parse data URI
    _, b64_data = data_uri.split(",")
    decoded_data = base64.b64decode(b64_data)

    # Load audio using soundfile - format is auto-detected from content
    audio_data, sample_rate = sf.read(io.BytesIO(decoded_data))
    return audio_data, sample_rate


@pytest.fixture
def base_config():
    return AudioConfig(
        length=AudioLengthConfig(
            mean=3.0,
            stddev=0.4,
        ),
        sample_rates=[44.1],
        depths=[16],
        format=AudioFormat.WAV,
        num_channels=1,
    )


@pytest.mark.parametrize(
    "expected_audio_length",
    [
        1.0,
        2.0,
    ],
)
def test_different_audio_length(expected_audio_length, base_config):
    base_config.length.mean = expected_audio_length
    base_config.length.stddev = 0.0  # make it deterministic

    audio_generator = AudioGenerator(base_config)
    data_uri = audio_generator.generate()

    audio_data, sample_rate = decode_audio(data_uri)
    actual_length = len(audio_data) / sample_rate
    assert abs(actual_length - expected_audio_length) < 0.1, (
        "audio length not as expected"
    )


def test_negative_length_raises_error(base_config):
    base_config.length.mean = -1.0
    audio_generator = AudioGenerator(base_config)

    with pytest.raises(GeneratorConfigurationError):
        audio_generator.generate()


@pytest.mark.parametrize(
    "mean, stddev, sampling_rate, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
def test_generator_deterministic(mean, stddev, sampling_rate, bit_depth, base_config):
    np.random.seed(123)
    random.seed(123)

    base_config.length.mean = mean
    base_config.length.stddev = stddev
    base_config.sample_rates = [sampling_rate]
    base_config.depths = [bit_depth]

    audio_generator = AudioGenerator(base_config)
    data_uri1 = audio_generator.generate()

    np.random.seed(123)
    random.seed(123)
    data_uri2 = audio_generator.generate()

    # Compare the actual audio data
    audio_data1, _ = decode_audio(data_uri1)
    audio_data2, _ = decode_audio(data_uri2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("audio_format", [AudioFormat.WAV, AudioFormat.MP3])
def test_audio_format(audio_format, base_config):
    # use sample rate supported by all formats (44.1kHz)
    base_config.format = audio_format

    audio_generator = AudioGenerator(base_config)
    data_uri = audio_generator.generate()

    # Check data URI format
    assert data_uri.startswith(f"{audio_format.name.lower()},"), (
        "incorrect data URI format"
    )

    # Verify the audio can be decoded
    audio_data, _ = decode_audio(data_uri)
    assert len(audio_data) > 0, "audio data is empty"


def test_unsupported_bit_depth(base_config):
    base_config.depths = [12]  # Unsupported bit depth

    with pytest.raises(GeneratorConfigurationError) as exc_info:
        audio_generator = AudioGenerator(base_config)
        audio_generator.generate()

    assert "Supported bit depths are:" in str(exc_info.value)


@pytest.mark.parametrize("channels", [1, 2])
def test_channels(channels, base_config):
    base_config.num_channels = channels

    audio_generator = AudioGenerator(base_config)
    data_uri = audio_generator.generate()

    audio_data, _ = decode_audio(data_uri)
    if channels == 1:
        assert len(audio_data.shape) == 1, "mono audio should be 1D array"
    else:
        assert len(audio_data.shape) == 2, "stereo audio should be 2D array"
        assert audio_data.shape[1] == 2, "stereo audio should have 2 channels"


@pytest.mark.parametrize(
    "sampling_rate_khz, bit_depth",
    [
        (44.1, 16),  # Common CD quality
        (48, 24),  # Studio quality
        (96, 32),  # High-res audio
    ],
)
def test_audio_parameters(sampling_rate_khz, bit_depth, base_config):
    base_config.sample_rates = [sampling_rate_khz]
    base_config.depths = [bit_depth]

    audio_generator = AudioGenerator(base_config)
    data_uri = audio_generator.generate()

    _, sample_rate = decode_audio(data_uri)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


def test_mp3_unsupported_sampling_rate(base_config):
    base_config.sample_rates = [96]  # 96kHz is not supported for MP3
    base_config.format = AudioFormat.MP3

    with pytest.raises(GeneratorConfigurationError) as exc_info:
        audio_generator = AudioGenerator(base_config)
        audio_generator.generate()

        assert "MP3 format only supports" in str(exc_info.value), (
            "error message should mention supported rates"
        )
