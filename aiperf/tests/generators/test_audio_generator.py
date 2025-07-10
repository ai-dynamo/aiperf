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
from aiperf.common.exceptions import ConfigurationError
from aiperf.services.dataset.generator.audio import (
    AudioGenerator,
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


@pytest.fixture
async def initialized_generator(base_config):
    generator = AudioGenerator(base_config)
    await generator.initialize()
    return generator


@pytest.mark.parametrize(
    "audio_length",
    [
        1.0,
        2.0,
    ],
)
async def test_different_audio_length(audio_length, initialized_generator):
    """Test that the audio length is as expected."""
    initialized_generator.config.length.mean = audio_length
    initialized_generator.config.length.stddev = 0.0  # make it deterministic

    data_uri = await initialized_generator.generate()

    audio_data, sample_rate = decode_audio(data_uri)
    actual_length = len(audio_data) / sample_rate
    assert abs(actual_length - audio_length) < 0.1, "audio length not as expected"


async def test_negative_length_raises_error(initialized_generator):
    """Test that setting a negative audio length raises an error."""
    initialized_generator.config.length.mean = -1.0

    with pytest.raises(ConfigurationError):
        await initialized_generator.generate()


@pytest.mark.parametrize(
    "mean, stddev, sampling_rate, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
async def test_generator_deterministic(
    mean, stddev, sampling_rate, bit_depth, initialized_generator
):
    """Test that setting random seed makes the generator deterministic."""
    initialized_generator.config = AudioConfig(
        length=AudioLengthConfig(mean=mean, stddev=stddev),
        sample_rates=[sampling_rate],
        depths=[bit_depth],
        format=AudioFormat.WAV,
        num_channels=1,
    )

    np.random.seed(123)
    random.seed(123)
    data_uri1 = await initialized_generator.generate()

    np.random.seed(123)
    random.seed(123)
    data_uri2 = await initialized_generator.generate()

    # Compare the actual audio data
    audio_data1, _ = decode_audio(data_uri1)
    audio_data2, _ = decode_audio(data_uri2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("audio_format", [AudioFormat.WAV, AudioFormat.MP3])
async def test_audio_format(audio_format, initialized_generator):
    """Test setting different audio format works as expected."""
    # use sample rate supported by all formats (44.1kHz)
    initialized_generator.config.format = audio_format

    data_uri = await initialized_generator.generate()

    # Check data URI format
    assert data_uri.startswith(f"{audio_format.name.lower()},"), (
        "incorrect data URI format"
    )

    # Verify the audio can be decoded
    audio_data, _ = decode_audio(data_uri)
    assert len(audio_data) > 0, "audio data is empty"


async def test_unsupported_bit_depth(initialized_generator):
    """Test that setting an unsupported bit depth raises an error."""
    initialized_generator.config.depths = [12]  # Unsupported bit depth

    with pytest.raises(ConfigurationError) as exc_info:
        await initialized_generator.generate()

    assert "Supported bit depths are:" in str(exc_info.value)


@pytest.mark.parametrize("channels", [1, 2])
async def test_channels(channels, initialized_generator):
    """Test that setting different number of channels works as expected."""
    initialized_generator.config.num_channels = channels

    data_uri = await initialized_generator.generate()

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
async def test_audio_parameters(sampling_rate_khz, bit_depth, initialized_generator):
    """Test that setting different audio parameters works as expected."""
    initialized_generator.config.sample_rates = [sampling_rate_khz]
    initialized_generator.config.depths = [bit_depth]

    data_uri = await initialized_generator.generate()

    _, sample_rate = decode_audio(data_uri)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


async def test_mp3_unsupported_sampling_rate(initialized_generator):
    """Test that setting an unsupported sampling rate for MP3 raises an error."""
    initialized_generator.config.sample_rates = [96]  # 96kHz is not supported for MP3
    initialized_generator.config.format = AudioFormat.MP3

    with pytest.raises(ConfigurationError) as exc_info:
        await initialized_generator.generate()

        assert "MP3 format only supports" in str(exc_info.value), (
            "error message should mention supported rates"
        )
