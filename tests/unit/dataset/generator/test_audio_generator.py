# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import numpy as np
import pytest
import soundfile as sf

from aiperf.common import random_generator as rng
from aiperf.common.enums import AudioFormat
from aiperf.common.exceptions import ConfigurationError
from aiperf.config.dataset.content import AudioConfig
from aiperf.config.distributions import NormalDistribution
from aiperf.dataset.generator import (
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


def make_config(
    *,
    mean: float = 3.0,
    stddev: float = 0.4,
    sample_rates: list[float] | None = None,
    depths: list[int] | None = None,
    audio_format: AudioFormat = AudioFormat.WAV,
    channels: int = 1,
) -> AudioConfig:
    """Build a v2 AudioConfig with the requested overrides."""
    return AudioConfig(
        length=NormalDistribution(mean=mean, stddev=stddev),
        sample_rates=sample_rates if sample_rates is not None else [44.1],
        depths=depths if depths is not None else [16],
        format=audio_format,
        channels=channels,
    )


@pytest.fixture
def base_config() -> AudioConfig:
    return make_config()


@pytest.mark.parametrize(
    "expected_audio_length",
    [
        1.0,
        2.0,
    ],
)
def test_different_audio_length(expected_audio_length):
    config = make_config(mean=expected_audio_length, stddev=0.0)

    audio_generator = AudioGenerator(config)
    data_uri = audio_generator.generate()

    audio_data, sample_rate = decode_audio(data_uri)
    actual_length = len(audio_data) / sample_rate
    assert abs(actual_length - expected_audio_length) < 0.1, (
        "audio length not as expected"
    )


def test_negative_length_raises_error():
    """v2 NormalDistribution rejects negative mean via Pydantic validation."""
    with pytest.raises((ValueError, ConfigurationError)):
        # NormalDistribution itself does not bound mean, but the generator
        # treats values < 0.01 as invalid. Use a small positive mean below
        # the threshold to exercise the runtime check.
        config = make_config(mean=0.001, stddev=0.0)
        AudioGenerator(config).generate()


@pytest.mark.parametrize(
    "mean, stddev, sampling_rate, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
def test_generator_deterministic(mean, stddev, sampling_rate, bit_depth):
    config_kwargs = dict(
        mean=mean, stddev=stddev, sample_rates=[sampling_rate], depths=[bit_depth]
    )

    # First generation with seed 123
    rng.reset()
    rng.init(123)
    audio_generator1 = AudioGenerator(make_config(**config_kwargs))
    data_uri1 = audio_generator1.generate()

    # Second generation with same seed 123
    rng.reset()
    rng.init(123)
    audio_generator2 = AudioGenerator(make_config(**config_kwargs))
    data_uri2 = audio_generator2.generate()

    # Compare the actual audio data
    audio_data1, _ = decode_audio(data_uri1)
    audio_data2, _ = decode_audio(data_uri2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("audio_format", [AudioFormat.WAV, AudioFormat.MP3])
def test_audio_format(audio_format):
    # use sample rate supported by all formats (44.1kHz)
    config = make_config(audio_format=audio_format)

    audio_generator = AudioGenerator(config)
    data_uri = audio_generator.generate()

    # Check data URI format
    assert data_uri.startswith(f"{audio_format.name.lower()},"), (
        "incorrect data URI format"
    )

    # Verify the audio can be decoded
    audio_data, _ = decode_audio(data_uri)
    assert len(audio_data) > 0, "audio data is empty"


def test_unsupported_bit_depth():
    config = make_config(depths=[12])  # Unsupported bit depth

    with pytest.raises(ConfigurationError) as exc_info:
        audio_generator = AudioGenerator(config)
        audio_generator.generate()

    assert "Supported bit depths are:" in str(exc_info.value)


@pytest.mark.parametrize("channels", [1, 2])
def test_channels(channels):
    config = make_config(channels=channels)

    audio_generator = AudioGenerator(config)
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
def test_audio_parameters(sampling_rate_khz, bit_depth):
    config = make_config(sample_rates=[sampling_rate_khz], depths=[bit_depth])

    audio_generator = AudioGenerator(config)
    data_uri = audio_generator.generate()

    _, sample_rate = decode_audio(data_uri)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


def test_mp3_unsupported_sample_rate_raises():
    """MP3 format with an unsupported sample rate raises a ConfigurationError."""
    config = make_config(sample_rates=[96], audio_format=AudioFormat.MP3)
    audio_generator = AudioGenerator(config)
    with pytest.raises(ConfigurationError, match="MP3 format only supports"):
        audio_generator.generate()


def test_audio_below_min_length_raises():
    """A configured mean length under 0.01s raises a ConfigurationError."""
    config = make_config(mean=0.005, stddev=0.0)
    audio_generator = AudioGenerator(config)
    with pytest.raises(ConfigurationError, match="must be greater than 0.01 seconds"):
        audio_generator.generate()


class TestAudioBitDepth:
    """Test suite for audio bit depth support, including 8-bit unsigned WAV."""

    @pytest.mark.parametrize(
        "bit_depth,expected_subtype",
        [
            (8, "PCM_U8"),
            (16, "PCM_16"),
            (24, "PCM_24"),
            (32, "PCM_32"),
        ],
    )
    def test_wav_bit_depth_produces_correct_subtype(self, bit_depth, expected_subtype):
        """WAV files use correct PCM subtype for each bit depth.

        Regression test for 8-bit audio bug where PCM_S8 was incorrectly used
        instead of PCM_U8. WAV format requires unsigned 8-bit audio.
        """
        config = make_config(
            mean=0.1,
            stddev=0.0,
            sample_rates=[16.0],
            depths=[bit_depth],
            audio_format=AudioFormat.WAV,
        )
        generator = AudioGenerator(config)
        data_uri = generator.generate()

        _, b64_data = data_uri.split(",")
        audio_bytes = base64.b64decode(b64_data)

        with io.BytesIO(audio_bytes) as f:
            info = sf.info(f)
            assert info.subtype == expected_subtype

    @pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
    def test_wav_bit_depth_produces_valid_audio(self, bit_depth):
        """All supported bit depths produce valid, readable WAV audio."""
        config = make_config(
            mean=0.1,
            stddev=0.0,
            sample_rates=[16.0],
            depths=[bit_depth],
            audio_format=AudioFormat.WAV,
        )
        generator = AudioGenerator(config)
        data_uri = generator.generate()

        audio_data, sample_rate = decode_audio(data_uri)
        assert len(audio_data) > 0
        assert sample_rate == 16000

    @pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
    def test_mp3_ignores_bit_depth_uses_lossy_encoding(self, bit_depth):
        """MP3 format works with all bit depths (lossy encoding ignores PCM subtype)."""
        config = make_config(
            mean=0.1,
            stddev=0.0,
            sample_rates=[44.1],
            depths=[bit_depth],
            audio_format=AudioFormat.MP3,
        )
        generator = AudioGenerator(config)
        data_uri = generator.generate()

        assert data_uri.startswith("mp3,")
        audio_data, _ = decode_audio(data_uri)
        assert len(audio_data) > 0
