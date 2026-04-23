# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aiperf.common import random_generator as rng
from aiperf.common.enums import AudioFormat
from aiperf.common.exceptions import AIPerfConfigurationError
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.dataset.generator import (
    AudioGenerator,
)


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test"))


_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def _make_config(**audio_overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with a single synthetic dataset containing audio config."""
    audio = {
        "batch_size": 1,
        "length": {"mean": 3.0, "stddev": 0.4},
        "sample_rates": [44.1],
        "depths": [16],
        "format": "wav",
        "channels": 1,
    }
    audio.update(audio_overrides)
    return AIPerfConfig(
        **_BASE,
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
                "audio": audio,
            }
        },
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
    return _make_config()


@pytest.mark.parametrize(
    "expected_audio_length",
    [
        1.0,
        2.0,
    ],
)
def test_different_audio_length(expected_audio_length, base_config):
    audio_generator = AudioGenerator(_make_run(base_config))
    audio_generator.audio_config.length.mean = expected_audio_length
    audio_generator.audio_config.length.stddev = 0.0  # make it deterministic

    data_uri = audio_generator.generate()

    audio_data, sample_rate = decode_audio(data_uri)
    actual_length = len(audio_data) / sample_rate
    assert abs(actual_length - expected_audio_length) < 0.1, (
        "audio length not as expected"
    )


def test_negative_length_raises_error(base_config):
    audio_generator = AudioGenerator(_make_run(base_config))
    audio_generator.audio_config.length.mean = -1.0

    with pytest.raises(AIPerfConfigurationError):
        audio_generator.generate()


@pytest.mark.parametrize(
    "mean, stddev, sampling_rate, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
def test_generator_deterministic(mean, stddev, sampling_rate, bit_depth, base_config):
    # First generation with seed 123
    rng.reset()
    rng.init(123)
    generator1 = AudioGenerator(_make_run(base_config))
    generator1.audio_config.length.mean = mean
    generator1.audio_config.length.stddev = stddev
    generator1.audio_config.sample_rates = [sampling_rate]
    generator1.audio_config.depths = [bit_depth]
    data_uri1 = generator1.generate()

    # Second generation with same seed 123
    rng.reset()
    rng.init(123)
    generator2 = AudioGenerator(_make_run(base_config))
    generator2.audio_config.length.mean = mean
    generator2.audio_config.length.stddev = stddev
    generator2.audio_config.sample_rates = [sampling_rate]
    generator2.audio_config.depths = [bit_depth]
    data_uri2 = generator2.generate()

    # Compare the actual audio data
    audio_data1, _ = decode_audio(data_uri1)
    audio_data2, _ = decode_audio(data_uri2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("audio_format", [AudioFormat.WAV, AudioFormat.MP3])
def test_audio_format(audio_format, base_config):
    # use sample rate supported by all formats (44.1kHz)
    audio_generator = AudioGenerator(_make_run(base_config))
    audio_generator.audio_config.format = audio_format

    data_uri = audio_generator.generate()

    # Check data URI format
    assert data_uri.startswith(f"{audio_format.name.lower()},"), (
        "incorrect data URI format"
    )

    # Verify the audio can be decoded
    audio_data, _ = decode_audio(data_uri)
    assert len(audio_data) > 0, "audio data is empty"


def test_unsupported_bit_depth(base_config):
    audio_generator = AudioGenerator(_make_run(base_config))
    audio_generator.audio_config.depths = [12]  # Unsupported bit depth

    with pytest.raises(AIPerfConfigurationError) as exc_info:
        audio_generator.generate()

    assert "Supported bit depths are:" in str(exc_info.value)


@pytest.mark.parametrize("channels", [1, 2])
def test_channels(channels, base_config):
    config = _make_config(channels=channels)

    audio_generator = AudioGenerator(_make_run(config))
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
    audio_generator = AudioGenerator(_make_run(base_config))
    audio_generator.audio_config.sample_rates = [sampling_rate_khz]
    audio_generator.audio_config.depths = [bit_depth]

    data_uri = audio_generator.generate()

    _, sample_rate = decode_audio(data_uri)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


@pytest.mark.parametrize(
    "config_changes,expected_error",
    [
        ({"sample_rates": [96], "format": AudioFormat.MP3}, "MP3 format only supports"),
        ({"channels": 3}, r"mono \(1\) and stereo \(2\)"),
        (
            {"length": {"mean": 0.005, "stddev": 0.0}},
            "must be greater than 0.01 seconds",
        ),
        ({"format": "UNSUPPORTED"}, "Unsupported audio format"),
    ],
)
def test_audio_validation_errors(base_config, config_changes, expected_error):
    """Test that invalid configurations raise appropriate ConfigurationErrors."""
    audio_generator = AudioGenerator(_make_run(base_config))

    # Apply configuration changes
    for key, value in config_changes.items():
        if key == "length":
            audio_generator.audio_config.length.mean = value["mean"]
            audio_generator.audio_config.length.stddev = value["stddev"]
        else:
            setattr(audio_generator.audio_config, key, value)

    with pytest.raises(AIPerfConfigurationError, match=expected_error):
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
        config = _make_config(
            length={"mean": 0.1, "stddev": 0.0},
            sample_rates=[16.0],
            depths=[bit_depth],
            format="wav",
            channels=1,
        )
        generator = AudioGenerator(_make_run(config))
        data_uri = generator.generate()

        _, b64_data = data_uri.split(",")
        audio_bytes = base64.b64decode(b64_data)

        with io.BytesIO(audio_bytes) as f:
            info = sf.info(f)
            assert info.subtype == expected_subtype

    @pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
    def test_wav_bit_depth_produces_valid_audio(self, bit_depth):
        """All supported bit depths produce valid, readable WAV audio."""
        config = _make_config(
            length={"mean": 0.1, "stddev": 0.0},
            sample_rates=[16.0],
            depths=[bit_depth],
            format="wav",
            channels=1,
        )
        generator = AudioGenerator(_make_run(config))
        data_uri = generator.generate()

        audio_data, sample_rate = decode_audio(data_uri)
        assert len(audio_data) > 0
        assert sample_rate == 16000

    @pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
    def test_mp3_ignores_bit_depth_uses_lossy_encoding(self, bit_depth):
        """MP3 format works with all bit depths (lossy encoding ignores PCM subtype)."""
        config = _make_config(
            length={"mean": 0.1, "stddev": 0.0},
            sample_rates=[44.1],
            depths=[bit_depth],
            format="mp3",
            channels=1,
        )
        generator = AudioGenerator(_make_run(config))
        data_uri = generator.generate()

        assert data_uri.startswith("mp3,")
        audio_data, _ = decode_audio(data_uri)
        assert len(audio_data) > 0
