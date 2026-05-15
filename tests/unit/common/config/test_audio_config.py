# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.config import AudioConfig, AudioLengthConfig
from aiperf.common.enums import AudioFormat


def test_audio_config_defaults():
    """Test the default values of the AudioConfig class."""
    config = AudioConfig()
    assert config.batch_size == 1
    assert config.length.mean == 0.0
    assert config.length.stddev == 0.0
    assert config.format == AudioFormat.WAV
    assert config.depths == [16]
    assert config.sample_rates == [16.0]
    assert config.num_channels == 1


def test_audio_config_custom_values():
    """Test AudioConfig correctly initializes with custom values."""
    custom_values = {
        "batch_size": 32,
        "length": AudioLengthConfig(mean=5.0, stddev=1.0),
        "format": AudioFormat.WAV,
        "depths": [16, 24],
        "sample_rates": [44, 48],
        "num_channels": 2,
    }
    config = AudioConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


class TestAudioEnabled:
    def test_enabled_when_all_conditions_met(self):
        config = AudioConfig(
            batch_size=1,
            length=AudioLengthConfig(mean=2.0),
        )
        assert config.audio_enabled() is True

    def test_disabled_by_default(self):
        config = AudioConfig()
        assert config.audio_enabled() is False

    def test_disabled_when_length_mean_zero(self):
        config = AudioConfig.model_construct(
            batch_size=1,
            length=AudioLengthConfig(mean=0),
        )
        assert config.audio_enabled() is False

    def test_disabled_when_batch_size_zero(self):
        config = AudioConfig.model_construct(
            batch_size=0,
            length=AudioLengthConfig(mean=2.0),
        )
        assert config.audio_enabled() is False


class TestAudioConfigValidation:
    def test_rejects_options_when_audio_disabled(self):
        """Non-default length params without enabling audio raise."""
        with pytest.raises(ValidationError, match="Audio generation is disabled"):
            AudioConfig(
                length=AudioLengthConfig(stddev=2.0),
            )

    def test_format_alone_does_not_require_audio_enabled(self):
        """Setting format without length is allowed (e.g., for external loaders)."""
        config = AudioConfig(format=AudioFormat.WAV)
        assert config.format == AudioFormat.WAV
        assert config.audio_enabled() is False

    def test_explicit_batch_size_zero_disable(self):
        """`--audio-batch-size 0` is treated as explicit disable, never raises."""
        config = AudioConfig(batch_size=0)
        assert config.audio_enabled() is False

    def test_explicit_batch_size_zero_with_length_disable(self):
        """`batch_size=0` overrides any length intent without raising."""
        config = AudioConfig(
            batch_size=0,
            length=AudioLengthConfig(mean=5.0, stddev=1.0),
        )
        assert config.audio_enabled() is False

    def test_default_valued_length_does_not_raise(self):
        """Loading a config dump with all default fields set must not raise."""
        config = AudioConfig(
            length=AudioLengthConfig(mean=0.0, stddev=0.0),
            format=AudioFormat.WAV,
            depths=[16],
            sample_rates=[16.0],
            num_channels=1,
        )
        assert config.audio_enabled() is False

    def test_config_yaml_roundtrip_does_not_raise(self):
        """Round-tripping a fully-serialized default config must not raise."""
        original = AudioConfig()
        dumped = original.model_dump()
        # All keys are present in the dump but at default values
        config = AudioConfig.model_validate(dumped)
        assert config.audio_enabled() is False
