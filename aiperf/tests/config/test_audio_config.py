#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import AudioConfig, AudioDefaults, AudioLengthConfig
from aiperf.common.enums import AudioFormat


def test_audio_config_defaults():
    """
    Test the default values of the AudioConfig class.

    This test verifies that the AudioConfig object is initialized with the correct
    default values as defined in the AudioDefaults class.
    """
    config = AudioConfig()
    assert config.batch_size == AudioDefaults.BATCH_SIZE
    assert config.length.mean == AudioDefaults.LENGTH_MEAN
    assert config.length.stddev == AudioDefaults.LENGTH_STDDEV
    assert config.format == AudioDefaults.FORMAT
    assert config.depths == AudioDefaults.DEPTHS
    assert config.sample_rates == AudioDefaults.SAMPLE_RATES
    assert config.num_channels == AudioDefaults.NUM_CHANNELS


def test_audio_config_custom_values():
    """
    This test ensures that the AudioConfig object is properly initialized
    when provided with custom input values. It verifies that the attributes
    of the object match the expected values specified in the test.

    Assertions:
    - Each attribute of the AudioConfig object matches the corresponding
        value in the custom_values dictionary.
    """

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
