# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import (
    ImageConfig,
    ImageDefaults,
    ImageHeightConfig,
    ImageWidthConfig,
)
from aiperf.common.enums import ImageFormat


def test_image_config_defaults():
    """
    Test the default values of the ImageConfig class.

    This test verifies that the ImageConfig object is initialized with the correct
    default values as defined in the ImageDefaults class.
    """
    config = ImageConfig()
    assert config.width.mean == ImageDefaults.WIDTH_MEAN
    assert config.width.stddev == ImageDefaults.WIDTH_STDDEV
    assert config.height.mean == ImageDefaults.HEIGHT_MEAN
    assert config.height.stddev == ImageDefaults.HEIGHT_STDDEV
    assert config.batch_size == ImageDefaults.BATCH_SIZE
    assert config.format == ImageDefaults.FORMAT


def test_image_config_custom_values():
    """
    Test the InputConfig class with custom values.

    This test verifies that the InputConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "width": ImageWidthConfig(mean=640.0, stddev=80.0),
        "height": ImageHeightConfig(mean=480.0, stddev=60.0),
        "batch_size": 16,
        "format": ImageFormat.JPEG,
    }
    config = ImageConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
