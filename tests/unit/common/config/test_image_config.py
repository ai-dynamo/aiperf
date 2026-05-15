# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.config import (
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
)
from aiperf.common.enums import ImageFormat, ImageSource


def test_image_config_defaults():
    """Test the default values of the ImageConfig class."""
    config = ImageConfig()
    assert config.width.mean == 0.0
    assert config.width.stddev == 0.0
    assert config.height.mean == 0.0
    assert config.height.stddev == 0.0
    assert config.batch_size == 1
    assert config.format == ImageFormat.PNG
    assert config.source == ImageSource.NOISE


def test_image_config_custom_values():
    """Test ImageConfig correctly initializes with custom values."""
    custom_values = {
        "width": ImageWidthConfig(mean=640.0, stddev=80.0),
        "height": ImageHeightConfig(mean=480.0, stddev=60.0),
        "batch_size": 16,
        "format": ImageFormat.JPEG,
    }
    config = ImageConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


class TestImagesEnabled:
    def test_enabled_when_all_conditions_met(self):
        config = ImageConfig(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
            batch_size=1,
        )
        assert config.images_enabled() is True

    def test_disabled_by_default(self):
        config = ImageConfig()
        assert config.images_enabled() is False

    def test_disabled_when_width_zero(self):
        config = ImageConfig.model_construct(
            width=ImageWidthConfig(mean=0),
            height=ImageHeightConfig(mean=10),
            batch_size=1,
        )
        assert config.images_enabled() is False

    def test_disabled_when_height_zero(self):
        config = ImageConfig.model_construct(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=0),
            batch_size=1,
        )
        assert config.images_enabled() is False

    def test_disabled_when_batch_size_zero(self):
        config = ImageConfig.model_construct(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
            batch_size=0,
        )
        assert config.images_enabled() is False


class TestImageConfigValidation:
    def test_rejects_options_when_images_disabled(self):
        """Non-default width/height params without enabling images raise."""
        with pytest.raises(ValidationError, match="Image generation is disabled"):
            ImageConfig(
                width=ImageWidthConfig(stddev=80.0),
            )

    def test_rejects_options_when_images_disabled_height_stddev(self):
        """Non-default height stddev alone without enabling images also raises."""
        with pytest.raises(ValidationError, match="Image generation is disabled"):
            ImageConfig(
                height=ImageHeightConfig(stddev=60.0),
            )

    def test_format_alone_does_not_require_images_enabled(self):
        """Setting format without dimensions is allowed (e.g., for external loaders)."""
        config = ImageConfig(format=ImageFormat.JPEG)
        assert config.format == ImageFormat.JPEG
        assert config.images_enabled() is False

    def test_source_alone_does_not_require_images_enabled(self):
        """Setting source without dimensions is allowed (e.g., for external loaders)."""
        config = ImageConfig(source=ImageSource.NOISE)
        assert config.source == ImageSource.NOISE
        assert config.images_enabled() is False

    def test_explicit_batch_size_zero_disable(self):
        """`--image-batch-size 0` is treated as explicit disable, never raises."""
        config = ImageConfig(batch_size=0)
        assert config.images_enabled() is False

    def test_explicit_batch_size_zero_with_dims_disable(self):
        """`batch_size=0` overrides any dimension intent without raising."""
        config = ImageConfig(
            batch_size=0,
            width=ImageWidthConfig(mean=640, stddev=80),
            height=ImageHeightConfig(mean=480, stddev=60),
        )
        assert config.images_enabled() is False

    def test_default_valued_dims_does_not_raise(self):
        """Loading a config dump with all default fields set must not raise."""
        config = ImageConfig(
            width=ImageWidthConfig(mean=0.0, stddev=0.0),
            height=ImageHeightConfig(mean=0.0, stddev=0.0),
            format=ImageFormat.PNG,
            source=ImageSource.ASSETS,
        )
        assert config.images_enabled() is False

    def test_config_yaml_roundtrip_does_not_raise(self):
        """Round-tripping a fully-serialized default config must not raise."""
        original = ImageConfig()
        dumped = original.model_dump()
        config = ImageConfig.model_validate(dumped)
        assert config.images_enabled() is False

    def test_batch_size_without_dimensions_logs_warning(self, caplog):
        """Warn when positive batch size cannot enable images without dimensions."""
        config = ImageConfig(batch_size=4)

        assert config.images_enabled() is False
        assert "--image-width-mean" in caplog.text
        assert "--image-height-mean" in caplog.text

    def test_default_config_roundtrip_does_not_warn(self, caplog):
        """Fully-serialized default configs should remain quiet."""
        ImageConfig.model_validate(ImageConfig().model_dump())

        assert caplog.text == ""


class TestImageSource:
    def test_source_defaults_to_noise(self):
        config = ImageConfig(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
        )
        assert config.source == ImageSource.NOISE

    def test_source_assets(self):
        config = ImageConfig(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
            source=ImageSource.ASSETS,
        )
        assert config.source == ImageSource.ASSETS

    def test_source_noise(self):
        config = ImageConfig(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
            source=ImageSource.NOISE,
        )
        assert config.source == ImageSource.NOISE

    def test_source_custom_path(self):
        config = ImageConfig(
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
            source=Path("/tmp/my_images"),
        )
        assert config.source == Path("/tmp/my_images")

    @pytest.mark.parametrize(
        "raw,expected",
        [
            param("noise", ImageSource.NOISE, id="enum-string-noise"),
            param("assets", ImageSource.ASSETS, id="enum-string-assets"),
            param("./my_images", Path("./my_images"), id="path-relative"),
            param("/tmp/my_images", Path("/tmp/my_images"), id="path-absolute"),
            param("~/my_images", Path("~/my_images").expanduser(), id="path-home"),
        ],
    )  # fmt: skip
    def test_source_str_coercion(self, raw, expected):
        """Strings from CLI/YAML must resolve to ImageSource enum or Path."""
        config = ImageConfig.model_validate(
            {
                "width": {"mean": 10},
                "height": {"mean": 10},
                "source": raw,
            }
        )
        assert config.source == expected
