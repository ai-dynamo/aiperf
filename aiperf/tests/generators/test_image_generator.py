# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common.config import ImageConfig, ImageHeightConfig, ImageWidthConfig
from aiperf.common.enums import ImageFormat
from aiperf.common.exceptions import InitializationError
from aiperf.services.dataset.generator.image import ImageGenerator


@pytest.fixture
def base_config():
    """Base configuration for ImageGenerator tests."""
    return ImageConfig(
        width=ImageWidthConfig(mean=10, stddev=2),
        height=ImageHeightConfig(mean=10, stddev=2),
        format=ImageFormat.PNG,
    )


@pytest.fixture
def config_random_format():
    """Configuration with no format specified (for random format selection)."""
    return ImageConfig(
        width=ImageWidthConfig(mean=10, stddev=2),
        height=ImageHeightConfig(mean=10, stddev=2),
        format=ImageFormat.RANDOM,
    )


@pytest.fixture
def config_fixed_dimensions():
    """Configuration with fixed dimensions (stddev=0)."""
    return ImageConfig(
        width=ImageWidthConfig(mean=10, stddev=0),
        height=ImageHeightConfig(mean=10, stddev=0),
        format=ImageFormat.PNG,
    )


@pytest.fixture
def mock_image() -> tuple[Mock, Mock]:
    """Mock PIL Image object for source image."""
    image = Mock(spec=Image.Image)
    resized_image = Mock(spec=Image.Image)
    image.resize.return_value = resized_image
    return image, resized_image


@pytest.fixture
def test_image() -> Image.Image:
    """Real PIL Image object for integration tests."""
    return Image.new("RGB", (5, 5), color="red")


@pytest.fixture
def mock_file_system():
    """Mock file system for testing source image sampling."""
    with (
        patch("aiperf.services.dataset.generator.image.glob.glob") as mock_glob,
        patch("aiperf.services.dataset.generator.image.random.choice") as mock_choice,
        patch("aiperf.services.dataset.generator.image.Image.open") as mock_open,
    ):
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value = mock_image

        yield {
            "mock_glob": mock_glob,
            "mock_choice": mock_choice,
            "mock_open": mock_open,
            "mock_image": mock_image,
        }


@pytest.fixture(
    params=[
        ImageConfig(
            width=ImageWidthConfig(mean=50, stddev=5),
            height=ImageHeightConfig(mean=75, stddev=8),
            format=ImageFormat.JPEG,
        ),
        ImageConfig(
            width=ImageWidthConfig(mean=200, stddev=20),
            height=ImageHeightConfig(mean=150, stddev=15),
            format=ImageFormat.RANDOM,
        ),
        ImageConfig(
            width=ImageWidthConfig(mean=1024, stddev=0),
            height=ImageHeightConfig(mean=768, stddev=0),
            format=ImageFormat.PNG,
        ),
    ]
)
def various_configs(request):
    """Parameterized fixture providing various ImageConfig configurations."""
    return request.param


@pytest.fixture(
    params=[
        (1, 0, 1, 0),  # Minimum size
        (100, 0, 50, 0),  # Fixed size
        (200, 50, 300, 75),  # Variable size
    ]
)
def dimension_params(request):
    """Parameterized fixture providing various dimension configurations."""
    width_mean, width_stddev, height_mean, height_stddev = request.param
    return ImageConfig(
        width=ImageWidthConfig(mean=width_mean, stddev=width_stddev),
        height=ImageHeightConfig(mean=height_mean, stddev=height_stddev),
        format=ImageFormat.PNG,
    )


@pytest.fixture
def initialized_generator(base_config):
    """Initialized ImageGenerator for testing."""
    generator = ImageGenerator(base_config)
    generator.data_initialized.set()
    return generator


@pytest.mark.asyncio
class TestImageGenerator:
    """Comprehensive test suite for ImageGenerator class."""

    async def test_init_with_config(self, base_config):
        """Test ImageGenerator initialization with valid config."""
        generator = ImageGenerator(base_config)
        assert generator.config == base_config
        assert hasattr(generator, "logger")
        assert not generator.data_initialized.is_set()

    async def test_init_with_different_configs(self, various_configs):
        """Test initialization with various config parameters."""
        generator = ImageGenerator(various_configs)
        assert generator.config == various_configs
        assert not generator.data_initialized.is_set()

    async def test_initialize(self, base_config, mock_file_system):
        """Test that initialize method loads images and sets initialized flag."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = [
            "/path/image1.jpg",
            "/path/image2.png",
        ]

        generator = ImageGenerator(base_config)

        assert not generator.data_initialized.is_set()
        assert len(generator.images) == 0

        await generator.initialize()

        assert generator.data_initialized.is_set()
        assert len(generator.images) == 2

    async def test_initialize_no_source_images(self, base_config, mock_file_system):
        """Test error handling when no source images are found."""
        mock_file_system["mock_glob"].return_value = []  # No files found

        generator = ImageGenerator(base_config)

        with pytest.raises(InitializationError) as exc_info:
            await generator.initialize()

        assert "No source images found" in str(exc_info.value)

    @pytest.mark.parametrize("format", [ImageFormat.PNG, ImageFormat.JPEG])
    async def test_generate_with_specified_format(
        self, initialized_generator, test_image, format
    ):
        """Test generate method with a specified image format."""
        initialized_generator.images = [test_image]
        initialized_generator.config.format = format

        result = await initialized_generator.generate()

        assert result.startswith(f"data:image/{format};base64")

    async def test_generate_with_random_format(self, initialized_generator, test_image):
        """Test generate method when format is random (random selection)."""
        initialized_generator.images = [test_image]
        initialized_generator.config.format = ImageFormat.RANDOM

        result = await initialized_generator.generate()
        assert result.startswith("data:image/")
        assert "random" not in result

    async def test_generate_different_images(self, initialized_generator, test_image):
        """Test that multiple generate calls can produce different images."""
        initialized_generator.images = [test_image]

        image1 = await initialized_generator.generate()
        image2 = await initialized_generator.generate()

        assert image1 != image2

    @pytest.mark.parametrize("width, height", [(12, 34), (100, 100), (200, 200)])
    async def test_generate_various_dimensions(
        self, initialized_generator, test_image, width, height
    ):
        """Integration test using a real image (mocked filesystem)."""
        initialized_generator.images = [test_image]
        initialized_generator.config = ImageConfig(
            width=ImageWidthConfig(mean=width, stddev=0),
            height=ImageHeightConfig(mean=height, stddev=0),
            format=ImageFormat.JPEG,
        )

        result = await initialized_generator.generate()

        # Verify we can decode the image
        _, base64_data = result.split(";base64,")
        decoded_data = base64.b64decode(base64_data)
        decoded_image = Image.open(BytesIO(decoded_data))

        # The image should have been resized to target dimensions
        assert decoded_image.size == (width, height)
        assert decoded_image.format == "JPEG"
