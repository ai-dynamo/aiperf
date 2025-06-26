# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common.enums import ImageFormat
from aiperf.services.dataset.config import ImageConfig
from aiperf.services.dataset.generator.image import ImageGenerator


@pytest.fixture
def base_config():
    """Base configuration for ImageGenerator tests."""
    return ImageConfig(
        width_mean=100,
        width_stddev=10,
        height_mean=100,
        height_stddev=10,
        format=ImageFormat.PNG,
    )


@pytest.fixture
def config_no_format():
    """Configuration with no format specified (for random format selection)."""
    return ImageConfig(
        width_mean=100, width_stddev=10, height_mean=100, height_stddev=10, format=None
    )


@pytest.fixture
def config_fixed_dimensions():
    """Configuration with fixed dimensions (stddev=0)."""
    return ImageConfig(
        width_mean=100,
        width_stddev=0,
        height_mean=100,
        height_stddev=0,
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
    return Image.new("RGB", (50, 50), color="red")


@pytest.fixture
def mock_image_generation_setup(mock_image):
    """Complete mock setup for image generation testing."""
    source_image, resized_image = mock_image

    with (
        patch.object(
            ImageGenerator, "_sample_source_image", return_value=source_image
        ) as mock_sample,
        patch(
            "aiperf.services.dataset.generator.image.utils.sample_positive_normal_integer"
        ) as mock_sample_int,
        patch(
            "aiperf.services.dataset.generator.image.utils.encode_image"
        ) as mock_encode,
    ):
        mock_sample_int.side_effect = [120, 80]  # width, height
        mock_encode.return_value = "fake_base64_string"

        yield {
            "mock_sample": mock_sample,
            "mock_sample_int": mock_sample_int,
            "mock_encode": mock_encode,
            "source_image": source_image,
            "resized_image": resized_image,
        }


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
            width_mean=50,
            width_stddev=5,
            height_mean=75,
            height_stddev=8,
            format=ImageFormat.JPEG,
        ),
        ImageConfig(
            width_mean=200,
            width_stddev=20,
            height_mean=150,
            height_stddev=15,
            format=None,
        ),
        ImageConfig(
            width_mean=1024,
            width_stddev=0,
            height_mean=768,
            height_stddev=0,
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
        width_mean=width_mean,
        width_stddev=width_stddev,
        height_mean=height_mean,
        height_stddev=height_stddev,
        format=ImageFormat.PNG,
    )


class TestImageGenerator:
    """Comprehensive test suite for ImageGenerator class."""

    def test_init_with_config(self, base_config):
        """Test ImageGenerator initialization with valid config."""
        generator = ImageGenerator(base_config)
        assert generator.config == base_config
        assert hasattr(generator, "logger")

    def test_init_with_different_configs(self, various_configs):
        """Test initialization with various config parameters."""
        generator = ImageGenerator(various_configs)
        assert generator.config == various_configs

    def test_generate_with_specified_format(
        self, base_config, mock_image_generation_setup
    ):
        """Test generate method with a specified image format."""
        mocks = mock_image_generation_setup

        generator = ImageGenerator(base_config)
        result = generator.generate()

        # Verify calls
        mocks["mock_sample_int"].assert_any_call(100, 10)  # width
        mocks["mock_sample_int"].assert_any_call(100, 10)  # height
        mocks["mock_sample"].assert_called_once()
        mocks["source_image"].resize.assert_called_once_with(size=(120, 80))
        mocks["mock_encode"].assert_called_once_with(mocks["resized_image"], "PNG")

        # Verify result
        expected_result = "data:image/png;base64,fake_base64_string"
        assert result == expected_result

    def test_generate_with_random_format(self, config_no_format, mock_image):
        """Test generate method when format is None (random selection)."""
        source_image, _ = mock_image

        with (
            patch.object(
                ImageGenerator, "_sample_source_image", return_value=source_image
            ),
            patch(
                "aiperf.services.dataset.generator.image.utils.sample_positive_normal_integer",
                side_effect=[150, 200],
            ),
            patch(
                "aiperf.services.dataset.generator.image.utils.encode_image",
                return_value="random_format_base64",
            ),
            patch(
                "aiperf.services.dataset.generator.image.random.choice",
                return_value=ImageFormat.JPEG,
            ) as mock_choice,
        ):
            generator = ImageGenerator(config_no_format)
            result = generator.generate()

            # Verify random choice was called with ImageFormat list
            mock_choice.assert_called_once_with(list(ImageFormat))

            expected_result = "data:image/jpeg;base64,random_format_base64"
            assert result == expected_result

    def test_generate_multiple_calls_different_results(self, base_config, mock_image):
        """Test that multiple generate calls can produce different results."""
        source_image, _ = mock_image

        with (
            patch.object(
                ImageGenerator, "_sample_source_image", return_value=source_image
            ),
            patch(
                "aiperf.services.dataset.generator.image.utils.sample_positive_normal_integer",
                side_effect=[100, 100, 150, 150],
            ),
            patch(
                "aiperf.services.dataset.generator.image.utils.encode_image",
                side_effect=["first_image", "second_image"],
            ) as mock_encode,
        ):
            generator = ImageGenerator(base_config)

            result1 = generator.generate()
            result2 = generator.generate()

            assert "first_image" in result1
            assert "second_image" in result2
            assert mock_encode.call_count == 2

    def test_sample_source_image_success(self, base_config, mock_file_system):
        """Test successful sampling of source image."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = [
            "/path/image1.jpg",
            "/path/image2.png",
            "/path/image3.gif",
        ]
        mocks["mock_choice"].return_value = "/path/image2.png"

        generator = ImageGenerator(base_config)
        result = generator._sample_source_image()

        # Verify the correct path was constructed
        mocks["mock_glob"].assert_called_once()
        glob_call_path = mocks["mock_glob"].call_args[0][0]
        assert "source_images" in glob_call_path and glob_call_path.endswith("*")

        mocks["mock_choice"].assert_called_once_with(
            ["/path/image1.jpg", "/path/image2.png", "/path/image3.gif"]
        )
        mocks["mock_open"].assert_called_once_with("/path/image2.png")
        assert result == mocks["mock_image"]

    def test_sample_source_image_no_images_found(self, base_config, mock_file_system):
        """Test error handling when no source images are found."""
        mock_file_system["mock_glob"].return_value = []  # No files found

        generator = ImageGenerator(base_config)

        with pytest.raises(ValueError) as exc_info:
            generator._sample_source_image()

        assert "No source images found" in str(exc_info.value)
        mock_file_system["mock_glob"].assert_called_once()

    def test_sample_source_image_single_file(self, base_config, mock_file_system):
        """Test sampling when only one source image exists."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = ["/path/single_image.jpg"]
        mocks["mock_choice"].return_value = "/path/single_image.jpg"

        generator = ImageGenerator(base_config)
        result = generator._sample_source_image()

        mocks["mock_choice"].assert_called_once_with(["/path/single_image.jpg"])
        mocks["mock_open"].assert_called_once_with("/path/single_image.jpg")
        assert result == mocks["mock_image"]

    def test_generate_integration_with_real_image(self, base_config, test_image):
        """Integration test using a real image (mocked filesystem)."""
        with patch.object(
            ImageGenerator, "_sample_source_image", return_value=test_image
        ):
            generator = ImageGenerator(base_config)
            result = generator.generate()

            # Verify the result is a valid data URL
            assert result.startswith("data:image/")
            assert ";base64," in result

            # Verify we can decode the image
            _, base64_data = result.split(";base64,")
            decoded_data = base64.b64decode(base64_data)
            decoded_image = Image.open(BytesIO(decoded_data))

            # The image should have been resized (not 50x50 anymore due to random sampling)
            assert decoded_image.size != (50, 50)
            assert decoded_image.format in ["PNG", "JPEG"]

    @pytest.mark.parametrize(
        "image_format, expected_prefix",
        [
            (ImageFormat.PNG, "data:image/png;base64,"),
            (ImageFormat.JPEG, "data:image/jpeg;base64,"),
        ],
    )
    def test_generate_different_formats(
        self, image_format, expected_prefix, test_image
    ):
        """Test generate method with different image formats."""
        config = ImageConfig(
            width_mean=100,
            width_stddev=0,
            height_mean=100,
            height_stddev=0,
            format=image_format,
        )

        with patch.object(
            ImageGenerator, "_sample_source_image", return_value=test_image
        ):
            generator = ImageGenerator(config)
            result = generator.generate()

            assert result.startswith(expected_prefix)

    def test_generate_various_dimensions(self, dimension_params, test_image):
        """Test generate method with various dimension configurations."""
        with patch.object(
            ImageGenerator, "_sample_source_image", return_value=test_image
        ):
            generator = ImageGenerator(dimension_params)
            result = generator.generate()

            # Verify it's a valid data URL
            assert result.startswith("data:image/png;base64,")

            # Decode and verify the image
            _, base64_data = result.split(";base64,")
            decoded_data = base64.b64decode(base64_data)
            decoded_image = Image.open(BytesIO(decoded_data))

            # We can verify it's a valid image
            assert decoded_image.size[0] > 0
            assert decoded_image.size[1] > 0
