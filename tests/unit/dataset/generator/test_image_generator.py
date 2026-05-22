# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common.enums import ImageFormat, ImageSource
from aiperf.config.dataset.content import ImageConfig
from aiperf.config.distributions import NormalDistribution
from aiperf.dataset.generator import ImageGenerator

# v1 had separate ImageWidthConfig / ImageHeightConfig dataclasses; v2 uses
# a single SamplingDistribution (Normal/Fixed/...). Tests authored against
# v1 still spell out the per-axis shape, so alias here so they keep reading
# naturally without rewriting every callsite.
ImageWidthConfig = NormalDistribution
ImageHeightConfig = NormalDistribution


def make_image_config(
    *,
    width_mean: float,
    width_stddev: float,
    height_mean: float,
    height_stddev: float,
    image_format: ImageFormat = ImageFormat.PNG,
    source: ImageSource | Path = ImageSource.ASSETS,
    batch_size: int = 1,
) -> ImageConfig:
    """Build a v2 ImageConfig from mean/stddev parameters.

    Defaults ``source`` to ``ASSETS`` so disk-loading and source-image sampling
    paths are exercised by these tests. NOISE bypasses disk entirely, so tests
    that need to verify file-loading behavior must keep the ASSETS default.
    Defaults ``batch_size`` to 1 so ``images_enabled()`` is True and the
    generator's RNGs and source dispatch run; tests that want a disabled
    ImageConfig should construct it directly.
    """
    return ImageConfig(
        width=NormalDistribution(mean=width_mean, stddev=width_stddev),
        height=NormalDistribution(mean=height_mean, stddev=height_stddev),
        format=image_format,
        source=source,
        batch_size=batch_size,
    )


@pytest.fixture
def base_config() -> ImageConfig:
    """Base configuration for ImageGenerator tests."""
    return make_image_config(
        width_mean=10, width_stddev=2, height_mean=10, height_stddev=2
    )


@pytest.fixture
def config_random_format() -> ImageConfig:
    """Configuration with random format selection."""
    return make_image_config(
        width_mean=10,
        width_stddev=2,
        height_mean=10,
        height_stddev=2,
        image_format=ImageFormat.RANDOM,
    )


@pytest.fixture
def config_fixed_dimensions() -> ImageConfig:
    """Configuration with fixed dimensions (stddev=0)."""
    return make_image_config(
        width_mean=10, width_stddev=0, height_mean=10, height_stddev=0
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
        patch("aiperf.dataset.generator.image.glob.glob") as mock_glob,
        patch("aiperf.dataset.generator.image.Image.open") as mock_open,
    ):
        # Create mock images with copy() method
        mock_image = Mock(spec=Image.Image)
        mock_image.copy.return_value = mock_image

        # Support context manager protocol
        mock_open.return_value.__enter__ = Mock(return_value=mock_image)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        yield {
            "mock_glob": mock_glob,
            "mock_open": mock_open,
            "mock_image": mock_image,
        }


@pytest.fixture(
    params=[
        dict(
            width_mean=50,
            width_stddev=5,
            height_mean=75,
            height_stddev=8,
            image_format=ImageFormat.JPEG,
        ),
        dict(
            width_mean=200,
            width_stddev=20,
            height_mean=150,
            height_stddev=15,
            image_format=ImageFormat.RANDOM,
        ),
        dict(
            width_mean=1024,
            width_stddev=0,
            height_mean=768,
            height_stddev=0,
            image_format=ImageFormat.PNG,
        ),
    ]
)
def various_configs(request) -> ImageConfig:
    """Parameterized fixture providing various ImageConfig configurations."""
    return make_image_config(**request.param)


@pytest.fixture(
    params=[
        (1, 0, 1, 0),  # Minimum size
        (100, 0, 50, 0),  # Fixed size
        (200, 50, 300, 75),  # Variable size
    ]
)
def dimension_params(request) -> ImageConfig:
    """Parameterized fixture providing various dimension configurations."""
    width_mean, width_stddev, height_mean, height_stddev = request.param
    return make_image_config(
        width_mean=width_mean,
        width_stddev=width_stddev,
        height_mean=height_mean,
        height_stddev=height_stddev,
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

    @patch(
        "aiperf.dataset.generator.image.utils.encode_image",
        return_value="fake_base64_string",
    )
    def test_generate_with_specified_format(self, mock_encode, base_config):
        """Test generate method with a specified image format."""
        generator = ImageGenerator(base_config)
        result = generator.generate()

        expected_result = "data:image/png;base64,fake_base64_string"
        assert result == expected_result

    def test_generate_with_random_format(self):
        """Test generate method when format is random (random selection)."""
        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=2),
            height=ImageHeightConfig(mean=10, stddev=2),
            format=ImageFormat.RANDOM,
            source=ImageSource.NOISE,
        )
        generator = ImageGenerator(config)
        result = generator.generate()
        assert result.startswith("data:image/")
        assert "random" not in result

    def test_generate_multiple_calls_different_results(self):
        """Test that multiple generate calls can produce different results."""
        from aiperf.common import random_generator as rng

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=2),
            height=ImageHeightConfig(mean=10, stddev=2),
            format=ImageFormat.PNG,
            source=ImageSource.NOISE,
        )
        rng.reset()
        rng.init(42)
        generator = ImageGenerator(config)
        image1 = generator.generate()
        image2 = generator.generate()

        assert image1 != image2

    def test_create_from_file_success(self, base_config, mock_file_system):
        """Test successful loading and sampling of source images."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = [
            "/path/image1.jpg",
            "/path/image2.png",
            "/path/image3.gif",
        ]
        mocks["mock_image"].resize.return_value = mocks["mock_image"]

        generator = ImageGenerator(base_config)

        mocks["mock_glob"].assert_called_once()
        glob_call_path = mocks["mock_glob"].call_args[0][0]
        assert "source_images" in glob_call_path and glob_call_path.endswith("*")
        assert mocks["mock_open"].call_count == 3

        result = generator._create_from_source_images(10, 10)
        assert result == mocks["mock_image"]

    def test_file_mode_no_images_found_raises(self, base_config, mock_file_system):
        """Test error handling when no source images are found."""
        mock_file_system["mock_glob"].return_value = []

        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(base_config)

        mock_file_system["mock_glob"].assert_called_once()

    def test_create_from_file_single_image(self, base_config, mock_file_system):
        """Test sampling when only one source image exists."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = ["/path/single_image.jpg"]
        mocks["mock_image"].resize.return_value = mocks["mock_image"]

        generator = ImageGenerator(base_config)

        mocks["mock_glob"].assert_called_once()
        mocks["mock_open"].assert_called_once_with("/path/single_image.jpg")

        result = generator._create_from_source_images(10, 10)
        assert result == mocks["mock_image"]

    def test_generate_integration_with_real_image(self):
        """Integration test with noise mode producing a decodable image."""
        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=2),
            height=ImageHeightConfig(mean=10, stddev=2),
            format=ImageFormat.PNG,
            source=ImageSource.NOISE,
        )
        generator = ImageGenerator(config)
        result = generator.generate()

        assert result.startswith("data:image/")
        assert ";base64," in result

        _, base64_data = result.split(";base64,")
        decoded_data = base64.b64decode(base64_data)
        decoded_image = Image.open(BytesIO(decoded_data))
        assert decoded_image.format in ["PNG", "JPEG"]

    @pytest.mark.parametrize(
        "image_format, expected_prefix",
        [
            (ImageFormat.PNG, "data:image/png;base64,"),
            (ImageFormat.JPEG, "data:image/jpeg;base64,"),
        ],
    )
    def test_generate_different_formats(self, image_format, expected_prefix):
        """Test generate method with different image formats."""
        config = make_image_config(
            width_mean=100,
            width_stddev=0,
            height_mean=100,
            height_stddev=0,
            image_format=image_format,
            source=ImageSource.NOISE,
        )
        generator = ImageGenerator(config)
        result = generator.generate()
        assert result.startswith(expected_prefix)

    @pytest.mark.parametrize(
        "width_mean, width_stddev, height_mean, height_stddev",
        [
            (1, 0, 1, 0),
            (100, 0, 50, 0),
            (200, 50, 300, 75),
        ],
    )
    def test_generate_various_dimensions(
        self, width_mean, width_stddev, height_mean, height_stddev
    ):
        """Test generate method with various dimension configurations."""
        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=width_mean, stddev=width_stddev),
            height=ImageHeightConfig(mean=height_mean, stddev=height_stddev),
            format=ImageFormat.PNG,
            source=ImageSource.NOISE,
        )
        generator = ImageGenerator(config)
        result = generator.generate()

        assert result.startswith("data:image/png;base64,")
        _, base64_data = result.split(";base64,")
        decoded_data = base64.b64decode(base64_data)
        decoded_image = Image.open(BytesIO(decoded_data))
        assert decoded_image.size[0] > 0
        assert decoded_image.size[1] > 0

    def test_deterministic_image_generation(self):
        """Test that image generation is deterministic with same seed."""
        from aiperf.common import random_generator as rng

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=2),
            height=ImageHeightConfig(mean=10, stddev=2),
            format=ImageFormat.PNG,
            source=ImageSource.NOISE,
        )

        def generate_with_seed(seed):
            rng.reset()
            rng.init(seed)
            generator = ImageGenerator(config)
            return generator.generate()

        assert generate_with_seed(12345) == generate_with_seed(12345)


class TestImageGeneratorNoiseMode:
    """Tests for noise source mode."""

    @pytest.fixture
    def noise_config(self):
        return ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            format=ImageFormat.PNG,
            source=ImageSource.NOISE,
        )

    def test_init_noise_mode_skips_disk(self, noise_config):
        generator = ImageGenerator(noise_config)
        assert not hasattr(generator, "_source_images")

    def test_generate_noise_returns_valid_data_url(self, noise_config):
        generator = ImageGenerator(noise_config)
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_noise_generates_correct_dimensions(self, noise_config):
        generator = ImageGenerator(noise_config)
        result = generator.generate()
        _, base64_data = result.split(";base64,")
        decoded = base64.b64decode(base64_data)
        img = Image.open(BytesIO(decoded))
        assert img.size == (10, 10)

    def test_noise_deterministic_with_same_seed(self, noise_config):
        from aiperf.common import random_generator as rng

        def generate_with_seed(seed):
            rng.reset()
            rng.init(seed)
            generator = ImageGenerator(noise_config)
            return generator.generate()

        assert generate_with_seed(42) == generate_with_seed(42)

    def test_noise_produces_different_images_per_call(self, noise_config):
        generator = ImageGenerator(noise_config)
        results = [generator.generate() for _ in range(5)]
        assert len(set(results)) == 5


class TestImageGeneratorCustomDirectory:
    """Tests for custom directory source mode."""

    def test_custom_directory_loads_images(self, tmp_path):
        img = Image.new("RGB", (5, 5), color="blue")
        img.save(tmp_path / "test.png")

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            format=ImageFormat.PNG,
            source=tmp_path,
        )
        generator = ImageGenerator(config)
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_custom_directory_skips_non_image_files(self, tmp_path):
        """Non-image entries (text, subdirs) must be skipped, not crash generation."""
        img = Image.new("RGB", (5, 5), color="red")
        img.save(tmp_path / "valid.png")
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "subdir").mkdir()

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            format=ImageFormat.PNG,
            source=tmp_path,
        )
        generator = ImageGenerator(config)
        assert len(generator._source_images) == 1
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_custom_directory_only_non_image_files_raises(self, tmp_path):
        """A directory with only non-image files raises rather than silently producing nothing."""
        (tmp_path / "notes.txt").write_text("hello")

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            source=tmp_path,
        )
        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(config)

    def test_custom_directory_not_found_raises(self):
        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            source=Path("/nonexistent/dir"),
        )
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ImageGenerator(config)

    def test_custom_directory_is_file_raises(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            source=file_path,
        )
        with pytest.raises(NotADirectoryError, match="is not a directory"):
            ImageGenerator(config)

    def test_custom_directory_empty_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            source=empty_dir,
        )
        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(config)


class TestImageGeneratorDisabled:
    """Tests for disabled image generation."""

    def test_disabled_images_skips_init(self):
        config = ImageConfig()
        generator = ImageGenerator(config)
        assert generator.config.images_enabled() is False
        assert not hasattr(generator, "_dimensions_rng")
