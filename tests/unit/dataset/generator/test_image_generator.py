# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common import random_generator as rng
from aiperf.common.enums import (
    ImageFormat,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.dataset.generator import ImageGenerator

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases=[
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
)


def _make_config(**image_overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with a single synthetic dataset containing image config.

    Defaults ``source`` to ``assets`` so disk-loading and source-image sampling
    paths are exercised; NOISE bypasses disk entirely, so tests that verify
    file-loading behavior must keep the ASSETS default.
    """
    images = {
        "batch_size": 1,
        "width": {"mean": 10, "stddev": 2},
        "height": {"mean": 10, "stddev": 2},
        "format": "png",
        "source": "assets",
    }
    images.update(image_overrides)
    return AIPerfConfig(
        benchmark={
            **_BASE,
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                    "images": images,
                }
            ],
        }
    )


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_id="test", cfg=config.benchmark, artifact_dir=Path("/tmp/test")
    )


def _run(**image_overrides) -> BenchmarkRun:
    return _make_run(_make_config(**image_overrides))


@pytest.fixture
def base_config():
    """Base configuration for ImageGenerator tests."""
    return _make_config()


@pytest.fixture
def config_random_format():
    """Configuration with random format selection."""
    return _make_config(format="random")


@pytest.fixture
def config_fixed_dimensions():
    """Configuration with fixed dimensions (stddev=0)."""
    return _make_config(
        width={"mean": 10, "stddev": 0},
        height={"mean": 10, "stddev": 0},
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
        mock_image.resize.return_value = mock_image

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
            width={"mean": 50, "stddev": 5},
            height={"mean": 75, "stddev": 8},
            format="jpeg",
        ),
        dict(
            width={"mean": 200, "stddev": 20},
            height={"mean": 150, "stddev": 15},
            format="random",
        ),
        dict(
            width={"mean": 1024, "stddev": 0},
            height={"mean": 768, "stddev": 0},
            format="png",
        ),
    ]
)
def various_configs(request):
    """Parameterized fixture providing various AIPerfConfig configurations."""
    return _make_config(**request.param)


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
    return _make_config(
        width={"mean": width_mean, "stddev": width_stddev},
        height={"mean": height_mean, "stddev": height_stddev},
    )


class TestImageGenerator:
    """Comprehensive test suite for ImageGenerator class."""

    def test_init_with_config(self, base_config):
        """Test ImageGenerator initialization with valid config."""
        generator = ImageGenerator(_make_run(base_config))
        assert (
            generator.image_config == base_config.benchmark.get_default_dataset().images
        )
        assert hasattr(generator, "logger")

    def test_init_with_different_configs(self, various_configs):
        """Test initialization with various config parameters."""
        generator = ImageGenerator(_make_run(various_configs))
        assert (
            generator.image_config
            == various_configs.benchmark.get_default_dataset().images
        )

    @patch(
        "aiperf.dataset.generator.image.utils.encode_image",
        return_value="fake_base64_string",
    )
    def test_generate_with_specified_format(
        self, mock_encode, base_config, mock_file_system
    ):
        """Test generate method with a specified image format."""
        mock_file_system["mock_glob"].return_value = ["/path/image1.png"]
        generator = ImageGenerator(_make_run(base_config))
        result = generator.generate()

        expected_result = "data:image/png;base64,fake_base64_string"
        assert result == expected_result

    def test_generate_with_random_format(self):
        """Test generate method when format is random (random selection)."""
        generator = ImageGenerator(_run(format="random", source="noise"))
        result = generator.generate()
        assert result.startswith("data:image/")
        assert "random" not in result

    def test_generate_multiple_calls_different_results(self):
        """Test that multiple generate calls can produce different results."""
        rng.reset()
        rng.init(42)
        generator = ImageGenerator(_run(source="noise"))
        image1 = generator.generate()
        image2 = generator.generate()

        assert image1 != image2

    def test_create_from_file_success(self, base_config, mock_file_system):
        """Test successful indexing and lazy sampling of source images."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = [
            "/path/image1.jpg",
            "/path/image2.png",
            "/path/image3.gif",
        ]

        generator = ImageGenerator(_make_run(base_config))

        mocks["mock_glob"].assert_called_once()
        glob_call_path = mocks["mock_glob"].call_args[0][0]
        assert "source_images" in glob_call_path and glob_call_path.endswith("*")
        assert len(generator._source_image_paths) == 3
        mocks["mock_open"].assert_not_called()

        result = generator._create_from_source_images(10, 10)
        assert result == mocks["mock_image"]
        mocks["mock_open"].assert_called_once()

    def test_file_mode_no_images_found_raises(self, base_config, mock_file_system):
        """Test error handling when no source images are found."""
        mock_file_system["mock_glob"].return_value = []

        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(_make_run(base_config))

        mock_file_system["mock_glob"].assert_called_once()

    def test_create_from_file_single_image(self, base_config, mock_file_system):
        """Test sampling when only one source image exists."""
        mocks = mock_file_system
        mocks["mock_glob"].return_value = ["/path/single_image.jpg"]

        generator = ImageGenerator(_make_run(base_config))

        mocks["mock_glob"].assert_called_once()
        mocks["mock_open"].assert_not_called()

        result = generator._create_from_source_images(10, 10)
        assert result == mocks["mock_image"]
        mocks["mock_open"].assert_called_once_with(Path("/path/single_image.jpg"))

    def test_generate_integration_with_real_image(self):
        """Integration test with noise mode producing a decodable image."""
        generator = ImageGenerator(_run(source="noise"))
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
        generator = ImageGenerator(
            _run(
                width={"mean": 100, "stddev": 0},
                height={"mean": 100, "stddev": 0},
                format=image_format.name.lower(),
                source="noise",
            )
        )
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
        generator = ImageGenerator(
            _run(
                width={"mean": width_mean, "stddev": width_stddev},
                height={"mean": height_mean, "stddev": height_stddev},
                source="noise",
            )
        )
        result = generator.generate()

        assert result.startswith("data:image/png;base64,")
        _, base64_data = result.split(";base64,")
        decoded_data = base64.b64decode(base64_data)
        decoded_image = Image.open(BytesIO(decoded_data))
        assert decoded_image.size[0] > 0
        assert decoded_image.size[1] > 0

    def test_deterministic_image_generation(self):
        """Test that image generation is deterministic with same seed."""

        def generate_with_seed(seed):
            rng.reset()
            rng.init(seed)
            generator = ImageGenerator(_run(source="noise"))
            return generator.generate()

        assert generate_with_seed(12345) == generate_with_seed(12345)


class TestImageGeneratorNoiseMode:
    """Tests for noise source mode."""

    def test_init_noise_mode_skips_disk(self):
        generator = ImageGenerator(
            _run(width={"mean": 10, "stddev": 0}, source="noise")
        )
        assert not hasattr(generator, "_source_image_paths")

    def test_generate_noise_returns_valid_data_url(self):
        generator = ImageGenerator(_run(source="noise"))
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_noise_generates_correct_dimensions(self):
        generator = ImageGenerator(
            _run(
                width={"mean": 10, "stddev": 0},
                height={"mean": 10, "stddev": 0},
                source="noise",
            )
        )
        result = generator.generate()
        _, base64_data = result.split(";base64,")
        decoded = base64.b64decode(base64_data)
        img = Image.open(BytesIO(decoded))
        assert img.size == (10, 10)

    def test_noise_deterministic_with_same_seed(self):
        def generate_with_seed(seed):
            rng.reset()
            rng.init(seed)
            generator = ImageGenerator(_run(source="noise"))
            return generator.generate()

        assert generate_with_seed(42) == generate_with_seed(42)

    def test_noise_produces_different_images_per_call(self):
        generator = ImageGenerator(_run(source="noise"))
        results = [generator.generate() for _ in range(5)]
        assert len(set(results)) == 5


class TestImageGeneratorCustomDirectory:
    """Tests for custom directory source mode."""

    def test_custom_directory_loads_images(self, tmp_path):
        img = Image.new("RGB", (5, 5), color="blue")
        img.save(tmp_path / "test.png")

        generator = ImageGenerator(
            _run(
                width={"mean": 10, "stddev": 0},
                height={"mean": 10, "stddev": 0},
                source=str(tmp_path),
            )
        )
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_custom_directory_shuffle_cycle_uses_pool_once(self, tmp_path):
        colors = {
            "blue.png": (0, 0, 255),
            "green.png": (0, 128, 0),
            "red.png": (255, 0, 0),
        }
        for filename, color in colors.items():
            Image.new("RGB", (5, 5), color=color).save(tmp_path / filename)

        generator = ImageGenerator(
            _run(
                width={"mean": 5, "stddev": 0},
                height={"mean": 5, "stddev": 0},
                source=str(tmp_path),
                source_sampling="shuffle-cycle",
            )
        )

        first_cycle = [
            generator._create_from_source_images(5, 5).getpixel((0, 0))
            for _ in range(len(colors))
        ]
        next_image = generator._create_from_source_images(5, 5).getpixel((0, 0))

        assert sorted(first_cycle) == sorted(colors.values())
        assert next_image in colors.values()

    def test_custom_directory_sequential_cycle_uses_sorted_order_and_wraps(
        self, tmp_path
    ):
        colors = {
            "blue.png": (0, 0, 255),
            "green.png": (0, 128, 0),
            "red.png": (255, 0, 0),
        }
        for filename, color in colors.items():
            Image.new("RGB", (5, 5), color=color).save(tmp_path / filename)

        generator = ImageGenerator(
            _run(
                width={"mean": 5, "stddev": 0},
                height={"mean": 5, "stddev": 0},
                source=str(tmp_path),
                source_sampling="sequential-cycle",
            )
        )

        sampled = [
            generator._create_from_source_images(5, 5).getpixel((0, 0))
            for _ in range(len(colors) + 1)
        ]

        assert sampled == [
            colors["blue.png"],
            colors["green.png"],
            colors["red.png"],
            colors["blue.png"],
        ]

    def test_source_sampling_rejects_noise_source(self):
        with pytest.raises(ValueError, match="requires image source"):
            _run(source="noise", source_sampling="shuffle-cycle")

    def test_custom_directory_skips_non_image_files(self, tmp_path):
        """Non-image entries (text, subdirs) must be skipped, not crash generation."""
        img = Image.new("RGB", (5, 5), color="red")
        img.save(tmp_path / "valid.png")
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "subdir").mkdir()

        generator = ImageGenerator(
            _run(
                width={"mean": 10, "stddev": 0},
                height={"mean": 10, "stddev": 0},
                source=str(tmp_path),
            )
        )
        assert generator._source_image_paths == [tmp_path / "valid.png"]
        result = generator.generate()
        assert result.startswith("data:image/png;base64,")

    def test_custom_directory_only_non_image_files_raises(self, tmp_path):
        """A directory with only non-image files raises rather than producing nothing."""
        (tmp_path / "notes.txt").write_text("hello")

        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(
                _run(
                    width={"mean": 10, "stddev": 0},
                    height={"mean": 10, "stddev": 0},
                    source=str(tmp_path),
                )
            )

    def test_custom_directory_unreadable_candidate_is_retired_lazily(self, tmp_path):
        (tmp_path / "corrupt.jpg").write_bytes(b"not actually an image")
        Image.new("RGB", (5, 5), color="red").save(tmp_path / "valid.png")

        generator = ImageGenerator(
            _run(
                width={"mean": 10, "stddev": 0},
                height={"mean": 10, "stddev": 0},
                source=str(tmp_path),
                source_sampling="sequential-cycle",
            )
        )

        result = generator.generate()

        assert result.startswith("data:image/png;base64,")
        assert generator._available_source_image_indexes == [1]

    def test_custom_directory_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ImageGenerator(
                _run(
                    width={"mean": 10, "stddev": 0},
                    height={"mean": 10, "stddev": 0},
                    source="/nonexistent/dir",
                )
            )

    def test_custom_directory_is_file_raises(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")

        with pytest.raises(NotADirectoryError, match="is not a directory"):
            ImageGenerator(
                _run(
                    width={"mean": 10, "stddev": 0},
                    height={"mean": 10, "stddev": 0},
                    source=str(file_path),
                )
            )

    def test_custom_directory_empty_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No source images found"):
            ImageGenerator(
                _run(
                    width={"mean": 10, "stddev": 0},
                    height={"mean": 10, "stddev": 0},
                    source=str(empty_dir),
                )
            )


class TestImageGeneratorDisabled:
    """Tests for disabled image generation."""

    def test_disabled_images_skips_init(self):
        generator = ImageGenerator(_run(batch_size=0))
        assert generator.image_config.images_enabled() is False
        assert not hasattr(generator, "_dimensions_rng")
