# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, UnidentifiedImageError

from aiperf.dataset import utils


class TestCheckFileExists:
    def test_existing_file(self):
        """Test that no exception is raised for an existing file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            utils.check_file_exists(Path(tmp_file.name))  # Should not raise

    def test_non_existing_file(self):
        """Test that FileNotFoundError is raised for a non-existing file."""
        non_existing_path = Path("/path/that/does/not/exist.txt")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            utils.check_file_exists(non_existing_path)


class TestOpenImage:
    def test_open_valid_image(self):
        """Test opening a valid image file."""
        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            img.save(tmp_file.name, "PNG")
            tmp_path = tmp_file.name

        try:
            opened_img = utils.open_image(tmp_path)
            assert opened_img.size == (10, 10)
            assert opened_img.format == "PNG"
        finally:
            Path(tmp_path).unlink()  # Clean up

    def test_open_non_existing_image(self):
        """Test that FileNotFoundError is raised for non-existing image."""
        with pytest.raises(FileNotFoundError):
            utils.open_image("/path/that/does/not/exist.jpg")

    def test_invalid_image_format(self):
        """Test that RuntimeError is raised for unsupported format."""
        # Create a file that's not a valid image
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not an image")
            tmp_path = tmp_file.name

        try:
            with pytest.raises(UnidentifiedImageError):
                utils.open_image(tmp_path)
        finally:
            Path(tmp_path).unlink()


class TestEncodeImage:
    @pytest.mark.parametrize("format", ["PNG", "JPEG"])
    def test_encode_image(self, format):
        """Test encoding a PNG image to base64."""
        img = Image.new("RGB", (2, 2), color="blue")
        encoded = utils.encode_image(img, format)

        # Verify it's valid base64
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(BytesIO(decoded_bytes))
        assert decoded_img.size == (2, 2)
        assert decoded_img.mode == "RGB"
        assert decoded_img.format == format

    def test_encode_rgba_to_jpeg(self):
        """Test that RGBA image is converted to RGB for JPEG format."""
        img = Image.new("RGBA", (2, 2), color=(255, 0, 0, 128))
        encoded = utils.encode_image(img, "JPEG")

        # Should successfully encode without error
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(BytesIO(decoded_bytes))
        assert decoded_img.mode == "RGB"


class TestLoadJsonStr:
    def test_valid_json(self):
        """Test loading valid JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = utils.load_json_str(json_str)
        assert result == {"key": "value", "number": 42}

    def test_invalid_json(self):
        """Test that RuntimeError is raised for invalid JSON."""
        json_str = '{"key": "value", "invalid": }'
        with pytest.raises(RuntimeError, match="Failed to parse JSON string"):
            utils.load_json_str(json_str)

    def test_validation_function(self):
        """Test loading JSON with a validation function."""
        json_str = '{"required_field": "value"}'

        def validate_func(obj):
            if "required_field" not in obj:
                raise ValueError("Missing required field")
            return obj

        result = utils.load_json_str(json_str, validate_func)
        assert result == {"required_field": "value"}

    def test_validation_failure(self):
        """Test that validation function can raise errors."""
        json_str = '{"wrong_field": "value"}'

        def validate_func(obj):
            if "required_field" not in obj:
                raise ValueError("Missing required field")
            return obj

        with pytest.raises(ValueError, match="Missing required field"):
            utils.load_json_str(json_str, validate_func)


class TestSampleNormal:
    @patch("numpy.random.normal", return_value=5.0)
    def test_basic(self, mock_normal):
        """Test basic normal distribution sampling."""
        result = utils.sample_normal(5.0, 1.0)
        assert result == 5.0

    @patch("numpy.random.normal", side_effect=[0.5, 2.0, 3.0])
    def test_with_bounds(self, mock_normal):
        """Test normal distribution sampling with bounds."""
        result = utils.sample_normal(2.0, 1.0, lower=1.0, upper=4.0)
        assert result == 2.0  # First value in bounds

    @patch("numpy.random.normal", side_effect=[10.0, -5.0, 2.5])
    def test_rejection_sampling(self, mock_normal):
        """Test that rejection sampling works for out-of-bounds values."""
        result = utils.sample_normal(2.0, 1.0, lower=0.0, upper=5.0)
        assert result == 2.5


class TestSamplePositiveNormal:
    @patch("numpy.random.normal", return_value=3.0)
    def test_valid_mean(self, mock_normal):
        """Test sampling from positive normal distribution."""
        result = utils.sample_positive_normal(3.0, 1.0)
        assert result == 3.0

    @patch("numpy.random.normal", return_value=-1.0)
    def test_negative_mean(self, mock_normal):
        """Test that ValueError is raised for negative mean."""
        with pytest.raises(ValueError, match="Mean value.*should be greater than 0"):
            _ = utils.sample_positive_normal(-1.0, 1.0)

    @patch("numpy.random.normal", return_value=2.0)
    def test_zero_mean(self, mock_normal):
        """Test that zero mean raises ValueError."""
        result = utils.sample_positive_normal(0.0, 1.0)
        assert result == 2.0


class TestSamplePositiveNormalInteger:
    @patch("aiperf.dataset.utils.sample_positive_normal", return_value=2.3)
    def test_basic(self, mock_sample_positive_normal):
        """Test sampling positive integer from normal distribution."""
        result = utils.sample_positive_normal_integer(2.0, 1.0)
        assert result == 3  # ceil(2.3)

    @patch("numpy.random.normal", return_value=-1.0)
    def test_negative_mean(self, mock_normal):
        """Test that ValueError is raised for negative mean."""
        with pytest.raises(ValueError, match="Mean value.*should be greater than 0"):
            _ = utils.sample_positive_normal(-1.0, 1.0)

    @patch("aiperf.dataset.utils.sample_positive_normal", return_value=0.1)
    def test_small_value(self, mock_sample_positive_normal):
        """Test that small positive values are ceiled to at least 1."""
        result = utils.sample_positive_normal_integer(0.5, 0.1)
        assert result == 1  # ceil(0.1)
