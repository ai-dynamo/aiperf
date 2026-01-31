# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
from io import BytesIO
from pathlib import Path

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
            opened_img.close()
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
