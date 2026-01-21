# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.composer.custom import CustomDatasetComposer


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filename = None

    def _create_file(content_lines):
        nonlocal filename
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filename = f.name
        return filename

    yield _create_file

    # Cleanup all created files
    if filename:
        Path(filename).unlink(missing_ok=True)


@pytest.fixture
def create_user_config_and_composer(mock_tokenizer_cls):
    """Create a UserConfig and CustomDatasetComposer for testing."""

    def _create():
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                file="test_data.jsonl",
                conversation=ConversationConfig(num=5),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=10, stddev=2),
                ),
            ),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        composer = CustomDatasetComposer(config, tokenizer)
        return config, composer

    return _create


@pytest.fixture
def default_user_config() -> UserConfig:
    """Create a default UserConfig for testing."""
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
def test_images(tmp_path):
    """Create temporary test images copied from source assets.

    Returns:
        A dictionary mapping image names to their temporary file paths.
    """
    # Get the source images directory
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")

    # Get some actual image files
    source_images = list(source_images_dir.glob("*.jpg"))[:4]

    if not source_images:
        # Create a minimal synthetic JPEG image if no source images found
        from PIL import Image

        synthetic_path = tmp_path / "image1.jpg"
        img = Image.new("RGB", (1, 1), color="red")
        img.save(synthetic_path, format="JPEG")
        return {"image1.jpg": str(synthetic_path)}

    # Create temporary copies preserving original file extensions
    image_map = {}
    for i, source_img in enumerate(source_images, 1):
        # Preserve the original file extension to avoid MIME/encoder mismatches
        dest_filename = f"image{i}{source_img.suffix}"
        dest_path = tmp_path / dest_filename
        shutil.copy(source_img, dest_path)
        image_map[dest_filename] = str(dest_path)

    return image_map


@pytest.fixture
def create_test_image(tmp_path):
    """Create a single test image copied from source assets.

    Returns:
        A function that creates a test image with the given name.
    """
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")
    source_images = list(source_images_dir.glob("*.jpg"))

    def _create_image(name: str = "test_image.jpg"):
        from PIL import Image

        dest_path = tmp_path / name
        requested_ext = Path(name).suffix.lower()

        if source_images:
            # Load the source image and save it in the requested format
            img = Image.open(source_images[0])
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                # Default to JPEG
                img.save(dest_path, format="JPEG")
        else:
            # Create a minimal synthetic image matching the requested format
            img = Image.new("RGB", (1, 1), color="red")
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                # Default to JPEG
                img.save(dest_path, format="JPEG")

        return str(dest_path)

    return _create_image


@pytest.fixture
def create_test_audio(tmp_path):
    """Create test audio files (WAV and MP3).

    Returns:
        A function that creates a test audio file with the given name.
    """
    import wave

    import numpy as np

    def _create_audio(name: str = "test_audio.wav"):
        dest_path = tmp_path / name

        # Generate simple sine wave audio
        sample_rate = 16000
        duration = 0.1  # 100ms
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(str(dest_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return str(dest_path)

    return _create_audio


@pytest.fixture
def create_test_video(tmp_path):
    """Create test video files (MP4).

    Returns:
        A function that creates a test video file with the given name.
    """
    from PIL import Image, ImageDraw

    def _create_video(name: str = "test_video.mp4"):
        dest_path = tmp_path / name

        # Try using ffmpeg-python if available, otherwise create a minimal MP4
        try:
            import tempfile

            import ffmpeg

            # Create a few simple frames
            temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_")
            for i in range(3):
                img = Image.new("RGB", (64, 64), (i * 80, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((10, 25), f"F{i}", fill=(255, 255, 255))
                img.save(f"{temp_frame_dir}/frame_{i:03d}.png")

            # Use ffmpeg to create video
            (
                ffmpeg.input(f"{temp_frame_dir}/frame_%03d.png", framerate=1)
                .output(str(dest_path), vcodec="libx264", pix_fmt="yuv420p", t=1)
                .overwrite_output()
                .run(quiet=True)
            )

            for file in Path(temp_frame_dir).glob("*.png"):
                file.unlink()
            Path(temp_frame_dir).rmdir()

        except (ImportError, Exception):
            # Fallback: create a minimal valid MP4 file
            # This is a minimal MP4 with just headers (won't play but is valid for testing)
            minimal_mp4 = bytes.fromhex(
                "000000186674797069736f6d0000020069736f6d69736f32617663310000"
                "0008667265650000002c6d6461740000001c6d6f6f7600000000006d7668"
                "6400000000000000000000000000000001000000"
            )
            with open(dest_path, "wb") as f:
                f.write(minimal_mp4)

        return str(dest_path)

    return _create_video
