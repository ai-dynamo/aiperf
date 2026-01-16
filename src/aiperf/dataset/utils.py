# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from aiperf.common.enums import AudioFormat, ImageFormat, VideoFormat


def check_file_exists(filename: Path) -> None:
    """Verifies that the file exists.

    Args:
        filename : The file path to verify.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filename.exists():
        raise FileNotFoundError(f"The file '{filename}' does not exist.")


def open_image(filename: str) -> Image:
    """Opens an image file.

    Args:
        filename : The file path to open.

    Returns:
        The opened PIL Image object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    check_file_exists(Path(filename))
    img = Image.open(filename)

    if img.format is None:
        raise RuntimeError(f"Failed to determine image format of '{filename}'.")

    if img.format.upper() not in list(ImageFormat):
        raise RuntimeError(
            f"'{img.format}' is not one of the supported image formats: "
            f"{', '.join(ImageFormat)}"
        )
    return img


def encode_image(img: Image, format: str) -> str:
    """Encodes an image into base64 encoded string.

    Args:
        img: The PIL Image object to encode.
        format: The image format to use (e.g., "JPEG", "PNG").

    Returns:
        A base64 encoded string representation of the image.
    """
    # JPEG does not support P or RGBA mode (commonly used for PNG) so it needs
    # to be converted to RGB before an image can be saved as JPEG format.
    if format == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")

    buffer = BytesIO()
    # Use explicit compression settings to ensure deterministic output across platforms
    # (macOS and Linux may have different library versions that produce different output)
    if format == "PNG":
        # PNG: Explicit compress_level and disable optimize to ensure consistent zlib compression
        img.save(buffer, format=format, compress_level=6, optimize=False)
    elif format == "JPEG":
        # JPEG: Explicit quality and subsampling to ensure consistent libjpeg output
        img.save(buffer, format=format, quality=85, subsampling=0)
    else:
        img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def open_audio(filename: str) -> tuple[bytes, AudioFormat]:
    """Opens an audio file and returns its bytes and format.

    Args:
        filename: The file path to open.

    Returns:
        A tuple of (audio_bytes, format) where format is 'wav' or 'mp3'.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the audio format is unsupported.
    """
    file_path = Path(filename)
    check_file_exists(file_path)

    # Determine format from extension
    suffix = file_path.suffix.lower()
    if suffix == ".wav":
        audio_format = AudioFormat.WAV
    elif suffix == ".mp3":
        audio_format = AudioFormat.MP3
    else:
        raise RuntimeError(
            f"'{suffix}' is not one of the supported audio formats: "
            f"{', '.join([str(f) for f in AudioFormat])}"
        )

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes, audio_format


def encode_audio(audio_bytes: bytes, format: AudioFormat) -> str:
    """Encodes audio bytes into base64 string with format prefix.

    Args:
        audio_bytes: The audio data as bytes.
        format: The audio format (e.g., AudioFormat.WAV, AudioFormat.MP3).

    Returns:
        A string in the format "format,base64_encoded_data".
    """
    base64_data = base64.b64encode(audio_bytes).decode("utf-8")
    return f"{format.lower()},{base64_data}"


def open_video(filename: str) -> tuple[bytes, VideoFormat]:
    """Opens a video file and returns its bytes and format.

    Args:
        filename: The file path to open.

    Returns:
        A tuple of (video_bytes, format) where format is VideoFormat.MP4.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the video format is unsupported.
    """
    file_path = Path(filename)
    check_file_exists(file_path)

    # Determine format from extension
    suffix = file_path.suffix.lower()
    if suffix == ".mp4":
        video_format = VideoFormat.MP4
    elif suffix == ".webm":
        video_format = VideoFormat.WEBM
    else:
        raise RuntimeError(
            f"'{suffix}' is not one of the supported video formats: "
            f"{', '.join([str(f) for f in VideoFormat])}"
        )

    with open(filename, "rb") as f:
        video_bytes = f.read()

    return video_bytes, video_format


def encode_video(video_bytes: bytes, format: VideoFormat) -> str:
    """Encodes video bytes into base64 data URL.

    Args:
        video_bytes: The video data as bytes.
        format: The video format (e.g., VideoFormat.MP4).

    Returns:
        A data URL string in the format "data:video/format;base64,encoded_data".
    """
    base64_data = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/{format.lower()};base64,{base64_data}"
