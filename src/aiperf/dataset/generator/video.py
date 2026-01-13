# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import math
import os
import platform
import shutil
import tempfile

import ffmpeg
from PIL import Image, ImageDraw

from aiperf.common.config.video_config import VideoConfig
from aiperf.common.enums import VideoFormat, VideoSynthType
from aiperf.dataset.generator.base import BaseGenerator


class VideoGenerator(BaseGenerator):
    """A class that generates synthetic videos.

    This class provides methods to create synthetic videos with different patterns
    like moving shapes or grid clocks. The videos are generated in MP4 or WebM format
    and returned as base64 encoded strings.
    """

    def __init__(self, config: VideoConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg binary is available in the system."""
        return shutil.which("ffmpeg") is not None

    def _get_ffmpeg_install_instructions(self) -> str:
        """Get platform-specific FFmpeg installation instructions."""
        system = platform.system().lower()

        if system == "linux":
            # Try to detect the distribution
            try:
                with open("/etc/os-release") as f:
                    os_info = f.read().lower()
                if "ubuntu" in os_info or "debian" in os_info:
                    return "sudo apt update && sudo apt install ffmpeg"
                elif "fedora" in os_info or "rhel" in os_info or "centos" in os_info:
                    return "sudo dnf install ffmpeg  # or: sudo yum install ffmpeg"
                elif "arch" in os_info:
                    return "sudo pacman -S ffmpeg"
            except (FileNotFoundError, PermissionError, OSError):
                pass
            return "sudo apt install ffmpeg  # (Ubuntu/Debian) or use your distribution's package manager"
        elif system == "darwin":  # macOS
            if shutil.which("brew"):
                return "brew install ffmpeg"
            elif shutil.which("port"):
                return "sudo port install ffmpeg"
            else:
                return (
                    "brew install ffmpeg  # (install Homebrew first: https://brew.sh)"
                )
        elif system == "windows":
            if shutil.which("choco"):
                return "choco install ffmpeg"
            elif shutil.which("winget"):
                return "winget install ffmpeg"
            else:
                return "Download from https://ffmpeg.org/download.html or use 'choco install ffmpeg'"
        else:
            return "Install FFmpeg using your system's package manager or download from https://ffmpeg.org"

    def generate(self, *args, **kwargs) -> str:
        """Generate a video with the configured parameters.

        Returns:
            A base64 encoded string of the generated video, or empty string if generation is disabled.
        """
        # Only generate videos if width and height are non-zero
        if not self.config.width or not self.config.height:
            self.logger.debug(
                f"Video generation disabled (width={self.config.width}, height={self.config.height})",
            )
            return ""

        self.logger.debug(
            "Generating video with width=%d, height=%d, duration=%.1fs, fps=%d, type=%s",
            self.config.width,
            self.config.height,
            self.config.duration,
            self.config.fps,
            self.config.synth_type,
        )

        # Generate frames
        frames = self._generate_frames()

        # Convert frames to video data and return base64
        return self._encode_frames_to_base64(frames)

    def _generate_frames(self) -> list[Image.Image]:
        """Generate frames based on the synthesis type."""
        total_frames = int(self.config.duration * self.config.fps)
        frames = []

        if self.config.synth_type == VideoSynthType.MOVING_SHAPES:
            frames = self._generate_moving_shapes_frames(total_frames)
        elif self.config.synth_type == VideoSynthType.GRID_CLOCK:
            frames = self._generate_grid_clock_frames(total_frames)
        else:
            raise ValueError(f"Unknown synthesis type: {self.config.synth_type}")

        return frames

    def _generate_moving_shapes_frames(self, total_frames: int) -> list[Image.Image]:
        """Generate frames with moving geometric shapes."""
        frames = []
        width, height = self.config.width, self.config.height

        # Create multiple moving objects
        shapes = [
            {
                "type": "circle",
                "color": (255, 0, 0),  # Red circle
                "size": 30,
                "start_x": 0,
                "start_y": height // 2,
                "dx": width / total_frames * 2,  # Move across screen in half duration
                "dy": 0,
            },
            {
                "type": "rectangle",
                "color": (0, 255, 0),  # Green rectangle
                "size": 25,
                "start_x": width // 2,
                "start_y": 0,
                "dx": 0,
                "dy": height / total_frames * 2,  # Move down
            },
            {
                "type": "circle",
                "color": (0, 0, 255),  # Blue circle
                "size": 20,
                "start_x": width,
                "start_y": height,
                "dx": -width / total_frames * 1.5,  # Move diagonally
                "dy": -height / total_frames * 1.5,
            },
        ]

        for frame_num in range(total_frames):
            # Create black background
            img = Image.new("RGB", (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Draw each shape at its current position
            for shape in shapes:
                x = shape["start_x"] + shape["dx"] * frame_num
                y = shape["start_y"] + shape["dy"] * frame_num

                # Wrap around screen edges
                x = x % width
                y = y % height

                size = shape["size"]
                color = shape["color"]

                if shape["type"] == "circle":
                    draw.ellipse(
                        [x - size // 2, y - size // 2, x + size // 2, y + size // 2],
                        fill=color,
                    )
                elif shape["type"] == "rectangle":
                    draw.rectangle(
                        [x - size // 2, y - size // 2, x + size // 2, y + size // 2],
                        fill=color,
                    )

            frames.append(img)

        return frames

    def _generate_grid_clock_frames(self, total_frames: int) -> list[Image.Image]:
        """Generate frames with a grid and clock-like animation."""
        frames = []
        width, height = self.config.width, self.config.height

        for frame_num in range(total_frames):
            # Create dark gray background
            img = Image.new("RGB", (width, height), (32, 32, 32))
            draw = ImageDraw.Draw(img)

            # Draw grid
            grid_size = 32
            for x in range(0, width, grid_size):
                draw.line([(x, 0), (x, height)], fill=(64, 64, 64), width=1)
            for y in range(0, height, grid_size):
                draw.line([(0, y), (width, y)], fill=(64, 64, 64), width=1)

            # Draw clock hands
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4

            # Frame-based rotation
            angle = (frame_num / total_frames) * 2 * math.pi

            # Hour hand (slower)
            hour_angle = angle / 12
            hour_x = center_x + radius * 0.6 * math.cos(hour_angle - math.pi / 2)
            hour_y = center_y + radius * 0.6 * math.sin(hour_angle - math.pi / 2)
            draw.line(
                [(center_x, center_y), (hour_x, hour_y)], fill=(255, 255, 0), width=3
            )

            # Minute hand
            min_x = center_x + radius * 0.9 * math.cos(angle - math.pi / 2)
            min_y = center_y + radius * 0.9 * math.sin(angle - math.pi / 2)
            draw.line(
                [(center_x, center_y), (min_x, min_y)], fill=(255, 255, 255), width=2
            )

            # Clock face circle
            draw.ellipse(
                [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                outline=(128, 128, 128),
                width=2,
            )

            # Center dot
            draw.ellipse(
                [center_x - 3, center_y - 3, center_x + 3, center_y + 3],
                fill=(255, 0, 0),
            )

            # Add frame number in corner
            draw.text((10, 10), f"Frame {frame_num}", fill=(255, 255, 255))

            frames.append(img)

        return frames

    def _encode_frames_to_base64(self, frames: list[Image.Image]) -> str:
        """Convert frames to video data and encode as base64 string.

        Creates video data using the format specified in config. Supports MP4 and WebM formats.
        """
        if not frames:
            return ""

        # Validate format
        from aiperf.common.enums import VideoFormat

        if self.config.format not in [VideoFormat.MP4, VideoFormat.WEBM]:
            raise ValueError(
                f"Unsupported video format: {self.config.format}. Only MP4 and WebM are supported."
            )

        # Check if FFmpeg is available before proceeding
        if not self._check_ffmpeg_availability():
            install_cmd = self._get_ffmpeg_install_instructions()
            raise RuntimeError(
                f"FFmpeg binary not found. Please install FFmpeg:\n\n"
                f"  Recommended: {install_cmd}\n\n"
                f"  Alternative: conda install -c conda-forge ffmpeg\n\n"
                f"After installation, restart your terminal and try again."
            )

        try:
            return self._create_video_with_ffmpeg(frames)
        except Exception as e:
            self.logger.error(
                f"Failed to create {self.config.format.upper()} with ffmpeg: {e}"
            )

            # Provide specific error messages based on the error type
            if "No such file or directory" in str(e) or "not found" in str(e):
                raise RuntimeError(
                    "FFmpeg binary not accessible. Please ensure FFmpeg is installed and in your PATH."
                ) from e
            elif "Codec" in str(e) or "codec" in str(e):
                raise RuntimeError(
                    f"Video codec '{self.config.codec}' is not supported. "
                    f"Please use a valid FFmpeg codec (e.g., libvpx-vp9, libx264, libx265, h264_nvenc)."
                ) from e
            else:
                raise RuntimeError(
                    f"FFmpeg failed to create video: {e}\n"
                    f"Codec: {self.config.codec}, Size: {self.config.width}x{self.config.height}, FPS: {self.config.fps}"
                ) from e

    def _create_video_with_ffmpeg(self, frames: list[Image.Image]) -> str:
        """Create video data using ffmpeg-python with improved error handling."""

        try:
            # First try the in-memory approach
            return self._create_video_with_pipes(frames)
        except (BrokenPipeError, OSError, RuntimeError) as e:
            self.logger.warning(
                f"Pipe method failed ({e}), falling back to temporary file method"
            )
            # Fall back to temporary file approach if pipes fail
            return self._create_video_with_temp_files(frames)

    def _prepare_frame_for_encoding(self, frame: Image.Image) -> bytes:
        """Prepare frame for encoding."""
        if frame.size != (self.config.width, self.config.height):
            frame = frame.resize((self.config.width, self.config.height), Image.LANCZOS)
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        return frame.tobytes()

    def _create_video_with_pipes(self, frames: list[Image.Image]) -> str:
        """Create video using pipes via stdin and either stdout or temp file output."""
        temp_output_path = None
        try:
            # Gather all frame data first to prevent deadlocks due to pipe input/output synchronization issues
            all_data = b"".join(
                self._prepare_frame_for_encoding(frame) for frame in frames
            )

            output_options = {
                "format": self.config.format,
                "vcodec": self.config.codec,
                "pix_fmt": "yuv420p",
            }

            # Determine output destination based on format
            if self.config.format == VideoFormat.MP4:
                # MP4 requires seekable output, use temp file
                output_options["movflags"] = "faststart"
                # Ignore SIM115 warning as we are closing the file in the finally block
                temp_output = tempfile.NamedTemporaryFile(  # noqa: SIM115
                    suffix=f".{self.config.format}", delete=False
                )
                temp_output_path = temp_output.name
                temp_output.close()
                output_dest = temp_output_path
            else:
                # WebM and other formats can use pipe output
                output_dest = "pipe:"

            # Run ffmpeg
            stdout, _ = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{self.config.width}x{self.config.height}",
                    r=self.config.fps,
                )
                .output(output_dest, **output_options)
                .overwrite_output()
                .run(input=all_data, capture_stdout=True, capture_stderr=True)
            )

            # Read output based on destination
            if temp_output_path is not None:
                with open(temp_output_path, "rb") as f:
                    video_data = f.read()
            else:
                video_data = stdout

            if not video_data:
                raise RuntimeError("FFmpeg produced no output")

            return f"data:video/{self.config.format};base64,{base64.b64encode(video_data).decode()}"

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"FFmpeg process failed: {error_msg}") from e
        finally:
            # Clean up temporary output file if it was created
            if temp_output_path and os.path.exists(temp_output_path):
                try:
                    os.unlink(temp_output_path)
                except OSError as e:
                    self.logger.debug(
                        f"Failed to remove temporary output file: {temp_output_path}: {e!r}"
                    )

    def _create_video_with_temp_files(self, frames: list[Image.Image]) -> str:
        """Create video using temporary files (fallback method)."""
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="aiperf_frames_")

        try:
            # Save frames as PNG files
            for i, frame in enumerate(frames):
                # Ensure frame is the correct size
                if frame.size != (self.config.width, self.config.height):
                    frame = frame.resize(
                        (self.config.width, self.config.height), Image.LANCZOS
                    )

                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                # Use explicit compression settings for deterministic output across platforms
                frame.save(frame_path, "PNG", compress_level=6, optimize=False)

            # Create output file in the same temp directory
            file_ext = self.config.format
            output_path = os.path.join(temp_dir, f"output.{file_ext}")
            frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

            # Build output options based on format
            output_options = {
                "format": self.config.format,
                "vcodec": self.config.codec,
                "pix_fmt": "yuv420p",
            }

            # Add format-specific options
            if self.config.format == VideoFormat.MP4:
                output_options["movflags"] = "faststart"

            # Use ffmpeg to create video from frames
            _ = (
                ffmpeg.input(frame_pattern, r=self.config.fps)
                .output(output_path, **output_options)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Read the output file
            with open(output_path, "rb") as f:
                video_data = f.read()

            if not video_data:
                raise RuntimeError("FFmpeg produced no output")

            # Encode as base64
            base64_data = base64.b64encode(video_data).decode("utf-8")
            return f"data:video/{self.config.format};base64,{base64_data}"

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode("utf-8") if e.stderr else "Unknown ffmpeg error"
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"FFmpeg process failed: {error_msg}") from e
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
