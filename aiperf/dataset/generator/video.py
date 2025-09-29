# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import math
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

from aiperf.common.config.video_config import VideoConfig
from aiperf.common.enums import VideoSynthType
from aiperf.dataset.generator.base import BaseGenerator


class VideoGenerator(BaseGenerator):
    """A class that generates synthetic videos.

    This class provides methods to create synthetic videos with different patterns
    like moving shapes or grid clocks. The videos are generated in MP4 format and
    returned as base64 encoded strings. Currently only MP4 format is supported.
    """

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.config = config

    def generate(self, *args, **kwargs) -> str:
        """Generate a video with the configured parameters.

        Returns:
            A base64 encoded string of the generated video, or empty string if generation is disabled.
        """
        # Only generate videos if width and height are non-zero
        if self.config.width <= 0 or self.config.height <= 0:
            self.logger.debug(
                "Video generation disabled (width=%d, height=%d)",
                self.config.width,
                self.config.height,
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

        Creates video data using the format specified in config. Currently only MP4 is supported.
        """
        if not frames:
            return ""

        # Validate format
        from aiperf.common.enums import VideoFormat

        if self.config.format != VideoFormat.MP4:
            raise ValueError(
                f"Unsupported video format: {self.config.format}. Only MP4 is supported."
            )

        try:
            # Try OpenCV first (most efficient)
            return self._create_mp4_with_opencv(frames)
        except ImportError:
            self.logger.debug("OpenCV not available, trying ffmpeg subprocess")
            try:
                return self._create_mp4_with_ffmpeg(frames)
            except Exception as e:
                self.logger.error(f"Failed to create MP4 with ffmpeg: {e}")
                raise RuntimeError(
                    "Unable to create MP4 video. Please install OpenCV (pip install opencv-python) or ffmpeg."
                ) from None
        except Exception as e:
            self.logger.error(
                f"Failed to encode video frames to {self.config.format}: {e}"
            )
            raise

    def _create_mp4_with_opencv(self, frames: list[Image.Image]) -> str:
        """Create MP4 data using OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("OpenCV (cv2) not available") from None

        # Create a temporary file for OpenCV (it needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Convert PIL frames to OpenCV format
            height, width = self.config.height, self.config.width
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_path, fourcc, self.config.fps, (width, height))

            for frame in frames:
                # Convert PIL Image to OpenCV format (BGR)
                frame_array = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()

            # Read the MP4 data back
            with open(temp_path, "rb") as f:
                mp4_data = f.read()

            # Clean up temp file
            os.unlink(temp_path)

            # Encode as base64
            base64_data = base64.b64encode(mp4_data).decode("utf-8")
            return f"data:video/{self.config.format.value};base64,{base64_data}"

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _create_mp4_with_ffmpeg(self, frames: list[Image.Image]) -> str:
        """Create MP4 data using ffmpeg subprocess."""

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="aiperf_frames_")

        try:
            # Save frames as PNG files
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame.save(frame_path, "PNG")
                frame_paths.append(frame_path)

            # Use ffmpeg to create MP4 from frames
            output_path = os.path.join(temp_dir, "output.mp4")
            frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

            cmd = [
                "ffmpeg",
                "-y",  # -y to overwrite output file
                "-r",
                str(self.config.fps),  # frame rate
                "-i",
                frame_pattern,  # input pattern
                "-c:v",
                "libx264",  # video codec
                "-pix_fmt",
                "yuv420p",  # pixel format for compatibility
                "-movflags",
                "+faststart",  # optimize for streaming
                output_path,
            ]

            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Read the MP4 data
            with open(output_path, "rb") as f:
                mp4_data = f.read()

            # Encode as base64
            base64_data = base64.b64encode(mp4_data).decode("utf-8")
            return f"data:video/{self.config.format.value};base64,{base64_data}"

        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
