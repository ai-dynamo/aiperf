# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import ffmpeg
import soundfile as sf
from PIL import Image

from aiperf.common import random_generator as rng
from aiperf.common.enums import VideoAudioCodec, VideoFormat, VideoSynthType
from aiperf.config.dataset import VIDEO_AUDIO_CODEC_MAP
from aiperf.dataset.generator.audio import SUPPORTED_BIT_DEPTHS
from aiperf.dataset.generator.base import BaseGenerator, generate_noise_signal
from aiperf.dataset.generator.ffmpeg_support import (
    check_ffmpeg_availability,
    get_ffmpeg_install_instructions,
)
from aiperf.dataset.generator.video_frames import (
    generate_grid_clock_frames,
    generate_moving_shapes_frames,
    generate_noise_frames,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class VideoGenerator(BaseGenerator):
    """A class that generates synthetic videos.

    This class provides methods to create synthetic videos with different patterns
    like moving shapes or grid clocks. The videos are generated in MP4 or WebM format
    and returned as base64 encoded strings.
    """

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        # Extract video config from dataset config
        dataset_config = run.cfg.get_default_dataset()
        self.video_config = getattr(dataset_config, "video", None)
        self._audio_rng = rng.derive("dataset.video.audio")
        self._noise_rng = rng.derive("dataset.video.noise")

    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg binary is available in the system."""
        return check_ffmpeg_availability()

    def _get_ffmpeg_install_instructions(self) -> str:
        """Get platform-specific FFmpeg installation instructions."""
        return get_ffmpeg_install_instructions()

    def generate(self, *args, **kwargs) -> str:
        """Generate a video with the configured parameters.

        Returns:
            A base64 encoded string of the generated video, or empty string if generation is disabled.
        """
        if self.video_config is None:
            return ""

        # Only generate videos if width and height are non-zero
        if not self.video_config.width or not self.video_config.height:
            self.logger.debug(
                f"Video generation disabled (width={self.video_config.width}, height={self.video_config.height})",
            )
            return ""

        self.logger.debug(
            "Generating video with width=%d, height=%d, duration=%.1fs, fps=%d, type=%s",
            self.video_config.width,
            self.video_config.height,
            self.video_config.duration,
            self.video_config.fps,
            self.video_config.synth_type,
        )

        # Generate frames
        frames = self._generate_frames()

        # Convert frames to video data and return base64
        return self._encode_frames_to_base64(frames)

    def _generate_frames(self) -> list[Image.Image]:
        """Generate frames based on the synthesis type."""
        total_frames = int(self.video_config.duration * self.video_config.fps)
        width, height = self.video_config.width, self.video_config.height
        synth_type = self.video_config.synth_type

        if synth_type == VideoSynthType.MOVING_SHAPES:
            return generate_moving_shapes_frames(total_frames, width, height)
        if synth_type == VideoSynthType.GRID_CLOCK:
            return generate_grid_clock_frames(total_frames, width, height)
        if synth_type == VideoSynthType.NOISE:
            return generate_noise_frames(total_frames, width, height, self._noise_rng)
        raise ValueError(f"Unknown synthesis type: {synth_type}")

    def _encode_frames_to_base64(self, frames: list[Image.Image]) -> str:
        """Convert frames to video data and encode as base64 string.

        Creates video data using the format specified in config. Supports MP4 and WebM formats.
        """
        if not frames:
            return ""

        if self.video_config.format not in [VideoFormat.MP4, VideoFormat.WEBM]:
            raise ValueError(
                f"Unsupported video format: {self.video_config.format}. Only MP4 and WebM are supported."
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
                f"Failed to create {self.video_config.format.upper()} with ffmpeg: {e}"
            )

            # Provide specific error messages based on the error type
            if "No such file or directory" in str(e) or "not found" in str(e):
                raise RuntimeError(
                    "FFmpeg binary not accessible. Please ensure FFmpeg is installed and in your PATH."
                ) from e
            elif "Codec" in str(e) or "codec" in str(e):
                raise RuntimeError(
                    f"Video codec '{self.video_config.codec}' is not supported. "
                    f"Please use a valid FFmpeg codec (e.g., libvpx-vp9, libx264, libx265, h264_nvenc)."
                ) from e
            else:
                raise RuntimeError(
                    f"FFmpeg failed to create video: {e}\n"
                    f"Codec: {self.video_config.codec}, Size: {self.video_config.width}x{self.video_config.height}, FPS: {self.video_config.fps}"
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

    def _generate_audio_data(self) -> bytes:
        """Generate Gaussian noise audio data matching video duration as WAV bytes."""
        num_samples = int(
            self.video_config.duration * self.video_config.audio.sample_rate
        )
        signal = generate_noise_signal(
            self._audio_rng, num_samples, self.video_config.audio.channels
        )

        # Scale to the appropriate bit depth range
        # Note: For 8-bit, we use int16 input and let soundfile convert to PCM_U8
        bit_depth = self.video_config.audio.depth
        numpy_type, subtype = SUPPORTED_BIT_DEPTHS[bit_depth]
        scale_depth = 16 if bit_depth == 8 else bit_depth
        max_val = 2 ** (scale_depth - 1) - 1
        audio_data = (signal * max_val).astype(numpy_type)

        output_buffer = io.BytesIO()
        sf.write(
            output_buffer,
            audio_data,
            self.video_config.audio.sample_rate,
            format="WAV",
            subtype=subtype,
        )
        return output_buffer.getvalue()

    def _resolve_audio_codec(self) -> VideoAudioCodec:
        """Resolve the audio codec, auto-selecting from format if not explicitly set."""
        if self.video_config.audio.codec is not None:
            return self.video_config.audio.codec
        codec = VIDEO_AUDIO_CODEC_MAP.get(self.video_config.format)
        if codec is None:
            raise ValueError(
                f"No default audio codec for format '{self.video_config.format}'. "
                f"Specify --video-audio-codec explicitly."
            )
        return codec

    def _build_ffmpeg_output(
        self,
        video_stream: ffmpeg.Stream,
        output_dest: str,
        output_options: dict,
        audio_dir: Path,
    ) -> ffmpeg.Stream:
        """Build ffmpeg output node, muxing audio if channels > 0.

        Writes a temp WAV file into audio_dir when audio is enabled.
        Caller is responsible for cleaning up audio_dir.
        """
        if self.video_config.audio.channels > 0:
            audio_path = audio_dir / "audio.wav"
            audio_path.write_bytes(self._generate_audio_data())

            audio_stream = ffmpeg.input(str(audio_path))
            merged_options = {
                **output_options,
                "acodec": self._resolve_audio_codec(),
                "shortest": None,
            }
            return ffmpeg.output(
                video_stream, audio_stream, output_dest, **merged_options
            ).overwrite_output()

        return video_stream.output(output_dest, **output_options).overwrite_output()

    def _prepare_frame_for_encoding(self, frame: Image.Image) -> bytes:
        """Prepare frame for encoding."""
        if frame.size != (self.video_config.width, self.video_config.height):
            frame = frame.resize(
                (self.video_config.width, self.video_config.height), Image.LANCZOS
            )
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        return frame.tobytes()

    def _create_video_with_pipes(self, frames: list[Image.Image]) -> str:
        """Create video using pipes via stdin and either stdout or temp file output."""
        temp_dir = Path(tempfile.mkdtemp(prefix="aiperf_pipes_"))
        try:
            # Gather all frame data first to prevent deadlocks due to pipe input/output synchronization issues
            all_data = b"".join(
                self._prepare_frame_for_encoding(frame) for frame in frames
            )

            output_options = {
                "format": self.video_config.format,
                "vcodec": self.video_config.codec,
                "pix_fmt": "yuv420p",
            }

            # Determine output destination based on format
            if self.video_config.format == VideoFormat.MP4:
                # MP4 requires seekable output, use temp file
                output_options["movflags"] = "faststart"
                output_dest = str(temp_dir / f"output.{self.video_config.format}")
            else:
                # WebM and other formats can use pipe output
                output_dest = "pipe:"

            video_stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self.video_config.width}x{self.video_config.height}",
                r=self.video_config.fps,
            )

            pipeline = self._build_ffmpeg_output(
                video_stream, output_dest, output_options, temp_dir
            )
            stdout, _ = pipeline.run(
                input=all_data, capture_stdout=True, capture_stderr=True
            )

            # Read output based on destination
            if output_dest != "pipe:":
                video_data = Path(output_dest).read_bytes()
            else:
                video_data = stdout

            if not video_data:
                raise RuntimeError("FFmpeg produced no output")

            return f"data:video/{self.video_config.format};base64,{base64.b64encode(video_data).decode()}"

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"FFmpeg process failed: {error_msg}") from e
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _create_video_with_temp_files(self, frames: list[Image.Image]) -> str:
        """Create video using temporary files (fallback method)."""
        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="aiperf_frames_"))

        try:
            # Save frames as PNG files
            for i, frame in enumerate(frames):
                # Ensure frame is the correct size
                if frame.size != (self.video_config.width, self.video_config.height):
                    frame = frame.resize(
                        (self.video_config.width, self.video_config.height),
                        Image.LANCZOS,
                    )

                frame_path = temp_dir / f"frame_{i:06d}.png"
                # Use explicit compression settings for deterministic output across platforms
                frame.save(frame_path, "PNG", compress_level=6, optimize=False)

            # Create output file in the same temp directory
            output_path = temp_dir / f"output.{self.video_config.format}"
            frame_pattern = str(temp_dir / "frame_%06d.png")

            # Build output options based on format
            output_options = {
                "format": self.video_config.format,
                "vcodec": self.video_config.codec,
                "pix_fmt": "yuv420p",
            }

            # Add format-specific options
            if self.video_config.format == VideoFormat.MP4:
                output_options["movflags"] = "faststart"

            video_stream = ffmpeg.input(frame_pattern, r=self.video_config.fps)

            pipeline = self._build_ffmpeg_output(
                video_stream, str(output_path), output_options, temp_dir
            )
            pipeline.run(capture_stdout=True, capture_stderr=True)

            # Read the output file
            video_data = output_path.read_bytes()

            if not video_data:
                raise RuntimeError("FFmpeg produced no output")

            # Encode as base64
            base64_data = base64.b64encode(video_data).decode("utf-8")
            return f"data:video/{self.video_config.format};base64,{base64_data}"

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode("utf-8") if e.stderr else "Unknown ffmpeg error"
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"FFmpeg process failed: {error_msg}") from e
        finally:
            # Clean up temporary files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
