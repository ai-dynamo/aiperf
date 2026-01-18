# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from contextlib import ExitStack
from unittest.mock import Mock, mock_open, patch

import pytest
from PIL import Image

from aiperf.common.config import VideoConfig
from aiperf.common.enums import VideoFormat, VideoSynthType
from aiperf.dataset.generator.video import VideoGenerator


@pytest.fixture
def base_config():
    """Base configuration for VideoGenerator tests."""
    return VideoConfig(
        width=64,
        height=64,
        duration=0.5,
        fps=2,
        format=VideoFormat.WEBM,
        codec="libvpx-vp9",
        synth_type=VideoSynthType.MOVING_SHAPES,
    )


class TestVideoGenerator:
    """Test suite for VideoGenerator class."""

    def test_init_with_config(self, base_config):
        """Test VideoGenerator initialization with valid config."""
        generator = VideoGenerator(base_config)
        assert generator.config == base_config

    @pytest.mark.parametrize(
        "ffmpeg_path,expected",
        [
            ("/usr/bin/ffmpeg", True),
            (None, False),
        ],
    )
    def test_check_ffmpeg_availability(self, base_config, ffmpeg_path, expected):
        """Test FFmpeg availability check."""
        with patch("shutil.which", return_value=ffmpeg_path):
            generator = VideoGenerator(base_config)
            assert generator._check_ffmpeg_availability() is expected

    @pytest.mark.parametrize(
        "platform_name,patches_and_expected",
        [
            # Linux distributions
            ("Linux", ({"open_return": io.StringIO("ID=ubuntu")}, "apt")),
            ("Linux", ({"open_return": io.StringIO("ID=fedora")}, "dnf")),
            ("Linux", ({"open_return": io.StringIO("ID=arch")}, "pacman")),
            ("Linux", ({"open_side_effect": FileNotFoundError}, "apt")),  # fallback
            # macOS
            ("Darwin",({"which": lambda x: "/brew" if x == "brew" else None}, "brew install")),
            ("Darwin",({"which": lambda x: "/port" if x == "port" else None}, "port install")),
            ("Darwin", ({"which": lambda x: None}, "brew.sh")),
            # Windows
            ("Windows",({"which": lambda x: "choco" if x == "choco" else None}, "choco install")),
            ("Windows",({"which": lambda x: "winget" if x == "winget" else None}, "winget install")),
            ("Windows", ({"which": lambda x: None}, "ffmpeg.org")),
            # Unknown OS
            ("UnknownOS", ({}, "ffmpeg.org")),
        ],
    )  # fmt: skip
    def test_get_ffmpeg_install_instructions(
        self, base_config, platform_name, patches_and_expected
    ):
        """Test platform-specific FFmpeg installation instructions."""
        patches_dict, expected_keyword = patches_and_expected
        generator = VideoGenerator(base_config)

        with ExitStack() as stack:
            stack.enter_context(patch("platform.system", return_value=platform_name))

            if "open_return" in patches_dict:
                stack.enter_context(
                    patch(
                        "builtins.open",
                        create=True,
                        return_value=patches_dict["open_return"],
                    )
                )
            elif "open_side_effect" in patches_dict:
                stack.enter_context(
                    patch("builtins.open", side_effect=patches_dict["open_side_effect"])
                )

            if "which" in patches_dict:
                stack.enter_context(
                    patch("shutil.which", side_effect=patches_dict["which"])
                )

            instructions = generator._get_ffmpeg_install_instructions()
            assert expected_keyword in instructions
            assert "ffmpeg" in instructions

    def test_generate_with_disabled_video(self):
        """Test that generate returns empty string when video is disabled."""
        config = VideoConfig(
            width=None,
            height=None,
            duration=1.0,
            fps=4,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )
        generator = VideoGenerator(config)
        result = generator.generate()
        assert result == ""

    @pytest.mark.parametrize(
        "synth_type,width,height,duration,fps",
        [
            (VideoSynthType.MOVING_SHAPES, 64, 64, 0.5, 2),
            (VideoSynthType.GRID_CLOCK, 128, 128, 1.0, 4),
        ],
    )
    def test_generate_frames(self, synth_type, width, height, duration, fps):
        """Test frame generation for different synthesis types."""
        config = VideoConfig(
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=synth_type,
        )
        generator = VideoGenerator(config)
        frames = generator._generate_frames()

        expected_frame_count = int(duration * fps)
        assert len(frames) == expected_frame_count

        # Verify all frames are PIL Images with correct dimensions
        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.size == (width, height)
            assert frame.mode == "RGB"

    def test_generate_frames_unknown_type(self, base_config):
        """Test that unknown synthesis type raises ValueError."""
        base_config.synth_type = "unknown_type"
        generator = VideoGenerator(base_config)

        with pytest.raises(ValueError, match="Unknown synthesis type"):
            generator._generate_frames()

    def test_encode_frames_to_base64_empty_frames(self, base_config):
        """Test encoding empty frame list returns empty string."""
        generator = VideoGenerator(base_config)
        result = generator._encode_frames_to_base64([])
        assert result == ""

    def test_encode_frames_to_base64_unsupported_format(self, base_config):
        """Test that unsupported format raises ValueError."""
        base_config.format = Mock(name="UNSUPPORTED", value="unsupported")
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with pytest.raises(ValueError, match="Unsupported video format"):
            generator._encode_frames_to_base64(frames)

    def test_encode_frames_ffmpeg_not_available(self, base_config):
        """Test that encoding fails gracefully when FFmpeg is not available."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with (
            patch.object(generator, "_check_ffmpeg_availability", return_value=False),
            pytest.raises(RuntimeError, match="FFmpeg binary not found"),
        ):
            generator._encode_frames_to_base64(frames)

    def test_encode_frames_codec_error(self, base_config):
        """Test handling of codec errors."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with (
            patch.object(generator, "_check_ffmpeg_availability", return_value=True),
            patch.object(
                generator,
                "_create_video_with_pipes",
                side_effect=Exception("Codec not supported"),
            ),
            patch.object(
                generator,
                "_create_video_with_temp_files",
                side_effect=Exception("Codec not supported"),
            ),
            pytest.raises(RuntimeError, match="[Cc]odec"),
        ):
            generator._encode_frames_to_base64(frames)

    def test_create_video_with_pipes_fallback(self, base_config):
        """Test fallback to temp files when pipes fail."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]
        mock_result = "data:video/webm;base64,FAKE_BASE64"

        with (
            patch.object(
                generator, "_create_video_with_pipes", side_effect=BrokenPipeError
            ),
            patch.object(
                generator, "_create_video_with_temp_files", return_value=mock_result
            ),
        ):
            result = generator._create_video_with_ffmpeg(frames)
            assert result == mock_result

    @pytest.mark.parametrize(
        "video_format,codec,expected_output,expected_movflags",
        [
            (VideoFormat.MP4, "libx264", "/tmp/test_video.mp4", "faststart"),
            (VideoFormat.WEBM, "libvpx-vp9", "pipe:", None),
        ],
    )
    def test_create_video_with_pipes_format_handling(
        self, video_format, codec, expected_output, expected_movflags
    ):
        """Test that MP4 uses temp file output and WebM uses pipe output."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=video_format,
            codec=codec,
            synth_type=VideoSynthType.MOVING_SHAPES,
        )
        generator = VideoGenerator(config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]

        # Use different data for file vs pipe to verify correct reading source
        file_data = b"file_video_data"
        pipe_data = b"pipe_video_data"
        expected_data = file_data if video_format == VideoFormat.MP4 else pipe_data

        with (
            patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg,
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
            patch("builtins.open", mock_open(read_data=file_data)) as mock_file_open,
        ):
            # Setup mock ffmpeg chain
            mock_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.return_value = mock_input
            mock_input.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (pipe_data, b"")

            # Mock temp file for MP4
            mock_file = Mock()
            mock_file.name = "/tmp/test_video.mp4"
            mock_tempfile.return_value = mock_file

            result = generator._create_video_with_pipes(frames)

            # Verify output destination
            output_call = mock_input.output.call_args
            assert output_call[0][0] == expected_output

            # Verify movflags
            if expected_movflags:
                assert output_call[1]["movflags"] == expected_movflags
            else:
                assert "movflags" not in output_call[1]

            # Verify correct reading source was used
            if video_format == VideoFormat.MP4:
                # MP4 should read from temp file
                mock_file_open.assert_called_once_with("/tmp/test_video.mp4", "rb")
            else:
                # WebM should NOT read from file (use pipe data from stdout)
                mock_file_open.assert_not_called()

            # Verify result contains data from correct source
            expected_base64 = base64.b64encode(expected_data).decode()
            assert result.startswith(f"data:video/{video_format};base64,")
            assert expected_base64 in result
