# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import soundfile as sf
from PIL import Image

from aiperf.common.config import VideoAudioConfig, VideoConfig
from aiperf.common.enums import VideoAudioCodec, VideoFormat, VideoSynthType
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
            (VideoSynthType.NOISE, 64, 64, 0.5, 2),
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
        "video_format,codec,expected_movflags",
        [
            (VideoFormat.MP4, "libx264", "faststart"),
            (VideoFormat.WEBM, "libvpx-vp9", None),
        ],
    )
    def test_create_video_with_pipes_format_handling(
        self, video_format, codec, expected_movflags
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

        file_data = b"file_video_data"
        pipe_data = b"pipe_video_data"
        expected_data = file_data if video_format == VideoFormat.MP4 else pipe_data
        temp_dir = "/tmp/aiperf_pipes_test"

        with (
            patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg,
            patch("tempfile.mkdtemp", return_value=temp_dir),
            patch.object(Path, "read_bytes", return_value=file_data),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.rmtree"),
        ):
            mock_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.return_value = mock_input
            mock_input.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (pipe_data, b"")

            result = generator._create_video_with_pipes(frames)

            # Verify output destination
            output_call = mock_input.output.call_args
            if video_format == VideoFormat.MP4:
                assert output_call[0][0] == f"{temp_dir}/output.mp4"
            else:
                assert output_call[0][0] == "pipe:"

            # Verify movflags
            if expected_movflags:
                assert output_call[1]["movflags"] == expected_movflags
            else:
                assert "movflags" not in output_call[1]

            # Verify result contains data from correct source
            expected_base64 = base64.b64encode(expected_data).decode()
            assert result.startswith(f"data:video/{video_format};base64,")
            assert expected_base64 in result


@pytest.fixture
def audio_config():
    """VideoConfig with audio enabled (channels=1)."""
    return VideoConfig(
        width=64,
        height=64,
        duration=0.5,
        fps=2,
        format=VideoFormat.WEBM,
        codec="libvpx-vp9",
        synth_type=VideoSynthType.MOVING_SHAPES,
        audio=VideoAudioConfig(sample_rate=44100, channels=1),
    )


class TestVideoGeneratorAudio:
    """Test suite for VideoGenerator audio muxing."""

    def test_generate_audio_data_produces_valid_wav(self, audio_config):
        """Audio data is valid WAV with non-empty content."""
        generator = VideoGenerator(audio_config)
        wav_bytes = generator._generate_audio_data()

        assert len(wav_bytes) > 44  # WAV header is 44 bytes minimum
        assert wav_bytes[:4] == b"RIFF"  # WAV magic bytes

        # Verify it's readable by soundfile
        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 44100
        assert len(data) > 0

    def test_generate_audio_data_duration_matches_video(self, audio_config):
        """Audio duration approximately matches video duration."""
        generator = VideoGenerator(audio_config)
        wav_bytes = generator._generate_audio_data()

        data, sr = sf.read(io.BytesIO(wav_bytes))
        audio_duration = len(data) / sr
        assert abs(audio_duration - audio_config.duration) < 0.01

    @pytest.mark.parametrize("channels", [1, 2])
    def test_generate_audio_data_channels(self, channels):
        """Mono produces 1D array, stereo produces 2D with shape[1]==2."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=channels),
        )
        generator = VideoGenerator(config)
        wav_bytes = generator._generate_audio_data()

        data, _ = sf.read(io.BytesIO(wav_bytes))
        if channels == 1:
            assert data.ndim == 1
        else:
            assert data.ndim == 2
            assert data.shape[1] == 2

    @pytest.mark.parametrize(
        "video_format,expected_codec",
        [
            (VideoFormat.WEBM, VideoAudioCodec.LIBVORBIS),
            (VideoFormat.MP4, VideoAudioCodec.AAC),
        ],
    )
    def test_resolve_audio_codec_auto_select(self, video_format, expected_codec):
        """Auto-selects correct codec based on video format."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=video_format,
            codec="libvpx-vp9" if video_format == VideoFormat.WEBM else "libx264",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, codec=None),
        )
        generator = VideoGenerator(config)
        assert generator._resolve_audio_codec() == expected_codec

    def test_resolve_audio_codec_unsupported_format_raises(self):
        """Unsupported format without explicit codec raises ValueError."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, codec=None),
        )
        generator = VideoGenerator(config)
        # Simulate an unsupported format by patching the config
        generator.config.format = "avi"
        with pytest.raises(ValueError, match="No default audio codec"):
            generator._resolve_audio_codec()

    @pytest.mark.parametrize(
        "explicit_codec", [VideoAudioCodec.LIBOPUS, VideoAudioCodec.AAC]
    )
    def test_resolve_audio_codec_explicit_override(self, explicit_codec):
        """Explicit codec beats auto-select."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, codec=explicit_codec),
        )
        generator = VideoGenerator(config)
        assert generator._resolve_audio_codec() == explicit_codec

    def test_audio_disabled_no_acodec_in_ffmpeg(self, base_config):
        """When audio is disabled, ffmpeg.output is not called with acodec."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]

        with patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg:
            mock_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.return_value = mock_input
            mock_input.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (b"video_data", b"")

            generator._create_video_with_pipes(frames)

            # Verify output was called on the stream (not ffmpeg.output with two streams)
            output_call = mock_input.output.call_args
            assert "acodec" not in output_call[1]

    def test_audio_enabled_adds_audio_stream(self, audio_config):
        """When audio is enabled, ffmpeg.output is called with both streams and acodec."""
        generator = VideoGenerator(audio_config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]

        with (
            patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg,
            patch("tempfile.mkdtemp", return_value="/tmp/aiperf_pipes_test"),
            patch.object(Path, "write_bytes"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.rmtree"),
        ):
            mock_video_input = Mock(name="video_input")
            mock_audio_input = Mock(name="audio_input")
            mock_output = Mock()

            # First call is video pipe, second is audio file
            mock_ffmpeg.input.side_effect = [mock_video_input, mock_audio_input]
            mock_ffmpeg.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (b"video_data", b"")

            generator._create_video_with_pipes(frames)

            # ffmpeg.output should be called with both streams
            mock_ffmpeg.output.assert_called_once()
            call_args = mock_ffmpeg.output.call_args
            assert call_args[0][0] is mock_video_input
            assert call_args[0][1] is mock_audio_input
            assert "acodec" in call_args[1]

    def test_audio_generation_deterministic(self, audio_config):
        """Same seed produces same WAV bytes."""
        generator1 = VideoGenerator(audio_config)
        generator2 = VideoGenerator(audio_config)
        assert generator1._generate_audio_data() == generator2._generate_audio_data()

    def test_pipes_cleans_temp_dir(self, audio_config):
        """Temp directory is cleaned up via shutil.rmtree in finally block."""
        generator = VideoGenerator(audio_config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]
        temp_dir = "/tmp/aiperf_pipes_test"

        with (
            patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg,
            patch("tempfile.mkdtemp", return_value=temp_dir),
            patch.object(Path, "write_bytes"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_video_input = Mock()
            mock_audio_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.side_effect = [mock_video_input, mock_audio_input]
            mock_ffmpeg.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (b"video_data", b"")

            generator._create_video_with_pipes(frames)

            mock_rmtree.assert_called_once_with(Path(temp_dir))

    def test_temp_files_with_audio(self):
        """Temp files path muxes audio correctly when enabled."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1),
        )
        generator = VideoGenerator(config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]

        with (
            patch("aiperf.dataset.generator.video.ffmpeg") as mock_ffmpeg,
            patch("tempfile.mkdtemp", return_value="/tmp/aiperf_frames_test"),
            patch.object(Image.Image, "save"),
            patch.object(Path, "write_bytes"),
            patch.object(Path, "read_bytes", return_value=b"video_data"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.rmtree"),
        ):
            mock_video_input = Mock(name="video_input")
            mock_audio_input = Mock(name="audio_input")
            mock_output = Mock()

            mock_ffmpeg.input.side_effect = [mock_video_input, mock_audio_input]
            mock_ffmpeg.output.return_value = mock_output
            mock_output.overwrite_output.return_value = mock_output
            mock_output.run.return_value = (b"", b"")

            generator._create_video_with_temp_files(frames)

            # ffmpeg.output should be called with both streams
            mock_ffmpeg.output.assert_called_once()
            call_args = mock_ffmpeg.output.call_args
            assert call_args[0][0] is mock_video_input
            assert call_args[0][1] is mock_audio_input
            assert "acodec" in call_args[1]


class TestVideoAudioBitDepth:
    """Test suite for video audio bit depth support, including 8-bit unsigned WAV."""

    @pytest.mark.parametrize(
        "bit_depth,expected_subtype",
        [
            (8, "PCM_U8"),
            (16, "PCM_16"),
            (24, "PCM_24"),
            (32, "PCM_32"),
        ],
    )
    def test_video_audio_bit_depth_produces_correct_subtype(
        self, bit_depth, expected_subtype
    ):
        """Video audio uses correct PCM subtype for each bit depth.

        Regression test for 8-bit audio bug where PCM_S8 was incorrectly used
        instead of PCM_U8. WAV format requires unsigned 8-bit audio.
        """
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.1,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, depth=bit_depth),
        )
        generator = VideoGenerator(config)
        wav_bytes = generator._generate_audio_data()

        with io.BytesIO(wav_bytes) as f:
            info = sf.info(f)
            assert info.subtype == expected_subtype

    @pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
    def test_video_audio_bit_depth_produces_valid_audio(self, bit_depth):
        """All supported bit depths produce valid, readable WAV audio."""
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.1,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, depth=bit_depth),
        )
        generator = VideoGenerator(config)
        wav_bytes = generator._generate_audio_data()

        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert len(data) > 0
        assert sr == 44100  # default sample rate

    def test_video_audio_8bit_is_unsigned(self):
        """8-bit video audio values are in unsigned range (0-255 centered at 128).

        This is a specific regression test for the PCM_U8 bug fix.
        """
        config = VideoConfig(
            width=64,
            height=64,
            duration=0.1,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
            audio=VideoAudioConfig(channels=1, depth=8),
        )
        generator = VideoGenerator(config)
        wav_bytes = generator._generate_audio_data()

        with io.BytesIO(wav_bytes) as f:
            info = sf.info(f)
            assert info.subtype == "PCM_U8"
            assert info.format == "WAV"
