# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.enums import VideoAudioCodec, VideoFormat
from aiperf.config import (
    VideoAudioConfig,
    VideoConfig,
)


class TestVideoAudioConfigDefaults:
    """Test VideoAudioConfig default values."""

    def test_video_audio_config_defaults(self):
        """Default values match expected defaults."""
        config = VideoAudioConfig()
        assert config.sample_rate == 44100
        assert config.channels == 0
        assert config.codec is None

    def test_video_audio_config_disabled_by_default(self):
        """Default channels=0 means audio is disabled."""
        config = VideoAudioConfig()
        assert config.channels == 0


class TestVideoAudioConfigValidation:
    """Test VideoAudioConfig field validation."""

    @pytest.mark.parametrize("channels", [0, 1, 2])
    def test_video_audio_config_valid_channels(self, channels):
        """Channels 0, 1, and 2 are valid."""
        config = VideoAudioConfig(channels=channels)
        assert config.channels == channels

    @pytest.mark.parametrize("channels", [3, -1])
    def test_video_audio_config_invalid_channels(self, channels):
        """Channels outside 0-2 raise ValidationError."""
        with pytest.raises(ValidationError):
            VideoAudioConfig(channels=channels)

    @pytest.mark.parametrize("sample_rate", [8000, 44100, 96000])
    def test_video_audio_config_valid_sample_rate(self, sample_rate):
        """Sample rates within 8000-96000 are valid."""
        config = VideoAudioConfig(sample_rate=sample_rate)
        assert config.sample_rate == sample_rate

    @pytest.mark.parametrize("sample_rate", [7999, 96001, 0, -1])
    def test_video_audio_config_invalid_sample_rate(self, sample_rate):
        """Sample rates outside 8000-96000 raise ValidationError."""
        with pytest.raises(ValidationError):
            VideoAudioConfig(sample_rate=sample_rate)

    @pytest.mark.parametrize(
        "codec",
        [VideoAudioCodec.AAC, VideoAudioCodec.LIBVORBIS, VideoAudioCodec.LIBOPUS],
    )
    def test_video_audio_config_valid_codec(self, codec):
        """All VideoAudioCodec values are valid when channels > 0."""
        config = VideoAudioConfig(codec=codec, channels=1)
        assert config.codec == codec

    def test_video_audio_config_codec_none(self):
        """None codec is valid (auto-select)."""
        config = VideoAudioConfig(codec=None)
        assert config.codec is None

    def test_video_audio_config_codec_without_channels_raises(self):
        """Setting codec with channels=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="--video-audio-num-channels is 0"):
            VideoAudioConfig(codec=VideoAudioCodec.AAC, channels=0)

    def test_video_audio_config_codec_with_channels_valid(self):
        """Setting codec with channels>0 is accepted."""
        config = VideoAudioConfig(codec=VideoAudioCodec.AAC, channels=1)
        assert config.codec == VideoAudioCodec.AAC


class TestVideoConfigWithAudio:
    """Test VideoConfig properly nests VideoAudioConfig."""

    def test_video_config_default_audio(self):
        """VideoConfig has default VideoAudioConfig nested with audio disabled."""
        config = VideoConfig()
        assert isinstance(config.audio, VideoAudioConfig)
        assert config.audio.channels == 0

    def test_video_config_with_custom_audio(self):
        """VideoConfig accepts custom VideoAudioConfig."""
        audio = VideoAudioConfig(sample_rate=48000, channels=2)
        config = VideoConfig(audio=audio)
        assert config.audio.sample_rate == 48000
        assert config.audio.channels == 2

    def test_video_config_preserves_existing_defaults(self):
        """Existing VideoConfig defaults are unchanged."""
        config = VideoConfig()
        assert config.batch_size == 0
        assert config.duration == 1.0
        assert config.fps == 4
        assert config.format is VideoFormat.WEBM
        assert config.codec == "libvpx-vp9"
