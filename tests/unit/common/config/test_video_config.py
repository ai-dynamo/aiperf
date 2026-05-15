# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    VideoAudioConfig,
    VideoAudioDefaults,
    VideoConfig,
)
from aiperf.common.enums import VideoAudioCodec, VideoFormat


class TestVideoAudioConfigDefaults:
    """Test VideoAudioConfig default values."""

    def test_video_audio_config_defaults(self):
        """Default values match VideoAudioDefaults."""
        config = VideoAudioConfig()
        assert config.sample_rate == VideoAudioDefaults.SAMPLE_RATE
        assert config.channels == VideoAudioDefaults.CHANNELS
        assert config.codec is VideoAudioDefaults.CODEC

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

    @pytest.mark.parametrize("sample_rate", [8.0, 16.0, 44.1, 48.0, 96.0])
    def test_video_audio_config_valid_sample_rate(self, sample_rate):
        """Sample rates within 8-96 kHz are valid."""
        config = VideoAudioConfig(sample_rate=sample_rate)
        assert config.sample_rate == sample_rate

    @pytest.mark.parametrize("sample_rate", [7.999, 96.001, 0, -1])
    def test_video_audio_config_invalid_sample_rate(self, sample_rate):
        """Sample rates outside 8-96 kHz raise ValidationError."""
        with pytest.raises(ValidationError):
            VideoAudioConfig(sample_rate=sample_rate)

    @pytest.mark.parametrize("depth", [8, 16, 24, 32, "8", "16", "24", "32"])
    def test_video_audio_config_depth_coerces_string(self, depth):
        """Depth accepts int or numeric string (from YAML/JSON configs)."""
        config = VideoAudioConfig(depth=depth)
        assert config.depth == int(depth)

    @pytest.mark.parametrize("depth", [0, 12, "12", "abc"])
    def test_video_audio_config_invalid_depth(self, depth):
        """Non-supported depth values raise ValidationError."""
        with pytest.raises(ValidationError):
            VideoAudioConfig(depth=depth)

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
        audio = VideoAudioConfig(sample_rate=48.0, channels=2)
        config = VideoConfig(audio=audio)
        assert config.audio.sample_rate == 48.0
        assert config.audio.channels == 2

    def test_video_config_preserves_existing_defaults(self):
        """Existing VideoConfig defaults are unchanged."""
        config = VideoConfig()
        assert config.batch_size == 1
        assert config.duration == 5.0
        assert config.fps == 4
        assert config.format == VideoFormat.WEBM
        assert config.codec == "libvpx-vp9"


class TestVideoConfigValidation:
    """Behavior tests for VideoConfig validators."""

    def test_default_valued_does_not_raise(self):
        """Loading a config with explicit default-valued width/height must not raise."""
        config = VideoConfig(
            batch_size=1,
            duration=5.0,
            fps=4,
            width=None,
            height=None,
        )
        assert config.videos_enabled() is False

    def test_config_yaml_roundtrip_does_not_raise(self):
        """Round-tripping a fully-serialized default config must not raise."""
        original = VideoConfig()
        dumped = original.model_dump()
        config = VideoConfig.model_validate(dumped)
        assert config.videos_enabled() is False

    def test_batch_size_zero_is_disabled(self):
        """`--video-batch-size 0` resolves to videos_enabled=False."""
        config = VideoConfig(batch_size=0)
        assert config.videos_enabled() is False

    def test_batch_size_zero_with_dims_is_disabled(self):
        """Width/height set with `batch_size=0` is treated as a valid explicit-disable."""
        config = VideoConfig(batch_size=0, width=640, height=480)
        assert config.videos_enabled() is False

    def test_width_without_height_raises(self):
        """Setting width but not height is rejected by validate_width_and_height."""
        with pytest.raises(
            ValidationError, match="Width is specified but height is not"
        ):
            VideoConfig(width=640)

    def test_height_without_width_raises(self):
        """Setting height but not width is rejected by validate_width_and_height."""
        with pytest.raises(
            ValidationError, match="Height is specified but width is not"
        ):
            VideoConfig(height=480)
