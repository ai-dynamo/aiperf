# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-value contract for dataset content configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import (
    AudioFormat,
    CacheBustTarget,
    ImageFormat,
    ImageSource,
    VideoAudioCodec,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config import SyntheticDataset
from aiperf.config.dataset.content import (
    AudioConfig,
    CacheBustConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    RankingsConfig,
)
from aiperf.config.dataset.video import VideoAudioConfig, VideoConfig
from aiperf.config.distributions import FixedDistribution, NormalDistribution


def test_image_config_defaults():
    """The default values of ImageConfig match the v2 field defaults."""
    config = ImageConfig()
    assert config.batch_size == 0
    assert isinstance(config.width, FixedDistribution)
    assert config.width.expected_value == 512.0
    assert isinstance(config.height, FixedDistribution)
    assert config.height.expected_value == 512.0
    assert config.format == ImageFormat.JPEG
    assert config.source == ImageSource.NOISE
    assert config.images_enabled() is False


def test_image_config_custom_values():
    """ImageConfig accepts custom distribution + format + batch_size values."""
    config = ImageConfig(
        width=640,
        height=480,
        batch_size=16,
        format=ImageFormat.PNG,
    )
    assert config.width.expected_value == 640.0
    assert config.height.expected_value == 480.0
    assert config.batch_size == 16
    assert config.format == ImageFormat.PNG
    assert config.images_enabled() is True


def test_audio_config_defaults():
    """The default values of AudioConfig match the v2 field defaults."""
    config = AudioConfig()
    assert config.batch_size == 0
    assert isinstance(config.length, FixedDistribution)
    assert config.length.expected_value == 10.0
    assert config.format == AudioFormat.WAV
    assert config.depths == [16]
    assert config.sample_rates == [16.0]
    assert config.channels == 1


def test_audio_config_custom_values():
    """AudioConfig accepts custom batch_size, length, format, depths, rates."""
    config = AudioConfig(
        batch_size=32,
        length=5.0,
        format=AudioFormat.MP3,
        depths=[16, 24],
        sample_rates=[44.1, 48.0],
        channels=2,
    )
    assert config.batch_size == 32
    assert config.length.expected_value == 5.0
    assert config.format == AudioFormat.MP3
    assert config.depths == [16, 24]
    assert config.sample_rates == [44.1, 48.0]
    assert config.channels == 2


class TestVideoAudioConfigDefaults:
    """VideoAudioConfig default values + validation contract."""

    def test_video_audio_config_defaults(self):
        """Default values match the v2 field defaults."""
        config = VideoAudioConfig()
        assert config.sample_rate == 44.1
        assert config.channels == 0
        assert config.codec is None
        assert config.depth == 16

    def test_video_audio_config_disabled_by_default(self):
        """Default channels=0 means audio is disabled."""
        config = VideoAudioConfig()
        assert config.channels == 0

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


class TestVideoConfigDefaults:
    """VideoConfig default values + nested audio contract."""

    def test_video_config_defaults(self):
        """Default values match the v2 field defaults."""
        config = VideoConfig()
        assert config.batch_size == 0
        assert config.duration == 1.0
        assert config.fps == 4
        assert config.width is None
        assert config.height is None
        assert config.format == VideoFormat.WEBM
        assert config.codec == "libvpx-vp9"
        assert config.synth_type == VideoSynthType.MOVING_SHAPES

    def test_video_config_default_audio(self):
        """VideoConfig nests a default VideoAudioConfig with audio disabled."""
        config = VideoConfig()
        assert isinstance(config.audio, VideoAudioConfig)
        assert config.audio.channels == 0

    def test_video_config_with_custom_audio(self):
        """VideoConfig accepts a custom VideoAudioConfig."""
        audio = VideoAudioConfig(sample_rate=48.0, channels=2)
        config = VideoConfig(audio=audio)
        assert config.audio.sample_rate == 48.0
        assert config.audio.channels == 2


def test_prompt_config_defaults():
    """The default values of PromptConfig match the v2 field defaults."""
    config = PromptConfig()
    assert config.isl is None
    assert config.osl is None
    assert config.block_size is None
    assert config.batch_size == 1
    assert config.corpus is None
    assert config.sequence_distribution is None


def test_prompt_config_isl_osl_custom_values():
    """PromptConfig hydrates isl/osl into SamplingDistribution instances."""
    config = PromptConfig(isl=100, osl={"mean": 200, "stddev": 10})
    assert isinstance(config.isl, FixedDistribution)
    assert config.isl.expected_value == 100.0
    assert isinstance(config.osl, NormalDistribution)
    assert config.osl.mean == 200.0
    assert config.osl.stddev == 10.0


def test_prefix_prompt_config_defaults():
    """The default values of PrefixPromptConfig are all None (v2)."""
    config = PrefixPromptConfig()
    assert config.pool_size is None
    assert config.length is None
    assert config.shared_system_length is None
    assert config.user_context_length is None


def test_prefix_prompt_config_custom_values():
    """PrefixPromptConfig accepts pool_size + length together."""
    config = PrefixPromptConfig(pool_size=100, length=10)
    assert config.pool_size == 100
    assert config.length == 10


def test_prefix_prompt_config_mutually_exclusive_groups_raise():
    """pool_size/length and shared/user_context lengths are mutually exclusive."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PrefixPromptConfig(pool_size=10, shared_system_length=128)


def test_cache_bust_config_default_is_none():
    """CacheBustConfig disables cache-busting by default."""
    cfg = CacheBustConfig()
    assert cfg.target == CacheBustTarget.NONE


def test_cache_bust_config_accepts_each_target():
    """Every CacheBustTarget enum value is an accepted target."""
    for target in CacheBustTarget:
        cfg = CacheBustConfig(target=target)
        assert cfg.target == target


def test_prompt_config_exposes_cache_bust():
    """PromptConfig nests a default CacheBustConfig (target=none)."""
    pc = PromptConfig()
    assert isinstance(pc.cache_bust, CacheBustConfig)
    assert pc.cache_bust.target == CacheBustTarget.NONE


def test_rankings_config_defaults():
    """The default values of RankingsConfig match the v2 field defaults."""
    config = RankingsConfig()
    assert config.passages.expected_value == 10.0
    assert config.passage_tokens.expected_value == 128.0
    assert config.query_tokens.expected_value == 32.0


def _synthetic(**overrides) -> SyntheticDataset:
    """Build a minimal valid SyntheticDataset, applying field overrides."""
    return SyntheticDataset.model_validate(
        {"name": "default", "type": "synthetic", **overrides}
    )


def test_synthetic_dataset_conversation_turn_defaults():
    """The v2 SyntheticDataset carries the conversation/turn contract."""
    config = _synthetic()
    assert config.turns is None
    assert config.turn_delay is None
    assert config.turn_delay_ratio == 1.0
    assert config.entries == 100
    assert config.random_seed is None


def test_synthetic_dataset_turn_custom_values():
    """SyntheticDataset hydrates turns/turn_delay into SamplingDistributions."""
    config = _synthetic(
        turns={"mean": 5.0, "stddev": 1.0},
        turn_delay={"mean": 10.0, "stddev": 2.0},
        turn_delay_ratio=1.5,
    )
    assert isinstance(config.turns, NormalDistribution)
    assert config.turns.mean == 5.0
    assert config.turns.stddev == 1.0
    assert isinstance(config.turn_delay, NormalDistribution)
    assert config.turn_delay.mean == 10.0
    assert config.turn_delay.stddev == 2.0
    assert config.turn_delay_ratio == 1.5


def test_synthetic_dataset_turns_below_one_rejected():
    """turns expected value must be >= 1 (the conversation-turn contract)."""
    with pytest.raises(ValidationError, match="turns expected value must be >= 1"):
        _synthetic(turns={"mean": 0.0})
