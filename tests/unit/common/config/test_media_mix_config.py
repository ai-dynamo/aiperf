# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.config.image_config import ImageHeightConfig, ImageWidthConfig
from aiperf.common.config.media_mix_config import (
    AudioProfileConfig,
    ImageProfileConfig,
    MediaMixArchetype,
    ModalityEntry,
    TextOverrideConfig,
    VideoProfileConfig,
)
from aiperf.common.config.prompt_config import InputTokensConfig, OutputTokensConfig
from aiperf.common.enums import ImageFormat


class TestImageProfileConfig:
    def test_valid_profile(self):
        profile = ImageProfileConfig(
            weight=1.0,
            width=ImageWidthConfig(mean=1024, stddev=128),
            height=ImageHeightConfig(mean=768, stddev=96),
            format=ImageFormat.JPEG,
        )
        assert profile.weight == 1.0
        assert profile.width.mean == 1024
        assert profile.format == ImageFormat.JPEG

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            ImageProfileConfig(weight=0, width=ImageWidthConfig(mean=1024))


class TestAudioProfileConfig:
    def test_valid_profile(self):
        profile = AudioProfileConfig(weight=0.5)
        assert profile.weight == 0.5
        assert profile.num_channels == 1


class TestVideoProfileConfig:
    def test_valid_profile(self):
        profile = VideoProfileConfig(weight=1.0, width=1280, height=720)
        assert profile.width == 1280
        assert profile.fps == 4


class TestModalityEntry:
    def test_image_with_image_profiles(self):
        entry = ModalityEntry(
            modality="image",
            batch_size=2,
            profiles=[
                ImageProfileConfig(
                    weight=1.0,
                    width=ImageWidthConfig(mean=256),
                    height=ImageHeightConfig(mean=256),
                )
            ],
        )
        assert entry.modality == "image"
        assert entry.batch_size == 2

    def test_audio_with_audio_profiles(self):
        entry = ModalityEntry(
            modality="audio",
            profiles=[AudioProfileConfig(weight=1.0)],
        )
        assert entry.modality == "audio"

    def test_video_with_video_profiles(self):
        entry = ModalityEntry(
            modality="video",
            profiles=[VideoProfileConfig(weight=1.0, width=640, height=480)],
        )
        assert entry.modality == "video"

    def test_mismatched_profile_type_raises(self):
        with pytest.raises(ValueError, match="Expected ImageProfileConfig"):
            ModalityEntry(
                modality="image",
                profiles=[AudioProfileConfig(weight=1.0)],
            )

    def test_empty_profiles_raises(self):
        with pytest.raises(ValueError):
            ModalityEntry(modality="image", profiles=[])


class TestMediaMixArchetype:
    def test_text_only_archetype(self):
        archetype = MediaMixArchetype(weight=1.0, modalities=[])
        assert archetype.include_text is True
        assert archetype.weight == 1.0

    def test_text_disabled_with_modalities(self):
        archetype = MediaMixArchetype(
            weight=1.0,
            text=False,
            modalities=[
                ModalityEntry(
                    modality="image",
                    profiles=[
                        ImageProfileConfig(
                            weight=1.0,
                            width=ImageWidthConfig(mean=256),
                            height=ImageHeightConfig(mean=256),
                        )
                    ],
                )
            ],
        )
        assert archetype.include_text is False

    def test_text_disabled_no_modalities_raises(self):
        with pytest.raises(ValueError, match="at least one modality or text enabled"):
            MediaMixArchetype(weight=1.0, text=False, modalities=[])

    def test_text_override_config(self):
        archetype = MediaMixArchetype(
            weight=1.0,
            text=TextOverrideConfig(
                input_tokens=InputTokensConfig(mean=100, stddev=20),
                output_tokens=OutputTokensConfig(mean=500),
            ),
        )
        assert archetype.include_text is True
        assert isinstance(archetype.text, TextOverrideConfig)
        assert archetype.text.input_tokens.mean == 100

    def test_text_none_means_enabled(self):
        archetype = MediaMixArchetype(weight=1.0)
        assert archetype.include_text is True

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            MediaMixArchetype(weight=0)


class TestTextOverrideConfig:
    def test_partial_override(self):
        config = TextOverrideConfig(
            input_tokens=InputTokensConfig(mean=100),
        )
        assert config.input_tokens.mean == 100
        assert config.output_tokens is None

    def test_full_override(self):
        config = TextOverrideConfig(
            input_tokens=InputTokensConfig(mean=100, stddev=10),
            output_tokens=OutputTokensConfig(mean=500, stddev=50),
        )
        assert config.input_tokens.mean == 100
        assert config.output_tokens.mean == 500

    def test_empty_override(self):
        config = TextOverrideConfig()
        assert config.input_tokens is None
        assert config.output_tokens is None


class TestArchetypeNameValidation:
    """InputConfig.validate_media_mix_archetype_names (Decision 9)."""

    @staticmethod
    def _single_image_modality() -> ModalityEntry:
        return ModalityEntry(
            modality="image",
            profiles=[
                ImageProfileConfig(
                    weight=1.0,
                    width=ImageWidthConfig(mean=256),
                    height=ImageHeightConfig(mean=256),
                )
            ],
        )

    def test_duplicate_archetype_names_rejected(self):
        from aiperf.common.config import InputConfig

        with pytest.raises(ValueError, match="Duplicate archetype names"):
            InputConfig(
                media_mix=[
                    MediaMixArchetype(
                        weight=0.5,
                        name="image",
                        modalities=[self._single_image_modality()],
                    ),
                    MediaMixArchetype(
                        weight=0.5,
                        name="image",  # duplicate
                        modalities=[self._single_image_modality()],
                    ),
                ]
            )

    def test_unnamed_archetypes_get_synthetic_names(self):
        from aiperf.common.config import InputConfig

        config = InputConfig(
            media_mix=[
                MediaMixArchetype(
                    weight=0.5,
                    modalities=[self._single_image_modality()],
                ),
                MediaMixArchetype(
                    weight=0.5,
                    modalities=[self._single_image_modality()],
                ),
            ]
        )
        names = [a.name for a in config.media_mix]
        assert names == ["_archetype_0", "_archetype_1"]
