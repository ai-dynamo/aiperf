# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.audio_config import AudioLengthConfig
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import (
    AudioDefaults,
    ImageDefaults,
    VideoDefaults,
)
from aiperf.common.config.image_config import ImageHeightConfig, ImageWidthConfig
from aiperf.common.config.prompt_config import InputTokensConfig, OutputTokensConfig
from aiperf.common.config.video_config import VideoAudioConfig
from aiperf.common.enums import (
    AudioFormat,
    ImageFormat,
    VideoFormat,
    VideoSynthType,
)

VALID_MODALITIES = {"image", "audio", "video"}


class TextOverrideConfig(BaseConfig):
    """Per-archetype overrides for text prompt generation (ISL/OSL)."""

    input_tokens: InputTokensConfig | None = Field(
        default=None,
        description="Override input sequence length (ISL) for this archetype. "
        "When omitted, the global --prompt-input-tokens-mean/stddev is used.",
    )
    output_tokens: OutputTokensConfig | None = Field(
        default=None,
        description="Override output sequence length (OSL) for this archetype. "
        "When omitted, the global --prompt-output-tokens-mean/stddev is used.",
    )


class ImageProfileConfig(BaseConfig):
    """Dimensional profile for synthetic image generation within a media mix archetype."""

    weight: float = Field(
        gt=0, description="Sampling probability weight for this profile."
    )
    width: ImageWidthConfig = Field(
        default_factory=ImageWidthConfig,
        description="Image width distribution (mean/stddev in pixels).",
    )
    height: ImageHeightConfig = Field(
        default_factory=ImageHeightConfig,
        description="Image height distribution (mean/stddev in pixels).",
    )
    format: ImageFormat = Field(
        default=ImageDefaults.FORMAT,
        description="Image file format: png, jpeg, or random.",
    )


class AudioProfileConfig(BaseConfig):
    """Dimensional profile for synthetic audio generation within a media mix archetype."""

    weight: float = Field(
        gt=0, description="Sampling probability weight for this profile."
    )
    length: AudioLengthConfig = Field(
        default_factory=AudioLengthConfig,
        description="Audio duration distribution (mean/stddev in seconds).",
    )
    format: AudioFormat = Field(
        default=AudioDefaults.FORMAT,
        description="Audio file format: wav or mp3.",
    )
    depths: list[int] = Field(
        default=AudioDefaults.DEPTHS,
        description="Bit depths to randomly select from (8, 16, 24, 32).",
    )
    sample_rates: list[float] = Field(
        default=AudioDefaults.SAMPLE_RATES,
        description="Sample rates in kHz to randomly select from.",
    )
    num_channels: int = Field(
        default=AudioDefaults.NUM_CHANNELS,
        ge=1,
        le=2,
        description="Number of audio channels: 1 (mono) or 2 (stereo).",
    )


class VideoProfileConfig(BaseConfig):
    """Dimensional profile for synthetic video generation within a media mix archetype."""

    weight: float = Field(
        gt=0, description="Sampling probability weight for this profile."
    )
    width: int = Field(ge=1, description="Video frame width in pixels.")
    height: int = Field(ge=1, description="Video frame height in pixels.")
    duration: float = Field(
        default=VideoDefaults.DURATION,
        ge=0.0,
        description="Video duration in seconds.",
    )
    fps: int = Field(
        default=VideoDefaults.FPS,
        ge=1,
        description="Frames per second.",
    )
    format: VideoFormat = Field(
        default=VideoDefaults.FORMAT,
        description="Video container format: webm or mp4.",
    )
    codec: str = Field(
        default=VideoDefaults.CODEC,
        description="FFmpeg video codec name.",
    )
    synth_type: VideoSynthType = Field(
        default=VideoDefaults.SYNTH_TYPE,
        description="Synthetic video pattern: moving_shapes, grid_clock, or noise.",
    )
    audio: VideoAudioConfig = Field(
        default_factory=VideoAudioConfig,
        description="Embedded audio track configuration.",
    )


class ModalityEntry(BaseConfig):
    """A modality within a media mix archetype, with weighted profiles."""

    modality: Literal["image", "audio", "video"] = Field(
        description="Media type: image, audio, or video.",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of items of this modality per request.",
    )
    profiles: list[ImageProfileConfig | AudioProfileConfig | VideoProfileConfig] = (
        Field(
            min_length=1,
            description="Weighted generation profiles for this modality.",
        )
    )

    @model_validator(mode="after")
    def validate_profiles_match_modality(self) -> Self:
        """Ensure all profiles match the declared modality type."""
        expected = {
            "image": ImageProfileConfig,
            "audio": AudioProfileConfig,
            "video": VideoProfileConfig,
        }
        expected_type = expected[self.modality]
        for i, profile in enumerate(self.profiles):
            if not isinstance(profile, expected_type):
                raise ValueError(
                    f"Profile {i} is {type(profile).__name__} but modality is '{self.modality}'. "
                    f"Expected {expected_type.__name__}."
                )
        return self


class MediaMixArchetype(BaseConfig):
    """A weighted request archetype defining which modalities appear and how."""

    weight: float = Field(
        gt=0,
        description="Sampling probability weight for this archetype.",
    )
    name: str | None = Field(
        default=None,
        description="Optional human-readable name for this archetype.",
    )
    text: TextOverrideConfig | bool | None = Field(
        default=None,
        description="Text prompt configuration. None or True: text enabled with global config. "
        "TextOverrideConfig: text enabled with ISL/OSL overrides. False: text disabled.",
    )
    modalities: list[ModalityEntry] = Field(
        default_factory=list,
        description="Media modalities included in this archetype.",
    )

    @model_validator(mode="after")
    def validate_has_content(self) -> Self:
        """Ensure the archetype produces at least some content."""
        text_enabled = self.text is not False
        has_modalities = len(self.modalities) > 0
        if not text_enabled and not has_modalities:
            raise ValueError(
                "Archetype must have at least one modality or text enabled. "
                "Got text=False with no modalities."
            )
        return self

    @property
    def include_text(self) -> bool:
        """Whether text prompt generation is enabled for this archetype."""
        return self.text is not False


def parse_media_mix(value: Any) -> Any:
    """BeforeValidator: parse shorthand string or pass through list/None.

    Shorthand format: "image:0.6,video:0.2,audio:0.2"
    Returns sentinel dicts with _shorthand=True for InputConfig to inflate.
    """
    if value is None or isinstance(value, list):
        return value

    if not isinstance(value, str):
        return value

    archetypes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid media mix shorthand '{part}'. Expected 'modality:weight' "
                f"(e.g., 'image:0.6,video:0.4')."
            )
        modality, weight_str = part.split(":", 1)
        modality = modality.strip().lower()
        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}'. Must be one of: {sorted(VALID_MODALITIES)}."
            )
        try:
            weight = float(weight_str.strip())
        except ValueError as e:
            raise ValueError(
                f"Invalid weight '{weight_str.strip()}' for modality '{modality}'. "
                "Must be a positive number."
            ) from e
        if weight <= 0:
            raise ValueError(f"Weight for '{modality}' must be positive, got {weight}.")
        archetypes.append({"_shorthand": True, "modality": modality, "weight": weight})

    if not archetypes:
        raise ValueError("Media mix shorthand cannot be empty.")

    return archetypes


def normalize_media_mix_input(value: Any) -> list | None:
    """Parse string shorthand to list, validate list is non-empty.

    Returns None if input is None/empty/invalid (caller should leave data unchanged).
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = parse_media_mix(value)
    if not isinstance(value, list) or not value:
        return None
    return value


def _as_dict(value: Any) -> dict:
    """Return value as dict, or empty dict if not a dict."""
    return value if isinstance(value, dict) else {}


def _build_image_modality_entry(image_cfg: dict) -> dict:
    """Build an image ModalityEntry dict from sibling --image-* config values."""
    return {
        "modality": "image",
        "batch_size": max(1, image_cfg.get("batch_size", 1)),
        "profiles": [
            {
                "weight": 1.0,
                "width": image_cfg.get("width", {}),
                "height": image_cfg.get("height", {}),
                "format": image_cfg.get("format", "png"),
            }
        ],
    }


def _build_audio_modality_entry(audio_cfg: dict) -> dict:
    """Build an audio ModalityEntry dict from sibling --audio-* config values."""
    return {
        "modality": "audio",
        "batch_size": max(1, audio_cfg.get("batch_size", 1)),
        "profiles": [
            {
                "weight": 1.0,
                "length": audio_cfg.get("length", {}),
                "format": audio_cfg.get("format", "wav"),
                "depths": audio_cfg.get("depths", [16]),
                "sample_rates": audio_cfg.get("sample_rates", [16.0]),
                "num_channels": audio_cfg.get("num_channels", 1),
            }
        ],
    }


def _build_video_modality_entry(video_cfg: dict) -> dict:
    """Build a video ModalityEntry dict from sibling --video-* config values."""
    return {
        "modality": "video",
        "batch_size": max(1, video_cfg.get("batch_size", 1)),
        "profiles": [
            {
                "weight": 1.0,
                "width": video_cfg.get("width") or 640,
                "height": video_cfg.get("height") or 480,
                "duration": video_cfg.get("duration", 5.0),
                "fps": video_cfg.get("fps", 4),
                "format": video_cfg.get("format", "webm"),
                "codec": video_cfg.get("codec", "libvpx-vp9"),
                "synth_type": video_cfg.get("synth_type", "moving_shapes"),
            }
        ],
    }


_MODALITY_BUILDERS = {
    "image": _build_image_modality_entry,
    "audio": _build_audio_modality_entry,
    "video": _build_video_modality_entry,
}


def inflate_shorthand_archetypes(media_mix: list, sibling_data: dict) -> list[dict]:
    """Inflate a list of shorthand sentinel dicts into full archetype dicts.

    Args:
        media_mix: List of shorthand sentinel dicts (each must have _shorthand=True).
        sibling_data: Raw InputConfig data dict for accessing sibling image/audio/video config.

    Returns:
        List of full archetype dicts ready for Pydantic validation.

    Raises:
        ValueError: If any entry is not a shorthand dict.
    """
    sibling_cfgs = {
        "image": _as_dict(sibling_data.get("image")),
        "audio": _as_dict(sibling_data.get("audio")),
        "video": _as_dict(sibling_data.get("video")),
    }
    return [_inflate_shorthand_entry(entry, sibling_cfgs) for entry in media_mix]


def _inflate_shorthand_entry(entry: Any, sibling_cfgs: dict[str, dict]) -> dict:
    """Inflate a single shorthand sentinel dict into a full MediaMixArchetype dict."""
    if not isinstance(entry, dict) or not entry.get("_shorthand"):
        raise ValueError(
            "Cannot mix shorthand and full media mix archetype definitions."
        )
    modality = entry["modality"]
    builder = _MODALITY_BUILDERS[modality]
    return {
        "weight": entry["weight"],
        "name": modality,
        "modalities": [builder(sibling_cfgs[modality])],
    }


def is_shorthand_list(media_mix: list) -> bool:
    """Return True if the list contains shorthand sentinel dicts."""
    return (
        bool(media_mix)
        and isinstance(media_mix[0], dict)
        and media_mix[0].get("_shorthand") is True
    )
