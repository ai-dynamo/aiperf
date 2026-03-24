# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from aiperf.common import random_generator as rng
from aiperf.common.config.audio_config import AudioConfig
from aiperf.common.config.image_config import ImageConfig
from aiperf.common.config.media_mix_config import (
    AudioProfileConfig,
    ImageProfileConfig,
    MediaMixArchetype,
    ModalityEntry,
    TextOverrideConfig,
    VideoProfileConfig,
)
from aiperf.common.config.video_config import VideoConfig
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.random_generator import RandomGenerator
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.video import VideoGenerator


@dataclass(slots=True)
class ResolvedTurn:
    """Per-turn media generation plan produced by MediaMixResolver."""

    include_text: bool = True
    input_tokens_mean: int | None = None
    input_tokens_stddev: float | None = None
    output_tokens_mean: int | None = None
    output_tokens_stddev: float | None = None
    image_generators: list[tuple[ImageGenerator, int]] = field(default_factory=list)
    audio_generators: list[tuple[AudioGenerator, int]] = field(default_factory=list)
    video_generators: list[tuple[VideoGenerator, int]] = field(default_factory=list)


class MediaMixResolver(AIPerfLoggerMixin):
    """Samples archetypes and profiles per turn, providing pre-created generators."""

    def __init__(self, archetypes: list[MediaMixArchetype]) -> None:
        super().__init__()
        self._archetypes = archetypes
        self._archetype_rng = rng.derive("media_mix.archetype")
        self._profile_rngs: dict[str, RandomGenerator] = {}
        self._generators: dict[tuple[int, str, int], BaseGenerator] = {}
        self._archetype_weights = [a.weight for a in archetypes]

        for a_idx, archetype in enumerate(archetypes):
            for entry in archetype.modalities:
                key = f"media_mix.{entry.modality}.{a_idx}"
                self._profile_rngs[key] = rng.derive(f"{key}.profile")
                for p_idx, profile in enumerate(entry.profiles):
                    ns = f"mm.{a_idx}.{p_idx}"
                    gen = self._create_generator(entry.modality, profile, ns)
                    self._generators[(a_idx, entry.modality, p_idx)] = gen

        self.info(
            lambda: f"Initialized MediaMixResolver with {len(archetypes)} archetypes, "
            f"{len(self._generators)} generators"
        )

    def resolve_turn(self) -> ResolvedTurn:
        """Sample an archetype, then sample a profile per modality.

        Returns:
            ResolvedTurn with generators and text override info for one turn.
        """
        a_idx = self._sample_weighted(self._archetype_weights, self._archetype_rng)
        archetype = self._archetypes[a_idx]

        resolved = ResolvedTurn(include_text=archetype.include_text)
        self._apply_text_overrides(archetype, resolved)

        for entry in archetype.modalities:
            self._resolve_modality(a_idx, entry, resolved)

        return resolved

    def _apply_text_overrides(
        self, archetype: MediaMixArchetype, resolved: ResolvedTurn
    ) -> None:
        """Extract text override ISL/OSL from archetype config."""
        if not isinstance(archetype.text, TextOverrideConfig):
            return
        if archetype.text.input_tokens is not None:
            resolved.input_tokens_mean = archetype.text.input_tokens.mean
            resolved.input_tokens_stddev = archetype.text.input_tokens.stddev
        if archetype.text.output_tokens is not None:
            resolved.output_tokens_mean = archetype.text.output_tokens.mean
            resolved.output_tokens_stddev = archetype.text.output_tokens.stddev

    def _resolve_modality(
        self, a_idx: int, entry: ModalityEntry, resolved: ResolvedTurn
    ) -> None:
        """Sample a profile for a modality entry and add its generator to resolved."""
        profile_weights = [p.weight for p in entry.profiles]
        key = f"media_mix.{entry.modality}.{a_idx}"
        p_idx = self._sample_weighted(profile_weights, self._profile_rngs[key])
        gen = self._generators[(a_idx, entry.modality, p_idx)]

        target = {
            "image": resolved.image_generators,
            "audio": resolved.audio_generators,
            "video": resolved.video_generators,
        }[entry.modality]
        target.append((gen, entry.batch_size))

    def _sample_weighted(self, weights: list[float], rand: RandomGenerator) -> int:
        """Sample an index from a weighted distribution.

        Args:
            weights: Positive weights (need not sum to 1).
            rand: RandomGenerator to use for sampling.

        Returns:
            Sampled index.
        """
        total = sum(weights)
        normalized = [w / total for w in weights]
        return int(rand.numpy_choice(len(normalized), p=normalized))

    def _create_generator(
        self,
        modality: str,
        profile: ImageProfileConfig | AudioProfileConfig | VideoProfileConfig,
        rng_namespace: str,
    ) -> BaseGenerator:
        """Create a generator from a profile config.

        Args:
            modality: "image", "audio", or "video".
            profile: Profile config for the modality.
            rng_namespace: Unique RNG namespace for this profile.

        Returns:
            A configured generator instance.
        """
        if modality == "image" and isinstance(profile, ImageProfileConfig):
            config = ImageConfig(
                width=profile.width,
                height=profile.height,
                batch_size=1,
                format=profile.format,
            )
            return ImageGenerator(config, rng_namespace=rng_namespace)

        if modality == "audio" and isinstance(profile, AudioProfileConfig):
            config = AudioConfig(
                batch_size=1,
                length=profile.length,
                format=profile.format,
                depths=profile.depths,
                sample_rates=profile.sample_rates,
                num_channels=profile.num_channels,
            )
            return AudioGenerator(config, rng_namespace=rng_namespace)

        if modality == "video" and isinstance(profile, VideoProfileConfig):
            config = VideoConfig(
                batch_size=1,
                width=profile.width,
                height=profile.height,
                duration=profile.duration,
                fps=profile.fps,
                format=profile.format,
                codec=profile.codec,
                synth_type=profile.synth_type,
                audio=profile.audio,
            )
            return VideoGenerator(config, rng_namespace=rng_namespace)

        raise ValueError(f"Unknown modality '{modality}' or mismatched profile type.")
