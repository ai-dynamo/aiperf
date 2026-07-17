# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Content generation configs (prompts, prefix prompts, images, audio, rankings)
used as building blocks inside dataset variants. Video configs live in
``video.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aiperf.common.enums import (
    AudioFormat,
    ImageFormat,
    ImageSource,
    ImageSourceSamplingStrategy,
)
from aiperf.config.base import BaseConfig
from aiperf.config.types import (
    FixedDistribution,
    SamplingDistribution,
    SequenceDistributionEntry,
    validate_probability_distribution,
)


def _parse_image_source(value: object) -> object:
    """Coerce ``--image-source`` input into ``ImageSource`` or ``Path``.

    Accepts an ``ImageSource`` enum member, a ``Path``, or a string. Strings
    that match an enum value resolve to the enum; anything else is treated
    as a filesystem path to a directory of source images.
    """
    if isinstance(value, ImageSource):
        return value
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str):
        try:
            return ImageSource(value)
        except ValueError:
            return Path(value).expanduser()
    return value


class PromptConfig(BaseConfig):
    """
    Configuration for prompt/token specifications in synthetic datasets.

    This is the core configuration for controlling input sequence length (ISL)
    and output sequence length (OSL) in synthetic data generation.
    """

    model_config = ConfigDict(extra="forbid")

    isl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Input sequence length in tokens. "
            "Can be a fixed integer (e.g., 512) or distribution {mean: 512, stddev: 50}. "
            "AIPerf generates prompts with lengths following a normal distribution "
            "around the mean (±stddev). Ignored when sequence_distribution is specified.",
        ),
    ]

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length (max tokens to request via max_completion_tokens). "
            "Can be a fixed integer or distribution {mean, stddev}. "
            "Controls response length for synthetic datasets. "
            "When not set, the model determines output length. "
            "Ignored when sequence_distribution is specified.",
        ),
    ]

    block_size: Annotated[
        int | None,
        Field(
            gt=0,
            default=None,
            description="Token block size for hash-based prompt caching in mooncake_trace datasets. "
            "When hash_ids are provided in trace entries, prompts are divided into blocks "
            "of this size. Each hash_id maps to a cached block, enabling simulation of "
            "KV-cache sharing patterns from production workloads. "
            "Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.",
        ),
    ]

    batch_size: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Number of text inputs to include in each request for batch processing endpoints. "
            "Supported by embeddings and rankings endpoint types where models can process "
            "multiple inputs simultaneously. Set to 1 for single-input requests. "
            "Not applicable to chat or completions endpoints.",
        ),
    ]

    sequence_distribution: Annotated[
        list[SequenceDistributionEntry] | None,
        Field(
            default=None,
            description="Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. "
            "Each entry specifies {isl, osl, probability}. "
            "Probabilities are percentages (0-100) and must sum to 100. "
            "When specified, requests are sampled from this distribution instead of using isl/osl fields.",
        ),
    ]

    @field_validator("sequence_distribution")
    @classmethod
    def validate_sequence_probabilities(
        cls, v: list[SequenceDistributionEntry] | None
    ) -> list[SequenceDistributionEntry] | None:
        if v is not None:
            validate_probability_distribution(v)
        return v


class PrefixPromptConfig(BaseConfig):
    """
    Configuration for prefix prompts (KV cache testing).

    Prefix prompts allow testing KV cache efficiency by generating
    requests that share common prefixes. This simulates scenarios
    like system prompts or shared context that can be cached.

    Note: pool_size/length are mutually exclusive with shared_system_length
    and user_context_length.
    """

    model_config = ConfigDict(extra="forbid")

    pool_size: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Number of distinct prefix prompts to generate for KV cache testing. "
            "Each prefix is prepended to user prompts, simulating cached context scenarios. "
            "Prefixes are randomly selected from pool per request. "
            "Mutually exclusive with shared_system_length/user_context_length.",
        ),
    ]

    length: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Token length for each prefix prompt in the pool. "
            "Only used when pool_size is set. "
            "Note: due to prefix and user prompts being concatenated, "
            "the final prompt token count may be off by one. "
            "Mutually exclusive with shared_system_length/user_context_length.",
        ),
    ]

    shared_system_length: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Length of shared system prompt in tokens. "
            "This prompt is identical across all sessions and appears as a system message. "
            "First part of a two-part prefix structure with high cache hit rate expected. "
            "Mutually exclusive with pool_size/length.",
        ),
    ]

    user_context_length: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Length of per-session user context prompt in tokens. "
            "Each dataset entry gets a unique user context prompt. "
            "Second part of two-part prefix structure with lower cache hit rate expected. "
            "Mutually exclusive with pool_size/length.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_prefix_exclusivity(self) -> Self:
        pool_group = (self.pool_size, self.length)
        system_group = (self.shared_system_length, self.user_context_length)
        has_pool = any(v is not None for v in pool_group)
        has_system = any(v is not None for v in system_group)
        if has_pool and has_system:
            raise ValueError(
                "pool_size/length and shared_system_length/user_context_length "
                "are mutually exclusive"
            )
        return self


class ImageConfig(BaseConfig):
    """
    Configuration for synthetic image generation in multimodal datasets.

    Controls the generation of synthetic images for vision-language
    model benchmarking. Images are generated by randomly sampling and
    resizing source images to specified dimensions.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"source": {"const": "noise"}}},
                    "then": {
                        "properties": {
                            "sourceSampling": {"const": "random-with-replacement"}
                        }
                    },
                }
            ]
        },
    )

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            default=0,
            description="Number of images to include in each multimodal request. "
            "Supported with chat endpoint type for vision-language models. "
            "Set to 0 to disable image inputs. "
            "Higher batch sizes test multi-image understanding and increase request payload size.",
        ),
    ]

    width: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=512),
            description="Image width in pixels. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Combined with height to determine image dimensions and file sizes "
            "for multimodal benchmarking.",
        ),
    ]

    height: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=512),
            description="Image height in pixels. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Used when batch_size > 0 for multimodal vision benchmarking.",
        ),
    ]

    format: Annotated[
        ImageFormat,
        Field(
            default=ImageFormat.JPEG,
            description="Image file format for generated images. "
            "png: lossless compression (larger files, best quality). "
            "jpeg: lossy compression (smaller files, good quality). "
            "random: randomly select between PNG and JPEG per image. "
            "Format affects file size in multimodal requests and encoding overhead.",
        ),
    ]

    source: Annotated[
        ImageSource | Path,
        BeforeValidator(_parse_image_source),
        Field(
            default=ImageSource.NOISE,
            description="Source image generation mode (default: noise). "
            "noise: generate random noise images on the fly — no files on disk, "
            "effectively unbounded variety so servers cannot dedupe identical inputs. "
            "assets: index images from the bundled assets/source_images directory and "
            "lazily load them at the requested dimensions. "
            "A path string indexes images from the given directory (e.g. ./source_images). "
            "Random-noise images are roughly incompressible, so payload bytes are larger "
            "than equivalent natural images.",
        ),
    ]

    source_sampling: Annotated[
        ImageSourceSamplingStrategy,
        Field(
            default=ImageSourceSamplingStrategy.RANDOM_WITH_REPLACEMENT,
            description="How source images are selected from finite image sources "
            "selected by source='assets' or a directory path. "
            "random-with-replacement: draw each source image independently; repeats "
            "may occur immediately. "
            "shuffle-cycle: draw every source image once per shuffled cycle, "
            "reshuffling after exhaustion. "
            "sequential-cycle: walk source images in sorted load order and wrap "
            "after exhaustion. For noise mode, only random-with-replacement is "
            "valid because there is no finite source pool.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_source_sampling_source(self) -> Self:
        if (
            self.source_sampling != ImageSourceSamplingStrategy.RANDOM_WITH_REPLACEMENT
            and self.source == ImageSource.NOISE
        ):
            raise ValueError(
                "images.source_sampling requires image source 'assets' or a directory "
                "path unless it is 'random-with-replacement'; noise has no finite source pool"
            )
        return self

    def images_enabled(self) -> bool:
        """Whether image generation is configured to produce images.

        Mirrors the v1 helper used by composers/generators: requires positive
        batch_size and positive expected width/height.
        """
        return (
            self.batch_size > 0
            and self.width.expected_value > 0
            and self.height.expected_value > 0
        )


class AudioConfig(BaseConfig):
    """
    Configuration for synthetic audio generation in multimodal datasets.

    Controls the generation of synthetic audio for speech-to-text
    and audio-language model benchmarking. Generated audio is random
    noise with specified sample rate, bit depth, and format.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            default=0,
            description="Number of audio inputs to include in each multimodal request. "
            "Supported with chat endpoint type for multimodal models. "
            "Set to 0 to disable audio inputs.",
        ),
    ]

    length: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=10.0),
            description="Audio duration in seconds. "
            "Can be a fixed value or {mean, stddev} distribution. "
            "Used when batch_size > 0 for multimodal benchmarking.",
        ),
    ]

    format: Annotated[
        AudioFormat,
        Field(
            default=AudioFormat.WAV,
            description="File format for generated audio files. "
            "wav: uncompressed PCM (larger files). "
            "mp3: compressed (smaller files). "
            "Format affects file size in multimodal requests but not audio characteristics.",
        ),
    ]

    sample_rates: Annotated[
        list[float],
        Field(
            default_factory=lambda: [16.0],
            description="List of audio sample rates in kHz to randomly select from. "
            "Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), "
            "48.0 (professional). Specify multiple values for mixed-quality testing.",
        ),
    ]

    depths: Annotated[
        list[int],
        Field(
            default_factory=lambda: [16],
            description="List of audio bit depths in bits to randomly select from. "
            "Each audio file is assigned a random depth from this list. "
            "Common values: 8 (low quality), 16 (CD quality), 24 (professional), "
            "32 (high-end). Specify multiple values for mixed-quality testing.",
        ),
    ]

    channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            default=1,
            description="Number of audio channels. "
            "1 = mono (single channel), 2 = stereo (left/right channels). "
            "Stereo doubles file size. Most speech models use mono.",
        ),
    ]


class RankingsConfig(BaseConfig):
    """
    Configuration for rankings/reranking endpoint datasets.

    Controls the generation of query-passage pairs for benchmarking
    reranking and ranking models. Each request contains one query
    and multiple passages to rank.
    """

    model_config = ConfigDict(extra="forbid")

    passages: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=10),
            description="Number of passages per ranking request. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Higher values test ranking at scale but increase request payload size "
            "and processing time.",
        ),
    ]

    passage_tokens: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=128),
            description="Token length for each passage in ranking requests. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Passages are synthetically generated text. "
            "Longer passages increase input processing demands and request size.",
        ),
    ]

    query_tokens: Annotated[
        SamplingDistribution,
        Field(
            default_factory=lambda: FixedDistribution(value=32),
            description="Token length for the query text in ranking requests. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Each ranking request contains one query and multiple passages.",
        ),
    ]
