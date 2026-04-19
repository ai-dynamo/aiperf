# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Datasets - Data sources, prompts, and multimodal content
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from aiperf.common.enums import (
    AudioFormat,
    DatasetFormat,
    DatasetType,
    ImageFormat,
    OslMode,
    PublicDatasetType,
    VideoAudioCodec,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config._base import BaseConfig
from aiperf.config.types import (
    FixedDistribution,
    SamplingDistribution,
    SequenceDistributionEntry,
    validate_probability_distribution,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


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

    model_config = ConfigDict(extra="forbid")

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


VIDEO_AUDIO_CODEC_MAP: dict[VideoFormat, VideoAudioCodec] = {
    VideoFormat.WEBM: VideoAudioCodec.LIBVORBIS,
    VideoFormat.MP4: VideoAudioCodec.AAC,
}


class VideoAudioConfig(BaseConfig):
    """Configuration for embedding an audio track in synthetic video files."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.codec is not None and self.channels == 0:
            raise ValueError(
                f"--video-audio-codec '{self.codec}' is set but --video-audio-num-channels is 0 "
                f"(audio disabled). Set --video-audio-num-channels to 1 or 2 to enable audio."
            )
        return self

    sample_rate: Annotated[
        int,
        Field(
            ge=8000,
            le=96000,
            default=44100,
            description="Audio sample rate in Hz for the embedded audio track. "
            "Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). "
            "Higher sample rates increase audio fidelity and file size.",
        ),
    ]

    channels: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            default=0,
            description="Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video duration "
            "is muxed into each video via FFmpeg.",
        ),
    ]

    codec: Annotated[
        VideoAudioCodec | None,
        Field(
            default=None,
            description="Audio codec for the embedded audio track. "
            "If not specified, auto-selects based on video format: "
            "aac for MP4, libvorbis for WebM. "
            "Options: aac, libvorbis, libopus.",
        ),
    ]

    depth: Annotated[
        Literal[8, 16, 24, 32],
        Field(
            default=16,
            description="Audio bit depth for the embedded audio track. "
            "Supported values: 8, 16, 24, or 32 bits. "
            "Higher bit depths provide greater dynamic range but increase file size.",
        ),
    ]


class VideoConfig(BaseConfig):
    """
    Configuration for synthetic video generation in multimodal datasets.

    Controls the generation of synthetic videos for video understanding
    model benchmarking. Requires FFmpeg for video generation.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            default=0,
            description="Number of video files to include in each multimodal request. "
            "Supported with chat endpoint type for video understanding models. "
            "Set to 0 to disable video inputs. "
            "Higher batch sizes significantly increase request payload size.",
        ),
    ]

    duration: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Duration in seconds for each generated video clip. "
            "Combined with fps, determines total frame count (frames = duration * fps). "
            "Longer durations increase file size and processing time. "
            "Typical values: 1-10 seconds for testing.",
        ),
    ]

    fps: Annotated[
        int,
        Field(
            ge=1,
            default=4,
            description="Frames per second for generated video. "
            "Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: 4 (minimal, recommended for Cosmos models), "
            "24 (cinematic), 30 (standard), 60 (high frame rate). "
            "Total frames = duration * fps.",
        ),
    ]

    width: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Video frame width in pixels. "
            "Determines video resolution and file size. "
            "Common values: 640 (SD), 1280 (HD), 1920 (Full HD). "
            "If not specified, uses codec/format defaults.",
        ),
    ]

    height: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Video frame height in pixels. "
            "Combined with width determines aspect ratio and total pixel count per frame. "
            "Common values: 480 (SD), 720 (HD), 1080 (Full HD). "
            "If not specified, uses codec/format defaults.",
        ),
    ]

    format: Annotated[
        VideoFormat,
        Field(
            default=VideoFormat.WEBM,
            description="Container format for generated video files. "
            "webm: VP9 codec, BSD-licensed, recommended for open-source workflows. "
            "mp4: H.264/H.265, widely compatible. "
            "avi: legacy, larger files. "
            "mkv: Matroska, flexible container. "
            "Format affects compatibility, file size, and encoding options.",
        ),
    ]

    codec: Annotated[
        str,
        Field(
            default="libvpx-vp9",
            description="Video codec for encoding. "
            "Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
            "libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), "
            "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
            "Any FFmpeg-supported codec can be used.",
        ),
    ]

    synth_type: Annotated[
        VideoSynthType,
        Field(
            default=VideoSynthType.MOVING_SHAPES,
            description="Algorithm for generating synthetic video content. "
            "Different types produce different visual patterns for testing. "
            "Content doesn't affect semantic meaning but may impact encoding "
            "efficiency and file size.",
        ),
    ]

    audio: Annotated[
        VideoAudioConfig,
        Field(
            default_factory=VideoAudioConfig,
            description="Audio track configuration for embedding audio in generated videos.",
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


class SynthesisConfig(BaseConfig):
    """
    Configuration for trace synthesis/transformation.

    Used with mooncake_trace format to transform production trace
    data before replay. Allows scaling timestamps, token lengths,
    and radix tree structure.
    """

    model_config = ConfigDict(extra="forbid")

    speedup_ratio: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for timestamp scaling in synthesized traces. "
            "1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.",
        ),
    ]

    prefix_len_multiplier: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for core prefix branch lengths in the radix tree. "
            "1.5 means prefix branches are 50%% longer.",
        ),
    ]

    prefix_root_multiplier: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Number of independent radix trees to distribute traces across. "
            "Higher values increase prefix diversity.",
        ),
    ]

    prompt_len_multiplier: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for leaf path (unique prompt) lengths. "
            "2.0 means prompts are 2x longer.",
        ),
    ]

    max_isl: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Maximum input sequence length filter. "
            "Traces with input_length > max_isl are skipped entirely.",
        ),
    ]

    max_osl: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Maximum output sequence length cap. "
            "Traces with output_length > max_osl are capped to this value (not filtered).",
        ),
    ]


class AugmentConfig(BaseConfig):
    """
    Configuration for augmenting file datasets with output length specifications.

    Used in composed datasets where file-based prompts need OSL control.
    """

    model_config = ConfigDict(extra="forbid")

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length to apply to augmented records. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Behavior depends on osl_mode setting.",
        ),
    ]

    osl_mode: Annotated[
        OslMode,
        Field(
            default=OslMode.FILL,
            description="How to apply OSL to records. "
            "fill: only apply if the record lacks an existing OSL value. "
            "override: always replace existing OSL.",
        ),
    ]

    output_distribution: Annotated[
        list[SequenceDistributionEntry] | None,
        Field(
            default=None,
            description="Output length probability distribution. "
            "When specified, overrides the osl field. "
            "Each entry specifies {isl (ignored), osl, probability}. "
            "Probabilities must sum to 100.",
        ),
    ]

    @field_validator("output_distribution")
    @classmethod
    def validate_output_probabilities(
        cls, v: list[SequenceDistributionEntry] | None
    ) -> list[SequenceDistributionEntry] | None:
        if v is not None:
            validate_probability_distribution(v)
        return v


# Dataset type variants using discriminated unions
class SyntheticDataset(BaseConfig):
    """
    Synthetic dataset configuration.

    Generates prompts programmatically based on token length
    specifications. Ideal for controlled experiments.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.SYNTHETIC],
        Field(description="Dataset type discriminator. Must be 'synthetic'."),
    ]

    entries: Annotated[
        int,
        Field(
            ge=1,
            default=100,
            description="Total number of unique entries to generate for the dataset. "
            "Each entry represents a unique prompt with sampled ISL/OSL. "
            "Entries are reused across conversations and turns according to "
            "the sampling strategy. Higher values provide more diversity.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic dataset generation. "
            "When set, makes synthetic prompts, sampling, and other random operations "
            "reproducible across runs. Essential for A/B testing and debugging. "
            "Overrides global random_seed for this dataset.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    prompts: Annotated[
        PromptConfig | None,
        Field(
            default=None,
            description="Prompt/token length configuration specifying ISL, OSL, "
            "sequence distributions, and batch processing settings.",
        ),
    ]

    prefix_prompts: Annotated[
        PrefixPromptConfig | None,
        Field(
            default=None,
            description="Shared prefix configuration for KV cache testing. "
            "Generates prefix prompts that are prepended to user prompts, "
            "simulating cached context scenarios.",
        ),
    ]

    turns: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Number of request-response turns per conversation. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Each turn consists of a user message and model response. "
            "Set to 1 for single-turn interactions. "
            "Multi-turn conversations enable testing of context retention "
            "and conversation history handling.",
        ),
    ]

    turn_delay: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Delay in milliseconds between consecutive turns within a "
            "multi-turn conversation. Can be a fixed value or {mean, stddev} distribution. "
            "Simulates user think time between receiving a response and sending "
            "the next message. Only applies when turns > 1. "
            "Set to 0 for back-to-back turns.",
        ),
    ]

    turn_delay_ratio: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for scaling all turn delays. "
            "Applied after mean/stddev calculation: actual_delay = calculated_delay * ratio. "
            "Values < 1 speed up conversations, > 1 slow them down. "
            "Set to 0 to eliminate delays entirely.",
        ),
    ]

    images: Annotated[
        ImageConfig | None,
        Field(
            default=None,
            description="Synthetic image configuration for multimodal vision-language testing.",
        ),
    ]

    audio: Annotated[
        AudioConfig | None,
        Field(
            default=None,
            description="Synthetic audio configuration for multimodal speech/audio testing.",
        ),
    ]

    video: Annotated[
        VideoConfig | None,
        Field(
            default=None,
            description="Synthetic video configuration for multimodal video understanding testing.",
        ),
    ]

    rankings: Annotated[
        RankingsConfig | None,
        Field(
            default=None,
            description="Rankings/reranking configuration for generating query-passage pairs. "
            "Only relevant for rankings endpoint types.",
        ),
    ]


class FileDataset(BaseConfig):
    """
    File-based dataset configuration.

    Loads prompts from a local file in various formats.
    Supports trace replay and custom sampling strategies.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Dataset type discriminator. Must be 'file'."),
    ]

    path: Annotated[
        Path,
        Field(
            description="Path to file or directory containing benchmark dataset. "
            "Can be absolute or relative. Supported formats depend on the format field: "
            "JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, "
            "directories for random_pool."
        ),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic and expected file structure. "
            "single_turn: JSONL with single prompt-response exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace: timestamped trace files for replay. "
            "random_pool: directory of reusable prompts.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    synthesis: Annotated[
        SynthesisConfig | None,
        Field(
            default=None,
            description="Trace synthesis/transformation configuration. "
            "Allows scaling timestamps and token lengths before replay. "
            "Only used with mooncake_trace format.",
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Limit number of records to use from file. "
            "If not specified, uses all records in the file.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic sampling. "
            "When set, makes random/shuffle sampling reproducible across runs. "
            "Overrides global random_seed for this dataset.",
        ),
    ]


class PublicDataset(BaseConfig):
    """
    Public dataset configuration.

    Uses well-known public benchmarking datasets that are
    automatically downloaded and processed by AIPerf.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.PUBLIC],
        Field(description="Dataset type discriminator. Must be 'public'."),
    ]

    name: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "AIPerf automatically downloads and parses these datasets. "
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Limit number of records to use from the dataset. "
            "If not specified, uses all available records.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic sampling from the dataset. "
            "Overrides global random_seed for this dataset.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    hf_subset: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace dataset subset/config name override (e.g. 'sharegpt4o'). "
            "Only applies for HuggingFace-backed public dataset loaders. "
            "Takes priority over the subset defined in the plugin registry.",
        ),
    ]


class FileSourceConfig(BaseConfig):
    """
    File source configuration for composed datasets.

    Simplified file dataset specification used within composed
    dataset source field.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Source type. Must be 'file' for composed datasets."),
    ]

    path: Annotated[
        Path,
        Field(description="Path to the source file. Can be absolute or relative."),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic. "
            "single_turn: JSONL with single exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace: timestamped trace files.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from the source file. "
            "sequential: iterate in order. "
            "random: randomly sample with replacement. "
            "shuffle: random permutation without replacement.",
        ),
    ]


class ComposedDataset(BaseConfig):
    """
    Composed dataset configuration (unique to AIPerf).

    Combines file-based data with synthetic augmentation.
    This enables advanced scenarios like:
    - Adding system prompts to existing queries
    - Testing KV cache with file-based prompts
    - Adding multimodal content to text datasets
    - Extending small datasets with padding
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.COMPOSED],
        Field(
            default=DatasetType.COMPOSED,
            description="Dataset type discriminator. Must be 'composed'.",
        ),
    ]

    source: Annotated[
        FileSourceConfig,
        Field(description="The base file dataset to augment."),
    ]

    augment: Annotated[
        AugmentConfig,
        Field(
            description="Augmentation configuration specifying prefixes, suffixes, "
            "OSL, multimodal content, and padding for the source data."
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Final dataset size after augmentation. "
            "If source has fewer entries and pad_to_count is set in augment, "
            "synthetic padding entries are generated to reach this count.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic augmentation. "
            "Overrides global random_seed for this dataset.",
        ),
    ]


# Union type for all dataset variants using discriminated union
DatasetConfig = Annotated[
    SyntheticDataset | FileDataset | PublicDataset | ComposedDataset,
    Discriminator("type"),
]
"""
Dataset configuration supporting multiple source types.

Discriminated by 'type' field or structure:
    - synthetic: Generated prompts (type: synthetic)
    - file: Local file data (type: file)
    - public: Public benchmark datasets (type: public)
    - composed: Combined dataset (has source + augment fields)
"""
