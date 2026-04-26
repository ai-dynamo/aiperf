# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 InputConfig and nested children.

Consolidates the 8 origin/main input config files (conversation, prompt, image,
audio, video, rankings, synthesis, input) into a single validator-free DTO
hierarchy. Validators are forbidden on v1 - AIPerfConfig owns validation.

`BeforeValidator(...)` Annotated metadata is preserved because cyclopts needs
those input-shape coercers; the no-validator rule only forbids
`@field_validator` and `@model_validator` decorators.
"""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field

from aiperf.common.enums import (
    AudioFormat,
    ImageFormat,
    VideoAudioCodec,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, Groups
from aiperf.config.parsing import (
    parse_file,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list_of_positive_values,
)
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
    PublicDatasetType,
)

# Default constants inlined from origin/main's config_defaults.py. The v2
# refactor dropped these classes from aiperf.config.defaults; we restore the
# literal values here so v1 stays self-contained.
_TURN_DELAY_MEAN = 0.0
_TURN_DELAY_STDDEV = 0.0
_TURN_DELAY_RATIO = 1.0
_TURN_MEAN = 1
_TURN_STDDEV = 0
_CONVERSATION_NUM: int | None = None
_PROMPT_NUM = 100
_PROMPT_BATCH_SIZE = 1
_INPUT_TOKENS_MEAN = 550
_INPUT_TOKENS_STDDEV = 0.0
_OUTPUT_TOKENS_STDDEV = 0
_PREFIX_PROMPT_POOL_SIZE = 0
_PREFIX_PROMPT_LENGTH = 0
_IMAGE_BATCH_SIZE = 1
_IMAGE_WIDTH_MEAN = 0.0
_IMAGE_WIDTH_STDDEV = 0.0
_IMAGE_HEIGHT_MEAN = 0.0
_IMAGE_HEIGHT_STDDEV = 0.0
_IMAGE_FORMAT = ImageFormat.PNG
_AUDIO_BATCH_SIZE = 1
_AUDIO_LENGTH_MEAN = 0.0
_AUDIO_LENGTH_STDDEV = 0.0
_AUDIO_FORMAT = AudioFormat.WAV
_AUDIO_DEPTHS = [16]
_AUDIO_SAMPLE_RATES = [16.0]
_AUDIO_NUM_CHANNELS = 1
_VIDEO_BATCH_SIZE = 1
_VIDEO_DURATION = 5.0
_VIDEO_FPS = 4
_VIDEO_WIDTH: int | None = None
_VIDEO_HEIGHT: int | None = None
_VIDEO_SYNTH_TYPE = VideoSynthType.MOVING_SHAPES
_VIDEO_FORMAT = VideoFormat.WEBM
_VIDEO_CODEC = "libvpx-vp9"
_VIDEO_AUDIO_SAMPLE_RATE = 44100
_VIDEO_AUDIO_CHANNELS = 0
_VIDEO_AUDIO_CODEC: VideoAudioCodec | None = None
_VIDEO_AUDIO_DEPTH = 16
_RANKINGS_PASSAGES_MEAN = 1
_RANKINGS_PASSAGES_STDDEV = 0
_RANKINGS_PASSAGES_PROMPT_TOKEN_MEAN = 550
_RANKINGS_PASSAGES_PROMPT_TOKEN_STDDEV = 0
_RANKINGS_QUERY_PROMPT_TOKEN_MEAN = 550
_RANKINGS_QUERY_PROMPT_TOKEN_STDDEV = 0
_INPUT_EXTRA: list = []
_INPUT_HEADERS: list = []
_INPUT_FILE = None
_INPUT_FIXED_SCHEDULE = False
_INPUT_FIXED_SCHEDULE_AUTO_OFFSET = False
_INPUT_FIXED_SCHEDULE_START_OFFSET: int | None = None
_INPUT_FIXED_SCHEDULE_END_OFFSET: int | None = None
_INPUT_PUBLIC_DATASET = None
_INPUT_CUSTOM_DATASET_TYPE = None
_INPUT_RANDOM_SEED: int | None = None
_INPUT_GOODPUT = None


# --- Conversation ---------------------------------------------------------


class TurnDelayConfig(BaseConfig):
    """Turn delay related settings."""

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between "
            "receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). "
            "Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-mean",
                "--session-turn-delay-mean",
            ),
            group=_CLI_GROUP,
        ),
    ] = _TURN_DELAY_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. "
            "Delays follow normal distribution. Set to 0 for deterministic delays. "
            "Models realistic human interaction patterns with variable response times.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-stddev",
                "--session-turn-delay-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = _TURN_DELAY_STDDEV

    ratio: Annotated[
        float,
        Field(
            ge=0,
            description="Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: "
            "`actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. "
            "Values < 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-ratio",
                "--session-delay-ratio",
            ),
            group=_CLI_GROUP,
        ),
    ] = _TURN_DELAY_RATIO


class TurnConfig(BaseConfig):
    """Turn related settings in a conversation."""

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean number of request-response turns per conversation. Each turn consists of a user message and model response. "
            "Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. "
            "Multi-turn conversations enable testing of context retention and conversation history handling.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-mean",
                "--session-turns-mean",
            ),
            group=_CLI_GROUP,
        ),
    ] = _TURN_MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating "
            "diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. "
            "Set to 0 for uniform conversation lengths.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-stddev",
                "--session-turns-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = _TURN_STDDEV

    delay: TurnDelayConfig = TurnDelayConfig()


class ConversationConfig(BaseConfig):
    """Conversations related settings."""

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    num: Annotated[
        int | None,
        Field(
            ge=1,
            description="The total number of unique conversations to generate.\n"
            "Each conversation represents a single request session between client and server.\n"
            "Supported on synthetic mode and the custom random_pool dataset. The number of conversations \n"
            "will be used to determine the number of entries in both the custom random_pool and synthetic \n"
            "datasets and will be reused until benchmarking is complete.",
        ),
        CLIParameter(
            name=(
                "--conversation-num",
                "--num-conversations",
                "--num-sessions",
            ),
            group=_CLI_GROUP,
        ),
    ] = _CONVERSATION_NUM

    num_dataset_entries: Annotated[
        int,
        Field(
            ge=1,
            description="Total number of unique entries to generate for the dataset. Each entry represents one user message that can be "
            "used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. "
            "Higher values provide more diversity.",
        ),
        CLIParameter(
            name=(
                "--num-dataset-entries",
                "--num-prompts",
            ),
            group=_CLI_GROUP,
        ),
    ] = _PROMPT_NUM

    turn: TurnConfig = TurnConfig()


# --- Prompt ---------------------------------------------------------------


class InputTokensConfig(BaseConfig):
    """Input token related settings."""

    _CLI_GROUP = Groups.ISL

    mean: Annotated[
        int,
        Field(
            ge=0,
            description="Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths "
            "following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, "
            "not custom or public datasets.",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-mean",
                "--synthetic-input-tokens-mean",
                "--isl",
            ),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_TOKENS_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, "
            "simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. "
            "Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-stddev",
                "--synthetic-input-tokens-stddev",
                "--isl-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_TOKENS_STDDEV

    block_size: Annotated[
        int | None,
        Field(
            default=None,
            description="Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, "
            "prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation "
            "of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. "
            "When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-block-size",
                "--synthetic-input-tokens-block-size",
                "--isl-block-size",
            ),
            group=_CLI_GROUP,
        ),
    ] = None


class OutputTokensConfig(BaseConfig):
    """Output token related settings."""

    _CLI_GROUP = Groups.OSL

    mean: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Mean number of tokens to request in model outputs via `max_completion_tokens` field. "
            "Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit "
            "generation length. When not set, model determines output length.",
        ),
        CLIParameter(
            name=(
                "--prompt-output-tokens-mean",
                "--output-tokens-mean",
                "--osl",
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    stddev: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            description="Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, "
            "simulating mixed response length requirements. Lengths follow normal distribution. "
            "Only applies when `--prompt-output-tokens-mean` is set.",
        ),
        CLIParameter(
            name=(
                "--prompt-output-tokens-stddev",
                "--output-tokens-stddev",
                "--osl-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = _OUTPUT_TOKENS_STDDEV


class PrefixPromptConfig(BaseConfig):
    """Prefix prompt related settings."""

    _CLI_GROUP = Groups.PREFIX_PROMPT

    pool_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, "
            "simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. "
            "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.",
        ),
        CLIParameter(
            name=(
                "--prompt-prefix-pool-size",
                "--prefix-prompt-pool-size",
                "--num-prefix-prompts",
            ),
            group=_CLI_GROUP,
        ),
    ] = _PREFIX_PROMPT_POOL_SIZE

    length: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "The number of tokens in each prefix prompt.\n"
                "This is only used if `--num-prefix-prompts` is greater than zero.\n"
                "Note that due to the prefix and user prompts being concatenated,\n"
                "the number of tokens in the final prompt may be off by one."
                "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`."
            ),
        ),
        CLIParameter(
            name=(
                "--prompt-prefix-length",
                "--prefix-prompt-length",
            ),
            group=_CLI_GROUP,
        ),
    ] = _PREFIX_PROMPT_LENGTH

    shared_system_prompt_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of shared system prompt in tokens.\n"
                "This prompt is identical across all sessions and appears as a system message.\n"
                "Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`."
            ),
        ),
        CLIParameter(
            name=("--shared-system-prompt-length",),
            group=_CLI_GROUP,
        ),
    ] = None

    user_context_prompt_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of per-session user context prompt in tokens.\n"
                "Each dataset entry gets a unique user context prompt.\n"
                "Requires --num-dataset-entries to be specified.\n"
                "Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size."
            ),
        ),
        CLIParameter(
            name=("--user-context-prompt-length",),
            group=_CLI_GROUP,
        ),
    ] = None


class PromptConfig(BaseConfig):
    """Prompt related settings."""

    _CLI_GROUP = Groups.PROMPT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` "
            "and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. "
            "Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.",
        ),
        CLIParameter(
            name=(
                "--prompt-batch-size",
                "--batch-size-text",
                "--batch-size",
                "-b",
            ),
            group=_CLI_GROUP,
        ),
    ] = _PROMPT_BATCH_SIZE

    input_tokens: InputTokensConfig = InputTokensConfig()
    output_tokens: OutputTokensConfig = OutputTokensConfig()
    prefix_prompt: PrefixPromptConfig = PrefixPromptConfig()

    sequence_distribution: Annotated[
        str | None,
        Field(
            default=None,
            description="Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. "
            "Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). "
            "Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. "
            "Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. "
            "Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.",
        ),
        CLIParameter(
            name=("--seq-dist", "--sequence-distribution"),
            group=Groups.ISL,
        ),
    ] = None


# --- Image ---------------------------------------------------------------


class ImageHeightConfig(BaseConfig):
    """Image height related settings."""

    _CLI_GROUP = Groups.IMAGE_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean height in pixels for synthetically generated images. Image heights follow a normal distribution "
            "around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. "
            "Generated images are resized from source images in `assets/source_images` directory.",
        ),
        CLIParameter(
            name=("--image-height-mean",),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_HEIGHT_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, "
            "simulating mixed-resolution image inputs. Heights follow normal distribution. "
            "Set to 0 for uniform image heights.",
        ),
        CLIParameter(
            name=("--image-height-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_HEIGHT_STDDEV


class ImageWidthConfig(BaseConfig):
    """Image width related settings."""

    _CLI_GROUP = Groups.IMAGE_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean width in pixels for synthetically generated images. Image widths follow a normal distribution "
            "around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions "
            "and file sizes for multimodal benchmarking.",
        ),
        CLIParameter(
            name=("--image-width-mean",),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_WIDTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, "
            "simulating mixed-resolution image inputs. Widths follow normal distribution. "
            "Set to 0 for uniform image widths.",
        ),
        CLIParameter(
            name=("--image-width-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_WIDTH_STDDEV


class ImageConfig(BaseConfig):
    """Image related settings."""

    _CLI_GROUP = Groups.IMAGE_INPUT

    width: ImageWidthConfig = ImageWidthConfig()
    height: ImageHeightConfig = ImageHeightConfig()
    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. "
            "Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. "
            "Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--image-batch-size",
                "--batch-size-image",
            ),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_BATCH_SIZE

    format: Annotated[
        ImageFormat,
        Field(
            description="Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), "
            "`jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. "
            "Format affects file size in multimodal requests and encoding overhead.",
        ),
        CLIParameter(
            name=("--image-format",),
            group=_CLI_GROUP,
        ),
    ] = _IMAGE_FORMAT


# --- Audio ---------------------------------------------------------------


class AudioLengthConfig(BaseConfig):
    """Audio length related settings."""

    _CLI_GROUP = Groups.AUDIO_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution "
            "around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. "
            "Generated audio is random noise with specified sample rate, bit depth, and format.",
        ),
        CLIParameter(
            name=("--audio-length-mean",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_LENGTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, "
            "simulating mixed-duration audio inputs. Durations follow normal distribution. "
            "Set to 0 for uniform audio lengths.",
        ),
        CLIParameter(
            name=("--audio-length-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_LENGTH_STDDEV


class AudioConfig(BaseConfig):
    """Audio related settings."""

    _CLI_GROUP = Groups.AUDIO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.",
        ),
        CLIParameter(
            name=(
                "--audio-batch-size",
                "--batch-size-audio",
            ),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_BATCH_SIZE

    length: AudioLengthConfig = AudioLengthConfig()

    format: Annotated[
        AudioFormat,
        Field(
            description="File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). "
            "Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).",
        ),
        CLIParameter(
            name=("--audio-format",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_FORMAT

    depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned "
            "a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). "
            "Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=("--audio-depths",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_DEPTHS

    sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\n"
            "Common sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=("--audio-sample-rates",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_SAMPLE_RATES

    num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). "
            "Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. "
            "Most speech models use mono.",
        ),
        CLIParameter(
            name=("--audio-num-channels",),
            group=_CLI_GROUP,
        ),
    ] = _AUDIO_NUM_CHANNELS


# --- Video ---------------------------------------------------------------


class VideoAudioConfig(BaseConfig):
    """Configuration for embedding an audio track in synthetic video files."""

    _CLI_GROUP = Groups.VIDEO_INPUT

    sample_rate: Annotated[
        int,
        Field(
            ge=8000,
            le=96000,
            description="Audio sample rate in Hz for the embedded audio track. "
            "Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). "
            "Higher sample rates increase audio fidelity and file size.",
        ),
        CLIParameter(
            name=("--video-audio-sample-rate",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_AUDIO_SAMPLE_RATE

    channels: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            description="Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video duration "
            "is muxed into each video via FFmpeg.",
        ),
        CLIParameter(
            name=("--video-audio-num-channels",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_AUDIO_CHANNELS

    codec: Annotated[
        VideoAudioCodec | None,
        Field(
            description="Audio codec for the embedded audio track. "
            "If not specified, auto-selects based on video format: "
            "aac for MP4, libvorbis for WebM. "
            "Options: aac, libvorbis, libopus.",
        ),
        CLIParameter(
            name=("--video-audio-codec",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_AUDIO_CODEC

    depth: Annotated[
        Literal[8, 16, 24, 32],
        Field(
            description="Audio bit depth for the embedded audio track. "
            "Supported values: 8, 16, 24, or 32 bits. "
            "Higher bit depths provide greater dynamic range but increase file size.",
        ),
        CLIParameter(
            name=("--video-audio-depth",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_AUDIO_DEPTH


class VideoConfig(BaseConfig):
    """Video related settings.

    Note: Video generation requires FFmpeg to be installed on your system.
    If FFmpeg is not found, you'll get installation instructions specific to your platform.
    """

    _CLI_GROUP = Groups.VIDEO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. "
            "Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. "
            "Higher batch sizes test multi-video understanding and significantly increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--video-batch-size",
                "--batch-size-video",
            ),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_BATCH_SIZE

    duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count "
            "(frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. "
            "Requires FFmpeg for video generation.",
        ),
        CLIParameter(
            name=("--video-duration",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_DURATION

    fps: Annotated[
        int,
        Field(
            ge=1,
            description="Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). "
            "Total frames = `--video-duration` × FPS.",
        ),
        CLIParameter(
            name=("--video-fps",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_FPS

    width: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution "
            "and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.",
        ),
        CLIParameter(
            name=("--video-width",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_WIDTH

    height: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width "
            "determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.",
        ),
        CLIParameter(
            name=("--video-height",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_HEIGHT

    synth_type: Annotated[
        VideoSynthType,
        Field(
            description="Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. "
            "Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). "
            "Content doesn't affect semantic meaning but may impact encoding efficiency and file size.",
        ),
        CLIParameter(
            name=("--video-synth-type",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_SYNTH_TYPE

    format: Annotated[
        VideoFormat,
        Field(
            description="Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). "
            "Format choice affects compatibility, file size, and encoding options. "
            "Use `webm` for open-source workflows, `mp4` for maximum compatibility.",
        ),
        CLIParameter(
            name=("--video-format",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_FORMAT

    codec: Annotated[
        str,
        Field(
            description=(
                "The video codec to use for encoding. Common options: "
                "libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
                "libx264 (CPU, GPL-licensed, widely compatible), "
                "libx265 (CPU, GPL-licensed, smaller files), "
                "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
                "Any FFmpeg-supported codec can be used."
            ),
        ),
        CLIParameter(
            name=("--video-codec",),
            group=_CLI_GROUP,
        ),
    ] = _VIDEO_CODEC

    audio: Annotated[
        VideoAudioConfig,
        Field(
            description="Audio track configuration for embedding audio in generated videos."
        ),
    ] = VideoAudioConfig()


# --- Rankings ------------------------------------------------------------


class RankingsPassagesConfig(BaseConfig):
    """Rankings passages related settings."""

    _CLI_GROUP = Groups.RANKINGS

    mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query "
            "and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). "
            "Higher values test ranking at scale but increase request payload size and processing time.",
        ),
        CLIParameter(
            name=("--rankings-passages-mean",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_PASSAGES_MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. "
            "Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.",
        ),
        CLIParameter(
            name=("--rankings-passages-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_PASSAGES_STDDEV

    prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths "
            "following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). "
            "Longer passages increase input processing demands and request size.",
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-mean",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_PASSAGES_PROMPT_TOKEN_MEAN

    prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating "
            "realistic heterogeneous document collections. Token lengths follow normal distribution. "
            "Set to 0 for uniform passage lengths.",
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_PASSAGES_PROMPT_TOKEN_STDDEV


class RankingsQueryConfig(BaseConfig):
    """Rankings query related settings."""

    _CLI_GROUP = Groups.RANKINGS

    prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. "
            "Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`). ",
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-mean",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_QUERY_PROMPT_TOKEN_MEAN

    prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating "
            "realistic user search patterns. Token lengths follow normal distribution. "
            "Set to 0 for uniform query lengths.",
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-stddev",),
            group=_CLI_GROUP,
        ),
    ] = _RANKINGS_QUERY_PROMPT_TOKEN_STDDEV


class RankingsConfig(BaseConfig):
    """Rankings related settings."""

    _CLI_GROUP = Groups.RANKINGS

    passages: RankingsPassagesConfig = RankingsPassagesConfig()
    query: RankingsQueryConfig = RankingsQueryConfig()


# --- Synthesis -----------------------------------------------------------


class SynthesisConfig(BaseConfig):
    """Configuration for synthetic trace generation with prefix patterns."""

    _CLI_GROUP = Groups.SYNTHESIS

    speedup_ratio: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for timestamp scaling in synthesized traces",
        ),
        CLIParameter(name=("--synthesis-speedup-ratio",), group=_CLI_GROUP),
    ] = 1.0

    prefix_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for core prefix branch lengths in radix tree",
        ),
        CLIParameter(name=("--synthesis-prefix-len-multiplier",), group=_CLI_GROUP),
    ] = 1.0

    prefix_root_multiplier: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Number of independent radix trees to distribute traces across",
        ),
        CLIParameter(name=("--synthesis-prefix-root-multiplier",), group=_CLI_GROUP),
    ] = 1

    prompt_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for leaf path (unique prompt) lengths",
        ),
        CLIParameter(name=("--synthesis-prompt-len-multiplier",), group=_CLI_GROUP),
    ] = 1.0

    max_isl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.",
        ),
        CLIParameter(name=("--synthesis-max-isl",), group=_CLI_GROUP),
    ] = None

    max_osl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.",
        ),
        CLIParameter(name=("--synthesis-max-osl",), group=_CLI_GROUP),
    ] = None


# --- InputConfig ---------------------------------------------------------


class InputConfig(BaseConfig):
    """Input related settings (top-level)."""

    _CLI_GROUP = Groups.INPUT

    extra: Annotated[
        Any,
        Field(
            description="Additional input parameters to include in every API request payload. Specify as `key:value` pairs "
            "(e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{\"temperature\": 0.7}'`). "
            "These parameters are merged with request-specific inputs and sent directly to the endpoint API.",
        ),
        CLIParameter(
            name=("--extra-inputs",),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
    ] = _INPUT_EXTRA

    headers: Annotated[
        Any,
        Field(
            description="Custom HTTP headers to include with every request. Specify as `Header:Value` pairs "
            "(e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. "
            "Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers "
            "(e.g., `Authorization` from `--api-key`).",
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        CLIParameter(
            name=(
                "--header",
                "-H",
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
    ] = _INPUT_HEADERS

    file: Annotated[
        Any,
        Field(
            description="Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. "
            "Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), "
            "directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.",
        ),
        BeforeValidator(parse_file),
        CLIParameter(
            name=("--input-file",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_FILE

    fixed_schedule: Annotated[
        bool,
        Field(
            description="Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays "
            "the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets."
        ),
        CLIParameter(
            name=("--fixed-schedule",),
            group=_CLI_GROUP,
            negative=None,
        ),
    ] = _INPUT_FIXED_SCHEDULE

    fixed_schedule_auto_offset: Annotated[
        bool,
        Field(
            description="Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. "
            "When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute "
            "offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.",
        ),
        CLIParameter(
            name=("--fixed-schedule-auto-offset",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_FIXED_SCHEDULE_AUTO_OFFSET

    fixed_schedule_start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing "
            "benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. "
            "Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. "
            "Must be ≤ `--fixed-schedule-end-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-start-offset",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_FIXED_SCHEDULE_START_OFFSET

    fixed_schedule_end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing "
            "benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. "
            "Must be ≥ `--fixed-schedule-start-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-end-offset",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_FIXED_SCHEDULE_END_OFFSET

    public_dataset: Annotated[
        PublicDatasetType | None,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). "
            "AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. "
            "Run `aiperf plugins public_dataset_loader` to list available datasets. "
            "Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.",
        ),
        CLIParameter(
            name=("--public-dataset",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_PUBLIC_DATASET

    hf_dataset_subset: Annotated[
        str | None,
        Field(
            description="HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). "
            "Only applies when using `--public-dataset` with a HuggingFace-backed loader. "
            "Takes priority over the subset defined in the plugin registry.",
        ),
        CLIParameter(
            name=("--hf-subset",),
            group=_CLI_GROUP,
        ),
    ] = None

    custom_dataset_type: Annotated[
        CustomDatasetType | None,
        Field(
            description="Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. "
            "Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), "
            "`mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; "
            "when using `random_pool`, `--conversation-num` defaults to 100 if not specified; "
            "batch sizes > 1 sample each modality independently from a flat pool and do not preserve "
            "per-entry associations - use `single_turn` if paired modalities must stay together). "
            "Requires `--input-file`. Mutually exclusive with `--public-dataset`.",
        ),
        CLIParameter(
            name=("--custom-dataset-type",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_CUSTOM_DATASET_TYPE

    dataset_sampling_strategy: Annotated[
        DatasetSamplingStrategy | None,
        Field(
            description="Strategy for selecting entries from dataset during benchmarking. "
            "`sequential`: Iterate through dataset in order, wrapping to start after end. "
            "`random`: Randomly sample with replacement (entries may repeat before all are used). "
            "`shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. "
            "Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).",
        ),
        CLIParameter(
            name=("--dataset-sampling-strategy",),
            group=_CLI_GROUP,
        ),
    ] = None

    random_seed: Annotated[
        int | None,
        Field(
            description="Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other "
            "random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. "
            "Initialized globally at config creation.",
        ),
        CLIParameter(
            name=("--random-seed",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_RANDOM_SEED

    goodput: Annotated[
        Any | None,
        Field(
            default=None,
            description="Specify service level objectives (SLOs) for goodput as space-separated "
            "'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the "
            "metric's display unit (falls back to its base unit if no display unit is defined). "
            "Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), "
            "`output_token_throughput_per_user:600` (tokens/s).\n"
            "Only metrics applicable to the current endpoint/config are considered. "
            "For more context on the definition of goodput, "
            "refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
            "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
        ),
        BeforeValidator(parse_str_as_numeric_dict),
        CLIParameter(
            name=("--goodput",),
            group=_CLI_GROUP,
        ),
    ] = _INPUT_GOODPUT

    audio: AudioConfig = AudioConfig()
    image: ImageConfig = ImageConfig()
    video: VideoConfig = VideoConfig()
    prompt: PromptConfig = PromptConfig()
    rankings: RankingsConfig = RankingsConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    conversation: ConversationConfig = ConversationConfig()


__all__ = [
    "AudioConfig",
    "AudioLengthConfig",
    "ConversationConfig",
    "ImageConfig",
    "ImageHeightConfig",
    "ImageWidthConfig",
    "InputConfig",
    "InputTokensConfig",
    "OutputTokensConfig",
    "PrefixPromptConfig",
    "PromptConfig",
    "RankingsConfig",
    "RankingsPassagesConfig",
    "RankingsQueryConfig",
    "SynthesisConfig",
    "TurnConfig",
    "TurnDelayConfig",
    "VideoAudioConfig",
    "VideoConfig",
]
