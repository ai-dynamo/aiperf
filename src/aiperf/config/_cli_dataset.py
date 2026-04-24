# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset-section builders for the CLI-to-config converter.

Split from ``cli_converter.py`` to keep each builder focused and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


_NON_TEXT_TEXT_FLAGS: dict[str, tuple[str, ...]] = {
    "isl_mean": (
        "--isl",
        "--synthetic-input-tokens-mean",
        "--prompt-input-tokens-mean",
    ),
    "isl_stddev": (
        "--isl-stddev",
        "--synthetic-input-tokens-stddev",
        "--prompt-input-tokens-stddev",
    ),
    "isl_block_size": (
        "--isl-block-size",
        "--synthetic-input-tokens-block-size",
        "--prompt-input-tokens-block-size",
    ),
    "prompt_batch_size": ("--batch-size-text", "--prompt-batch-size"),
    "sequence_distribution": ("--seq-dist", "--sequence-distribution"),
    "prefix_prompt_length": (
        "--prefix-prompt-length",
        "--prompt-prefix-length",
    ),
    "num_prefix_prompts": (
        "--prefix-prompt-pool-size",
        "--prompt-prefix-pool-size",
        "--num-prefix-prompts",
    ),
}
_NON_TEXT_TOKENIZER_FLAGS: dict[str, tuple[str, ...]] = {
    "tokenizer_name": ("--tokenizer",),
    "tokenizer_trust_remote_code": ("--tokenizer-trust-remote-code",),
}

_AUGMENT_TRIGGER_FIELDS = frozenset(
    {
        "num_prefix_prompts",
        "prefix_prompt_length",
        "shared_system_prompt_length",
        "user_context_prompt_length",
        "image_batch_size",
        "audio_batch_size",
        "video_batch_size",
        "osl_mean",
    }
)


def _validate_non_text_endpoint(endpoint_type: Any, s: set[str]) -> None:
    """Raise if text/tokenizer options are used with a non-text endpoint."""
    bad = [
        "/".join(flags) for field, flags in _NON_TEXT_TEXT_FLAGS.items() if field in s
    ]
    if bad:
        raise ValueError(
            f"{', '.join(bad)} cannot be used with --endpoint-type {endpoint_type}."
        )
    bad = [
        "/".join(flags)
        for field, flags in _NON_TEXT_TOKENIZER_FLAGS.items()
        if field in s
    ]
    if bad:
        raise ValueError(
            f"Tokenizer options ({', '.join(bad)}) cannot be used with "
            f"--endpoint-type {endpoint_type}."
        )


def _build_prompts(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    prompts: dict[str, Any] = {}
    isl: dict[str, Any] = {}
    if "isl_mean" in s:
        isl["mean"] = cli.isl_mean
    if "isl_stddev" in s:
        isl["stddev"] = cli.isl_stddev
    if isl:
        prompts["isl"] = isl
    osl: dict[str, Any] = {}
    if "osl_mean" in s:
        osl["mean"] = cli.osl_mean
    if "osl_stddev" in s:
        osl["stddev"] = cli.osl_stddev
    if osl:
        prompts["osl"] = osl
    if "isl_block_size" in s:
        prompts["block_size"] = cli.isl_block_size
    if "prompt_batch_size" in s:
        prompts["batch_size"] = cli.prompt_batch_size
    return prompts


def _build_prefix_prompts(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    prefix: dict[str, Any] = {}
    if "num_prefix_prompts" in s:
        prefix["pool_size"] = cli.num_prefix_prompts
    if "prefix_prompt_length" in s:
        prefix["length"] = cli.prefix_prompt_length
    if "shared_system_prompt_length" in s:
        prefix["shared_system_length"] = cli.shared_system_prompt_length
    if "user_context_prompt_length" in s:
        prefix["user_context_length"] = cli.user_context_prompt_length
    return prefix


def _build_rankings(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    rankings: dict[str, Any] = {}
    if "passages_mean" in s:
        rankings.setdefault("passages", {})["mean"] = cli.passages_mean
    if "passages_stddev" in s:
        rankings.setdefault("passages", {})["stddev"] = cli.passages_stddev
    if "passages_prompt_token_mean" in s:
        rankings.setdefault("passage_tokens", {})["mean"] = (
            cli.passages_prompt_token_mean
        )
    if "passages_prompt_token_stddev" in s:
        rankings.setdefault("passage_tokens", {})["stddev"] = (
            cli.passages_prompt_token_stddev
        )
    if "query_prompt_token_mean" in s:
        rankings.setdefault("query_tokens", {})["mean"] = cli.query_prompt_token_mean
    if "query_prompt_token_stddev" in s:
        rankings.setdefault("query_tokens", {})["stddev"] = (
            cli.query_prompt_token_stddev
        )
    return rankings


def _build_synthesis(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    synthesis: dict[str, Any] = {}
    mapping = {
        "synthesis_speedup_ratio": "speedup_ratio",
        "synthesis_prefix_len_multiplier": "prefix_len_multiplier",
        "synthesis_prefix_root_multiplier": "prefix_root_multiplier",
        "synthesis_prompt_len_multiplier": "prompt_len_multiplier",
        "synthesis_max_isl": "max_isl",
        "synthesis_max_osl": "max_osl",
    }
    for field, key in mapping.items():
        if field in s:
            synthesis[key] = getattr(cli, field)
    return synthesis


def _build_audio(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    audio: dict[str, Any] = {}
    if "audio_length_mean" in s:
        audio.setdefault("length", {})["mean"] = cli.audio_length_mean
    if "audio_length_stddev" in s:
        audio.setdefault("length", {})["stddev"] = cli.audio_length_stddev
    if "audio_batch_size" in s:
        audio["batch_size"] = cli.audio_batch_size
    if "audio_format" in s:
        audio["format"] = cli.audio_format
    if "audio_depths" in s:
        audio["depths"] = cli.audio_depths
    if "audio_sample_rates" in s:
        audio["sample_rates"] = cli.audio_sample_rates
    if "audio_num_channels" in s:
        audio["channels"] = cli.audio_num_channels
    return audio


def _build_images(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    images: dict[str, Any] = {}
    if "image_height_mean" in s:
        images.setdefault("height", {})["mean"] = cli.image_height_mean
    if "image_height_stddev" in s:
        images.setdefault("height", {})["stddev"] = cli.image_height_stddev
    if "image_width_mean" in s:
        images.setdefault("width", {})["mean"] = cli.image_width_mean
    if "image_width_stddev" in s:
        images.setdefault("width", {})["stddev"] = cli.image_width_stddev
    if "image_batch_size" in s:
        images["batch_size"] = cli.image_batch_size
    if "image_format" in s:
        images["format"] = cli.image_format
    return images


def _build_video(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    video: dict[str, Any] = {}
    fields = {
        "video_batch_size": "batch_size",
        "video_duration": "duration",
        "video_fps": "fps",
        "video_width": "width",
        "video_height": "height",
        "video_synth_type": "synth_type",
        "video_format": "format",
        "video_codec": "codec",
    }
    for field, key in fields.items():
        if field in s:
            video[key] = getattr(cli, field)
    video_audio: dict[str, Any] = {}
    if "video_audio_sample_rate" in s:
        video_audio["sample_rate"] = cli.video_audio_sample_rate
    if "video_audio_num_channels" in s:
        video_audio["channels"] = cli.video_audio_num_channels
    if video_audio:
        video["audio"] = video_audio
    return video


def _apply_dataset_type(d: dict[str, Any], cli: BaseModel, needs_text: bool) -> None:
    from aiperf.common.enums import DatasetType

    entries = cli.request_count or cli.num_sessions
    if cli.public_dataset:
        d["type"] = DatasetType.PUBLIC
        if entries is not None:
            d["entries"] = entries
    elif cli.input_file:
        d["type"] = DatasetType.FILE
    else:
        d["type"] = DatasetType.SYNTHETIC
        d.setdefault("entries", entries or cli.num_dataset_entries)
        if needs_text:
            d.setdefault("prompts", {}).setdefault("isl", {}).setdefault("mean", 550)


def _apply_sequence_distribution(d: dict[str, Any], cli: BaseModel) -> None:
    if not cli.sequence_distribution:
        return
    from aiperf.common.models.sequence_distribution import DistributionParser

    dist = DistributionParser.parse(cli.sequence_distribution)
    d.setdefault("prompts", {})["sequence_distribution"] = [
        {
            "isl": {"mean": p.input_seq_len, "stddev": p.input_seq_len_stddev},
            "osl": {"mean": p.output_seq_len, "stddev": p.output_seq_len_stddev},
            "probability": p.probability,
        }
        for p in dist.pairs
    ]


def _apply_turns(d: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    if "num_turns_mean" in s:
        d["turns"] = {"mean": cli.num_turns_mean, "stddev": cli.num_turns_stddev}
    if "turn_delay_mean" in s:
        d["turn_delay"] = {
            "mean": cli.turn_delay_mean,
            "stddev": cli.turn_delay_stddev,
        }
    if "turn_delay_ratio" in s:
        d["turn_delay_ratio"] = cli.turn_delay_ratio


def _apply_implicit_media_batch(d: dict[str, Any], s: set[str]) -> None:
    triggers = {
        "images": ("image_width_mean", "image_height_mean", "image_batch_size"),
        "audio": ("audio_length_mean", "audio_batch_size"),
        "video": (
            "video_batch_size",
            "video_width",
            "video_height",
            "video_duration",
            "video_fps",
            "video_synth_type",
        ),
    }
    for media_key, trig in triggers.items():
        media = d.get(media_key)
        if media and "batch_size" not in media and any(f in s for f in trig):
            media["batch_size"] = 1


def _needs_augment(s: set[str]) -> bool:
    return bool(_AUGMENT_TRIGGER_FIELDS & s)


_FLAT_DATASET_FIELDS = {
    "input_file": "path",
    "public_dataset": "name",
    "hf_subset": "hf_subset",
    "custom_dataset_type": "format",
    "dataset_sampling_strategy": "sampling",
    "num_dataset_entries": "entries",
}


def _flat_dataset_fields(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for field, key in _FLAT_DATASET_FIELDS.items():
        if field in s:
            d[key] = getattr(cli, field)
    return d


def _attach_subtables(d: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    builders = (
        ("prompts", _build_prompts),
        ("prefix_prompts", _build_prefix_prompts),
        ("rankings", _build_rankings),
        ("synthesis", _build_synthesis),
        ("audio", _build_audio),
        ("images", _build_images),
        ("video", _build_video),
    )
    for key, builder in builders:
        if value := builder(cli, s):
            d[key] = value


def _determine_needs_text(cli: BaseModel, s: set[str]) -> bool:
    from aiperf.plugin.plugins import get_endpoint_metadata

    endpoint_type = getattr(cli, "endpoint_type", None)
    if endpoint_type is None:
        return True
    meta = get_endpoint_metadata(endpoint_type)
    needs_text = meta.tokenizes_input or meta.produces_tokens
    if not needs_text:
        _validate_non_text_endpoint(endpoint_type, s)
    return needs_text


def build_dataset(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    """Build dataset dict."""
    needs_text = _determine_needs_text(cli, s)
    if cli.input_file and _needs_augment(s):
        return build_composed_dataset(cli, s)

    d = _flat_dataset_fields(cli, s)
    _attach_subtables(d, cli, s)
    _apply_dataset_type(d, cli, needs_text)
    _apply_sequence_distribution(d, cli)
    _apply_turns(d, cli, s)
    _apply_implicit_media_batch(d, s)
    return d


def _composed_augment_prefix(cli: BaseModel, s: set[str]) -> dict[str, Any] | None:
    if "prefix_prompt_length" in s or "num_prefix_prompts" in s:
        return {
            "length": cli.prefix_prompt_length or 128,
            "pool_size": cli.num_prefix_prompts or 1,
        }
    if "shared_system_prompt_length" in s:
        return {"length": cli.shared_system_prompt_length, "pool_size": 1}
    if "user_context_prompt_length" in s:
        return {"user_context_length": cli.user_context_prompt_length}
    return None


def _composed_augment_images(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    images: dict[str, Any] = {}
    if "image_batch_size" in s:
        images["batch_size"] = cli.image_batch_size
    if "image_height_mean" in s:
        images.setdefault("height", {})["mean"] = cli.image_height_mean
    if "image_width_mean" in s:
        images.setdefault("width", {})["mean"] = cli.image_width_mean
    if "image_format" in s:
        images["format"] = cli.image_format
    return images


def _composed_augment_audio(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    audio: dict[str, Any] = {}
    if "audio_batch_size" in s:
        audio["batch_size"] = cli.audio_batch_size
    if "audio_length_mean" in s:
        audio.setdefault("length", {})["mean"] = cli.audio_length_mean
    if "audio_format" in s:
        audio["format"] = cli.audio_format
    return audio


def _composed_augment_video(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    video: dict[str, Any] = {}
    if "video_batch_size" in s:
        video["batch_size"] = cli.video_batch_size
    if "video_duration" in s:
        video["duration"] = cli.video_duration
    if "video_fps" in s:
        video["fps"] = cli.video_fps
    return video


def build_composed_dataset(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    """File dataset with augmentation overlay."""
    from aiperf.common.enums import DatasetType

    source: dict[str, Any] = {"type": DatasetType.FILE}
    if "input_file" in s:
        source["path"] = cli.input_file
    if "custom_dataset_type" in s:
        source["format"] = cli.custom_dataset_type
    if "dataset_sampling_strategy" in s:
        source["sampling"] = cli.dataset_sampling_strategy

    augment: dict[str, Any] = {}
    if "osl_mean" in s:
        osl: dict[str, Any] = {"mean": cli.osl_mean}
        if "osl_stddev" in s:
            osl["stddev"] = cli.osl_stddev
        augment["osl"] = osl

    if (aug_prefix := _composed_augment_prefix(cli, s)) is not None:
        augment["prefix"] = aug_prefix
    if images := _composed_augment_images(cli, s):
        augment["images"] = images
    if audio := _composed_augment_audio(cli, s):
        augment["audio"] = audio
    if video := _composed_augment_video(cli, s):
        augment["video"] = video

    return {
        "type": DatasetType.COMPOSED,
        "source": source,
        "augment": augment,
        "entries": cli.request_count or cli.num_sessions,
        "random_seed": cli.random_seed,
    }
