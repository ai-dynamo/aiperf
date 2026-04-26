# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 -> v2 dataset-section converter.

Ports the logic of :mod:`aiperf.config._cli_dataset` (which reads a flat ``cli``
model) to the nested ``user.input.*`` layout exposed by ``UserConfig``. The
discrimination tree, augment-trigger logic, and field name mappings are
preserved 1:1; only the read-side traversal changes.

Returns a *dict* (not a wrapped ``DatasetConfig``) — wrapping with
``{"name": "main", **out}`` happens in the top-level converter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1 import UserConfig
    from aiperf.config.v1._input import InputConfig

# --- explicit-set helpers -------------------------------------------------


def _set(model: Any, field: str) -> bool:
    """Return True iff ``field`` was explicitly provided on ``model``."""
    return model is not None and field in model.model_fields_set


def _input(user: UserConfig) -> InputConfig | None:
    return user.input


# --- prompt / ISL / OSL ---------------------------------------------------


def _build_prompts(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    prompts: dict[str, Any] = {}
    pcfg = inp.prompt
    isl: dict[str, Any] = {}
    if _set(pcfg.input_tokens, "mean"):
        isl["mean"] = pcfg.input_tokens.mean
    if _set(pcfg.input_tokens, "stddev"):
        isl["stddev"] = pcfg.input_tokens.stddev
    if isl:
        prompts["isl"] = isl
    osl: dict[str, Any] = {}
    if _set(pcfg.output_tokens, "mean") and pcfg.output_tokens.mean is not None:
        osl["mean"] = pcfg.output_tokens.mean
    if _set(pcfg.output_tokens, "stddev") and pcfg.output_tokens.stddev is not None:
        osl["stddev"] = pcfg.output_tokens.stddev
    if osl:
        prompts["osl"] = osl
    if _set(pcfg.input_tokens, "block_size") and pcfg.input_tokens.block_size:
        prompts["block_size"] = pcfg.input_tokens.block_size
    if _set(pcfg, "batch_size"):
        prompts["batch_size"] = pcfg.batch_size
    return prompts


def _build_prefix_prompts(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    pp = inp.prompt.prefix_prompt
    out: dict[str, Any] = {}
    if _set(pp, "pool_size"):
        out["pool_size"] = pp.pool_size
    if _set(pp, "length"):
        out["length"] = pp.length
    if (
        _set(pp, "shared_system_prompt_length")
        and pp.shared_system_prompt_length is not None
    ):
        out["shared_system_length"] = pp.shared_system_prompt_length
    if (
        _set(pp, "user_context_prompt_length")
        and pp.user_context_prompt_length is not None
    ):
        out["user_context_length"] = pp.user_context_prompt_length
    return out


# --- rankings -------------------------------------------------------------


def _mean_stddev_pair(model: Any, mean_field: str, stddev_field: str) -> dict[str, Any]:
    """Return ``{"mean": ..., "stddev": ...}`` for whichever of the two fields was set."""
    out: dict[str, Any] = {}
    if _set(model, mean_field):
        out["mean"] = getattr(model, mean_field)
    if _set(model, stddev_field):
        out["stddev"] = getattr(model, stddev_field)
    return out


def _build_rankings(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    r = inp.rankings
    out: dict[str, Any] = {}
    if passages := _mean_stddev_pair(r.passages, "mean", "stddev"):
        out["passages"] = passages
    if passage_tokens := _mean_stddev_pair(
        r.passages, "prompt_token_mean", "prompt_token_stddev"
    ):
        out["passage_tokens"] = passage_tokens
    if query_tokens := _mean_stddev_pair(
        r.query, "prompt_token_mean", "prompt_token_stddev"
    ):
        out["query_tokens"] = query_tokens
    return out


# --- media (audio / images / video) ---------------------------------------


def _build_audio(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    a = inp.audio
    out: dict[str, Any] = {}
    length: dict[str, Any] = {}
    if _set(a.length, "mean"):
        length["mean"] = a.length.mean
    if _set(a.length, "stddev"):
        length["stddev"] = a.length.stddev
    if length:
        out["length"] = length
    if _set(a, "batch_size"):
        out["batch_size"] = a.batch_size
    if _set(a, "format"):
        out["format"] = a.format
    if _set(a, "depths"):
        out["depths"] = a.depths
    if _set(a, "sample_rates"):
        out["sample_rates"] = a.sample_rates
    if _set(a, "num_channels"):
        out["channels"] = a.num_channels
    return out


def _build_images(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    img = inp.image
    out: dict[str, Any] = {}
    height: dict[str, Any] = {}
    if _set(img.height, "mean"):
        height["mean"] = img.height.mean
    if _set(img.height, "stddev"):
        height["stddev"] = img.height.stddev
    if height:
        out["height"] = height
    width: dict[str, Any] = {}
    if _set(img.width, "mean"):
        width["mean"] = img.width.mean
    if _set(img.width, "stddev"):
        width["stddev"] = img.width.stddev
    if width:
        out["width"] = width
    if _set(img, "batch_size"):
        out["batch_size"] = img.batch_size
    if _set(img, "format"):
        out["format"] = img.format
    return out


def _build_video(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    v = inp.video
    out: dict[str, Any] = {}
    direct = {
        "batch_size": "batch_size",
        "duration": "duration",
        "fps": "fps",
        "width": "width",
        "height": "height",
        "synth_type": "synth_type",
        "format": "format",
        "codec": "codec",
    }
    for src, dst in direct.items():
        if _set(v, src):
            out[dst] = getattr(v, src)
    audio: dict[str, Any] = {}
    if _set(v.audio, "sample_rate"):
        audio["sample_rate"] = v.audio.sample_rate
    if _set(v.audio, "channels"):
        audio["channels"] = v.audio.channels
    if audio:
        out["audio"] = audio
    return out


# --- top-level dataset assembly -------------------------------------------


def _flat_dataset_fields(user: UserConfig) -> dict[str, Any]:
    """Top-level fields that move through verbatim."""
    inp = _input(user)
    if inp is None:
        return {}
    out: dict[str, Any] = {}
    if _set(inp, "file"):
        out["path"] = inp.file
    if _set(inp, "public_dataset"):
        out["dataset"] = inp.public_dataset
    if _set(inp, "hf_dataset_subset") and inp.hf_dataset_subset is not None:
        out["hf_subset"] = inp.hf_dataset_subset
    if _set(inp, "custom_dataset_type") and inp.custom_dataset_type is not None:
        out["format"] = inp.custom_dataset_type
    if (
        _set(inp, "dataset_sampling_strategy")
        and inp.dataset_sampling_strategy is not None
    ):
        out["sampling"] = inp.dataset_sampling_strategy
    if (
        inp.conversation is not None
        and "num_dataset_entries" in inp.conversation.model_fields_set
    ):
        out["entries"] = inp.conversation.num_dataset_entries
    return out


def _attach_subtables(d: dict[str, Any], user: UserConfig) -> None:
    builders = (
        ("prompts", _build_prompts),
        ("prefix_prompts", _build_prefix_prompts),
        ("rankings", _build_rankings),
        ("audio", _build_audio),
        ("images", _build_images),
        ("video", _build_video),
    )
    for key, builder in builders:
        if value := builder(user):
            d[key] = value


def _resolve_entries(user: UserConfig) -> int | None:
    """request_count (loadgen) > conversation.num > conversation.num_dataset_entries.

    Mirrors ``cli.request_count or cli.num_sessions`` from the flat model.
    """
    if user.loadgen is not None and user.loadgen.request_count is not None:
        return user.loadgen.request_count
    inp = _input(user)
    if (
        inp is not None
        and inp.conversation is not None
        and inp.conversation.num is not None
    ):
        return inp.conversation.num
    return None


def _apply_dataset_type(d: dict[str, Any], user: UserConfig, needs_text: bool) -> None:
    from aiperf.common.enums import DatasetType

    inp = _input(user)
    entries = _resolve_entries(user)
    if inp is not None and inp.public_dataset:
        d["type"] = DatasetType.PUBLIC
        if entries is not None:
            d["entries"] = entries
        return
    if inp is not None and inp.file:
        d["type"] = DatasetType.FILE
        return
    d["type"] = DatasetType.SYNTHETIC
    fallback_entries = entries
    if fallback_entries is None and inp is not None and inp.conversation is not None:
        fallback_entries = inp.conversation.num_dataset_entries
    d.setdefault("entries", fallback_entries)
    if needs_text:
        d.setdefault("prompts", {}).setdefault("isl", {}).setdefault("mean", 550)


def _apply_sequence_distribution(d: dict[str, Any], user: UserConfig) -> None:
    inp = _input(user)
    if inp is None or not inp.prompt.sequence_distribution:
        return
    from aiperf.common.models.sequence_distribution import DistributionParser

    dist = DistributionParser.parse(inp.prompt.sequence_distribution)
    d.setdefault("prompts", {})["sequence_distribution"] = [
        {
            "isl": {"mean": p.input_seq_len, "stddev": p.input_seq_len_stddev},
            "osl": {"mean": p.output_seq_len, "stddev": p.output_seq_len_stddev},
            "probability": p.probability,
        }
        for p in dist.pairs
    ]


def _apply_turns(d: dict[str, Any], user: UserConfig) -> None:
    inp = _input(user)
    if inp is None or inp.conversation is None:
        return
    turn = inp.conversation.turn
    delay = turn.delay
    if "mean" in turn.model_fields_set or "stddev" in turn.model_fields_set:
        d["turns"] = {"mean": turn.mean, "stddev": turn.stddev}
    if "mean" in delay.model_fields_set or "stddev" in delay.model_fields_set:
        d["turn_delay"] = {"mean": delay.mean, "stddev": delay.stddev}
    if "ratio" in delay.model_fields_set:
        d["turn_delay_ratio"] = delay.ratio


def _apply_implicit_media_batch(d: dict[str, Any], user: UserConfig) -> None:
    """Default batch_size=1 when any media-shape field is set without batch_size."""
    inp = _input(user)
    if inp is None:
        return
    img_set = (
        inp.image.model_fields_set
        | inp.image.width.model_fields_set
        | inp.image.height.model_fields_set
    )
    aud_set = inp.audio.model_fields_set | inp.audio.length.model_fields_set
    vid_set = inp.video.model_fields_set
    triggers = {
        "images": (
            "width",
            "height",
            "batch_size",
            "mean",
        ),
        "audio": ("length", "batch_size", "mean"),
        "video": (
            "batch_size",
            "width",
            "height",
            "duration",
            "fps",
            "synth_type",
        ),
    }
    set_maps = {"images": img_set, "audio": aud_set, "video": vid_set}
    for media_key, trig in triggers.items():
        media = d.get(media_key)
        if (
            media
            and "batch_size" not in media
            and any(f in set_maps[media_key] for f in trig)
        ):
            media["batch_size"] = 1


# --- augment-trigger detection (composed dataset) -------------------------


def _augment_triggers_set(user: UserConfig) -> bool:
    """True iff any composed-dataset augment trigger was explicitly set.

    Mirrors ``_AUGMENT_TRIGGER_FIELDS`` from ``_cli_dataset.py``.
    """
    inp = _input(user)
    if inp is None:
        return False
    pp = inp.prompt.prefix_prompt
    if (
        "pool_size" in pp.model_fields_set
        or "length" in pp.model_fields_set
        or pp.shared_system_prompt_length is not None
        or pp.user_context_prompt_length is not None
    ):
        return True
    out_tokens = inp.prompt.output_tokens
    if "mean" in out_tokens.model_fields_set and out_tokens.mean is not None:
        return True
    if "batch_size" in inp.image.model_fields_set:
        return True
    if "batch_size" in inp.audio.model_fields_set:
        return True
    return "batch_size" in inp.video.model_fields_set


# --- composed-dataset assembly -------------------------------------------


def _composed_augment_prefix(user: UserConfig) -> dict[str, Any] | None:
    inp = _input(user)
    if inp is None:
        return None
    pp = inp.prompt.prefix_prompt
    if "length" in pp.model_fields_set or "pool_size" in pp.model_fields_set:
        return {
            "length": pp.length or 128,
            "pool_size": pp.pool_size or 1,
        }
    if pp.shared_system_prompt_length is not None:
        return {"length": pp.shared_system_prompt_length, "pool_size": 1}
    if pp.user_context_prompt_length is not None:
        return {"user_context_length": pp.user_context_prompt_length}
    return None


def _composed_augment_images(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    img = inp.image
    out: dict[str, Any] = {}
    if "batch_size" in img.model_fields_set:
        out["batch_size"] = img.batch_size
    if "mean" in img.height.model_fields_set:
        out.setdefault("height", {})["mean"] = img.height.mean
    if "mean" in img.width.model_fields_set:
        out.setdefault("width", {})["mean"] = img.width.mean
    if "format" in img.model_fields_set:
        out["format"] = img.format
    return out


def _composed_augment_audio(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    a = inp.audio
    out: dict[str, Any] = {}
    if "batch_size" in a.model_fields_set:
        out["batch_size"] = a.batch_size
    if "mean" in a.length.model_fields_set:
        out.setdefault("length", {})["mean"] = a.length.mean
    if "format" in a.model_fields_set:
        out["format"] = a.format
    return out


def _composed_augment_video(user: UserConfig) -> dict[str, Any]:
    inp = _input(user)
    if inp is None:
        return {}
    v = inp.video
    out: dict[str, Any] = {}
    if "batch_size" in v.model_fields_set:
        out["batch_size"] = v.batch_size
    if "duration" in v.model_fields_set:
        out["duration"] = v.duration
    if "fps" in v.model_fields_set:
        out["fps"] = v.fps
    return out


def _composed_source(user: UserConfig) -> dict[str, Any]:
    from aiperf.common.enums import DatasetType

    source: dict[str, Any] = {"type": DatasetType.FILE}
    inp = _input(user)
    if inp is None:
        return source
    if "file" in inp.model_fields_set and inp.file is not None:
        source["path"] = inp.file
    if (
        "custom_dataset_type" in inp.model_fields_set
        and inp.custom_dataset_type is not None
    ):
        source["format"] = inp.custom_dataset_type
    if (
        "dataset_sampling_strategy" in inp.model_fields_set
        and inp.dataset_sampling_strategy is not None
    ):
        source["sampling"] = inp.dataset_sampling_strategy
    return source


def _composed_augment_osl(user: UserConfig) -> dict[str, Any] | None:
    inp = _input(user)
    if inp is None:
        return None
    out_tokens = inp.prompt.output_tokens
    if "mean" not in out_tokens.model_fields_set or out_tokens.mean is None:
        return None
    osl: dict[str, Any] = {"mean": out_tokens.mean}
    if "stddev" in out_tokens.model_fields_set and out_tokens.stddev is not None:
        osl["stddev"] = out_tokens.stddev
    return osl


def _composed_augment(user: UserConfig) -> dict[str, Any]:
    augment: dict[str, Any] = {}
    if (osl := _composed_augment_osl(user)) is not None:
        augment["osl"] = osl
    if (aug_prefix := _composed_augment_prefix(user)) is not None:
        augment["prefix"] = aug_prefix
    if images := _composed_augment_images(user):
        augment["images"] = images
    if audio := _composed_augment_audio(user):
        augment["audio"] = audio
    if video := _composed_augment_video(user):
        augment["video"] = video
    return augment


def _build_composed_dataset(user: UserConfig) -> dict[str, Any]:
    """File dataset with augmentation overlay."""
    from aiperf.common.enums import DatasetType

    inp = _input(user)
    random_seed = inp.random_seed if inp is not None else None
    return {
        "type": DatasetType.COMPOSED,
        "source": _composed_source(user),
        "augment": _composed_augment(user),
        "entries": _resolve_entries(user),
        "random_seed": random_seed,
    }


# --- text-endpoint validation -------------------------------------------


_NON_TEXT_TEXT_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("prompt.input_tokens.mean", "--isl/--prompt-input-tokens-mean"),
    ("prompt.input_tokens.stddev", "--isl-stddev/--prompt-input-tokens-stddev"),
    (
        "prompt.input_tokens.block_size",
        "--isl-block-size/--prompt-input-tokens-block-size",
    ),
    ("prompt.batch_size", "--prompt-batch-size/--batch-size-text"),
    ("prompt.sequence_distribution", "--seq-dist/--sequence-distribution"),
    ("prompt.prefix_prompt.length", "--prompt-prefix-length"),
    ("prompt.prefix_prompt.pool_size", "--prompt-prefix-pool-size"),
)


def _is_set_path(root: Any, path: str) -> bool:
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        if cur is None:
            return False
        cur = getattr(cur, p, None)
    if cur is None:
        return False
    return parts[-1] in cur.model_fields_set


def _determine_needs_text(user: UserConfig) -> bool:
    """True iff the configured endpoint type tokenizes input or produces tokens.

    Reads ``user.endpoint.type`` (if available) and consults the plugin
    registry; on a non-text endpoint, raises if any text-only flag was set.
    """
    from aiperf.plugin.plugins import get_endpoint_metadata

    endpoint_type = (
        getattr(user.endpoint, "type", None) if user.endpoint is not None else None
    )
    if endpoint_type is None:
        return True
    meta = get_endpoint_metadata(endpoint_type)
    needs_text = meta.tokenizes_input or meta.produces_tokens
    if not needs_text and user.input is not None:
        violations = [
            flag
            for path, flag in _NON_TEXT_TEXT_TRIGGERS
            if _is_set_path(user.input, path)
        ]
        if violations:
            raise ValueError(
                f"{', '.join(violations)} cannot be used with --endpoint-type "
                f"{endpoint_type}."
            )
    return needs_text


# --- public entrypoint ----------------------------------------------------


def build_dataset(user: UserConfig) -> dict[str, Any]:
    """Build a single dataset entry (without the wrapping ``name`` field).

    Discriminates among synthetic / file / public / composed based on the
    populated ``user.input.*`` fields, then assembles ~70 sub-fields into the
    correct v2 dataset shape.

    Returns:
        A dict suitable for ``DatasetConfig.model_validate({"name": "main", **out})``.
    """
    needs_text = _determine_needs_text(user)
    inp = _input(user)
    if inp is not None and inp.file and _augment_triggers_set(user):
        return _build_composed_dataset(user)

    d = _flat_dataset_fields(user)
    _attach_subtables(d, user)
    _apply_dataset_type(d, user, needs_text)
    _apply_sequence_distribution(d, user)
    _apply_turns(d, user)
    _apply_implicit_media_batch(d, user)
    if inp is not None and "random_seed" in inp.model_fields_set:
        d["random_seed"] = inp.random_seed
    return d
