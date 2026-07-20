# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> AIPerfConfig dataset-section converter.

Translates the flat ``cli.<field>`` layout (modality, prompt, conversation,
file, etc.) into the AIPerfConfig dataset dict (discrimination tree,
augment-trigger logic, field name mappings).

Returns a *dict* (not a wrapped ``DatasetConfig``) — wrapping with
``{"name": "main", **out}`` happens in the top-level converter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.config.flags._section_fields import (
    TOKENIZER_FIELDS,
)

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def _normalize_sample_rate_khz(value: float | int) -> float:
    """Auto-convert Hz inputs to kHz for the kHz-scoped audio schema.

    Pre-redesign cyclopts CLI flags accepted Hz-shaped values like ``16000``
    while the kHz schema caps at 96 (96 kHz = pro audio). Auto-divide
    values above the cap by 1000 to preserve the historical invocation
    shape. Why: chaos suite + tutorials still pass ``16000`` for 16 kHz
    speech audio.
    """
    v = float(value)
    return v / 1000.0 if v > 96.0 else v


# --- explicit-set helpers -------------------------------------------------


def _set(model: Any, field: str) -> bool:
    """Return True iff ``field`` was explicitly provided on ``model``."""
    return model is not None and field in model.model_fields_set


# --- prompt / ISL / OSL ---------------------------------------------------


def _build_prompts(cli: CLIConfig) -> dict[str, Any]:
    prompts: dict[str, Any] = {}
    s = cli.model_fields_set
    isl: dict[str, Any] = {}
    if "prompt_input_tokens_mean" in s:
        # Magic-list flags hoist the list to the sweep block; the base
        # config keeps the first element as a placeholder so AIPerfConfig
        # validation passes (each variation overrides per-cell at expand
        # time). See `_promote_cli_dataset_magic_lists`.
        v = cli.prompt_input_tokens_mean
        isl["mean"] = v[0] if isinstance(v, list) and v else v
    if "prompt_input_tokens_stddev" in s:
        v = cli.prompt_input_tokens_stddev
        isl["stddev"] = v[0] if isinstance(v, list) and v else v
    if isl:
        prompts["isl"] = isl
    osl: dict[str, Any] = {}
    if "prompt_output_tokens_mean" in s and cli.prompt_output_tokens_mean is not None:
        v = cli.prompt_output_tokens_mean
        osl["mean"] = v[0] if isinstance(v, list) and v else v
    if (
        "prompt_output_tokens_stddev" in s
        and cli.prompt_output_tokens_stddev is not None
    ):
        v = cli.prompt_output_tokens_stddev
        osl["stddev"] = v[0] if isinstance(v, list) and v else v
    if osl:
        prompts["osl"] = osl
    if "prompt_input_tokens_block_size" in s and cli.prompt_input_tokens_block_size:
        prompts["block_size"] = cli.prompt_input_tokens_block_size
    if "prompt_batch_size" in s:
        prompts["batch_size"] = cli.prompt_batch_size
    return prompts


def _build_prefix_prompts(cli: CLIConfig) -> dict[str, Any]:
    s = cli.model_fields_set
    out: dict[str, Any] = {}
    if "prompt_prefix_pool_size" in s:
        out["pool_size"] = cli.prompt_prefix_pool_size
    if "prompt_prefix_length" in s:
        out["length"] = cli.prompt_prefix_length
    if (
        "prompt_prefix_shared_system_length" in s
        and cli.prompt_prefix_shared_system_length is not None
    ):
        out["shared_system_length"] = cli.prompt_prefix_shared_system_length
    if (
        "prompt_prefix_user_context_length" in s
        and cli.prompt_prefix_user_context_length is not None
    ):
        out["user_context_length"] = cli.prompt_prefix_user_context_length
    return out


# --- rankings -------------------------------------------------------------


def _mean_stddev_pair(
    cli: CLIConfig, mean_field: str, stddev_field: str
) -> dict[str, Any]:
    """Return ``{"mean": ..., "stddev": ...}`` for whichever of the two fields was set."""
    s = cli.model_fields_set
    out: dict[str, Any] = {}
    if mean_field in s:
        out["mean"] = getattr(cli, mean_field)
    if stddev_field in s:
        out["stddev"] = getattr(cli, stddev_field)
    return out


def _build_rankings(cli: CLIConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if passages := _mean_stddev_pair(
        cli, "rankings_passages_mean", "rankings_passages_stddev"
    ):
        out["passages"] = passages
    if passage_tokens := _mean_stddev_pair(
        cli,
        "rankings_passages_prompt_token_mean",
        "rankings_passages_prompt_token_stddev",
    ):
        out["passage_tokens"] = passage_tokens
    if query_tokens := _mean_stddev_pair(
        cli, "rankings_query_prompt_token_mean", "rankings_query_prompt_token_stddev"
    ):
        out["query_tokens"] = query_tokens
    return out


# --- media (audio / images / video) ---------------------------------------


def _build_audio(cli: CLIConfig) -> dict[str, Any]:
    s = cli.model_fields_set
    out: dict[str, Any] = {}
    length: dict[str, Any] = {}
    if "audio_length_mean" in s:
        length["mean"] = cli.audio_length_mean
    if "audio_length_stddev" in s:
        length["stddev"] = cli.audio_length_stddev
    if length:
        out["length"] = length
    if "audio_batch_size" in s:
        out["batch_size"] = cli.audio_batch_size
    if "audio_format" in s:
        out["format"] = cli.audio_format
    if "audio_depths" in s:
        out["depths"] = cli.audio_depths
    if "audio_sample_rates" in s:
        out["sample_rates"] = [
            _normalize_sample_rate_khz(r) for r in cli.audio_sample_rates
        ]
    if "audio_num_channels" in s:
        out["channels"] = cli.audio_num_channels
    return out


def _build_images(cli: CLIConfig) -> dict[str, Any]:
    s = cli.model_fields_set
    out: dict[str, Any] = {}
    height: dict[str, Any] = {}
    if "image_height_mean" in s:
        height["mean"] = cli.image_height_mean
    if "image_height_stddev" in s:
        height["stddev"] = cli.image_height_stddev
    if height:
        out["height"] = height
    width: dict[str, Any] = {}
    if "image_width_mean" in s:
        width["mean"] = cli.image_width_mean
    if "image_width_stddev" in s:
        width["stddev"] = cli.image_width_stddev
    if width:
        out["width"] = width
    direct = {
        "image_batch_size": "batch_size",
        "image_format": "format",
        "image_source": "source",
        "image_source_sampling": "source_sampling",
    }
    for src, dst in direct.items():
        if src in s:
            out[dst] = getattr(cli, src)
    return out


def _build_video(cli: CLIConfig) -> dict[str, Any]:
    s = cli.model_fields_set
    out: dict[str, Any] = {}
    direct = {
        "video_batch_size": "batch_size",
        "video_duration": "duration",
        "video_fps": "fps",
        "video_width": "width",
        "video_height": "height",
        "video_synth_type": "synth_type",
        "video_format": "format",
        "video_codec": "codec",
    }
    for src, dst in direct.items():
        if src in s:
            out[dst] = getattr(cli, src)
    audio: dict[str, Any] = {}
    if "video_audio_sample_rate" in s:
        audio["sample_rate"] = _normalize_sample_rate_khz(cli.video_audio_sample_rate)
    if "video_audio_channels" in s:
        audio["channels"] = cli.video_audio_channels
    if "video_audio_codec" in s:
        audio["codec"] = cli.video_audio_codec
    if "video_audio_depth" in s:
        audio["depth"] = cli.video_audio_depth
    if audio:
        out["audio"] = audio
    return out


# --- top-level dataset assembly -------------------------------------------


def _parse_dataset_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(
                f"Invalid --dataset-filter {item!r}; expected non-empty key=value"
            )
        if key in filters:
            raise ValueError(f"Duplicate --dataset-filter key {key!r}")
        filters[key] = value
    return filters


# (cli field, dataset key, forward explicit None). The _set gate alone
# forwards explicitly-set booleans of both polarities (e.g.
# --no-open-loop-replay); a bool is never None, so keep_none is
# irrelevant for the boolean rows.
_VERBATIM_DATASET_FIELDS = (
    ("input_file", "path", True),
    ("public_dataset", "dataset", True),
    ("hf_dataset_subset", "hf_subset", False),
    ("custom_dataset_type", "format", False),
    ("dataset_sampling_strategy", "sampling", False),
    ("conversation_num_dataset_entries", "entries", True),
    ("trace_session_sample_ratio", "trace_session_sample_ratio", False),
    ("max_idle_gap_cap_seconds", "max_idle_gap_cap_seconds", False),
    ("replay_speedup", "replay_speedup", False),
    ("open_loop_replay", "open_loop_replay", False),
    ("open_loop_strict", "open_loop_strict", False),
    ("omit_kv_hints", "omit_kv_hints", False),
    ("force_min_tokens", "force_min_tokens", False),
)


def _flat_dataset_fields(cli: CLIConfig) -> dict[str, Any]:
    """Top-level fields that move through verbatim."""
    out: dict[str, Any] = {}
    for field, key, keep_none in _VERBATIM_DATASET_FIELDS:
        value = getattr(cli, field)
        if _set(cli, field) and (keep_none or value is not None):
            out[key] = value
    if _set(cli, "dataset_filters"):
        out["filters"] = _parse_dataset_filters(cli.dataset_filters)
    return out


def _attach_subtables(d: dict[str, Any], cli: CLIConfig) -> None:
    builders = (
        ("prompts", _build_prompts),
        ("prefix_prompts", _build_prefix_prompts),
        ("rankings", _build_rankings),
        ("audio", _build_audio),
        ("images", _build_images),
        ("video", _build_video),
    )
    for key, builder in builders:
        if value := builder(cli):
            d[key] = value


def _resolve_entries(cli: CLIConfig) -> int | None:
    """Return user-set entry count, or None if no source field was user-set.

    Resolution order:
      1. ``cli.conversation_num_dataset_entries`` (explicitly set) — the
         field that directly names the dataset entry count wins when the user
         set it on purpose.
      2. ``cli.conversation_num`` (explicitly set) — ``--num-conversations N``
         names the count of unique sessions/conversations to materialize.
         Wins over ``--request-count`` so users sweeping concurrency or
         request_count against a fixed-size dataset get exactly N unique
         conversations (the runner recycles them to fill request_count).
      3. ``cli.request_count`` (explicitly set) — fallback so a single
         ``--request-count N`` invocation produces ``N`` unique entries when
         the user did not pin the conversation count separately.

    Returns None when none was explicitly set. The caller MUST omit the
    ``entries`` key from the output dict in that case so the dataset class's
    own Pydantic default applies (``SyntheticDataset.entries=100``;
    ``File/Public.entries=None``). Emitting ``entries=None`` into the
    dict would crash AIPerfConfig validation on synthetic
    (``int_type, got NoneType``).
    """
    s = cli.model_fields_set
    if "conversation_num_dataset_entries" in s:
        return cli.conversation_num_dataset_entries
    if "conversation_num" in s:
        # Magic-list sweep on --num-conversations: phase.sessions varies
        # per-variation, but the dataset entries pool needs ONE scalar.
        # Use max(list) so every variation has its full unique-session set.
        v = cli.conversation_num
        if isinstance(v, list):
            return max(v) if v else None
        return v
    if "request_count" in s:
        v = cli.request_count
        if isinstance(v, list):
            return max(v) if v else None
        return v
    return None


def _apply_dataset_type(d: dict[str, Any], cli: CLIConfig, needs_text: bool) -> None:
    from aiperf.common.enums import DatasetType

    entries = _resolve_entries(cli)
    if cli.public_dataset:
        d["type"] = DatasetType.PUBLIC
        if entries is not None:
            d["entries"] = entries
        # PublicDataset doesn't carry per-modality subtables.
        for key in (
            "prompts",
            "prefix_prompts",
            "rankings",
            "audio",
            "images",
            "video",
        ):
            d.pop(key, None)
        return
    if cli.input_file:
        d["type"] = DatasetType.FILE
        if entries is not None:
            d["entries"] = entries
        # FileDataset only carries synthesis + osl as auxiliary fields. The
        # synthetic-only subtables are dropped here; --osl is handled by
        # _apply_file_osl.
        for key in (
            "prompts",
            "prefix_prompts",
            "rankings",
            "audio",
            "images",
            "video",
        ):
            d.pop(key, None)
        return
    d["type"] = DatasetType.SYNTHETIC
    if entries is not None:
        d.setdefault("entries", entries)
    # else: omit; SyntheticDataset.entries=100 default applies
    if needs_text:
        d.setdefault("prompts", {}).setdefault("isl", {}).setdefault("mean", 550)


def _apply_sequence_distribution(d: dict[str, Any], cli: CLIConfig) -> None:
    if not cli.prompt_sequence_distribution:
        return
    from aiperf.common.models.sequence_distribution import DistributionParser

    dist = DistributionParser.parse(cli.prompt_sequence_distribution)
    d.setdefault("prompts", {})["sequence_distribution"] = [
        {
            "isl": {"mean": p.input_seq_len, "stddev": p.input_seq_len_stddev},
            "osl": {"mean": p.output_seq_len, "stddev": p.output_seq_len_stddev},
            "probability": p.probability,
        }
        for p in dist.pairs
    ]


def _apply_random_range_ratio(d: dict[str, Any], cli: CLIConfig) -> None:
    if not cli.prompt_random_range_ratio:
        return
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    RangeRatioDistribution.parse_cli_value(
        cli.prompt_random_range_ratio, cli.prompt_random_range_ratio_mode
    )
    d.setdefault("prompts", {})["random_range_ratio"] = cli.prompt_random_range_ratio
    if "prompt_random_range_ratio_mode" in cli.model_fields_set:
        d["prompts"]["random_range_ratio_mode"] = cli.prompt_random_range_ratio_mode


def _apply_turns(d: dict[str, Any], cli: CLIConfig) -> None:
    fields_set = cli.model_fields_set
    if (
        "conversation_turn_mean" in fields_set
        or "conversation_turn_stddev" in fields_set
    ):
        # Magic-list on --conversation-turn-mean: keep first element as
        # placeholder; the sweep block carries the full list.
        v = cli.conversation_turn_mean
        turn_mean = v[0] if isinstance(v, list) and v else v
        d["turns"] = {
            "mean": turn_mean,
            "stddev": cli.conversation_turn_stddev,
        }
    if (
        "conversation_turn_delay_mean" in fields_set
        or "conversation_turn_delay_stddev" in fields_set
    ):
        d["turn_delay"] = {
            "mean": cli.conversation_turn_delay_mean,
            "stddev": cli.conversation_turn_delay_stddev,
        }
    if "conversation_turn_delay_ratio" in fields_set:
        d["turn_delay_ratio"] = cli.conversation_turn_delay_ratio


def _apply_synthesis(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``cli.synthesis_*`` fields to ``FileDataset.synthesis``.

    Synthesis is only meaningful for trace-format file datasets (the
    Synthesizer is invoked from BaseTraceDatasetLoader). The synthesis
    fields live flat on CLIConfig (post-Task-13), so we only emit a
    ``synthesis`` sub-dict when the resulting dataset is a FileDataset and
    at least one field was explicitly set or carries a non-default value.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") != DatasetType.FILE:
        return
    set_fields = cli.model_fields_set
    out: dict[str, Any] = {}
    for cli_attr, dst_key in (
        ("synthesis_speedup_ratio", "speedup_ratio"),
        ("synthesis_prefix_len_multiplier", "prefix_len_multiplier"),
        ("synthesis_prefix_root_multiplier", "prefix_root_multiplier"),
        ("synthesis_prompt_len_multiplier", "prompt_len_multiplier"),
        ("synthesis_output_len_multiplier", "output_len_multiplier"),
        ("synthesis_max_isl", "max_isl"),
        ("synthesis_max_osl", "max_osl"),
    ):
        if cli_attr in set_fields:
            value = getattr(cli, cli_attr)
            if value is not None:
                out[dst_key] = value
    if out:
        d["synthesis"] = out


def _apply_implicit_media_batch(d: dict[str, Any], cli: CLIConfig) -> None:
    """Default batch_size=1 when any media-shape field is set without batch_size."""
    s = cli.model_fields_set
    triggers = {
        "images": (
            "image_width_mean",
            "image_width_stddev",
            "image_height_mean",
            "image_height_stddev",
            "image_batch_size",
            "image_source",
            "image_source_sampling",
        ),
        "audio": ("audio_length_mean", "audio_length_stddev", "audio_batch_size"),
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


# --- file-dataset incompatibility validation -----------------------------


_FILE_DATASET_INCOMPATIBLE_TRIGGERS: tuple[tuple[str, str], ...] = (
    (
        "prompt_prefix_length",
        "--prompt-prefix-length/--prefix-prompt-length",
    ),
    (
        "prompt_prefix_pool_size",
        "--prompt-prefix-pool-size/--prefix-prompt-pool-size",
    ),
    (
        "prompt_prefix_shared_system_length",
        "--shared-system-prompt-length",
    ),
    (
        "prompt_prefix_user_context_length",
        "--user-context-prompt-length",
    ),
    # ISL / prompt-shaping flags only apply to synthetic generation. File
    # datasets (including mooncake_trace) source ISL from the trace records
    # themselves — silently dropping these flags hid bugs (e.g. trace replay
    # using the hardcoded block_size=512 fallback while ignoring user's
    # --isl-block-size). Reject at convert-time with a clear error.
    (
        "prompt_input_tokens_mean",
        "--isl/--prompt-input-tokens-mean/--synthetic-input-tokens-mean",
    ),
    (
        "prompt_input_tokens_stddev",
        "--isl-stddev/--prompt-input-tokens-stddev/--synthetic-input-tokens-stddev",
    ),
    (
        "prompt_input_tokens_block_size",
        "--isl-block-size/--prompt-input-tokens-block-size/--synthetic-input-tokens-block-size",
    ),
    ("prompt_batch_size", "--prompt-batch-size/--batch-size-text"),
    ("prompt_sequence_distribution", "--seq-dist/--sequence-distribution"),
    ("prompt_random_range_ratio", "--random-range-ratio"),
    ("image_batch_size", "--image-batch-size"),
    ("image_source", "--image-source"),
    ("image_source_sampling", "--image-source-sampling"),
    ("audio_batch_size", "--audio-batch-size"),
    ("video_batch_size", "--video-batch-size"),
)


def _reject_file_dataset_incompatible(cli: CLIConfig) -> None:
    """Reject synthetic-only flags when --input-file is set.

    Flags rejected: prefix prompts, ISL shaping (--isl/--isl-stddev/
    --isl-block-size), --prompt-batch-size, --seq-dist, multimodal
    batch_size. These are only meaningful for synthetic datasets; on file
    datasets they were previously silently dropped by the strip in
    ``_apply_dataset_type`` (or worse, leaked through and crashed
    AIPerfConfig validation with ``extra_forbidden`` on the
    FileDataset). Surface a clear message instead.

    --osl / --osl-stddev are NOT rejected — they're routed onto
    ``FileDataset.osl`` by ``_apply_file_osl`` as a per-record fallback.
    """
    if not cli.input_file:
        return
    s = cli.model_fields_set
    violations = [
        flag for attr, flag in _FILE_DATASET_INCOMPATIBLE_TRIGGERS if attr in s
    ]
    if violations:
        raise ValueError(
            f"{', '.join(violations)} is only supported with synthetic datasets; "
            "use a synthetic dataset (no --input-file) to apply synthetic-only "
            "prompt shaping (ISL, prefix prompts, multimodal generation, etc)."
        )


_BASETEN_ONLY_TRACE_FLAGS: tuple[tuple[str, str], ...] = (
    ("trace_session_sample_ratio", "--trace-session-sample-ratio"),
    ("replay_speedup", "--replay-speedup"),
    ("max_idle_gap_cap_seconds", "--max-idle-gap-cap-seconds"),
)

# Boolean knobs are never None, so an explicit set of either polarity
# (membership in model_fields_set) is the guard signal.
_BASETEN_ONLY_TRACE_BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("open_loop_replay", "--open-loop-replay/--no-open-loop-replay"),
    ("open_loop_strict", "--open-loop-strict"),
    ("omit_kv_hints", "--omit-kv-hints"),
    ("force_min_tokens", "--force-min-tokens/--no-force-min-tokens"),
)


def _reject_baseten_only_trace_flags(cli: CLIConfig) -> None:
    """Reject baseten_trace-only replay knobs on incompatible datasets.

    These knobs are only consumed by the baseten_trace loader; on any other
    dataset they would silently no-op (or crash AIPerfConfig validation with
    a raw ``extra_forbidden`` on synthetic/public datasets), hiding user
    error. Rejected when the dataset cannot be file-based (--public-dataset
    set or no --input-file), or when --custom-dataset-type is explicitly set
    to a different loader.
    """
    from aiperf.plugin.enums import CustomDatasetType

    set_flags = [
        flag
        for attr, flag in _BASETEN_ONLY_TRACE_FLAGS
        if attr in cli.model_fields_set and getattr(cli, attr) is not None
    ]
    set_flags += [
        flag
        for attr, flag in _BASETEN_ONLY_TRACE_BOOL_FLAGS
        if attr in cli.model_fields_set
    ]
    if not set_flags:
        return
    msg = f"{', '.join(set_flags)} is only supported by the baseten_trace loader"
    if cli.public_dataset or not cli.input_file:
        raise ValueError(
            f"{msg}; provide --input-file and --custom-dataset-type baseten_trace."
        )
    if (
        cli.custom_dataset_type is not None
        and cli.custom_dataset_type != CustomDatasetType.BASETEN_TRACE
    ):
        raise ValueError(
            f"{msg}, but --custom-dataset-type is {cli.custom_dataset_type}."
        )


def _reject_baseten_trace_unsupported_synthesis(cli: CLIConfig) -> None:
    """Reject synthesis knobs that cannot apply to baseten_trace replay.

    baseten_trace replay is paced by --replay-speedup; synthesis speedup
    rescales the raw trace timestamps before replay, so it compounds with
    --replay-speedup and desyncs the closed-loop think-time subtraction and
    the open-loop idle-gap cap (both divide by replay_speedup only).
    Prompt-shaping multipliers reshape hash_ids while the wire still sends
    the original recorded prompt, so the forwarded KV hints would desync
    from the prompt. Output-length synthesis and the max-ISL/OSL filter/cap
    remain valid. The auto-detected dataset-type path is guarded at load
    time by the loader.
    """
    from aiperf.plugin.enums import CustomDatasetType

    if cli.custom_dataset_type != CustomDatasetType.BASETEN_TRACE:
        return
    if cli.synthesis_speedup_ratio != 1.0:
        raise ValueError(
            "--synthesis-speedup-ratio is not supported with "
            "--custom-dataset-type baseten_trace; use --replay-speedup to "
            "scale replay pacing."
        )
    reshaping_flags = [
        flag
        for attr, flag, default in (
            (
                "synthesis_prefix_len_multiplier",
                "--synthesis-prefix-len-multiplier",
                1.0,
            ),
            (
                "synthesis_prefix_root_multiplier",
                "--synthesis-prefix-root-multiplier",
                1,
            ),
            (
                "synthesis_prompt_len_multiplier",
                "--synthesis-prompt-len-multiplier",
                1.0,
            ),
        )
        if getattr(cli, attr) != default
    ]
    if reshaping_flags:
        verb = "is" if len(reshaping_flags) == 1 else "are"
        raise ValueError(
            f"{', '.join(reshaping_flags)} {verb} not supported with "
            "--custom-dataset-type baseten_trace: it replays recorded "
            "prompts verbatim, so hash-reshaping synthesis cannot change "
            "the sent prompt and would desync the forwarded hash_ids KV "
            "hints."
        )


def _reject_baseten_trace_extra_input_collisions(cli: CLIConfig) -> None:
    """Reject --extra-inputs keys the baseten_trace loader injects per-turn.

    Loader-injected per-turn values (``min_tokens`` from the recorded output
    length, ``hash_ids``/``block_size`` KV hints) overwrite endpoint-level
    extras, so the user's value would be silently clobbered on the wire.
    Each collision has an opt-out flag that stops the injection so the user
    value goes through. ``max_tokens`` is not guarded: user extras win over
    the loader for that key.
    """
    from aiperf.plugin.enums import CustomDatasetType

    if cli.custom_dataset_type != CustomDatasetType.BASETEN_TRACE:
        return
    extra = dict(cli.extra_inputs or ())
    collisions: list[tuple[str, str]] = []
    if cli.force_min_tokens and "min_tokens" in extra:
        collisions.append(("min_tokens", "--no-force-min-tokens"))
    if not cli.omit_kv_hints:
        collisions.extend(
            (key, "--omit-kv-hints")
            for key in ("hash_ids", "block_size")
            if key in extra
        )
    if collisions:
        raise ValueError(
            "; ".join(
                f"--extra-inputs {key} is overwritten per-turn by the "
                f"baseten_trace loader; pass {flag} to send your value instead"
                for key, flag in collisions
            )
        )


def _apply_file_osl(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--osl`` onto ``FileDataset.osl`` when --input-file is set.

    Synthetic datasets carry OSL on ``prompts.osl`` (handled by
    ``_build_prompts``). For file datasets, route the same value to the
    flat ``FileDataset.osl`` field as a per-record fallback.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") != DatasetType.FILE:
        return
    s = cli.model_fields_set
    if "prompt_output_tokens_mean" not in s or cli.prompt_output_tokens_mean is None:
        return
    v = cli.prompt_output_tokens_mean
    osl: dict[str, Any] = {"mean": v[0] if isinstance(v, list) and v else v}
    if (
        "prompt_output_tokens_stddev" in s
        and cli.prompt_output_tokens_stddev is not None
    ):
        osl["stddev"] = cli.prompt_output_tokens_stddev
    d["osl"] = osl


def _apply_inter_turn_delay_cap(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--inter-turn-delay-cap-seconds`` onto ``FileDataset``.

    The cap clamps per-turn replay delays (read from JSONL trace files)
    so long pre-recorded waits don't stall the benchmark. Only meaningful
    on file datasets (synthetic datasets compute their own delays).
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") != DatasetType.FILE:
        return
    if (
        "inter_turn_delay_cap_seconds" not in cli.model_fields_set
        or cli.inter_turn_delay_cap_seconds is None
    ):
        return
    d["inter_turn_delay_cap_seconds"] = cli.inter_turn_delay_cap_seconds


# --- text-endpoint validation -------------------------------------------


_NON_TEXT_TEXT_TRIGGERS: tuple[tuple[str, str], ...] = (
    (
        "prompt_input_tokens_mean",
        "--isl/--prompt-input-tokens-mean/--synthetic-input-tokens-mean",
    ),
    (
        "prompt_input_tokens_stddev",
        "--isl-stddev/--prompt-input-tokens-stddev/--synthetic-input-tokens-stddev",
    ),
    (
        "prompt_input_tokens_block_size",
        "--isl-block-size/--prompt-input-tokens-block-size/--synthetic-input-tokens-block-size",
    ),
    ("prompt_batch_size", "--prompt-batch-size/--batch-size-text"),
    ("prompt_sequence_distribution", "--seq-dist/--sequence-distribution"),
    ("prompt_random_range_ratio", "--random-range-ratio"),
)

# Tokenizer options are also rejected for non-tokenizing endpoints
# (image_retrieval, embeddings, etc.).
_NON_TEXT_TOKENIZER_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("tokenizer_name", "--tokenizer"),
    ("trust_remote_code", "--tokenizer-trust-remote-code"),
    ("tokenizer_revision", "--tokenizer-revision"),
)


def _determine_needs_text(cli: CLIConfig) -> bool:
    """True iff the configured endpoint type tokenizes input or produces tokens.

    Reads ``cli.endpoint_type`` (if available) and consults the plugin
    registry; on a non-text endpoint, raises if any text-only flag was set.
    """
    from aiperf.plugin.plugins import get_endpoint_metadata

    endpoint_type = getattr(cli, "endpoint_type", None)
    if endpoint_type is None:
        return True
    meta = get_endpoint_metadata(endpoint_type)
    needs_text = meta.tokenizes_input or meta.produces_tokens
    if not needs_text:
        s = cli.model_fields_set
        violations = [flag for attr, flag in _NON_TEXT_TEXT_TRIGGERS if attr in s]
        if violations:
            raise ValueError(
                f"{', '.join(violations)} cannot be used with --endpoint-type "
                f"{endpoint_type}."
            )
        prefix_prompt_fields = {f for f in s if f.startswith("prompt_prefix_")}
        if prefix_prompt_fields:
            raise ValueError(
                f"Prefix prompt options ({', '.join(sorted(prefix_prompt_fields))}) "
                f"cannot be used with --endpoint-type {endpoint_type}."
            )
    if not needs_text:
        tok_set = cli.model_fields_set & TOKENIZER_FIELDS
        tok_violations = [
            flag for field, flag in _NON_TEXT_TOKENIZER_TRIGGERS if field in tok_set
        ]
        if tok_violations:
            raise ValueError(
                f"Tokenizer options ({', '.join(tok_violations)}) cannot be used "
                f"with --endpoint-type {endpoint_type}."
            )
    return needs_text


# --- public entrypoint ----------------------------------------------------


def build_dataset(cli: CLIConfig) -> dict[str, Any]:
    """Build a single dataset entry (without the wrapping ``name`` field).

    Discriminates among synthetic / file / public based on the populated
    flat input fields and sub-config holders on ``cli``, then assembles the
    sub-fields into the correct dataset shape. Rejects synthetic-only
    flags (prefix, ISL shaping, batch_size, seq-dist, multimodal batch_size)
    when --input-file is set.

    Returns:
        A dict suitable for ``DatasetConfig.model_validate({"name": "main", **out})``.
    """
    needs_text = _determine_needs_text(cli)
    _reject_file_dataset_incompatible(cli)
    _reject_baseten_only_trace_flags(cli)
    _reject_baseten_trace_unsupported_synthesis(cli)
    _reject_baseten_trace_extra_input_collisions(cli)
    if cli.dataset_filters and not cli.public_dataset:
        raise ValueError("--dataset-filter requires --public-dataset")

    d = _flat_dataset_fields(cli)
    _attach_subtables(d, cli)
    _apply_dataset_type(d, cli, needs_text)
    _apply_sequence_distribution(d, cli)
    _apply_random_range_ratio(d, cli)
    _apply_turns(d, cli)
    _apply_synthesis(d, cli)
    _apply_implicit_media_batch(d, cli)
    _apply_file_osl(d, cli)
    _apply_inter_turn_delay_cap(d, cli)
    if "random_seed" in cli.model_fields_set:
        d["random_seed"] = cli.random_seed
    return d
