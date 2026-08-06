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
    if "cache_bust" in s:
        prompts["cache_bust"] = {"target": cli.cache_bust}
    if "prompt_corpus" in s and cli.prompt_corpus is not None:
        prompts["corpus"] = cli.prompt_corpus
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
    ("hf_weka_dataset", "hf_weka_dataset", False),
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


def _implies_public_dataset(cli: CLIConfig) -> bool:
    """True when the CLI selects a public dataset (explicit or via --hf-weka-dataset).

    ``--hf-weka-dataset`` alone auto-selects ``weka_hf``; reject/type paths that
    key off ``cli.public_dataset`` must treat that as public too.
    """
    if cli.public_dataset:
        return True
    return _set(cli, "hf_weka_dataset") and cli.hf_weka_dataset is not None


def _flat_dataset_fields(cli: CLIConfig) -> dict[str, Any]:
    """Top-level fields that move through verbatim."""
    out: dict[str, Any] = {}
    for field, key, keep_none in _VERBATIM_DATASET_FIELDS:
        value = getattr(cli, field)
        if _set(cli, field) and (keep_none or value is not None):
            out[key] = value
    if _set(cli, "dataset_filters"):
        out["filters"] = _parse_dataset_filters(cli.dataset_filters)
    # --hf-weka-dataset alone auto-selects --public-dataset weka_hf (docs +
    # PublicDataset._validate_weka_hf expect the pairing).
    if "hf_weka_dataset" in out and "dataset" not in out:
        from aiperf.plugin.enums import PublicDatasetType

        out["dataset"] = PublicDatasetType.WEKA_HF
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
    # ``entries`` legitimately absorbs the --num-conversations / --request-count
    # fallback (see _resolve_entries), so its mere presence cannot tell whether
    # the user actually named --num-dataset-entries. Public-dataset provenance
    # must report num_dataset_entries ONLY for explicit intent, so whenever the
    # converter writes ``entries`` from a fallback it pins the
    # ``_entries_explicit`` sentinel to the true intent.
    entries_explicit = "conversation_num_dataset_entries" in cli.model_fields_set
    if _implies_public_dataset(cli) or d.get("dataset") is not None:
        d["type"] = DatasetType.PUBLIC
        if entries is not None:
            d["entries"] = entries
            d["_entries_explicit"] = entries_explicit
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

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
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
        ("allow_dataset_wrap", "allow_dataset_wrap"),
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
    # ISL mean/stddev only apply to synthetic generation. File datasets
    # (including mooncake_trace) source ISL from the trace records themselves --
    # silently dropping these flags hid bugs. Reject at convert-time with a clear
    # error. NOTE: --isl-block-size is NOT here -- it is the hash-id block
    # granularity that the trace loaders DO consume, so it is routed onto
    # FileDataset.block_size by _apply_block_size (and rejected only for weka,
    # which carries its own inline per-block sizes).
    (
        "prompt_input_tokens_mean",
        "--isl/--prompt-input-tokens-mean/--synthetic-input-tokens-mean",
    ),
    (
        "prompt_input_tokens_stddev",
        "--isl-stddev/--prompt-input-tokens-stddev/--synthetic-input-tokens-stddev",
    ),
    ("prompt_batch_size", "--prompt-batch-size/--batch-size-text"),
    ("prompt_sequence_distribution", "--seq-dist/--sequence-distribution"),
    ("image_batch_size", "--image-batch-size"),
    ("image_source", "--image-source"),
    ("image_source_sampling", "--image-source-sampling"),
    ("audio_batch_size", "--audio-batch-size"),
    ("video_batch_size", "--video-batch-size"),
    # Multi-turn conversation GENERATION knobs: synthetic-only. Trace datasets
    # carry their own turn structure, so these previously crashed FileDataset
    # validation with extra_forbidden. Reject with a clear message instead.
    ("conversation_turn_mean", "--conversation-turn-mean/--session-turns-mean"),
    ("conversation_turn_stddev", "--conversation-turn-stddev/--session-turns-stddev"),
    ("conversation_turn_delay_mean", "--conversation-turn-delay-mean"),
    ("conversation_turn_delay_stddev", "--conversation-turn-delay-stddev"),
    ("conversation_turn_delay_ratio", "--conversation-turn-delay-ratio"),
)


def _reject_file_dataset_incompatible(cli: CLIConfig) -> None:
    """Reject synthetic-only flags on FILE or PUBLIC (trace) datasets.

    Flags rejected: prefix prompts, ISL shaping (--isl/--isl-stddev/
    --isl-block-size), --prompt-batch-size, --seq-dist, multimodal batch_size,
    and multi-turn conversation generation (--conversation-turn-*). These are
    only meaningful for synthetic datasets; on file/public trace datasets the
    value source is the trace, so they were previously silently dropped by the
    ``_apply_dataset_type`` strip (or worse, leaked through and crashed
    AIPerfConfig validation with ``extra_forbidden``). Surface a clear message
    instead.

    --osl / --osl-stddev are NOT rejected — they're routed onto
    ``FileDataset.osl`` / ``PublicDataset.osl`` by ``_apply_file_osl`` as a
    per-record fallback.
    """
    if not cli.input_file and not _implies_public_dataset(cli):
        return
    s = cli.model_fields_set
    violations = [
        flag for attr, flag in _FILE_DATASET_INCOMPATIBLE_TRIGGERS if attr in s
    ]
    if violations:
        raise ValueError(
            f"{', '.join(violations)} is only supported with synthetic datasets; "
            "remove --input-file / --public-dataset (use a synthetic dataset) to "
            "apply synthetic-only prompt shaping (ISL, prefix prompts, multimodal "
            "generation, multi-turn conversation, etc)."
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
    if _implies_public_dataset(cli) or not cli.input_file:
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
    """Route ``--osl`` onto ``FileDataset.osl`` / ``PublicDataset.osl``.

    Synthetic datasets carry OSL on ``prompts.osl`` (handled by
    ``_build_prompts``). For file AND public (HF-backed weka) trace datasets,
    route the same value to the flat ``osl`` field as a per-record fallback
    (both models carry it).
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
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


def _apply_file_block_size(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--isl-block-size`` onto ``FileDataset.block_size`` when
    --input-file is set.

    Synthetic datasets carry block size on ``prompts.block_size`` (handled by
    ``_build_prompts``, then stripped for file datasets in
    ``_apply_dataset_type``). For trace/file datasets, route the same value to
    the flat ``FileDataset.block_size`` field, which overrides the loader's
    ``default_block_size`` plugin metadata (needed when a trace was recorded at
    a block size different from its loader default).
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") != DatasetType.FILE:
        return
    if (
        "prompt_input_tokens_block_size" not in cli.model_fields_set
        or cli.prompt_input_tokens_block_size is None
    ):
        return
    d["block_size"] = cli.prompt_input_tokens_block_size


def _apply_inter_turn_delay_cap(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--inter-turn-delay-cap-seconds`` onto ``FileDataset``/``PublicDataset``.

    The cap clamps per-turn replay delays (read from trace records) so long
    pre-recorded waits don't stall the benchmark. Meaningful on file-based
    trace datasets AND the HF-backed ``weka_hf`` public dataset (both replay
    traces); synthetic datasets compute their own delays. ``PublicDataset``
    carries the same field, so route there too.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    if (
        "inter_turn_delay_cap_seconds" not in cli.model_fields_set
        or cli.inter_turn_delay_cap_seconds is None
    ):
        return
    d["inter_turn_delay_cap_seconds"] = cli.inter_turn_delay_cap_seconds


def _apply_trace_delay_flags(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route trace-replay delay knobs onto ``FileDataset``/``PublicDataset``.

    These flags live on both FILE and PUBLIC dataset models. Ignore/think-only
    affect ``Turn.delay`` / ``Turn.timestamp`` at load time; the trace idle cap
    is consumed by the AgentX runtime watchdog without rewriting either field.
    Without this route the CLI flags are silently dropped (YAML / scenario
    paths set the fields directly).
    Mutual exclusivity of ignore vs think-only is enforced by the dataset
    model validators after conversion.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    s = cli.model_fields_set
    if "ignore_trace_delays" in s:
        d["ignore_trace_delays"] = cli.ignore_trace_delays
    if "use_think_time_only" in s:
        d["use_think_time_only"] = cli.use_think_time_only
    if "trace_idle_gap_cap_seconds" in s and cli.trace_idle_gap_cap_seconds is not None:
        d["trace_idle_gap_cap_seconds"] = cli.trace_idle_gap_cap_seconds


def _is_weka_dataset(d: dict[str, Any]) -> bool:
    """True when ``d`` is a Weka file format or Weka public-dataset alias."""
    from aiperf.common.enums import DatasetType

    dtype = d.get("type")
    if dtype == DatasetType.FILE:
        fmt = d.get("format")
        return fmt is not None and str(fmt) == "weka_trace"
    if dtype == DatasetType.PUBLIC:
        public = d.get("dataset")
        return public is not None and "weka" in str(public).lower()
    return False


def _is_definitely_non_weka_dataset(d: dict[str, Any]) -> bool:
    """True only when ``d`` is provably NOT weka.

    A content-auto-detected FILE trace has ``format=None`` here (weka itself is
    auto-detected via ``can_load``), so it is ambiguous, not provably non-weka.
    Only an explicit non-weka file format or a non-weka public dataset is
    definite.
    """
    from aiperf.common.enums import DatasetType

    dtype = d.get("type")
    if dtype == DatasetType.FILE:
        fmt = d.get("format")
        # format=None is a content-auto-detected trace (possibly weka): ambiguous.
        if fmt is None:
            return False
        return str(fmt) != "weka_trace"
    if dtype == DatasetType.PUBLIC:
        public = d.get("dataset")
        return public is None or "weka" not in str(public).lower()
    # Synthetic and every other non-file/public type is provably not weka.
    return True


def _apply_max_context_length(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--max-context-length`` onto Weka ``FileDataset``/``PublicDataset``.

    Weka selection filters traces whose *recorded* peak prompt+output exceeds
    this ceiling before applying ``--num-dataset-entries``. Provably non-Weka
    datasets do not consume the flag (there is no DatasetManager tokenize path),
    so reject those loudly. An auto-detected FILE trace (format=None) may still
    be weka, so it is allowed through -- the loader ignores the field if it does
    not implement the filter.
    """
    if (
        "max_context_length" not in cli.model_fields_set
        or cli.max_context_length is None
    ):
        return
    if _is_definitely_non_weka_dataset(d):
        raise ValueError(
            "--max-context-length only applies to Weka trace replay "
            "(--custom-dataset-type weka_trace or a weka --public-dataset). "
            "It filters by recorded peak prompt+output length at load time; "
            "other formats do not implement this filter. Drop "
            "--max-context-length or switch to a Weka dataset."
        )
    d["max_context_length"] = cli.max_context_length


# FILE custom_dataset_types whose loaders decode hash_ids into token blocks of
# ``block_size`` (the BaseTraceDatasetLoader family). These CONSUME
# --isl-block-size. weka is excluded deliberately: it carries its own inline
# per-block sizes in the trace, so a global override is meaningless there.
_BLOCK_SIZE_TRACE_FORMATS = frozenset(
    {
        "mooncake_trace",
        "bailian_trace",
        "burst_gpt_trace",
        "sagemaker_data_capture",
    }
)


def _apply_block_size(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--isl-block-size`` onto ``FileDataset.block_size`` for hash-id
    trace datasets.

    block_size is fundamentally a TRACE field: the mooncake/bailian/burst_gpt/
    sagemaker loaders decode each ``hash_id`` into a cached block of this many
    tokens (default 512 / 16 from plugin metadata). Synthetic datasets carry it
    on ``prompts.block_size`` (written by ``_build_prompts``, then stripped for
    FILE/PUBLIC by ``_apply_dataset_type``), so for FILE traces it must be
    re-routed onto the flat field here -- after the strip -- or it silently
    no-ops (the loader falls back to the hardcoded default, ignoring the user).

    Weka datasets REJECT it: weka traces carry their own inline per-block sizes,
    so an override would be wrong. Public datasets reject it too (the only
    public traces are weka; non-trace public datasets do not decode hash blocks).
    """
    from aiperf.common.enums import DatasetType

    s = cli.model_fields_set
    if (
        "prompt_input_tokens_block_size" not in s
        or not cli.prompt_input_tokens_block_size
    ):
        return
    # Synthetic: handled by _build_prompts -> prompts.block_size.
    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return

    fmt = str(d["format"]) if d.get("format") is not None else None
    public = str(d["dataset"]) if d.get("dataset") is not None else None
    is_weka = fmt == "weka_trace" or (public is not None and "weka" in public.lower())
    if is_weka:
        raise ValueError(
            "--isl-block-size is not supported with weka datasets: weka traces "
            "carry their own inline per-block sizes. Drop --isl-block-size to "
            "replay the trace's own block sizes."
        )
    if fmt in _BLOCK_SIZE_TRACE_FORMATS:
        d["block_size"] = cli.prompt_input_tokens_block_size
        return
    # Content-auto-detected FILE traces (mooncake/bailian/etc. selected via
    # can_load, not an explicit --custom-dataset-type) have format=None here.
    # We cannot tell hash-id traces from other file shapes at convert time, so
    # restore #1159 by routing onto the flat FileDataset.block_size; loaders
    # that do not decode hash blocks ignore it. Explicit weka is already
    # rejected above.
    if fmt is None and d.get("type") == DatasetType.FILE:
        d["block_size"] = cli.prompt_input_tokens_block_size
        return
    raise ValueError(
        "--isl-block-size only applies to synthetic generation or hash-id trace "
        "replay (mooncake_trace, bailian_trace, burst_gpt_trace, "
        "sagemaker_data_capture). The selected dataset does not decode hash-id "
        "token blocks; drop --isl-block-size."
    )


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
    if cli.dataset_filters and not _implies_public_dataset(cli):
        raise ValueError("--dataset-filter requires --public-dataset")

    d = _flat_dataset_fields(cli)
    _attach_subtables(d, cli)
    _apply_dataset_type(d, cli, needs_text)
    _apply_sequence_distribution(d, cli)
    _apply_turns(d, cli)
    _apply_synthesis(d, cli)
    _apply_implicit_media_batch(d, cli)
    _apply_file_osl(d, cli)
    # block_size for FILE hash-id traces is owned by _apply_block_size (which
    # also rejects weka / non-hash-id formats). Do not also call
    # _apply_file_block_size — that helper is broader and redundant here.
    _apply_inter_turn_delay_cap(d, cli)
    _apply_trace_delay_flags(d, cli)
    _apply_max_context_length(d, cli)
    _apply_block_size(d, cli)
    _apply_corpus_and_cache_bust(d, cli)
    if "random_seed" in cli.model_fields_set:
        d["random_seed"] = cli.random_seed
    return d


def _apply_corpus_and_cache_bust(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--prompt-corpus`` / ``--cache-bust`` onto FILE/PUBLIC datasets.

    Corpus is re-attached as ``prompts.corpus`` after the synthetic prompts
    subtable is stripped. Cache-bust remains a top-level ``cache_bust`` field.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    s = cli.model_fields_set
    if "prompt_corpus" in s and cli.prompt_corpus is not None:
        prompts = d.get("prompts")
        if not isinstance(prompts, dict):
            prompts = {}
            d["prompts"] = prompts
        prompts["corpus"] = cli.prompt_corpus
    if "cache_bust" in s:
        d["cache_bust"] = {"target": cli.cache_bust}
