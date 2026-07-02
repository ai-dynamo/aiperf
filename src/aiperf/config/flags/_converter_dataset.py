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
        prompts["prompt_corpus"] = cli.prompt_corpus
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


def _resolve_public_dataset_type(cli: CLIConfig) -> Any | None:
    """Resolve the public dataset type, honoring the --hf-weka-dataset auto-select.

    ``--hf-weka-dataset`` names a HuggingFace repo for the generic weka_hf
    loader, so passing it implies ``--public-dataset weka_hf``. Auto-select it
    when the user didn't name a dataset, so the repo flag works on its own
    instead of erroring. Setting it alongside any other --public-dataset or
    --custom-dataset-type is an error.
    """
    from aiperf.plugin.enums import PublicDatasetType

    if _set(cli, "public_dataset"):
        return cli.public_dataset
    if _set(cli, "hf_weka_dataset") and cli.hf_weka_dataset is not None:
        if _set(cli, "custom_dataset_type") and cli.custom_dataset_type is not None:
            raise ValueError(
                "--hf-weka-dataset selects --public-dataset weka_hf, which "
                "cannot be combined with --custom-dataset-type"
            )
        # ``weka_hf`` is registered by the Weka loader plugin (plugins.yaml).
        # Resolve via the extensible enum so this wiring stays correct once
        # the loader is registered; surface a clear error if it isn't yet.
        try:
            return PublicDatasetType("weka_hf")
        except ValueError as exc:
            raise ValueError(
                "--hf-weka-dataset requires the 'weka_hf' public dataset loader, "
                "which is not registered in this build. Install/enable the Weka "
                "dataset loader plugin, or pass --public-dataset/--custom-dataset-type."
            ) from exc
    return None


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


def _flat_dataset_fields(cli: CLIConfig) -> dict[str, Any]:
    """Top-level fields that move through verbatim."""
    out: dict[str, Any] = {}
    if _set(cli, "input_file"):
        out["path"] = cli.input_file
    if (public := _resolve_public_dataset_type(cli)) is not None:
        out["dataset"] = public
    if _set(cli, "hf_weka_dataset") and cli.hf_weka_dataset is not None:
        out["hf_weka_dataset"] = cli.hf_weka_dataset
    if _set(cli, "hf_dataset_subset") and cli.hf_dataset_subset is not None:
        out["hf_subset"] = cli.hf_dataset_subset
    if _set(cli, "dataset_filters"):
        out["filters"] = _parse_dataset_filters(cli.dataset_filters)
    if _set(cli, "custom_dataset_type") and cli.custom_dataset_type is not None:
        out["format"] = cli.custom_dataset_type
    if (
        _set(cli, "dataset_sampling_strategy")
        and cli.dataset_sampling_strategy is not None
    ):
        out["sampling"] = cli.dataset_sampling_strategy
    if "conversation_num_dataset_entries" in cli.model_fields_set:
        out["entries"] = cli.conversation_num_dataset_entries
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
    # ``_entries_explicit`` sentinel to the true intent. The PublicDataset model
    # maps the sentinel onto its ``entries_explicit`` field; a YAML/programmatic
    # config that sets ``entries`` directly (no sentinel) is treated as explicit.
    entries_explicit = "conversation_num_dataset_entries" in cli.model_fields_set
    # ``d["dataset"]`` is set by ``_flat_dataset_fields`` and reflects the
    # --hf-weka-dataset auto-select, so it (not raw ``cli.public_dataset``)
    # is the source of truth for the PUBLIC discriminator.
    if d.get("dataset") is not None:
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
        # FileDataset only carries synthesis + osl + trace-replay knobs as
        # auxiliary fields. The synthetic-only subtables are dropped here;
        # --osl is handled by _apply_file_osl, trace fields by
        # _apply_weka_trace_fields.
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
    # turns / turn_delay / turn_delay_ratio are SyntheticDataset-only fields
    # (multi-turn conversation GENERATION knobs). File/public trace datasets
    # source their turn structure from the trace itself and forbid these keys
    # (extra="forbid"), so writing them there crashed AIPerfConfig validation
    # with an opaque ``extra_forbidden``. Only emit them for synthetic; the
    # FILE rejection below surfaces a clear flag-level error instead.
    from aiperf.common.enums import DatasetType

    if d.get("type") != DatasetType.SYNTHETIC:
        return
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

    Synthesis is meaningful for trace datasets -- file-based traces AND the
    HF-backed ``weka_hf`` public dataset, where ``--max-isl``/``--max-osl`` cap
    the replayed traces' lengths (the weka loader reads ``synthesis.max_osl``).
    The synthesis fields live flat on CLIConfig, so we only emit a ``synthesis``
    sub-dict when the resulting dataset is a FILE/PUBLIC dataset and at least
    one field was explicitly set.
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


def _reject_non_synthetic_incompatible(cli: CLIConfig) -> None:
    """Reject synthetic-only flags on FILE or PUBLIC (trace) datasets.

    Flags rejected: prefix prompts, ISL shaping (--isl/--isl-stddev/
    --isl-block-size), --prompt-batch-size, --seq-dist, multimodal batch_size,
    and multi-turn conversation generation (--conversation-turn-* / the
    --session-turns-* aliases). These are only meaningful for synthetic
    datasets; on file/public trace datasets the value source is the trace, so
    they were previously silently dropped by the ``_apply_dataset_type`` strip
    (or worse, leaked through and crashed AIPerfConfig with ``extra_forbidden``
    -- including the magic-list forms promoted onto a sweep block the
    FileDataset/PublicDataset forbids). Surface a clear flag-level message
    instead. Runs in ``build_dataset`` before the magic-list promoter, so it
    catches both scalar and list forms.

    --osl / --osl-stddev are NOT rejected -- they route onto ``FileDataset.osl``
    / ``PublicDataset.osl`` (via ``_apply_file_osl``) as a per-record fallback.
    """
    if not cli.input_file and _resolve_public_dataset_type(cli) is None:
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


def _apply_file_osl(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--osl`` onto ``FileDataset.osl`` / ``PublicDataset.osl``.

    Synthetic datasets carry OSL on ``prompts.osl`` (handled by
    ``_build_prompts``). For file AND public (HF-backed weka) trace datasets,
    route the same value to the flat ``osl`` field as a per-record fallback
    (both models carry it; the composer's ``_osl_distribution`` reads it for
    either type).
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


def _apply_inter_turn_delay_cap(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--inter-turn-delay-cap-seconds`` onto ``FileDataset``/``PublicDataset``.

    The cap clamps per-turn replay delays (read from trace records) so long
    pre-recorded waits don't stall the benchmark. Meaningful on file-based
    trace datasets AND the HF-backed ``weka_hf`` public dataset (both replay
    traces through ``WekaTraceLoader``); synthetic datasets compute their own
    delays. ``PublicDataset`` carries the same field, so route there too.
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
    raise ValueError(
        "--isl-block-size only applies to synthetic generation or hash-id trace "
        "replay (mooncake_trace, bailian_trace, burst_gpt_trace, "
        "sagemaker_data_capture). The selected dataset does not decode hash-id "
        "token blocks; drop --isl-block-size."
    )


def _apply_corpus_and_cache_bust(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--prompt-corpus`` and ``--cache-bust`` onto FILE/PUBLIC datasets.

    For synthetic datasets these live on ``prompts.{prompt_corpus,cache_bust}``
    (written by ``_build_prompts``). FILE/PUBLIC datasets drop the entire
    ``prompts`` subtable in ``_apply_dataset_type``, so the values must be
    routed to the flat top-level ``FileDataset``/``PublicDataset`` fields here
    -- AFTER the strip -- or ``--prompt-corpus`` and ``--cache-bust`` silently
    no-op on trace replay (the corpus reconstruction falls back to the loader
    default and KV-cache-bust experiments do nothing). Both fields exist on
    ``FileDataset`` and ``PublicDataset``; ``cache_bust`` is a ``CacheBustConfig``
    keyed by ``target`` (same shape ``_build_prompts`` uses for synthetic).
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    s = cli.model_fields_set
    if "prompt_corpus" in s and cli.prompt_corpus is not None:
        d["prompt_corpus"] = cli.prompt_corpus
    if "cache_bust" in s:
        d["cache_bust"] = {"target": cli.cache_bust}


def _apply_weka_trace_fields(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route the Weka/trace-replay flags onto ``FileDataset``/``PublicDataset``.

    ``--ignore-trace-delays``, ``--use-think-time-only`` and
    ``--trace-idle-gap-cap-seconds`` govern trace replay timing;
    ``--max-context-length`` drops over-length conversations at load. These
    carry onto file-based trace datasets and onto the HF-backed ``weka_hf``
    public dataset (both replay Weka traces through ``WekaTraceLoader``).
    Synthetic datasets compute their own delays and have no over-length
    trace records to drop, so they are skipped.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    s = cli.model_fields_set
    if "ignore_trace_delays" in s:
        d["ignore_trace_delays"] = cli.ignore_trace_delays
    if "use_think_time_only" in s:
        d["use_think_time_only"] = cli.use_think_time_only
    if "use_end_to_start_delays" in s:
        d["use_end_to_start_delays"] = cli.use_end_to_start_delays
    if "trace_idle_gap_cap_seconds" in s and cli.trace_idle_gap_cap_seconds is not None:
        d["trace_idle_gap_cap_seconds"] = cli.trace_idle_gap_cap_seconds
    if "max_context_length" in s and cli.max_context_length is not None:
        d["max_context_length"] = cli.max_context_length


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
    _reject_non_synthetic_incompatible(cli)
    # Resolve rather than read cli.public_dataset directly so the
    # --hf-weka-dataset auto-select reaches the composer's accurate
    # "does not support --dataset-filter" error instead of this one.
    if cli.dataset_filters and _resolve_public_dataset_type(cli) is None:
        raise ValueError("--dataset-filter requires --public-dataset")

    d = _flat_dataset_fields(cli)
    _attach_subtables(d, cli)
    _apply_dataset_type(d, cli, needs_text)
    _apply_sequence_distribution(d, cli)
    _apply_turns(d, cli)
    _apply_synthesis(d, cli)
    _apply_implicit_media_batch(d, cli)
    _apply_file_osl(d, cli)
    _apply_block_size(d, cli)
    _apply_inter_turn_delay_cap(d, cli)
    _apply_weka_trace_fields(d, cli)
    _apply_corpus_and_cache_bust(d, cli)
    if "random_seed" in cli.model_fields_set:
        d["random_seed"] = cli.random_seed
    return d
