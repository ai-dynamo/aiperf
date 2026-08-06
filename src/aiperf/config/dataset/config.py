# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Datasets - Data source variants and their discriminated union.

Content-generation sub-configs (prompts, images, audio, video, rankings) and
trace synthesis sub-configs live in sibling ``content.py`` / ``trace.py`` /
``video.py`` modules and are re-exported here so existing
``from aiperf.config.dataset import X`` imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    DatasetFormat,
    DatasetType,
)
from aiperf.config.base import BaseConfig
from aiperf.config.dataset.content import (
    AudioConfig,
    CacheBustConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    PromptSelectionConfig,
    RankingsConfig,
)
from aiperf.config.dataset.trace import (
    SynthesisConfig,
)
from aiperf.config.dataset.video import (
    VIDEO_AUDIO_CODEC_MAP,
    VideoAudioConfig,
    VideoConfig,
)
from aiperf.config.loader.normalizers import _hoist_synthetic_prompt_fields
from aiperf.config.types import SamplingDistribution
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType

_logger = AIPerfLogger(__name__)

__all__ = [
    "VIDEO_AUDIO_CODEC_MAP",
    "AudioConfig",
    "CacheBustConfig",
    "DatasetConfig",
    "FileDataset",
    "ImageConfig",
    "PrefixPromptConfig",
    "PromptConfig",
    "PromptSelectionConfig",
    "PublicDataset",
    "RankingsConfig",
    "SynthesisConfig",
    "SyntheticDataset",
    "VideoAudioConfig",
    "VideoConfig",
]


# Shared name field for all DatasetConfig subclasses — extracted to keep each
# class definition compact (file-size budget under tools/ergonomics_baseline).
_DatasetName = Annotated[
    str,
    Field(
        min_length=1,
        description="Dataset identifier — used in result file paths. "
        "Defaults to 'default' for the singular `dataset:` shorthand.",
    ),
]


# Dataset type variants using discriminated unions
class SyntheticDataset(BaseConfig):
    """
    Synthetic dataset configuration.

    Generates prompts programmatically based on token length
    specifications. Ideal for controlled experiments.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

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

    isl: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `prompts.isl`. Accepts a fixed integer or "
                "distribution dict. Hoisted into `prompts.isl` by the before-"
                "validator and excluded from serialization."
            ),
        ),
    ]

    osl: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `prompts.osl`. Accepts a fixed integer or "
                "distribution dict. Hoisted into `prompts.osl` by the before-"
                "validator and excluded from serialization."
            ),
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
            ge=0.0,
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

    @model_validator(mode="before")
    @classmethod
    def _hoist_isl_osl_shortcuts(cls, data: Any) -> Any:
        """Hoist top-level isl/osl into prompts.{isl,osl} for direct validation.

        AIPerfConfig.parse_datasets already runs this hoist at the list level via
        `_normalize_single_dataset_listed`. This validator covers direct
        `SyntheticDataset.model_validate({...isl...})` callers (programmatic use).
        """
        if isinstance(data, dict):
            _hoist_synthetic_prompt_fields(data)
        return data

    @model_validator(mode="after")
    def _validate_turns_at_least_one(self) -> SyntheticDataset:
        # NormalDistribution.mean keeps ge=0.0 to support OSL=0 / turn_delay=0,
        # so turns must enforce its tighter "at least 1 turn per conversation"
        # contract here. Without this, --conversation-turn-mean 0 (or YAML
        # turns: {mean: 0}) is silently floored to 1 by the composer.
        if self.turns is not None and self.turns.expected_value < 1.0:
            raise ValueError(
                "turns expected value must be >= 1 "
                f"(got {self.turns.expected_value}); set --conversation-turn-mean "
                "to at least 1 or omit it for single-turn conversations."
            )
        return self


class FileDataset(BaseConfig):
    """
    File-based dataset configuration.

    Loads prompts from a local file in various formats.
    Supports trace replay and custom sampling strategies.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Dataset type discriminator. Must be 'file'."),
    ]

    path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to file or directory containing benchmark dataset. "
            "Can be absolute or relative. Mutually exclusive with `records:`. "
            "Supported formats depend on the format field: "
            "JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace/"
            "bailian_trace, Parquet for baseten_trace, directories for random_pool.",
        ),
    ]

    records: Annotated[
        list[dict[str, Any]] | dict[str, list[dict[str, Any]]] | None,
        Field(
            default=None,
            description="Inline benchmark records, embedded directly in the YAML config. "
            "Mutually exclusive with `path:`. The element schema is determined by `format:` "
            "(same shape as one line of the equivalent JSONL file). "
            "For `format: random_pool`, may be either a flat list (single pool) or a "
            "dict-of-lists (multi-pool, mirrors the directory-of-JSONLs file mode). "
            "All other formats require a flat list.",
        ),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic and expected file structure. "
            "single_turn: JSONL with single prompt-response exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace / bailian_trace / baseten_trace / burst_gpt_trace: "
            "timestamped trace files for replay. "
            "sagemaker_data_capture: JSONL captured by SageMaker DataCapture. "
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
            "Applies to trace formats such as mooncake_trace and baseten_trace, "
            "except speedup_ratio, which is rejected for baseten_trace "
            "(use replay_speedup / --replay-speedup to scale replay pacing there).",
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

    trace_session_sample_ratio: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=1.0,
            default=None,
            description="Fraction of trace sessions to keep for replay, sampled "
            "whole-session to preserve multi-turn integrity; deterministic when "
            "``random_seed`` is set. Only supported by the baseten_trace loader.",
        ),
    ]

    inter_turn_delay_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Clamp per-turn replay delays to at most this many "
            "seconds; ``None`` disables the cap. Honored by the DAG JSONL loader "
            "and the baseten_trace loader's closed-loop think-times; the clamp "
            "count is reported at end of load.",
        ),
    ]

    max_idle_gap_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description="Collapse idle gaps between consecutive requests (across "
            "all sessions) to at most this many seconds, so a sparse or "
            "session-sampled trace does not replay dead air; ``None`` disables "
            "the cap. The cap is in replay wall-clock seconds, applied after "
            "replay_speedup compression, so it bounds actual benchmark idle "
            "time regardless of speedup. Only supported by the baseten_trace "
            "loader.",
        ),
    ]

    replay_speedup: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description="Trace replay wall-clock compression (10 = 10x faster "
            "than recorded): divides normalized timestamps and inter-turn delays; "
            "``None`` = real time. Unlike synthesis speedup_ratio, hash_ids stay "
            "untouched (KV-cache fidelity). Only supported by the baseten_trace "
            "loader.",
        ),
    ]

    open_loop_replay: Annotated[
        bool,
        Field(
            default=True,
            description="Open-loop replay (the default): each session starts at "
            "its absolute, speedup-scaled recorded timestamp; continuation turns "
            "fire at max(recorded timestamp, prior-turn completion). Set "
            "``False`` for closed-loop back-pressure: continuation turns fire a "
            "think-time (recorded start-to-start gap minus recorded e2e duration) "
            "after the prior turn completes, keeping sessions causally ordered "
            "when replayed service times differ from recorded (e.g. A/A "
            "comparisons). Only honored by the baseten_trace loader.",
        ),
    ]

    open_loop_strict: Annotated[
        bool,
        Field(
            default=False,
            description="In open-loop replay, fire every trace row at its "
            "absolute recorded timestamp as an independent single-turn session, "
            "trading away multi-turn grouping and session metrics. Only honored "
            "by the baseten_trace loader.",
        ),
    ]

    omit_kv_hints: Annotated[
        bool,
        Field(
            default=False,
            description="Drop recorded KV-cache hints (``hash_ids``, "
            "``block_size``) from replayed request bodies, for strict frontends "
            "that reject unknown parameters. Only honored by the baseten_trace "
            "loader.",
        ),
    ]

    force_min_tokens: Annotated[
        bool,
        Field(
            default=True,
            description="Pin ``min_tokens`` to the recorded output length so "
            "replayed generations match recorded lengths; disable to let EOS end "
            "generations naturally (some servers reject ``min_tokens``). Only "
            "honored by the baseten_trace loader.",
        ),
    ]

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length to apply when records do not specify one. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Per-line `output_length` values in the file always take precedence.",
        ),
    ]

    block_size: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Hash-id block granularity for trace replay (--isl-block-size). "
            "hash-id trace loaders (mooncake_trace, bailian_trace, burst_gpt_trace, "
            "sagemaker_data_capture) decode each hash_id into a cached block of this "
            "many tokens; total ISL = (num_hash_ids - 1) * block_size + final_block_size. "
            "When unset, the loader's plugin-metadata default applies (e.g. 512 for "
            "mooncake_trace, 16 for bailian_trace). Not used by weka, which carries its "
            "own inline per-block sizes.",
        ),
    ]

    trace_idle_gap_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Hard ceiling (seconds) for observed runtime idle time within "
            "each AgentX trajectory tree. The idle clock covers initial future work and "
            "restarts when the final in-flight request across the root and all descendant "
            "streams completes. If no request from that tree reaches the wire before the "
            "cap, AIPerf uniformly advances "
            "only that tree's pending replay timers. Dataset timestamps are unchanged, "
            "and request order, spawn/join dependencies, and replay barriers remain "
            "authoritative. Defaults to None (no per-trace runtime idle cap).",
        ),
    ]

    ignore_trace_delays: Annotated[
        bool,
        Field(
            default=False,
            description="Strip per-turn timestamps and inter-turn delays from trace datasets at load time. "
            "With this flag, Turn.timestamp and Turn.delay are emitted as None so concurrency / "
            "request-rate timing modes dispatch turns back-to-back instead of reproducing the recorded "
            "user think-time gaps. No effect under fixed-schedule (timestamps drive that mode before "
            "they could be ignored -- combine with --no-fixed-schedule if you want both behaviors). "
            "Mutually exclusive with use_think_time_only.",
        ),
    ]

    use_think_time_only: Annotated[
        bool,
        Field(
            default=False,
            description="For weka_trace inputs, emit Turn.delay using only the recorded per-request `think_time` "
            "(client-side delay before each request) instead of the full `t_curr - t_prev` inter-request delta. "
            "Compresses replay wall time against zero-latency mocks because the recorded `api_time` portion of "
            "each gap is dropped. Mirrors kv-cache-tester's default `--timing-strategy think-only`. Falls back to "
            "the full delta for turns whose recorded `think_time` is null. Mutually exclusive with "
            "ignore_trace_delays. No effect on non-weka trace loaders.",
        ),
    ]

    max_context_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Maximum peak prompt+output context length (tokens) per Weka root "
                "trace. Only honored by format weka_trace: the loader drops whole "
                "traces whose *recorded* peak exceeds this ceiling at load time "
                "(filter-then-cap before entries). Not a DatasetManager tokenize "
                "filter; rejected for non-Weka formats."
            ),
        ),
    ]

    cache_bust: Annotated[
        CacheBustConfig,
        Field(
            default_factory=CacheBustConfig,
            description="Per-conversation cache-bust marker injection for trace "
            "replay (mirror of the synthetic dataset's prompts.cache_bust). The "
            "engine reads the active dataset's target via "
            "``BenchmarkConfig.get_cache_bust_target()``; 'none' (default) "
            "disables it.",
        ),
    ]

    prompts: Annotated[
        PromptSelectionConfig | None,
        Field(
            default=None,
            description="Prompt synthesis selection for this dataset. "
            "Set ``corpus`` to choose sonnet vs coding when content "
            "is synthesized (trace hash_id reconstruction). Verbatim formats ignore it.",
        ),
    ]

    _use_think_time_only_explicitly_set: bool = False

    @model_validator(mode="after")
    def _validate_trace_delay_exclusivity(self) -> FileDataset:
        """Reject the mutually-exclusive trace-delay flags and snapshot intent."""
        self._use_think_time_only_explicitly_set = (
            "use_think_time_only" in self.model_fields_set
        )
        # Both flags set Turn.delay differently (None / recorded think_time), so
        # at most one may be active.
        if self.ignore_trace_delays and self.use_think_time_only:
            raise ValueError(
                "--ignore-trace-delays and --use-think-time-only are mutually "
                "exclusive (each sets Turn.delay differently)."
            )
        return self

    @model_validator(mode="after")
    def _validate_max_context_length_weka_only(self) -> FileDataset:
        """Reject max_context_length on provably non-Weka file formats.

        Only weka_trace consumes this field (recorded peak filter-then-cap).
        File Weka traces are content-auto-detected (``WekaTraceLoader.can_load``)
        and arrive with the default ``single_turn`` format, so ``single_turn`` is
        ambiguous (may resolve to Weka) and must be allowed through -- the loader
        ignores the field if it does not implement the filter. Only an explicit
        non-Weka trace/pool format is provably unsupported. The CLI converter
        (``_apply_max_context_length``) still loudly rejects an explicit non-Weka
        ``--custom-dataset-type`` before construction.
        """
        if self.max_context_length is None:
            return self
        if self.format not in (DatasetFormat.WEKA_TRACE, DatasetFormat.SINGLE_TURN):
            raise ValueError(
                "max_context_length (--max-context-length) only applies to "
                f"format weka_trace; got format {self.format}. It filters by "
                "recorded peak prompt+output length at load time."
            )
        return self

    @model_validator(mode="after")
    def _validate_source_xor(self) -> FileDataset:
        path_set = self.path is not None
        records_set = self.records is not None
        if path_set == records_set:
            raise ValueError(
                "FileDataset requires exactly one source: set either `path:` "
                "(load from disk) or `records:` (embed in YAML), not both. "
                f"Got path={self.path!r}, records={'<set>' if records_set else None}."
            )

        if records_set and isinstance(self.records, dict):
            if self.format != DatasetFormat.RANDOM_POOL:
                raise ValueError(
                    "`records:` as a dict-of-lists (multi-pool) is only valid "
                    f"for format: random_pool, got format: {self.format}."
                )
            if not self.records:
                raise ValueError("`records:` dict must contain at least one pool.")
            for pool_name, pool_items in self.records.items():
                if not pool_items:
                    raise ValueError(
                        f"`records:` pool '{pool_name}' is empty; "
                        "every pool must contain at least one record."
                    )

        if records_set and isinstance(self.records, list) and not self.records:
            raise ValueError("`records:` must contain at least one record.")

        return self

    @model_validator(mode="after")
    def _validate_open_loop_strict_requires_open_loop(self) -> FileDataset:
        # open_loop_strict is an open-loop-only modifier; the loader would
        # silently ignore it in closed-loop replay. Strict defaults False and
        # open-loop defaults True, so strict=True with open-loop=False can
        # only come from an explicitly contradictory config.
        if self.open_loop_strict and not self.open_loop_replay:
            raise ValueError(
                "--open-loop-strict requires open-loop replay; remove "
                "--no-open-loop-replay (or drop --open-loop-strict)."
            )
        return self

    @model_validator(mode="after")
    def _warn_large_inline_records(self) -> FileDataset:
        if self.records is None:
            return self
        if isinstance(self.records, dict):
            total = sum(len(p) for p in self.records.values())
        else:
            total = len(self.records)

        from aiperf.common.environment import Environment

        if total > Environment.DATASET.INLINE_RECORDS_WARN_THRESHOLD:
            _logger.warning(
                f"Inline records: dataset '{self.name}' has {total} records inline, which is "
                f"large enough to make the YAML hard to scan. Consider moving "
                f"the dataset to a JSONL file and switching to `path:` instead."
            )
        return self


class PublicDataset(BaseConfig):
    """
    Public dataset configuration.

    Uses well-known public benchmarking datasets that are
    automatically downloaded and processed by AIPerf.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

    type: Annotated[
        Literal[DatasetType.PUBLIC],
        Field(description="Dataset type discriminator. Must be 'public'."),
    ]

    dataset: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "Name of the HuggingFace public dataset enum (e.g. 'sharegpt', 'alpaca'). "
            "AIPerf automatically downloads and parses these datasets.",
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

    filters: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Dataset-specific filters forwarded to public dataset loaders. "
            "Supported keys and values depend on the selected dataset.",
        ),
    ]

    hf_weka_dataset: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace dataset repo for the generic Weka loader (e.g. "
            "`semianalysisai/cc-traces-weka-061526`). Only applies with "
            "`dataset: weka_hf`; setting it with any other public dataset is an error. "
            "Passing `--hf-weka-dataset` on the CLI auto-selects `--public-dataset weka_hf`, "
            "so the repo flag works on its own. Pinned Weka public dataset aliases keep "
            "their registry-defined repo names.",
        ),
    ]

    inter_turn_delay_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Clamp per-turn replay delays to at most this many "
            "seconds for HF-backed Weka trace replay (mirror of "
            "``FileDataset.inter_turn_delay_cap_seconds``). ``None`` disables "
            "the cap.",
        ),
    ]

    trace_idle_gap_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Hard ceiling (seconds) for observed runtime idle time within "
            "each AgentX trajectory tree for HF-backed Weka replay (mirror of "
            "``FileDataset.trace_idle_gap_cap_seconds``). Dataset timestamps remain "
            "unchanged. Defaults to None (no per-trace runtime idle cap).",
        ),
    ]

    ignore_trace_delays: Annotated[
        bool,
        Field(
            default=False,
            description="Strip per-turn timestamps and inter-turn delays from "
            "HF-backed Weka trace replay at load time (mirror of "
            "``FileDataset.ignore_trace_delays``). Mutually exclusive with "
            "use_think_time_only.",
        ),
    ]

    use_think_time_only: Annotated[
        bool,
        Field(
            default=False,
            description="For HF-backed Weka replay, emit Turn.delay using only "
            "the recorded per-request ``think_time`` instead of the full "
            "inter-request delta (mirror of ``FileDataset.use_think_time_only``). "
            "Mutually exclusive with ignore_trace_delays.",
        ),
    ]

    max_context_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Maximum peak prompt+output context length (tokens) per "
                "conversation for HF-backed Weka replay; drops over-length "
                "traces at load using *recorded* lengths (mirror of "
                "``FileDataset.max_context_length``). Rejected for non-Weka "
                "public datasets."
            ),
        ),
    ]

    cache_bust: Annotated[
        CacheBustConfig,
        Field(
            default_factory=CacheBustConfig,
            description="Per-conversation cache-bust marker injection for "
            "HF-backed Weka trace replay (mirror of "
            "``FileDataset.cache_bust``). 'none' (default) disables it.",
        ),
    ]

    prompts: Annotated[
        PromptSelectionConfig | None,
        Field(
            default=None,
            description="Prompt synthesis selection for this dataset. "
            "Set ``corpus`` to choose sonnet vs coding when content "
            "is synthesized (trace hash_id reconstruction). Verbatim formats ignore it.",
        ),
    ]

    synthesis: Annotated[
        SynthesisConfig | None,
        Field(
            default=None,
            description="Trace synthesis/transformation configuration for "
            "HF-backed Weka trace replay (mirror of ``FileDataset.synthesis``). "
            "``--max-isl``/``--max-osl`` cap the per-conversation input/output "
            "lengths of the replayed HF Weka traces; without this they applied "
            "only to file-based traces. ``None`` disables synthesis.",
        ),
    ]

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length to apply when records do not "
            "specify one (mirror of ``FileDataset.osl``). Can be a fixed integer "
            "or {mean, stddev} distribution. Per-record output lengths always "
            "take precedence.",
        ),
    ]

    entries_explicit: Annotated[
        bool,
        Field(
            default=False,
            exclude=True,
            alias="_entries_explicit",
            description=(
                "Internal provenance flag: True when the user explicitly chose the "
                "entry count, gating the num_dataset_entries key in public-dataset "
                "provenance. The CLI converter sets the ``_entries_explicit`` "
                "sentinel to the true intent whenever it writes ``entries`` (which "
                "otherwise absorbs the --num-conversations / --request-count "
                "fallback and cannot signal intent). A YAML/programmatic config "
                "that sets ``entries`` directly (no sentinel) is treated as "
                "explicit by ``_resolve_entries_explicit``. Excluded from "
                "serialization."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _resolve_entries_explicit(self) -> PublicDataset:
        """Treat a direct (sentinel-less) ``entries`` set as explicit intent.

        The CLI converter always pins ``_entries_explicit`` when it writes
        ``entries``, so ``entries_explicit in model_fields_set`` is True for any
        CLI-derived config and the sentinel value stands. A YAML/programmatic
        config sets ``entries`` with no sentinel; there, a present ``entries``
        means the author named the count on purpose (cquil's model_fields_set
        semantics), so promote it to explicit.
        """
        if (
            "entries_explicit" not in self.model_fields_set
            and "entries" in self.model_fields_set
        ):
            self.entries_explicit = True
        return self

    @model_validator(mode="after")
    def _validate_trace_delay_exclusivity(self) -> PublicDataset:
        """Reject the mutually-exclusive trace-delay flags and snapshot intent."""
        self._use_think_time_only_explicitly_set = (
            "use_think_time_only" in self.model_fields_set
        )
        # Both flags set Turn.delay differently (None / recorded think_time), so
        # at most one may be active.
        if self.ignore_trace_delays and self.use_think_time_only:
            raise ValueError(
                "--ignore-trace-delays and --use-think-time-only are mutually "
                "exclusive (each sets Turn.delay differently)."
            )
        return self

    @model_validator(mode="after")
    def _validate_max_context_length_weka_only(self) -> PublicDataset:
        """Reject max_context_length on non-Weka public datasets.

        Only Weka HF loaders consume this field (recorded peak filter-then-cap).
        Other public datasets would silently store and ignore it.
        """
        if self.max_context_length is None:
            return self
        if "weka" not in str(self.dataset).lower():
            raise ValueError(
                "max_context_length (--max-context-length) only applies to "
                f"Weka public datasets; got dataset {self.dataset}. It filters "
                "by recorded peak prompt+output length at load time."
            )
        return self

    @model_validator(mode="after")
    def _validate_weka_hf(self) -> PublicDataset:
        """Fail fast on weka_hf <-> hf_weka_dataset inconsistency.

        The CLI path is safe (the converter auto-selects weka_hf only when
        --hf-weka-dataset is set), but a config file declaring
        ``dataset: weka_hf`` with no ``hf_weka_dataset`` would otherwise reach
        the generic Weka loader's required repo argument as None and surface an
        opaque TypeError. Mirror v1's composer-level guard at config-load time.
        Pinned Weka aliases (semianalysis_*) are distinct enum values and keep
        their registry-defined repos, so they are untouched here.
        """
        is_weka_hf = self.dataset == PublicDatasetType.WEKA_HF
        if self.hf_weka_dataset is not None and not is_weka_hf:
            raise ValueError(
                "hf_weka_dataset (--hf-weka-dataset) can only be used with "
                "dataset weka_hf (--public-dataset weka_hf)"
            )
        if is_weka_hf:
            repo = (self.hf_weka_dataset or "").strip()
            if not repo:
                raise ValueError(
                    "dataset weka_hf (--public-dataset weka_hf) requires a "
                    "non-empty hf_weka_dataset (--hf-weka-dataset) HuggingFace repo"
                )
            self.hf_weka_dataset = repo
        return self


# Union type for all dataset variants using discriminated union
DatasetConfig = Annotated[
    SyntheticDataset | FileDataset | PublicDataset,
    Discriminator("type"),
]
"""
Dataset configuration supporting multiple source types.

Discriminated by 'type' field:
    - synthetic: Generated prompts (type: synthetic)
    - file: Local file data (type: file)
    - public: Public benchmark datasets (type: public)
"""
