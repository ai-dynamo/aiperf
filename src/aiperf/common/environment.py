# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Environment Configuration Module

Provides a hierarchical, type-safe configuration system using Pydantic BaseSettings.
All settings can be configured via environment variables with the AIPERF_ prefix.

Structure:
    Environment.ACCURACY.*       - Accuracy benchmark settings
    Environment.API_SERVER.*     - API server settings
    Environment.COMPRESSION.*    - Compression settings for streaming file transfers
    Environment.DATASET.*        - Dataset management
    Environment.DEV.*            - Development and debugging settings
    Environment.GPU.*            - GPU telemetry collection
    Environment.HTTP.*           - HTTP client socket and connection settings
    Environment.LOGGING.*        - Logging configuration
    Environment.METRICS.*        - Metrics collection and storage
    Environment.MLFLOW.*         - MLflow export settings
    Environment.OTEL.*           - OTel metrics streaming
    Environment.RECORD.*         - Record processing
    Environment.SEARCH_PLANNER.* - Adaptive-search planner tunables
    Environment.SERVER_METRICS.* - Server metrics collection
    Environment.SERVICE.*        - Service lifecycle and communication
    Environment.TIMING.*         - Timing manager settings
    Environment.TOKENIZER.*      - Tokenizer pre-warm and loading
    Environment.UI.*             - User interface settings
    Environment.WANDB.*          - Weights & Biases export settings
    Environment.WORKER.*         - Worker management and scaling
    Environment.ZMQ.*            - ZMQ communication settings

Examples:
    # Via environment variables:
    AIPERF_HTTP_SO_RCVBUF=20971520
    AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8

    # In code:
    print(f"Buffer: {Environment.HTTP.SO_RCVBUF}")
    print(f"Workers: {Environment.WORKER.CPU_UTILIZATION_FACTOR}")
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import IS_WINDOWS
from aiperf.config.loader.parsing import (
    parse_service_types,
    parse_str_or_csv_list,
)
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.plugin.enums import UIType

_logger = AIPerfLogger(__name__)

__all__ = ["Environment"]


class _AccuracySettings(BaseSettings):
    """Accuracy benchmark settings.

    Tunables for accuracy benchmarking: the cancel-path result-wait timeout and
    the LiveCodeBench dataset release pin, so accuracy behavior and numbers are
    reproducible across runs without requiring source edits.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_ACCURACY_")

    CANCEL_RESULT_WAIT_SEC: float = Field(
        default=5.0,
        ge=0.0,
        description="Bounded time (seconds) the SystemController waits on the "
        "cancel (Ctrl+C) path for the RecordsManager's "
        "ProcessAccuracyResultMessage before stopping. The normal completion "
        "path blocks on the accuracy shutdown gate indefinitely, but the cancel "
        "path must not hang forever, so it waits at most this long for the "
        "graded accuracy summary to arrive over pub/sub before proceeding to "
        "export. Set to 0 to skip the wait entirely.",
    )

    LCB_GRADE_TIMEOUT_MAX_S: float = Field(
        default=300.0,
        gt=0.0,
        description="Hard ceiling (seconds) on the client-side wall-clock timeout "
        "for a single LiveCodeBench code-execution grade. The per-grade timeout "
        "scales with the problem's test-case count (lighteval's internal budget "
        "plus a margin) but is capped here so one wedged grading worker cannot "
        "stall the whole run. Raise it if legitimately slow large problems are "
        "being prematurely failed; lower it to fail wedged workers faster. "
        "Consumed by ``aiperf.accuracy.graders.code_execution._derive_grade_timeout``.",
    )

    LCB_RELEASE_TAG: str = Field(
        default="v4_v5",
        description="LiveCodeBench dataset subset (HF config name) "
        "passed as the positional ``name`` arg to "
        '``load_dataset("livecodebench/code_generation_lite", name, split="test", trust_remote_code=True)``. '
        "Pins which monthly snapshot LCB serves so accuracy numbers "
        "are reproducible across runs and branches. Default "
        "``v4_v5`` matches lighteval's base subset; bump (e.g. to "
        "``v6``) when the team rebaselines against a newer snapshot. "
        "``trust_remote_code=True`` is required because LCB ships a "
        "repository loading script; this is only compatible with "
        "``datasets<4`` (``datasets>=4`` dropped loading-script support "
        "entirely — the loader surfaces a clear error with a "
        "``datasets<4`` pin). Consumed by "
        "``aiperf.accuracy.benchmarks.lcb_codegeneration``.",
    )


class _AgentXSettings(BaseSettings):
    """Settings for the InferenceX AgentX scenario family.

    Controls runtime knobs for the agentx scenario: the substring allowlist
    and rate limit used to classify and gate context-overflow errors
    (RFC 2026-04-26 §7), and the AgenticReplayStrategy double-recycle guard
    window (RECYCLE_GUARD_MAX_WINDOW).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_AGENTX_",
    )

    CONTEXT_OVERFLOW_SUBSTRINGS: list[str] = Field(
        default=[
            "context length",
            "maximum context",
            "context_length_exceeded",
            "prompt is too long",
        ],
        description="Case-insensitive substring allowlist used to classify a "
        "server error response as a context-overflow event. Matched against "
        "the raw response body and the OpenAI-style nested 'error.message' "
        "field. Extend via AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS to "
        "support additional inference-server vocabularies (vLLM, TGI, "
        "TensorRT-LLM, ...). Empty list disables runtime detection.",
    )
    CONTEXT_OVERFLOW_RATE_LIMIT: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Strict upper bound on the per-run context-overflow rate "
        "(context_overflow_count / total_responses) before a scenario "
        "submission is flipped to submission_valid=false with reason "
        "'context_overflow_rate_exceeded'. Default 0.01 (1%) matches the "
        "scenario spec RFC 2026-04-26 §7. Comparison is strictly greater-than: "
        "rate exactly equal to the limit is accepted. Has no effect on "
        "non-scenario runs (no --scenario flag) or runs with zero responses.",
    )
    RECYCLE_GUARD_MAX_WINDOW: int = Field(
        ge=1,
        default=1_000_000,
        description="Maximum number of recently-recycled root correlation_ids "
        "retained by AgenticReplayStrategy's double-recycle guard (which raises "
        "if a final-turn credit return is delivered twice and would re-spawn a "
        "session). Without a bound the guard retains one entry per recycled "
        "session for the entire PROFILING phase -- hundreds of MB of "
        "unreclaimable memory on long, high-throughput durability ramps. Oldest "
        "entries are evicted FIFO once the window is full; a duplicate delivered "
        "after this many intervening recycles is no longer caught. Duplicate "
        "deliveries are near-immediate in practice, so the default window is far "
        "larger than any real gap; raise it for very high concurrency.",
    )


class _APIServerSettings(BaseSettings):
    """API server settings.

    Controls the host and port of the API server.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_API_SERVER_")

    HOST: str = Field(
        default="127.0.0.1",
        description="Host to bind the API server to",
    )
    PORT: int | None = Field(
        ge=1,
        le=65535,
        default=None,
        description="Port to bind the API server to",
    )
    CORS_ORIGINS: list[str] = Field(
        default=[],
        description="List of CORS origins to allow (empty = no CORS, ['*'] = all origins)",
    )
    SHUTDOWN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Timeout in seconds for graceful API server shutdown before force-cancelling",
    )
    POST_COMPLETE_GRACE: float = Field(
        ge=0.0,
        le=300.0,
        default=5.0,
        description="Seconds the API listener stays open after a benchmark terminates "
        "so polling clients can observe the final status before the server shuts down. "
        "Set to 0 to skip the grace window and shut down immediately.",
    )


class _ChatSettings(BaseSettings):
    """Settings for the interactive ``aiperf chat`` command."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_CHAT_",
    )

    CONNECT_TIMEOUT: float = Field(
        gt=0.0,
        default=10.0,
        description="Seconds to wait to establish a connection to the endpoint "
        "before a turn fails. Kept short so an unreachable URL fails fast.",
    )
    READ_TIMEOUT: float = Field(
        gt=0.0,
        default=300.0,
        description="Seconds to wait for the next streamed chunk before a turn "
        "fails. No overall (total) timeout is applied, so long generations are "
        "never truncated mid-reply; this only fires if the server stalls.",
    )


class _CompressionSettings(BaseSettings):
    """Compression settings for streaming file transfers.

    Controls chunk size and compression levels for zstd and gzip encodings
    used in dataset and results file transfers.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_COMPRESSION_",
    )

    CHUNK_SIZE: int = Field(
        ge=1024,
        le=1048576,
        default=65536,
        description="Chunk size in bytes for streaming compressed data (default: 64KB)",
    )
    ZSTD_LEVEL: int = Field(
        ge=1,
        le=22,
        default=3,
        description="Zstandard compression level (1=fastest, 22=best compression, default: 3)",
    )
    GZIP_LEVEL: int = Field(
        ge=1,
        le=9,
        default=6,
        description="Gzip compression level (1=fastest, 9=best compression, default: 6)",
    )


class _CLIRunnerSettings(BaseSettings):
    """CLI runner post-run callback behavior.

    Controls whether OnComplete callback exceptions abort the run after all
    callbacks attempt or are isolated and logged. Default is isolated so that
    a single misbehaving callback (e.g. auto-plot in strict mode, third-party
    hook) cannot bypass the deliberate ``os._exit`` hang-protection that
    guards against multiprocessing/ZMQ teardown hangs in the parent process.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
    )

    RAISE_ON_CALLBACK_ERROR: bool = Field(
        default=False,
        description="When true, re-raise the first OnComplete callback exception "
        "after running all remaining callbacks but before os._exit. Provides a "
        "strict-mode contract where a callback raise propagates out "
        "of the runner. When false (default) the exception is logged with full "
        "traceback, the exit code is forced non-zero, and the process still "
        "terminates via os._exit so leftover ZMQ/multiprocessing state cannot "
        "hang the interpreter.",
    )


class _DatasetSettings(BaseSettings):
    """Dataset loading and configuration.

    Controls timeouts and behavior for dataset loading operations,
    as well as memory-mapped dataset storage settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DATASET_",
    )

    CONFIGURATION_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for dataset configuration operations",
    )
    MMAP_BASE_PATH: Path | None = Field(
        default=None,
        description="Base path for memory-mapped dataset files. If None, uses system temp directory. "
        "Set to a shared filesystem path for Kubernetes mounted volumes. "
        "Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc "
        "creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/",
    )
    MMAP_CACHE_ENABLED: bool = Field(
        default=True,
        description="If True, AIPerf reuses memory-mapped dataset files across runs whose "
        "input bytes, tokenizer identity, and prompt/input settings are byte-identical. "
        "Set to False to force every run to re-tokenize and re-write its mmap files. "
        "Cache misses still produce byte-identical mmap files to a non-cached run.",
    )
    MMAP_CACHE_DIR: Path | None = Field(
        default=None,
        description="Directory holding the content-addressed mmap cache. If None, defaults to "
        "~/.cache/aiperf/dataset_mmap. Each cache entry lives under a `dir/key` subpath and contains "
        "dataset.dat, index.dat, manifest.json, and (when produced) inputs.json. "
        "No automatic eviction is implemented yet -- delete the directory to reclaim disk.",
    )
    PREFORMAT_PAYLOADS: bool = Field(
        default=False,
        description="If True, pre-encode single-turn / self-contained synthetic "
        "conversations to the PAYLOAD_BYTES mmap fast path at dataset-build time "
        "so workers stream the bytes verbatim and skip per-request encoding. This "
        "is a throughput optimization that DROPS input-tokenization metrics "
        "(input_sequence_length, image counts) because the structured prompt is "
        "discarded. Default False keeps the structured-turns (CONVERSATION) path "
        "so those metrics are computed. Datasets that natively ship raw payloads "
        "(raw_payload / inputs_json / mooncake-with-payload) always use "
        "PAYLOAD_BYTES regardless of this flag; cache-bust runs always use "
        "CONVERSATION regardless.",
    )
    PUBLIC_DATASET_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for public dataset loading operations",
    )
    MEDIA_DOWNLOAD_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=60.0,
        description="Timeout in seconds per media URL download when inline encoding is required",
    )
    MEDIA_DOWNLOAD_MAX_CONCURRENCY: int = Field(
        ge=1,
        le=100,
        default=10,
        description="Maximum number of concurrent media URL downloads",
    )
    INLINE_RECORDS_WARN_THRESHOLD: int = Field(
        ge=1,
        default=500,
        description="Soft warning threshold for the number of inline `records:` "
        "entries on a `FileDataset`. When total inline records exceed this "
        "value, the config loader logs a warning suggesting the user move the "
        "dataset to a JSONL file. No hard cap.",
    )

    WEKA_PARALLEL_WORKERS: int = Field(
        ge=0,
        le=256,
        default=0,
        description="Number of worker processes for WekaTraceLoader parallel "
        "reconstruction. 0 = auto (min(cpu_count - 1, 16, num_traces)). Set to 1 "
        "to force serial reconstruction.",
    )
    WEKA_PARALLEL_THRESHOLD: int = Field(
        ge=1,
        le=100000,
        default=8,
        description="Minimum number of parent traces required before "
        "WekaTraceLoader switches to the multi-process parallel reconstruction "
        "path. Below this, the in-process serial path is used (Pool startup "
        "overhead exceeds the speedup for tiny corpora).",
    )
    WEKA_SPLIT_FLATTENED_AGENTS: bool = Field(
        default=True,
        description="When True (default), WekaTraceLoader runs hash_id LCP "
        "chain detection at both layers: untagged agent fan-outs recorded as "
        "flat top-level requests split into per-agent child conversations "
        "(::fa:NNN), and each subagent entry's inner requests split into "
        "per-context-chain children (`::sa:agent_id` plus :fa:NNN siblings), "
        "all with SPAWN/SPAWN_JOIN linkage so replay reproduces the recorded "
        "concurrency. Set to False to disable detection at both layers: all "
        "top-level requests serialize into one root conversation and each "
        "subagent emits exactly one child with its inner requests in time "
        "order. Detected chains at both layers are further split into genuine "
        "agents and auxiliary one-shot sidecars (top-level ::fa: vs ::aux:; "
        "subagent overflow :fa: vs :aux:) per WEKA_AUX_MAX_REQUESTS / "
        "WEKA_AUX_ISL_RATIO / WEKA_AUX_ISL_FLOOR.",
    )
    WEKA_TOOL_SHAPED_MESSAGES: bool = Field(
        default=False,
        description="When True, WekaTraceLoader emits the OpenAI tool-call "
        "wire shape for turns classified as tool-result continuations: the "
        "same-delta assistant message gains a synthetic tool_calls entry and "
        "the turn's new input is sent as a role='tool' message instead of "
        "plain user text (content unchanged). Exercises the server's "
        "tool-message chat-template path at the cost of exact ISL fidelity "
        "(tool messages tokenize differently than plain user text). Only "
        "turns with a recorded tool signal (input_types / prior stop) shape; "
        "legacy traces are unaffected. Default False keeps the byte-exact "
        "plain-user replay shape.",
    )
    WEKA_SEAM_MAX_GAP_SECONDS: float = Field(
        ge=0.0,
        default=3600.0,
        description="LCP chain-detection seam guard: the maximum wall-clock gap "
        "(seconds) between a chain's last request and a candidate continuation "
        "before that continuation is only accepted when it also keeps enough of "
        "the prior context (see WEKA_SEAM_MIN_OVERLAP_RATIO). A genuine context "
        "compaction continues promptly (seconds to minutes), so a low-overlap "
        "join hours later is treated as a distinct session that merely shares a "
        "base prefix and is spawned as its own conversation instead of being "
        "stitched onto the chain (which would fabricate a multi-hour intra-"
        "conversation idle gap). The guard fires only when BOTH this gap is "
        "exceeded AND overlap is below the ratio, so prompt compactions at any "
        "overlap and verbatim long-gap resumes at high overlap are preserved. "
        "Raise toward infinity to disable the temporal half of the guard.",
    )
    WEKA_SEAM_MIN_OVERLAP_RATIO: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="LCP chain-detection seam guard: the minimum shared-prefix "
        "ratio (continuation's fork depth / the chain tail's block count) for a "
        "far-future continuation to still be accepted as the same agent. Below "
        "this, a continuation past WEKA_SEAM_MAX_GAP_SECONDS is spawned as a new "
        "conversation rather than spliced on. Corpus data is bimodal -- real "
        "compactions and verbatim resumes keep at least 94% of the prefix, while "
        "coincidental base-prefix mis-merges keep under 50% -- so 0.5 sits in a wide "
        "safe valley. Set to 0.0 to disable the overlap half of the guard.",
    )
    WEKA_AUX_MAX_REQUESTS: int = Field(
        ge=0,
        default=1,
        description="Auxiliary (sidecar) classification: a detected worker "
        "chain with at most this many requests is eligible to be reclassified "
        "as an auxiliary one-shot call -- a tool-issued sidecar (web "
        "fetch/search summary, title generation, a classifier) rather than a "
        "sustained agent -- when it also passes the WEKA_AUX_ISL_* size test. "
        "Applies to both top-level flat chains (::fa: -> ::aux:) and a "
        "subagent's nested-LCP overflow (:fa: -> :aux:). Corpus sidecars are "
        "overwhelmingly single-request, so the default is 1. Set to 0 to "
        "disable aux classification (every worker chain keeps its agent tag). "
        "Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True.",
    )
    WEKA_AUX_ISL_RATIO: float = Field(
        ge=0.0,
        default=0.10,
        description="Auxiliary (sidecar) classification: an aux-eligible chain "
        "(see WEKA_AUX_MAX_REQUESTS) is reclassified to a sidecar only when its "
        "first request's input length is below max(WEKA_AUX_ISL_FLOOR, this "
        "ratio * the enclosing main chain's peak input length -- the trace's for "
        "flat chains, the subagent's for overflow). The ratio catches calls "
        "small relative to a large conversation's accumulated context; the floor "
        "catches them in absolute terms. Sidecars start from a fresh "
        "few-thousand-token context vs the agent's tens-to-hundreds of "
        "thousands.",
    )
    WEKA_AUX_ISL_FLOOR: int = Field(
        ge=0,
        default=16384,
        description="Auxiliary (sidecar) classification: absolute input-length "
        "floor (tokens) for the aux size test (see WEKA_AUX_ISL_RATIO). A chain "
        "whose first-request input length is below max(this, ratio * main peak "
        "ISL) is treated as an auxiliary one-shot sidecar. Keeps small "
        "fresh-context calls classified as sidecars even when the enclosing "
        "conversation is itself small.",
    )
    WEKA_AUX_CROSS_MODEL: bool = Field(
        default=True,
        description="Auxiliary (sidecar) classification: when True (default), an "
        "aux-eligible chain (at most WEKA_AUX_MAX_REQUESTS requests) whose first "
        "request runs on a different model than its enclosing main chain is "
        "treated as a sidecar regardless of input length. An agent does not "
        "switch models for its own reasoning, so a one-shot on a different model "
        "is a tool-internal call -- e.g. a Haiku WebFetch summary fired by an "
        "Opus agent, which can carry a large fetched-page payload and so escape "
        "the WEKA_AUX_ISL_* size test. Set to False to classify purely by size.",
    )
    WEKA_AUX_REDUCTION_OSL_MAX: int = Field(
        ge=0,
        default=4000,
        description="Auxiliary (sidecar) classification, reduction arm: a "
        "single-request worker chain on the SAME model as its enclosing main "
        "chain is reclassified to an auxiliary one-shot when its output length "
        "is in (0, this) tokens AND its input length is at least "
        "WEKA_AUX_ISL_FLOOR AND its input/output ratio exceeds "
        "WEKA_AUX_REDUCTION_RATIO. This catches large-input/short-output "
        "reductions (context compaction, subagent-result summaries, tool-output "
        "digests) that the size and cross-model arms miss because they are "
        "same-model and large. The bound separates a bounded summary from "
        "generative agent output (a real agent emits long completions); corpus "
        "reductions cap well below 4k output across every capture. Reductions "
        "are emitted as ordinary ::aux: sidecars. Set to 0 to disable the "
        "reduction arm. Only applies when "
        "WEKA_SPLIT_FLATTENED_AGENTS is True.",
    )
    WEKA_AUX_REDUCTION_RATIO: float = Field(
        ge=0.0,
        default=20.0,
        description="Auxiliary (sidecar) classification, reduction arm: the "
        "minimum input-to-output token ratio for a same-model single-request "
        "large-input chain to be treated as a reduction sidecar (see "
        "WEKA_AUX_REDUCTION_OSL_MAX). A reduction consumes a large body and "
        "emits a short summary, so input/output is high (corpus median ~120); "
        "20 is a conservative floor that still excludes balanced request/"
        "response calls. Only applies when WEKA_AUX_REDUCTION_OSL_MAX > 0.",
    )
    WEKA_WORKER_GROUP_MIN: int = Field(
        ge=0,
        default=3,
        description="Parallel worker-group tagging: a coordinated parallel fan-"
        "out must BOTH share a deep spawned context AND run concurrently. Workers "
        "that forked from shared context (fork depth > 0) are first scoped by "
        "their fork point (the parent request they branched off), then within "
        "each scope split into connected components of overlapping active "
        "[t0, t1) intervals; a component with at least this many members is "
        "emitted as ::wg:{group}_{member} (group = the concurrent fan-out, member "
        "= index by start time) instead of the generic ::fa: agent marker. The "
        "fork-point scope keeps unrelated fan-outs apart (pure interval overlap "
        "bridges a busy trace into one blob); the overlap split drops members "
        "that share the fork point but never run concurrently. This isolates "
        "genuine parallel sub-agent fan-out (the dominant agent population) from "
        "solo agents, unlike keying on the first context block (shared by ~every "
        "worker all session). Auxiliary chains are classified first, so a one-"
        "shot sidecar never becomes a worker-group member. Set to 0 to disable "
        "worker-group tagging (parallel workers keep the generic ::fa: tag). Only "
        "applies when WEKA_SPLIT_FLATTENED_AGENTS is True.",
    )


class _DagSettings(BaseSettings):
    """Settings for DAG benchmark mode (`dag_jsonl` input type)."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DAG_",
    )

    FAIL_FAST: bool = Field(
        default=False,
        description="When True, a single DAG child error aborts the parent and "
        "every orphan sibling under the same branch (releases sticky refcounts "
        "and calls issuer.abort_session); unrelated root sessions continue. "
        "Default False - the orchestrator counts the error in "
        "BranchStats.children_errored, releases the join slot, drains pending "
        "siblings, and continues. Set via AIPERF_DAG_FAIL_FAST=1 for strict CI "
        "assertions.",
    )


class _DeveloperSettings(BaseSettings):
    """Development and debugging configuration.

    Controls developer-focused features like debug logging, profiling, and internal metrics.
    These settings are typically disabled in production environments.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DEV_",
    )

    DEBUG_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable DEBUG logging for (comma-separated or multiple flags)",
    )
    ENABLE_YAPPI: bool = Field(
        default=False,
        description="Enable yappi profiling (Yet Another Python Profiler) for performance analysis. "
        "Requires 'pip install yappi snakeviz'",
    )
    MODE: bool = Field(
        default=False,
        description="Enable AIPerf Developer mode for internal metrics and debugging",
    )
    SHOW_EXPERIMENTAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show experimental metrics in output (requires DEV_MODE)",
    )
    SHOW_INTERNAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show internal and hidden metrics in output (requires DEV_MODE)",
    )
    TRACE_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable TRACE logging for (comma-separated or multiple flags)",
    )


class _GPUSettings(BaseSettings):
    """GPU telemetry collection configuration.

    Controls GPU metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from DCGM endpoints at the specified interval.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_GPU_",
        env_parse_enums=True,
    )

    COLLECTION_INTERVAL: float = Field(
        ge=0.01,
        le=300.0,
        default=0.333,
        description="GPU telemetry metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    DEFAULT_DCGM_ENDPOINTS: Annotated[
        str | list[str],
        BeforeValidator(parse_str_or_csv_list),
    ] = Field(
        default=["http://localhost:9400/metrics", "http://localhost:9401/metrics"],
        description="Default DCGM endpoint URLs to check for GPU telemetry (comma-separated string or JSON array)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for telemetry record export results processor",
    )
    FINAL_SCRAPE_GRACE_NS: int = Field(
        ge=0,
        le=60_000_000_000,
        default=666_000_000,
        description=(
            "Grace window in nanoseconds appended to phase end_ns when computing "
            "the GPU energy-counter delta. Energy is scraped on a cadence "
            "(see COLLECTION_INTERVAL), so the trailing scrape often lands after "
            "the phase ends; this grace lets it be included while bounding the "
            "window so cooldown/idle samples and subsequent-phase samples don't "
            "leak into the delta. Default 666_000_000 ns ~= 2x the default "
            "333 ms COLLECTION_INTERVAL; raise this if you also raise "
            "COLLECTION_INTERVAL."
        ),
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking GPU telemetry endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down GPU telemetry service to allow command response transmission",
    )


class _HTTPSettings(BaseSettings):
    """HTTP client socket and connection configuration.

    Controls low-level socket options, keepalive settings, DNS caching, and connection
    pooling for HTTP clients. These settings optimize performance for high-throughput
    streaming workloads.

    Video Generation Polling:
        For async video generation APIs that use job polling (e.g., SGLang /v1/videos),
        the poll interval is controlled by AIPERF_HTTP_VIDEO_POLL_INTERVAL. The max poll time uses
        the --request-timeout-seconds CLI argument.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_HTTP_",
    )

    CONNECTION_LIMIT: int = Field(
        ge=1,
        le=65000,
        default=2500,
        description="Maximum number of concurrent HTTP connections",
    )
    KEEPALIVE_TIMEOUT: int = Field(
        ge=0,
        le=10000,
        default=300,
        description="HTTP connection keepalive timeout in seconds for connection pooling",
    )
    SO_RCVBUF: int = Field(
        ge=1024,
        default=10485760,  # 10MB
        description="Socket receive buffer size in bytes (default: 10MB for high-throughput streaming)",
    )
    SO_SNDBUF: int = Field(
        ge=1024,
        default=10485760,  # 10MB
        description="Socket send buffer size in bytes (default: 10MB for high-throughput streaming)",
    )
    TCP_KEEPCNT: int = Field(
        ge=1,
        le=100,
        default=1,
        description="Maximum number of keepalive probes to send before considering the connection dead",
    )
    TCP_KEEPIDLE: int = Field(
        ge=1,
        le=100000,
        default=60,
        description="Time in seconds before starting TCP keepalive probes on idle connections",
    )
    TCP_KEEPINTVL: int = Field(
        ge=1,
        le=100000,
        default=30,
        description="Interval in seconds between TCP keepalive probes",
    )
    TCP_USER_TIMEOUT: int = Field(
        ge=1,
        le=1000000,
        default=30000,
        description="TCP user timeout in milliseconds (Linux-specific, detects dead connections)",
    )
    TTL_DNS_CACHE: int = Field(
        ge=0,
        le=1000000,
        default=300,
        description="DNS cache TTL in seconds for aiohttp client sessions",
    )
    FORCE_CLOSE: bool = Field(
        default=False,
        description="Force close connections after each request",
    )
    ENABLE_CLEANUP_CLOSED: bool = Field(
        default=False,
        description="Enable cleanup of closed ssl connections",
    )
    USE_DNS_CACHE: bool = Field(
        default=True,
        description="Enable DNS cache",
    )
    SSL_VERIFY: bool = Field(
        default=True,
        description="Enable SSL certificate verification. Set to False to disable verification. "
        "WARNING: Disabling this is insecure and should only be used for testing in a trusted environment.",
    )
    REQUEST_CANCELLATION_SEND_TIMEOUT: float = Field(
        ge=10.0,
        le=3600.0,
        default=300.0,
        description="Safety net timeout in seconds for waiting for HTTP request to be fully sent "
        "when request cancellation is enabled. Used as fallback when no explicit timeout is configured "
        "to prevent hanging indefinitely while waiting for the request to be written to the socket.",
    )
    IP_VERSION: Literal["4", "6", "auto"] = Field(
        default="4",
        description="IP version for HTTP socket connections. "
        "Options: '4' (AF_INET, default), '6' (AF_INET6), or 'auto' (AF_UNSPEC, system chooses).",
    )
    TRUST_ENV: bool = Field(
        default=False,
        description="Trust environment variables for HTTP client configuration. "
        "When enabled, aiohttp will read proxy settings from HTTP_PROXY, HTTPS_PROXY, "
        "and NO_PROXY environment variables.",
    )
    X_SESSION_ID_FROM_CORRELATION_ID: bool = Field(
        default=False,
        description="Also send X-Session-ID with the stable X-Correlation-ID value. "
        "This transport setting is the supported way to enable generic HTTP session "
        "affinity. It is ADDITIVE (both headers are sent); --session-header only "
        "RENAMES the single correlation header.",
    )
    X_SMG_ROUTING_KEY_FROM_CORRELATION_ID: bool = Field(
        default=False,
        description="Also send X-SMG-Routing-Key with the stable X-Correlation-ID value. "
        "This transport setting is the supported affinity path for the SGLang Model "
        "Gateway manual routing policy.",
    )
    X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID: bool = Field(
        default=False,
        description="Also send X-Dynamo-Session-ID with the stable X-Correlation-ID value, "
        "plus X-Dynamo-Parent-Session-ID on subagent children. This transport setting "
        "is the supported affinity path for a Dynamo frontend running "
        "--router-session-affinity-ttl-secs, pinning every turn of a session to the "
        "replica holding its KV prefix.",
    )
    VIDEO_POLL_INTERVAL: float = Field(
        ge=0.001,
        le=10.0,
        default=0.1,
        description="Interval in seconds between status polls for async video generation jobs. "
        "Lower values provide faster completion detection but increase server load. "
        "Applies to the aiohttp transport.",
    )


class _LoggingSettings(BaseSettings):
    """Logging system configuration.

    Controls multiprocessing log queue size and other logging behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_LOGGING_",
    )

    QUEUE_MAXSIZE: int = Field(
        ge=1,
        le=1000000,
        default=1000,
        description="Maximum size of the multiprocessing logging queue",
    )


class _MetricsSettings(BaseSettings):
    """Metrics collection and storage configuration.

    Controls metrics storage allocation and collection behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_METRICS_",
    )

    ARRAY_INITIAL_CAPACITY: int = Field(
        ge=100,
        le=1000000,
        default=10000,
        description="Initial array capacity for metric storage dictionaries to minimize reallocation",
    )
    EXPORT_FLUSH_INTERVAL: float = Field(
        ge=0.05,
        le=60.0,
        default=1.0,
        description="Periodic flush interval (seconds) for buffered JSONL stream exporters (raw record writer, record export, gpu/server-metrics JSONL writers). Bounds the worst-case freshness of low-throughput export files when the in-memory batch never reaches batch_size.",
    )
    USAGE_PCT_DIFF_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=10.0,
        description="Percentage difference threshold for flagging discrepancies between API usage and client token counts (default: 10%)",
    )
    OSL_MISMATCH_PCT_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=5.0,
        description="Percentage difference threshold for flagging discrepancies between requested and actual output sequence length (default: 5%)",
    )
    OSL_MISMATCH_MAX_TOKEN_THRESHOLD: int = Field(
        ge=1,
        default=50,
        description="Maximum absolute token threshold for OSL mismatch. The effective threshold is min(requested_osl * pct_threshold, this value). Makes threshold tighter for large OSL values (default: 50 tokens)",
    )
    TDIGEST_COMPRESSION: int = Field(
        ge=20,
        le=10000,
        default=500,
        description="t-digest sketch compression for list-valued record metric aggregation. Higher = more centroids, tighter percentile accuracy, larger sketch. Default 500 measured to keep worst-case relative percentile error under 0.05% on 50M-sample workloads (40x under the 0.5% claimed accuracy band) at ~4 KB sketch size.",
    )
    LIST_BACKEND: Literal["ragged", "tdigest"] = Field(
        default="ragged",
        description="Storage backend for list-valued RECORD metrics (today: only inter_chunk_latency). 'ragged' (default) keeps every value, enabling exact percentiles and ICL-aware throughput / tokens-in-flight sweep curves. 'tdigest' uses a bounded-memory crick.TDigest sketch (~4 KB regardless of sample count) — percentiles are approximate (at most 0.05% relative error at default compression), and ICL-aware sweep curves silently fall back to their non-ICL equivalents that use only request-level (start_ns, generation_start_ns, end_ns) timing. Choose tdigest when records-manager pod memory at 1M+ request scale is the binding constraint.",
    )


class _OTelSettings(BaseSettings):
    """OpenTelemetry metrics streaming configuration.

    Controls buffering and flush behavior for OTLP metric streaming.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_OTEL_",
    )

    FLUSH_INTERVAL_SECONDS: float = Field(
        ge=0.1,
        le=60.0,
        default=2.0,
        description="Interval in seconds between periodic OTel metrics flushes",
    )
    MAX_BATCH_RECORDS: int = Field(
        ge=1,
        le=1000000,
        default=500,
        description="Maximum number of metric records to include in a single OTel flush",
    )
    MAX_BUFFERED_RECORDS: int = Field(
        ge=1,
        le=10000000,
        default=10000,
        description="Maximum number of buffered metric records before oldest records are dropped",
    )
    REQUEST_TIMEOUT_SECONDS: float = Field(
        ge=0.1,
        le=300.0,
        default=10.0,
        description="Timeout in seconds for OTel collector HTTP requests",
    )


class _MLflowSettings(BaseSettings):
    """MLflow export configuration.

    Controls timeout behavior for post-run MLflow artifact uploads.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_MLFLOW_",
    )

    EXPORT_TIMEOUT_SECONDS: float = Field(
        ge=1.0,
        le=600.0,
        default=30.0,
        description="Timeout in seconds for the post-run MLflow export operation. "
        "If the MLflow tracking server is unreachable, the export will be abandoned "
        "after this duration rather than blocking indefinitely.",
    )


class _RecordSettings(BaseSettings):
    """Record processing and export configuration.

    Controls batch sizes, processor scaling, and progress reporting for record processing.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_RECORD_",
    )

    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for record export results processor",
    )
    RAW_EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=10,
        description="Batch size for raw record writer processor",
    )
    PROCESSOR_SCALE_FACTOR: int = Field(
        ge=1,
        le=100,
        default=4,
        description="Scale factor for number of record processors to spawn based on worker count. "
        "Formula: 1 record processor for every X workers",
    )
    PROGRESS_REPORT_INTERVAL: float = Field(
        ge=0.1,
        le=600.0,
        default=2.0,
        description="Interval in seconds between records progress report messages",
    )
    PROCESS_RECORDS_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for processing record results",
    )
    STRIP_PAYLOAD_BYTES: bool | None = Field(
        default=None,
        description="Tri-state control for omitting canonical request payload "
        "bytes from RecordContext after a request is sent, which substantially "
        "reduces record-pipeline memory for very large prompts. None (default) "
        "auto-detects: bytes are stripped only when no downstream record consumer "
        "needs them (client-side input tokenization disabled, no synthetic image/"
        "audio/video inputs, and raw payload export off). True forces stripping "
        "even when a consumer wants the bytes, disabling client-side input "
        "tokenization, media counting from request bodies, and raw request "
        "payload export. False always retains them. Auto-detection does not see "
        "media embedded in custom dataset payloads under server-token-count mode; "
        "set False explicitly for that case.",
    )


class _SearchPlannerSettings(BaseSettings):
    """Adaptive-search planner tunables.

    Controls precision targets, warmup-phase injection, and request-count
    presets for the smooth-isotonic and monotonic SLA-saturation search
    planners. All values are read at planner-construction or
    iteration-mutate time, so changes take effect on the next search run.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SEARCH_PLANNER_",
    )

    SLA_PRECISION_DEFAULT: float = Field(
        gt=0.0,
        lt=1.0,
        default=0.05,
        description="Default SLA boundary search precision target. "
        "The bisection / smooth-isotonic bracket halts when "
        "(infeasible_min - feasible_max) / infeasible_min stays below this value, "
        "and the cliff detector requires bracket_gap > this * x_hi to "
        "report a cliff. 5% mirrors perf_analyzer's --binary-search default.",
    )
    DEFAULT_WARMUP_SECONDS: float = Field(
        gt=0.0,
        le=100000.0,
        default=30.0,
        description="Smooth-isotonic SLA planner: default warmup phase duration "
        "in seconds injected into each iteration's cfg when "
        "``cfg.sla_warmup_seconds`` is unset. Spec calls for "
        "max(30s, 3*inter-batch-time) but inter-batch-time is unknown at "
        "planner-time, so 30s is the safe floor. Must be strictly positive: "
        "zero defeats the cold-KV-cache rationale that motivates the floor.",
    )
    FIRST_PROBE_WARMUP_FLOOR: float = Field(
        gt=0.0,
        le=100000.0,
        default=60.0,
        description="Smooth-isotonic SLA planner: minimum warmup duration in "
        "seconds for the first probe at each swept-dim value. Cold KV-cache "
        "and CUDA-graph compilation cost is largest the first time we hit a "
        "given concurrency, so floor that probe at 60s. Must be strictly "
        "positive: zero defeats the cold-KV-cache rationale.",
    )
    REPLICATE_WARMUP_FLOOR: float = Field(
        gt=0.0,
        le=100000.0,
        default=15.0,
        description="Smooth-isotonic SLA planner: minimum warmup duration in "
        "seconds for replicate probes at an already-probed swept-dim value. "
        "Replicates reuse the warm KV-cache / CUDA-graph state, so a shorter "
        "warmup suffices. Must be strictly positive: zero defeats the floor.",
    )
    SLA_PRECISION_REQUESTS: dict[str, Annotated[int, Field(gt=0)]] = Field(
        default={
            "tight": 10000,
            "normal": 1000,
            "coarse": 300,
        },
        description="Mapping from ``cfg.sla_precision`` preset name to the "
        "``phases.profiling.requests`` value injected when the user did not "
        "set ``requests`` explicitly on the profiling phase. Drives p99 CI "
        "width. Each value must be strictly positive — zero/negative request "
        "counts surface as iteration-time failures otherwise. Override via "
        "JSON, e.g. "
        "``AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS='{\"tight\": 20000}'``.",
    )


class _ServerMetricsSettings(BaseSettings):
    """Server metrics collection configuration.

    Controls server metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from Prometheus-compatible endpoints at the specified interval.
    Use `--no-server-metrics` CLI flag to disable collection.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SERVER_METRICS_",
        env_parse_enums=True,
    )

    COLLECTION_FLUSH_PERIOD: float = Field(
        ge=0.0,
        le=30.0,
        default=2.0,
        description="Time in seconds to continue collecting metrics after profiling completes, "
        "allowing server-side metrics to flush/finalize before shutting down (default: 2.0s)",
    )
    COLLECTION_INTERVAL: float = Field(
        ge=0.001,
        le=300.0,
        default=0.333,
        description="Server metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for server metrics jsonl writer export results processor",
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking server metrics endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down server metrics service to allow command response transmission",
    )


class _NetworkLatencySettings(BaseSettings):
    """Network latency calibration configuration.

    Controls the TCP-handshake RTT probes used to estimate the client-to-endpoint
    network round-trip time so it can be subtracted from latency metrics. Probes run
    throughout the profiling phase. Enable with `--network-latency-automatic`.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_NETWORK_LATENCY_",
        env_parse_enums=True,
    )

    DEFAULT_PROBE_INTERVAL: float = Field(
        ge=0.001,
        le=300.0,
        default=1.0,
        description="Default seconds between RTT probes when --network-latency-ping-interval is unset (default: 1.0s, ~1Hz)",
    )
    MIN_SAMPLES: int = Field(
        ge=1,
        le=100000,
        default=5,
        description="Minimum number of successful RTT samples to collect; extra probes are issued at "
        "profile completion if a short run did not reach this floor",
    )
    CONNECT_TIMEOUT: float = Field(
        ge=0.001,
        le=300.0,
        default=5.0,
        description="Timeout in seconds for a single TCP-handshake RTT probe",
    )
    COMPLETE_TOPUP_TIMEOUT: float = Field(
        ge=0.0,
        le=30.0,
        default=3.0,
        description="Wall-clock budget in seconds for the final MIN_SAMPLES top-up probes "
        "at PROFILE_COMPLETE, kept well under the command-response budget so a slow "
        "endpoint cannot stall completion",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for the network latency jsonl writer export results processor",
    )


class _TimingSettings(BaseSettings):
    """Timing manager configuration.

    Controls timing-related settings for credit phase execution and scheduling.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_TIMING_",
    )

    CANCEL_DRAIN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=10.0,
        description="Timeout in seconds for waiting for cancelled credits to drain after phase timeout",
    )
    RATE_RAMP_UPDATE_INTERVAL: float = Field(
        ge=0.01,
        le=10.0,
        default=0.1,
        description="Update interval in seconds for continuous rate ramping (default 0.1s = 100ms)",
    )


class _ServiceSettings(BaseSettings):
    """Service lifecycle and inter-service communication configuration.

    Controls timeouts for service registration, startup, shutdown, command handling,
    connection probing, heartbeats, and profile operations.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SERVICE_",
    )

    COMMAND_RESPONSE_TIMEOUT: float = Field(
        ge=1.0,
        le=1000.0,
        default=30.0,
        description="Timeout in seconds for command responses",
    )
    COMMS_REQUEST_TIMEOUT: float = Field(
        ge=1.0,
        le=1000.0,
        default=90.0,
        description="Timeout in seconds for requests from req_clients to rep_clients",
    )
    CONNECTION_PROBE_INTERVAL: float = Field(
        ge=0.1,
        le=600.0,
        default=0.1,
        description="Interval in seconds for connection probes while waiting for initial connection to the zmq message bus",
    )
    CONNECTION_PROBE_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=90.0,
        description="Maximum time in seconds to wait for connection probe response while waiting for initial connection to the zmq message bus",
    )
    CREDIT_PROGRESS_REPORT_INTERVAL: float = Field(
        ge=1,
        le=100000.0,
        default=2.0,
        description="Interval in seconds between credit progress report messages",
    )
    WARMUP_PROGRESS_LOG_INTERVAL: float = Field(
        ge=0.0,
        le=100000.0,
        default=30.0,
        description="Interval in seconds between warmup progress heartbeat log messages. "
        "Set to 0 to disable.",
    )
    DISABLE_UVLOOP: bool = Field(
        default=False,
        description="Disable uvloop and use default asyncio event loop instead",
    )
    HEARTBEAT_INTERVAL: float = Field(
        ge=1.0,
        le=100000.0,
        default=5.0,
        description="Interval in seconds between heartbeat messages for component services",
    )
    PROFILE_CONFIGURE_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=600.0,
        description="Timeout in seconds for profile configure command",
    )
    PROFILE_START_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=60.0,
        description="Timeout in seconds for profile start command",
    )
    PROFILE_CANCEL_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=10.0,
        description="Timeout in seconds for profile cancel command",
    )
    REGISTRATION_INTERVAL: float = Field(
        ge=1.0,
        le=100000.0,
        default=1.0,
        description="Interval in seconds between registration attempts for component services",
    )
    REGISTRATION_MAX_ATTEMPTS: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Maximum number of registration attempts before giving up",
    )
    REGISTRATION_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=30.0,
        description="Timeout in seconds for service registration",
    )
    START_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=30.0,
        description="Timeout in seconds for service start operations. Also bounds the per-phase wait for the first worker to register with the credit router before credit issuance begins; exceeding it fails the phase.",
    )
    TASK_CANCEL_TIMEOUT_SHORT: float = Field(
        ge=1.0,
        le=100000.0,
        default=2.0,
        description="Maximum time in seconds to wait for simple tasks to complete when cancelling",
    )
    # Event loop health monitoring settings
    EVENT_LOOP_HEALTH_ENABLED: bool = Field(
        default=True,
        description="Enable event loop health monitoring to detect blocked event loops. "
        "When enabled, TimingManager and Worker services periodically check if the event loop is responsive "
        "and log warnings when latency exceeds the threshold.",
    )
    EVENT_LOOP_HEALTH_INTERVAL: float = Field(
        ge=0.05,
        le=10.0,
        default=0.25,
        description="Interval in seconds between event loop health checks (default: 250ms). "
        "The monitor sleeps for this duration and measures actual elapsed time to detect blocking.",
    )
    EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS: float = Field(
        gt=1.0,
        le=10000.0,
        default=25.0,
        description="Warning threshold in milliseconds for event loop latency (default: 25ms). "
        "If the actual sleep duration exceeds the expected duration by this amount, a warning is logged.",
    )
    # Health server settings for Kubernetes probes
    HEALTH_ENABLED: bool = Field(
        default=False,
        description="Enable the lightweight health server for Kubernetes liveness/readiness probes. "
        "When enabled, non-API services will start an HTTP server serving /healthz and /readyz endpoints.",
    )
    HEALTH_HOST: str = Field(
        default="127.0.0.1",
        description="Host to bind the health server to. Use '0.0.0.0' for Kubernetes deployments.",
    )
    HEALTH_PORT: int = Field(
        ge=1,
        le=65535,
        default=8080,
        description="Port for the health server HTTP endpoints (/healthz, /readyz).",
    )
    HEALTH_REQUEST_TIMEOUT: float = Field(
        ge=0.1,
        le=60.0,
        default=5.0,
        description="Timeout in seconds for reading health check HTTP requests.",
    )
    # Windows-only: TCP-loopback fallback for ZMQ IPC sockets (pyzmq on
    # Windows does not support ipc://). Per-endpoint ports are derived
    # deterministically from the IPC path; these settings move or resize
    # the port window if the default conflicts with another service or
    # if a collision fires (see `_validate_no_port_collisions`).
    WINDOWS_TCP_BASE_PORT: int = Field(
        ge=1024,
        le=65535,
        default=28000,
        description="Windows-only: starting port for the ZMQ IPC TCP-loopback "
        "fallback range. Per-endpoint ports are derived as "
        "``base + (sha256_hash mod range)``. No-op on POSIX where ipc:// "
        "is used directly.",
    )
    WINDOWS_TCP_PORT_RANGE: int = Field(
        ge=64,
        le=60000,
        default=20000,
        description="Windows-only: size of the TCP-loopback port window for "
        "the ZMQ IPC fallback. Birthday-paradox collision probability for "
        "n sockets is ``1 - exp(-n*n/(2*range))``. Widen if AIPerf grows "
        "to many more sockets per run, or relocate via "
        "``AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT`` if 28000-48000 conflicts.",
    )

    @model_validator(mode="after")
    def auto_disable_uvloop_on_windows(self) -> Self:
        """Automatically disable uvloop on Windows as it's not supported.

        Validator fires on every ``_ServiceSettings()`` construction, which
        runs once in the main process AND once per spawned child service.
        Gate the log line to the main process so the user sees it once, not
        ~9 times per aiperf run on Windows.
        """
        if IS_WINDOWS and not self.DISABLE_UVLOOP:
            import multiprocessing

            if multiprocessing.parent_process() is None:
                _logger.info(
                    "Windows detected: automatically disabling uvloop (not supported on Windows)"
                )
            self.DISABLE_UVLOOP = True
        return self

    @model_validator(mode="after")
    def validate_windows_port_window_fits_in_tcp_range(self) -> Self:
        """Fail fast if the configured Windows TCP-fallback port window would
        emit invalid ports above the TCP max (65535).

        ``build_socket_address`` derives an endpoint port as
        ``WINDOWS_TCP_BASE_PORT + (sha256_hash mod WINDOWS_TCP_PORT_RANGE)``.
        With the two fields bounded independently, a config like base 65000
        + range 20000 would silently produce ``tcp://127.0.0.1:84999`` —
        invalid, and only blowing up later at ZMQ bind/connect time with a
        misleading error. Validate the sum at config time so misconfigured
        envs fail with a clear, actionable message.

        No-op on POSIX where ipc:// is used and these fields are dead. The
        check fires anyway because ``_ServiceSettings`` validation runs on
        every platform — but the only cost is the addition, which is cheap.
        """
        max_port = self.WINDOWS_TCP_BASE_PORT + self.WINDOWS_TCP_PORT_RANGE - 1
        if max_port > 65535:
            raise ValueError(
                f"AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT "
                f"({self.WINDOWS_TCP_BASE_PORT}) + "
                f"AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE "
                f"({self.WINDOWS_TCP_PORT_RANGE}) - 1 = {max_port} exceeds "
                f"the max TCP port (65535). Lower either value so the "
                f"window stays within 1024-65535."
            )
        return self


class _TokenizerSettings(BaseSettings):
    """Tokenizer pre-warm and loading configuration.

    Controls how the CLI parent pre-warms tokenizer caches before
    spawning AIPerf services. Pre-warming runs in subprocesses so the
    parent never imports the heavy native libraries (``transformers``,
    Rust-backed ``tokenizers``, ``tiktoken``).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_TOKENIZER_",
    )

    PRELOAD_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=120.0,
        description=(
            "Timeout in seconds for the parent's tokenizer pre-warm phase. "
            "Bounds the total wall-clock time for all parallel subprocess "
            "pre-warms. On timeout, subprocesses are killed and AIPerf "
            "continues; child services may then download tokenizers "
            "themselves on first use."
        ),
    )
    SKIP_PRELOAD: bool = Field(
        default=False,
        description=(
            "Skip parent-process tokenizer cache pre-warming. Intended for "
            "test harnesses that replace tokenizer loading and must avoid "
            "forked prefetch subprocesses. Production defaults to preloading."
        ),
    )


class _WandbSettings(BaseSettings):
    """Weights & Biases export configuration.

    Controls timeout behavior for the post-run W&B upload.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_WANDB_",
    )

    EXPORT_TIMEOUT_SECONDS: float = Field(
        ge=1.0,
        le=600.0,
        default=30.0,
        description="Timeout in seconds for the post-run Weights & Biases export operation. "
        "If the W&B backend is unreachable, the export will be abandoned "
        "after this duration rather than blocking indefinitely.",
    )


class _UISettings(BaseSettings):
    """User interface and dashboard configuration.

    Controls refresh rates, update thresholds, and notification behavior for the
    various UI modes (dashboard, tqdm, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_UI_",
    )

    CONSOLE_EXPORT_WIDTH: int = Field(
        ge=40,
        le=10000,
        default=140,
        description=(
            "Fixed column width used to render the post-run console exporter "
            "tables. Applied both to the recording console that produces "
            "profile_export_console.txt and to the live console when stdout "
            "is not a tty (so non-tty CI logs match the saved artifact)."
        ),
    )
    LOG_REFRESH_INTERVAL: float = Field(
        ge=0.01,
        le=100000.0,
        default=0.1,
        description="Log viewer refresh interval in seconds (default: 10 FPS)",
    )
    MIN_UPDATE_PERCENT: float = Field(
        ge=0.01,
        le=100.0,
        default=1.0,
        description="Minimum percentage difference from last update to trigger a UI update (for non-dashboard UIs)",
    )
    REALTIME_METRICS_INTERVAL: float | None = Field(
        ge=0.0,
        le=1000.0,
        default=None,
        description=(
            "Interval in seconds between real-time metrics messages (and the "
            "per-tick stats log block). 0 disables the log block; dashboards "
            "still poll. When None, `realtime_metrics_interval(ui_type)` "
            "auto-defaults to 5.0 under --ui dashboard, 30.0 otherwise."
        ),
    )
    REALTIME_METRICS_ENABLED: bool = Field(
        default=False,
        description="Enable real-time metrics collection and reporting despite UI type",
    )

    def realtime_metrics_interval(self, ui_type: "UIType") -> float:
        """Resolve the realtime metrics tick interval, applying the auto-default by UI type."""
        if self.REALTIME_METRICS_INTERVAL is not None:
            return self.REALTIME_METRICS_INTERVAL
        from aiperf.plugin.enums import UIType as _UIType  # local import: avoid cycle

        return 5.0 if ui_type == _UIType.DASHBOARD else 30.0

    SPINNER_REFRESH_RATE: float = Field(
        ge=0.1,
        le=100.0,
        default=0.1,
        description="Progress spinner refresh rate in seconds (default: 10 FPS)",
    )


class _WorkerSettings(BaseSettings):
    """Worker management and auto-scaling configuration.

    Controls worker pool sizing, health monitoring, load detection, and recovery behavior.
    The CPU_UTILIZATION_FACTOR is used in the auto-scaling formula:
    max_workers = max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP))
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_WORKER_",
    )

    CHECK_INTERVAL: float = Field(
        ge=0.1,
        le=100000.0,
        default=1.0,
        description="Interval in seconds between worker status checks by WorkerManager",
    )
    CPU_UTILIZATION_FACTOR: float = Field(
        ge=0.1,
        le=1.0,
        default=0.75,
        description="Factor multiplied by CPU count to determine default max workers (0.0-1.0). "
        "Formula: max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP))",
    )
    ERROR_RECOVERY_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=3.0,
        description="Time in seconds from last error before worker is considered healthy again",
    )
    HEALTH_CHECK_INTERVAL: float = Field(
        ge=0.1,
        le=1000.0,
        default=2.0,
        description="Interval in seconds between worker health check messages",
    )
    HIGH_LOAD_CPU_USAGE: float = Field(
        ge=50.0,
        le=100.0,
        default=85.0,
        description="CPU usage percentage threshold for considering a worker under high load",
    )
    HIGH_LOAD_RECOVERY_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=5.0,
        description="Time in seconds from last high load before worker is considered recovered",
    )
    MAX_WORKERS_CAP: int = Field(
        ge=1,
        le=10000,
        default=32,
        description="Absolute maximum number of workers to spawn, regardless of CPU count",
    )
    STALE_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=10.0,
        description="Time in seconds from last status report before worker is considered stale",
    )
    STATUS_SUMMARY_INTERVAL: float = Field(
        ge=0.1,
        le=1000.0,
        default=0.5,
        description="Interval in seconds between worker status summary messages",
    )


class _ZMQSettings(BaseSettings):
    """ZMQ socket and communication configuration.

    Controls ZMQ socket timeouts, keepalive settings, retry behavior, and concurrency limits.
    These settings affect reliability and performance of the internal message bus.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_ZMQ_",
    )

    PULL_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ PULL clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    REPLY_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received requests from ZMQ ROUTER reply clients. "
        "Prevents event loop starvation during request bursts. "
        "0 disables yielding, 1 yields after every request, 10 yields every 10 requests, etc.",
    )
    REQUEST_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received responses from ZMQ DEALER request clients. "
        "Prevents event loop starvation during response bursts. "
        "0 disables yielding, 1 yields after every response, 10 yields every 10 responses, etc.",
    )
    STREAMING_DEALER_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ streaming DEALER clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    STREAMING_ROUTER_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ streaming ROUTER clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    SUB_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ SUB clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    PULL_MAX_CONCURRENCY: int = Field(
        ge=1,
        le=10000000,
        default=100_000,
        description="Maximum concurrency for ZMQ PULL clients",
    )
    PUSH_MAX_RETRIES: int = Field(
        ge=1,
        le=100,
        default=2,
        description="Maximum number of retry attempts when pushing messages to ZMQ PUSH socket",
    )
    PUSH_RETRY_DELAY: float = Field(
        ge=0.1,
        le=1000.0,
        default=0.1,
        description="Delay in seconds between retry attempts for ZMQ PUSH operations",
    )
    RCVTIMEO: int = Field(
        ge=1,
        le=10000000,
        default=300000,  # 5 minutes
        description="Socket receive timeout in milliseconds (default: 5 minutes)",
    )
    SNDTIMEO: int = Field(
        ge=1,
        le=10000000,
        default=300000,  # 5 minutes
        description="Socket send timeout in milliseconds (default: 5 minutes)",
    )
    TCP_KEEPALIVE_IDLE: int = Field(
        ge=1,
        le=100000,
        default=60,
        description="Time in seconds before starting TCP keepalive probes on idle ZMQ connections",
    )
    TCP_KEEPALIVE_INTVL: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Interval in seconds between TCP keepalive probes for ZMQ connections",
    )
    EVENT_BUS_PROXY_FRONTEND_PORT: int = Field(
        ge=1,
        le=65535,
        default=5663,
        description="Default TCP port for the event-bus XPUB/XSUB proxy frontend "
        "(producers connect here). Single source of truth for the non-k8s comm "
        "configs (TCP, dual-bind); k8s pod manifests pull the same value via "
        "``K8sEnvironment.PORTS.EVENT_BUS_PROXY_PUB_FRONTEND`` (defaults match).",
    )
    EVENT_BUS_PROXY_BACKEND_PORT: int = Field(
        ge=1,
        le=65535,
        default=5664,
        description="Default TCP port for the event-bus XPUB/XSUB proxy backend "
        "(subscribers connect here). See ``EVENT_BUS_PROXY_FRONTEND_PORT``.",
    )


class _Environment(BaseSettings):
    """
    Root environment configuration with nested subsystem settings.

    This is a singleton instance that loads configuration from environment variables
    with the AIPERF_ prefix. Settings are organized into logical subsystems for
    better discoverability and maintainability.

    All nested settings can be configured via environment variables using the pattern:
    AIPERF_{SUBSYSTEM}_{SETTING_NAME}

    Example:
        AIPERF_HTTP_CONNECTION_LIMIT=5000
        AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8
        AIPERF_ZMQ_RCVTIMEO=600000
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Nested subsystem settings (alphabetically ordered)
    ACCURACY: _AccuracySettings = Field(
        default_factory=_AccuracySettings,
        description="Accuracy benchmark settings (dataset version pins, etc.)",
    )
    AGENTX: _AgentXSettings = Field(
        default_factory=_AgentXSettings,
        description="InferenceX AgentX scenario settings (context-overflow detection, etc.)",
    )
    API_SERVER: _APIServerSettings = Field(
        default_factory=_APIServerSettings,
        description="API server settings",
    )
    CHAT: _ChatSettings = Field(
        default_factory=_ChatSettings,
        description="Interactive `aiperf chat` command settings",
    )
    COMPRESSION: _CompressionSettings = Field(
        default_factory=_CompressionSettings,
        description="Compression settings for streaming file transfers",
    )
    CLI_RUNNER: _CLIRunnerSettings = Field(
        default_factory=_CLIRunnerSettings,
        description="CLI runner post-run callback isolation settings",
    )
    DATASET: _DatasetSettings = Field(
        default_factory=_DatasetSettings,
        description="Dataset loading and configuration settings",
    )
    DAG: _DagSettings = Field(
        default_factory=_DagSettings,
        description="DAG benchmark mode settings (dag_jsonl input type)",
    )
    DEV: _DeveloperSettings = Field(
        default_factory=_DeveloperSettings,
        description="Development and debugging settings",
    )
    GPU: _GPUSettings = Field(
        default_factory=_GPUSettings,
        description="GPU telemetry collection settings",
    )
    HTTP: _HTTPSettings = Field(
        default_factory=_HTTPSettings,
        description="HTTP client socket and connection settings",
    )
    LOGGING: _LoggingSettings = Field(
        default_factory=_LoggingSettings,
        description="Logging system settings",
    )
    METRICS: _MetricsSettings = Field(
        default_factory=_MetricsSettings,
        description="Metrics collection and storage settings",
    )
    MLFLOW: _MLflowSettings = Field(
        default_factory=_MLflowSettings,
        description="MLflow export settings",
    )
    OTEL: _OTelSettings = Field(
        default_factory=_OTelSettings,
        description="OpenTelemetry metrics streaming settings",
    )
    RECORD: _RecordSettings = Field(
        default_factory=_RecordSettings,
        description="Record processing and export settings",
    )
    SEARCH_PLANNER: _SearchPlannerSettings = Field(
        default_factory=_SearchPlannerSettings,
        description="Adaptive-search planner tunables",
    )
    NETWORK_LATENCY: _NetworkLatencySettings = Field(
        default_factory=_NetworkLatencySettings,
        description="Network latency calibration settings",
    )
    SERVER_METRICS: _ServerMetricsSettings = Field(
        default_factory=_ServerMetricsSettings,
        description="Server metrics collection settings",
    )
    SERVICE: _ServiceSettings = Field(
        default_factory=_ServiceSettings,
        description="Service lifecycle and communication settings",
    )
    TIMING: _TimingSettings = Field(
        default_factory=_TimingSettings,
        description="Timing manager settings",
    )
    TOKENIZER: _TokenizerSettings = Field(
        default_factory=_TokenizerSettings,
        description="Tokenizer pre-warm and loading settings",
    )
    UI: _UISettings = Field(
        default_factory=_UISettings,
        description="User interface and dashboard settings",
    )
    WANDB: _WandbSettings = Field(
        default_factory=_WandbSettings,
        description="Weights & Biases export settings",
    )
    WORKER: _WorkerSettings = Field(
        default_factory=_WorkerSettings,
        description="Worker management and scaling settings",
    )
    ZMQ: _ZMQSettings = Field(
        default_factory=_ZMQSettings,
        description="ZMQ communication settings",
    )

    @model_validator(mode="after")
    def validate_dev_mode(self) -> Self:
        """Validate that developer mode is enabled for features that require it."""
        if self.DEV.SHOW_INTERNAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_INTERNAL_METRICS"
            )
            self.DEV.SHOW_INTERNAL_METRICS = False

        if self.DEV.SHOW_EXPERIMENTAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS"
            )
            self.DEV.SHOW_EXPERIMENTAL_METRICS = False

        return self

    @model_validator(mode="after")
    def validate_profile_configure_timeout(self) -> Self:
        """Validate that the profile configure timeout is at least as long as the dataset configuration timeout."""
        if self.SERVICE.PROFILE_CONFIGURE_TIMEOUT < self.DATASET.CONFIGURATION_TIMEOUT:
            raise ValueError(
                f"AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT: {self.SERVICE.PROFILE_CONFIGURE_TIMEOUT} must be greater than or equal to AIPERF_DATASET_CONFIGURATION_TIMEOUT: {self.DATASET.CONFIGURATION_TIMEOUT}"
            )
        return self


# Global singleton instance
Environment = _Environment()
