# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed disk cache for memory-mapped dataset files.

Re-runs whose input bytes, tokenizer identity, and prompt/input settings are
byte-identical reuse the previously-tokenized ``dataset.dat`` / ``index.dat``
pair instead of re-tokenizing from scratch.

Cache key inputs:
    - sha256 of the input file bytes (None if no file -- e.g. synthetic)
    - public_dataset name (e.g. "openai/openai_humaneval") if any
    - custom_dataset_type (e.g. "mooncake_trace") if any
    - tokenizer identity tuple: (name, revision, trust_remote_code, apply_chat_template)
    - input/prompt config dump that affects tokenization or layout, including
      num_conversations, num_dataset_entries, sampling_strategy, and the entire
      ``input.prompt`` config (excluding the cache_bust subtree -- see below)
    - aiperf release-tag-or-rev when AIPERF_VERSION is set; absent otherwise

Cache-bust deliberately does NOT enter the key. The mmap holds template bytes
that the worker re-randomizes per-request, so two runs with different
cache_bust settings can safely share the same cached mmap.

On-disk layout::

    <cache_dir>/<key>/
        dataset.dat         # mmap data file (or .dat.zst when compress_only)
        index.dat           # mmap index file (or .dat.zst when compress_only)
        manifest.json       # orjson; version + side-data needed to skip the composer
        inputs.json         # optional; copied from artifact dir on populate

Concurrency: writers populate to ``<cache_dir>/<key>.tmp.<pid>`` and atomically
``os.replace`` the directory into place. A reader that finds a partial entry
(missing manifest.json) treats the entry as a MISS and overwrites it.

Manifest version:
    Bumped whenever the on-disk layout, the side-data schema, or the decoded
    content the loaders produce for a given key changes. Mismatches are treated
    as a MISS.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from pydantic import Field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.dataset.mmap_cache_lock import acquire_cache_lock as _acquire_cache_lock
from aiperf.plugin import plugins

if TYPE_CHECKING:
    from aiperf.common.config import UserConfig

_logger = AIPerfLogger(__name__)

# Bump when the on-disk layout, side-data schema, OR the decoded content the
# loaders produce for a given key changes -- the key has no source-code
# component, so a content-semantics fix must bump this or warm caches keep
# serving the old (wrong) dataset. Version 7 invalidates entries built before
# the weka context-loss rule (a turn whose truncation removes every user
# segment now resumes at a USER turn instead of opening with a fabricated
# assistant segment -- role boundaries moved for seam/reset re-emits).
# Version 6 invalidates entries built before
# the weka subagent hash_id-scope fix (subagents now share the parent trace's
# scope, so shared blocks decode to different tokens than v5 produced).
# Version 5 fixed the Conversation.metadata() projection of per-turn
# theoretical prefix-cache block counts for realtime infinite-cache hit rate.
MANIFEST_VERSION = (
    # v21: Weka traces carry explicit api_time interval-frontier metadata in
    # DatasetMetadata (replay_scope_id + per-turn replay_predecessors). Cached
    # manifests produced before v21 deserialize with empty defaults, silently
    # disabling fan-out/join barriers even though dataset.dat remains usable.
    # Rebuild so the manifest sidecar contains the inferred dependency graph.
    # v20: DAG datasets (any FORK/SPAWN branch) are no longer preformatted into
    # the PAYLOAD_BYTES mmap fast path -- they are delta-compressed and
    # accumulate context across the tree (FORK children seed from the parent's
    # live session), which payload_bytes cannot represent. A pre-v20 warm cache
    # could hold a poisoned PAYLOAD_BYTES entry for a single-turn-root-with-branch
    # dataset; bumping invalidates those so they rebuild as CONVERSATION.
    # v19: worker-group grouping now requires BOTH a shared fork point AND
    # temporal overlap (the corpus research + graph adapter prescription:
    # overlapping intervals AND a shared prefix). Workers are scoped by fork
    # point, then split into connected components of overlapping [t0,t1)
    # intervals within each scope. Pure overlap alone bridged a busy trace into
    # one blob (a chain of overlaps spans the session); the fork-point scope
    # prevents that, and the overlap split drops fork-point members that never
    # run concurrently (e.g. a seam-re-keyed phantom fork). ::wg:{group}_{member}
    # membership changed.
    # v18: worker-group grouping re-keyed from hash_ids[0] (block-0) to the
    # fork point (fork.parent_chain + fork.fork_outer_idx) -- the deep spawn
    # relationship that actually identifies a coordinated fan-out. block-0 is the
    # shallow common root (~system prompt) shared by ~all of a session's workers,
    # so it lumped unrelated fan-outs into one coarse blob. The coordinate drops
    # to ::wg:{group}_{member} (the temporal burst is gone -- fork-point groups
    # are inherently time-tight), so ::wg: child session ids changed again.
    # WEKA_WORKER_GROUP_BURST_GAP_SECONDS removed from the cache key.
    # v17: worker-group session ids now encode the parallel-fan-out coordinate
    # as ::wg:{lineage}_{burst}_{member} (was ::wg:{NNN}). lineage = shared-
    # spawn-block group; burst = temporal dispatch wave split at
    # WEKA_WORKER_GROUP_BURST_GAP_SECONDS; member = index within the burst. Those
    # ::wg: child session ids changed; ::fa:/::aux:/::aux:red: are unchanged.
    # v16: aux classification gained a reduction arm and worker-group tagging.
    # A same-model single-request large-input/short-output one-shot (context
    # compaction, subagent-result summary, tool-output digest) is now a sidecar
    # at ::aux:red: (and :aux:red: under subagents) via WEKA_AUX_REDUCTION_*;
    # and a worker chain sharing a spawn block with >= WEKA_WORKER_GROUP_MIN
    # siblings and a deep fork is tagged ::wg: (parallel fan-out agent) instead
    # of the generic ::fa:, so those child session ids changed.
    # v15: aux classification gained a cross-model arm -- a one-shot worker
    # chain on a different model than its enclosing main chain (e.g. a Haiku
    # WebFetch summary under an Opus agent) is now a sidecar regardless of
    # payload size, so large cross-model singletons move ::fa: -> ::aux: (and
    # :fa: -> :aux: under subagents). Gated by WEKA_AUX_CROSS_MODEL.
    # v14: subagent nested-LCP overflow session ids restructured to match the
    # top-level vocabulary -- the agent marker is renamed and separated
    # (:cNNN -> :fa:NNN) and short, small-fresh-context overflow splits off as
    # sidecars (:aux:NNN), so those child session ids changed. Same WEKA_AUX_*
    # knobs; yardstick is the subagent's own main-chain peak ISL.
    # v13: flattened-agent worker chains that are short, small-fresh-context
    # one-shot calls are reclassified ::fa: -> ::aux: (auxiliary sidecars), so
    # those child session ids changed (::fa:{NNN} -> ::aux:{NNN}). Governed by
    # WEKA_AUX_MAX_REQUESTS / WEKA_AUX_ISL_RATIO / WEKA_AUX_ISL_FLOOR (also in
    # the cache key, so tuning them re-keys without a version bump).
    # v12: weka subagent inner requests split by nested LCP chain detection
    # instead of time-interval stream packing -- child session ids changed
    # (::sa:{agent_id} main chain + :c{NNN} spawned chains replace the
    # :s{i} streams), child counts and SPAWN branch memberships changed,
    # child turn timestamps moved to root-trace coordinates, and
    # spawned-chain turn-0 tool/system attribution is proof-gated.
    # v11: the system role is never fabricated from the observed
    # namespace-group prefix (0/0-declared chains bake all-user turn 0s) --
    # role boundaries changed again relative to v10.
    # v10: merge of the flattened-agent-splitting lineage and the
    # tool-shaping lineage (boundary-cut overhang strip; shaping decided at
    # first emission so reset re-emits reproduce the first-sent shape).
    21
)
MANIFEST_FILENAME = "manifest.json"
INPUTS_JSON_FILENAME = "inputs.json"

# Re-exported with cache_dir resolver pre-bound.
acquire_cache_lock = functools.partial(
    _acquire_cache_lock, cache_dir_resolver=lambda: cache_dir()
)

# Bytes hashed in one read pass. 8 MiB strikes a balance between memory use
# and syscall count for very large input files.
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _default_cache_dir() -> Path:
    """Resolve the default cache directory (``~/.cache/aiperf/dataset_mmap``)."""
    return Path.home() / ".cache" / "aiperf" / "dataset_mmap"


def cache_dir() -> Path:
    """Return the active cache directory, honouring environment overrides."""
    configured = Environment.DATASET.MMAP_CACHE_DIR
    return Path(configured) if configured is not None else _default_cache_dir()


def cache_enabled() -> bool:
    """Return True when the mmap cache is enabled."""
    return bool(Environment.DATASET.MMAP_CACHE_ENABLED)


def hash_file_bytes(path: Path) -> str:
    """Return the hex-encoded sha256 of the bytes in ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def hash_dir_contents(path: Path) -> str:
    """Return a sha256 over the relative paths and bytes of every file under ``path``.

    Walks ``path`` recursively in sorted order so the digest is stable regardless
    of filesystem traversal order. Used so directory inputs (e.g. the weka_trace
    one-file-per-trace corpus) get a content-addressed cache key that
    differentiates two directories with the same name but different contents.
    """
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with child.open("rb") as f:
            while chunk := f.read(_HASH_CHUNK_BYTES):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _hash_input_path(path: Path) -> str:
    """Return a content digest for ``path`` (file or directory)."""
    return hash_dir_contents(path) if path.is_dir() else hash_file_bytes(path)


def compute_cache_key(
    *,
    input_file: Path | None,
    public_dataset: str | None,
    custom_dataset_type: str | None,
    tokenizer_identity: dict[str, object],
    settings_payload: dict[str, object],
    aiperf_version: str | None = None,
) -> str:
    """Build the content+settings cache key.

    Args:
        input_file: Path to the user-supplied input file or directory, or None
            for synthetic. Directories are hashed via :func:`hash_dir_contents`
            so two directories with the same name but different contents (e.g.
            distinct weka_trace corpora under tmp_path) produce distinct keys.
        public_dataset: Public-dataset name (None when not used).
        custom_dataset_type: Custom-dataset-type identifier (None when not used).
        tokenizer_identity: Stable dict identifying the tokenizer.
        settings_payload: Stable dict of input/prompt settings that influence
            tokenization or mmap layout. MUST NOT contain cache_bust settings.
        aiperf_version: Optional AIPerf version/rev string included in the hash.

    Returns:
        A 32-character hex digest used as the cache subdirectory name.
    """
    payload: dict[str, object] = {
        "v": MANIFEST_VERSION,
        "input_file_sha256": (
            _hash_input_path(input_file) if input_file is not None else None
        ),
        "input_file_name": input_file.name if input_file is not None else None,
        "public_dataset": public_dataset,
        "custom_dataset_type": custom_dataset_type,
        "tokenizer": tokenizer_identity,
        "settings": settings_payload,
        "aiperf_version": aiperf_version,
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:32]


class CacheManifest(AIPerfBaseModel):
    """Side-data persisted alongside dataset.dat/index.dat in a cache entry.

    Bumping ``version`` invalidates older entries (treated as MISS).
    """

    version: int = Field(
        default=MANIFEST_VERSION,
        description="Manifest format version. Bumped on any on-disk layout or schema change.",
    )
    cache_key: str = Field(
        ..., description="The content+settings hash that produced this entry."
    )
    created_at: float = Field(
        ..., description="Unix epoch time at which the entry was populated."
    )
    aiperf_version: str | None = Field(
        default=None,
        description="AIPerf version/rev that produced this entry, when known.",
    )
    num_conversations: int = Field(
        ..., ge=0, description="Number of conversations in the cached dataset."
    )
    total_size_bytes: int = Field(
        ..., ge=0, description="Total uncompressed size of the cached dataset bytes."
    )
    compressed: bool = Field(
        default=False,
        description="If True, dataset.dat/index.dat are zstd-compressed (compress_only mode).",
    )
    compressed_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Size of the compressed dataset file when compressed=True.",
    )
    mmap_format: str = Field(
        ...,
        description="Stored MemoryMapFormat value (conversation or payload_bytes).",
    )
    default_context_mode: str | None = Field(
        default=None,
        description="ConversationContextMode the loader assigned, if any.",
    )
    all_turns_source_loaded_payloads: bool = Field(
        default=False,
        description="Whether every turn carried a source-loaded raw_payload before pre-formatting.",
    )
    dataset_metadata_json: str = Field(
        ...,
        description="DatasetMetadata serialized as JSON string for cross-version restore.",
    )
    has_inputs_json: bool = Field(
        default=False,
        description="True when the cache entry has a sibling inputs.json blob.",
    )


class CacheHit(AIPerfBaseModel):
    """Resolved paths and side-data returned on a cache HIT."""

    entry_dir: Path = Field(..., description="Directory holding the cache entry.")
    data_path: Path = Field(..., description="Cached dataset.dat (or .dat.zst) path.")
    index_path: Path = Field(..., description="Cached index.dat (or .dat.zst) path.")
    inputs_json_path: Path | None = Field(
        default=None,
        description="Cached inputs.json path when has_inputs_json=True; None otherwise.",
    )
    manifest: CacheManifest = Field(..., description="Decoded manifest contents.")


def _read_manifest(entry_dir: Path) -> CacheManifest | None:
    """Decode and return the manifest, or None if missing/invalid/version-mismatched."""
    manifest_path = entry_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        raw = orjson.loads(manifest_path.read_bytes())
        manifest = CacheManifest.model_validate(raw)
    except Exception as e:
        _logger.warning(f"Ignoring corrupt cache manifest at {manifest_path}: {e!r}")
        return None
    if manifest.version != MANIFEST_VERSION:
        _logger.info(
            lambda: (
                f"Cache entry {entry_dir.name} has manifest version "
                f"{manifest.version} != current {MANIFEST_VERSION}; treating as MISS."
            )
        )
        return None
    return manifest


def lookup(cache_key: str, *, compressed: bool) -> CacheHit | None:
    """Return a CacheHit for ``cache_key`` if a complete entry exists, else None.

    Args:
        cache_key: The content+settings hash returned by ``compute_cache_key``.
        compressed: When True, expect ``.dat.zst`` files (compress_only mode).

    Returns:
        A populated CacheHit on HIT; None on MISS (including partial/corrupt entries).
    """
    entry_dir = cache_dir() / cache_key
    if not entry_dir.is_dir():
        return None
    manifest = _read_manifest(entry_dir)
    if manifest is None:
        return None
    if manifest.compressed != compressed:
        _logger.info(
            lambda: (
                f"Cache entry {cache_key} compressed={manifest.compressed} but caller "
                f"requested compressed={compressed}; treating as MISS."
            )
        )
        return None

    ext = ".dat.zst" if compressed else ".dat"
    data_path = entry_dir / f"dataset{ext}"
    index_path = entry_dir / f"index{ext}"
    if not data_path.exists() or not index_path.exists():
        _logger.warning(
            f"Cache entry {cache_key} is missing dataset/index files; treating as MISS."
        )
        return None

    inputs_json_path: Path | None = None
    if manifest.has_inputs_json:
        candidate = entry_dir / INPUTS_JSON_FILENAME
        if candidate.exists():
            inputs_json_path = candidate

    return CacheHit(
        entry_dir=entry_dir,
        data_path=data_path,
        index_path=index_path,
        inputs_json_path=inputs_json_path,
        manifest=manifest,
    )


def _restore_file(src: Path, dst: Path) -> str:
    start = time.perf_counter()
    try:
        os.link(src, dst)
        method = "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        method = "copy"
    duration = time.perf_counter() - start
    size_gib = src.stat().st_size / (1024**3)
    _logger.info(
        f"Restored mmap cache file {dst.name} via {method} "
        f"in {duration:.3f}s ({size_gib:.2f} GiB)"
    )
    return method


def restore_to_run_dir(
    hit: CacheHit, run_data_path: Path, run_index_path: Path
) -> None:
    """Restore cached dataset/index files into the run directory."""
    total_start = time.perf_counter()
    run_data_path.parent.mkdir(parents=True, exist_ok=True)
    data_method = _restore_file(hit.data_path, run_data_path)
    index_method = _restore_file(hit.index_path, run_index_path)
    total_duration = time.perf_counter() - total_start
    data_size_gib = hit.data_path.stat().st_size / (1024**3)
    index_size_gib = hit.index_path.stat().st_size / (1024**3)
    _logger.info(
        f"Restored mmap cache files in {total_duration:.3f}s via "
        f"dataset={data_method} index={index_method} "
        f"(dataset={data_size_gib:.2f} GiB, index={index_size_gib:.2f} GiB)"
    )


def populate(
    *,
    cache_key: str,
    run_data_path: Path,
    run_index_path: Path,
    manifest: CacheManifest,
    inputs_json_path: Path | None = None,
) -> Path | None:
    """Populate the cache with the artifacts a successful run produced.

    Writes a tmp dir and atomically renames it into ``<cache_dir>/<cache_key>``.
    A pre-existing entry at the same key is left in place (winner-stays).

    Args:
        cache_key: Cache key for the new entry.
        run_data_path: Source dataset.dat (or .dat.zst) from the run.
        run_index_path: Source index.dat (or .dat.zst) from the run.
        manifest: Manifest to serialize into the entry.
        inputs_json_path: Optional inputs.json to copy alongside.

    Returns:
        The committed entry directory, or None when no entry was committed
        (a concurrent populate already won, or an error rendered the entry partial).
    """
    base = cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    final_dir = base / cache_key

    if final_dir.exists() and (final_dir / MANIFEST_FILENAME).exists():
        _logger.debug(lambda: f"Cache entry {cache_key} already populated; skipping.")
        return final_dir

    tmp_dir = base / f".{cache_key}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        ext = run_data_path.suffix
        ext_index = run_index_path.suffix
        # Use the source file extension verbatim so .dat.zst stays .dat.zst.
        cache_data = tmp_dir / (
            "dataset.dat.zst"
            if str(run_data_path).endswith(".dat.zst")
            else f"dataset{ext}"
        )
        cache_index = tmp_dir / (
            "index.dat.zst"
            if str(run_index_path).endswith(".dat.zst")
            else f"index{ext_index}"
        )
        shutil.copyfile(run_data_path, cache_data)
        shutil.copyfile(run_index_path, cache_index)

        manifest.has_inputs_json = False

        manifest_bytes = orjson.dumps(
            manifest.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2,
        )
        (tmp_dir / MANIFEST_FILENAME).write_bytes(manifest_bytes)

        try:
            os.replace(tmp_dir, final_dir)
        except OSError:
            # Another writer beat us; leave their entry, drop ours.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return final_dir if final_dir.exists() else None
        _logger.info(f"Populated mmap cache entry {final_dir}")
        return final_dir
    except Exception as e:
        _logger.warning(f"Failed to populate mmap cache entry {cache_key}: {e!r}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def _aiperf_version() -> str | None:
    """Return AIPERF_VERSION env var if set, else None."""
    return os.environ.get("AIPERF_VERSION") or None


def _tokenizer_identity_from_user_config(
    user_config: UserConfig,
) -> dict[str, object]:
    """Stable dict identifying the tokenizer.

    Mirrors the fields ``DatasetManager._configure_tokenizer`` consumes plus
    ``apply_chat_template`` since chat-template wrapping changes tokenized ISL.
    """
    model_name = user_config.endpoint.model_names[0]
    tokenizer_config = user_config.tokenizer
    tokenizer_name = tokenizer_config.get_tokenizer_name_for_model(model_name)
    return {
        "name": tokenizer_name,
        "revision": tokenizer_config.revision,
        "trust_remote_code": bool(tokenizer_config.trust_remote_code),
        "apply_chat_template": bool(tokenizer_config.apply_chat_template),
    }


def _public_dataset_source_from_user_config(
    user_config: UserConfig,
) -> dict[str, object] | None:
    inp = user_config.input
    if inp.public_dataset is None:
        return None

    public_dataset = str(inp.public_dataset)
    metadata = plugins.get_public_dataset_loader_metadata(public_dataset)
    hf_dataset_name = inp.hf_weka_dataset or metadata.hf_dataset_name
    if hf_dataset_name is None:
        return {"plugin": public_dataset}

    hf_subset = (
        inp.hf_dataset_subset
        if inp.hf_dataset_subset is not None
        else metadata.hf_subset
    )
    source = metadata.model_dump(mode="json", exclude_none=True)
    source["hf_dataset_name"] = hf_dataset_name
    source["hf_split"] = metadata.hf_split
    source["hf_subset"] = hf_subset
    return source


def _settings_payload_from_user_config(
    user_config: UserConfig,
) -> dict[str, object]:
    """Stable dict of input/prompt settings that affect mmap layout.

    Excludes ``cache_bust`` deliberately. Cache-bust mutates per-request bytes
    at the worker, not the mmap template -- two runs differing only in
    cache-bust settings can safely share the cached mmap.
    """
    inp = user_config.input
    prompt_dump = inp.prompt.model_dump(mode="json", exclude_none=False)
    prompt_dump.pop("cache_bust", None)
    return {
        "num_dataset_entries": inp.conversation.num_dataset_entries,
        "dataset_sampling_strategy": str(inp.dataset_sampling_strategy),
        "custom_dataset_type": (
            str(inp.custom_dataset_type)
            if inp.custom_dataset_type is not None
            else None
        ),
        "public_dataset_source": _public_dataset_source_from_user_config(user_config),
        "prompt": prompt_dump,
        "endpoint_type": str(user_config.endpoint.type),
        "model_name": user_config.endpoint.model_names[0],
        "fixed_schedule_start_offset": inp.fixed_schedule_start_offset,
        "fixed_schedule_end_offset": inp.fixed_schedule_end_offset,
        # Load-time timing knobs bake into the cached Turn timestamps/delays
        # (applied during reconstruction, not at request time), so they must
        # key the cache or a warm entry silently serves the other mode.
        "ignore_trace_delays": inp.ignore_trace_delays,
        "use_think_time_only": inp.use_think_time_only,
        "inter_turn_delay_cap_seconds": (
            user_config.loadgen.inter_turn_delay_cap_seconds
        ),
        "trace_idle_gap_cap_seconds": getattr(
            user_config.loadgen, "trace_idle_gap_cap_seconds", None
        ),
        "weka_live_assistant_responses": (
            Environment.DATASET.WEKA_LIVE_ASSISTANT_RESPONSES
        ),
        "weka_split_flattened_agents": (
            Environment.DATASET.WEKA_SPLIT_FLATTENED_AGENTS
        ),
        "weka_aux_max_requests": Environment.DATASET.WEKA_AUX_MAX_REQUESTS,
        "weka_aux_isl_ratio": Environment.DATASET.WEKA_AUX_ISL_RATIO,
        "weka_aux_isl_floor": Environment.DATASET.WEKA_AUX_ISL_FLOOR,
        "weka_aux_cross_model": Environment.DATASET.WEKA_AUX_CROSS_MODEL,
        "weka_aux_reduction_osl_max": (Environment.DATASET.WEKA_AUX_REDUCTION_OSL_MAX),
        "weka_aux_reduction_ratio": Environment.DATASET.WEKA_AUX_REDUCTION_RATIO,
        "weka_worker_group_min": Environment.DATASET.WEKA_WORKER_GROUP_MIN,
        "weka_tool_shaped_messages": (Environment.DATASET.WEKA_TOOL_SHAPED_MESSAGES),
        "max_isl": inp.synthesis.max_isl,
        "max_osl": inp.synthesis.max_osl,
        "max_context_length": inp.max_context_length,
    }


def compute_cache_key_from_user_config(user_config: UserConfig) -> str | None:
    """Build a cache key for ``user_config`` or return None when caching is unsafe.

    Returns None for synthetic-only runs (no input file, no public dataset, no
    custom dataset type) -- those are cheap and the seed/distribution interplay
    makes content-addressing brittle. Returns None for accuracy mode (loader
    has its own dataset semantics that don't share mmap shape with normal mode).
    """
    if user_config.accuracy.enabled:
        return None
    inp = user_config.input
    input_file: Path | None = None
    if inp.file is not None:
        candidate = Path(inp.file)
        if candidate.is_file() or candidate.is_dir():
            input_file = candidate
    has_source = (
        input_file is not None
        or inp.public_dataset is not None
        or inp.custom_dataset_type is not None
    )
    if not has_source:
        return None

    return compute_cache_key(
        input_file=input_file,
        public_dataset=None,
        custom_dataset_type=(
            str(inp.custom_dataset_type)
            if inp.custom_dataset_type is not None
            else None
        ),
        tokenizer_identity=_tokenizer_identity_from_user_config(user_config),
        settings_payload=_settings_payload_from_user_config(user_config),
        aiperf_version=_aiperf_version(),
    )
