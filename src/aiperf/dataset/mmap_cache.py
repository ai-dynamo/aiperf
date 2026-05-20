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
    - tokenizer identity tuple (name, revision, trust_remote_code, ...)
    - input/prompt config dump that affects tokenization or layout
    - aiperf release-tag-or-rev when AIPERF_VERSION is set; absent otherwise

The cache module exposes the low-level primitives (``compute_cache_key``,
``lookup``, ``populate``, ``restore_to_run_dir``) plus manifest data classes.
Higher-level helpers that build the key from a ``BenchmarkRun`` belong in the
dataset manager integration PR, since the per-config field mapping is
config-shape sensitive.

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
    Bumped whenever the on-disk layout or the side-data schema changes.
    Mismatches are treated as a MISS.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import orjson
from pydantic import Field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.common.models.base_models import AIPerfBaseModel

_logger = AIPerfLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
INPUTS_JSON_FILENAME = "inputs.json"

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
    of filesystem traversal order. Used so directory inputs (e.g. a one-file-
    per-trace corpus) get a content-addressed cache key that differentiates two
    directories with the same name but different contents.
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
            so two directories with the same name but different contents
            produce distinct keys.
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
        ge=0,
        description="Manifest format version. Bumped on any on-disk layout or schema change.",
    )
    cache_key: str = Field(
        ..., description="The content+settings hash that produced this entry."
    )
    created_at: float = Field(
        ...,
        ge=0.0,
        description="Unix epoch time at which the entry was populated.",
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
        description="Stored memory-map format identifier (e.g. 'conversation', 'payload_bytes').",
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
    except Exception as e:  # noqa: BLE001 -- corrupt cache is recoverable; downgrade to MISS
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


def restore_to_run_dir(
    hit: CacheHit, run_data_path: Path, run_index_path: Path
) -> None:
    """Copy cached dataset/index files into the run directory.

    The run directory is created if needed. Files are copied (not symlinked) so the
    backing-store cleanup hook can ``unlink`` them at run end without nuking the cache.
    """
    run_data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hit.data_path, run_data_path)
    shutil.copyfile(hit.index_path, run_index_path)


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

        if inputs_json_path is not None and inputs_json_path.exists():
            shutil.copyfile(inputs_json_path, tmp_dir / INPUTS_JSON_FILENAME)
            manifest.has_inputs_json = True
        else:
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
    except Exception as e:  # noqa: BLE001 -- cache populate failure must not break the run
        _logger.warning(f"Failed to populate mmap cache entry {cache_key}: {e!r}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def aiperf_version() -> str | None:
    """Return AIPERF_VERSION env var if set, else None."""
    return os.environ.get("AIPERF_VERSION") or None
