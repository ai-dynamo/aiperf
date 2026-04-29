# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace cache detection helpers.

Split out from ``tokenizer.py`` to keep that file under the ergonomics
file-size ceiling. ``aiperf.common.tokenizer`` re-exports these so callers
do not need to know about the split.
"""

from __future__ import annotations

from pathlib import Path


def _find_hf_cache_aliases(name: str) -> list[Path]:
    """Find HF cache directories matching a model name alias.

    Scans the HF hub cache for ``models--*--<name>`` directories
    (case-insensitive suffix match).

    Returns:
        List of matching cache directory paths.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.is_dir():
        return []

    suffix = f"--{name.lower()}"
    return [
        entry
        for entry in cache_dir.iterdir()
        if entry.is_dir()
        and entry.name.startswith("models--")
        and entry.name.lower().endswith(suffix)
    ]


def _is_revision_snapshot_cached(model_dir: Path, revision: str) -> bool:
    """Check if a specific revision snapshot exists in an HF model cache directory.

    Supports both named refs (``main``, ``v1.2``) and direct commit hashes.
    """
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    # Named ref: refs/<revision> contains the commit hash
    refs_file = model_dir / "refs" / revision
    if refs_file.is_file():
        commit_hash = refs_file.read_text().strip()
        return (snapshots_dir / commit_hash).is_dir()
    # Direct commit hash
    return (snapshots_dir / revision).is_dir()


def _is_hf_cached(name: str, revision: str | None = None) -> bool:
    """Check if a HuggingFace model is available in the local cache.

    Looks for ``models--<name>/`` (with ``/`` replaced by ``--``) inside the
    HF hub cache directory.  Also handles alias-style short names, returning
    True only when a single unambiguous match exists.

    When *revision* is given, also verifies that the specific revision snapshot
    is present — a model directory from a different revision is not sufficient.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.is_dir():
        return False

    # Exact match: "meta-llama/Llama-2-7b-hf" -> "models--meta-llama--Llama-2-7b-hf"
    exact = cache_dir / f"models--{name.replace('/', '--')}"
    if exact.is_dir():
        model_dir = exact
    else:
        aliases = _find_hf_cache_aliases(name)
        if len(aliases) != 1:
            return False
        model_dir = aliases[0]

    if revision is None:
        return True
    return _is_revision_snapshot_cached(model_dir, revision)
