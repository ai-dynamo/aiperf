# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job index for fast lookups without scanning individual result files.

Maintains a ``jobs_index.json`` file at the PVC root that is updated at two
points in the job lifecycle:

1. **on_create** — records the CR spec, model, endpoint, start time.
2. **on_completion** — records end time, phase, key metrics, file list.

The index is a simple JSON object keyed by ``{namespace}/{job_id}``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from aiperf.operator.environment import OperatorEnvironment

logger = logging.getLogger(__name__)

INDEX_FILENAME = "jobs_index.json"

# Serializes concurrent read-modify-write access to jobs_index.json. kopf
# dispatches handlers for different AIPerfJob CRs concurrently; without this
# lock, two handlers can read the same snapshot and the later writer silently
# clobbers the earlier writer's update. Operator runs single-replica, so an
# in-process asyncio.Lock is sufficient.
_index_lock = asyncio.Lock()


def _index_path() -> Path:
    return OperatorEnvironment.RESULTS.DIR / INDEX_FILENAME


def _read_index() -> dict[str, Any]:
    """Read the current index, returning empty dict if missing or corrupt."""
    path = _index_path()
    if not path.exists():
        return {}
    try:
        return orjson.loads(path.read_bytes())
    except Exception as e:
        logger.warning(f"Failed to read job index: {e}")
        return {}


def _write_index(data: dict[str, Any]) -> None:
    """Atomically write the index file."""
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.replace(path)


def _job_key(namespace: str, job_id: str) -> str:
    return f"{namespace}/{job_id}"


async def index_job_created(
    namespace: str,
    job_id: str,
    spec: dict[str, Any],
) -> None:
    """Record a new job in the index at creation time.

    Saves the original CR spec plus extracted metadata for fast lookups.
    """

    def _do() -> None:
        index = _read_index()
        key = _job_key(namespace, job_id)

        # Extract key metadata from the spec for quick access
        benchmark = spec.get("benchmark", spec)
        endpoint_cfg = benchmark.get("endpoint", {})
        models_cfg = benchmark.get("models", {})

        # models can be: {"items": [{"name": "x"}]}, {"modelNames": ["x"]}, or just ["x"]
        if isinstance(models_cfg, list):
            model_items = models_cfg
        else:
            model_items = models_cfg.get("items", models_cfg.get("modelNames", []))
        model_name = None
        if isinstance(model_items, list) and model_items:
            first = model_items[0]
            model_name = (
                first.get("name", first) if isinstance(first, dict) else str(first)
            )

        urls = endpoint_cfg.get("urls", endpoint_cfg.get("url", []))
        endpoint_url = (
            urls[0]
            if isinstance(urls, list) and urls
            else (urls if isinstance(urls, str) else None)
        )

        index[key] = {
            "namespace": namespace,
            "job_id": job_id,
            "model": model_name,
            "endpoint": endpoint_url,
            "phase": "Pending",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "throughput_rps": None,
            "latency_p99_ms": None,
            "error": None,
            "file_count": 0,
            "spec": spec,
        }
        _write_index(index)
        logger.info(f"Indexed job {key} at creation (model={model_name})")

    async with _index_lock:
        await asyncio.to_thread(_do)


async def index_job_completed(
    namespace: str,
    job_id: str,
    phase: str,
    metrics: dict[str, Any] | None = None,
    downloaded_files: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Update the index when a job completes (or fails).

    Merges completion data into the existing index entry.
    """

    def _do() -> None:
        index = _read_index()
        key = _job_key(namespace, job_id)

        entry = index.get(key, {"namespace": namespace, "job_id": job_id})
        entry["phase"] = phase
        entry["end_time"] = datetime.now(timezone.utc).isoformat()
        entry["error"] = error

        if downloaded_files:
            entry["file_count"] = len(downloaded_files)

        if metrics:
            rt = metrics.get("request_throughput", {})
            rl = metrics.get("request_latency", {})
            entry["throughput_rps"] = rt.get("avg")
            entry["latency_p99_ms"] = rl.get("p99")

        index[key] = entry
        _write_index(index)
        logger.info(f"Indexed job {key} completion (phase={phase})")

    async with _index_lock:
        await asyncio.to_thread(_do)


async def index_job_failed(
    namespace: str,
    job_id: str,
    error: str,
) -> None:
    """Update the index when a job fails."""
    await index_job_completed(namespace, job_id, phase="Failed", error=error)


async def get_index() -> dict[str, Any]:
    """Read the full index (async-safe)."""
    return await asyncio.to_thread(_read_index)


async def get_job_spec(namespace: str, job_id: str) -> dict[str, Any] | None:
    """Get the original CR spec for a specific job."""
    index = await get_index()
    key = _job_key(namespace, job_id)
    entry = index.get(key)
    if entry is None:
        return None
    return entry.get("spec")


async def save_job_spec_file(namespace: str, job_id: str, spec: dict[str, Any]) -> None:
    """Save the CR spec as a standalone JSON file in the job's results directory.

    This is a belt-and-suspenders approach: the spec is also in the index,
    but having it as a standalone file makes it available even if the index
    is regenerated. Disk I/O is offloaded to a worker thread so we don't block
    the kopf event loop under reconcile load.
    """
    dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
    path = dest_dir / "job_spec.json"
    payload = orjson.dumps(spec, option=orjson.OPT_INDENT_2)

    def _write() -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    await asyncio.to_thread(_write)
    logger.info(f"Saved CR spec to {path}")
