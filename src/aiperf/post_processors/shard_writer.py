# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared per-processor shard-file machinery for record observers.

Record observers that persist output write one shard file per ``RecordProcessor``
so parallel processors never contend on a single file; a downstream aggregator
concatenates the shards at profile completion. Every such observer names its
file the same way and every aggregator merges the same way -- this module owns
both halves so the writers stop hand-rolling (and drifting on) that logic:

- :class:`ShardWriterMixin` resolves ``<shard_dir>/<prefix>_<sanitized id>.<ext>``.
- :class:`ShardAggregatorMixin` concatenates the shards back into one file.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import aiofiles

from aiperf.common.mixins import AIPerfLoggerMixin

_SHARD_ID_TRANSLATION = str.maketrans({"/": "_", ":": "_", " ": "_"})


class ShardWriterMixin:
    """Resolve this processor's private shard file within a shared shard directory.

    Mixed into a buffered record observer (raw records, output fragments, ...)
    to replace the copy-pasted ``output_dir`` / ``safe_id`` / ``output_file``
    boilerplate with one naming rule. Pair with :class:`ShardAggregatorMixin`
    on the corresponding exporter to merge the shards back.
    """

    @staticmethod
    def sanitize_shard_id(service_id: str | None) -> str:
        """Turn a ``service_id`` into a filesystem-safe shard suffix.

        Args:
            service_id: The owning processor's id, or ``None`` for a lone writer.

        Returns:
            ``service_id`` with ``/``, ``:`` and spaces replaced by ``_``,
            defaulting to ``"processor"`` when no id is supplied.
        """
        return (service_id or "processor").translate(_SHARD_ID_TRANSLATION)

    def shard_output_file(
        self,
        artifacts_dir: Path,
        folder: Path,
        *,
        prefix: str,
        ext: str,
        service_id: str | None,
    ) -> Path:
        """Build (and ensure the directory for) this processor's shard file.

        Args:
            artifacts_dir: The run's artifact directory (``artifacts.dir``).
            folder: Shard subdirectory under ``artifacts_dir`` (e.g. ``raw_records``).
            prefix: Filename stem shared by every shard (e.g. ``raw_records``).
            ext: File extension without the dot (e.g. ``jsonl``).
            service_id: The owning processor's id; sanitized into the filename.

        Returns:
            ``<artifacts_dir>/<folder>/<prefix>_<sanitized id>.<ext>``.
        """
        shard_dir = artifacts_dir / folder
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{prefix}_{self.sanitize_shard_id(service_id)}.{ext}"


class ShardAggregatorMixin(AIPerfLoggerMixin):
    """Concatenate per-processor shard files matching a glob into a single output file."""

    async def _concat_shards(
        self,
        shard_dir: Path,
        glob: str,
        output_file: Path,
        *,
        header_from_first: bool = False,
    ) -> int:
        """Merge every shard matching ``glob`` in ``shard_dir`` into ``output_file``.

        Shards are consumed in sorted order and deleted as they are merged; the
        shard directory is removed when empty. When ``header_from_first`` is set,
        the first line of every shard after the first is skipped so a repeated
        CSV header does not leak into the merged output.

        Args:
            shard_dir: Directory holding the per-processor shard files.
            glob: Glob pattern selecting shards, e.g. ``"raw_records_*.jsonl"``.
            output_file: Destination file; truncated before merging.
            header_from_first: Skip the leading line of shards after the first.

        Returns:
            The number of non-empty data lines written to ``output_file``,
            excluding the CSV header line (when ``header_from_first`` is set).
        """
        shards = sorted(shard_dir.glob(glob))
        if not shards:
            return 0
        output_file.unlink(missing_ok=True)
        count = 0
        async with aiofiles.open(output_file, "wb") as out:
            for i, shard in enumerate(shards):
                count += await self._append_shard(
                    shard,
                    out,
                    header_from_first=header_from_first,
                    is_first_shard=i == 0,
                )
                shard.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            shard_dir.rmdir()
        return count

    async def _append_shard(
        self,
        shard: Path,
        out,
        *,
        header_from_first: bool,
        is_first_shard: bool,
    ) -> int:
        """Append one shard's data lines to an already-open output file.

        When ``header_from_first`` is set, ``shard``'s first line is treated as a
        CSV header: written only when ``is_first_shard`` is True, and never counted
        as a data row either way, so repeated headers don't leak into the merged
        output and don't inflate the returned row count. When ``header_from_first``
        is False (e.g. JSONL merging), the first line is ordinary data.

        Args:
            shard: The shard file to read and append.
            out: The open output file handle to append lines to.
            header_from_first: Whether ``shard``'s first line is a CSV header to
                deduplicate rather than a plain data row.
            is_first_shard: Whether ``shard`` is the first shard being merged
                (its header, if any, is the one kept).

        Returns:
            The number of non-empty data lines appended (excluding the header).
        """
        count = 0
        async with aiofiles.open(shard, "rb") as f:
            first_line = True
            async for line in f:
                if header_from_first and first_line:
                    first_line = False
                    if is_first_shard and line.strip():
                        await out.write(line)
                    continue
                first_line = False
                if line.strip():
                    count += 1
                    await out.write(line)
        return count
