# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TraceLab agentic-coding trace loader.

Reads the TraceLab (uw-syfi) corpus, which publishes one JSONL row per LLM
round, groups those rounds into sessions, and hands the result to
:class:`WekaTraceLoader` for reconstruction. File-based Weka replay and
TraceLab replay therefore share one reconstruction body: the same hash_id
replay, model mapping, branch / spawn-join linkage and delay capping.

Two things this loader does that a plain format adapter does not, both because
the corpus does not record them:

* **Block ids.** TraceLab has no content hashes of any kind. It does record an
  engine-reported ``prefix_tokens`` / ``newly_append_tokens`` split of every
  round's input, which is enough to mint per-session virtual block ids that
  reproduce the recorded prefix-reuse to block granularity. See
  :func:`~aiperf.dataset.loader._tracelab_convert.synthesize_hash_ids`.
* **Subagent nesting.** A subagent round appears in the corpus as its own
  top-level ``session_id`` with no parent link of any kind. The link is
  recovered by timing containment against the spawning tool call's window. See
  :func:`~aiperf.dataset.loader._tracelab_convert.build_join_index`.

Usage::

    aiperf profile --custom-dataset-type tracelab \\
        --input-file syfi_coding_trace.jsonl.gz ...

Conversion runs in memory; no intermediate files are written.
"""

from __future__ import annotations

import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.common.utils import open_text_maybe_gzip
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader._tracelab_convert import (
    build_join_index,
    build_trace,
    group_children_by_parent,
    safe_trace_id,
)
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# Historical default, and the value this loader declares in its plugin
# metadata. Kept as a named constant so the fallback and the metadata cannot
# drift apart silently.
DEFAULT_BLOCK_SIZE = 64

# Fields every TraceLab row carries. Used only for content auto-detection, so
# it is deliberately a small distinctive subset rather than the full schema:
# ``prefix_tokens`` + ``newly_append_tokens`` + ``timing_events`` together do
# not co-occur in any other format this repo reads.
_DETECT_KEYS = frozenset(
    {
        "session_id",
        "round_index",
        "input_tokens_total",
        "prefix_tokens",
        "newly_append_tokens",
        "timing_events",
    }
)

_JSONL_SUFFIXES = (".jsonl", ".jsonl.gz", ".json.gz", ".gz")


class TraceLabTraceDatasetLoader(BaseFileLoader):
    """Dataset loader for the TraceLab agentic-coding corpus.

    Parses TraceLab JSONL (optionally gzipped), builds :class:`WekaTrace`
    objects in memory, and delegates conversation reconstruction to
    :class:`WekaTraceLoader` so this path and file-based Weka replay cannot
    diverge.

    Each recovered session produces:

    - one root :class:`Conversation` from its rounds, and
    - one child :class:`Conversation` per recovered subagent, linked by
      SPAWN + SPAWN_JOIN prerequisites on the parent's turns.

    Behaviour knobs live on the ``AIPERF_DATASET_TRACELAB_*`` environment
    variables rather than the CLI, since they select between reconstruction
    hypotheses rather than benchmark parameters.

    Raises:
        DatasetLoaderError: unreadable input, or a converted session that does
            not satisfy the trace schema.
    """

    tag: ClassVar[str] = "TraceLabTrace"

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        run: BenchmarkRun | None = None,
        prompt_generator: PromptGenerator | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, run=run, **kwargs)
        self.prompt_generator = prompt_generator

        dataset = self.run.cfg.get_default_dataset()
        # --isl-block-size lands on the flat FileDataset.block_size (routed by
        # the config converter for hash-id trace formats); a synthetic-via-file
        # dataset carries it on prompts.block_size. Plugin metadata is the
        # fallback, and DEFAULT_BLOCK_SIZE the floor. Unlike Weka -- whose
        # traces declare their own per-file block size, so an override would be
        # meaningless -- TraceLab has no recorded block size at all: the value
        # chosen here is what the block ids are synthesized AT, so it is a real
        # knob and the user's value must win.
        prompts = getattr(dataset, "prompts", None)
        block_size = getattr(dataset, "block_size", None)
        if block_size is None and prompts is not None:
            block_size = getattr(prompts, "block_size", None)
        if block_size is None:
            block_size = default_block_size
        self._block_size = (
            int(block_size) if block_size is not None else DEFAULT_BLOCK_SIZE
        )
        if self._block_size <= 0:
            raise DatasetLoaderError(
                f"TraceLab block size must be positive, got {self._block_size}."
            )

        self._join_subagents = Environment.DATASET.TRACELAB_SUBAGENT_JOIN
        self._join_codex = Environment.DATASET.TRACELAB_CODEX_SUBAGENT_JOIN
        self._min_spawn_ms = Environment.DATASET.TRACELAB_MIN_SPAWN_MS

        # The delegate is constructed in HF-delegation mode (filename=None): it
        # never touches a file, it only reconstructs the traces handed to it.
        # Selection (--num-dataset-entries / --max-context-length) is therefore
        # this loader's job, exactly as it is the HF loader's.
        self._weka = WekaTraceLoader(
            filename=None,
            run=self.run,
            prompt_generator=prompt_generator,
            default_block_size=self._block_size,
        )

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Return True when the source looks like TraceLab JSONL.

        Auto-detection is content-based off the first record when the caller
        already parsed one, and single-read otherwise. Directories are rejected:
        TraceLab is distributed as a single JSONL file.
        """
        if data is not None:
            return _DETECT_KEYS.issubset(data.keys())
        if filename is None:
            return False
        path = Path(filename)
        try:
            if path.is_dir() or not path.is_file():
                return False
            if not str(path).endswith(_JSONL_SUFFIXES):
                return False
            first = cls._first_record(path)
        except Exception:
            return False
        return first is not None and _DETECT_KEYS.issubset(first.keys())

    @staticmethod
    def _first_record(path: Path) -> dict[str, Any] | None:
        from aiperf.common.utils import load_json_str

        with open_text_maybe_gzip(path) as handle:
            for line in handle:
                if line := line.strip():
                    record = load_json_str(line)
                    return record if isinstance(record, dict) else None
        return None

    def load_dataset(self) -> dict[str, list[WekaTrace]]:
        """Read the corpus, group rounds into sessions, return ``{id: [trace]}``.

        Overrides :meth:`BaseFileLoader._iter_record_dicts` indirectly by not
        using it: that helper opens the target in text mode, and the released
        corpus ships gzipped.
        """
        if self.filename is None:
            raise DatasetLoaderError(
                "TraceLab replay requires --input-file pointing at a TraceLab "
                "JSONL file (optionally gzipped)."
            )
        t0 = time.monotonic()
        sessions = self._read_sessions(self.filename)
        if not sessions:
            raise DatasetLoaderError(
                f"No TraceLab sessions found in '{self.filename}'. Every row "
                "must carry a 'session_id'."
            )

        children_by_parent: dict[str, dict[str, tuple[Any, list[dict[str, Any]]]]] = (
            defaultdict(dict)
        )
        if self._join_subagents:
            links, stats = build_join_index(
                sessions,
                min_spawn_ms=self._min_spawn_ms,
                enable_codex=self._join_codex,
            )
            children_by_parent = group_children_by_parent(sessions, links, stats)
            self.info(f"TraceLab subagent join: {stats.summary()}")

        nested_ids = {csid for kids in children_by_parent.values() for csid in kids}
        traces = self._build_traces(sessions, children_by_parent, nested_ids)
        data = self._select(traces)
        self.info(
            f"TraceLab: {len(sessions)} sessions -> {len(data)} traces "
            f"(block_size={self._block_size}) in {time.monotonic() - t0:.1f}s"
        )
        return data

    def _read_sessions(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        """Group every row of the corpus by ``session_id``, preserving file order."""
        from aiperf.common.utils import load_json_str

        sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        try:
            with open_text_maybe_gzip(path) as handle:
                for lineno, line in enumerate(handle, start=1):
                    if not (line := line.strip()):
                        continue
                    try:
                        row = load_json_str(line)
                    except ValueError as e:
                        raise DatasetLoaderError(
                            f"Invalid JSON in TraceLab file {path} at line {lineno}: {e}"
                        ) from None
                    if isinstance(row, dict) and (sid := row.get("session_id")):
                        sessions[sid].append(row)
        except (OSError, UnicodeDecodeError, EOFError, zlib.error) as e:
            raise DatasetLoaderError(f"Cannot read TraceLab file {path}: {e}") from e
        return dict(sessions)

    def _build_traces(
        self,
        sessions: dict[str, list[dict[str, Any]]],
        children_by_parent: dict[str, dict[str, tuple[Any, list[dict[str, Any]]]]],
        nested_ids: set[str],
    ) -> dict[str, WekaTrace]:
        """Convert each root session into a validated :class:`WekaTrace`."""
        traces: dict[str, WekaTrace] = {}
        actually_nested: set[str] = set()

        for sid, rows in sessions.items():
            if sid in nested_ids:
                continue
            try:
                blob = build_trace(
                    sid,
                    rows,
                    self._block_size,
                    children_by_parent.get(sid),
                    placed_sids=actually_nested,
                )
            except (ValueError, TypeError, AttributeError) as e:
                raise DatasetLoaderError(
                    f"Failed to convert TraceLab session '{sid}': {e}"
                ) from e
            if blob is None or not blob["requests"]:
                continue
            self._add_trace(traces, blob, sid)

        fallback = nested_ids - actually_nested
        if fallback:
            self.warning(
                f"TraceLab: {len(fallback)} subagent session(s) failed to nest "
                "and will be emitted as standalone traces."
            )
        for csid in sessions:
            if csid not in fallback:
                continue
            try:
                blob = build_trace(csid, sessions[csid], self._block_size)
            except (ValueError, TypeError, AttributeError) as e:
                raise DatasetLoaderError(
                    f"Failed to convert TraceLab session '{csid}': {e}"
                ) from e
            if blob is None or not blob["requests"]:
                continue
            self._add_trace(traces, blob, csid)

        return traces

    def _add_trace(
        self, traces: dict[str, WekaTrace], blob: dict[str, Any], sid: str
    ) -> None:
        """Validate and register one converted trace blob."""
        trace_id = safe_trace_id(sid)
        blob["id"] = trace_id
        if trace_id in traces:
            raise DatasetLoaderError(
                f"Duplicate TraceLab trace id '{trace_id}': session "
                f"'{sid}' collides with an earlier session."
            )
        try:
            traces[trace_id] = WekaTrace.model_validate(blob)
        except ValidationError as e:
            raise DatasetLoaderError(
                f"Converted TraceLab session '{sid}' does not satisfy the "
                f"trace schema: {e}"
            ) from e

    def _select(self, traces: dict[str, WekaTrace]) -> dict[str, list[WekaTrace]]:
        """Apply --max-context-length filtering then --num-dataset-entries cap.

        The delegate skips its own selection in delegation mode, so it happens
        here. Filtering strictly precedes the cap: capping the raw prefix first
        would silently shrink the eligible pool.
        """
        from aiperf.dataset.loader.selection import (
            filter_then_cap,
            log_selection_summary,
        )
        from aiperf.dataset.loader.weka_trace import _trace_peak_context_length

        dataset = self.run.cfg.get_default_dataset()
        entries = getattr(dataset, "entries", None)
        explicit = bool(getattr(dataset, "entries_explicit", False)) or (
            "entries" in dataset.model_fields_set and entries is not None
        )
        num_entries = entries if explicit else None
        max_ctx = getattr(dataset, "max_context_length", None)
        synthesis = getattr(dataset, "synthesis", None)
        max_osl = getattr(synthesis, "max_osl", None) if synthesis else None

        if num_entries is None and max_ctx is None:
            return {tid: [trace] for tid, trace in traces.items()}

        candidates = (
            ((tid, trace), _trace_peak_context_length(trace, max_osl=max_osl))
            for tid, trace in traces.items()
        )
        kept, stats = filter_then_cap(
            candidates,
            num_dataset_entries=num_entries,
            max_context_length=max_ctx,
        )
        log_selection_summary(
            stats,
            source=str(self.filename),
            num_dataset_entries=num_entries,
            max_context_length=max_ctx,
        )
        if not kept:
            msg = (
                f"No eligible TraceLab traces in '{self.filename}' after "
                f"filter-then-cap (scanned {stats.scanned}, "
                f"--max-context-length={max_ctx}, "
                f"--num-dataset-entries={num_entries})."
            )
            if stats.smallest_observed > 0:
                msg += (
                    f"\nSmallest trace requires {stats.smallest_observed:,} tokens; "
                    f"raise --max-context-length to at least that to admit any trace."
                )
            raise DatasetLoaderError(msg)
        return {tid: [trace] for tid, trace in kept}

    def convert_to_conversations(
        self, data: dict[str, list[WekaTrace]]
    ) -> list[Conversation]:
        """Delegate reconstruction to the Weka loader (same code path)."""
        return self._weka.convert_to_conversations(data)

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
