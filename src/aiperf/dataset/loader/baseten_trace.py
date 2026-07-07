# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseten Parquet trace replay loader."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation
from aiperf.dataset.loader._baseten_replay_timemodel import reflow_idle_gaps
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_trace_loader import (
    BaseTraceDatasetLoader,
    _has_meaningful_synthesis,
)
from aiperf.dataset.loader.baseten_trace_models import (
    METADATA_COLUMNS,
    METADATA_COLUMNS_POOR_MAN_SESSION,
    METADATA_COLUMNS_SESSION,
    METADATA_COLUMNS_TIME,
    REQUIRED_COLUMNS,
    BasetenTrace,
    choose_baseten_session_key,
)

__all__ = [
    "BasetenTrace",
    "BasetenTraceDatasetLoader",
    "choose_baseten_session_key",
    "count_baseten_parquet_records_and_sessions",
]

_SESSION_KEY_PROBE_ROWS = 10_000


def count_baseten_parquet_records_and_sessions(file_path: str) -> tuple[int, int]:
    """Return row and session counts for a Baseten Parquet trace file."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        schema_names = set(pq.read_schema(file_path).names)
        session_columns = {
            METADATA_COLUMNS_SESSION,
            METADATA_COLUMNS_POOR_MAN_SESSION,
        } & schema_names
        if not session_columns:
            return row_count, row_count

        table = pq.read_table(file_path, columns=sorted(session_columns))
    except (OSError, pa.ArrowException):
        return 0, 0

    rows = table.to_pylist()
    session_key = choose_baseten_session_key(
        [row.get(METADATA_COLUMNS_SESSION) for row in rows],
        [row.get(METADATA_COLUMNS_POOR_MAN_SESSION) for row in rows],
    )
    if session_key is None:
        return row_count, row_count

    values = {str(row[session_key]) for row in rows if row.get(session_key) is not None}
    return row_count, len(values) if values else row_count


class BasetenTraceDatasetLoader(BaseTraceDatasetLoader[BasetenTrace]):
    """Loader for Baseten completion traces exported as Parquet."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dataset = self.run.cfg.get_default_dataset()
        self._session_sample_ratio = getattr(
            dataset, "trace_session_sample_ratio", None
        )
        gap_cap_s = getattr(dataset, "max_idle_gap_cap_seconds", None)
        self._max_idle_gap_cap_ms = (
            int(gap_cap_s * 1000) if gap_cap_s is not None else None
        )
        self._delay_cap = DelayCapTracker(
            cap_seconds=getattr(dataset, "inter_turn_delay_cap_seconds", None)
        )
        self._speedup = getattr(dataset, "replay_speedup", None) or 1.0
        self._open_loop = getattr(dataset, "open_loop_replay", False)
        self._open_loop_strict = getattr(dataset, "open_loop_strict", False)
        self._omit_kv_hints = getattr(dataset, "omit_kv_hints", False)
        self._force_min_tokens = getattr(dataset, "force_min_tokens", True)
        self._rng = rng.derive("dataset.loader.baseten_trace.session_sampling")
        self._floored_zero_osl = 0

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        if filename is None or Path(filename).suffix.lower() != ".parquet":
            return False

        if data is not None:
            return REQUIRED_COLUMNS.issubset(data.keys())

        try:
            schema = pq.read_schema(filename)
        except (FileNotFoundError, OSError, pa.ArrowException):
            return False

        return REQUIRED_COLUMNS.issubset(schema.names)

    def _parse_trace(self, record: dict) -> BasetenTrace:
        raise NotImplementedError("BasetenTraceDatasetLoader reads Parquet, not JSONL.")

    def _preprocess_trace(self, trace: BasetenTrace) -> None:
        trace.timestamp = int(trace.timestamp_start_unix_ms)
        trace.input_length = int(trace.input_tokens)
        # Real traces contain canceled requests with output_tokens=0, but
        # Turn.max_tokens requires >= 1.
        trace.output_length = max(1, int(trace.output_tokens))
        if trace.output_tokens == 0:
            self._floored_zero_osl += 1
        trace.text_input = trace.prompt
        trace.hash_ids = list(trace.total_hashes or [])

    def _set_request_body(self, trace: BasetenTrace) -> None:
        if trace.hash_ids is None:
            trace.hash_ids = list(trace.total_hashes or [])
        trace.request_body = {}
        if self._force_min_tokens:
            trace.request_body["min_tokens"] = trace.output_length
        # KV-cache-aware routing hints. Inert at 1P1D (no routing choice); some
        # strict frontends (Dynamo v1.2) 400 on these unknown params. Opt out via
        # omit_kv_hints to keep request bodies identical across legs whose
        # frontends differ in param tolerance.
        if not self._omit_kv_hints:
            if trace.hash_ids:
                trace.request_body["hash_ids"] = trace.hash_ids
            if trace.block_size is not None:
                trace.request_body["block_size"] = trace.block_size

    def _infer_context_mode(
        self, traces: list[BasetenTrace]
    ) -> ConversationContextMode | None:
        if len(traces) > 1:
            return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        return None

    def _group_traces(self, items: list[BasetenTrace]) -> dict[str, list[BasetenTrace]]:
        if not items:
            return {}

        session_key = self._choose_session_key(items)
        if session_key is None:
            self.info(
                "No repeated Baseten trace session key found; generating session IDs."
            )
        else:
            self.info(f"Using Baseten trace session key: {session_key}")

        groups: dict[str, list[BasetenTrace]] = defaultdict(list)
        for trace in items:
            if (
                session_key == METADATA_COLUMNS_SESSION
                and trace.provided_session_id is not None
            ):
                session_id = str(trace.provided_session_id)
            elif (
                session_key == METADATA_COLUMNS_POOR_MAN_SESSION
                and trace.poor_man_session_id is not None
            ):
                session_id = str(trace.poor_man_session_id)
            else:
                session_id = self.session_id_generator.next()
            groups[session_id].append(trace)

        for traces in groups.values():
            traces.sort(key=lambda trace: int(trace.timestamp or 0))

        return self._order_groups(groups)

    def _order_groups(
        self, groups: dict[str, list[BasetenTrace]]
    ) -> dict[str, list[BasetenTrace]]:
        session_entries = [
            (
                min(int(trace.timestamp or 0) for trace in traces),
                session_id,
                traces,
            )
            for session_id, traces in groups.items()
            if traces
        ]

        session_entries.sort(key=lambda item: (item[0], item[1]))
        return {session_id: traces for _, session_id, traces in session_entries}

    def _read_metadata_table(self) -> pa.Table:
        schema = pq.read_schema(self.filename)
        metadata_columns = [
            column for column in METADATA_COLUMNS if column in set(schema.names)
        ]
        return pq.read_table(self.filename, columns=metadata_columns)

    def _choose_session_key_from_metadata_rows(
        self, rows: list[dict[str, Any]]
    ) -> str | None:
        return choose_baseten_session_key(
            [row.get(METADATA_COLUMNS_SESSION) for row in rows],
            [row.get(METADATA_COLUMNS_POOR_MAN_SESSION) for row in rows],
        )

    def _sample_session_ids(
        self,
    ) -> tuple[int | None, str | None, set[str | int] | None, set[int] | None]:
        metadata_table = self._read_metadata_table()

        if metadata_table.num_rows == 0:
            return None, None, None, None

        probe_size = min(_SESSION_KEY_PROBE_ROWS, metadata_table.num_rows)
        probe_rows = metadata_table.slice(offset=0, length=probe_size).to_pylist()
        session_key = self._choose_session_key_from_metadata_rows(probe_rows)
        metadata_rows = metadata_table.to_pylist()

        min_timestamp: int | None = None
        session_first_ts: dict[str | int, int] = {}
        null_row_count = 0

        for row in metadata_rows:
            timestamp = int(row[METADATA_COLUMNS_TIME])
            min_timestamp = (
                timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            )

            if session_key is None:
                continue

            session_id = row.get(session_key)
            if session_id is None:
                null_row_count += 1
                continue

            session_first_ts[session_id] = min(
                timestamp, session_first_ts.get(session_id, timestamp)
            )

        if self._session_sample_ratio is None or self._session_sample_ratio >= 1.0:
            return min_timestamp, session_key, None, None

        if session_key is None:
            self.warning(
                "trace_session_sample_ratio requested, but neither provided_session_id "
                "nor poor_man_session_id forms multi-row sessions; skipping sampling."
            )
            return min_timestamp, None, None, None

        session_entries = sorted(
            ((ts, sid) for sid, ts in session_first_ts.items()),
            key=lambda item: (item[0], str(item[1])),
        )
        original_count = len(session_entries)
        sampled_entries = [
            entry
            for entry in session_entries
            if self._rng.uniform(0.0, 1.0) < self._session_sample_ratio
        ]

        if not sampled_entries and original_count > 0:
            sampled_entries = [self._rng.choice(session_entries)]

        # Null-session rows become synthesized single-turn sessions downstream, so
        # sample each (keyed by stable file row order) at the same ratio instead
        # of letting the pyarrow "in" filter drop them all.
        sampled_null_rows = {
            ordinal
            for ordinal in range(null_row_count)
            if self._rng.uniform(0.0, 1.0) < self._session_sample_ratio
        }

        self.info(
            f"Sampled {len(sampled_entries):,} of {original_count:,} sessions and "
            f"{len(sampled_null_rows):,} of {null_row_count:,} null-session rows "
            f"using {session_key} with "
            f"trace_session_sample_ratio={self._session_sample_ratio}"
        )
        return (
            min_timestamp,
            session_key,
            {session_id for _, session_id in sampled_entries},
            sampled_null_rows,
        )

    def load_dataset(self) -> dict[str, list[BasetenTrace]]:
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        self._floored_zero_osl = 0

        min_timestamp, sampled_session_key, sampled_session_ids, sampled_null_rows = (
            self._sample_session_ids()
        )
        table_kwargs: dict[str, Any] = {}
        sampling = sampled_session_key is not None and sampled_session_ids is not None
        if sampling:
            table_kwargs["filters"] = (
                pc.field(sampled_session_key).isin(sorted(sampled_session_ids))
                | pc.field(sampled_session_key).is_null()
            )

        table = pq.read_table(self.filename, **table_kwargs)
        items: list[BasetenTrace] = []
        null_ordinal = 0

        for row in table.to_pylist():
            if sampling and row.get(sampled_session_key) is None:
                kept = null_ordinal in sampled_null_rows
                null_ordinal += 1
                if not kept:
                    continue

            if "__version__" in row and "dataset_version" not in row:
                row["dataset_version"] = row.pop("__version__")

            trace = BasetenTrace.model_validate(row)
            self._preprocess_trace(trace)
            if min_timestamp is not None and trace.timestamp is not None:
                trace.timestamp = int(trace.timestamp) - int(min_timestamp)
                if self._speedup != 1.0:
                    # Compress wall-clock once here; gap-cap + back-pressure delays
                    # downstream inherit the compressed times. Never touches hash_ids.
                    trace.timestamp = trace.timestamp / self._speedup

            if not self._filter_and_cap_trace(trace):
                continue

            items.append(trace)

        if self._max_idle_gap_cap_ms is not None:
            self._apply_idle_gap_cap(items)

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} Baseten traces "
                f"across {len(data):,} sessions from {self.filename}"
            )
        )

        if _has_meaningful_synthesis(self._synthesis):
            data = self._apply_synthesis(data)

        data = self._cap_grouped_traces_max_osl(data)
        for traces in data.values():
            for trace in traces:
                self._set_request_body(trace)

        self._log_filtering_summary()
        if self._floored_zero_osl > 0:
            self.info(
                f"Floored {self._floored_zero_osl:,} traces with output_tokens=0 "
                f"(e.g. canceled requests) to an output_length of 1"
            )
        return data

    def _apply_idle_gap_cap(self, items: list[BasetenTrace]) -> None:
        """Collapse global idle gaps so a sparse (sampled) trace does not idle
        through long dead-air stretches under fixed-schedule replay.

        Operates on the normalized per-row timestamps across ALL sessions, before
        grouping, so the global schedule stays monotonic. Pure timing rewrite —
        does not touch hash_ids/prompt, so KV-cache fidelity is preserved.
        """
        if not items:
            return
        original = [int(trace.timestamp or 0) for trace in items]
        reflowed = reflow_idle_gaps(original, self._max_idle_gap_cap_ms)
        for trace, new_ts in zip(items, reflowed, strict=True):
            trace.timestamp = new_ts

    def convert_to_conversations(
        self, data: dict[str, list[BasetenTrace]]
    ) -> list[Conversation]:
        """Apply session back-pressure, then build conversations.

        For multi-turn sessions, continuation turns (index > 0) replay
        closed-loop — turn N+1 fires only after turn N completes — rather than at
        an absolute pre-recorded time. This keeps each session's prefix cached in
        order (faithful KV reuse) and is the correct model for this trace (~93%
        multi-turn). Turn 0 keeps its absolute arrival time (session start).
        Open-loop ('no-mercy') replay skips back-pressure entirely: every turn
        keeps its absolute (speedup-scaled) timestamp and fires on the schedule.
        With ``open_loop_strict`` additionally set, sessions are exploded so
        every trace row becomes its own independent single-turn conversation:
        no session grouping, no multi-turn context mode, every request fires at
        its absolute recorded time.
        """
        if self._open_loop and self._open_loop_strict:
            data = {
                f"{session_id}#{index}": [trace]
                for session_id, traces in data.items()
                for index, trace in enumerate(traces)
            }
        elif not self._open_loop:
            self._apply_back_pressure(data)
        conversations = super().convert_to_conversations(data)
        self._delay_cap.log_summary(logger_name=__name__)
        return conversations

    def _apply_back_pressure(self, data: dict[str, list[BasetenTrace]]) -> None:
        """Convert continuation turns from absolute timestamps to inter-turn
        delays (clamped by ``inter_turn_delay_cap_seconds``). Clearing the
        absolute timestamp makes the timing strategy take its delay branch.

        The recorded start-to-start gap already includes the prior turn's
        service time, and fixed_schedule applies ``delay`` AFTER the prior turn
        completes in replay — so subtract the prior turn's recorded end-to-end
        duration to avoid double-counting server time (replay inter-arrival
        would otherwise be replay_service + recorded_service + think). Fall back
        to the raw gap when duration_e2e_ms is absent. duration_e2e_ms is not
        speedup-scaled, so divide it to match the already-scaled timestamps."""
        for traces in data.values():
            ordered = sorted(traces, key=lambda t: int(t.timestamp or 0))
            prev_ts: int | None = None
            prev_e2e_ms: float = 0.0
            for i, trace in enumerate(ordered):
                ts = int(trace.timestamp or 0)
                if i == 0:
                    prev_ts = ts
                    prev_e2e_ms = float(trace.duration_e2e_ms or 0) / self._speedup
                    continue
                gap = float(max(0, ts - prev_ts))
                trace.delay = self._delay_cap.clamp(max(0.0, gap - prev_e2e_ms))
                trace.timestamp = None
                prev_ts = ts
                prev_e2e_ms = float(trace.duration_e2e_ms or 0) / self._speedup

    def _choose_session_key(self, items: list[BasetenTrace]) -> str | None:
        return choose_baseten_session_key(
            [trace.provided_session_id for trace in items],
            [trace.poor_man_session_id for trace in items],
        )

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset(
            {
                "duration_e2e_ms",
                "duration_ttft_ms",
                "request_canceled",
                "cached_tokens_reference",
                "model_name",
                "org_id",
                "block_size",
                "features",
                "speculation_ratio",
                "output_text",
                "dataset_version",
                "total_hashes_len",
                "provided_session_id",
                "poor_man_session_id",
                "request_body",
                "prompt",
                "text_input",
                "total_hashes",
            }
        )

    def _reconstruct_traces(
        self, originals: list[BasetenTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[BasetenTrace]:
        result: list[BasetenTrace] = []
        for i, synth_dict in enumerate(synth_dicts):
            original = originals[i] if i < len(originals) else originals[-1]
            merged = original.model_dump()
            merged.update(synth_dict)
            trace = BasetenTrace.model_validate(merged)
            self._set_request_body(trace)
            result.append(trace)
        return result
