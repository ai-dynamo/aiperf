# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseten Parquet trace replay loader."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.base_trace_loader import (
    BaseTraceDatasetLoader,
    _has_meaningful_synthesis,
)
from aiperf.dataset.loader.models import BasetenTrace

_METADATA_COLUMNS_TIME = "timestamp_start_unix_ms"
_METADATA_COLUMNS_SESSION = "provided_session_id"
_METADATA_COLUMNS_POOR_MAN_SESSION = "poor_man_session_id"
_METADATA_COLUMNS = {
    _METADATA_COLUMNS_TIME,
    _METADATA_COLUMNS_SESSION,
    _METADATA_COLUMNS_POOR_MAN_SESSION,
}

_REQUIRED_COLUMNS = {
    _METADATA_COLUMNS_TIME,
    "prompt",
    "input_tokens",
    "output_tokens",
}
_SESSION_KEY_PROBE_ROWS = 10_000


def _score_session_groups(
    session_ids: list[str | int | None],
) -> tuple[int, int]:
    counts = Counter(session_id for session_id in session_ids if session_id is not None)
    repeated_group_sizes = [count for count in counts.values() if count > 1]
    return (sum(repeated_group_sizes), len(repeated_group_sizes))


def _choose_repeated_session_key(
    provided_session_ids: list[str | int | None],
    poor_man_session_ids: list[int | None],
) -> str | None:
    provided_score = _score_session_groups(provided_session_ids)
    poor_score = _score_session_groups(poor_man_session_ids)

    if provided_score > poor_score and provided_score[0] > 0:
        return _METADATA_COLUMNS_SESSION
    if poor_score > provided_score and poor_score[0] > 0:
        return _METADATA_COLUMNS_POOR_MAN_SESSION
    if provided_score == poor_score and provided_score[0] > 0:
        return _METADATA_COLUMNS_SESSION
    return None


class BasetenTraceDatasetLoader(BaseTraceDatasetLoader[BasetenTrace]):
    """Loader for Baseten completion traces exported as Parquet."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dataset = self.run.cfg.get_default_dataset()
        self._session_sample_ratio = getattr(
            dataset, "trace_session_sample_ratio", None
        )
        self._rng = rng.derive("dataset.loader.baseten_trace.session_sampling")

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        if filename is None or Path(filename).suffix.lower() != ".parquet":
            return False

        if data is not None:
            return _REQUIRED_COLUMNS.issubset(data.keys())

        try:
            schema = pq.read_schema(filename)
        except (FileNotFoundError, OSError, pa.ArrowException):
            return False

        return _REQUIRED_COLUMNS.issubset(schema.names)

    def _parse_trace(self, record: dict) -> BasetenTrace:
        raise NotImplementedError("BasetenTraceDatasetLoader reads Parquet, not JSONL.")

    def _preprocess_trace(self, trace: BasetenTrace) -> None:
        trace.timestamp = int(trace.timestamp_start_unix_ms)
        trace.input_length = int(trace.input_tokens)
        trace.output_length = int(trace.output_tokens)
        trace.text_input = trace.prompt
        trace.hash_ids = list(trace.total_hashes or [])

    def _set_request_body(self, trace: BasetenTrace) -> None:
        if trace.hash_ids is None:
            trace.hash_ids = list(trace.total_hashes or [])
        trace.request_body = {"min_tokens": trace.output_length}
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
            self.info("No repeated Baseten trace session key found; generating session IDs.")
        else:
            self.info(f"Using Baseten trace session key: {session_key}")

        groups: dict[str, list[BasetenTrace]] = defaultdict(list)
        for trace in items:
            if (
                session_key == _METADATA_COLUMNS_SESSION
                and trace.provided_session_id is not None
            ):
                session_id = str(trace.provided_session_id)
            elif (
                session_key == _METADATA_COLUMNS_POOR_MAN_SESSION
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
            column for column in _METADATA_COLUMNS if column in set(schema.names)
        ]
        return pq.read_table(self.filename, columns=metadata_columns)

    def _choose_session_key_from_metadata_rows(
        self, rows: list[dict[str, Any]]
    ) -> str | None:
        return _choose_repeated_session_key(
            [row.get(_METADATA_COLUMNS_SESSION) for row in rows],
            [row.get(_METADATA_COLUMNS_POOR_MAN_SESSION) for row in rows],
        )

    def _sample_session_ids(
        self,
    ) -> tuple[int | None, str | None, set[str | int] | None]:
        metadata_table = self._read_metadata_table()

        if metadata_table.num_rows == 0:
            return None, None, None

        probe_size = min(_SESSION_KEY_PROBE_ROWS, metadata_table.num_rows)
        probe_rows = metadata_table.slice(offset=0, length=probe_size).to_pylist()
        session_key = self._choose_session_key_from_metadata_rows(probe_rows)
        metadata_rows = metadata_table.to_pylist()

        min_timestamp: int | None = None
        session_first_ts: dict[str | int, int] = {}

        for row in metadata_rows:
            timestamp = int(row[_METADATA_COLUMNS_TIME])
            min_timestamp = (
                timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            )

            if session_key is None:
                continue

            session_id = row.get(session_key)
            if session_id is None:
                continue

            session_first_ts[session_id] = min(
                timestamp,
                session_first_ts.get(session_id, timestamp),
            )

        if self._session_sample_ratio is None or self._session_sample_ratio >= 1.0:
            return min_timestamp, session_key, None

        if session_key is None:
            self.warning(
                "trace_session_sample_ratio requested, but neither provided_session_id "
                "nor poor_man_session_id forms multi-row sessions; skipping sampling."
            )
            return min_timestamp, None, None

        session_entries = sorted(
            (
                (first_ts, session_id)
                for session_id, first_ts in session_first_ts.items()
            ),
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

        self.info(
            f"Sampled {len(sampled_entries):,} of {original_count:,} sessions using "
            f"{session_key} "
            f"with trace_session_sample_ratio={self._session_sample_ratio}"
        )
        return (
            min_timestamp,
            session_key,
            {session_id for _, session_id in sampled_entries},
        )

    def load_dataset(self) -> dict[str, list[BasetenTrace]]:
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0

        min_timestamp, sampled_session_key, sampled_session_ids = (
            self._sample_session_ids()
        )
        table_kwargs: dict[str, Any] = {}
        if sampled_session_key and sampled_session_ids:
            table_kwargs["filters"] = [
                (sampled_session_key, "in", sorted(sampled_session_ids))
            ]

        table = pq.read_table(self.filename, **table_kwargs)
        items: list[BasetenTrace] = []

        for row in table.to_pylist():
            if "__version__" in row and "dataset_version" not in row:
                row["dataset_version"] = row.pop("__version__")

            trace = BasetenTrace.model_validate(row)
            self._preprocess_trace(trace)
            if min_timestamp is not None and trace.timestamp is not None:
                trace.timestamp = int(trace.timestamp) - int(min_timestamp)

            if not self._filter_and_cap_trace(trace):
                continue

            self._set_request_body(trace)
            items.append(trace)

        self._log_filtering_summary()
        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} Baseten traces "
                f"across {len(data):,} sessions from {self.filename}"
            )
        )

        if _has_meaningful_synthesis(self._synthesis):
            data = self._apply_synthesis(data)

        return data

    def _choose_session_key(self, items: list[BasetenTrace]) -> str | None:
        return _choose_repeated_session_key(
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
