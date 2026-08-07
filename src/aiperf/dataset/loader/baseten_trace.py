# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseten Parquet and Arrow IPC trace replay loader."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - platform-dependent
    # pyarrow has no Windows-on-ARM wheel (apache/arrow#47195) and source-builds
    # fail there. Import lazily-tolerant so AIPerf still installs and runs without
    # it; the loader self-disables (can_load returns False, __init__ raises) when
    # pyarrow is unavailable.
    pa = None
    pc = None
    ipc = None
    pq = None

from pydantic import Field, field_validator

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode
from aiperf.common.environment import Environment
from aiperf.common.models import AIPerfBaseModel, Conversation
from aiperf.dataset.loader._baseten_replay_timemodel import reflow_idle_gaps
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_trace_loader import (
    BaseTraceDatasetLoader,
    _has_meaningful_synthesis,
)

METADATA_COLUMNS_TIME = "timestamp_start_unix_ms"
METADATA_COLUMNS_SESSION = "provided_session_id"
METADATA_COLUMNS_POOR_MAN_SESSION = "poor_man_session_id"
_VALIDATION_SAMPLE_ROWS = 10
_PARQUET_BATCH_SIZE = 128
_ARROW_IPC_SUFFIXES = frozenset({".arrow", ".ipc"})

_REQUIRED_COLUMNS = {
    METADATA_COLUMNS_TIME,
    "prompt",
    "input_tokens",
    "output_tokens",
}

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class BasetenTrace(AIPerfBaseModel):
    """Schema for Baseten completion traces exported as Parquet or Arrow IPC."""

    timestamp_start_unix_ms: NonNegativeInt = Field(
        description="Recorded request start timestamp in Unix milliseconds."
    )
    prompt: str = Field(description="Literal completion prompt sent to the server.")
    input_tokens: NonNegativeInt = Field(description="Recorded prompt token count.")
    output_tokens: NonNegativeInt = Field(
        description="Recorded completion token count."
    )
    total_hashes: list[NonNegativeInt] = Field(
        default_factory=list,
        description="Optional KV-cache block hashes aligned to block_size.",
    )
    provided_session_id: str | NonNegativeInt | None = Field(
        default=None,
        description="Session identifier exported directly from the source trace.",
    )
    poor_man_session_id: NonNegativeInt | None = Field(
        default=None,
        description="Fallback derived session identifier.",
    )
    duration_e2e_ms: NonNegativeInt | None = Field(
        default=None,
        description="Recorded end-to-end request duration in milliseconds.",
    )
    block_size: PositiveInt | None = Field(
        default=None,
        description="KV-cache block size associated with total_hashes.",
    )

    timestamp: NonNegativeInt | NonNegativeFloat | None = Field(
        default=None,
        description="Normalized timestamp in milliseconds since the first event.",
    )
    delay: NonNegativeFloat | None = Field(
        default=None,
        description="Per-turn replay delay in ms, set on continuation turns under "
        "back-pressure: turn N+1 fires this long after turn N completes.",
    )
    input_length: NonNegativeInt | None = Field(
        default=None,
        description="Alias field used by shared trace filtering logic.",
    )
    output_length: NonNegativeInt | None = Field(
        default=None,
        description="Alias field used by shared trace filtering logic.",
    )
    text_input: str | None = Field(
        default=None,
        description="Alias field used by shared trace conversation conversion logic.",
    )
    hash_ids: list[NonNegativeInt] | None = Field(
        default=None,
        description="Alias field used by per-turn request-body forwarding.",
    )
    request_body: dict[str, Any] | None = Field(
        default=None,
        description="Optional per-row payload fields to merge into the outgoing request.",
    )

    @field_validator("total_hashes", mode="before")
    @classmethod
    def _coerce_null_hashes(cls, value: Any) -> Any:
        if value is None:
            return []
        return value


def _baseten_session_key_from_schema(schema_names: set[str]) -> str | None:
    """Choose the canonical session column without scanning trace rows."""
    configured = Environment.DATASET.BASETEN_SESSION_COLUMN
    if configured in schema_names:
        return configured
    fallback = (
        METADATA_COLUMNS_POOR_MAN_SESSION
        if configured == METADATA_COLUMNS_SESSION
        else METADATA_COLUMNS_SESSION
    )
    if fallback in schema_names:
        return fallback
    return None


def _parquet_min_timestamp(parquet_file: pq.ParquetFile) -> int | None:
    """Return the minimum timestamp only when every row group has statistics.

    Returning ``None`` for any missing statistic makes the caller scan the
    timestamp column, preserving correctness when statistics are incomplete.
    """
    timestamp_index = parquet_file.schema.names.index(METADATA_COLUMNS_TIME)
    minimum: int | None = None
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(timestamp_index)
            .statistics
        )
        if statistics is None or not statistics.has_min_max:
            return None
        value = int(statistics.min)
        minimum = value if minimum is None else min(minimum, value)
    return minimum


def _is_arrow_ipc(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() in _ARROW_IPC_SUFFIXES


@contextmanager
def _open_arrow_ipc(file_path: str | Path) -> Iterator[Any]:
    """Open a memory-mapped Arrow IPC file for the lifetime of the reader."""
    source = pa.memory_map(str(file_path), "r")
    try:
        yield ipc.open_file(source)
    finally:
        source.close()


def count_baseten_records(file_path: str) -> int:
    """Return the row count for a Baseten Parquet or Arrow trace."""
    if pq is None:  # pragma: no cover - platform-dependent
        return 0
    try:
        if not _is_arrow_ipc(file_path):
            return pq.ParquetFile(file_path).metadata.num_rows
        with _open_arrow_ipc(file_path) as reader:
            return sum(
                reader.get_batch(index).num_rows
                for index in range(reader.num_record_batches)
            )
    except (OSError, pa.ArrowException):
        return 0


def count_baseten_parquet_records_and_sessions(file_path: str) -> tuple[int, int]:
    """Return row and session counts for a Baseten Parquet or Arrow trace."""
    if pq is None:  # pragma: no cover - platform-dependent
        return 0, 0
    try:
        if _is_arrow_ipc(file_path):
            with _open_arrow_ipc(file_path) as reader:
                schema_names = set(reader.schema.names)
                session_key = _baseten_session_key_from_schema(schema_names)
                row_count = 0
                session_ids: set[str | int] = set()
                null_count = 0
                for index in range(reader.num_record_batches):
                    batch = reader.get_batch(index)
                    row_count += batch.num_rows
                    if session_key is None:
                        continue
                    session_column = batch.column(
                        batch.schema.get_field_index(session_key)
                    )
                    null_count += session_column.null_count
                    session_ids.update(
                        value
                        for value in pc.unique(session_column).to_pylist()
                        if value is not None
                    )
                session_count = len(session_ids) + null_count
        else:
            parquet_file = pq.ParquetFile(file_path)
            row_count = parquet_file.metadata.num_rows
            schema_names = set(parquet_file.schema_arrow.names)
            session_key = _baseten_session_key_from_schema(schema_names)
            if session_key is None:
                return row_count, row_count
            session_ids = set()
            null_count = 0
            for batch in parquet_file.iter_batches(
                columns=[session_key],
                batch_size=_PARQUET_BATCH_SIZE,
                use_threads=True,
            ):
                session_column = batch.column(0)
                null_count += session_column.null_count
                session_ids.update(
                    value
                    for value in pc.unique(session_column).to_pylist()
                    if value is not None
                )
            session_count = len(session_ids) + null_count
    except (OSError, pa.ArrowException):
        return 0, 0

    if session_key is None:
        return row_count, row_count
    return row_count, session_count or row_count


class BasetenTraceDatasetLoader(BaseTraceDatasetLoader[BasetenTrace]):
    """Loader for Baseten completion traces exported as Parquet or Arrow IPC."""

    def __init__(self, *args, **kwargs) -> None:
        if pq is None:
            raise ValueError(
                "baseten_trace requires pyarrow, which is not installed (no "
                "Windows-on-ARM wheel is published; see apache/arrow#47195). "
                "Install pyarrow to replay Parquet or Arrow IPC traces."
            )
        super().__init__(*args, **kwargs)
        dataset = self.run.cfg.get_default_dataset()
        self._session_sample_ratio = getattr(
            dataset, "trace_session_sample_ratio", None
        )
        gap_cap_s = getattr(dataset, "max_idle_gap_cap_seconds", None)
        # Keep the cap as float ms; int-truncation would turn a sub-ms cap
        # into 0 and reflow_idle_gaps rejects non-positive caps.
        self._max_idle_gap_cap_ms = gap_cap_s * 1000 if gap_cap_s is not None else None
        self._delay_cap = DelayCapTracker(
            cap_seconds=getattr(dataset, "inter_turn_delay_cap_seconds", None)
        )
        self._speedup = getattr(dataset, "replay_speedup", None) or 1.0
        # Reject unsupported synthesis on every config path (YAML,
        # auto-detected type). Speedup compounds with replay_speedup and
        # bypasses the think-time subtraction in back-pressure.
        synthesis_speedup = getattr(self._synthesis, "speedup_ratio", 1.0)
        if synthesis_speedup != 1.0:
            raise ValueError(
                f"synthesis speedup_ratio={synthesis_speedup} is not supported "
                "by the baseten_trace loader; use --replay-speedup for "
                "wall-clock compression."
            )
        # Prompt-shaping multipliers reshape hash_ids while the wire still
        # sends the original recorded prompt, desyncing the KV hints.
        for attr, default in (
            ("prefix_len_multiplier", 1.0),
            ("prefix_root_multiplier", 1),
            ("prompt_len_multiplier", 1.0),
        ):
            value = getattr(self._synthesis, attr, default)
            if value != default:
                raise ValueError(
                    f"synthesis {attr}={value} is not supported by the "
                    "baseten_trace loader: it replays recorded prompts "
                    "verbatim, so hash-reshaping synthesis cannot change the "
                    "sent prompt and would desync the forwarded hash_ids KV "
                    "hints."
                )
        self._open_loop = getattr(dataset, "open_loop_replay", True)
        self._open_loop_strict = getattr(dataset, "open_loop_strict", False)
        self._omit_kv_hints = getattr(dataset, "omit_kv_hints", False)
        self._force_min_tokens = getattr(dataset, "force_min_tokens", True)
        # Mirror of the CLI converter's collision guard
        # (_reject_baseten_trace_extra_input_collisions), which only sees
        # explicit --custom-dataset-type baseten_trace; this catches
        # auto-detected Parquet traces too.
        self._reject_extra_input_collisions()
        self._rng = rng.derive("dataset.loader.baseten_trace.session_sampling")
        self._floored_zero_osl = 0
        # Session key used to filter rows during sampling; grouping must reuse
        # it because re-scoring the filtered subset can flip to the other
        # column and silently shred the sessions sampling kept whole.
        self._sampled_session_key: str | None = None
        self._schema_names: set[str] | None = None
        self._parquet_file: Any | None = None

    def _reject_extra_input_collisions(self) -> None:
        """Reject endpoint extra-inputs keys this loader injects per-turn.

        Loader-injected per-turn values (``min_tokens`` from the recorded
        output length, ``hash_ids``/``block_size`` KV hints) overwrite
        endpoint-level extras, so the user's value would be silently clobbered
        on the wire. Each collision has an opt-out flag that stops the
        injection so the user value goes through. ``max_tokens`` is not
        guarded: user extras win over the loader for that key.
        """
        extra = self.run.cfg.endpoint.extra or {}
        collisions: list[tuple[str, str]] = []
        if self._force_min_tokens and "min_tokens" in extra:
            collisions.append(("min_tokens", "--no-force-min-tokens"))
        if not self._omit_kv_hints:
            collisions.extend(
                (key, "--omit-kv-hints")
                for key in ("hash_ids", "block_size")
                if key in extra
            )
        if collisions:
            raise ValueError(
                "; ".join(
                    f"--extra-inputs {key} is overwritten per-turn by the "
                    f"baseten_trace loader; pass {flag} to send your value instead"
                    for key, flag in collisions
                )
            )

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        if pq is None:  # pragma: no cover - platform-dependent
            return False
        if filename is None:
            return False

        suffix = Path(filename).suffix.lower()
        if suffix not in {".parquet", *_ARROW_IPC_SUFFIXES}:
            return False

        if data is not None:
            return _REQUIRED_COLUMNS.issubset(data.keys())

        try:
            if suffix == ".parquet":
                schema_names = set(pq.read_schema(filename).names)
            else:
                with _open_arrow_ipc(filename) as reader:
                    schema_names = set(reader.schema.names)
        except (FileNotFoundError, OSError, pa.ArrowException):
            return False

        return _REQUIRED_COLUMNS.issubset(schema_names)

    def _parse_trace(self, record: dict) -> BasetenTrace:
        raise NotImplementedError(
            "BasetenTraceDatasetLoader reads columnar files, not JSONL."
        )

    def _preprocess_trace(self, trace: BasetenTrace) -> None:
        trace.total_hashes = list(trace.total_hashes or [])
        trace.timestamp = int(trace.timestamp_start_unix_ms)
        trace.input_length = int(trace.input_tokens)
        # Real traces contain canceled requests with output_tokens=0, but
        # Turn.max_tokens requires >= 1.
        trace.output_length = max(1, int(trace.output_tokens))
        trace.text_input = trace.prompt
        trace.hash_ids = list(trace.total_hashes)

    def _set_request_body(self, trace: BasetenTrace) -> None:
        if trace.hash_ids is None:
            trace.hash_ids = list(trace.total_hashes or [])
        trace.request_body = {}
        if self._force_min_tokens:
            trace.request_body["min_tokens"] = trace.output_length
        # KV-cache-aware routing hints. Inert when there is no routing choice
        # (single replica per role); some strict frontends reject unknown body
        # params with HTTP 400. Opt out via omit_kv_hints for frontends that do
        # not tolerate extra params.
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

        session_key = self._sampled_session_key
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

    def _source_schema_names(self) -> set[str]:
        if self._schema_names is None:
            if _is_arrow_ipc(self.filename):
                with _open_arrow_ipc(self.filename) as reader:
                    self._schema_names = set(reader.schema.names)
            else:
                assert self._parquet_file is not None
                self._schema_names = set(self._parquet_file.schema_arrow.names)
        return self._schema_names

    def _iter_source_batches(self, columns: list[str]) -> Iterator[pa.RecordBatch]:
        """Yield projected batches from Parquet or memory-mapped Arrow IPC."""
        if _is_arrow_ipc(self.filename):
            with _open_arrow_ipc(self.filename) as reader:
                for index in range(reader.num_record_batches):
                    yield reader.get_batch(index).select(columns)
            return

        assert self._parquet_file is not None
        yield from self._parquet_file.iter_batches(
            columns=columns,
            batch_size=_PARQUET_BATCH_SIZE,
            use_threads=True,
        )

    def _minimum_timestamp(self) -> int | None:
        if not _is_arrow_ipc(self.filename):
            assert self._parquet_file is not None
            minimum = _parquet_min_timestamp(self._parquet_file)
            if minimum is not None:
                return minimum
        minimum = None
        for batch in self._iter_source_batches([METADATA_COLUMNS_TIME]):
            batch_minimum = pc.min(batch.column(0)).as_py()
            if batch_minimum is not None:
                minimum = (
                    int(batch_minimum)
                    if minimum is None
                    else min(minimum, int(batch_minimum))
                )
        return minimum

    def _sample_session_ids(
        self,
    ) -> tuple[int | None, str | None, set[str | int] | None, set[int] | None]:
        session_key = _baseten_session_key_from_schema(self._source_schema_names())
        self._sampled_session_key = session_key
        if self._session_sample_ratio is None or self._session_sample_ratio >= 1.0:
            return self._minimum_timestamp(), session_key, None, None

        min_timestamp: int | None = None
        session_first_ts: dict[str | int, int] = {}
        null_row_count = 0

        columns = [METADATA_COLUMNS_TIME]
        if session_key is not None:
            columns.append(session_key)
        for batch in self._iter_source_batches(columns):
            for row in batch.to_pylist():
                timestamp = int(row[METADATA_COLUMNS_TIME])
                min_timestamp = (
                    timestamp
                    if min_timestamp is None
                    else min(min_timestamp, timestamp)
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

        if min_timestamp is None:
            return None, session_key, None, None

        if session_key is None:
            self.warning(
                "trace_session_sample_ratio requested, but neither provided_session_id "
                "nor poor_man_session_id exists; skipping sampling."
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

        if not _is_arrow_ipc(self.filename):
            self._parquet_file = pq.ParquetFile(self.filename)
            self._schema_names = set(self._parquet_file.schema_arrow.names)
        try:
            items = self._read_traces(*self._sample_session_ids())
        finally:
            if self._parquet_file is not None:
                self._parquet_file.close()
                self._parquet_file = None

        # Closed-loop replay defers the gap cap to convert_to_conversations so
        # think-time delays derive from the recorded timestamps, not reflowed ones.
        if self._open_loop and self._max_idle_gap_cap_ms is not None:
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

    def _read_traces(
        self,
        min_timestamp: int | None,
        session_key: str | None,
        session_ids: set[str | int] | None,
        null_rows: set[int] | None,
    ) -> list[BasetenTrace]:
        """Read, sample, normalize, and filter trace rows from the source file."""
        sampling = session_key is not None and session_ids is not None
        projected_columns = self._projected_trace_columns(session_key)

        items: list[BasetenTrace] = []
        null_ordinal = 0
        validated_rows = 0
        for row in self._iter_trace_rows(
            projected_columns,
            session_key if sampling else None,
            session_ids,
        ):
            if sampling and row.get(session_key) is None:
                kept = null_ordinal in null_rows
                null_ordinal += 1
                if not kept:
                    continue

            if validated_rows < _VALIDATION_SAMPLE_ROWS or self._row_needs_validation(
                row
            ):
                trace = BasetenTrace.model_validate(row)
                validated_rows += 1
            else:
                trace = BasetenTrace.model_construct(**row)
            self._preprocess_trace(trace)
            normalized = min_timestamp is not None and trace.timestamp is not None
            if normalized:
                trace.timestamp = int(trace.timestamp) - int(min_timestamp)

            # Filter on normalized-but-uncompressed ms so the offset window
            # selects recorded time regardless of replay_speedup.
            if not self._filter_and_cap_trace(trace):
                continue

            if normalized and self._speedup != 1.0:
                # Compress wall-clock once here; gap-cap + back-pressure delays
                # downstream inherit the compressed times. Never touches hash_ids.
                trace.timestamp = trace.timestamp / self._speedup

            # Count after filtering so skipped rows do not inflate the summary.
            if trace.output_tokens == 0:
                self._floored_zero_osl += 1
            items.append(trace)
        return items

    @staticmethod
    def _row_needs_validation(row: dict[str, Any]) -> bool:
        """Route malformed required fields through Pydantic's clear errors."""
        if not isinstance(row.get("prompt"), str):
            return True
        return any(
            not isinstance(value := row.get(field), int) or value < 0
            for field in (
                METADATA_COLUMNS_TIME,
                "input_tokens",
                "output_tokens",
            )
        )

    def _projected_trace_columns(self, session_key: str | None) -> list[str]:
        """Return source columns needed by the configured replay behavior."""
        schema_names = self._source_schema_names()
        columns = {
            METADATA_COLUMNS_TIME,
            "prompt",
            "input_tokens",
            "output_tokens",
        }
        if session_key is not None:
            columns.add(session_key)
        if not self._omit_kv_hints:
            columns.update(("total_hashes", "block_size"))
        if not self._open_loop:
            columns.add("duration_e2e_ms")
        return sorted(columns & schema_names)

    def _iter_trace_rows(
        self,
        projected_columns: list[str],
        session_key: str | None,
        session_ids: set[str | int] | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield projected rows after applying sampling in bounded Arrow batches."""
        sampled_ids: pa.Array | None = None
        for batch in self._iter_source_batches(projected_columns):
            if session_key is not None and session_ids is not None:
                if sampled_ids is None:
                    sampled_ids = pa.array(
                        sorted(session_ids),
                        type=batch.schema.field(session_key).type,
                    )
                session_column = batch.column(batch.schema.get_field_index(session_key))
                selected = pc.is_in(session_column, value_set=sampled_ids)
                selected = pc.or_(selected, pc.is_null(session_column))
                if _is_arrow_ipc(self.filename):
                    columns = tuple(zip(batch.schema.names, batch.columns, strict=True))
                    selected_rows = pc.indices_nonzero(selected).to_pylist()
                    for row_index in selected_rows:
                        yield {
                            name: column[row_index].as_py() for name, column in columns
                        }
                    continue
                batch = batch.filter(selected)
            yield from batch.to_pylist()

    def _apply_idle_gap_cap(self, items: list[BasetenTrace]) -> None:
        """Collapse global idle gaps so a sparse (sampled) trace does not idle
        through long dead-air stretches under fixed-schedule replay.

        Operates on the normalized timestamps of every trace that still carries
        one: all rows in open-loop replay (before grouping), only session-start
        turns in closed-loop replay (after back-pressure clears continuation
        timestamps). Pure timing rewrite — does not touch hash_ids/prompt, so
        KV-cache fidelity is preserved.
        """
        timed = [trace for trace in items if trace.timestamp is not None]
        if not timed:
            return
        reflowed = reflow_idle_gaps(
            [trace.timestamp for trace in timed], self._max_idle_gap_cap_ms
        )
        for trace, new_ts in zip(timed, reflowed, strict=True):
            trace.timestamp = new_ts

    def convert_to_conversations(
        self, data: dict[str, list[BasetenTrace]]
    ) -> list[Conversation]:
        """Apply session back-pressure, then build conversations.

        For multi-turn sessions, continuation turns (index > 0) replay
        closed-loop — turn N+1 fires only after turn N completes — rather than at
        an absolute pre-recorded time. This keeps each session's prefix cached in
        order (faithful KV reuse) and is the correct model for heavily
        multi-turn traces. Turn 0 keeps its absolute arrival time (session start).
        Open-loop replay skips back-pressure entirely: every turn
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
            # Back-pressure first so think-time delays derive from the RECORDED
            # start-to-start gaps; the idle-gap cap then reflows only the
            # remaining absolute session-start timestamps.
            self._apply_back_pressure(data)
            if self._max_idle_gap_cap_ms is not None:
                self._apply_idle_gap_cap(
                    [trace for traces in data.values() for trace in traces]
                )
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

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset(
            {
                "duration_e2e_ms",
                "block_size",
                "provided_session_id",
                "poor_man_session_id",
                "request_body",
                "prompt",
                "text_input",
                "total_hashes",
                # The wire sends the recorded prompt verbatim, so the KV hints
                # must stay the recorded ones even under output-len synthesis.
                # input_length is deliberately NOT excluded: the synthesizer
                # would reset a missing value to its default block_size.
                "hash_ids",
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
            result.append(BasetenTrace.model_validate(merged))
        return result
