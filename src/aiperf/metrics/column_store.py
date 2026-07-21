# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-indexed NaN-sparse columnar storage for per-record metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.metrics._column_store_handlers import (
    make_list_handler as _make_list_handler,
)
from aiperf.metrics._column_store_handlers import (
    make_numeric_handler as _make_numeric_handler,
)
from aiperf.metrics._column_store_handlers import (
    make_string_handler as _make_string_handler,
)
from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.ragged_series import RaggedSeries

_logger = AIPerfLogger(__name__)

# Backends both implement: ``add_for_record(idx, values)``,
# ``to_result(tag, header, unit)`` (only on TDigest) or per-record accessors
# (only on RaggedSeries), plus ``SUPPORTS_PER_RECORD_REPLAY`` class flag.
ListMetricBackendT = RaggedSeries | TDigestListMetricAggregator


def _resolve_list_backend_class() -> type[ListMetricBackendT]:
    """Pick the list-metric backend class from ``Environment.METRICS.LIST_BACKEND``.

    Resolved on each ColumnStore construction so test-time monkey-patching of
    the env singleton takes effect without a process restart.
    """
    # Imported here to avoid a circular import at module load: environment ->
    # _env_data -> (no metrics dep), but _env_data is read at module init of
    # several siblings — safer to defer.
    from aiperf.common.environment import Environment

    if Environment.METRICS.LIST_BACKEND == "tdigest":
        return TDigestListMetricAggregator
    return RaggedSeries


_BOOL_MISSING = np.uint8(255)
"""Sentinel for an absent ``metadata_bool`` value (NaN-equivalent for uint8)."""

_CATEGORICAL_MISSING = np.int32(-1)
"""Sentinel for an absent ``metadata_categorical`` code. int32 (max ~2.1 B
unique values) avoids the int16 overflow at >32k unique values that
``x_correlation_id`` can hit on single-turn workloads."""


class ColumnStore:
    """Request-indexed NaN-sparse columnar storage for per-record metrics.

    Uses session_num (credit issuance index) as the canonical array index.
    Pre-filled with NaN/None; records write to their slot on arrival in any order.
    """

    __slots__ = (
        "_capacity",
        "_count",
        "_numeric",
        "_string",
        "_ragged",
        "_list_backend_cls",
        "_sums",
        "_counts",
        "_tag_handlers",
        "_metadata_numeric",
        "_metadata_string",
        "_metadata_bool",
        "_metadata_categorical",
        "_metadata_categories",
        "start_ns",
        "end_ns",
        "generation_start_ns",
    )

    def __init__(
        self,
        initial_capacity: int = 1024,
        *,
        list_backend_cls: type[ListMetricBackendT] | None = None,
    ) -> None:
        self._capacity = initial_capacity
        self._count = 0
        self._numeric: dict[str, NDArray[np.float64]] = {}
        self._string: dict[str, list[str | None]] = {}
        self._ragged: dict[str, ListMetricBackendT] = {}
        self._list_backend_cls = list_backend_cls or _resolve_list_backend_class()
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        # Per-tag setter closures, resolved on first sighting of each metric tag
        # (via Python type dispatch: list -> ragged backend, str -> string column,
        # numeric -> float64 column). Subsequent records skip the isinstance
        # ladder and the ``_ensure_*_column`` lookups entirely. Cleared by
        # ``_grow()`` because numeric/metadata-numeric arrays get reallocated;
        # closures captured the old array references and would write to garbage.
        self._tag_handlers: dict[str, Callable[[int, Any], None]] = {}
        # Metadata columns — separate from metric columns so _compute_results()
        # doesn't pick them up. Caller picks the storage type per field based
        # on cardinality + semantics; see ``ingest_metadata`` for the trade-off.
        self._metadata_numeric: dict[str, NDArray[np.float64]] = {}
        self._metadata_string: dict[str, list[str | None]] = {}
        self._metadata_bool: dict[str, NDArray[np.uint8]] = {}
        self._metadata_categorical: dict[str, NDArray[np.int32]] = {}
        # Per-tag intern table: ``categories[tag][string] = int code``.
        self._metadata_categories: dict[str, dict[str, int]] = {}
        self.start_ns = np.full(initial_capacity, np.nan, dtype=np.float64)
        self.end_ns = np.full(initial_capacity, np.nan, dtype=np.float64)
        self.generation_start_ns = np.full(initial_capacity, np.nan, dtype=np.float64)

    @property
    def count(self) -> int:
        """Number of records written (max session_num + 1)."""
        return self._count

    def numeric(self, tag: str) -> NDArray[np.float64]:
        """Return the float64 column for `tag`, sliced to count.

        Returns a NaN-filled array if no record has ingested a value for `tag`.
        Logs a warning when the column is missing on a non-empty store, since
        the most common cause is a typo'd tag name silently producing a
        useless all-NaN result downstream.
        """
        col = self._numeric.get(tag)
        if col is None:
            if self._count > 0:
                _logger.warning(
                    f"ColumnStore.numeric: unknown tag '{tag}' on a non-empty store "
                    f"(known numeric tags: {sorted(self._numeric.keys())}). "
                    "Returning NaN-fill — check for a typo or missing ingestion."
                )
            return np.full(self._count, np.nan, dtype=np.float64)
        return col[: self._count]

    def numeric_tags(self) -> list[str]:
        """Return all numeric column tags."""
        return list(self._numeric.keys())

    def string(self, tag: str) -> list[str | None]:
        """Return the string column for `tag`, sliced to count. None where missing."""
        col = self._string.get(tag)
        if col is None:
            return [None] * self._count
        return col[: self._count]

    def ragged(self, tag: str) -> ListMetricBackendT:
        """Return the list-valued backend for ``tag``.

        Concrete type is :class:`RaggedSeries` (default) or
        :class:`TDigestListMetricAggregator` depending on
        ``Environment.METRICS.LIST_BACKEND``. Both expose
        ``add_for_record(idx, values)``; only the ragged backend exposes
        per-record replay accessors (``values``, ``record_indices``,
        ``offsets``, ``grouped_cumsum``, ``get_values_for_mask``). Consumers
        that need replay must gate on
        ``backend.SUPPORTS_PER_RECORD_REPLAY``.
        """
        return self._ragged[tag]

    def ragged_tags(self) -> list[str]:
        """Return all ragged column tags."""
        return list(self._ragged.keys())

    def numeric_sum(self, tag: str) -> float:
        """Return the running sum for a numeric column (O(1))."""
        return self._sums.get(tag, 0.0)

    def numeric_count(self, tag: str) -> int:
        """Return the count of values ingested for a numeric column (O(1))."""
        return self._counts.get(tag, 0)

    def metadata_numeric(self, tag: str) -> NDArray[np.float64]:
        """Return the metadata float64 column for `tag`, sliced to count. NaN where missing."""
        col = self._metadata_numeric.get(tag)
        if col is None:
            return np.full(self._count, np.nan, dtype=np.float64)
        return col[: self._count]

    def metadata_string(self, tag: str) -> list[str | None]:
        """Return the metadata string column for `tag`, sliced to count. None where missing."""
        col = self._metadata_string.get(tag)
        if col is None:
            return [None] * self._count
        return col[: self._count]

    def metadata_bool(self, tag: str) -> NDArray[np.uint8]:
        """Return the metadata bool column for `tag`, sliced to count.

        Encoding: 0=False, 1=True, 255=missing. Compare against
        ``_BOOL_MISSING`` (255) to detect absence; cast to ``bool`` otherwise.
        """
        col = self._metadata_bool.get(tag)
        if col is None:
            return np.full(self._count, _BOOL_MISSING, dtype=np.uint8)
        return col[: self._count]

    def metadata_categorical(self, tag: str) -> NDArray[np.int32]:
        """Return the per-record category codes for `tag`. -1 = missing.

        Decode via ``metadata_category_strings(tag)[code]`` (when ``code != -1``).
        """
        col = self._metadata_categorical.get(tag)
        if col is None:
            return np.full(self._count, _CATEGORICAL_MISSING, dtype=np.int32)
        return col[: self._count]

    def metadata_category_strings(self, tag: str) -> list[str]:
        """Reverse lookup: code -> original string for a categorical column."""
        table = self._metadata_categories.get(tag, {})
        out = [""] * len(table)
        for s, code in table.items():
            out[code] = s
        return out

    def metadata_categorical_tags(self) -> list[str]:
        """Return all categorical metadata tags (e.g. for grouping enumeration)."""
        return list(self._metadata_categorical.keys())

    def unique_categorical_values(self, tag: str) -> list[str]:
        """Return the unique values that have appeared in categorical column ``tag``.

        Same data as :meth:`metadata_category_strings`; named for the
        per-X-grouping use case where the caller wants to iterate over
        groups (e.g. "for each x_correlation_id, compute per-conversation
        latency stats").
        """
        return self.metadata_category_strings(tag)

    def mask_for_categorical(self, tag: str, value: str) -> NDArray[np.bool_]:
        """Return a boolean mask of records whose ``tag`` column equals ``value``.

        Use case: per-group analyzer queries. Combine with
        :meth:`MetricsAccumulator.compute_results_for_mask` to compute
        windowed metrics for a single group:

        .. code-block:: python

            for value in store.unique_categorical_values("x_correlation_id"):
                mask = store.mask_for_categorical("x_correlation_id", value)
                results = accumulator.compute_results_for_mask(mask)

        Returns an empty mask if the tag has no column or the value never
        appeared (no false-positive matches via the missing-sentinel).
        """
        table = self._metadata_categories.get(tag)
        if table is None:
            return np.zeros(self._count, dtype=np.bool_)
        code = table.get(value)
        if code is None:
            return np.zeros(self._count, dtype=np.bool_)
        col = self._metadata_categorical.get(tag)
        if col is None:
            return np.zeros(self._count, dtype=np.bool_)
        return col[: self._count] == code

    def query_time_range(self, start_ns: float, end_ns: float) -> NDArray[np.bool_]:
        """Return a boolean mask of records overlapping ``[start_ns, end_ns]``.

        A record overlaps the window when ``start_ns <= record.end_ns`` and
        ``record.start_ns <= end_ns``. NaN slots (uningested or partial) are
        excluded by the standard NaN comparison semantics: every comparison
        with NaN returns False, so unfilled rows never match. The window
        endpoints are inclusive.
        """
        if self._count == 0:
            return np.zeros(0, dtype=np.bool_)
        rec_start = self.start_ns[: self._count]
        rec_end = self.end_ns[: self._count]
        return (rec_start <= end_ns) & (rec_end >= start_ns)

    # --- Write API (called from MetricsAccumulator.process_record) ---

    def ingest(
        self,
        idx: int,
        *,
        record_metrics: dict[str, Any],
        start_ns: float,
        end_ns: float,
        generation_start_ns: float | None,
    ) -> None:
        """Write a record's data to slot `idx` (= session_num).

        Grows capacity if idx >= _capacity. Dispatches metric values via cached
        per-tag setter closures — the isinstance ladder and ``_ensure_*_column``
        lookups run only on the first record per tag. Profiling at 50k records
        shows this hoists ~30% of ingest wall time vs the per-record dispatch.
        """
        if idx >= self._capacity:
            self._grow(idx)

        if idx >= self._count:
            self._count = idx + 1

        self.start_ns[idx] = start_ns
        self.end_ns[idx] = end_ns
        if generation_start_ns is not None:
            self.generation_start_ns[idx] = generation_start_ns

        handlers = self._tag_handlers
        for tag, value in record_metrics.items():
            handler = handlers.get(tag)
            if handler is None:
                handler = self._resolve_tag_handler(tag, value)
                if handler is None:
                    continue
                handlers[tag] = handler
            handler(idx, value)

    def _resolve_tag_handler(
        self, tag: str, value: Any
    ) -> Callable[[int, Any], None] | None:
        """First-sighting type dispatch: pick a setter closure for ``tag``.

        Bound on first record only; subsequent records reuse the cached
        closure. Returns ``None`` for unsupported value types so ``ingest``
        can skip the tag without re-dispatching.
        """
        if isinstance(value, list):
            backend = self._ensure_ragged_column(tag)
            return _make_list_handler(backend)
        if isinstance(value, str):
            col = self._ensure_string_column(tag)
            return _make_string_handler(col)
        if isinstance(value, (int, float)):
            col = self._ensure_numeric_column(tag)
            return _make_numeric_handler(col, tag, self._sums, self._counts)
        return None

    def ingest_metadata(
        self,
        idx: int,
        metadata_numeric: dict[str, float | None],
        metadata_string: dict[str, str | None],
        *,
        metadata_bool: dict[str, bool | None] | None = None,
        metadata_categorical: dict[str, str | None] | None = None,
    ) -> None:
        """Write per-record metadata to slot `idx`.

        Metadata columns are kept separate from metric columns so that
        _compute_results() does not treat them as metrics. Caller picks the
        storage type per field based on cardinality + semantics:

        - ``metadata_numeric``: float64 (NaN missing) — high-resolution numbers.
        - ``metadata_string``: list[str|None] — high-cardinality strings (UUIDs).
        - ``metadata_bool``: uint8 with sentinel 255 — saves 8x vs float64.
        - ``metadata_categorical``: int32 + per-tag interning table — saves
          ~25x vs raw strings even at full cardinality, much more on
          low-cardinality fields like ``worker_id``.
        """
        if idx >= self._capacity:
            self._grow(idx)

        for tag, num_value in metadata_numeric.items():
            if num_value is not None:
                self._ensure_metadata_numeric_column(tag)[idx] = float(num_value)

        for tag, str_value in metadata_string.items():
            self._ensure_metadata_string_column(tag)[idx] = str_value

        if metadata_bool:
            self._ingest_bool_metadata(idx, metadata_bool)
        if metadata_categorical:
            self._ingest_categorical_metadata(idx, metadata_categorical)

    def _ingest_bool_metadata(self, idx: int, values: dict[str, bool | None]) -> None:
        for tag, bool_value in values.items():
            if bool_value is not None:
                self._ensure_metadata_bool_column(tag)[idx] = 1 if bool_value else 0

    def _ingest_categorical_metadata(
        self, idx: int, values: dict[str, str | None]
    ) -> None:
        for tag, cat_value in values.items():
            if cat_value is None:
                continue
            # Order matters: ensure the column (which seeds the per-tag
            # categories table) BEFORE interning, since Python evaluates
            # the RHS before the LHS in chained subscript assignments.
            col = self._ensure_metadata_categorical_column(tag)
            col[idx] = self._intern_category(tag, cat_value)

    def _grow(self, min_idx: int) -> None:
        """Double capacity until min_idx fits. Numeric column reallocation
        invalidates ``_tag_handlers`` (cached setter closures held old array
        refs); list/string columns grow in place. Grow runs ~log2(N) times
        so handler-rebuild overhead is negligible.
        """
        new_cap = self._capacity
        while new_cap <= min_idx:
            new_cap *= 2

        for attr in ("start_ns", "end_ns", "generation_start_ns"):
            old = getattr(self, attr)
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            setattr(self, attr, new)

        for tag, old in self._numeric.items():
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            self._numeric[tag] = new

        for tag, old in self._string.items():
            old.extend([None] * (new_cap - self._capacity))
            self._string[tag] = old

        for tag, old in self._metadata_numeric.items():
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            self._metadata_numeric[tag] = new

        for tag, old in self._metadata_string.items():
            old.extend([None] * (new_cap - self._capacity))
            self._metadata_string[tag] = old

        for tag, old in self._metadata_bool.items():
            new = np.full(new_cap, _BOOL_MISSING, dtype=np.uint8)
            new[: self._capacity] = old[: self._capacity]
            self._metadata_bool[tag] = new

        for tag, old in self._metadata_categorical.items():
            new = np.full(new_cap, _CATEGORICAL_MISSING, dtype=np.int32)
            new[: self._capacity] = old[: self._capacity]
            self._metadata_categorical[tag] = new

        # Numeric metric columns were reallocated; cached setter closures
        # captured the old array references. Drop them so the next ingest
        # rebuilds them against the new arrays.
        self._tag_handlers.clear()

        self._capacity = new_cap

    def _ensure_numeric_column(self, tag: str) -> NDArray[np.float64]:
        col = self._numeric.get(tag)
        if col is None:
            col = np.full(self._capacity, np.nan, dtype=np.float64)
            self._numeric[tag] = col
            self._sums[tag] = 0.0
            self._counts[tag] = 0
        return col

    def _ensure_string_column(self, tag: str) -> list[str | None]:
        col = self._string.get(tag)
        if col is None:
            col = [None] * self._capacity
            self._string[tag] = col
        return col

    def _ensure_ragged_column(self, tag: str) -> ListMetricBackendT:
        ragged = self._ragged.get(tag)
        if ragged is None:
            ragged = self._list_backend_cls()
            self._ragged[tag] = ragged
        return ragged

    def _ensure_metadata_numeric_column(self, tag: str) -> NDArray[np.float64]:
        col = self._metadata_numeric.get(tag)
        if col is None:
            col = np.full(self._capacity, np.nan, dtype=np.float64)
            self._metadata_numeric[tag] = col
        return col

    def _ensure_metadata_string_column(self, tag: str) -> list[str | None]:
        col = self._metadata_string.get(tag)
        if col is None:
            col = [None] * self._capacity
            self._metadata_string[tag] = col
        return col

    def _ensure_metadata_bool_column(self, tag: str) -> NDArray[np.uint8]:
        col = self._metadata_bool.get(tag)
        if col is None:
            col = np.full(self._capacity, _BOOL_MISSING, dtype=np.uint8)
            self._metadata_bool[tag] = col
        return col

    def _ensure_metadata_categorical_column(self, tag: str) -> NDArray[np.int32]:
        col = self._metadata_categorical.get(tag)
        if col is None:
            col = np.full(self._capacity, _CATEGORICAL_MISSING, dtype=np.int32)
            self._metadata_categorical[tag] = col
            self._metadata_categories[tag] = {}
        return col

    def _intern_category(self, tag: str, value: str) -> int:
        """Look up or insert ``value`` in the per-tag category table; return the int32 code."""
        table = self._metadata_categories[tag]
        code = table.get(value)
        if code is None:
            code = len(table)
            table[value] = code
        return code
