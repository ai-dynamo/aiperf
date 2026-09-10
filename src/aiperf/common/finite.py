# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized NaN/inf discipline for AIPerf.

Every numeric metric value that crosses a serialization boundary (orjson,
Pydantic ``model_dump_json``, CSV writer) or feeds a numerical algorithm
(``np.mean``, ``polyfit``, BO acquisitions) must be either **finite** or
**explicitly None**. NaN/inf values look benign in memory but corrupt
downstream artifacts and analyses in three distinct ways:

1. ``orjson.dumps`` and Pydantic's ``model_dump_json`` silently coerce
   NaN/+inf/-inf to JSON ``null``. Once on disk, ``null`` is
   indistinguishable from "metric was missing" — the contract used by
   sentinels like ``SLABreachKnee.breaches[].observed`` collapses.
2. Naive CSV ``f"{value:.2f}"`` formatting writes the literal strings
   ``"nan"``/``"inf"``, which downstream pandas/duckdb readers parse
   inconsistently (string column on mixed input, float NaN on uniform).
3. ``np.mean``/``np.std``/``polyfit`` on arrays containing NaN poison
   subsequent decision logic (Pareto fronts, BO acquisition maxima,
   plateau detectors) without raising.

This module centralizes the discipline as four primitives:

- :data:`FiniteFloat` -- a Pydantic float type that *rejects* NaN/inf at
  validation time. Use it for any new metric field without finite=missing
  semantics.
- :func:`is_finite_value` -- duck-typed finiteness check that works on
  Python ``int``/``float`` AND numpy scalar types (``numpy.float32``,
  ``numpy.float64``, ``numpy.int64``); rejects ``bool`` by design.
- :func:`scrub_non_finite` -- recursively rewrites non-finite numeric
  values to ``None`` in dict/list/tuple structures. Apply before every
  ``orjson.dumps`` call that may carry metric data.
- :func:`nan_safe_mean` / :func:`nan_safe_std` -- aggregations that
  ignore non-finite inputs and return ``None`` when no finite values
  remain (rather than silently returning NaN).
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import AfterValidator

__all__ = [
    "FiniteFloat",
    "is_finite_value",
    "nan_safe_mean",
    "nan_safe_std",
    "scrub_non_finite",
]


def _native_scalar(x: Any) -> Any:
    """Return the native Python counterpart of a numpy scalar, else ``x``.

    Numpy's ``.item()`` maps each scalar to the Python type it actually
    means -- ``numpy.bool_`` to ``bool``, ``numpy.int64`` to ``int``,
    ``numpy.float32`` to ``float`` -- which ``float()`` alone would flatten.
    Used instead of ``import numpy`` so this module stays import-light; it
    is pulled in on per-record paths. Python scalars have no ``.item()`` and
    cost one failed attribute lookup.
    """
    item = getattr(x, "item", None)
    if not callable(item):
        return x
    try:
        return item()
    except (TypeError, ValueError):
        return x


def is_finite_value(x: Any) -> bool:
    """Return True if ``x`` is a finite real number.

    Returns False for ``None``, ``bool`` (semantic: a bool is not a metric
    value even though Python treats it as numeric), NaN, +inf, -inf, and
    anything that cannot be coerced to ``float``. Works on Python
    ``int``/``float`` and numpy scalar types (``float32``, ``float64``,
    ``int64``, ...) because ``float(np.float64(...))`` round-trips.

    ``numpy.bool_`` is rejected alongside Python ``bool``. It does not
    subclass ``bool``, so without an explicit check it would be admitted as
    the value ``1.0`` and skew :func:`nan_safe_mean` / :func:`nan_safe_std`
    -- the same input would average differently depending only on which
    bool type produced it.

    Strings, bytes, lists, dicts and other non-numeric types return False
    (the ``float()`` coercion either raises ``ValueError`` or
    ``TypeError``, both of which are caught).
    """
    if x is None or isinstance(_native_scalar(x), bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check_finite(x: float) -> float:
    """Pydantic AfterValidator that rejects NaN/+inf/-inf in float fields.

    Raises ``ValueError`` with a message that includes the rejected value
    so the failure is debuggable in nested validation contexts.
    """
    if not math.isfinite(x):
        raise ValueError(f"value must be finite, got {x!r}")
    return x


FiniteFloat = Annotated[float, AfterValidator(_check_finite)]
"""Pydantic float type that rejects NaN/+inf/-inf at validation time.

Use for any metric field that has no finite=missing semantic. For fields
where ``None`` means missing, use ``FiniteFloat | None`` -- the validator
only fires when a non-None value is provided.

Example::

    class MetricSummary(AIPerfBaseModel):
        mean: FiniteFloat = Field(description="Sample mean (must be finite)")
        std: FiniteFloat | None = Field(
            default=None,
            description="Sample stddev; None means insufficient samples",
        )
"""


def _scrub_scalar(obj: Any) -> Any:
    """Normalize one non-container value for :func:`scrub_non_finite`.

    Finite numerics come back as their native Python type, non-finite ones
    as ``None``, and anything non-numeric passes through untouched.
    """
    if isinstance(obj, float):
        # ``numpy.float64`` is the one numpy scalar type that subclasses
        # ``float``, so it matches here rather than taking the ``.item()``
        # path below. Returning it unconverted leaks it into orjson.dumps(),
        # which rejects it outright ("Type is not JSON serializable").
        # float() is a no-op on an exact float -- CPython returns the same
        # object, and the call is cheaper than the isfinite() beside it.
        return float(obj) if math.isfinite(obj) else None
    if not hasattr(obj, "__float__") or isinstance(obj, int):
        return obj
    # Numpy scalar or other duck-typed number. ``numpy.int64(7)`` must stay
    # ``7`` rather than widen to ``7.0``, and ``numpy.bool_(True)`` must stay
    # ``True`` rather than collapse to ``1.0``.
    native = _native_scalar(obj)
    # bool is caught here too, since bool subclasses int; returning
    # ``native`` keeps True a bool instead of flattening it to 1.
    if isinstance(native, int):
        return native
    try:
        f = float(native)
    except (TypeError, ValueError):
        return obj
    return f if math.isfinite(f) else None


def scrub_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite numeric values with ``None``.

    Walks ``dict``, ``list``, and ``tuple`` containers; leaves ``str``,
    ``bytes``, and ``bytearray`` alone (a string literal ``"nan"`` is not a
    numeric NaN and must not be rewritten). Numpy scalars are normalized to
    their native Python counterpart rather than uniformly to ``float``, so
    ``numpy.int64(7)`` yields ``7`` (not ``7.0``) and ``numpy.bool_(True)``
    yields ``True`` (not ``1.0``). ``numpy.float64`` needs an explicit cast
    because it subclasses ``float``; the rest go through ``.item()``.

    Use before ``orjson.dumps`` on any payload that may contain metric
    values. This is the guard for two distinct orjson behaviors: it
    silently coerces NaN/inf to JSON ``null`` (indistinguishable from
    explicit-None semantics downstream), and it raises outright on numpy
    scalars ("Type is not JSON serializable: numpy.float64").

    The returned structure preserves the input container types (dict stays
    dict, tuple stays tuple). Booleans are passed through unchanged because
    they are not metric values -- a non-finite check does not apply to them.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_non_finite(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_non_finite(v) for v in obj)
    return _scrub_scalar(obj)


def nan_safe_mean(values: Any) -> float | None:
    """Return the mean of finite values in ``values``, or None if none exist.

    Filters non-finite entries (NaN/+inf/-inf/None/non-numeric) before
    averaging. Returns ``None`` rather than NaN when the input contains
    no finite values, so callers can distinguish "no data" from "data
    averaged to NaN".
    """
    finite = [float(v) for v in values if is_finite_value(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def nan_safe_std(values: Any, ddof: int = 1) -> float | None:
    """Return the sample stddev of finite values, or None if too few.

    Filters non-finite entries first; returns ``None`` when fewer than
    ``1 + ddof`` finite values remain (the minimum sample size for the
    requested degrees of freedom). Default ``ddof=1`` matches the textbook
    sample-stddev / pandas convention; numpy's ``np.std`` defaults to
    ddof=0.
    """
    finite = [float(v) for v in values if is_finite_value(v)]
    if len(finite) < 1 + ddof:
        return None
    mean = sum(finite) / len(finite)
    sq = sum((v - mean) ** 2 for v in finite)
    return math.sqrt(sq / (len(finite) - ddof))
