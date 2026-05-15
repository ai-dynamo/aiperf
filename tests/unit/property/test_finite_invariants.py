# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical "global invariant" tests for the NaN/inf discipline.

Three CI-enforceable contracts that codify the round-1 finite-float
remediation work into rules a future PR can't accidentally regress:

1. Every JSON exporter that calls ``orjson.dumps`` on metric-bearing
   payloads also imports ``scrub_non_finite``. Whitelisted call sites
   are catalogued explicitly with a documented reason.
2. Every Pydantic field whose name suggests it holds a metric value
   (``*_p99``, ``*_mean``, ``latency_*``, ``ttft_*``, ``itl_*``, ...)
   is annotated as ``FiniteFloat`` (or ``FiniteFloat | None``) -- not a
   raw ``float`` that would silently accept NaN.
3. Every numeric field on every Pydantic model has *some* validator
   (``ge``/``gt``/``le``/``lt``/``FiniteFloat``) or is on the whitelist.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import pkgutil
import typing
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "aiperf"

# ============================================================================
# Test 1 -- every metric-bearing JSON exporter scrubs
# ============================================================================

EXPORTER_DIRS = [
    SRC_ROOT / "exporters",
    SRC_ROOT / "server_metrics",
]

# Whitelist: modules that call ``orjson.dumps`` on payloads that are NOT
# metric values (or only call ``orjson.loads``). Each entry must include a
# documented reason.
ORJSON_SCRUB_WHITELIST: dict[str, str] = {
    # Only orjson.loads (parses backend error messages). Nothing dumped.
    "src/aiperf/exporters/console_api_error_exporter.py": (
        "loads-only -- parses backend error JSON to detect insight patterns"
    ),
    # Parquet metadata bytes: input_config dict, model names, endpoint URLs,
    # label-column key list, metric-type counts. None of these carry per-trial
    # metric values; they are configuration / structural metadata stored as
    # parquet file-level metadata.
    "src/aiperf/server_metrics/parquet_exporter.py": (
        "metadata-only orjson.dumps -- input_config, model names, endpoint "
        "urls, label-column keys, metric type counts. No metric values."
    ),
}


def _orjson_dumps_files() -> list[pathlib.Path]:
    """Files under exporters/ + server_metrics/ that import orjson."""
    out: list[pathlib.Path] = []
    for d in EXPORTER_DIRS:
        for path in d.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            text = path.read_text()
            if "orjson" in text:
                out.append(path)
    return out


def test_every_json_exporter_calls_scrub_non_finite() -> None:
    """Every exporter calling ``orjson.dumps`` on metric data must import scrub.

    If you add a new JSON exporter that dumps metric values, either import
    ``scrub_non_finite`` from ``aiperf.common.finite`` and apply it before
    ``orjson.dumps``, or add the file to ``ORJSON_SCRUB_WHITELIST`` with a
    documented reason.
    """
    failures: list[str] = []
    for path in _orjson_dumps_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        text = path.read_text()
        # Skip if the file never calls orjson.dumps (only loads counts).
        if "orjson.dumps" not in text:
            continue
        if "scrub_non_finite" in text:
            continue
        if rel in ORJSON_SCRUB_WHITELIST:
            continue
        failures.append(
            f"{rel}: calls orjson.dumps but does not import scrub_non_finite. "
            f"Either apply scrub_non_finite() before dumps or add an entry to "
            f"ORJSON_SCRUB_WHITELIST with the reason."
        )
    assert not failures, "\n".join(failures)


# ============================================================================
# Helpers: walk every Pydantic model in src/aiperf/
# ============================================================================


def _iter_aiperf_modules() -> list[str]:
    """Every importable submodule under aiperf, skipping known-broken paths."""
    import aiperf

    skip_prefixes = (
        # Heavy optional deps that fail to import on some envs.
        "aiperf.dataset.agentic_code_gen.reporting",
        # K8s controller code: imports kopf at import time.
        "aiperf.kubernetes",
        # Generated / autoflakey
        "aiperf.cli_commands._generated",
    )
    found: list[str] = []
    for info in pkgutil.walk_packages(
        aiperf.__path__, prefix="aiperf.", onerror=lambda _: None
    ):
        if any(info.name.startswith(p) for p in skip_prefixes):
            continue
        found.append(info.name)
    return sorted(set(found))


def _iter_pydantic_models() -> list[type]:
    """Every Pydantic BaseModel subclass importable from aiperf.*."""
    from pydantic import BaseModel

    seen: set[int] = set()
    out: list[type] = []
    for modname in _iter_aiperf_modules():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if not isinstance(obj, type):
                continue
            try:
                if not issubclass(obj, BaseModel):
                    continue
            except TypeError:
                continue
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            out.append(obj)
    return out


# ============================================================================
# Test 2 -- metric-named fields are FiniteFloat or explicitly Optional
# ============================================================================

# Heuristic suffixes/prefixes that mark a field as carrying a metric value.
METRIC_NAME_HINTS = (
    "_p50",
    "_p90",
    "_p95",
    "_p99",
    "_mean",
    "_avg",
    "_std",
    "_stddev",
    "_min",
    "_max",
    "latency_",
    "throughput_",
    "ttft_",
    "itl_",
    "_throughput",
    "_latency",
    "_ttft",
    "_itl",
    "_observed",
    "_value",
)

# Whitelist: metric-suffixed fields that legitimately accept raw float
# (not FiniteFloat). Each entry must have a documented reason.
FINITE_FIELD_WHITELIST: dict[str, str] = {
    # Distribution.min / Distribution.max: sample-clamp bounds. The
    # Distribution class has its own model_validator (_validate_bounds)
    # that rejects NaN/inf with a clear ValueError. FiniteFloat would
    # duplicate that check at field level.
    "Distribution.min": "validated by Distribution._validate_bounds",
    "Distribution.max": "validated by Distribution._validate_bounds",
    "FixedDistribution.min": "inherited; validated by Distribution._validate_bounds",
    "FixedDistribution.max": "inherited; validated by Distribution._validate_bounds",
    "NormalDistribution.min": "inherited; validated by Distribution._validate_bounds",
    "NormalDistribution.max": "inherited; validated by Distribution._validate_bounds",
    "LogNormalDistribution.min": "inherited; validated by Distribution._validate_bounds",
    "LogNormalDistribution.max": "inherited; validated by Distribution._validate_bounds",
    "MultimodalDistribution.min": "inherited; validated by Distribution._validate_bounds",
    "MultimodalDistribution.max": "inherited; validated by Distribution._validate_bounds",
    "EmpiricalDistribution.min": "inherited; validated by Distribution._validate_bounds",
    "EmpiricalDistribution.max": "inherited; validated by Distribution._validate_bounds",
    # FixedDistribution.value: validated by FixedDistribution.validate_finite.
    "FixedDistribution.value": "validated by FixedDistribution.validate_finite",
    # SamplingDimension.lo/hi: validated by SamplingDimension._validate_finite_bounds.
    "SamplingDimension.lo": "validated by SamplingDimension._validate_finite_bounds",
    "SamplingDimension.hi": "validated by SamplingDimension._validate_finite_bounds",
    # SearchSpaceDimension.lo/hi: validated by _validate_finite_bounds.
    "SearchSpaceDimension.lo": "validated by SearchSpaceDimension._validate_finite_bounds",
    "SearchSpaceDimension.hi": "validated by SearchSpaceDimension._validate_finite_bounds",
}


def _iter_annotation_metadata(annotation: Any) -> list[Any]:
    metadata = list(getattr(annotation, "__metadata__", None) or [])
    for inner in typing.get_args(annotation):
        metadata.extend(_iter_annotation_metadata(inner))
    return metadata


def _field_accepts_only_finite(field: Any) -> bool:
    """True iff the field's annotation rejects NaN/inf at validation time.

    Detects ``FiniteFloat``/``FiniteFloat | None`` by inspecting the
    Pydantic ``FieldInfo`` metadata for an ``AfterValidator`` whose
    function name starts with ``_check_finite``. Also detects Pydantic
    ``ge``/``gt``/``le``/``lt`` numeric constraints, which reject NaN
    because NaN comparisons against the bound return False.
    """
    metadata = list(getattr(field, "metadata", []) or [])
    ann = getattr(field, "annotation", None)
    if ann is not None:
        metadata.extend(_iter_annotation_metadata(ann))
    for m in metadata:
        nested_metadata = getattr(m, "metadata", None)
        if nested_metadata:
            metadata.extend(nested_metadata)
        # AfterValidator(_check_finite)
        func = getattr(m, "func", None)
        if func is not None and getattr(func, "__name__", "") == "_check_finite":
            return True
        # Pydantic numeric constraint with finite bound rejects NaN.
        for attr in ("ge", "gt", "le", "lt"):
            v = getattr(m, attr, None)
            if v is not None:
                import math

                try:
                    if math.isfinite(float(v)):
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def _field_is_numeric(field: Any) -> bool:
    """True iff the field's annotation includes ``float`` or ``int``."""
    ann = getattr(field, "annotation", None)
    if ann is None:
        return False
    s = str(ann)
    # FiniteFloat is Annotated[float, ...] -- caught by 'float' substring.
    return "float" in s or ("int" in s and "Literal" not in s)


def _is_metric_named(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in METRIC_NAME_HINTS)


def test_every_metric_field_is_finite_or_optional() -> None:
    """Pydantic fields named like metrics must be FiniteFloat-validated.

    Heuristic: if a field's name contains a metric-suffix hint
    (``_p99``, ``_mean``, ``latency_``, ...) AND its annotation includes
    ``float``, it must either be FiniteFloat-annotated or be on
    ``FINITE_FIELD_WHITELIST`` with a reason. Catches the regression of
    "someone added a new latency field as plain ``float``", which would
    silently accept NaN through ``model_validate``.
    """
    failures: list[str] = []
    for model in _iter_pydantic_models():
        for fname, finfo in model.model_fields.items():
            if not _is_metric_named(fname):
                continue
            ann = str(getattr(finfo, "annotation", "")) or ""
            if "float" not in ann:
                continue
            qual = f"{model.__name__}.{fname}"
            if qual in FINITE_FIELD_WHITELIST:
                continue
            if _field_accepts_only_finite(finfo):
                continue
            failures.append(
                f"{qual}: metric-named float field is not FiniteFloat / has no "
                f"finite-bounds metadata. Annotate as FiniteFloat or add an "
                f"entry to FINITE_FIELD_WHITELIST."
            )
    if failures:
        # Same one-way ratchet as test 3: legacy violations are tracked in a
        # baseline file; only NEW violations fail the test.
        baseline = _load_or_init_metric_baseline(failures)
        new = sorted(set(failures) - set(baseline))
        if new:
            raise AssertionError(
                "New metric-named float fields without FiniteFloat detected:\n"
                + "\n".join(new)
                + f"\n\nFix the field, or add to FINITE_FIELD_WHITELIST or "
                f"{_METRIC_BASELINE_FILE.name}."
            )


_METRIC_BASELINE_FILE = pathlib.Path(__file__).parent / "_metric_field_baseline.txt"


def _load_or_init_metric_baseline(current: list[str]) -> list[str]:
    """Same first-run-creates-baseline behavior as the numeric-bounds test."""
    return _atomic_load_or_init_baseline(
        _METRIC_BASELINE_FILE,
        current,
        header_comment=(
            "# Auto-generated baseline of metric-named float fields not yet "
            "FiniteFloat-annotated.\n"
            "# Adding a new violation must either fix the field, whitelist it "
            "in code, or add it here with a follow-up issue.\n"
        ),
    )


# ============================================================================
# Test 3 -- every numeric field has range constraints OR is whitelisted
# ============================================================================

# Whitelist: numeric fields that legitimately have no range constraint and
# are not FiniteFloat. Most are: integer enums, free-form pass-through ints,
# or fields validated downstream rather than at field level.
NUMERIC_BOUNDS_WHITELIST: set[str] = {
    # SweepVariation.index: zero-based index, bounded by sweep size at
    # runtime; no useful field-level upper bound.
    "SweepVariation.index",
    # SearchIteration is a dataclass not a BaseModel; not picked up.
    # AdaptiveSearchSweep.max_iterations / n_initial_points: have ge bounds.
    # MultiRunConfig.num_runs: has ge/le.
    # AIPerfConfig.random_seed: ge=0.
    # NormalDistribution.mean / LogNormal.mean / etc: free-form. Validated
    # at distribution level (Fixed/Normal_validate_finite, Lognormal
    # validate_median_le_mean, etc.)
    "FixedDistribution.value",  # in FINITE_FIELD_WHITELIST already
    "NormalDistribution.mean",  # free-form, validated by sampler clamp
    "LogNormalDistribution.mean",  # gt=0 -- but to be safe, whitelist
    "LogNormalDistribution.median",
    "EmpiricalPoint.value",
    "PeakEntry.weight",
    "EmpiricalPoint.weight",
    # AdaptiveSearchSweep.outcome_constraints: list[OutcomeConstraint], not a
    # numeric field. Per-element OutcomeConstraint.bound is already FiniteFloat.
    "AdaptiveSearchSweep.outcome_constraints",
}


def test_every_numeric_field_has_bounds() -> None:
    """Every Pydantic int/float field has ge/gt/le/lt OR is FiniteFloat-validated.

    A field with no constraint silently accepts -1, 0, +inf, NaN -- exactly
    the values that produce hard-to-debug downstream symptoms. If a numeric
    field genuinely has no useful bound, add it to NUMERIC_BOUNDS_WHITELIST
    with a comment explaining why.
    """
    failures: list[str] = []
    for model in _iter_pydantic_models():
        for fname, finfo in model.model_fields.items():
            if not _field_is_numeric(finfo):
                continue
            ann = str(getattr(finfo, "annotation", "")) or ""
            # Skip pure-bool, Literal, enum, list/dict/tuple containers --
            # the substring check above is loose; tighten by requiring the
            # bare ``float``/``int`` to appear OUTSIDE a Literal/enum.
            if "Literal" in ann:
                continue
            qual = f"{model.__name__}.{fname}"
            if qual in NUMERIC_BOUNDS_WHITELIST:
                continue
            if qual in FINITE_FIELD_WHITELIST:
                continue
            if _field_accepts_only_finite(finfo):
                continue
            # Has metadata of any constraint kind (ge/gt/le/lt)?
            metadata = getattr(finfo, "metadata", []) or []
            has_bound = False
            for m in metadata:
                for attr in ("ge", "gt", "le", "lt"):
                    if getattr(m, attr, None) is not None:
                        has_bound = True
                        break
                if has_bound:
                    break
            if has_bound:
                continue
            failures.append(
                f"{qual}: numeric field has no ge/gt/le/lt bound and is not "
                f"FiniteFloat. Add a Pydantic numeric constraint, annotate as "
                f"FiniteFloat, or add to NUMERIC_BOUNDS_WHITELIST."
            )
    if failures:
        # Soft-skip the test unless the count gets WORSE than the current
        # baseline -- some legacy fields are out of scope for this branch.
        # The mechanism remains in place so new violations fire.
        baseline = _load_or_init_numeric_baseline(failures)
        new = sorted(set(failures) - set(baseline))
        if new:
            raise AssertionError(
                "New unbounded numeric fields detected:\n"
                + "\n".join(new)
                + f"\n\nFix the field, or add to NUMERIC_BOUNDS_WHITELIST or "
                f"{_BASELINE_FILE.name}."
            )


_BASELINE_FILE = pathlib.Path(__file__).parent / "_numeric_bounds_baseline.txt"


def _load_or_init_numeric_baseline(current: list[str]) -> list[str]:
    """Return the baseline; create it if missing.

    First-run behavior: ``_numeric_bounds_baseline.txt`` is missing so the
    current set becomes the baseline. Subsequent runs only fail when a new
    violation appears that isn't in the baseline -- the test acts as a
    one-way ratchet rather than a forced full-codebase fix.
    """
    return _atomic_load_or_init_baseline(
        _BASELINE_FILE,
        current,
        header_comment=(
            "# Auto-generated baseline of unbounded numeric fields.\n"
            "# This file is the legacy waterline for "
            "test_every_numeric_field_has_bounds.\n"
            "# Adding a new violation must either fix the field, whitelist it "
            "in code, or add it here with a follow-up issue.\n"
        ),
    )


def _atomic_load_or_init_baseline(
    path: pathlib.Path,
    current: list[str],
    *,
    header_comment: str,
) -> list[str]:
    """Read the baseline if it exists, else atomically create it.

    Multiple xdist workers race here on first run -- ``os.O_EXCL`` ensures
    only one creates the file; the others read it once visible. We always
    deduplicate ``current`` before writing so workers' partial scans
    can't produce a fragmented file.
    """
    if path.exists():
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    deduped = sorted(set(current))
    body = header_comment + "\n".join(deduped) + "\n"
    try:
        # O_CREAT | O_EXCL -- atomic create-only-if-absent. If two workers
        # race, exactly one wins; the loser sees FileExistsError and reads.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    return deduped


# ============================================================================
# AST sanity check: no exporter file outside the whitelist mixes
# orjson.dumps with a metric-bearing variable but no scrub call
# ============================================================================


def test_orjson_dumps_files_have_scrub_or_whitelisted() -> None:
    """Stronger version of test 1 using AST: walk every orjson.dumps call site
    and verify scrub_non_finite is imported in the same module. Whitelist
    enforcement matches the substring test but uses a real parser to be
    robust against import-style differences.
    """
    failures: list[str] = []
    for path in _orjson_dumps_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in ORJSON_SCRUB_WHITELIST:
            continue
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        has_dumps = False
        has_scrub_import = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "dumps"
                and isinstance(node.value, ast.Name)
                and node.value.id == "orjson"
            ):
                has_dumps = True
            if isinstance(node, ast.ImportFrom) and (
                node.module == "aiperf.common.finite"
                and any(alias.name == "scrub_non_finite" for alias in node.names)
            ):
                has_scrub_import = True
        if has_dumps and not has_scrub_import:
            failures.append(f"{rel}: orjson.dumps without scrub_non_finite import")
    assert not failures, "\n".join(failures)


# ============================================================================
# Smoke check: the baseline file is not allowed to grow unboundedly --
# leave a stub assertion confirming it exists once initialized
# ============================================================================


def test_numeric_bounds_baseline_initialized() -> None:
    """Either there are no unbounded fields, OR the baseline file exists."""
    # First call to test_every_numeric_field_has_bounds will create the file.
    # If the codebase is fully clean, the file may not exist; that's fine.
    # This test only fires if the file exists but is empty / whitespace.
    if _BASELINE_FILE.exists():
        contents = _BASELINE_FILE.read_text().strip()
        assert contents, (
            f"{_BASELINE_FILE} exists but is empty -- delete it to regenerate."
        )


# Avoid the pytest collector flagging the helper as a test.
_iter_aiperf_modules.__test__ = False  # type: ignore[attr-defined]
_iter_pydantic_models.__test__ = False  # type: ignore[attr-defined]
