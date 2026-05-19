# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QMC sweep expansion (Sobol, Latin Hypercube).

Lives separately from `expand.py` so the scipy.stats.qmc import
stays isolated to QMC-only code paths.
"""

from __future__ import annotations

import copy
from math import ceil, exp, floor, log
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.sweep.config import SweepVariation
    from aiperf.config.sweep.sampling import SamplingDimension


def _qmc_engine(sweep_type: str, d: int, seed: int | None, opts: dict[str, Any]):
    """Return a scipy QMC engine. Imported lazily to keep scipy out of
    grid-sweep code paths.
    """
    from scipy.stats import qmc

    if sweep_type == "sobol":
        return qmc.Sobol(d=d, scramble=opts.get("scramble", True), seed=seed)
    if sweep_type == "latin_hypercube":
        return qmc.LatinHypercube(d=d, optimization=opts.get("optimization"), seed=seed)
    raise ValueError(f"unknown sampling sweep type: {sweep_type!r}")


def _map_dim(u: float, dim: SamplingDimension) -> Any:
    """Map a unit-cube coord u in [0, 1) to a value per dim spec."""
    if dim.choices is not None:
        idx = min(int(floor(u * len(dim.choices))), len(dim.choices) - 1)
        return dim.choices[idx]
    if dim.scale == "log":
        v = exp(log(dim.lo) + u * (log(dim.hi) - log(dim.lo)))
    else:
        v = dim.lo + u * (dim.hi - dim.lo)
    if dim.kind == "int":
        # Why: banker's rounding can push values outside the user-declared
        # [lo, hi] -- e.g. lo=0.5, hi=1.0, scale=log, u=0.0 maps to v=0.5,
        # round(0.5)=0, which is below lo. Clamp to [ceil(lo), floor(hi)]
        # so produced ints always honor the declared range. The narrow-int
        # range warning on SamplingDimension already alerts users when the
        # range is too tight for sampling diversity, but doesn't block.
        return max(ceil(dim.lo), min(int(round(v)), floor(dim.hi)))
    return float(v)


def expand_qmc_sweep(
    data: dict[str, Any],
    *,
    sweep_type: str,
    samples: int,
    seed: int | None,
    dimensions: list[SamplingDimension],
    options: dict[str, Any],
    label_format: str = "index",
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Expand a QMC sweep into (variant_dict, SweepVariation) tuples.

    Same return shape as `_expand_grid_sweep`, so the orchestrator and
    aggregator do not need to know this code exists.
    """
    import warnings

    from aiperf.config.sweep import SweepVariation
    from aiperf.config.sweep.expand import _set_nested_value

    if sweep_type == "sobol" and samples & (samples - 1) != 0:
        warnings.warn(
            f"Sobol balance is best at powers of 2 (8, 16, 32, ...); "
            f"got samples={samples}. Consider {1 << (samples.bit_length())}.",
            stacklevel=2,
        )
    if sweep_type == "sobol" and options.get("scramble") is False:
        warnings.warn(
            "Sobol with scramble=False uses the raw scipy sequence whose "
            "first row is all-zeros — the first variant will be (lo, lo, ...) "
            "for every dimension and small `samples` produces degenerate "
            "corner-only coverage. Prefer scramble=True unless you have a "
            "specific reason to want raw Sobol points.",
            stacklevel=2,
        )

    engine = _qmc_engine(sweep_type, d=len(dimensions), seed=seed, opts=options)
    if sweep_type == "sobol" and samples & (samples - 1) == 0:
        samples_unit_cube = engine.random_base2(m=samples.bit_length() - 1)
    elif sweep_type == "sobol":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol' points require n to be a power of 2.",
                category=UserWarning,
            )
            samples_unit_cube = engine.random(n=samples)
    else:
        samples_unit_cube = engine.random(n=samples)

    base = {k: v for k, v in data.items() if k != "sweep"}
    out: list[tuple[dict[str, Any], SweepVariation]] = []
    for i, row in enumerate(samples_unit_cube):
        values = {
            dim.path: _map_dim(float(u), dim)
            for dim, u in zip(dimensions, row, strict=True)
        }
        variant = copy.deepcopy(base)
        body = variant.setdefault("benchmark", {})
        for path, value in values.items():
            # Mirror the grid-sweep convention: dotted paths are body-rooted
            # under `benchmark:`. The single envelope-level escape is
            # `variables.<name>`, which writes into the root `variables:`
            # Jinja block per variation. SamplingDimension's path validator
            # rejects other top-level prefixes (sweep/multi_run/random_seed/
            # benchmark) so by here the only envelope path we accept is
            # `variables.*`.
            if path.split(".", 1)[0] == "variables":
                _set_nested_value(variant, path, value)
            else:
                _set_nested_value(body, path, value)
        if label_format == "kv":
            label = ", ".join(f"{k}={v}" for k, v in values.items())
        else:
            label = f"{sweep_type}_{i:04d}"
        out.append((variant, SweepVariation(index=i, label=label, values=values)))
    return out
