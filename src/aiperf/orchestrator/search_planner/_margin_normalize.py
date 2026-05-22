# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sigma-normalized multi-SLO margin aggregation.

vLLM's ``max(margin_i)`` aggregates raw ``(observed - threshold)`` values,
implicitly weighting constraints by numeric scale: a 50 ms TTFT margin with
sigma=5 reads "looser" than a 500 ms TPOT margin with sigma=80 even though
the TTFT constraint is statistically tighter. ``normalize_margins`` divides
by the per-constraint replicate stddev (floored at ``1% * |threshold|`` to
avoid blow-up on degenerate noise-free constraints) so the binding
constraint is the one with the worst *signal-to-noise* margin, not the
worst raw margin.

Falls back to raw max-of-margins when ``sigmas`` is unavailable (first
iteration, before any replicate has landed).
"""

from __future__ import annotations

__all__ = ["normalize_margins"]


def normalize_margins(
    margins: dict[str, float],
    sigmas: dict[str, float] | None,
    thresholds: dict[str, float],
    sigma_floor_frac: float = 0.01,
) -> tuple[float, str]:
    """Return ``(binding_normalized_margin, binding_constraint_key)``.

    Each input maps the SLA-filter constraint key (e.g. ``"ttft.p95"``) to its
    scalar:

    * ``margins`` — raw ``observed - threshold`` per constraint.
    * ``sigmas`` — per-replicate standard deviation of margins, or None /
      empty when no replicates have run yet.
    * ``thresholds`` — SLA threshold per constraint, used to floor a
      degenerate ``sigma == 0`` at ``sigma_floor_frac * |threshold|``.

    When ``sigmas`` is ``None`` or empty, falls back to vLLM-style raw
    ``max(margins)``; the binding key is ``argmax`` over raw margins. With
    sigmas, each ``sigma`` is floored at ``max(sigma, sigma_floor_frac *
    |threshold|)`` so a noise-free constraint can't drive the normalized
    margin to infinity, and we return the ``argmax`` of
    ``margins[k] / sigma_floored[k]``.
    """
    if not margins:
        raise ValueError("normalize_margins requires at least one constraint margin")
    if not sigmas:
        binding_key = max(margins, key=lambda k: margins[k])
        return (float(margins[binding_key]), binding_key)
    normalized: dict[str, float] = {}
    for key, raw_margin in margins.items():
        sigma = sigmas.get(key, 0.0)
        threshold = thresholds.get(key, 0.0)
        sigma_floored = max(sigma, sigma_floor_frac * abs(threshold))
        if sigma_floored == 0.0:
            normalized[key] = raw_margin
        else:
            normalized[key] = raw_margin / sigma_floored
    binding_key = max(normalized, key=lambda k: normalized[k])
    return (float(normalized[binding_key]), binding_key)
