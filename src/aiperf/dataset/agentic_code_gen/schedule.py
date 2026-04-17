# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand a ConcurrencyScheduleConfig into a dense (time_sec, concurrency) tick stream."""

from __future__ import annotations

import numpy as np

from aiperf.dataset.agentic_code_gen.models import ConcurrencyScheduleConfig


def expand_schedule(
    cfg: ConcurrencyScheduleConfig, rng: np.random.Generator
) -> list[tuple[float, int]]:
    """Walk cfg.anchors at cfg.tick_sec granularity, interpolate, apply noise.

    Returns a chronological list of (time_sec, concurrency) ticks where every
    concurrency is a positive int. Values are rounded and clamped to >= 1.

    The caller-supplied RNG makes noise deterministic under a given seed.
    """
    end = cfg.anchors[-1].time_sec
    ticks: list[tuple[float, int]] = []

    step = cfg.tick_sec
    num_ticks = int(end / step) + 1
    seg_idx = 0
    for i in range(num_ticks):
        t = round(i * step, 6)
        while (
            seg_idx + 1 < len(cfg.anchors) - 1
            and cfg.anchors[seg_idx + 1].time_sec <= t
        ):
            seg_idx += 1
        a = cfg.anchors[seg_idx]
        b = cfg.anchors[min(seg_idx + 1, len(cfg.anchors) - 1)]

        if cfg.interpolation == "step":
            # Leading-edge step: at the exact b.time_sec we've already jumped.
            value = float(b.concurrency if t >= b.time_sec else a.concurrency)
        elif a.time_sec == b.time_sec:
            value = float(a.concurrency)
        else:
            alpha = (t - a.time_sec) / (b.time_sec - a.time_sec)
            alpha = max(0.0, min(1.0, alpha))
            value = a.concurrency + alpha * (b.concurrency - a.concurrency)

        if cfg.noise_sigma > 0:
            value *= max(0.0, 1.0 + float(rng.normal(0.0, cfg.noise_sigma)))

        ticks.append((t, max(1, int(round(value)))))

    if ticks[-1][0] < end:
        # Ensure the final anchor is explicitly emitted even if end is not a
        # multiple of tick_sec.
        final = cfg.anchors[-1]
        ticks.append((round(final.time_sec, 6), max(1, int(final.concurrency))))

    return ticks
