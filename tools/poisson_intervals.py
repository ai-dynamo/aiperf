# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded Poisson inter-arrival schedule via the parity RNG, emitted as JSONL.

Python counterpart to ``rust/aiperf/examples/poisson_intervals.rs``. Uses the byte-exact
``aiperf.common.rng_parity`` backend to reproduce the Rust ``aiperf::timing`` Poisson
generator bit-for-bit: BLAKE3-derive the ``timing.request.poisson_interval`` seed off the
root, then emit ``secs_to_ns(expovariate(rate))`` intervals. Output is identical JSONL to
the Rust example, so the two files diff clean.

Usage: ``python tools/poisson_intervals.py <root_seed> <rate> <count>``
"""

from __future__ import annotations

import math
import sys

from aiperf.common.rng_parity import ParityRandomGenerator, RngRoot, namespace

_NANOS_PER_SECOND = 1_000_000_000.0


def _f64_round_half_away(value: float) -> int:
    """Match Rust ``f64::round`` — round half away from zero (values here are > 0)."""
    floor = math.floor(value)
    return int(floor) + 1 if value - floor >= 0.5 else int(floor)


def secs_to_ns(secs: float) -> int:
    """Port of ``intervals.rs`` ``secs_to_ns``: nearest ns, ties away from zero."""
    if not math.isfinite(secs) or secs <= 0.0:
        return 0
    return _f64_round_half_away(secs * _NANOS_PER_SECOND)


def main(argv: list[str]) -> int:
    root_seed = int(argv[1]) if len(argv) > 1 else 42
    rate = float(argv[2]) if len(argv) > 2 else 50.0
    count = int(argv[3]) if len(argv) > 3 else 64

    seed = RngRoot(root_seed).derive_seed(namespace.TIMING_REQUEST_POISSON_INTERVAL)
    generator = ParityRandomGenerator.from_seed(seed)

    cumulative_ns = 0
    for i in range(count):
        interval_ns = secs_to_ns(generator.expovariate(rate))
        cumulative_ns += interval_ns
        print(f'{{"i":{i},"interval_ns":{interval_ns},"cumulative_ns":{cumulative_ns}}}')
    print(
        f"poisson: root_seed={root_seed} derived_seed={seed} rate={rate} count={count}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
