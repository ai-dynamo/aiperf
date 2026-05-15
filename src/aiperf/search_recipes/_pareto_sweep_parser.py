# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the ``--isl-osl-pairs`` CLI flag into ``(isl, osl)`` tuples."""

from __future__ import annotations

__all__ = ["parse_isl_osl_pairs"]


def parse_isl_osl_pairs(raw: str) -> list[tuple[int, int]]:
    """Parse ``"128/128,256/256"`` into ``[(128, 128), (256, 256)]``.

    Whitespace is tolerated around commas and slashes. Each pair is
    ``<isl>/<osl>`` with positive ints. Empty input, malformed pairs,
    non-positive values, and duplicate pairs all raise ``ValueError``.
    """
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError(
            "--isl-osl-pairs: at least one pair required, got empty string"
        )

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for chunk in cleaned.split(","):
        token = chunk.strip()
        if not token:
            continue
        sides = token.split("/")
        if len(sides) != 2:
            raise ValueError(
                f"--isl-osl-pairs: {token!r} expected '<isl>/<osl>' (one slash)"
            )
        try:
            isl = int(sides[0].strip())
            osl = int(sides[1].strip())
        except ValueError as e:
            raise ValueError(
                f"--isl-osl-pairs: {token!r} both sides must be a positive int"
            ) from e
        if isl <= 0 or osl <= 0:
            raise ValueError(
                f"--isl-osl-pairs: {token!r} both sides must be a positive int"
            )
        pair = (isl, osl)
        if pair in seen:
            raise ValueError(f"--isl-osl-pairs: duplicate pair {token!r}")
        seen.add(pair)
        pairs.append(pair)

    if not pairs:
        raise ValueError("--isl-osl-pairs: at least one pair required after parsing")
    return pairs
