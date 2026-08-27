# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors for the parity RNG.

Ported from ``rust/aiperf/src/rng/error.rs``. The Rust crate uses a closed
``RngError`` enum; here a single exception with a ``kind`` discriminant carries the
same validation semantics. Message text need not match Rust byte-for-byte — only the
*decision* to raise (parity of validation behavior) matters.
"""

from __future__ import annotations

__all__ = ["RngError"]


class RngError(ValueError):
    """Validation/construction error raised by the parity RNG.

    Mirrors the ``RngError`` variants in ``error.rs``: empty range/sequence,
    oversized sample, invalid parameter/bounds/weights, invalid probability sum.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind

    @classmethod
    def empty_range(cls, what: str) -> RngError:
        return cls(f"empty range for {what}", kind="empty_range")

    @classmethod
    def empty_sequence(cls, what: str) -> RngError:
        return cls(f"empty sequence for {what}", kind="empty_sequence")

    @classmethod
    def sample_too_large(cls, k: int, length: int) -> RngError:
        return cls(
            f"sample size {k} exceeds population length {length}",
            kind="sample_too_large",
        )

    @classmethod
    def invalid_parameter(cls, what: str, value: float) -> RngError:
        return cls(f"invalid parameter {what}={value}", kind="invalid_parameter")

    @classmethod
    def invalid_bounds(cls, lower: float, upper: float) -> RngError:
        return cls(
            f"invalid bounds: lower ({lower}) > upper ({upper})",
            kind="invalid_bounds",
        )

    @classmethod
    def invalid_weights(cls, reason: str) -> RngError:
        return cls(f"invalid weights: {reason}", kind="invalid_weights")

    @classmethod
    def invalid_probability_sum(cls, total: float) -> RngError:
        return cls(
            f"probabilities must sum to 100.0, got {total}",
            kind="invalid_probability_sum",
        )
