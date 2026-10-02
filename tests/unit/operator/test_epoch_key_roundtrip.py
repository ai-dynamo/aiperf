# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An epoch KEY is not a POSIX timestamp, and reading it as one 422s the API.

``epoch_key_from_body`` emits epoch-seconds optionally carrying a six-digit
suffix -- real microseconds for a fractional timestamp, or a uid-derived
disambiguator for a whole-second Kubernetes one. ``sweep_union`` read the
directory name straight into ``datetime.fromtimestamp``, so the first sweep
created after suffixes were introduced raised "year 56594345 is out of range"
and FastAPI returned 422 for ``GET /api/v1/sweeps`` -- ONE unreadable
directory taking down the whole list, older sweeps included.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aiperf.operator.results_layout import epoch_key_from_body, epoch_key_seconds


def test_the_suffixed_key_that_broke_the_live_cluster() -> None:
    """gemma-bo5's real epoch. Raw, this is year 56594345."""
    seconds = epoch_key_seconds("1785882875890543")

    assert seconds == 1785882875
    assert datetime.fromtimestamp(seconds, tz=UTC).year == 2026


def test_unsuffixed_keys_still_read_correctly() -> None:
    """Directories written before suffixes exist are still on the PVC."""
    assert epoch_key_seconds("1785866923") == 1785866923


@pytest.mark.parametrize(
    "body",
    [
        # Whole-second Kubernetes timestamp WITH a uid -> uid-derived suffix.
        {"metadata": {"creationTimestamp": "2026-08-04T22:34:35Z", "uid": "abc-123"}},
        # Whole second, no uid -> bare seconds.
        {"metadata": {"creationTimestamp": "2026-08-04T22:34:35Z"}},
        # Fractional -> real microsecond suffix.
        {"metadata": {"creationTimestamp": "2026-08-04T22:34:35.890543Z"}},
    ],
    ids=["uid-suffixed", "bare-seconds", "microsecond-suffixed"],
)  # fmt: skip
def test_every_producer_shape_round_trips_to_the_same_second(body: dict) -> None:
    """The parser is the inverse of the producer for all three shapes it emits."""
    expected = int(
        datetime.fromisoformat(
            body["metadata"]["creationTimestamp"].replace("Z", "+00:00")
        ).timestamp()
    )

    assert epoch_key_seconds(epoch_key_from_body(body)) == expected


@pytest.mark.parametrize(
    "key", ["", "garbage", "17858x2875890543", "-1785866923"],
    ids=["empty", "alpha", "mixed", "negative"],
)  # fmt: skip
def test_unparseable_keys_are_skipped_not_raised(key: str) -> None:
    """A malformed directory must not take down the listing for every sweep."""
    assert epoch_key_seconds(key) is None
