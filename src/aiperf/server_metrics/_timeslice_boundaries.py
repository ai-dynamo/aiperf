# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timeslice boundary computation for server metrics export statistics."""

import numpy as np

from aiperf.common.constants import NANOS_PER_SECOND


def _compute_timeslice_boundaries(
    range_start_ns: int,
    range_end_ns: int,
    slice_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute timeslice start/end boundaries and completeness flags.

    Generates evenly-spaced timeslice boundaries covering the time range, including
    a partial final timeslice if the range doesn't align with slice boundaries.

    Best practice: Include all data (even partial slices) for completeness, but mark
    which slices are complete vs partial so aggregate statistics can filter appropriately.

    Args:
        range_start_ns: Start of the time range in nanoseconds (inclusive)
        range_end_ns: End of the time range in nanoseconds (inclusive)
        slice_duration: Duration of each timeslice in seconds

    Returns:
        Tuple of (starts, ends, is_complete) numpy arrays where:
        - starts[i] and ends[i] define the i-th timeslice boundaries
        - is_complete[i] is True if the slice covers the full duration, False for partial
        Returns None if slice_duration > range duration (no slices fit).

    Example:
        >>> # 10 second range with 3 second slices
        >>> starts, ends, complete = _compute_timeslice_boundaries(0, 10_000_000_000, 3.0)
        >>> # Returns: [0, 3s, 6s, 9s], [3s, 6s, 9s, 10s], [True, True, True, False]
        >>> #          ^-- 3 complete slices + 1 partial (1s duration)
    """
    timeslice_size_ns = int(slice_duration * NANOS_PER_SECOND)

    # Generate all complete timeslice starts
    timeslice_starts = np.arange(
        range_start_ns, range_end_ns, timeslice_size_ns, dtype=np.int64
    )

    if len(timeslice_starts) == 0:
        return None

    # Compute corresponding ends
    timeslice_ends = timeslice_starts + timeslice_size_ns

    # Mark which timeslices are complete (end <= range_end_ns)
    is_complete = timeslice_ends <= range_end_ns

    # Clip ends to range_end_ns (converts incomplete slices to partial)
    timeslice_ends = np.minimum(timeslice_ends, range_end_ns)

    # Add partial final slice if there's remaining time after last complete slice
    if not is_complete[-1]:
        # Last slice is already partial, no need to add another
        pass
    elif timeslice_ends[-1] < range_end_ns:
        # Add partial final slice from last complete slice end to range end
        final_start = timeslice_ends[-1]
        timeslice_starts = np.append(timeslice_starts, final_start)
        timeslice_ends = np.append(timeslice_ends, range_end_ns)
        is_complete = np.append(is_complete, False)

    return timeslice_starts, timeslice_ends, is_complete
