#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import time
from enum import Enum, auto


class RequestTimerKind(Enum):
    """Timestamp kinds for request handling stages."""

    # TODO: how should this handle streaming vs non-streaming requests?

    REQUEST_START = auto()
    """The timestamp when the client first initiates the request to the backend."""

    SEND_START = auto()
    """The timestamp when the client starts sending the payload."""

    SEND_END = auto()
    """The timestamp when the client finishes sending the payload."""

    RECV_START = auto()
    """The timestamp when the client first starts receiving the response from the backend.
    - For non-streaming requests, this is considered the timestamp when the first byte of the response is received.
    - For streaming requests, this is the timestamp when the server acknowledges the request.
    """

    RECV_CHUNK_START = auto()
    """The timestamp of the start of a single chunk of the response. Only used for streaming requests."""

    RECV_CHUNK_END = auto()
    """The timestamp of the end of a single chunk of the response. Only used for streaming requests."""

    RECV_END = auto()
    """The timestamp when the client finishes receiving the response.
    - For non-streaming requests, this is considered the timestamp when the last byte of the response is received.
    - For streaming requests, this is the timestamp when the server finishes sending the response.
    """

    REQUEST_END = auto()
    """The timestamp when the entire request has finished."""


class RequestTimers:
    """Records timestamps for different stages of request handling."""

    def __init__(self):
        """Initialize timer with zeroed timestamps."""
        self.timestamps: dict[RequestTimerKind, int] = {}
        self.chunk_start_timestamps: list[int] = []
        self.chunk_end_timestamps: list[int] = []

    def reset(self) -> None:
        """Reset all timestamp values to zero. Must be called before re-using the timer."""
        self.timestamps = {}
        self.chunk_start_timestamps = []
        self.chunk_end_timestamps = []

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks captured."""
        return len(self.chunk_start_timestamps)

    def capture_chunk_start_timestamp(self, timestamp_ns: int | None = None) -> int:
        """Capture a timestamp for the start of a chunk and return it."""
        ts = timestamp_ns or time.perf_counter_ns()
        self.chunk_start_timestamps.append(ts)
        return ts

    def capture_chunk_end_timestamp(self, timestamp_ns: int | None = None) -> int:
        """Capture a timestamp for the end of a chunk and return it."""
        ts = timestamp_ns or time.perf_counter_ns()
        self.chunk_end_timestamps.append(ts)
        return ts

    def has_timestamp(self, kind: RequestTimerKind) -> bool:
        """Check if a timestamp has been captured for a kind."""
        return kind in self.timestamps

    def get_timestamp(self, kind: RequestTimerKind) -> int | None:
        """Get the timestamp, in nanoseconds, for a kind.

        Args:
            kind: The timestamp kind.

        Returns:
            The timestamp in nanoseconds.
        """
        return self.timestamps.get(kind)

    def capture_timestamp(self, kind: RequestTimerKind) -> int:
        """Set a timestamp to the current time, in nanoseconds.

        Args:
            kind: The timestamp kind.

        Returns:
            The timestamp in nanoseconds.
        """
        ts = time.perf_counter_ns()
        self.timestamps[kind] = ts
        return ts

    def duration(self, start: RequestTimerKind, end: RequestTimerKind) -> int | None:
        """Return the duration between start timestamp and end timestamp in nanoseconds.

        Args:
            start: The start timestamp kind.
            end: The end timestamp kind.

        Returns:
            Duration in nanoseconds, or None if the duration could not be calculated.
        """
        start_ts = self.get_timestamp(start)
        end_ts = self.get_timestamp(end)

        # If the start or end timestamp is None then can't calculate the
        # duration, so return None to indicate error.
        if start_ts is None or end_ts is None:
            return None

        return end_ts - start_ts
