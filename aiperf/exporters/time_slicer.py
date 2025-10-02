# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import numpy as np

from aiperf.common.constants import MILLIS_PER_SECOND, NANOS_PER_MILLISECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.metrics.metric_dicts import MetricArray
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric


class TimeSlice:
    """Represents a single time slice of the benchmark."""

    def __init__(
        self,
        slice_index: int,
        start_ns: int,
        end_ns: int,
        slice_duration_ms: int,
    ):
        self.slice_index = slice_index
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.slice_duration_ms = slice_duration_ms

    @property
    def start_ms(self) -> float:
        """Start time in milliseconds."""
        return self.start_ns / NANOS_PER_MILLISECOND

    @property
    def end_ms(self) -> float:
        """End time in milliseconds."""
        return self.end_ns / NANOS_PER_MILLISECOND

    def __repr__(self) -> str:
        return f"TimeSlice(index={self.slice_index}, start_ms={self.start_ms:.2f}, end_ms={self.end_ms:.2f})"


class TimeSlicer(AIPerfLoggerMixin):
    """Splits benchmark results into time-based slices for analysis."""

    def __init__(
        self,
        results: ProfileResults,
        slice_duration_ms: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.results = results
        self.slice_duration_ms = slice_duration_ms
        self._slice_duration_ns = slice_duration_ms * NANOS_PER_MILLISECOND

    def create_time_slices(self) -> list[TimeSlice]:
        """Create time slices based on the benchmark duration and slice duration.

        Returns:
            List of TimeSlice objects representing each time window.
        """
        if not self.results.records:
            self.warning("No records available for time slicing")
            return []

        benchmark_start_ns = self.results.start_ns
        benchmark_end_ns = self.results.end_ns
        total_duration_ns = benchmark_end_ns - benchmark_start_ns

        num_slices = int(np.ceil(total_duration_ns / self._slice_duration_ns))

        self.debug(
            f"Creating {num_slices} time slices of {self.slice_duration_ms}ms each"
        )

        slices = []
        for i in range(num_slices):
            start_ns = benchmark_start_ns + (i * self._slice_duration_ns)
            end_ns = min(start_ns + self._slice_duration_ns, benchmark_end_ns)
            slices.append(
                TimeSlice(
                    slice_index=i,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    slice_duration_ms=self.slice_duration_ms,
                )
            )

        return slices

    def get_metrics_for_slice(
        self,
        time_slice: TimeSlice,
        original_records: list[MetricResult],
    ) -> Mapping[str, MetricResult]:
        """Compute metrics for a specific time slice by filtering the original records.

        Args:
            time_slice: The time slice to compute metrics for
            original_records: The original metric results containing all data

        Returns:
            Dictionary mapping metric tags to MetricResult objects for this slice
        """
        self.debug(f"Computing metrics for {time_slice}")

        # First, get the raw metric data that we need to slice
        # We'll look for the timestamp metric to determine which requests fall in this slice
        timestamp_metric = None
        for record in original_records:
            if record.tag == MinRequestTimestampMetric.tag:
                timestamp_metric = record
                break

        if not timestamp_metric:
            self.warning(
                "Could not find timestamp metric - cannot perform time slicing"
            )
            return {}

        # Build index mask for records that fall within this time slice
        # We need to reconstruct the timestamp data from somewhere...
        # Since we only have aggregated MetricResult objects, we need access to raw arrays
        # This is a limitation - we'll need to pass raw metric data instead

        sliced_metrics = {}
        return sliced_metrics

    def compute_sliced_metrics(
        self,
        time_slice: TimeSlice,
        raw_metric_data: Mapping[str, MetricArray],
        timestamp_data: np.ndarray,
    ) -> dict[str, MetricResult]:
        """Compute metrics for a time slice given raw metric arrays.

        Args:
            time_slice: The time slice to compute metrics for
            raw_metric_data: Raw metric arrays (tag -> MetricArray)
            timestamp_data: Array of request timestamps in nanoseconds

        Returns:
            Dictionary mapping metric tags to MetricResult objects for this slice
        """
        # Create mask for records within this time slice
        mask = (timestamp_data >= time_slice.start_ns) & (
            timestamp_data < time_slice.end_ns
        )
        num_records = np.sum(mask)

        self.debug(
            f"Found {num_records} records in time slice {time_slice.slice_index}"
        )

        if num_records == 0:
            self.warning(f"No records found in {time_slice}")
            return {}

        sliced_metrics = {}

        # For each metric, filter to only the records in this time slice
        for tag, metric_array in raw_metric_data.items():
            if not isinstance(metric_array, MetricArray):
                # For aggregate metrics (not arrays), include as-is
                # This is a simplification - ideally we'd recompute them for the slice
                continue

            # Create a new MetricArray with only the sliced data
            sliced_array = MetricArray(initial_capacity=num_records)
            filtered_data = metric_array.data[mask]

            if len(filtered_data) > 0:
                sliced_array.extend(filtered_data.tolist())

                # Get metric metadata
                metric_class = MetricRegistry.get_class(tag)

                # Compute statistics for the sliced data
                metric_result = sliced_array.to_result(
                    tag=tag,
                    header=metric_class.header,
                    unit=str(metric_class.unit),
                )

                sliced_metrics[tag] = metric_result

        return sliced_metrics
