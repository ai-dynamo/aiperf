# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import defaultdict

import orjson

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.hooks import on_stop
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import (
    MetricSchema,
    ServerMetricsMetadata,
    ServerMetricsMetadataFile,
    ServerMetricsRecord,
    ServerMetricsSlimRecord,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_EXPORT)
class ServerMetricsExportResultsProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[ServerMetricsSlimRecord]
):
    """Exports per-record server metrics data to JSONL files in slim format.

    This processor converts full ServerMetricsRecord objects to slim format before writing,
    excluding static metadata (metric types, help text) to minimize file size.
    Writes one JSON line per collection cycle.

    Deduplication Logic for ServerMetricsRecords:
        Consecutive identical records are suppressed to save
        storage while preserving complete timeline information. The strategy:

        1. First occurrence → always written (marks start of period)
        2. Duplicates → skipped and counted
        3. Change detected → last duplicate written, then new record
           (provides end timestamp of previous period + start of new period)

        Example: Input A,A,A,B,B,C,D,D,D,D → Output A,A,B,B,C,D,D

        Why write the last occurrence? Time-series data needs actual observations:
            Without: A@t1, B@t4 ← You could guess A ended at ~t3, but no proof
            With:    A@t1, A@t3, B@t4 ← A was observed until t3

        Without the last occurrence, you'd rely on interpolation/assumptions rather
        than actual measured data. This enables accurate duration calculations,
        timeline visualization (Grafana), and time-weighted averages. Essential
        for metrics requiring precise change detection.

        Deduplication uses equality (==) on the metrics dictionary for each separate endpoint.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - endpoint_latency_ns: Time taken to collect the metrics from the endpoint
        - endpoint_url: Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')
        - metrics: Dict mapping metric names to sample lists (flat structure)
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        output_file = user_config.output.server_metrics_export_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self._metadata_file = user_config.output.server_metrics_metadata_json_file
        self._metadata_file.unlink(missing_ok=True)

        # Store metadata for all endpoints using Pydantic model
        self._metadata_file_model = ServerMetricsMetadataFile()
        self._metadata_file_lock = asyncio.Lock()

        self.info(f"Server metrics export enabled: {self.output_file}")
        self.info(f"Server metrics metadata file: {self._metadata_file}")

        # Lock for safe access to creating dynamic locks
        self._lock_creation_lock = asyncio.Lock()
        self._dupe_locks: dict[str, asyncio.Lock] = {}
        self._dupe_counts = defaultdict(int)
        self._previous_records: dict[str, ServerMetricsSlimRecord] = {}

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.
        On first record from each endpoint, extracts metadata and writes metadata file.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        url = record.endpoint_url
        # First check without lock
        if (
            url not in self._metadata_file_model.endpoints
            or self._should_update_metadata(
                record, self._metadata_file_model.endpoints[url]
            )
        ):
            # Second check with lock
            async with self._metadata_file_lock:
                if (
                    url not in self._metadata_file_model.endpoints
                    or self._should_update_metadata(
                        record, self._metadata_file_model.endpoints[url]
                    )
                ):
                    self._extract_and_store_metadata(record)
                    await self._write_metadata_file()

        # Convert to slim format before writing to reduce file size
        slim_record = record.to_slim()
        records_to_write = [slim_record]

        # Create a lock for this endpoint if it doesn't exist
        if url not in self._dupe_locks:
            async with self._lock_creation_lock:
                if url not in self._dupe_locks:
                    self._dupe_locks[url] = asyncio.Lock()

        # Check for duplicates and update the records to write
        async with self._dupe_locks[url]:
            if url in self._previous_records:
                # Only check the metrics for equality, as the timestamp and endpoint latency
                # are unique per scrape.
                if self._previous_records[url].metrics == slim_record.metrics:
                    self._dupe_counts[url] += 1
                    # clear instead of return so the previous record is still updated
                    records_to_write.clear()

                # If we have duplicates, we need to write the previous record before the current record,
                # in order to know when the change actually occurs.
                elif self._dupe_counts[url] > 0:
                    self._dupe_counts[url] = 0
                    records_to_write.insert(0, self._previous_records[url])

            self._previous_records[url] = slim_record

        for rec in records_to_write:
            await self.buffered_write(rec)

    def _should_update_metadata(
        self, record: ServerMetricsRecord, existing_metadata: ServerMetricsMetadata
    ) -> bool:
        """Check if metadata should be updated based on record changes.

        Detects:
        - New metric names not in existing metadata
        - New histogram buckets for existing histogram metrics
        - New summary quantiles for existing summary metrics
        - New unique label values for existing metrics

        Args:
            record: ServerMetricsRecord to check
            existing_metadata: Existing metadata for the endpoint

        Returns:
            True if metadata needs updating, False otherwise
        """
        existing_schemas = existing_metadata.metric_schemas

        # Check for new metric names
        new_metrics = set(record.metrics.keys()) - set(existing_schemas.keys())
        if new_metrics:
            self.info(
                f"Detected new metrics for {record.endpoint_url}: {sorted(new_metrics)}"
            )
            return True

        # Check for histogram bucket changes, summary quantile changes, and label value changes
        max_values = Environment.SERVER_METRICS.MAX_UNIQUE_LABEL_VALUES
        for metric_name, metric_family in record.metrics.items():
            if metric_name not in existing_schemas:
                continue

            existing_schema = existing_schemas[metric_name]

            # Check histogram buckets
            if metric_family.samples and metric_family.samples[0].histogram:
                current_buckets = set(metric_family.samples[0].histogram.buckets.keys())
                existing_buckets = set(existing_schema.bucket_labels or [])

                new_buckets = current_buckets - existing_buckets
                if new_buckets:
                    # Use manual check instead of lambda to avoid binding loop variables
                    if self.is_debug_enabled:
                        self.debug(
                            f"Detected new histogram buckets for {record.endpoint_url}/{metric_name}: {sorted(new_buckets, key=lambda x: float(x))}"
                        )
                    return True

            # Check summary quantiles
            if metric_family.samples and metric_family.samples[0].summary:
                current_quantiles = set(
                    metric_family.samples[0].summary.quantiles.keys()
                )
                existing_quantiles = set(existing_schema.quantile_labels or [])

                new_quantiles = current_quantiles - existing_quantiles
                if new_quantiles:
                    # Use manual check instead of lambda to avoid binding loop variables
                    if self.is_debug_enabled:
                        self.debug(
                            f"Detected new summary quantiles for {record.endpoint_url}/{metric_name}: {sorted(new_quantiles, key=lambda x: float(x))}"
                        )
                    return True

            # Check for new label values
            existing_label_values = existing_schema.unique_label_values or {}
            for sample in metric_family.samples:
                if sample.labels:
                    for label_key, label_value in sample.labels.items():
                        existing_values = set(existing_label_values.get(label_key, []))
                        # Only check if we haven't hit the limit yet
                        if (
                            len(existing_values) < max_values
                            and label_value not in existing_values
                        ):
                            # Use manual check instead of lambda to avoid binding loop variables
                            if self.is_debug_enabled:
                                self.debug(
                                    f"Detected new label value for {record.endpoint_url}/{metric_name}: {label_key}={label_value}"
                                )
                            return True

        return False

    def _extract_and_store_metadata(self, record: ServerMetricsRecord) -> None:
        """Extract metadata from a ServerMetricsRecord and merge with existing metadata.

        Extracts endpoint URL, metric schemas (type, help text, bucket labels, quantile labels,
        unique label values) from the record and merges with existing metadata if present.
        This ensures that new metrics, histogram buckets, summary quantiles, or label values
        are captured even if they appear in later scrapes.

        Args:
            record: ServerMetricsRecord to extract metadata from
        """
        # Get existing metadata if available
        existing_metadata = self._metadata_file_model.endpoints.get(record.endpoint_url)
        existing_schemas = existing_metadata.metric_schemas if existing_metadata else {}

        # Start with existing schemas (copy to avoid mutation)
        metric_schemas: dict[str, MetricSchema] = {
            name: MetricSchema(
                type=schema.type,
                help=schema.help,
                bucket_labels=list(schema.bucket_labels)
                if schema.bucket_labels
                else None,
                quantile_labels=list(schema.quantile_labels)
                if schema.quantile_labels
                else None,
                unique_label_values={
                    k: list(v) for k, v in schema.unique_label_values.items()
                }
                if schema.unique_label_values
                else None,
            )
            for name, schema in existing_schemas.items()
        }

        for metric_name, metric_family in record.metrics.items():
            # Extract bucket labels for histogram metrics
            bucket_labels = None
            if metric_family.samples and metric_family.samples[0].histogram:
                current_buckets = set(metric_family.samples[0].histogram.buckets.keys())

                # Merge with existing buckets if present
                if (
                    metric_name in existing_schemas
                    and existing_schemas[metric_name].bucket_labels
                ):
                    existing_buckets = set(existing_schemas[metric_name].bucket_labels)
                    merged_buckets = current_buckets | existing_buckets
                else:
                    merged_buckets = current_buckets

                bucket_labels = sorted(merged_buckets, key=lambda x: float(x))

            # Extract quantile labels for summary metrics
            quantile_labels = None
            if metric_family.samples and metric_family.samples[0].summary:
                current_quantiles = set(
                    metric_family.samples[0].summary.quantiles.keys()
                )

                # Merge with existing quantiles if present
                if (
                    metric_name in existing_schemas
                    and existing_schemas[metric_name].quantile_labels
                ):
                    existing_quantiles = set(
                        existing_schemas[metric_name].quantile_labels
                    )
                    merged_quantiles = current_quantiles | existing_quantiles
                else:
                    merged_quantiles = current_quantiles

                quantile_labels = sorted(merged_quantiles, key=lambda x: float(x))

            # Extract unique label values from all samples
            unique_label_values: dict[str, set[str]] = defaultdict(set)
            max_values = Environment.SERVER_METRICS.MAX_UNIQUE_LABEL_VALUES

            # Merge with existing label values if present
            if (
                metric_name in existing_schemas
                and existing_schemas[metric_name].unique_label_values
            ):
                for key, values in existing_schemas[
                    metric_name
                ].unique_label_values.items():
                    unique_label_values[key] = set(values)

            # Collect label values from current samples
            for sample in metric_family.samples:
                if sample.labels:
                    for label_key, label_value in sample.labels.items():
                        # Only track up to max_values per label key
                        if len(unique_label_values[label_key]) < max_values:
                            unique_label_values[label_key].add(label_value)

            # Convert to sorted lists for consistent output (None if no labels)
            final_label_values: dict[str, list[str]] | None = None
            if unique_label_values:
                final_label_values = {
                    k: sorted(v) for k, v in sorted(unique_label_values.items())
                }

            # Create or update metric schema
            metric_schemas[metric_name] = MetricSchema(
                type=metric_family.type,
                help=metric_family.help,
                bucket_labels=bucket_labels,
                quantile_labels=quantile_labels,
                unique_label_values=final_label_values,
            )

        metadata = ServerMetricsMetadata(
            endpoint_url=record.endpoint_url,
            endpoint_display=record.endpoint_url,  # Can be enhanced with display name
            metric_schemas=metric_schemas,
        )

        self._metadata_file_model.endpoints[record.endpoint_url] = metadata

    @on_stop
    async def _flush_remaining_dupes(self) -> None:
        """Flush the remaining duplicates for all endpoints.

        Flushes the remaining duplicates for all endpoints during shutdown. This is
        in case the latest record was a duplicate, we still want to write the final record for that endpoint.
        """
        async with self._lock_creation_lock:
            urls = list(self._dupe_locks.keys())
        for url in urls:
            async with self._dupe_locks[url]:
                if self._dupe_counts[url] > 0:
                    await self.buffered_write(self._previous_records[url])
                    self._dupe_counts[url] = 0

    async def _write_metadata_file(self) -> None:
        """Write the complete metadata file for all seen endpoints.

        Re-writes the entire metadata file with all endpoints seen so far.
        Uses Pydantic model serialization with orjson for efficient JSON writing.
        """
        # Serialize the Pydantic model to JSON bytes using orjson
        metadata_json = orjson.dumps(
            self._metadata_file_model.model_dump(exclude_none=True, mode="json"),
            option=orjson.OPT_INDENT_2,
        )

        # Write to file
        self._metadata_file.write_bytes(metadata_json)

        self.debug(
            lambda: f"Wrote metadata file with {len(self._metadata_file_model.endpoints)} endpoints"
        )

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        Returns:
            Empty list (export processors don't generate metric results).
        """
        return []
