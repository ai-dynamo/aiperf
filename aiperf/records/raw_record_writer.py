# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writer for exporting raw request/response data with per-record metrics."""

import aiofiles
import orjson

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict


class RawRecordWriter(AIPerfLoggerMixin):
    """Writes raw request/response data with per-record metrics to JSONL files.

    Each RecordProcessor instance writes to its own file to avoid contention
    and enable efficient parallel I/O in distributed setups.

    File format: JSONL (newline-delimited JSON)
    One complete record per line for streaming efficiency.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.service_id = service_id or "processor"
        self.user_config = user_config

        # Construct output file path: raw_records/raw_records_processor_{id}.jsonl
        output_dir = (
            user_config.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each processor writes to its own file - avoids locking/contention
        # Sanitize service_id for filename (replace special chars)
        safe_id = self.service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        self.output_file = output_dir / f"raw_records_{safe_id}.jsonl"

        self.record_count = 0
        self.info(
            f"RawRecordWriter initialized: {self.output_file} - "
            "FULL request/response data will be exported (files may be large)"
        )

    async def write_record(
        self,
        parsed_record: ParsedResponseRecord,
        metric_results: list[MetricRecordDict | BaseException],
        worker_id: str,
    ) -> None:
        """Write a single record with raw data and metrics.

        Args:
            parsed_record: The parsed response record with request/response data
            metric_results: List of per-record metrics computed for this record (may include exceptions)
            worker_id: ID of the worker that generated this request
        """
        try:
            # Collect all metrics (filter out exceptions)
            all_metrics = {}
            for metric_dict in metric_results:
                if not isinstance(metric_dict, BaseException) and metric_dict:
                    all_metrics.update(metric_dict)

            # Build export record with metadata + full parsed record + all metrics
            record_export = {
                "worker_id": worker_id,
                "processor_id": self.service_id,
                "parsed_record": parsed_record.model_dump(mode="json"),
                "metrics": all_metrics,
            }

            # Serialize to JSONL (one line per record) using orjson for speed
            json_bytes = orjson.dumps(
                record_export,
                option=orjson.OPT_APPEND_NEWLINE,  # Automatically adds newline
            )

            async with aiofiles.open(self.output_file, mode="ab") as f:
                await f.write(json_bytes)

            self.record_count += 1
            if self.record_count % 100 == 0:
                self.debug(
                    f"Wrote {self.record_count} raw records to {self.output_file.name}"
                )

        except Exception as e:
            self.error(f"Failed to write raw record: {e}")
            # Don't raise - we don't want export failures to break benchmarking

    async def close(self) -> None:
        """Close the writer and log final statistics."""
        self.info(
            f"RawRecordWriter closed: {self.record_count} records written to {self.output_file}"
        )
