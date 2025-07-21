# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf metrics.

"""

import logging

import pytest

from aiperf.common.models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
)

logging.basicConfig(level=logging.DEBUG)


class ParsedResponseRecordBuilder:
    """Builder class for creating ParsedResponseRecord instances with flexible configuration.

    Supports building single or multiple ParsedResponseRecord instances with custom
    requests and responses for comprehensive testing scenarios.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the builder to default values."""
        self._records = []  # List of record configurations
        self._current_record = self._new_record_config()
        return self

    def _new_record_config(self):
        """Create a new record configuration with default values."""
        return {
            "worker_id": "worker_1",
            "request_start_perf_ns": 100,
            "request_kwargs": {},
            "responses": [],
        }

    def with_worker_id(self, worker_id: str):
        """Set the worker ID for the current record."""
        self._current_record["worker_id"] = worker_id
        return self

    def with_request_start_time(self, start_perf_ns: int):
        """Set the request start time for the current record."""
        self._current_record["request_start_perf_ns"] = start_perf_ns
        return self

    def with_request_kwargs(self, **kwargs):
        """Add additional kwargs to the RequestRecord for the current record."""
        self._current_record["request_kwargs"].update(kwargs)
        return self

    def add_response(
        self,
        perf_ns: int,
        raw_text: list[str] = None,
        parsed_text: list[str] = None,
        **kwargs,
    ):
        """Add a response to the current record."""
        if raw_text is None:
            raw_text = []
        if parsed_text is None:
            parsed_text = []

        response_data = ResponseData(
            perf_ns=perf_ns, raw_text=raw_text, parsed_text=parsed_text, **kwargs
        )
        self._current_record["responses"].append(response_data)
        return self

    def add_responses(self, *response_configs):
        """Add multiple responses to the current record. Each config should be a dict with response parameters."""
        for config in response_configs:
            self.add_response(**config)
        return self

    def new_record(self):
        """Finish the current record and start a new one. Returns self for chaining."""
        self._records.append(self._current_record.copy())
        self._current_record = self._new_record_config()
        return self

    def add_request(
        self, worker_id: str = None, start_perf_ns: int = None, **request_kwargs
    ):
        """Add a new request record. Automatically starts a new record."""
        self.new_record()

        if worker_id is not None:
            self.with_worker_id(worker_id)
        if start_perf_ns is not None:
            self.with_request_start_time(start_perf_ns)
        if request_kwargs:
            self.with_request_kwargs(**request_kwargs)

        return self

    def build(self) -> ParsedResponseRecord:
        """Build and return a single ParsedResponseRecord (for backward compatibility)."""
        records = self.build_all()
        return records[0]

    def build_all(self) -> list[ParsedResponseRecord]:
        """Build and return all configured ParsedResponseRecord instances."""
        # Add the current record if it has content
        all_records = self._records.copy()
        all_records.append(self._current_record)

        parsed_records = []
        for record_config in all_records:
            request = RequestRecord(
                conversation_id="test-conversation",
                turn_index=0,
                model_name="test-model",
                start_perf_ns=record_config["request_start_perf_ns"],
                **record_config["request_kwargs"],
            )

            parsed_record = ParsedResponseRecord(
                worker_id=record_config["worker_id"],
                request=request,
                responses=record_config["responses"].copy(),
            )
            parsed_records.append(parsed_record)

        return parsed_records


@pytest.fixture
def parsed_response_record_builder():
    """Fixture that provides a builder for creating ParsedResponseRecord instances."""
    return ParsedResponseRecordBuilder()
