# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
from typing import Any

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import BurstGPTTrace


class BurstGPTTraceDatasetLoader(BaseTraceDatasetLoader[BurstGPTTrace]):
    """Dataset loader for BurstGPT real-world bursty LLM traffic traces.

    Loads a BurstGPT CSV file where each row is an independent request prescribing
    request/response token counts. AIPerf synthesizes prompts of the prescribed
    length rather than replaying actual prompts.

    Expected CSV columns: Timestamp, Request tokens, Response tokens
    (additional columns such as Model, Total tokens, Log Type are ignored)

    Timestamps are seconds since the start of the trace and are converted to
    milliseconds internally.
    """

    _REQUIRED_COLUMNS = frozenset({"Timestamp", "Request tokens", "Response tokens"})

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Detect BurstGPT CSV by checking for the expected header columns."""
        if filename is None:
            return False
        try:
            with open(filename, newline="") as f:
                reader = csv.DictReader(f)
                return cls._REQUIRED_COLUMNS.issubset(set(reader.fieldnames or []))
        except (OSError, csv.Error, UnicodeDecodeError):
            return False

    # ------------------------------------------------------------------
    # Template-method hooks (see BaseTraceDatasetLoader)
    # ------------------------------------------------------------------

    def _parse_trace(self, line: str) -> BurstGPTTrace:
        # BurstGPT is CSV format; load_dataset() is overridden to use csv.DictReader.
        raise NotImplementedError

    def _parse_row(self, row: dict[str, str]) -> BurstGPTTrace | None:
        """Parse a single CSV row into a BurstGPTTrace, returning None on bad data."""
        try:
            return BurstGPTTrace(
                timestamp=float(row["Timestamp"]),
                input_length=int(row["Request tokens"]),
                output_length=int(row["Response tokens"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _preprocess_trace(self, trace: BurstGPTTrace) -> None:
        """Convert timestamp from seconds to milliseconds."""
        trace.timestamp = trace.timestamp * 1000.0

    def _group_traces(
        self, items: list[BurstGPTTrace]
    ) -> dict[str, list[BurstGPTTrace]]:
        """Each BurstGPT row is an independent request; assign a unique session ID per trace."""
        return {self.session_id_generator.next(): [trace] for trace in items}

    # ------------------------------------------------------------------
    # load_dataset — override to read CSV instead of JSONL
    # ------------------------------------------------------------------

    def load_dataset(self) -> dict[str, list[BurstGPTTrace]]:
        """Load, filter, group, and optionally synthesize BurstGPT trace data."""
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        items: list[BurstGPTTrace] = []

        with open(self.filename, newline="") as f:
            reader = csv.DictReader(f)
            missing = self._REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise DatasetLoaderError(
                    f"Missing required columns {missing} in '{self.filename}'"
                )
            for row in reader:
                trace = self._parse_row(row)
                if trace is None:
                    continue

                self._preprocess_trace(trace)

                if not self._filter_and_cap_trace(trace):
                    continue

                items.append(trace)

        self._log_filtering_summary()

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} traces "
                f"across {len(data):,} sessions from {self.filename}"
            )
        )

        if self.user_config.input.synthesis.should_synthesize():
            data = self._apply_synthesis(data)

        return data

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset()

    def _reconstruct_traces(
        self, originals: list[BurstGPTTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[BurstGPTTrace]:
        if not originals:
            raise ValueError("originals must not be empty")
        if len(synth_dicts) != len(originals):
            raise ValueError(
                f"synth_dicts length ({len(synth_dicts)}) != originals length ({len(originals)})"
            )
        result: list[BurstGPTTrace] = []
        for i, synth_dict in enumerate(synth_dicts):
            original = originals[i]
            result.append(
                BurstGPTTrace(
                    timestamp=synth_dict.get("timestamp", original.timestamp),
                    input_length=synth_dict["input_length"],
                    output_length=synth_dict["output_length"],
                )
            )
        return result
