# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.base_trace_loader import (
    BaseTraceDatasetLoader,
    _has_meaningful_synthesis,
)
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.output_replay import (
    effective_replay_key,
    merge_output_replay_annotation,
)
from aiperf.dataset.loader.speed_bench import is_speed_bench_row


class MooncakeTraceDatasetLoader(BaseTraceDatasetLoader[MooncakeTrace]):
    """A dataset loader that loads Mooncake trace data from a file.

    Loads Mooncake trace data from a file and converts the data into
    a list of conversations for dataset manager.

    Each line in the file represents a single trace entry and will be
    converted to a separate conversation with a unique session ID.

    Example:
    Fixed schedule version
    ```json
    {"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
    ```

    Multi-turn version
    ```json
    {"session_id": "abc-123", "input_length": 300, "output_length": 40},
    {"session_id": "abc-123", "delay": 2, "input_length": 150, "output_length": 20}
    ```
    """

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For mooncake trace data, simply validate the data against the MooncakeTrace model.
        This will handle all of the validation logic for the different input combinations.
        """
        if data is None:
            return False
        if is_speed_bench_row(data):
            return False

        try:
            MooncakeTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    # ------------------------------------------------------------------
    # Template-method hooks (see BaseTraceDatasetLoader.load_dataset)
    # ------------------------------------------------------------------

    def _parse_trace(self, record: dict) -> MooncakeTrace:
        return MooncakeTrace.model_validate(record)

    def _iter_records_with_line_index(self) -> Iterator[tuple[int, dict[str, Any]]]:
        if self.inline_records is not None:
            if isinstance(self.inline_records, dict):
                raise ValueError(
                    "Mooncake trace inline records must be a list of records, not a dict of pools."
                )
            yield from enumerate(self.inline_records)
            return

        with open(self.filename, encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                if line := line.strip():
                    try:
                        yield line_index, orjson.loads(line)
                    except orjson.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in dataset file {self.filename} at line {line_index + 1}: {e}"
                        ) from None

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        items: list[MooncakeTrace] = []
        session_turns: dict[str, int] = {}

        for line_index, record_dict in self._iter_records_with_line_index():
            trace = self._parse_trace(record_dict)
            turn_index = (
                session_turns.get(trace.session_id, 0)
                if trace.session_id is not None
                else 0
            )
            if trace.output_token_ids is not None:
                if (
                    self._max_osl is not None
                    and trace.output_length is not None
                    and trace.output_length > self._max_osl
                ):
                    raise ValueError(
                        "mooncake trace output_token_ids cannot be combined with max_osl capping"
                    )
                trace._output_replay_key = effective_replay_key(
                    trace.request_id,
                    trace.session_id,
                    turn_index,
                    line_index,
                )
            if trace.session_id is not None:
                session_turns[trace.session_id] = turn_index + 1

            self._preprocess_trace(trace)
            if not self._filter_and_cap_trace(trace):
                continue
            items.append(trace)

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} traces "
                f"across {len(data):,} sessions "
                f"from {self.filename if self.filename else '<inline records>'}"
            )
        )

        if _has_meaningful_synthesis(self._synthesis):
            data = self._apply_synthesis(data)

        data = self._cap_grouped_traces_max_osl(data)
        self._log_filtering_summary()

        return data

    def _group_traces(
        self, items: list[MooncakeTrace]
    ) -> dict[str, list[MooncakeTrace]]:
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        for trace in items:
            session_id = trace.session_id or self.session_id_generator.next()
            data[session_id].append(trace)
        return dict(data)

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _infer_context_mode(
        self, traces: list[MooncakeTrace]
    ) -> ConversationContextMode | None:
        """Auto-detect MESSAGE_ARRAY_WITH_RESPONSES when all traces are self-contained.

        Self-contained traces (pre-built `messages` or verbatim `payload`) bypass
        endpoint formatting and need to be replayed verbatim. Mixed sessions that
        combine self-contained traces with synthesized prompts, or mix `messages`
        and `payload` modes, are rejected.
        """
        msg_trace_count = sum(1 for trace in traces if trace.messages is not None)
        payload_trace_count = sum(1 for trace in traces if trace.payload is not None)
        self_contained_count = msg_trace_count + payload_trace_count

        if msg_trace_count > 0 and payload_trace_count > 0:
            raise ValueError(
                f"mooncake trace: mixed session contains {msg_trace_count} "
                f"`messages` trace(s) and {payload_trace_count} `payload` "
                f"trace(s); each session must use exactly one mode. Split "
                f"the offending sessions or convert all entries to a single "
                f"self-contained mode."
            )
        if self_contained_count == len(traces) and self_contained_count > 0:
            return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        if self_contained_count > 0:
            raise ValueError(
                "Mixed Mooncake sessions with both raw `messages`/`payload` and synthesized prompts are unsupported."
            )
        return None

    def _get_text_input(self, trace: MooncakeTrace) -> str | None:
        if trace.messages is not None or trace.payload is not None:
            return ""
        return trace.text_input

    def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
        # Verbatim payload/messages turns must honor --inter-turn-delay-cap-seconds
        # too: clamp the recorded delay via the shared DelayCapTracker, matching
        # both v1 and the synthesized-prompt branch (super()._build_turn, which
        # clamps in BaseTraceDatasetLoader). Without this the cap was silently
        # ignored for self-contained Mooncake sessions.
        if trace.payload is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=self._delay_cap_tracker.clamp(trace.delay),
                max_tokens=trace.output_length,
                raw_payload=merge_output_replay_annotation(
                    trace.payload, trace._output_replay_key
                ),
                extra_body=trace.extra,
            )
        extra_body = merge_output_replay_annotation(
            trace.extra, trace._output_replay_key
        )
        if trace.messages is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=self._delay_cap_tracker.clamp(trace.delay),
                max_tokens=trace.output_length,
                raw_messages=trace.messages,
                raw_tools=trace.tools,
                extra_body=extra_body,
            )
        turn = super()._build_turn(trace, prompt)
        if extra_body is not None:
            turn.extra_body = extra_body
        return turn

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset({"type"})

    def _reconstruct_traces(
        self, originals: list[MooncakeTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[MooncakeTrace]:
        traces = [MooncakeTrace.model_validate(t) for t in synth_dicts]
        for trace, original in zip(traces, originals, strict=False):
            trace._output_replay_key = original._output_replay_key
        return traces
