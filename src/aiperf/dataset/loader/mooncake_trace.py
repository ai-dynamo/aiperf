# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import AssistantResponseMode, ConversationContextMode
from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.speed_bench import is_speed_bench_row


def _looks_like_implicit_message_deltas(traces: list[MooncakeTrace]) -> bool:
    """Identify likely incremental input without overriding recorded replay."""
    if len(traces) < 2:
        return False
    message_rows: list[list[dict[str, Any]]] = []
    for trace in traces:
        if trace.messages is None or "assistant_responses" in trace.model_fields_set:
            return False
        message_rows.append(trace.messages)

    if not any(
        message["role"] == "assistant"
        for messages in message_rows
        for message in messages
    ):
        return True
    return any(
        len(current) < len(previous)
        or any(
            before != after for before, after in zip(previous, current, strict=False)
        )
        for previous, current in pairwise(message_rows)
    )


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

    Message-delta version (opt-in via assistant_responses; the first entry carries the
    initial history, later entries carry only the new messages for that turn,
    and live assistant responses are threaded into the history between them)
    ```json
    {"session_id": "abc-123", "assistant_responses": "live", "messages": [{"role": "user", "content": "Hi"}]},
    {"session_id": "abc-123", "assistant_responses": "live", "delay": 2, "messages": [{"role": "user", "content": "More"}]}
    ```
    """

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        Explicit response-mode fields identify Mooncake before validation, so
        malformed values cannot fall through to raw-payload replay. Other rows
        are probed against the MooncakeTrace schema.
        """
        if data is None:
            return False
        if is_speed_bench_row(data):
            return False
        if isinstance(data, dict) and {
            "assistant_responses",
            "message_mode",
        }.intersection(data):
            # Explicit Mooncake fields must reach schema validation, even when
            # their values are invalid, instead of selecting another loader.
            return True

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

    def _preprocess_trace(self, trace: MooncakeTrace) -> None:
        """Fail closed: timestamp-offset cropping is rejected for delta entries.

        ``start_offset``/``end_offset`` drop individual timestamped rows in
        :meth:`BaseTraceDatasetLoader.load_dataset` before sessions are
        grouped, but each ``assistant_responses='live'`` row depends on every
        earlier row in its session (the initial history plus the live
        assistant turns threaded between them), so cropping would silently
        truncate the assembled conversation. This hook runs before
        :meth:`BaseTraceDatasetLoader._filter_and_cap_trace`, so the load
        aborts on the first delta row whenever either offset is configured --
        even if offsets already dropped earlier (e.g. history-mode) rows in
        the file, nothing partially filtered is ever returned. Both offsets
        default to ``None``, so ordinary runs are unaffected; history-mode and
        synthesized traces keep the existing offset filtering.
        """
        if trace.assistant_responses == AssistantResponseMode.LIVE and (
            self._start_offset is not None or self._end_offset is not None
        ):
            raise ValueError(
                f"mooncake trace: timestamp offset filtering (start_offset="
                f"{self._start_offset}, end_offset={self._end_offset}) cannot "
                f"be combined with assistant_responses='live' entries. Offsets drop "
                f"individual rows before sessions are assembled, which would "
                f"silently remove the initial history or preceding turns that "
                f"later delta rows depend on. Remove the offsets and construct "
                f"a trace file containing only the complete delta sessions to "
                f"replay."
            )

    def _group_traces(
        self, items: list[MooncakeTrace]
    ) -> dict[str, list[MooncakeTrace]]:
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        for trace in items:
            session_id = trace.session_id or self.session_id_generator.next()
            data[session_id].append(trace)
        # Synthesis serializes defaults, so inspect omission before that round-trip.
        suspicious_count = 0
        first_suspicious_id: str | None = None
        for session_id, traces in data.items():
            if _looks_like_implicit_message_deltas(traces):
                suspicious_count += 1
                if first_suspicious_id is None:
                    first_suspicious_id = session_id
        if suspicious_count:
            self.warning(
                f"Mooncake trace: {suspicious_count} multi-row messages session(s), "
                f"including {first_suspicious_id!r}, omit assistant_responses and "
                "may contain incremental rows. Defaulting to recorded replays "
                "each row independently; live replies and preceding rows are "
                "not accumulated. Set assistant_responses='live' on every row "
                "for incremental replay, or explicitly set 'recorded' for "
                "intentional self-contained requests."
            )
        return dict(data)

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _infer_context_mode(
        self, traces: list[MooncakeTrace]
    ) -> ConversationContextMode | None:
        """Auto-detect the context mode for self-contained sessions.

        Self-contained traces (pre-built `messages` or verbatim `payload`) bypass
        prompt synthesis. All-`messages` sessions resolve by ``assistant_responses``:
        the default 'recorded' replays each entry verbatim
        (MESSAGE_ARRAY_WITH_RESPONSES) while opt-in 'live' accumulates entries
        and threads live assistant responses into the history
        (DELTAS_WITHOUT_RESPONSES). Mixed sessions that combine self-contained
        traces with synthesized prompts, mix `messages` and `payload` modes, or
        mix 'recorded' and 'live' entries (including entries that omit
        ``assistant_responses`` and default to 'recorded') are rejected.
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

        live_count = sum(
            1
            for trace in traces
            if trace.messages is not None
            and trace.assistant_responses == AssistantResponseMode.LIVE
        )
        if 0 < live_count < msg_trace_count:
            raise ValueError(
                f"mooncake trace: mixed session contains {live_count} "
                f"assistant_responses='live' trace(s) and "
                f"{msg_trace_count - live_count} assistant_responses='recorded' "
                f"trace(s) (omitted assistant_responses defaults to 'recorded'); each "
                f"session must use exactly one assistant_responses value. Mark every "
                f"`messages` entry in the session with assistant_responses='live' or "
                f"none of them."
            )

        if self_contained_count == len(traces) and self_contained_count > 0:
            if live_count > 0:
                self._validate_endpoint_extra_for_live()
                return ConversationContextMode.DELTAS_WITHOUT_RESPONSES
            return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        if self_contained_count > 0:
            raise ValueError(
                "Mixed Mooncake sessions with both raw `messages`/`payload` and synthesized prompts are unsupported."
            )
        return None

    def _validate_endpoint_extra_for_live(self) -> None:
        """Fail closed when an endpoint-global extra input would clobber history.

        Endpoint-level ``extra`` entries (``--extra-inputs`` /
        ``endpoint.extra``) are merged into every request body after the
        formatter builds ``messages`` (or ``input`` for the Responses API),
        so either key could replace the assembled history in every request.
        """
        extra = self.run.cfg.endpoint.extra
        if history_keys := {"messages", "input"}.intersection(extra):
            raise ValueError(
                "mooncake trace: assistant_responses='live' cannot be combined with "
                f"an endpoint-level extra input named {sorted(history_keys)} "
                "(--extra-inputs or endpoint.extra); it "
                "would overwrite the assembled conversation history in every "
                "request"
            )

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
                raw_payload=trace.payload,
                extra_body=trace.extra,
            )
        if trace.messages is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=self._delay_cap_tracker.clamp(trace.delay),
                max_tokens=trace.output_length,
                raw_messages=trace.messages,
                raw_tools=trace.tools,
                extra_body=trace.extra,
            )
        turn = super()._build_turn(trace, prompt)
        if trace.extra is not None:
            turn.extra_body = trace.extra
        return turn

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset({"type"})

    def _reconstruct_traces(
        self, originals: list[MooncakeTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[MooncakeTrace]:
        return [MooncakeTrace.model_validate(t) for t in synth_dicts]
