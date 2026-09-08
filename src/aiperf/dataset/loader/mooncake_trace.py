# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
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

    Message-delta version (opt-in via message_mode; the first entry carries the
    initial history, later entries carry only the new messages for that turn,
    and live assistant responses are threaded into the history between them)
    ```json
    {"session_id": "abc-123", "message_mode": "delta", "messages": [{"role": "user", "content": "Hi"}]},
    {"session_id": "abc-123", "message_mode": "delta", "delay": 2, "messages": [{"role": "user", "content": "More"}]}
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

    def _preprocess_trace(self, trace: MooncakeTrace) -> None:
        """Fail closed: timestamp-offset cropping is rejected for delta entries.

        ``start_offset``/``end_offset`` drop individual timestamped rows in
        :meth:`BaseTraceDatasetLoader.load_dataset` before sessions are
        grouped, but each ``message_mode='delta'`` row depends on every
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
        if trace.message_mode == "delta" and (
            self._start_offset is not None or self._end_offset is not None
        ):
            raise ValueError(
                f"mooncake trace: timestamp offset filtering (start_offset="
                f"{self._start_offset}, end_offset={self._end_offset}) cannot "
                f"be combined with message_mode='delta' entries. Offsets drop "
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
        return dict(data)

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _infer_context_mode(
        self, traces: list[MooncakeTrace]
    ) -> ConversationContextMode | None:
        """Auto-detect the context mode for self-contained sessions.

        Self-contained traces (pre-built `messages` or verbatim `payload`) bypass
        prompt synthesis. All-`messages` sessions resolve by ``message_mode``:
        the default 'history' replays each entry verbatim
        (MESSAGE_ARRAY_WITH_RESPONSES) while opt-in 'delta' accumulates entries
        and threads live assistant responses into the history
        (DELTAS_WITHOUT_RESPONSES). Mixed sessions that combine self-contained
        traces with synthesized prompts, mix `messages` and `payload` modes, or
        mix 'history' and 'delta' entries (including entries that omit
        ``message_mode`` and default to 'history') are rejected.
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

        delta_count = sum(
            1
            for trace in traces
            if trace.messages is not None and trace.message_mode == "delta"
        )
        if 0 < delta_count < msg_trace_count:
            raise ValueError(
                f"mooncake trace: mixed session contains {delta_count} "
                f"message_mode='delta' trace(s) and "
                f"{msg_trace_count - delta_count} message_mode='history' "
                f"trace(s) (omitted message_mode defaults to 'history'); each "
                f"session must use exactly one message_mode. Mark every "
                f"`messages` entry in the session with message_mode='delta' or "
                f"none of them."
            )

        if self_contained_count == len(traces) and self_contained_count > 0:
            if delta_count > 0:
                self._validate_endpoint_extra_for_delta()
                return ConversationContextMode.DELTAS_WITHOUT_RESPONSES
            return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        if self_contained_count > 0:
            raise ValueError(
                "Mixed Mooncake sessions with both raw `messages`/`payload` and synthesized prompts are unsupported."
            )
        return None

    def _validate_endpoint_extra_for_delta(self) -> None:
        """Fail closed when an endpoint-global extra input would clobber history.

        Endpoint-level ``extra`` entries (``--extra-inputs`` /
        ``endpoint.extra``) are merged into every request body after the
        formatter builds ``messages``, so an entry named 'messages' would
        silently replace the assembled delta history in every request.
        """
        endpoint = getattr(self.run.cfg, "endpoint", None)
        extra = getattr(endpoint, "extra", None) or {}
        if "messages" in extra:
            raise ValueError(
                "mooncake trace: message_mode='delta' cannot be combined with "
                "an endpoint-level extra input named 'messages' "
                "(--extra-inputs messages:... or endpoint.extra.messages); it "
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
